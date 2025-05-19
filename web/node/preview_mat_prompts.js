import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

const _ID = "PreviewMaterialPrompts";


app.registerExtension({
    name: 'pseudocomfy.' + _ID,
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _ID) return;
        //console.log("[pseudocomfy] Registering node extension for " + _ID);

        nodeType.prototype.onNodeCreated = function () {
            //console.log("[pseudocomfy] onNodeCreated");

            // to ensure node height will show controls, 
            // add some temporary divs and store their widgets in domWidgets
            for (let i = 0; i < 6; i++) {
                const fillerDiv = document.createElement("div");
                fillerDiv.style.width = "100%";
                fillerDiv.style.backgroundColor = "yellow";
                fillerDiv.style.visibility = "hidden"; // Invisible but takes up space

                if (this.addDOMWidget) {
                    const widget = this.addDOMWidget(`Filler${i + 1}`, "custom", fillerDiv, {});
                    if (!this.domWidgets) this.domWidgets = [];
                    this.domWidgets.push(widget);
                }
            }
        };

        nodeType.prototype.onExecuted = function (message) {


            // Properly remove previous widgets if needed
            if (this.domWidgets && this.domWidgets.length) {
                for (const w of this.domWidgets) {
                    // Remove DOM element if present
                    if (w.element && w.element.parentNode) w.element.parentNode.removeChild(w.element);
                    // Remove from node's widgets array if present
                    if (this.widgets && Array.isArray(this.widgets)) {
                        const idx = this.widgets.indexOf(w);
                        if (idx !== -1) this.widgets.splice(idx, 1);
                    }
                }
                this.domWidgets = [];
            } else {
                this.domWidgets = [];
            }

            // Defensive: ensure mat_msks and mat_txts are arrays
            const matTxts = Array.isArray(message.mat_txts) ? message.mat_txts : [];
            const matImgs = Array.isArray(message.mat_imgs) ? message.mat_imgs : [];
            const maskImgs = Array.isArray(message.mat_msks) ? message.mat_msks : [];
            
            if (!this._itemState) this._itemState = { idx: 0 };
            const state = this._itemState;
            if (state.idx >= maskImgs.length) state.idx = maskImgs.length - 1;
            if (state.idx < 0) state.idx = 0;

            let promptWidgetEl = null;
            let controlsInfoDiv = null;
            let itemCount = maskImgs.length;

            // Info widget needs to be updated by buttons and image
            const updateInfo = () => {
                if (controlsInfoDiv && controlsInfoDiv.infoWidgetEl) {
                    controlsInfoDiv.infoWidgetEl.textContent = maskImgs.length
                        ? `${state.idx + 1} of ${maskImgs.length}`
                        : "No materials found";
                }
                // update promptWidgetEl layout if needed ---
                if (promptWidgetEl && promptWidgetEl.parentNode) {
                    // Check if layout type needs to change
                    const hasImg = matImgs.length > 0 && matImgs[state.idx];
                    const isSideBySide = promptWidgetEl.style.flexDirection === "row";
                    if ((hasImg && !isSideBySide) || (!hasImg && isSideBySide)) {
                        // Remove old widget and insert new one
                        const newPromptWidget = createPromptWidget(matTxts, matImgs, state);
                        promptWidgetEl.parentNode.replaceChild(newPromptWidget, promptWidgetEl);
                        promptWidgetEl = newPromptWidget;
                    } else {
                        // Just update value
                        promptWidgetEl.value = matTxts.length > 0 ? matTxts[state.idx] : "";
                    }
                }
            };

            // Create widgets
            const maskDiv = createMaskWidget(maskImgs, state, updateInfo);
            controlsInfoDiv = createControlsInfoWidget(itemCount, state, maskDiv.img, updateInfo);
            promptWidgetEl = createPromptWidget(matTxts, matImgs, state);

            // Combine all into a single parent with top/bottom layout
            const parent = document.createElement("div");
            parent.style.display = "flex";
            parent.style.flexDirection = "column";
            parent.style.alignItems = "stretch";
            parent.style.overflow = "hidden";
            parent.style.height = "100%";

            // Top: maskDiv + promptWidget (flexible)
            const topDiv = document.createElement("div");
            topDiv.style.display = "flex";
            topDiv.style.flexDirection = "column";
            topDiv.style.flex = "1 1 0"; // Flexible, fills available space
            topDiv.style.overflow = "hidden";
            topDiv.appendChild(maskDiv);
            topDiv.appendChild(promptWidgetEl);

            // Bottom: controlsInfoDiv (fixed height)
            const bottomDiv = document.createElement("div");
            bottomDiv.style.flex = "0 0 auto";
            bottomDiv.style.height = "40px"; // Fixed height, adjust as needed
            bottomDiv.style.display = "flex";
            bottomDiv.style.alignItems = "center";
            bottomDiv.style.justifyContent = "center";
            bottomDiv.appendChild(controlsInfoDiv);

            parent.appendChild(topDiv);
            parent.appendChild(bottomDiv);

            // Add as a single DOM widget
            const widget = this.addDOMWidget?.("Material Mask", "custom", parent, {});
            if (widget) this.domWidgets.push(widget);
        };
    },
});




function createMaskWidget(images, state, updateInfo) {
    const wrapper = document.createElement("div");
    wrapper.style.width = "100%";
    wrapper.style.display = "flex";
    wrapper.style.justifyContent = "center";
    wrapper.style.alignItems = "center";
    wrapper.style.marginBottom = "8px";

    const img = document.createElement("img");
    img.style.width = "100%";
    img.style.maxWidth = "192px";
    img.style.height = "auto";
    img.style.display = "block";
    img.src = images.length > 0 ? images[state.idx] : "";

    img.updateSrc = () => {
        img.src = images.length > 0 ? images[state.idx] : "";
        updateInfo && updateInfo();
    };

    wrapper.appendChild(img);
    // Expose img for controls
    wrapper.img = img;
    return wrapper;
}



function createControlsInfoWidget(itemCount, state, img, updateInfo) {
    const wrapper = document.createElement("div");
    wrapper.style.width = "100%";
    wrapper.style.height = "40px"; // Fixed height, adjust as needed
    wrapper.style.display = "flex";
    wrapper.style.flexDirection = "row";
    wrapper.style.alignItems = "center";
    wrapper.style.justifyContent = "center";
    //wrapper.style.marginBottom = "8px";
    wrapper.style.gap = "8px";

    // Button wrappers to allow flex-grow
    const btnPrevWrapper = document.createElement("div");
    btnPrevWrapper.style.flex = "1 1 0";
    btnPrevWrapper.style.display = "flex";
    btnPrevWrapper.style.justifyContent = "flex-end";

    const btnPrev = document.createElement("button");
    btnPrev.textContent = "<";
    btnPrev.style.width = "100%";
    btnPrev.onclick = () => {
        if (itemCount > 0) {
            state.idx = (state.idx - 1 + itemCount) % itemCount;
            img.updateSrc();
            updateInfo && updateInfo();
        }
    };
    btnPrevWrapper.appendChild(btnPrev);

    const info = document.createElement("div");
    info.style.textAlign = "center";
    info.style.fontSize = "0.95em";
    info.style.color = "#333";
    info.style.flex = "0 1 auto";
    info.style.whiteSpace = "nowrap";
    info.textContent = itemCount
        ? `${state.idx + 1} of ${itemCount}`
        : "No materials found";

    const btnNextWrapper = document.createElement("div");
    btnNextWrapper.style.flex = "1 1 0";
    btnNextWrapper.style.display = "flex";
    btnNextWrapper.style.justifyContent = "flex-start";

    const btnNext = document.createElement("button");
    btnNext.textContent = ">";
    btnNext.style.width = "100%";
    btnNext.onclick = () => {
        if (itemCount > 0) {
            state.idx = (state.idx + 1) % itemCount;
            img.updateSrc();
            updateInfo && updateInfo();
        }
    };
    btnNextWrapper.appendChild(btnNext);

    // Expose info for updateInfo
    wrapper.infoWidgetEl = info;

    wrapper.appendChild(btnPrevWrapper);
    wrapper.appendChild(info);
    wrapper.appendChild(btnNextWrapper);
    return wrapper;
}



function createPromptWidget(txts, imgs, state) {
    //console.log("[pseudocomfy] createPromptWidget");
    //console.log("[pseudocomfy] txts:", txts);
    //console.log("[pseudocomfy] imgs:", imgs);

    const wrapper = document.createElement("div");
    wrapper.style.width = "100%";
    wrapper.style.margin = "8px 0";
    wrapper.style.flex = "1 1 0";
    wrapper.style.display = "flex";
    wrapper.style.flexDirection = "column";

    // Defensive: ensure txts and imgs are arrays of the same length
    const idx = state.idx;
    const txt = (Array.isArray(txts) && txts.length > idx) ? txts[idx] : "";
    const imgSrc = (Array.isArray(imgs) && imgs.length > idx) ? imgs[idx] : null;

    // If no image, use original layout (textarea only)
    if (!imgSrc) {
        const textarea = document.createElement("textarea");
        textarea.readOnly = true;
        textarea.style.width = "100%";
        textarea.style.height = "100%";
        textarea.style.flex = "1 1 0";
        textarea.style.resize = "none";
        textarea.style.background = "#f8f8f8";
        textarea.style.color = "#333";
        textarea.style.fontSize = "0.95em";
        textarea.style.border = "1px solid #ccc";
        textarea.style.borderRadius = "4px";
        textarea.style.padding = "4px";
        textarea.value = txt;

        wrapper.appendChild(textarea);
        // Expose textarea for updateInfo
        Object.defineProperty(wrapper, "value", {
            get: () => textarea.value,
            set: v => { textarea.value = v; }
        });
        return wrapper;
    }

    // If image is present, use side-by-side layout
    wrapper.style.flexDirection = "row";
    wrapper.style.alignItems = "stretch";
    wrapper.style.gap = "8px";

    // Textarea (left)
    const textarea = document.createElement("textarea");
    textarea.readOnly = true;
    textarea.style.width = "100%";
    textarea.style.height = "100%";
    textarea.style.flex = "1 1 0";
    textarea.style.resize = "none";
    textarea.style.background = "#f8f8f8";
    textarea.style.color = "#333";
    textarea.style.fontSize = "0.95em";
    textarea.style.border = "1px solid #ccc";
    textarea.style.borderRadius = "4px";
    textarea.style.padding = "4px";
    textarea.value = txt;

    // Image (right)
    const imgDiv = document.createElement("div");
    imgDiv.style.display = "flex";
    imgDiv.style.alignItems = "center";
    imgDiv.style.justifyContent = "center";
    imgDiv.style.height = "100%";
    imgDiv.style.flex = "0 0 128px"; // Fixed width for image column
    imgDiv.style.maxWidth = "128px";
    imgDiv.style.minWidth = "64px";

    const img = document.createElement("img");
    img.src = imgSrc;
    img.style.maxWidth = "100%";
    img.style.maxHeight = "128px";
    img.style.display = "block";
    img.style.objectFit = "contain";
    imgDiv.appendChild(img);

    wrapper.appendChild(textarea);
    wrapper.appendChild(imgDiv);

    // Expose textarea for updateInfo
    Object.defineProperty(wrapper, "value", {
        get: () => textarea.value,
        set: v => { textarea.value = v; }
    });
    return wrapper;
}