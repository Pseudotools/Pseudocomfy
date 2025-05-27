import { app } from "../../../scripts/app.js";

const _ID = "PseudoProcessImagePrompt";

app.registerExtension({
    name: 'pseudocomfy.' + _ID,
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _ID) return;

        nodeType.prototype.onNodeCreated = function () {
            // Add invisible filler divs to ensure node height
            for (let i = 0; i < 2; i++) {
                const fillerDiv = document.createElement("div");
                fillerDiv.style.width = "100%";
                fillerDiv.style.backgroundColor = "yellow";
                fillerDiv.style.visibility = "hidden";
                if (this.addDOMWidget) {
                    const widget = this.addDOMWidget(`Filler${i + 1}`, "custom", fillerDiv, {});
                    if (!this.domWidgets) this.domWidgets = [];
                    this.domWidgets.push(widget);
                }
            }
        };

        nodeType.prototype.onExecuted = function (message) {
            // Remove previous DOM widgets (except fillers)
            if (this.domWidgets && this.domWidgets.length) {
                for (const w of this.domWidgets) {
                    if (w.element && w.element.parentNode) w.element.parentNode.removeChild(w.element);
                    if (this.widgets && Array.isArray(this.widgets)) {
                        const idx = this.widgets.indexOf(w);
                        if (idx !== -1) this.widgets.splice(idx, 1);
                    }
                }
                this.domWidgets = [];
            } else {
                this.domWidgets = [];
            }

            // Extract info from message
            const given_width = message.given_width;
            const given_height = message.given_height;
            const scaled_width = message.scaled_width;
            const scaled_height = message.scaled_height;
            const img_b64 = message.img;




            // the image
            const topDiv = document.createElement("div");
            topDiv.style.display = "flex";
            topDiv.style.alignItems = "center";
            topDiv.style.justifyContent = "center";
            topDiv.style.width = "100%";
            topDiv.style.height = "auto";
            topDiv.style.overflow = "hidden";

            if (img_b64) {
                const img = document.createElement("img");
                img.src = img_b64;
                img.style.maxWidth = "100%";
                img.style.maxHeight = "100%";
                img.style.objectFit = "contain";
                img.style.border = "1px solid #aaa";
                topDiv.appendChild(img);
            } else {
                const noImg = document.createElement("div");
                noImg.textContent = "No image";
                noImg.style.color = "#888";
                topDiv.appendChild(noImg);
            }

            // single centered label
            const bottomDiv = document.createElement("div");
            bottomDiv.style.display = "flex";
            bottomDiv.style.justifyContent = "center";
            bottomDiv.style.alignItems = "center";
            bottomDiv.style.padding = "8px";
            bottomDiv.style.fontSize = "0.75em";
            bottomDiv.textContent = `(${given_width},${given_height}) -> (${scaled_width},${scaled_height})`;



            // Parent container
            const parent = document.createElement("div");
            parent.style.display = "flex";
            parent.style.flexDirection = "column";
            parent.style.width = "100%";
            parent.style.height = "100%";
            parent.appendChild(topDiv);
            parent.appendChild(bottomDiv);

            // Add as a single DOM widget
            const widget = this.addDOMWidget?.("Process Image Prompt", "custom", parent, {});
            if (widget) this.domWidgets.push(widget);
        };
    },
});


