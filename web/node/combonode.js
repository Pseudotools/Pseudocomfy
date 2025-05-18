import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";

const _ID = "ComboNodeCozy";

app.registerExtension({
    name: 'pseudocomfy.' + _ID,
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _ID) return;

        const orig_nodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            orig_nodeCreated?.apply(this, arguments);

            // Create an <img> element for the preview
            const imgEl = $el("img", {
                style: {
                    display: "block",
                    margin: "8px auto",
                    background: "#222",
                    borderRadius: "6px",
                    width: "128px",
                    height: "128px",
                    objectFit: "contain",
                    border: "1px solid #444"
                }
            });
            
            console.log("[pseudocomfy]\t\t ComboNodeCozy widget created"); 


            // Custom widget definition
            const widget = {
                type: "HTML",
                name: "blank_image", // must match RETURN_NAMES in Python
                inputEl: imgEl,
                draw(ctx, node, widget_width, y, widget_height) {
                    // Optionally position the image
                },
                onNodeValueChanged(node, value) {
                    console.log("[pseudocomfy]\t\t onNodeValueChanged called with:", value);

                    // value is a base64 PNG string from ComfyUI
                    if (value && typeof value === "string" && value.startsWith("data:image")) {
                        this.inputEl.src = value;
                    } else {
                        this.inputEl.src = "";
                    }
                }
            };

            document.body.appendChild(imgEl);
            this.addCustomWidget(widget);

            this.onRemoved = function () { imgEl.remove(); };
            this.serialize_widgets = false;
        };
    },
});