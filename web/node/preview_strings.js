import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Displays a string (or list of strings) on a node

const _ID = "PseudoPreviewStrings";

app.registerExtension({
    name: 'pseudocomfy.' + _ID,
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _ID) return;

        function populate(strings) {
            // Remove all widgets except the input widget (if present)
            if (this.widgets) {
                const isConvertedWidget = +!!this.inputs?.[0]?.widget;
                for (let i = isConvertedWidget; i < this.widgets.length; i++) {
                    this.widgets[i].onRemove?.();
                }
                this.widgets.length = isConvertedWidget;
            }

            // Accepts either a string or a list of strings
            let displayStrings = Array.isArray(strings) ? strings : [strings];
            displayStrings.forEach((str, idx) => {
                const w = ComfyWidgets["STRING"](
                    this,
                    `String${displayStrings.length > 1 ? " " + (idx + 1) : ""}`,
                    ["STRING", { multiline: true }],
                    app
                ).widget;
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.6;
                w.value = str;
                w.inputEl.placeholder = "String";
            });

            requestAnimationFrame(() => {
                const sz = this.computeSize();
                if (sz[0] < this.size[0]) sz[0] = this.size[0];
                if (sz[1] < this.size[1]) sz[1] = this.size[1];
                this.onResize?.(sz);
                app.graph.setDirtyCanvas(true, false);
            });
        }

        // When the node is executed we will be sent the input string(s), display in the widgets
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            populate.call(this, message.strings);
        };

        // Optionally handle configure/onConfigure for persistence (not required for preview)
    },
});