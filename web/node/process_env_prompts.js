import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Displays input text on a node

const _ID = "PseudoProcessEnvironmentalPrompts";

app.registerExtension({
    name: 'pseudocomfy.' + _ID,
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _ID) return;

        function populate(env_scene, env_style, env_negative) {
            // Remove all widgets except the input widget (if present)
            if (this.widgets) {
                const isConvertedWidget = +!!this.inputs?.[0]?.widget;
                for (let i = isConvertedWidget; i < this.widgets.length; i++) {
                    this.widgets[i].onRemove?.();
                }
                this.widgets.length = isConvertedWidget;
            }

            function addSimpleWidget(node, label, value) {
                const w = ComfyWidgets["STRING"](
                    node,
                    label,
                    ["STRING", { multiline: true }],
                    app
                ).widget;
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.6;
                w.value = value;
                w.inputEl.placeholder = label;
                return w;
            }

            addSimpleWidget(this, "Scene", env_scene);
            addSimpleWidget(this, "Style", env_style);
            addSimpleWidget(this, "Negative", env_negative);

            requestAnimationFrame(() => {
                const sz = this.computeSize();
                if (sz[0] < this.size[0]) sz[0] = this.size[0];
                if (sz[1] < this.size[1]) sz[1] = this.size[1];
                this.onResize?.(sz);
                app.graph.setDirtyCanvas(true, false);
            });
        }

        // When the node is executed we will be sent the input text, display this in the widgets
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            //console.log("env_scene:", message.env_scene);
            populate.call(this, message.env_scene, message.env_style, message.env_negative);
        };

        const VALUES = Symbol();
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            this[VALUES] = {
                env_scene: arguments[0]?.env_scene,
                env_style: arguments[0]?.env_style,
                env_negative: arguments[0]?.env_negative
            };
            return configure?.apply(this, arguments);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            const vals = this[VALUES];
            if (vals && (vals.env_scene && vals.env_style && vals.env_negative)) {
                //console.log("env_scene:", vals.env_scene);
                requestAnimationFrame(() => {
                    populate.call(this, vals.env_scene, vals.env_style, vals.env_negative);
                });
            }
        };
    },
});