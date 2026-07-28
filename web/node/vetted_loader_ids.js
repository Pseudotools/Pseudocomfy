import { app } from "../../../scripts/app.js";

const VETTED_LOADER_CONFIGS = [
    { nodeName: "PseudoVettedCheckpointLoader", comboWidget: "model", idWidget: "model_id" },
    { nodeName: "PseudoVettedControlNetLoader", comboWidget: "model", idWidget: "model_id" },
    { nodeName: "PseudoVettedLoraLoader",       comboWidget: "lora",  idWidget: "model_id" },
    { nodeName: "PseudoVettedClipLoader",       comboWidget: "model", idWidget: "model_id" },
    { nodeName: "PseudoVettedVaeLoader",        comboWidget: "model", idWidget: "model_id" },
];

for (const { nodeName, comboWidget, idWidget } of VETTED_LOADER_CONFIGS) {
    app.registerExtension({
        name: `pseudocomfy.${nodeName}`,
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== nodeName) return;

            const modelIds = nodeData.input?.required?.[comboWidget]?.[1]?.model_ids ?? {};

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const comboW = this.widgets?.find(w => w.name === comboWidget);
                const idW = this.widgets?.find(w => w.name === idWidget);
                if (!comboW || !idW) return;

                const origCallback = comboW.callback;
                comboW.callback = function (value) {
                    origCallback?.apply(this, arguments);
                    idW.value = modelIds[value] ?? "";
                };
            };
        },
    });
}
