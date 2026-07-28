from .node_conditioning import *
from .node_io import *
from .node_ipadapter_loader import *
from .node_loaders_snapshot import *
from .node_loaders_vetted import *
from .node_processing import *
from .node_utils import *
from .node_vars import *
from . import api


# =============================================================================
# === GLOBAL ===
# =============================================================================

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


# =============================================================================
# === REGISTRATION ===
# =============================================================================

NODE_CLASS_MAPPINGS = {

    # loaders
    "PseudoLoadModelSnapshot": PseudoLoadModelSnapshot,
    "PseudoUnpackModelSnapshot": PseudoUnpackModelSnapshot,
    "PseudoVettedCheckpointLoader": PseudoVettedCheckpointLoader,

    # io
    "PseudoSaveImageWithEmbeddedMasks": PseudoSaveImageWithEmbeddedMasks,

    # conditioning
    "PseudoApplyDenseDiffusionSDXL": PseudoApplyDenseDiffusionSDXL,
    "PseudoApplyIPAdaperSDXL": PseudoApplyIPAdaperSDXL,
    "PseudoIPAdapterUnifiedLoaderClone": PseudoIPAdapterUnifiedLoaderClone,

    # processing
    "PseudoProcessEnvironmentalPrompts": PseudoProcessEnvironmentalPrompts,
    "PseudoProcessMaterialPrompts": PseudoProcessMaterialPrompts,
    "PseudoProcessImagePrompt": PseudoProcessImagePrompt,

    # utils
    "PseudoMaskBlur": PseudoMaskBlur,
    "PseudoMaskClamp": PseudoMaskClamp,
    "PseudoMaskRemap": PseudoMaskRemap,
    "PseudoMaskInvert": PseudoMaskInvert,
    "PseudoMaskReshape": PseudoMaskReshape,
    "PseudoMaskAggregate": PseudoMaskAggregate,

    "PseudoPreviewStrings": PseudoPreviewStrings,
    "PseudoConcatStrings": PseudoConcatStrings,

    "PseudoRemapNormalizedFloat": PseudoRemapNormalizedFloat,
    "PseudoFloatToInt": PseudoFloatToInt,

    # variables
    "PseudoVarFloat": PseudoVarFloat,
    "PseudoVarInt": PseudoVarInt,
    "PseudoVarString": PseudoVarString,

    # seed
    "PseudoSeed": PseudoSeed,
}

NODE_DISPLAY_NAME_MAPPINGS = {

    # loaders
    "PseudoLoadModelSnapshot": "Load Model Snapshot",
    "PseudoUnpackModelSnapshot": "Unpack Model Snapshot",
    "PseudoVettedCheckpointLoader": "Vetted Checkpoint Loader",

    # io
    "PseudoSaveImageWithEmbeddedMasks": "Save Image with Embedded Masks",

    # conditioning
    "PseudoApplyDenseDiffusionSDXL": "Apply Dense Diffusion Conditioning (SDXL)",
    "PseudoApplyIPAdaperSDXL": "Apply IPAdaper Conditioning (SDXL)",
    "PseudoIPAdapterUnifiedLoaderClone": "IPAdapter Unified Loader (Clone)",

    # processing
    "PseudoProcessEnvironmentalPrompts": "Process Environmental Prompt Guidence", 
    "PseudoProcessMaterialPrompts": "Process Material Prompt Guidence",  
    "PseudoProcessImagePrompt": "Process Image Prompt", 

    # utils
    "PseudoMaskBlur": "Blur Mask",
    "PseudoMaskClamp": "Clamp Mask",
    "PseudoMaskRemap": "Remap Mask",
    "PseudoMaskInvert": "Invert Mask",
    "PseudoMaskReshape": "Reshape Mask",
    "PseudoMaskAggregate": "Aggregate Masks",

    "PseudoPreviewStrings": "Preview Strings",
    "PseudoConcatStrings": "Concat Strings",

    "PseudoRemapNormalizedFloat": "Remap Float",
    "PseudoFloatToInt": "Float to Int",

    #variables
    "PseudoVarFloat": "Float Variable",
    "PseudoVarInt": "Int Variable",
    "PseudoVarString": "String Variable",

    # seed
    "PseudoSeed": "Seed",
}