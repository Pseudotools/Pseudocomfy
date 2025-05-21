from .node_loaders import *
from .node_processors import *
from .node_utils import *
from .node_ipadapter_loader import *
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
    #"MakeMaskBatch": MakeMaskBatch,

    #"LoadJSONAuto": LoadJSONAuto,
    "LoadModelSnapshot": LoadModelSnapshot,
    "UnpackModelSnapshot": UnpackModelSnapshot,

    "ApplyDenseDiffusion": ApplyDenseDiffusion,
    "ApplyIPAdaper": ApplyIPAdaper,
    "IPAdapterUnifiedLoaderClone": IPAdapterUnifiedLoaderClone,

    "PreviewEnvironmentalPrompts": PreviewEnvironmentalPrompts,
    "PreviewMaterialPrompts": PreviewMaterialPrompts,
    "ProcessImagePrompt": ProcessImagePrompt,
    "BlurMask": BlurMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    #"MakeMaskBatch": "Batch Masks",

    #"LoadJSONAuto": "Load JSON (Auto)",
    "LoadModelSnapshot": "Load Model Snapshot",
    "UnpackModelSnapshot": "Unpack Model Snapshot",

    "ApplyDenseDiffusion": "Apply Dense Diffusion Conditioning",
    "ApplyIPAdaper": "Apply IPAdaper Conditioning",
    "IPAdapterUnifiedLoaderClone": "IPAdapter Unified Loader (Clone)",

    "PreviewEnvironmentalPrompts": "Preview Environmental Prompt Guidence", 
    "PreviewMaterialPrompts": "Preview Material Prompt Guidence",  
    "ProcessImagePrompt": "Process Image Prompt", 
    "BlurMask": "Blur Mask",
}