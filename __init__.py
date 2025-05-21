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
    "LoadModelSnapshot": LoadModelSnapshot,
    "UnpackModelSnapshot": UnpackModelSnapshot,

    "ApplyDenseDiffusionSDXL": ApplyDenseDiffusionSDXL,
    "ApplyIPAdaperSDXL": ApplyIPAdaperSDXL,
    "IPAdapterUnifiedLoaderClone": IPAdapterUnifiedLoaderClone,

    "PreviewEnvironmentalPrompts": PreviewEnvironmentalPrompts,
    "PreviewMaterialPrompts": PreviewMaterialPrompts,
    "ProcessImagePrompt": ProcessImagePrompt,

    "BlurMask": BlurMask,

    "PreviewStrings": PreviewStrings,
    "ConcatStrings": ConcatStrings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadModelSnapshot": "Load Model Snapshot",
    "UnpackModelSnapshot": "Unpack Model Snapshot",

    "ApplyDenseDiffusionSDXL": "Apply Dense Diffusion Conditioning (SDXL)",
    "ApplyIPAdaperSDXL": "Apply IPAdaper Conditioning (SDXL)",
    "IPAdapterUnifiedLoaderClone": "IPAdapter Unified Loader (Clone)",

    "PreviewEnvironmentalPrompts": "Preview Environmental Prompt Guidence", 
    "PreviewMaterialPrompts": "Preview Material Prompt Guidence",  
    "ProcessImagePrompt": "Process Image Prompt", 

    "BlurMask": "Blur Mask",

    "PreviewStrings": "Preview Strings",
    "ConcatStrings": "Concat Strings",
}