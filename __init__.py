from .node_conditioning import *
from .node_io import *
from .node_ipadapter_loader import *
from .node_loaders import *
from .node_processing import *
from .node_utils import *
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
    "LoadModelSnapshot": LoadModelSnapshot,
    "UnpackModelSnapshot": UnpackModelSnapshot,

    # io
    "SaveImageWithEmbeddedMasks": SaveImageWithEmbeddedMasks,

    # conditioning
    "ApplyDenseDiffusionSDXL": ApplyDenseDiffusionSDXL,
    "ApplyIPAdaperSDXL": ApplyIPAdaperSDXL,
    "IPAdapterUnifiedLoaderClone": IPAdapterUnifiedLoaderClone,

    # processing
    "ProcessEnvironmentalPrompts": ProcessEnvironmentalPrompts,
    "ProcessMaterialPrompts": ProcessMaterialPrompts,
    "ProcessImagePrompt": ProcessImagePrompt,

    # utils
    "BlurMask": BlurMask,
    "PreviewStrings": PreviewStrings,
    "ConcatStrings": ConcatStrings,
}

NODE_DISPLAY_NAME_MAPPINGS = {

    # loaders
    "LoadModelSnapshot": "Load Model Snapshot",
    "UnpackModelSnapshot": "Unpack Model Snapshot",

    # io
    "SaveImageWithEmbeddedMasks": "Save Image with Embedded Masks",

    # conditioning
    "ApplyDenseDiffusionSDXL": "Apply Dense Diffusion Conditioning (SDXL)",
    "ApplyIPAdaperSDXL": "Apply IPAdaper Conditioning (SDXL)",
    "IPAdapterUnifiedLoaderClone": "IPAdapter Unified Loader (Clone)",

    # processing
    "ProcessEnvironmentalPrompts": "Process Environmental Prompt Guidence", 
    "ProcessMaterialPrompts": "Process Material Prompt Guidence",  
    "ProcessImagePrompt": "Process Image Prompt", 

    # utils
    "BlurMask": "Blur Mask",
    "PreviewStrings": "Preview Strings",
    "ConcatStrings": "Concat Strings",
}