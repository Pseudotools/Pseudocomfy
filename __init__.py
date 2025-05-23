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
    "MaskBlur": MaskBlur,
    "MaskClamp": MaskClamp,
    "MaskRemap": MaskRemap,
    "MaskInvert": MaskInvert,
    "MaskReshape": MaskReshape,
    "MaskAggregate": MaskAggregate,

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
    "MaskBlur": "Blur Mask",
    "MaskClamp": "Clamp Mask",
    "MaskRemap": "Remap Mask",
    "MaskInvert": "Invert Mask",
    "MaskReshape": "Reshape Mask",
    "MaskAggregate": "Aggregate Masks",

    "PreviewStrings": "Preview Strings",
    "ConcatStrings": "Concat Strings",
}