from .loaders import *
from .processors import *
from .utils import *
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

    "PreviewEnvironmentalPrompts": PreviewEnvironmentalPrompts,
    "PreviewMaterialPrompts": PreviewMaterialPrompts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    #"MakeMaskBatch": "Batch Masks",

    #"LoadJSONAuto": "Load JSON (Auto)",
    "LoadModelSnapshot": "Load Model Snapshot",
    "UnpackModelSnapshot": "Unpack Model Snapshot",

    "ApplyDenseDiffusion": "Apply Dense Diffusion",
    "ApplyIPAdaper": "Apply IPAdaper",

    "PreviewEnvironmentalPrompts": "Preview Environmental Prompts", 
    "PreviewMaterialPrompts": "Preview Material Prompts",   
}