from .loaders import *
from .processors import *
from .utils import *
from . import api

NODE_CLASS_MAPPINGS = {
    "MakeMaskBatch": MakeMaskBatch,

    "LoadJSONAuto": LoadJSONAuto,
    "LoadJSONFromFolder": LoadJSONFromFolder,

    "ProcessJSON": ProcessJSON,
    "Combiner": Combiner,
    "MixedBuiltinCombinerIPAdaper": MixedBuiltinCombinerIPAdaper,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MakeMaskBatch": "Create Mask Batch",

    "LoadJSONAuto": "Load JSON (Auto)",
    "LoadJSONFromFolder": "Load JSON (From Folder)",

    "ProcessJSON": "Process JSON",
    "Combiner": "Combiner",
    "MixedBuiltinCombinerIPAdaper": "Mixed Builtin Combiner (with IPAdaper)",
}