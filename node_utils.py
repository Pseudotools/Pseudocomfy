# ==============================================================================
# This file contains code that has been adapted or directly copied from the
# ComfyUI-Impact-Pack package by Dr.Lt.Data ("ltdrdata").
# Original source: https://github.com/ltdrdata/ComfyUI-Impact-Pack
#
# This code is used under the terms of the original license, with modifications
# made to suit the needs of this project.
# ==============================================================================
import torch
import time, hashlib
from .helpers.imgutil import tensor_to_base64
import copy
import base64, io
from PIL import Image

from .helpers.imgutil import make_multiple_of_64, scale_tensor_image



class PreviewEnvironmentalPrompts:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "env_scene": ("STRING", {"forceInput": True}),
                "env_style": ("STRING", {"forceInput": True}),
                "env_negative": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    #INPUT_IS_LIST = False
    RETURN_TYPES = ("STRING","STRING","STRING",)
    RETURN_NAMES = ("env_scene","env_style","env_negative",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    #OUTPUT_IS_LIST = (False, False, False,)

    CATEGORY = "Pseudocomfy/Utils"

    def notify(self, env_scene, env_style, env_negative, unique_id=None, extra_pnginfo=None):
        if unique_id is not None and extra_pnginfo is not None:
            # it looks like extra_pnginfo is only a list in earlier versions of comfyui
            # this component might work perfectly fine without it
            if (
                isinstance(extra_pnginfo, list)
                and len(extra_pnginfo) > 0
                and isinstance(extra_pnginfo[0], dict)
                and "workflow" in extra_pnginfo[0]
            ):
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["env_scene"] = env_scene
                    node["env_style"] = env_style
                    node["env_negative"] = env_negative
            else:
                pass
                #print("[pseudocomfy] PreviewEnvironmentalPrompts\n\tError: extra_pnginfo is not a valid list or missing 'workflow' key")

        return { #comfyui expects all values in ui to be wrapped in a list
            "ui": {"env_scene": [env_scene], "env_style": [env_style], "env_negative": [env_negative]}, 
            "result": (env_scene, env_style, env_negative)
            }
    

class PreviewMaterialPrompts:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mat_txts": ("STRING", {"forceInput": True}),
                "mat_imgs": ("IMAGE", {"forceInput": True}),
                "mat_msks": ("IMAGE", {"forceInput": True}),
            }
        }

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,True,True,)
    RETURN_TYPES = ("STRING","IMAGE","IMAGE",)
    RETURN_NAMES = ("mat_txts", "mat_imgs", "mat_msks",)
    FUNCTION = "notify"
    OUTPUT_NODE = True

    CATEGORY = "Pseudocomfy/Utils"

    def notify(self, mat_txts, mat_imgs, mat_msks):
        #print("[pseudocomfy] PreviewMaterialPrompts\n\t mat_msks is len: ", len(mat_msks))
        
        mat_imgs_b64 = [tensor_to_base64(t) for t in mat_imgs]
        mat_msks_b64 = [tensor_to_base64(t) for t in mat_msks]
        
        return { #comfyui expects all values in ui to be wrapped in a list, since these are all lists we're fine.
            "ui": {"mat_txts": mat_txts, "mat_imgs": mat_imgs_b64, "mat_msks": mat_msks_b64}, 
            "result": (copy.deepcopy(mat_txts),copy.deepcopy(mat_imgs),copy.deepcopy(mat_msks),)
            }
    
class ProcessImagePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "given_width": ("INT", {"forceInput": True}),
                "given_height": ("INT", {"forceInput": True}),
                "img": ("IMAGE", {"forceInput": True}),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "IMAGE",)
    RETURN_NAMES = ("scaled_width", "scaled_height", "img",)
    FUNCTION = "func"
    OUTPUT_NODE = True
    CATEGORY = "Pseudocomfy/Utils"

    def func(self, given_width, given_height, img, scale_by):
        print(f"[pseudocomfy] ProcessImagePrompt\n\tgiven: {given_width}x{given_height}\n\tscale_by: {scale_by}\n\timg: {img.shape}")
        
        scaled_width = int(make_multiple_of_64(given_width * scale_by))
        scaled_height = int(make_multiple_of_64(given_height * scale_by))
        image = scale_tensor_image(img, scaled_width, scaled_height)
        print("completed scaling to: ", scaled_width, scaled_height, img.shape)
        #return (scaled_width, scaled_height,image,)
        return {
                "ui": { #comfyui expects all values in ui to be wrapped in a list
                        "img": [tensor_to_base64(image)], 
                        "given_width": [given_width], 
                        "given_height": [given_height], 
                        "scaled_width": [scaled_width], 
                        "scaled_height": [scaled_height]
                    }, 
                "result": (
                    scaled_width,
                    scaled_height,
                    image,
                )
            }
    
