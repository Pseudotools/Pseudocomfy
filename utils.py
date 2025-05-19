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
from .helpers.helpers import mask_to_image, tensor_to_base64
import copy
import base64, io
from PIL import Image

#print("[pseudocomfy]\t\t init from utils.py")

class MakeMaskBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK_LIST",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "append"

    CATEGORY = "Pseudocomfy/Utils"

    def append(self, masks):
        if not masks:
            # Create a blank image with the same shape as a normal mask image
            # Try to get shape from a dummy mask if possible
            dummy = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
            blank = mask_to_image(dummy)
            return (blank,)
        
        result = mask_to_image(masks[0])
        if len(masks) > 1:
            for i in range(1, len(masks)):
                result = torch.cat((result,  mask_to_image(masks[i])), 0)


        return (result,)
    

class ComboNodeCozy:
    """
    A class to represent a dynamic combo changer in ComfyUI.
    """
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blank_image",)
    CATEGORY = "Pseudocomfy/Utils"
    OUTPUT_NODE = True # marks this node as an output node, executes even with nothing attached
    FUNCTION = "func"

    #print("[pseudocomfy]\t\t init from ComboNodeCozy")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "test_input": ("IMAGE",),
        }}

    def func(self, test_input):
        # Return a blank RGB image (1, 3, 256, 256)
        #print("[pseudocomfy]\t\t ComboNodeCozy.func() called")
        blank = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
        return (blank,)
    
    @classmethod
    def IS_CHANGED(s, test_input):
        m = hashlib.sha256()
        current_time = str(time.time())
        m.update(current_time.encode('utf-8'))

        return m.digest().hex()


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
                print("[pseudocomfy]\t\tError: extra_pnginfo is not a valid list or missing 'workflow' key")

        return {
            "ui": {"env_scene": [env_scene], "env_style": [env_style], "env_negative": [env_negative]}, # not sure why these need to be wrapped in a list 
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
        print("[pseudocomfy]\t\t PreviewMaterialPrompts.notify() called")
        print("[pseudocomfy]\t\t mat_msks is len: ", len(mat_msks))
        
        mat_imgs_b64 = [tensor_to_base64(t) for t in mat_imgs]
        mat_msks_b64 = [tensor_to_base64(t) for t in mat_msks]
        
        return {
            "ui": {"mat_txts": mat_txts, "mat_imgs": mat_imgs_b64, "mat_msks": mat_msks_b64}, 
            "result": (copy.deepcopy(mat_txts),copy.deepcopy(mat_imgs),copy.deepcopy(mat_msks),)
            }
    
