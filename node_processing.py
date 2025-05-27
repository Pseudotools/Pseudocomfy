
import copy
import torch

from .helpers.imgutil import tensor_to_base64
from .helpers.imgutil import make_multiple_of_64, scale_tensor_image


class PseudoProcessEnvironmentalPrompts:
    """
    Utility class for previewing environmental prompts.
    Returns environmental prompt inputs unaltered.
    Inputs:
        env_scene (str): Scene description for the environment prompt.
        env_style (str): Style description for the environment prompt.
        env_negative (str): Negative prompt for the environment.
    Outputs:
        env_scene (str): Deep-copied scene description.
        env_style (str): Deep-copied style description.
        env_negative (str): Deep-copied negative prompt.
        env_all (str): Concatenated string of scene, style, and negative prompt.
    """
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
    RETURN_TYPES = ("STRING","STRING","STRING","STRING",)
    RETURN_NAMES = ("env_scene","env_style","env_negative","env_all",)
    FUNCTION = "notify"
    OUTPUT_NODE = True

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

        parts = [s for s in [env_scene, env_style, env_negative] if s is not None and str(s).strip() != ""]
        env_all = "; ".join(parts)

        return { #comfyui expects all values in ui to be wrapped in a list
            "ui": {"env_scene": [env_scene], "env_style": [env_style], "env_negative": [env_negative]}, 
            "result": (env_scene, env_style, env_negative, env_all,)
            }
   

class PseudoProcessMaterialPrompts:
    """
    Utility class for previewing material prompts
    Returns inputs unaltered.
    Inputs:
        mat_txts_lst (list of str): List of material prompt texts.
        mat_imgs_lst (list of tensor): List of image tensors as [1, H, W, 3] corresponding to the material prompts.
        mat_msks_lst (list of tensor): List of mask tensors as [1, H, W] corresponding to the material prompts.
    Outputs:
        mat_txts (list of str): Deep-copied list of material prompt texts.
        mat_imgs (list of tensor): Deep-copied list of image tensors.
        mat_msks (list of tensor): Deep-copied list of mask tensors.
        mat_txts_all (str): Concatenated string of all material prompt texts.
    Additional Information:
        - The class encodes image and mask tensors to base64 for UI display.
        - All outputs are wrapped in lists to comply with ComfyUI requirements.
        - Designed for use in the "Pseudocomfy/Utils" category.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mat_txts_lst": ("STRING", {"forceInput": True}),
                "mat_imgs_lst": ("IMAGE", {"forceInput": True}),
                "mat_msks_lst": ("MASK", {"forceInput": True}),
            }
        }

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,True,True,False,)
    RETURN_TYPES = ("STRING", "IMAGE", "MASK", "STRING",)
    RETURN_NAMES = ("mat_txts", "mat_imgs", "mat_msks", "mat_txts_all",)
    FUNCTION = "func"
    OUTPUT_NODE = True

    CATEGORY = "Pseudocomfy/Utils"

    def func(self, mat_txts_lst, mat_imgs_lst, mat_msks_lst):
        print("[pseudocomfy] ProcessMaterialPrompts")
        
        # all inputs are expected to be lists
        mat_txts = mat_txts_lst 
        mat_imgs = mat_imgs_lst
        mat_msks = mat_msks_lst
        
        mat_imgs_b64 = [tensor_to_base64(t) for t in mat_imgs]
        mat_msks_b64 = [tensor_to_base64(t) for t in mat_msks]
        
        # concatenate all mask texts
        parts = [s for s in mat_txts if s is not None and str(s).strip() != ""]
        mat_txts_all = "; ".join(parts)
        #print(f"\tmat_txts_all: '{mat_txts_all}'", parts)

        return { #comfyui expects all values in ui to be wrapped in a list, since these are all lists we're fine.
            "ui": {"mat_txts": mat_txts, "mat_imgs": mat_imgs_b64, "mat_msks": mat_msks_b64}, 
            "result": (
                    copy.deepcopy(mat_txts),
                    copy.deepcopy(mat_imgs),
                    copy.deepcopy(mat_msks),
                    mat_txts_all,
                )
            }


class PseudoProcessImagePrompt:
    """
    Utility class for scaling images and returning both the scaled image and relevant metadata.
    Inputs:
        given_width (int): The original width of the image.
        given_height (int): The original height of the image.
        img (tensor): The input image tensor, expected shape [1, H, W, 3].
        scale_by (float): The scaling factor to apply to the image dimensions (default: 2.0, min: 1.0, max: 4.0, step: 0.5).
    Outputs:
        scaled_width (int): The width of the scaled image (multiple of 64).
        scaled_height (int): The height of the scaled image (multiple of 64).
        img (tensor): The scaled image tensor, same shape as given.
    """
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
        print(f"[pseudocomfy] ProcessImagePrompt\n\tgiven w,h: ({given_width},{given_height})\n\tscale_by: {scale_by}\n\timg: {tuple(img.shape)}")
        
        scaled_width = int(make_multiple_of_64(given_width * scale_by))
        scaled_height = int(make_multiple_of_64(given_height * scale_by))
        image = scale_tensor_image(img, scaled_width, scaled_height)
        #print("completed scaling to: ", scaled_width, scaled_height, img.shape)
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
 
