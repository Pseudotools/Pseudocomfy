
import copy
import torch

from .helpers.imgutil import tensor_to_base64
from .helpers.imgutil import tensor_image_resize_and_crop_to_multiple_of_64


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

    CATEGORY = "Pseudocomfy/Processing"

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
        scale_msk_to (str): The target dimension for the shorter side of the given masks (options: 512, 1024).
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
                "scale_msk_to": (["512", "1024"], {"default": "1024"}),               
            }
        }

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,True,True,False,False,False)
    RETURN_TYPES = ("STRING", "IMAGE", "MASK", "STRING","INT", "INT",)
    RETURN_NAMES = ("mat_txts", "mat_imgs", "mat_msks", "mat_txts_all", "scaled_width", "scaled_height",)
    FUNCTION = "func"
    OUTPUT_NODE = True

    CATEGORY = "Pseudocomfy/Processing"

    def func(self, mat_txts_lst, mat_imgs_lst, mat_msks_lst, scale_msk_to):
        print("[pseudocomfy] ProcessMaterialPrompts")
        
        # mat inputs are expected to be lists
        mat_txts = mat_txts_lst 
        mat_imgs = mat_imgs_lst
        mat_msks = mat_msks_lst

        # if scale_to is a list, use the first element
        if isinstance(scale_msk_to, list) and len(scale_msk_to)>0: scale_msk_to = scale_msk_to[0]     

        # verify scale_to value - ComfyUI is behaving weirdly about the default value
        if not scale_img_to or scale_img_to not in ["512", "1024"]:
            scale_img_to = "1024"        

        # Convert scale_to from string to int
        scale_to_int = int(scale_msk_to)
        if scale_to_int not in [512, 1024]:
            raise ValueError(f"Invalid scale_to value: {scale_msk_to}. Expected 512 or 1024.")             


        # convert all image and mask tensors to base64 for UI display        
        mat_imgs_b64 = [tensor_to_base64(t) for t in mat_imgs]
        mat_msks_b64 = [tensor_to_base64(t) for t in mat_msks]
        
        # Resize and crop all masks to the target scale
        mat_msks_resized = []
        scaled_width = None
        scaled_height = None
        for m in mat_msks:
            sw, sh, m_resized = tensor_image_resize_and_crop_to_multiple_of_64(m, scale_to_int)
            mat_msks_resized.append(m_resized)
            # Store the scaled dimensions from the first mask (assuming all are the same)
            if scaled_width is None and scaled_height is None:
                scaled_width = sw
                scaled_height = sh

        # concatenate all mask texts
        parts = [s for s in mat_txts if s is not None and str(s).strip() != ""]
        mat_txts_all = "; ".join(parts)
        #print(f"\tmat_txts_all: '{mat_txts_all}'", parts)

        return { #comfyui expects all values in ui to be wrapped in a list, since these are all lists we're fine.
            "ui": {"mat_txts": mat_txts, "mat_imgs": mat_imgs_b64, "mat_msks": mat_msks_b64}, 
            "result": (
                    copy.deepcopy(mat_txts),
                    copy.deepcopy(mat_imgs),
                    mat_msks_resized,
                    mat_txts_all,
                )
            }


class PseudoProcessImagePrompt:
    """
    Utility class for scaling images and returning both the scaled image and relevant metadata.
    Inputs:
        img (tensor): The input image tensor, expected shape [1, H, W, 3].
        scale_img_to (str): The target dimension for the shorter side of the given image ("512" or "1024").
    Outputs:
        scaled_width (int): The width of the scaled image (multiple of 64).
        scaled_height (int): The height of the scaled image (multiple of 64).
        img (tensor): The scaled and cropped image tensor.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "img": ("IMAGE", {"forceInput": True}),
                "scale_img_to": (["512", "1024"], {"default": "1024"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "IMAGE",)
    RETURN_NAMES = ("scaled_width", "scaled_height", "img",)
    FUNCTION = "func"
    OUTPUT_NODE = True
    CATEGORY = "Pseudocomfy/Processing"

    def func(self, img, scale_img_to):
        print(f"[pseudocomfy] ProcessImagePrompt\n\tscale_to: {scale_img_to}\n\timg: {tuple(img.shape)}")

        # verify scale_to value - ComfyUI is behaving weirdly about the default value
        if not scale_img_to or scale_img_to not in ["512", "1024"]:
            scale_img_to = "1024"

        # Convert scale_to from string to int
        scale_to_int = int(scale_img_to)
        if scale_to_int not in [512, 1024]:
            raise ValueError(f"Invalid scale_to value: {scale_img_to}. Expected 512 or 1024.")        
        
        given_height, given_width = img.shape[1], img.shape[2]
        scaled_width, scaled_height, image = tensor_image_resize_and_crop_to_multiple_of_64(img, scale_to_int)
        return {
            "ui": { #comfyui expects all values in ui to be wrapped in a list
                "given_width": [given_width],
                "given_height": [given_height],                
                "img": [tensor_to_base64(image)],
                "scaled_width": [scaled_width],
                "scaled_height": [scaled_height]
            },
            "result": (
                scaled_width,
                scaled_height,
                image,
            )
        }
