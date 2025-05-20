# ==============================================================================
# This file contains code that has been adapted or directly copied from the
# ComfyUI-Impact-Pack package by Dr.Lt.Data ("ltdrdata").
# Original source: https://github.com/ltdrdata/ComfyUI-Impact-Pack
#
# This code is used under the terms of the original license, with modifications
# made to suit the needs of this project.
# ==============================================================================
import torch
import torch.nn.functional as F

import time, hashlib
from .helpers.imgutil import tensor_to_base64
import copy
import base64, io
from PIL import Image

from .helpers.imgutil import make_multiple_of_64, scale_tensor_image


class PreviewEnvironmentalPrompts:
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
    """
    Utility class for previewing material prompts
    Returns inputs unaltered.
    Inputs:
        mat_txts (list of str): List of material prompt texts.
        mat_imgs (list of tensor): List of image tensors as [1, H, W, 3] corresponding to the material prompts.
        mat_msks (list of tensor): List of mask tensors as [1, H, W] corresponding to the material prompts.
    Outputs:
        mat_txts (list of str): Deep-copied list of material prompt texts.
        mat_imgs (list of tensor): Deep-copied list of image tensors.
        mat_msks (list of tensor): Deep-copied list of mask tensors.
    Additional Information:
        - The class encodes image and mask tensors to base64 for UI display.
        - All outputs are wrapped in lists to comply with ComfyUI requirements.
        - Designed for use in the "Pseudocomfy/Utils" category.
    """
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
        print(f"[pseudocomfy] ProcessImagePrompt\n\tgiven: {given_width}x{given_height}\n\tscale_by: {scale_by}\n\timg: {img.shape}")
        
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
 

class BlurMask:
    """
    Utility class for applying a Gaussian blur to image masks.
    Inputs:
        msk (tensor): The input image tensor. Expected shape is [1, H, W]
        blur_radius (int): The radius of the Gaussian blur kernel (default: 1, min: 1, max: 31, step: 1).
        sigma (float): The standard deviation of the Gaussian kernel (default: 1.0, min: 0.1, max: 10.0, step: 0.1).
    Outputs:
        msk (tensor): The blurred image tensor, with the same shape as the input.
    """        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "msk": ("IMAGE",),
                "blur_radius": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 31,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("msk",)
    FUNCTION = "blur"
    CATEGORY = "Pseudocomfy/Utils"

    def blur(self, msk: torch.Tensor, blur_radius: int, sigma: float):
        """
        Expects (1, H, W)
        may also work with (B, H, W, C), or (B, C, H, W) tensors (untested).
        """
        if blur_radius == 0:
            return (msk,)

        device = msk.device
        # Handle (1, H, W) grayscale, (B, H, W, C) color, or (B, C, H, W)
        if msk.ndim == 3:  # (1, H, W) or (B, H, W)
            msk = msk.unsqueeze(-1)  # (1, H, W, 1)
        if msk.ndim == 4 and msk.shape[-1] <= 4:  # (B, H, W, C)
            msk = msk.permute(0, 3, 1, 2)  # (B, C, H, W)
        # Now image is (B, C, H, W)
        B, C, H, W = msk.shape

        kernel_size = blur_radius * 2 + 1
        kernel = self.gaussian_kernel(kernel_size, sigma, device=device)
        kernel = kernel.expand(C, 1, kernel_size, kernel_size)

        pad = blur_radius
        padded_image = F.pad(msk, (pad, pad, pad, pad), mode='reflect')
        blurred = F.conv2d(padded_image, kernel, padding=0, groups=C)
        # Remove extra padding
        blurred = blurred[:, :, pad:-pad, pad:-pad]

        # Return to (B, H, W, C) if input was that, or (1, H, W) if grayscale
        if blurred.shape[1] == 1:
            blurred = blurred.permute(0, 2, 3, 1).squeeze(-1)  # (B, H, W)
        else:
            blurred = blurred.permute(0, 2, 3, 1)  # (B, H, W, C)
        return (blurred,)
    
    def gaussian_kernel(kernel_size, sigma, device):
        """Create a 2D Gaussian kernel."""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
        grid = coords.unsqueeze(0) ** 2 + coords.unsqueeze(1) ** 2
        kernel = torch.exp(-0.5 * grid / sigma ** 2)
        kernel = kernel / kernel.sum()
        return kernel
