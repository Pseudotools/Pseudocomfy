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
import base64, io
from PIL import Image




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
        print(f"[pseudocomfy] BlurMask blur_radius:{blur_radius} sigma:{sigma} msk shape:{tuple(msk.shape)}")

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
    
    def gaussian_kernel(self, kernel_size, sigma, device):
        """Create a 2D Gaussian kernel."""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
        grid = coords.unsqueeze(0) ** 2 + coords.unsqueeze(1) ** 2
        kernel = torch.exp(-0.5 * grid / sigma ** 2)
        kernel = kernel / kernel.sum()
        return kernel


class PreviewStrings:
    """
    Utility class for previewing a single string in the ComfyUI UI.
    Inputs:
        string (str): The string(s) to preview.
    Outputs:
        none
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strings": ("STRING", {"forceInput": True}),
            },
        }
    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "notify"
    OUTPUT_NODE = True
    CATEGORY = "Pseudocomfy/Utils"

    def notify(self, strings):
        # Optionally handle extra_pnginfo/unique_id if needed
        return {
            "ui": {"strings": strings}
        }


class ConcatStrings:
    """
    Utility class for concatenating two strings with a selectable separator.
    Inputs:
        str_a (str): The first string.
        str_b (str): The second string.
        separator (str): The separator to use ("space", "comma", or "semicolon").
    Outputs:
        result (str): The concatenated string.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "str_a": ("STRING", {"forceInput": True}),
                "str_b": ("STRING", {"forceInput": True}),
                "separator": (["space", "comma", "semicolon"],{}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("str",)
    FUNCTION = "concat"
    CATEGORY = "Pseudocomfy/Utils"

    def concat(self, str_a, str_b, separator):
        sep_map = {
            "space": " ",
            "comma": ", ",
            "semicolon": "; "
        }
        sep = sep_map.get(separator, " ")
        result = f"{str_a}{sep}{str_b}"
        return (result,)


