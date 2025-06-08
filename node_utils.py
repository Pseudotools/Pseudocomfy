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




class PseudoMaskBlur:
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
                "msks_list": ("MASK",),
                "blur_radius": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 51,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
                "invert": ("BOOLEAN", {"default": False}),
            },
        }

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("msks",)
    FUNCTION = "blur"
    CATEGORY = "Pseudocomfy/Utils"

    def blur(self, msks_list, blur_radius: int, sigma: float, invert):
        """
        Apply Gaussian blur to a list of masks.
        Expects each mask to be (1, H, W) or (B, H, W)
        """
        # given that INPUT_IS_LIST, msks_list is a list of masks
        # if inputs that are meant to be singletons are a list, use the first element
        if isinstance(blur_radius, list) and len(blur_radius)>0: blur_radius = blur_radius[0]
        if isinstance(sigma, list) and len(sigma)>0: sigma = sigma[0]
        if isinstance(invert, list) and len(invert)>0: invert = invert[0]

        
        print(f"[pseudocomfy] BlurMask blur_radius:{blur_radius} sigma:{sigma}")

        results = []
        for msk in msks_list:
            print(f"\tmsk shape:{tuple(msk.shape)}")

            if blur_radius == 0:
                results.append(msk)
                continue

            device = msk.device

            # Ensure batch dimension
            if msk.ndim == 2:
                msk = msk.unsqueeze(0)  # (1, H, W)

            B, H, W = msk.shape

            # Add channel dimension for conv2d
            msk = msk.unsqueeze(1)  # (B, 1, H, W)

            kernel_size = blur_radius * 2 + 1
            kernel = self.gaussian_kernel(kernel_size, sigma, device=device)
            kernel = kernel.expand(1, 1, kernel_size, kernel_size)

            pad = blur_radius
            padded_image = F.pad(msk, (pad, pad, pad, pad), mode='reflect')
            blurred = F.conv2d(padded_image, kernel, padding=0, groups=1)
            blurred = blurred[:, :, pad:-pad, pad:-pad]  # Remove extra padding

            # Remove channel dimension
            result = blurred.squeeze(1)  # (B, H, W)

            if invert:
                result = 1.0 - result
                result = torch.clamp(result, 0.0, 1.0)

            results.append(result)
        return (results,)
    
    def gaussian_kernel(self, kernel_size, sigma, device):
        """Create a 2D Gaussian kernel."""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
        grid = coords.unsqueeze(0) ** 2 + coords.unsqueeze(1) ** 2
        kernel = torch.exp(-0.5 * grid / sigma ** 2)
        kernel = kernel / kernel.sum()
        return kernel

class PseudoMaskClamp:
    """
    Utility class for clamping mask values to a specified range.
    Inputs:
        msk (tensor): The input mask tensor. Expected shape is [1, H, W] or [B, H, W], values in [0, 1].
        min_val (float): Minimum value to clamp to (default: 0.0, min: 0.0, max: 1.0, step: 0.01).
        max_val (float): Maximum value to clamp to (default: 1.0, min: 0.0, max: 1.0, step: 0.01).
    Outputs:
        msk (tensor): The clamped mask tensor, same shape as input.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "msks_list": ("MASK",),
                "min_val": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "max_val": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
            }
        }

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("msks",)
    FUNCTION = "clamp"
    CATEGORY = "Pseudocomfy/Utils"

    def clamp(self, msks_list, min_val: float, max_val: float):
        """
        Clamp mask values to [min_val, max_val] for a list of masks.
        """
        # given that INPUT_IS_LIST, msks_list is a list of masks
        # if inputs that are meant to be singletons are a list, use the first element
        if isinstance(min_val, list) and len(min_val)>0: min_val = min_val[0]
        if isinstance(max_val, list) and len(max_val)>0: max_val = max_val[0]

        print(f"[pseudocomfy] ClampMask")

        results = []
        for msk in msks_list:
            print(f"\tmsk shape:{tuple(msk.shape)}")
            print(f"\tclamping mask of ({msk.min():.3f} -> {msk.max():.3f}) to ({min_val:.3f} -> {max_val:.3f})")
            clamped = torch.clamp(msk, min=min_val, max=max_val)
            results.append(clamped)
        return (results,)

class PseudoMaskRemap:
    """
    Utility class for remapping mask values from a source range to a target range.
    Inputs:
        msk (tensor): The input mask tensor. Expected shape is [1, H, W] or [B, H, W], values in [0, 1].
        src_min (float, optional): Source minimum value to remap from. If not set, uses mask min.
        src_max (float, optional): Source maximum value to remap from. If not set, uses mask max.
        tgt_min (float, optional): Target minimum value to remap to. If not set, uses 0.0.
        tgt_max (float, optional): Target maximum value to remap to. If not set, uses 1.0.
    Outputs:
        msk (tensor): The remapped mask tensor, same shape as input.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "msks_list": ("MASK",),
            },
            "optional": {
                "src_min": ("FLOAT", {"forceInput": True}), # force input b/c can't have widget for None
                "src_max": ("FLOAT", {"forceInput": True}), # force input b/c can't have widget for None
                "tgt_min": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "tgt_max": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("msks",)
    FUNCTION = "remap"
    CATEGORY = "Pseudocomfy/Utils"

    def remap(self, msks_list, src_min: float = None, src_max: float = None, tgt_min: float = 0.0, tgt_max: float = 1.0):
        """
        Remap mask values from [src_min, src_max] to [tgt_min, tgt_max] for a list of masks.
        """
        # given that INPUT_IS_LIST, msks_list is a list of masks
        # if inputs that are meant to be singletons are a list, use the first element
        if isinstance(src_min, list) and len(src_min)>0: src_min = src_min[0]
        if isinstance(src_max, list) and len(src_max)>0: src_max = src_max[0]
        if isinstance(tgt_min, list) and len(tgt_min)>0: tgt_min = tgt_min[0]
        if isinstance(tgt_max, list) and len(tgt_max)>0: tgt_max = tgt_max[0]

        print(f"[pseudocomfy] RemapMask")

        results = []
        for msk in msks_list:
            mask_min = float(msk.min())
            mask_max = float(msk.max())
            from_min = src_min if src_min is not None else mask_min
            from_max = src_max if src_max is not None else mask_max
            to_min = tgt_min
            to_max = tgt_max
            
            print(f"\tmsk shape:{tuple(msk.shape)}")
            print(f"\tremapping mask of ({mask_min:.3f} -> {mask_max:.3f}) from ({from_min:.3f} -> {from_max:.3f}) to ({to_min:.3f} -> {to_max:.3f})")

            # Avoid division by zero
            if from_max - from_min == 0:
                remapped = torch.full_like(msk, to_min)
            else:
                norm = (msk - from_min) / (from_max - from_min)
                remapped = norm * (to_max - to_min) + to_min
                remapped = torch.clamp(remapped, min(min(to_min, to_max), 0.0), max(max(to_min, to_max), 1.0))
            results.append(remapped)
        return (results,)

class PseudoMaskInvert:
    """
    Utility class for inverting mask values (1 - mask).
    Inputs:
        msk (tensor): The input mask tensor. Expected shape is [1, H, W] or [B, H, W], values in [0, 1].
    Outputs:
        msk (tensor): The inverted mask tensor, same shape as input.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "msks_list": ("MASK",),
            }
        }

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("msks",)
    FUNCTION = "invert"
    CATEGORY = "Pseudocomfy/Utils"

    def invert(self, msks_list):
        """
        Invert mask values (1 - mask) for a list of masks.
        """
        # given that INPUT_IS_LIST, msks_list is a list of masks
        
        print(f"[pseudocomfy] MaskInvert")        

        results = []
        for msk in msks_list:
            mask_min = float(msk.min())
            mask_max = float(msk.max())
            print(f"\tmsk shape:{tuple(msk.shape)}")
            print(f"\tinverting mask of ({mask_min:.3f} -> {mask_max:.3f}) to ({1-mask_max:.3f} -> {1-mask_min:.3f})")
            inverted = 1.0 - msk
            inverted = torch.clamp(inverted, 0.0, 1.0)
            results.append(inverted)
        return (results,)

class PseudoMaskReshape:
    """
    Utility class for morphological operations on masks: Erode (shrink) and Dilate (expand) white regions.
    Inputs:
        msk (tensor): The input mask tensor. Expected shape is [1, H, W] or [B, H, W], values in [0, 1].
        operation (str): "erode" or "dilate".
        kernel_size (int): Size of the square structuring element (default: 3, min: 1, max: 31, step: 2).
        iterations (int): Number of times to apply the operation (default: 1, min: 1, max: 10, step: 1).
    Outputs:
        msk (tensor): The processed mask tensor, same shape as input.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "msks_list": ("MASK",),
                "operation": (["erode (shrink white areas)", "dilate (grow white areas)"], {"default": "dilate (grow white areas)"}),
                "kernel_size": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 31,
                    "step": 2
                }),
                "iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("msks",)
    FUNCTION = "morph"
    CATEGORY = "Pseudocomfy/Utils"

    def morph(self, msks_list, operation: str, kernel_size: int, iterations: int, invert):
        """
        Apply morphological operation to a list of masks.
        """
        # given that INPUT_IS_LIST, msks_list is a list of masks
        # if inputs that are meant to be singletons are a list, use the first element
        if isinstance(operation, list) and len(operation)>0: operation = operation[0]
        if isinstance(kernel_size, list) and len(kernel_size)>0: kernel_size = kernel_size[0]
        if isinstance(iterations, list) and len(iterations)>0: iterations = iterations[0]
        if isinstance(invert, list) and len(invert)>0: invert = invert[0]

        
        print(f"[pseudocomfy] MaskMorphology")
        print(f"\toperation: {operation}, kernel_size: {kernel_size}, iterations: {iterations}, invert: {invert}")

        results = []
        for msk in msks_list:
            print(f"\tmsk shape:{tuple(msk.shape)}")

            # Ensure batch dimension
            if msk.ndim == 2:
                msk = msk.unsqueeze(0)  # (1, H, W)
            B, H, W = msk.shape

            # Add channel dimension for conv2d
            msk = msk.unsqueeze(1)  # (B, 1, H, W)

            # Create structuring element (kernel)
            kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=msk.dtype, device=msk.device)

            result = msk
            for _ in range(iterations):
                if operation[:3] == "ero":
                    result = torch.nn.functional.max_pool2d(
                        1.0 - result, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
                    )
                    result = 1.0 - result
                elif operation[:3] == "dil":
                    result = torch.nn.functional.max_pool2d(
                        result, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
                    )
                else:
                    raise ValueError(f"Unknown operation: {operation}")

            # Remove channel dimension
            result = result.squeeze(1)  # (B, H, W)
            
            if invert:
                result = 1.0 - result
                result = torch.clamp(result, 0.0, 1.0)

            print(f"\tresult range: ({result.min():.3f} -> {result.max():.3f})")
            results.append(result)
        return (results,)


class PseudoMaskAggregate:
    """
    Utility class for combining a list of masks into a single mask using various arithmetic operations.
    Inputs:
        msks_list (list[tensor]): List of mask tensors, each [1, H, W] or [B, H, W], values in [0, 1].
        operation (str): Operation to perform. One of:
            - "sum_clamped": Sum all masks, clamp to [0, 1].
            - "sum_normalized": Sum all masks, then normalize result to [0, 1].
            - "average": Pixelwise mean of all masks.
            - "max": Pixelwise max of all masks.
            - "min": Pixelwise min of all masks.
        invert (bool): If True, invert the result mask (1 - mask).
    Outputs:
        msk (tensor): The combined mask tensor, same shape as input.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "msks_list": ("MASK", {"forceInput": True, "isList": True}),
                "operation": (["sum_clamped", "sum_normalized", "average", "max", "min"], {}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("msk",)
    FUNCTION = "combine"
    CATEGORY = "Pseudocomfy/Utils"

    def combine(self, msks_list, operation, invert):
        # given that INPUT_IS_LIST, msks_list is a list of masks
        # if inputs that are meant to be singletons are a list, use the first element
        if isinstance(operation, list) and len(operation)>0: operation = operation[0]
        if isinstance(invert, list) and len(invert)>0: invert = invert[0]

        # Stack masks to shape (N, H, W) or (N, 1, H, W)
        msks = [m.float() for m in msks_list]
        stack = torch.stack(msks, dim=0)
        print(f"[pseudocomfy] CombineMasks")
        print(f"\toperation: {operation}")
        print(f"\tinput count: {len(msks)}")
        print(f"\tstack shape: {tuple(stack.shape)}")

        if operation == "sum_clamped":
            result = stack.sum(dim=0)
            result = torch.clamp(result, 0.0, 1.0)
        elif operation == "sum_normalized":
            result = stack.sum(dim=0)
            minv, maxv = result.min(), result.max()
            if maxv - minv == 0:
                result = torch.zeros_like(result)
            else:
                result = (result - minv) / (maxv - minv)
        elif operation == "average":
            result = stack.mean(dim=0)
        elif operation == "max":
            result = stack.max(dim=0).values
        elif operation == "min":
            result = stack.min(dim=0).values
        else:
            raise ValueError(f"Unknown operation: {operation}")

        if invert:
            result = 1.0 - result
            result = torch.clamp(result, 0.0, 1.0)

        print(f"\tresult shape: {tuple(result.shape)}")
        print(f"\tresult range: ({result.min():.3f} -> {result.max():.3f})")
        return (result,)



class PseudoPreviewStrings:
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


class PseudoConcatStrings:
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


class PseudoRemapNormalizedFloat:
    """
    Utility class for remapping a float value from a source range to a target range.
    Inputs:
        value (float): The input float value.
        src_min (float): Source minimum value (default: 0.0).
        src_max (float): Source maximum value (default: 1.0).
        tgt_min (float): Target minimum value (default: 0.0).
        tgt_max (float): Target maximum value (default: 1.0).
    Outputs:
        value (float): The remapped float value.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "val": ("FLOAT", {"forceInput": True}),
                "tgt_min": ("FLOAT", {"default": 0.0}),
                "tgt_max": ("FLOAT", {"default": 1.0}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("val",)
    FUNCTION = "remap"
    CATEGORY = "Pseudocomfy/Utils"

    def remap(self, val, tgt_min, tgt_max):
        print(f"[pseudocomfy] RemapFloat")
        print(f"\tvalue: {val}")
        print(f"\t0-1 ==> {tgt_min}-{tgt_max}")

        src_min, src_max = 0.0, 1.0  # Default source range

        # Ensure min <= max
        if tgt_min > tgt_max:
            tgt_min, tgt_max = tgt_max, tgt_min

        # Clamp value to source domain
        val = max(min(val, src_max), src_min)

        if src_max - src_min == 0:
            remapped = tgt_min
        else:
            norm = (val - src_min) / (src_max - src_min)
            remapped = norm * (tgt_max - tgt_min) + tgt_min
        return (remapped,)
    

class PseudoFloatToInt:
    """
    Utility class for converting a float value to an integer.
    Inputs:
        val (float): The input float value.
        rounding (str): Rounding method: "round" (default), "floor", or "ceil".
    Outputs:
        value (int): The converted integer value.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "val": ("FLOAT", {"forceInput": True}),
                "rounding": (["round", "floor", "ceil"], {"default": "round"}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "convert"
    CATEGORY = "Pseudocomfy/Utils"

    def convert(self, val, rounding="round"):
        import math
        if rounding == "floor":
            result = int(math.floor(val))
        elif rounding == "ceil":
            result = int(math.ceil(val))
        else:
            result = int(round(val))
        print(f"[pseudocomfy] FloatToInt: {val} -> {result} (method: {rounding})")
        return (result,)