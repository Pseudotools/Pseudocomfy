# ==============================================================================
# This file includes snippets adapted from the built-in nodes of ComfyUI.
# Original source: https://github.com/comfyanonymous/ComfyUI
# ==============================================================================
import base64
import gzip
from PIL import Image, ImageOps
import numpy as np
import io
import torch
import torch.nn.functional as F



def create_solid_mask(value, width, height): # from builtin nodes: Create Solid Mask
    mask = torch.full((1, height, width), value, dtype=torch.float32, device="cpu")
    return mask


def make_multiple_of_64(num):
    return ((num + 63) // 64) * 64



def scale_image(input_image, width, height):
    new_width = int(width)
    new_height = int(height)
    """
    Image.ANTIALIAS filter is used for high-quality downsampling. 
    We can replace it with other filters like Image.NEAREST, Image.BILINEAR, or 
    Image.BICUBIC depending on the desired quality and performance.
    """
    scaled_image = input_image.resize((new_width, new_height), Image.BICUBIC)
    
    return scaled_image

def tensor_to_base64(tensor):
    
    """
    Converts a torch tensor (C,H,W), (1,H,W), (3,H,W), (H,W), (H,W,3), or (1,H,W,3) to a base64 PNG string.
    """
    if tensor is None:
        return None

    #print("[pseudocomfy] tensor_to_base64()\n\ttensor shape =", tuple(tensor.shape))

    arr = tensor.detach().cpu().numpy()

    # Remove batch dimension if present
    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            raise ValueError(f"Unsupported 4D tensor shape: {arr.shape}")

    # Handle channel-first (C,H,W)
    if arr.ndim == 3:
        if arr.shape[0] == 1:  # Grayscale (1,H,W)
            arr = arr[0]
            mode = "L"
        elif arr.shape[0] == 3:  # RGB (3,H,W)
            arr = arr.transpose(1, 2, 0)
            mode = "RGB"
        elif arr.shape[2] == 3:  # (H,W,3)
            mode = "RGB"
        elif arr.shape[2] == 1:  # (H,W,1)
            arr = arr[:, :, 0]
            mode = "L"
        else:
            raise ValueError(f"Unsupported 3D tensor shape: {arr.shape}")
    elif arr.ndim == 2:
        mode = "L"
    else:
        raise ValueError(f"Unsupported tensor shape for image: {arr.shape}")

    arr = (arr * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr, mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def scale_tensor_image(image_tensor, width, height):
    # image_tensor: [1, H, W, 3] or [1, H, W]
    arr = image_tensor.squeeze(0).cpu().numpy()
    if arr.ndim == 3:  # HWC
        pil_img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
        pil_img = pil_img.resize((width, height), Image.BILINEAR)
        arr = np.array(pil_img).astype(np.float32) / 255.0
        arr = arr[None, ...]  # [1, H, W, 3]
    else:  # HW
        pil_img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
        pil_img = pil_img.resize((width, height), Image.BILINEAR)
        arr = np.array(pil_img).astype(np.float32) / 255.0
        arr = arr[None, ...]  # [1, H, W]
    return torch.from_numpy(arr)



def tensor_image_resize_and_crop_to_multiple_of_64(img, target_short_side):
    # img: [1, H, W, 3] or [1, H, W]
    shape = img.shape
    if len(shape) == 4:
        _, h, w, c = shape
        is_color = True
    elif len(shape) == 3:
        _, h, w = shape
        is_color = False
    else:
        raise ValueError(f"Unsupported tensor shape: {shape}")

    if h < w:
        scale = target_short_side / h
        new_h = target_short_side
        new_w = int(round(w * scale))
    else:
        scale = target_short_side / w
        new_w = target_short_side
        new_h = int(round(h * scale))

    # Resize
    resized = scale_tensor_image(img, new_w, new_h)

    # Crop to nearest multiple of 64
    crop_w = (new_w // 64) * 64
    crop_h = (new_h // 64) * 64
    if is_color:
        cropped = resized[:, :crop_h, :crop_w, :]
    else:
        cropped = resized[:, :crop_h, :crop_w]

    print(f"[pseudocomfy] resized image to: ({new_w}, {new_h}), cropped to: ({crop_w}, {crop_h})")
    return crop_w, crop_h, cropped