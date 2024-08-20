import base64
import gzip
from PIL import Image, ImageOps
import numpy as np
import io
import torch

import node_helpers

from .ipadapter_helpers import ipadapter_execute


def clip_text_encode(clip, str):
    tokens = clip.tokenize(str)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]



def create_solid_mask(value, width, height): # from builtin nodes: create a solid mask
    mask = torch.full((1, height, width), value, dtype=torch.float32, device="cpu")
    return mask



def make_multiple_of_64(num):
    return ((num + 63) // 64) * 64



def mask_to_image(mask): # from builtin nodes: MaskToImage
    result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    return result



def scale_image(input_image, scale_factor, width, height):
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    """
    Image.ANTIALIAS filter is used for high-quality downsampling. 
    We can replace it with other filters like Image.NEAREST, Image.BILINEAR, or 
    Image.BICUBIC depending on the desired quality and performance.
    """
    scaled_image = input_image.resize((new_width, new_height), Image.BICUBIC)
    
    return scaled_image



def decode_and_scale_mask(base64_mask, scale_factor, width, height):
    image_data = base64.b64decode(base64_mask) # Decode the base64 data
    decompressed_data = gzip.decompress(image_data) # Decompress the gzip data
    flat_array = np.frombuffer(decompressed_data, dtype=np.uint8) # Convert the decompressed data to a numpy array
    reshaped_array = flat_array.reshape((height, width))
    grayscale_image = Image.fromarray(reshaped_array.astype('uint8')*255, 'L') # Convert the numpy array to a greyscale image
    
    scaled_pil = scale_image(grayscale_image, scale_factor, width, height)

    image_array = np.array(scaled_pil).astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # Shape becomes [1, 1, H, W]

    return image_tensor



def decode_and_scale_depth(base64_depth, scale_factor, width, height):
    image_data = base64.b64decode(base64_depth) # becomes binary data
    pil_img = node_helpers.pillow(Image.open, io.BytesIO(image_data)) # builtin function - handles any potential errors during image loading
    # io.BytesIO allows to convert binary data into a in-memory file-like obj that PIL can read

    if pil_img.mode == 'I':
        pil_img = pil_img.point(lambda i: i * (1 / 255)) # pixels: -+2,147,000,000 -> [0,1]

    pil_img = pil_img.convert("RGB") # converting / ensuring image is in RGB format 

    scaled_pil = scale_image(pil_img, scale_factor, width, height)

    image_array = np.array(scaled_pil).astype(np.float32) / 255.0 # -> numpy array, cuz PIL images aren't directly compatible with pytorch tensors
    # .../255: normalizing [0,255] -> [0,1]
    image_tensor = torch.from_numpy(image_array)[None,]

    return image_tensor



def decode_image_prompt(base64_img):
    image_data = base64.b64decode(base64_img) # becomes binary data
    pil_img = node_helpers.pillow(Image.open, io.BytesIO(image_data)) # builtin function - handles any potential errors during image loading
    # io.BytesIO allows to convert binary data into a in-memory file-like obj that PIL can read

    if pil_img.mode == 'I':
        pil_img = pil_img.point(lambda i: i * (1 / 255)) # pixels: -+2,147,000,000 -> [0,1]

    pil_img = pil_img.convert("RGB") # converting / ensuring image is in RGB format 

    image_array = np.array(pil_img).astype(np.float32) / 255.0 # -> numpy array, cuz PIL images aren't directly compatible with pytorch tensors
    # .../255: normalizing [0,255] -> [0,1]
    image_tensor = torch.from_numpy(image_array)[None,]

    return image_tensor



def getMaskFromColor(semantic_base64, color):
    image_data = base64.b64decode(semantic_base64)
    pil_image = node_helpers.pillow(Image.open, io.BytesIO(image_data))
    
    # following a source code's processing steps
    pil_image = node_helpers.pillow(ImageOps.exif_transpose, pil_image)

    if pil_image.getbands() != ("R", "G", "B", "A"):
        if pil_image.mode == 'I':
            pil_image = pil_image.point(lambda i: i * (1 / 255))
        pil_image = pil_image.convert("RGBA")
    
    # for efficient processing:
    image_np = np.array(pil_image)

    # color to a numpy array
    target_color_np = np.array(color)

    # creating a mask where the color matches the target color
    mask = np.all(image_np[:, :, :3] == target_color_np, axis=-1)

    # setting matching pixels to transparent
    image_np[mask] = [0, 0, 0, 0]

    # converting back to a PIL image
    image_with_alpha = Image.fromarray(image_np)

    # extract the alpha channel and create a mask
    mask = np.array(image_with_alpha.getchannel('A')).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask)
    mask = 1.0 - mask 

    return mask.unsqueeze(0)



def conditioning_set_mask(conditioning, mask, set_cond_area="default", strength=1.0): # "append" func of the ConditioningSetMask node
        if not (0.0 <= strength <= 10.0):
            raise ValueError("Strength must be between 0.0 and 10.0.")

        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        cond = node_helpers.conditioning_set_values(conditioning, {"mask": mask,
                                                                "set_area_to_bounds": set_area_to_bounds,
                                                                "mask_strength": strength})
        return cond



def conditioning_combine(conditioning_1, conditioning_2): # ConditioningCombine node
    return conditioning_1 + conditioning_2



def apply_ipadapter_cloned(model, ipadapter, image, weight, start_at, end_at, weight_type, attn_mask=None):
        if weight_type.startswith("style"):
            weight_type = "style transfer"
        elif weight_type == "prompt is more important":
            weight_type = "ease out"
        else:
            weight_type = "linear"

        ipa_args = {
            "image": image,
            "weight": weight,
            "start_at": start_at,
            "end_at": end_at,
            "attn_mask": attn_mask,
            "weight_type": weight_type,
            "insightface": ipadapter['insightface']['model'] if 'insightface' in ipadapter else None,
        }
        
        if 'ipadapter' not in ipadapter:
            raise Exception("IPAdapter model not present in the pipeline. Please load the models with the IPAdapterUnifiedLoader node.")
        if 'clipvision' not in ipadapter:
            raise Exception("CLIPVision model not present in the pipeline. Please load the models with the IPAdapterUnifiedLoader node.")

        return ipadapter_execute(model.clone(), ipadapter['ipadapter']['model'], ipadapter['clipvision']['model'], **ipa_args)
