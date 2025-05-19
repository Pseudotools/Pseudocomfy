from pathlib import Path
import hashlib
import json
import time
import requests
import urllib.parse
import base64
from PIL import Image, ImageOps
import numpy as np
import io
import torch
import gzip

import folder_paths
import node_helpers

from .helpers.imgutil import make_multiple_of_64, scale_tensor_image


CUSTOM_NODES_DIR = Path(folder_paths.folder_names_and_paths["custom_nodes"][0][0])
SP_DIR = CUSTOM_NODES_DIR.joinpath("Pseudocomfy", "snapshots")



class LoadModelSnapshot:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string_path": ("STRING", {"default": ""})
            },
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("json_data",)

    FUNCTION = "load"

    CATEGORY = "Pseudocomfy/Loaders"

    def load(self, string_path):
        if urllib.parse.urlparse(string_path).scheme in ('http', 'https'):
            response = requests.get(string_path)
            response.raise_for_status()
            json_data = response.json()
            return (json_data,)
        else:
            path = Path(string_path)
            json_list = [str(file.name) for file in sorted(path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)]
            
            if not json_list:
                raise FileNotFoundError(f"No JSON files found in the folder: {string_path}")

            json_file = json_list[0]

            with open(path.joinpath(json_file), 'r') as f:
                json_data = json.load(f)

            return (json_data,)
    
    @classmethod
    def IS_CHANGED(s, string_path):
        m = hashlib.sha256()
        current_time = str(time.time())
        m.update(current_time.encode('utf-8'))
        
        return m.digest().hex()


class UnpackModelSnapshot:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_data": ("DICT", ),
            },
        }
    
    RETURN_TYPES = (
                        "STRING",
                        "IMAGE",
                        "IMAGE",
                        "STRING",
                        "STRING",
                        "STRING",
                        "INT",
                        "INT",
                        "IMAGE",
                        "IMAGE",
                        "IMAGE",
                    )
    
    RETURN_NAMES = (
                        "mat_txts",
                        "mat_imgs",
                        "mat_msks",
                        "env_scene",
                        "env_style",
                        "env_negative",
                        "width",
                        "height",
                        "img_depth",
                        "img_edge",
                        "img_style",
                    ) 
    
    OUTPUT_IS_LIST = (
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,                       
                    )



    FUNCTION = "process_json"

    CATEGORY = "Pseudocomfy/Processors"

    def process_json(self, json_data):
        #print("[pseudocomfy]\t\t ProcessJSON.process_json() called")
        map_semantic = json_data['map_semantic']
        mat_txts = [entry['pmt_txt'] for entry in map_semantic]
        mat_imgs_base64 = [entry['pmt_img'] for entry in map_semantic]
        
        masks_base64 = [entry['mask'] for entry in map_semantic]

        if len(mat_txts) != len(masks_base64):
            raise ValueError("Number of prompts and masks must be equal.")        

        
        # base prompts (pos/neg):
        pmts_environment = json_data['pmts_environment']
        env_scene = pmts_environment['pmt_scene']
        env_style = pmts_environment['pmt_style']
        env_negative = pmts_environment['pmt_negative']

        width = make_multiple_of_64(json_data['width'])
        height = make_multiple_of_64(json_data['height'])
        #scaled_width = int(width * scale_img_by)
        #scaled_height = int(height * scale_img_by)

        # depth image:
        img_depth = json_data['img_depth']
        #depth_tensor = decode_and_scale_depth(img_depth, scale_img_by, width, height)
        depth_tensor = scale_tensor_image( decode_rgb_image(img_depth), width, height )

        mat_msks = []
        for img in masks_base64:
            #scaled_mask = decode_and_scale_mask(img, scale_img_by, width, height)
            scaled_mask = scale_tensor_image( decode_mask(img, width, height), width, height )
            mat_msks.append(scaled_mask)

        mat_imgs = []
        for img in mat_imgs_base64:
            if img is not None:
                #img = decode_image_prompt(img)
                img = decode_rgb_image(img)        
            mat_imgs.append(img)
       
        '''
        print("depth_tensor shape:", depth_tensor.shape) # we expect [1, H, W, 3]
        for i, mask in enumerate(mat_msks):
            print(f"mat_msks[{i}] shape:", mask.shape) # we expect [1, H, W]
            print(f"mat_msks[{i}] value range: min={mask[0].min().item()}, max={mask[0].max().item()}")
        for i, img in enumerate(mat_imgs):
            if img is not None:
                print(f"mat_imgs[{i}] shape:", img.shape) # we expect [1, H, W, 3]
            else:
                print(f"mat_imgs[{i}] is None")

        print("given w,h:", width, height)
        '''

        return (
            mat_txts,
            mat_imgs,
            mat_msks,
            env_scene,
            env_style,
            env_negative,
            width,
            height,
            depth_tensor,
            None, # no edge image support yet
            None, # no style image support yet
        )



# ==============================================================================
# utility functions
# ==============================================================================


def decode_mask(base64_mask, width, height):
    image_data = base64.b64decode(base64_mask)
    decompressed_data = gzip.decompress(image_data)
    flat_array = np.frombuffer(decompressed_data, dtype=np.uint8)
    reshaped_array = flat_array.reshape((height, width))
    # scale up to 0/255 for display, but output shape [1, H, W]
    image_tensor = torch.from_numpy((reshaped_array * 255).astype(np.float32) / 255.0).unsqueeze(0)
    return image_tensor  # [1, H, W]

def decode_rgb_image(base64_img):
    image_data = base64.b64decode(base64_img)
    pil_img = node_helpers.pillow(Image.open, io.BytesIO(image_data))
    if pil_img.mode == 'I':
        pil_img = pil_img.point(lambda i: i * (1 / 255))
    pil_img = pil_img.convert("RGB")
    image_array = np.array(pil_img).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)[None, ...]  # [1, H, W, 3]
    return image_tensor

# not actually used.
def decode_gray_image(base64_img):
    image_data = base64.b64decode(base64_img)
    pil_img = node_helpers.pillow(Image.open, io.BytesIO(image_data))
    pil_img = pil_img.convert("L")
    image_array = np.array(pil_img).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)[None, ...]  # [1, H, W]
    return image_tensor


'''

def decode_and_scale_mask(base64_mask, scale_factor, width, height):
    image_data = base64.b64decode(base64_mask) # Decode the base64 data
    decompressed_data = gzip.decompress(image_data) # Decompress the gzip data
    flat_array = np.frombuffer(decompressed_data, dtype=np.uint8) # Convert the decompressed data to a numpy array
    reshaped_array = flat_array.reshape((height, width))
    grayscale_image = Image.fromarray(reshaped_array.astype('uint8')*255, 'L') # Convert the numpy array to a greyscale image
    
    scaled_pil = dumb_scale_image(grayscale_image, scale_factor, width, height)

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

    scaled_pil = dumb_scale_image(pil_img, scale_factor, width, height)

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


def dumb_scale_image(input_image, scale_factor, width, height):
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    """
    Image.ANTIALIAS filter is used for high-quality downsampling. 
    We can replace it with other filters like Image.NEAREST, Image.BILINEAR, or 
    Image.BICUBIC depending on the desired quality and performance.
    """
    scaled_image = input_image.resize((new_width, new_height), Image.BICUBIC)
    
    return scaled_image


'''