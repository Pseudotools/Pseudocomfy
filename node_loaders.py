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
    """
    Loader class for retrieving model snapshot data from a local directory or a remote URL.
    Inputs:
        string_path (str): Path to a directory containing JSON files or a URL pointing to a JSON resource.
            - If a URL (http/https), the JSON is fetched via HTTP GET.
            - If a local directory, the most recently modified JSON file is loaded.
    Outputs:
        json_data (dict): The loaded JSON data from the selected file or URL.
    Additional Information:
        - When loading from a directory, the loader searches for all JSON files and selects the most recently modified one.
        - Raises FileNotFoundError if no JSON files are found in the specified directory.
    """
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
        print(f"[pseudocomfy] LoadModelSnapshot\n\tstring_path: {string_path}")
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
    """
    Processor class for unpacking a model snapshot JSON into its constituent components for further processing.
    Inputs:
        json_data (DICT): Dictionary containing the model snapshot data, including material prompts, images, masks, environment prompts, and image metadata.
    Outputs:
        mat_txts (list of str): List of material prompt texts, one for each material/object in the scene.
        mat_imgs (list of tensor or None): List of decoded RGB image tensors ([1, H, W, 3]) for each material prompt, or None if not available.
        mat_msks (list of tensor): List of mask tensors ([1, H, W]) corresponding to each material prompt.
        env_scene (str): Scene description prompt from the environment.
        env_style (str): Style description prompt from the environment.
        env_negative (str): Negative prompt for conditioning from the environment.
        width (int): Target width for all masks and outputs, rounded to the nearest multiple of 64.
        height (int): Target height for all masks and outputs, rounded to the nearest multiple of 64.
        img_depth (tensor): Decoded and resized depth image tensor ([1, H, W, 3]).
        img_edge (None): Placeholder for edge image output (not supported yet).
        img_style (None): Placeholder for style image output (not supported yet).
    Additional Information:
        - The number of material prompts, images, and masks must be equal.
        - All images and masks are decoded and resized to the specified width and height.
        - The processor expects specific keys in the input JSON: 
                'map_semantic', 
                'pmts_environment', 
                'width', 
                'height', 
                'img_depth'
        - Edge and style image outputs are currently not supported and will be returned as None.
    """
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
                        "MASK",
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
        expected_keys = [
            'pseudorandom_spatial_package_version',
            'map_semantic',
            'pmts_environment',
            'width',
            'height',
            'img_depth'
        ]
        missing_keys = [k for k in expected_keys if k not in json_data]
        if missing_keys: raise KeyError(f"Missing required keys in json_data: {missing_keys}")      

        package_version = json_data['pseudorandom_spatial_package_version']
        print(f"[pseudocomfy] UnpackModelSnapshot\t spatial_package_version: {package_version}")
        # TODO: check package_version against current min_version (0.0 at time of writing)

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
                img = decode_rgb_image(img) # produces [1, H, W, 3], same as other rgb images
            mat_imgs.append(img)
       
        
        print(f"\tgiven w,h: ({width}, {height})")
        print(f"\tdepth_tensor shape: {tuple(depth_tensor.shape)} ([1, H, W, 3] expected)")
        print(f"\tmat txts/imgs/msks lengths: {len(mat_txts)},{len(mat_imgs)},{len(mat_msks)} (all should be equal)")
        if len(mat_msks) > 1: print(f"\tmat_msks shape:{tuple(mat_msks[0].shape)} ([1, H, W] expected)")
        '''
        for i, mask in enumerate(mat_msks):
            print(f"mat_msks[{i}] shape:", mask.shape) # we expect [1, H, W]
            print(f"mat_msks[{i}] value range: min={mask[0].min().item()}, max={mask[0].max().item()}")
        for i, img in enumerate(mat_imgs):
            if img is not None:
                print(f"mat_imgs[{i}] shape:", img.shape) # we expect [1, H, W, 3]
            else:
                print(f"mat_imgs[{i}] is None")
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

