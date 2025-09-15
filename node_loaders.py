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

from .helpers.imgutil import make_multiple_of_64


CUSTOM_NODES_DIR = Path(folder_paths.folder_names_and_paths["custom_nodes"][0][0])
SP_DIR = CUSTOM_NODES_DIR.joinpath("Pseudocomfy", "snapshots")



class PseudoLoadModelSnapshot:
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

    CATEGORY = "Pseudocomfy/IO"

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


class PseudoUnpackModelSnapshot:
    """
    Processor class for unpacking a model snapshot JSON into its constituent components for further processing.
    Supports both v0.1 (legacy) and v0.4 (current) snapshot formats with automatic detection and backwards compatibility.
    
    Inputs:
        json_data (DICT): Dictionary containing the model snapshot data, including material prompts, images, masks, environment prompts, and image metadata.
                         Must include 'pseudorandom_snapshot_version' field to determine format version.
    Outputs:
        mat_txts (list of str): List of material prompt texts, one for each material/object in the scene.
        mat_imgs (list of tensor or None): List of decoded RGB image tensors ([1, H, W, 3]) for each material prompt, or None if not available.
        mat_msks (list of tensor): List of mask tensors ([1, H, W]) corresponding to each material prompt.
        env_scene (str): Scene description prompt from the environment.
        env_style (str): Style description prompt from the environment.
        env_negative (str): Negative prompt for conditioning from the environment.
        img_depth (tensor): Decoded depth image tensor ([1, H, W, 3]).
        img_edge (None): Placeholder for edge image output (not supported yet).
        img_style (None): Placeholder for style image output (not supported yet).
    Additional Information:
        - The number of material prompts, images, and masks must be equal.
        - All images and masks are decoded and resized to the specified width and height.
        - Supports v0.1 format with keys: 'map_semantic', 'pmts_environment', 'width', 'height', 'img_depth'
        - Supports v0.4 format with keys: 'global_guidance', 'regional_guidance', 'spatial_guidance', 'width', 'height'
        - Minimum supported version is 0.1
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
                    )

    FUNCTION = "process_json"
    CATEGORY = "Pseudocomfy/IO"

    def process_json(self, json_data):
        MIN_SUPPORTED_VERSION = 0.1
        MAX_SUPPORTED_VERSION = 0.4


        # Check for required version field
        if 'pseudorandom_snapshot_version' not in json_data:
            raise KeyError("Missing required key: pseudorandom_snapshot_version")
            
        package_version = json_data['pseudorandom_snapshot_version']
        print(f"[pseudocomfy] UnpackModelSnapshot\tspatial_package_version: {package_version}")
        
        # Check package_version against minimum supported version
        min_supported_version = MIN_SUPPORTED_VERSION
        if package_version < min_supported_version:
            raise ValueError(f"[pseudocomfy] UnpackModelSnapshot\tUnsupported spatial package version: {package_version}. Minimum supported version is {min_supported_version}")

        # Handle different protocol versions
        if package_version == MAX_SUPPORTED_VERSION:
            return self._process_v04_json(json_data)
        else:
            return self._process_v01_json(json_data)

    def _process_v01_json(self, json_data):
        """Process v0.1 format JSON data"""
        expected_keys = [
            'map_semantic',
            'pmts_environment',
            'width',
            'height',
            'img_depth'
        ]
        missing_keys = [k for k in expected_keys if k not in json_data]
        if missing_keys: raise KeyError(f"Missing required keys in json_data: {missing_keys}")      

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

        width_given = json_data['width']
        height_given = json_data['height']

        # depth image:
        img_depth = json_data['img_depth']
        depth_tensor = decode_rgb_image(img_depth)

        mat_msks = []
        for img in masks_base64:
            mat_msks.append(decode_mask(img, width_given, height_given, 0.1))

        mat_imgs = []
        for img in mat_imgs_base64:
            if img is not None:
                img = decode_rgb_image(img) # produces [1, H, W, 3], same as other rgb images
            mat_imgs.append(img)

        return self._return_processed_data(mat_txts, mat_imgs, mat_msks, env_scene, env_style, env_negative, depth_tensor, width_given, height_given)

    def _process_v04_json(self, json_data):
        """Process v0.4 format JSON data"""
        expected_keys = [
            'global_guidance',
            'regional_guidance',
            'width',
            'height'
        ]
        missing_keys = [k for k in expected_keys if k not in json_data]
        if missing_keys: raise KeyError(f"Missing required keys in json_data: {missing_keys}")      

        # Extract regional guidance data
        regional_guidance = json_data['regional_guidance']
        mat_txts = [entry.get('txt') for entry in regional_guidance]
        mat_imgs_base64 = [entry.get('img') for entry in regional_guidance]
        masks_base64 = [entry['mask'] for entry in regional_guidance]

        if len(mat_txts) != len(masks_base64):
            raise ValueError("Number of prompts and masks must be equal.")        

        # Extract global guidance data
        global_guidance = json_data['global_guidance']
        env_scene = global_guidance['txt_scene']
        env_style = global_guidance['txt_style']
        env_negative = global_guidance['txt_negative']

        width_given = json_data['width']
        height_given = json_data['height']

        # Extract spatial guidance data (depth image)
        spatial_guidance = json_data.get('spatial_guidance', {})
        img_depth = spatial_guidance.get('depth')
        if img_depth is None:
            raise KeyError("Missing required spatial_guidance.depth in v0.4 format")
        depth_tensor = decode_rgb_image(img_depth)

        mat_msks = []
        for img in masks_base64:
            mat_msks.append(decode_mask(img, width_given, height_given, 0.4))

        mat_imgs = []
        for img in mat_imgs_base64:
            if img is not None:
                img = decode_rgb_image(img) # produces [1, H, W, 3], same as other rgb images
            mat_imgs.append(img)

        return self._return_processed_data(mat_txts, mat_imgs, mat_msks, env_scene, env_style, env_negative, depth_tensor, width_given, height_given)

    def _return_processed_data(self, mat_txts, mat_imgs, mat_msks, env_scene, env_style, env_negative, depth_tensor, width_given, height_given):
        print(f"\tgiven w,h: ({width_given}, {height_given})")
        print(f"\tdepth_tensor shape: {tuple(depth_tensor.shape)} ([1, H, W, 3] expected)")
        print(f"\tmat txts/imgs/msks lengths: {len(mat_txts)},{len(mat_imgs)},{len(mat_msks)} (all should be equal)")
        if len(mat_msks) > 1: print(f"\tmat_msks shape:{tuple(mat_msks[0].shape)} ([1, H, W] expected)")

        return (
            mat_txts,
            mat_imgs,
            mat_msks,
            env_scene,
            env_style,
            env_negative,
            depth_tensor,
            None, # no edge image support yet
            None, # no style image support yet
        )



# ==============================================================================
# utility functions
# ==============================================================================


def decode_mask(base64_mask, width, height, package_version):

    if package_version == 0.0:
        # old masks were stored a zipped flat binary array
        image_data = base64.b64decode(base64_mask)
        decompressed_data = gzip.decompress(image_data)
        flat_array = np.frombuffer(decompressed_data, dtype=np.uint8)
        reshaped_array = flat_array.reshape((height, width))
        # scale up to 0/255 for display, but output shape [1, H, W]
        image_tensor = torch.from_numpy((reshaped_array * 255).astype(np.float32) / 255.0).unsqueeze(0)
        return image_tensor  # [1, H, W]

    # masks are stored as greyscale images
    image_data = base64.b64decode(base64_mask)
    pil_img = node_helpers.pillow(Image.open, io.BytesIO(image_data))
    pil_img = pil_img.convert("L")
    image_array = np.array(pil_img).astype(np.float32) / 255.0  # shape [H, W]
    #print(f"[decode_mask] numpy image_array shape: {image_array.shape}, dtype: {image_array.dtype}")
    if image_array.ndim == 2:
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # [1, H, W]
        #print(f"[decode_mask] torch image_tensor shape (after unsqueeze): {image_tensor.shape}")
    else:       
        print(f"[pseudocomfy] PseudoUnpackModelSnapshot.decode_mask\t numpy image_array shape: {image_array.shape}, dtype: {image_array.dtype}")
        print(f"[pseudocomfy] PseudoUnpackModelSnapshot.decode_mask\t torch image_tensor shape (after unsqueeze): {image_tensor.shape} (expected [1, H, W])")
        raise ValueError(f"Decoded mask has unexpected shape: {image_array.shape}")
    return image_tensor


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

