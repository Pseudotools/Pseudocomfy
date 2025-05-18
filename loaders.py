from pathlib import Path
import hashlib
import json
import time
import requests
import urllib.parse

import folder_paths

from .helpers.helpers import *


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
                "scale_img_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.5})
            },
        }
    
    RETURN_TYPES = ("STRING",
                    "IMAGE",
                    "IMAGE",
                    "STRING",
                    "STRING",
                    "STRING",
                    "INT",
                    "INT",
                    "IMAGE",
                    "IMAGE",)
    
    RETURN_NAMES = ("mat_txts",
                    "mat_imgs",
                    "mat_msks",
                    "env_scene",
                    "env_style",
                    "env_negative",
                    "width",
                    "height",
                    "img_depth",
                    "img_edge",) 
    
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
                    )



    FUNCTION = "process_json"

    CATEGORY = "Pseudocomfy/Processors"

    def process_json(self, json_data, scale_img_by):
        #print("[pseudocomfy]\t\t ProcessJSON.process_json() called")
        map_semantic = json_data['map_semantic']
        obj_txts = [entry['pmt_txt'] for entry in map_semantic]
        obj_imgs_base64 = [entry['pmt_img'] for entry in map_semantic]
        
        masks_base64 = [entry['mask'] for entry in map_semantic]

        if len(obj_txts) != len(masks_base64):
            raise ValueError("Number of prompts and masks must be equal.")        

        
        # base prompts (pos/neg):
        pmts_environment = json_data['pmts_environment']
        env_scene = pmts_environment['pmt_scene']
        env_style = pmts_environment['pmt_style']
        env_negative = pmts_environment['pmt_negative']

        width = make_multiple_of_64(json_data['width'])
        height = make_multiple_of_64(json_data['height'])


        # depth image:
        img_depth = json_data['img_depth']
        depth_tensor = decode_and_scale_depth(img_depth, scale_img_by, width, height)

        obj_msks = []
        for img in masks_base64:
            scaled_mask = decode_and_scale_mask(img, scale_img_by, width, height)
            obj_msks.append(scaled_mask)

        obj_imgs = []
        for img in obj_imgs_base64:
            if img is not None:
                img = decode_image_prompt(img)
            
            obj_imgs.append(img)


        width = int(width * scale_img_by)
        height = int(height * scale_img_by) # wrapping in int cuz that's the format for empty mask and latent
        
        
        return (
            obj_txts,
            obj_imgs,
            obj_msks,
            env_scene,
            env_style,
            env_negative,
            width,
            height,
            depth_tensor,
            [],
        )





'''
class LoadModelSnapshotAuto:
    @classmethod
    def INPUT_TYPES(s):
        json_files = [str(file.name) for file in sorted(SP_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)]
        # json_files is sorted based on modification time
        return {
            "required": {
                "json_file": (json_files,),
            },
        }
    
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("json_data",)

    FUNCTION = "load"

    CATEGORY = "Pseudocomfy/Loaders"

    def load(self, json_file):

        with open(SP_DIR.joinpath(json_file), 'r') as f:
            json_data = json.load(f)

        return (json_data,)
    
    @classmethod
    def IS_CHANGED(s, json_file):
        m = hashlib.sha256()
        current_time = str(time.time())
        m.update(current_time.encode('utf-8'))

        return m.digest().hex()
'''
