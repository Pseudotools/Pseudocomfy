from pathlib import Path
import hashlib
import json
import time
import requests
import urllib.parse

import folder_paths


CUSTOM_NODES_DIR = Path(folder_paths.folder_names_and_paths["custom_nodes"][0][0])
SP_DIR = CUSTOM_NODES_DIR.joinpath("Pseudocomfy", "spatial_packages")

class LoadJSONAuto:
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
    RETURN_NAMES = ("json_path",)

    FUNCTION = "load"

    CATEGORY = "Pseudocomfy/Loaders"

    def load(self, json_file):

        with open(SP_DIR.joinpath(json_file), 'r') as f:
            json_data = json.load(f)

        json_path = json_data #TODO: LATER remove this line and return json_data directly.
        return (json_path,)
    
    @classmethod
    def IS_CHANGED(s, json_file):
        m = hashlib.sha256()

        json_list = [str(file.name) for file in sorted(SP_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)]
        json_file = json_list[0]
        json_path = SP_DIR.joinpath(json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)

        data_str = json.dumps(data, sort_keys=True)
        m.update(data_str.encode('utf-8'))

        current_time = str(time.time())
        m.update(current_time.encode('utf-8'))

        return m.digest().hex()



class LoadJSONFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string_path": ("STRING", {"default": ""})
            },
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("json_path",)

    FUNCTION = "load"

    CATEGORY = "Pseudocomfy/Loaders"

    def load(self, string_path):
        if urllib.parse.urlparse(string_path).scheme in ('http', 'https'):
            response = requests.get(string_path)
            response.raise_for_status()
            json_data = response.json()
            json_path = json_data #TODO: LATER remove this line and return json_data directly.
            return (json_path,)
        else:
            path = Path(string_path)
            json_list = [str(file.name) for file in sorted(path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)]
            
            if not json_list:
                raise FileNotFoundError(f"No JSON files found in the folder: {string_path}")

            json_file = json_list[0]

            with open(path.joinpath(json_file), 'r') as f:
                json_data = json.load(f)
            
            json_path = json_data #TODO: LATER remove this line and return json_data directly.

            return (json_path,)
    
    @classmethod
    def IS_CHANGED(s, string_path):
        m = hashlib.sha256()

        path = Path(string_path)
        json_list = [str(file.name) for file in sorted(path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)]
        json_file = json_list[0]
        json_path = path.joinpath(json_file) 
        with open(json_path, 'r') as f:
            data = json.load(f)

        data_str = json.dumps(data, sort_keys=True)
        m.update(data_str.encode('utf-8'))

        current_time = str(time.time())
        m.update(current_time.encode('utf-8'))
        
        return m.digest().hex()
