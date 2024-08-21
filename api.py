from __future__ import annotations
from aiohttp import web
import os

import folder_paths
import server


def get_first_level_items(folder_path):
    """Get first-level directories from the given folder."""
    excluded_dirs = {"__pycache__"}
    try:
        dirs = [item for item in os.listdir(folder_path) 
                if os.path.isdir(os.path.join(folder_path, item)) and item not in excluded_dirs]
        dirs.sort()
        return dirs
    except FileNotFoundError:
        return []


#### Code snippet from IPAdapterPlus package: since ipadapter folder is not builtin in models ####
if "ipadapter" not in folder_paths.folder_names_and_paths:
    ipadapter_folder = [os.path.join(folder_paths.models_dir, "ipadapter")]
else:
    ipadapter_folder, _ = folder_paths.folder_names_and_paths["ipadapter"]
folder_paths.folder_names_and_paths["ipadapter"] = (ipadapter_folder, folder_paths.supported_pt_extensions)
########################################################################################################


if _server := getattr(server.PromptServer, "instance", None):
    @_server.routes.get("/api/pseudorandom/affordances")
    async def installed_checkpoints(request):
        
        folder_names = [
            "checkpoints", "clip", "clip_vision", "configs", "controlnet",
            "diffusers", "embeddings", "gligen", "hypernetworks", "ipadapter",
            "loras", "photomaker", "style_models", "unet", "upscale_models", 
            "vae", "vae_approx"
        ]

        # dict: folder_name: contents
        folder_contents = {name: folder_paths.get_filename_list(name) for name in folder_names}

        # we only want the first lvl of custom_nodes folder:
        custom_nodes_folder_path = folder_paths.get_folder_paths("custom_nodes")[0] # get_folder_paths returns a list of paths
        custom_nodes = get_first_level_items(custom_nodes_folder_path)

        folder_contents["custom_nodes"] = custom_nodes

        sorted_folder_contents = dict(sorted(folder_contents.items())) # sort alphabetically

        return web.json_response(sorted_folder_contents)