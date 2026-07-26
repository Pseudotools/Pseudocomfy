import re

import folder_paths
from comfy.clip_vision import load as load_clip_vision
import comfy.utils
from comfy.sd import load_lora_for_models

from .helpers.ipadapter import *



"""
Adapted from the original code by Matteo Spinelli ("Matt3o/Cubiq")**  
  Original source: [https://github.com/cubiq/ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Main IPAdapter Class (IPAdapterPlus.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


class PseudoIPAdapterUnifiedLoaderClone:
    def __init__(self):
        self.lora = None
        self.clipvision = { "file": None, "model": None }
        self.ipadapter = { "file": None, "model": None }
        self.insightface = { "provider": None, "model": None }

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", ),
            "preset": (['STANDARD (medium strength)', 'VIT-G (medium strength)', 'PLUS (high strength)'], ),
        },
        "optional": {
            "ipadapter": ("IPADAPTER", ),
        }}

    RETURN_TYPES = ("MODEL", "IPADAPTER", )
    RETURN_NAMES = ("model", "ipadapter", )
    FUNCTION = "load_models"
    CATEGORY = "ipadapter"

    def load_models(self, model, preset, lora_strength=0.0, provider="CPU", ipadapter=None):
        pipeline = { "clipvision": { 'file': None, 'model': None }, "ipadapter": { 'file': None, 'model': None }, "insightface": { 'provider': None, 'model': None } }
        if ipadapter is not None:
            pipeline = ipadapter

        # 1. Load the clipvision model
        clipvision_file = get_clipvision_file(preset)
        if clipvision_file is None:
            raise Exception("ClipVision model not found.")

        if clipvision_file != self.clipvision['file']:
            if clipvision_file != pipeline['clipvision']['file']:
                self.clipvision['file'] = clipvision_file
                self.clipvision['model'] = load_clip_vision(clipvision_file)
                print(f"\033[33mINFO: Clip Vision model loaded from {clipvision_file}\033[0m")
            else:
                self.clipvision = pipeline['clipvision']

        # 2. Load the ipadapter model
        is_sdxl = isinstance(model.model, (comfy.model_base.SDXL, comfy.model_base.SDXLRefiner, comfy.model_base.SDXL_instructpix2pix))
        ipadapter_file, is_insightface, lora_pattern = get_ipadapter_file(preset, is_sdxl)
        if ipadapter_file is None:
            raise Exception("IPAdapter model not found.")

        if ipadapter_file != self.ipadapter['file']:
            if pipeline['ipadapter']['file'] != ipadapter_file:
                self.ipadapter['file'] = ipadapter_file
                self.ipadapter['model'] = ipadapter_model_loader(ipadapter_file)
                print(f"\033[33mINFO: IPAdapter model loaded from {ipadapter_file}\033[0m")
            else:
                self.ipadapter = pipeline['ipadapter']

        # 3. Load the lora model if needed
        if lora_pattern is not None:
            lora_file = get_lora_file(lora_pattern)
            lora_model = None
            if lora_file is None:
                raise Exception("LoRA model not found.")

            if self.lora is not None:
                if lora_file == self.lora['file']:
                    lora_model = self.lora['model']
                else:
                    self.lora = None
                    torch.cuda.empty_cache()

            if lora_model is None:
                lora_model = comfy.utils.load_torch_file(lora_file, safe_load=True)
                self.lora = { 'file': lora_file, 'model': lora_model }
                print(f"\033[33mINFO: LoRA model loaded from {lora_file}\033[0m")

            if lora_strength > 0:
                model, _ = load_lora_for_models(model, None, lora_model, lora_strength, 0)

        # 4. Load the insightface model if needed
        if is_insightface:
            if provider != self.insightface['provider']:
                if pipeline['insightface']['provider'] != provider:
                    self.insightface['provider'] = provider
                    self.insightface['model'] = insightface_loader(provider)
                    print(f"\033[33mINFO: InsightFace model loaded with {provider} provider\033[0m")
                else:
                    self.insightface = pipeline['insightface']

        return (model, { 'clipvision': self.clipvision, 'ipadapter': self.ipadapter, 'insightface': self.insightface }, )
    

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 UTIL (utils.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

def get_clipvision_file(preset):
    preset = preset.lower()
    clipvision_list = folder_paths.get_filename_list("clip_vision")

    if preset.startswith("vit-g"):
        pattern = r'(ViT.bigG.14.*39B.b160k|ipadapter.*sdxl|sdxl.*model\.(bin|safetensors))'
    else:
        pattern = r'(ViT.H.14.*s32B.b79K|ipadapter.*sd15|sd1.?5.*model\.(bin|safetensors))'
    clipvision_file = [e for e in clipvision_list if re.search(pattern, e, re.IGNORECASE)]

    clipvision_file = folder_paths.get_full_path("clip_vision", clipvision_file[0]) if clipvision_file else None

    return clipvision_file

def get_ipadapter_file(preset, is_sdxl):
    preset = preset.lower()
    ipadapter_list = folder_paths.get_filename_list("ipadapter")
    is_insightface = False
    lora_pattern = None

    if preset.startswith("standard"):
        if is_sdxl:
            pattern = r'ip.adapter.sdxl.vit.h\.(safetensors|bin)$'
        else:
            pattern = r'ip.adapter.sd15\.(safetensors|bin)$'
    elif preset.startswith("vit-g"):
        if is_sdxl:
            pattern = r'ip.adapter.sdxl\.(safetensors|bin)$'
        else:
            pattern = r'sd15.vit.g\.(safetensors|bin)$'
    elif preset.startswith("plus ("):
        if is_sdxl:
            pattern = r'plus.sdxl.vit.h\.(safetensors|bin)$'
        else:
            pattern = r'ip.adapter.plus.sd15\.(safetensors|bin)$'
    else:
        raise Exception(f"invalid type '{preset}'")

    ipadapter_file = [e for e in ipadapter_list if re.search(pattern, e, re.IGNORECASE)]
    ipadapter_file = folder_paths.get_full_path("ipadapter", ipadapter_file[0]) if ipadapter_file else None

    return ipadapter_file, is_insightface, lora_pattern

def ipadapter_model_loader(file):
    model = comfy.utils.load_torch_file(file, safe_load=True)

    if file.lower().endswith(".safetensors"):
        st_model = {"image_proj": {}, "ip_adapter": {}}
        for key in model.keys():
            if key.startswith("image_proj."):
                st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
            elif key.startswith("ip_adapter."):
                st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
        model = st_model
        del st_model

    if not "ip_adapter" in model.keys() or not model["ip_adapter"]:
        raise Exception("invalid IPAdapter model {}".format(file))

    if 'plusv2' in file.lower():
        model["faceidplusv2"] = True
    
    if 'unnorm' in file.lower():
        model["portraitunnorm"] = True

    return model

def get_lora_file(pattern):
    lora_list = folder_paths.get_filename_list("loras")
    lora_file = [e for e in lora_list if re.search(pattern, e, re.IGNORECASE)]
    lora_file = folder_paths.get_full_path("loras", lora_file[0]) if lora_file else None

    return lora_file

def insightface_loader(provider):
    raise Exception("Kyle disabled insightface model loading.")
    '''
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise Exception(e)

    path = os.path.join(folder_paths.models_dir, "insightface")
    model = FaceAnalysis(name="buffalo_l", root=path, providers=[provider + 'ExecutionProvider',])
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model
    '''












