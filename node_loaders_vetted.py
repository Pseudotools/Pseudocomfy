import requests
import torch
import folder_paths
import comfy.sd
import comfy.controlnet
import comfy.utils


SUPABASE_URL = "https://psfxsrilludczykwdyxz.supabase.co/rest/v1"

# This is the Supabase anon key — intentionally public-facing. Access is
# governed by Row Level Security policies on the Supabase side.
SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBzZnhzcmlsbHVkY3p5a3dkeXh6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODM1NDMwNDgsImV4cCI6MjA5OTExOTA0OH0"
    ".WewS_QERP0nhd9O7uIXQ0evKhfjci04QCGBHJeH8pLk"
)


def _fetch_vetted_models():
    try:
        response = requests.get(
            f"{SUPABASE_URL}/models",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            },
            params={
                "select": "id,name,file_name,category_id",
                "vetting_status_id": "eq.3",
                "order": "name",
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[pseudocomfy] failed to fetch vetted models: {e}")
        return []


_VETTED_MODELS = _fetch_vetted_models()

_CHECKPOINT_MODELS = [m for m in _VETTED_MODELS if m["category_id"] == 1]
_CHECKPOINT_NAMES = [m["file_name"] for m in _CHECKPOINT_MODELS] or ["(no vetted checkpoints available)"]
_CHECKPOINT_ID_MAP = {m["file_name"]: m["id"] for m in _CHECKPOINT_MODELS}
_CHECKPOINT_DEFAULT_ID = _CHECKPOINT_MODELS[0]["id"] if _CHECKPOINT_MODELS else ""

_CONTROLNET_MODELS = [m for m in _VETTED_MODELS if m["category_id"] == 3]
_CONTROLNET_NAMES = [m["file_name"] for m in _CONTROLNET_MODELS] or ["(no vetted controlnet models available)"]
_CONTROLNET_ID_MAP = {m["file_name"]: m["id"] for m in _CONTROLNET_MODELS}
_CONTROLNET_DEFAULT_ID = _CONTROLNET_MODELS[0]["id"] if _CONTROLNET_MODELS else ""

_LORA_MODELS = [m for m in _VETTED_MODELS if m["category_id"] == 5]
_LORA_NAMES = [m["file_name"] for m in _LORA_MODELS] or ["(no vetted lora models available)"]
_LORA_ID_MAP = {m["file_name"]: m["id"] for m in _LORA_MODELS}
_LORA_DEFAULT_ID = _LORA_MODELS[0]["id"] if _LORA_MODELS else ""

_CLIP_MODELS = [m for m in _VETTED_MODELS if m["category_id"] == 7]
_CLIP_NAMES = [m["file_name"] for m in _CLIP_MODELS] or ["(no vetted CLIP models available)"]
_CLIP_ID_MAP = {m["file_name"]: m["id"] for m in _CLIP_MODELS}
_CLIP_DEFAULT_ID = _CLIP_MODELS[0]["id"] if _CLIP_MODELS else ""

_VAE_MODELS = [m for m in _VETTED_MODELS if m["category_id"] == 6]
_VAE_NAMES = [m["file_name"] for m in _VAE_MODELS] or ["(no vetted VAE models available)"]
_VAE_ID_MAP = {m["file_name"]: m["id"] for m in _VAE_MODELS}
_VAE_DEFAULT_ID = _VAE_MODELS[0]["id"] if _VAE_MODELS else ""


class PseudoVettedCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (_CHECKPOINT_NAMES, {"model_ids": _CHECKPOINT_ID_MAP}),
                "model_id": ("STRING", {"default": _CHECKPOINT_DEFAULT_ID}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Loaders"

    def func(self, model, model_id=""):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", model)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        print(f"[pseudocomfy] PseudoVettedCheckpointLoader: {model}")
        return out[:3]


class PseudoVettedControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (_CONTROLNET_NAMES, {"model_ids": _CONTROLNET_ID_MAP}),
                "model_id": ("STRING", {"default": _CONTROLNET_DEFAULT_ID}),
            },
        }

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Loaders"

    def func(self, model, model_id=""):
        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", model)
        controlnet = comfy.controlnet.load_controlnet(controlnet_path)
        if controlnet is None:
            raise RuntimeError(f"[pseudocomfy] PseudoVettedControlNetLoader: invalid controlnet file: {model}")
        print(f"[pseudocomfy] PseudoVettedControlNetLoader: {model}")
        return (controlnet,)


class PseudoVettedLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora": (_LORA_NAMES, {"model_ids": _LORA_ID_MAP}),
                "lora_model_id": ("STRING", {"default": _LORA_DEFAULT_ID}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Loaders"

    def func(self, model, lora, lora_model_id="", strength_model=1.0):
        if strength_model == 0:
            return (model,)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora)

        if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
            lora_weights, lora_metadata = self.loaded_lora[1], self.loaded_lora[2]
        else:
            lora_weights, lora_metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
            self.loaded_lora = (lora_path, lora_weights, lora_metadata)

        model_out, _ = comfy.sd.load_lora_for_models(model, None, lora_weights, strength_model, 0, lora_metadata=lora_metadata)
        print(f"[pseudocomfy] PseudoVettedLoraLoader: {lora} (strength: {strength_model})")
        return (model_out,)


class PseudoVettedClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (_CLIP_NAMES, {"model_ids": _CLIP_ID_MAP}),
                "model_id": ("STRING", {"default": _CLIP_DEFAULT_ID}),
                "type": (["stable_diffusion", "stable_cascade", "sd3", "flux"], {}),
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Loaders"

    def func(self, model, model_id="", type="stable_diffusion", device="default"):
        if model == "(no vetted CLIP models available)":
            raise RuntimeError("[pseudocomfy] PseudoVettedClipLoader: no vetted CLIP models available in the database.")

        clip_type_map = {
            "stable_cascade": comfy.sd.CLIPType.STABLE_CASCADE,
            "sd3": comfy.sd.CLIPType.SD3,
            "flux": comfy.sd.CLIPType.FLUX,
        }
        clip_type = clip_type_map.get(type, comfy.sd.CLIPType.STABLE_DIFFUSION)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip_path = folder_paths.get_full_path_or_raise("clip", model)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options,
        )
        print(f"[pseudocomfy] PseudoVettedClipLoader: {model}")
        return (clip,)


class PseudoVettedVaeLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (_VAE_NAMES, {"model_ids": _VAE_ID_MAP}),
                "model_id": ("STRING", {"default": _VAE_DEFAULT_ID}),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Loaders"

    def func(self, model, model_id=""):
        if model == "(no vetted VAE models available)":
            raise RuntimeError("[pseudocomfy] PseudoVettedVaeLoader: no vetted VAE models available in the database.")
        vae_path = folder_paths.get_full_path_or_raise("vae", model)
        sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        print(f"[pseudocomfy] PseudoVettedVaeLoader: {model}")
        return (vae,)
