import concurrent.futures
import requests
import torch
import folder_paths
import comfy.sd
import comfy.controlnet
import comfy.utils


# Model listings and cards are served through the Pseudotools API rather
# than fetched from huggingface.co directly, so the node never talks to a
# third party and pseudotools.com stays the single place that controls
# what "vetted" means (caching, rate limits, backing store) for every
# ComfyUI install that has this node.
PSEUDOTOOLS_API = "https://tools.pseudotools.com/api/models"


def _fetch_requirement(record_id):
    try:
        resp = requests.get(f"{PSEUDOTOOLS_API}/hf/{record_id}", timeout=10)
        if not resp.ok:
            return None
        return resp.json().get("requirement") or None
    except Exception:
        return None


def _fetch_vetted_models():
    try:
        resp = requests.get(PSEUDOTOOLS_API, timeout=10)
        resp.raise_for_status()
        repos = resp.json()
    except Exception as e:
        print(f"[pseudocomfy] failed to fetch models from Pseudotools API: {e}")
        return []

    def enrich(repo):
        requirement = _fetch_requirement(repo["record_id"])
        return {"record_id": repo["record_id"], "category": repo["category"], "requirement": requirement} if requirement else None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for result in executor.map(enrich, repos):
            if result:
                results.append(result)

    results.sort(key=lambda m: m["requirement"])
    print(f"[pseudocomfy] loaded {len(results)} models from Pseudotools API")
    return results


_VETTED_MODELS = _fetch_vetted_models()

_CHECKPOINT_MODELS = [m for m in _VETTED_MODELS if m["category"] == "checkpoints"]
_CHECKPOINT_NAMES = [m["requirement"] for m in _CHECKPOINT_MODELS] or ["(no vetted checkpoints available)"]
_CHECKPOINT_ID_MAP = {m["requirement"]: m["record_id"] for m in _CHECKPOINT_MODELS}
_CHECKPOINT_DEFAULT_ID = _CHECKPOINT_MODELS[0]["record_id"] if _CHECKPOINT_MODELS else ""

_CONTROLNET_MODELS = [m for m in _VETTED_MODELS if m["category"] == "controlnet"]
_CONTROLNET_NAMES = [m["requirement"] for m in _CONTROLNET_MODELS] or ["(no vetted controlnet models available)"]
_CONTROLNET_ID_MAP = {m["requirement"]: m["record_id"] for m in _CONTROLNET_MODELS}
_CONTROLNET_DEFAULT_ID = _CONTROLNET_MODELS[0]["record_id"] if _CONTROLNET_MODELS else ""

_LORA_MODELS = [m for m in _VETTED_MODELS if m["category"] == "loras"]
_LORA_NAMES = [m["requirement"] for m in _LORA_MODELS] or ["(no vetted lora models available)"]
_LORA_ID_MAP = {m["requirement"]: m["record_id"] for m in _LORA_MODELS}
_LORA_DEFAULT_ID = _LORA_MODELS[0]["record_id"] if _LORA_MODELS else ""

_CLIP_MODELS = [m for m in _VETTED_MODELS if m["category"] == "clip_vision"]
_CLIP_NAMES = [m["requirement"] for m in _CLIP_MODELS] or ["(no vetted CLIP models available)"]
_CLIP_ID_MAP = {m["requirement"]: m["record_id"] for m in _CLIP_MODELS}
_CLIP_DEFAULT_ID = _CLIP_MODELS[0]["record_id"] if _CLIP_MODELS else ""


class PseudoVettedCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (_CHECKPOINT_NAMES, {"record_ids": _CHECKPOINT_ID_MAP}),
                "record_id": ("STRING", {"default": _CHECKPOINT_DEFAULT_ID}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Loaders"

    def func(self, model, record_id=""):
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
                "model": (_CONTROLNET_NAMES, {"record_ids": _CONTROLNET_ID_MAP}),
                "record_id": ("STRING", {"default": _CONTROLNET_DEFAULT_ID}),
            },
        }

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Loaders"

    def func(self, model, record_id=""):
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
                "model_input": ("MODEL",),
                "model": (_LORA_NAMES, {"record_ids": _LORA_ID_MAP}),
                "record_id": ("STRING", {"default": _LORA_DEFAULT_ID}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Loaders"

    def func(self, model_input, model, record_id="", strength_model=1.0):
        if strength_model == 0:
            return (model_input,)

        lora_path = folder_paths.get_full_path_or_raise("loras", model)

        if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
            lora_weights, lora_metadata = self.loaded_lora[1], self.loaded_lora[2]
        else:
            lora_weights, lora_metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
            self.loaded_lora = (lora_path, lora_weights, lora_metadata)

        model_out, _ = comfy.sd.load_lora_for_models(model_input, None, lora_weights, strength_model, 0, lora_metadata=lora_metadata)
        print(f"[pseudocomfy] PseudoVettedLoraLoader: {model} (strength: {strength_model})")
        return (model_out,)


class PseudoVettedClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (_CLIP_NAMES, {"record_ids": _CLIP_ID_MAP}),
                "record_id": ("STRING", {"default": _CLIP_DEFAULT_ID}),
                "type": (["stable_diffusion", "stable_cascade", "sd3", "flux"], {}),
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Loaders"

    def func(self, model, record_id="", type="stable_diffusion", device="default"):
        if model == "(no vetted CLIP models available)":
            raise RuntimeError("[pseudocomfy] PseudoVettedClipLoader: no vetted CLIP models available.")

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
