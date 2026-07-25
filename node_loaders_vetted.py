import requests
import folder_paths
import comfy.sd
import comfy.controlnet


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
                "select": "name,file_name,category_id",
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

_CONTROLNET_NAMES = (
    [m["file_name"] for m in _VETTED_MODELS if m["category_id"] == 3]
    or ["(no vetted controlnet models available)"]
)


class PseudoVettedCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (_CHECKPOINT_NAMES, {}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Loaders"

    def func(self, model):
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
                "model": (_CONTROLNET_NAMES, {}),
            },
        }

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Loaders"

    def func(self, model):
        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", model)
        controlnet = comfy.controlnet.load_controlnet(controlnet_path)
        if controlnet is None:
            raise RuntimeError(f"[pseudocomfy] PseudoVettedControlNetLoader: invalid controlnet file: {model}")
        print(f"[pseudocomfy] PseudoVettedControlNetLoader: {model}")
        return (controlnet,)
