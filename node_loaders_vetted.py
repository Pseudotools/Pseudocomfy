import requests


SUPABASE_URL = "https://psfxsrilludczykwdyxz.supabase.co/rest/v1"

# This is the Supabase anon key — intentionally public-facing. Access is
# governed by Row Level Security policies on the Supabase side.
SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBzZnhzcmlsbHVkY3p5a3dkeXh6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODM1NDMwNDgsImV4cCI6MjA5OTExOTA0OH0"
    ".WewS_QERP0nhd9O7uIXQ0evKhfjci04QCGBHJeH8pLk"
)

_CATEGORY_NAMES = {
    1: "checkpoints",
    2: "clip_vision",
    3: "controlnet",
    4: "ipadapter",
    5: "loras",
}


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
        print(f"[pseudocomfy] PseudoVettedModelLoader: failed to fetch models: {e}")
        return []


_VETTED_MODELS = _fetch_vetted_models()

_MODEL_LOOKUP = {
    m["name"]: (m["file_name"], _CATEGORY_NAMES.get(m["category_id"], "unknown"))
    for m in _VETTED_MODELS
}

_MODEL_NAMES = list(_MODEL_LOOKUP.keys()) or ["(no vetted models available)"]


class PseudoVettedModelLoader:
    """
    Presents a dropdown of vetted models from the Pseudorandom Supabase database.
    The list is fetched once at server startup.

    Unlike the standard ComfyUI loader nodes, this does not load the model file
    itself — it outputs the filename and category as strings so they can be wired
    into whichever standard loader node is appropriate (Load Checkpoint, Load
    ControlNet Model, etc.).

    Inputs:
        model (str): Selected model display name from the dropdown.
    Outputs:
        file_name (str): The model filename (e.g. "sd_xl_base_1.0.safetensors").
        category (str): The model category (e.g. "checkpoints", "loras", "controlnet").
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (_MODEL_NAMES, {}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_name", "category")
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Loaders"

    def func(self, model):
        file_name, category = _MODEL_LOOKUP.get(model, (model, "unknown"))
        print(f"[pseudocomfy] PseudoVettedModelLoader: {model} → {file_name} ({category})")
        return (file_name, category)
