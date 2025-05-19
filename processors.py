import node_helpers

from .helpers.dense_diffusion import dd_combine, dd_apply
from .helpers.ipadapter import apply_ipadapter
from .helpers.imgutil import create_solid_mask

class ApplyDenseDiffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"forceInput": True}),
                "clip": ("CLIP", {"forceInput": True}),
                "mat_txts": ("STRING", {"forceInput": True}),
                "mat_msks": ("IMAGE", {"forceInput": True}),
                "env_scene": ("STRING", {"forceInput": True}),
                "env_style": ("STRING", {"forceInput": True}),
                "env_negative": ("STRING", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
            },
        }
    
    INPUT_IS_LIST = True # All inputs of ``type`` will become ``list[type]``, regardless of how many items are passed in.

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("model", "positive", "negative",)
    OUTPUT_IS_LIST = (False, False, False,)

    FUNCTION = "combiner"

    CATEGORY = "Pseudocomfy/Processors"

    def combiner(self, model, clip, mat_txts, mat_msks, env_scene, env_style, env_negative, width, height):
        print("[pseudocomfy]\t\t mat_msks is len: ", len(mat_msks))

        # if model or clip is a list, use the first element
        if isinstance(model, list) and len(model)>0: model = model[0]
        if isinstance(clip, list) and len(clip)>0: clip = clip[0]

        # If env_x are a list, concatenate into a single string
        if isinstance(env_scene, list): env_scene = ", ".join(env_scene)
        if isinstance(env_style, list): env_style = ", ".join(env_style)
        if isinstance(env_negative, list): env_negative = ", ".join(env_negative)

        # if width or height are a list, use the first element
        if isinstance(width, list) and len(width)>0: width = width[0]
        if isinstance(height, list) and len(height)>0: height = height[0]
        print("[pseudocomfy] ApplyDenseDiffusion\t\t width, height: ", width, height)


        styled_material_prompts = [prompt + ", " + env_style for prompt in mat_txts] # adding styles to each object prompt
        # turning the list of strings into a list of conditionings:
        mat_pmts_cond = [clip_text_encode(clip, prompt) for prompt in styled_material_prompts] # appending as a list - format of comfy when returning CONDITIONING type

        combined_txt_list = [env_scene, env_style] + mat_txts # list containing all scene, style and object prompts 
        env_positive = "; ".join(combined_txt_list) # combine all prompts into a single string

        positive_prompt_cond = clip_text_encode(clip, env_positive) # wrapping in a list - format of comfy when returning CONDITIONING type
        negative_prompt_cond = clip_text_encode(clip, env_negative)

        empty_mask = create_solid_mask(1.0, width, height)
        model = dd_combine(model, positive_prompt_cond, empty_mask, 1.0) # first combining with dense diffusion

        for i in range(len(mat_pmts_cond)):
            model = dd_combine(model, mat_pmts_cond[i], mat_msks[i], 1.0)

        work_model, cond = dd_apply(model)

        return (work_model, cond, negative_prompt_cond)
    

class ApplyIPAdaper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "object_txts": ("STRING_LIST",),
                "masks": ("MASK_LIST",),
                "pmt_scene": ("STRING", {"forceInput": True}),
                "pmt_style": ("STRING", {"forceInput": True}),
                "pmt_negative": ("STRING", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "base_cond_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.01}),
                "object_cond_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "object_imgs": ("IMAGE_LIST",),
                "model": ("MODEL",),
                "ipadapter": ("IPADAPTER",),
                "ipadapter_weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")

    RETURN_NAMES = ("model", "positive", "negative")

    FUNCTION = "combiner"

    CATEGORY = "Pseudocomfy/Processors"

    def combiner(self, clip, object_txts, masks, pmt_scene, pmt_style, pmt_negative, width, height, base_cond_strength, object_cond_strength, ipadapter_weight, object_imgs=None, model=None, ipadapter=None):
        # adding styles to each object prompt
        styled_object_txt_prompts = [
            (pmt_style if txt_prompt is None or txt_prompt == '' else txt_prompt + ", " + pmt_style)
            for txt_prompt in object_txts
        ]
        
        # turning the list of strings into a list of conditionings:
        obj_pmts_conds_list = [clip_text_encode(clip, txt_prompt) for txt_prompt in styled_object_txt_prompts] # appending as a list - format of comfy when returning CONDITIONING type

        combined_scene_pmt_list = [pmt_scene, pmt_style]
        pmt_positive = "; ".join(combined_scene_pmt_list) # combine all scene prompts into a single string

        positive_prompt_cond = clip_text_encode(clip, pmt_positive) # wrapping in a list - format of comfy when returning CONDITIONING type
        negative_prompt_cond = clip_text_encode(clip, pmt_negative)

        empty_mask = create_solid_mask(1.0, width, height)
        positive_prompt_cond = conditioning_set_mask(positive_prompt_cond, empty_mask, strength=base_cond_strength) # first combining, using default vals for other params

        for i in range(len(obj_pmts_conds_list)):
            temp = conditioning_set_mask( obj_pmts_conds_list[i], masks[i], strength=object_cond_strength) # using default vals for other params
            positive_prompt_cond = conditioning_combine(positive_prompt_cond, temp)

        if object_imgs is not None:
            if model is not None and ipadapter is not None:
                for i in range(len(obj_pmts_conds_list)):
                    
                    if object_imgs[i] is not None:
                        model, _ = apply_ipadapter(model, ipadapter, object_imgs[i], ipadapter_weight, 0.0, 1.0, 'standard', masks[i])

        
        return (model, positive_prompt_cond, negative_prompt_cond)
    

# ==============================================================================
# utility functions
# ==============================================================================


def clip_text_encode(clip, str):
    tokens = clip.tokenize(str)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]



def conditioning_set_mask(conditioning, mask, set_cond_area="default", strength=1.0): # from builtin nodes: "append" func of the ConditioningSetMask node
        if not (0.0 <= strength <= 10.0):
            raise ValueError("Strength must be between 0.0 and 10.0.")

        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        cond = node_helpers.conditioning_set_values(conditioning, {"mask": mask,
                                                                "set_area_to_bounds": set_area_to_bounds,
                                                                "mask_strength": strength})
        return cond


def conditioning_combine(conditioning_1, conditioning_2): # from builtin nodes: ConditioningCombine
    return conditioning_1 + conditioning_2
