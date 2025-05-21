import node_helpers
import os

from .helpers.dense_diffusion import dd_combine, dd_apply
from .helpers.ipadapter import apply_ipadapter
from .helpers.imgutil import create_solid_mask
from .helpers.imgutil import scale_tensor_image

class ApplyDenseDiffusion:
    """
    Processor class for applying dense diffusion to material prompts within a scene context.
    Inputs:
        model (MODEL): The base model to apply dense diffusion to.
        clip (CLIP): The CLIP model used for text encoding.
        mat_txts (list of str): List of material prompt texts, one for each object/material.
        mat_msks (list of tensor): List of mask tensors as [1, H, W], one for each material prompt.
        env_scene (str): Scene description prompt.
        env_style (str): Style description prompt to be appended to each material prompt.
        env_negative (str): Negative prompt for conditioning.
        width (int): Target width for all masks and outputs.
        height (int): Target height for all masks and outputs.
    Outputs:
        model (MODEL): The processed model after applying dense diffusion with all prompts and masks.
        positive (CONDITIONING): The positive conditioning tensor for the combined scene and material prompts.
        negative (CONDITIONING): The negative conditioning tensor for the negative prompt.
    Additional Information:
        - Material masks are automatically resized to the specified width and height if needed.
        - Each material prompt is combined with the style prompt for conditioning.
    """
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

    FUNCTION = "func"

    CATEGORY = "Pseudocomfy/Processors"

    def func(self, model, clip, mat_txts, mat_msks, env_scene, env_style, env_negative, width, height):
        
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
        
        # Create a report string pairing each mat_txt with its corresponding mat_msk shape
        mat_report = "\n".join(
            f"\tmat[{i}]: {tuple(msk.shape)} '{txt[:20]}...'"
            for i, (txt, msk) in enumerate(zip(mat_txts, mat_msks))
        )
        print(f"[pseudocomfy] ApplyDenseDiffusion\n\tenv_scene: '{env_scene[:20]}...'\n\tenv_style: '{env_style[:20]}...'\n\tenv_negative: '{env_negative[:20]}...'\n\twidth: {width}, height: {height}\n{mat_report}")

        # Ensure all masks have shape [1, width, height]
        for i in range(len(mat_msks)):
            mask = mat_msks[i]
            # Check mask has three dimensions and the first dimension is 1
            if not (isinstance(mask.shape, tuple) and len(mask.shape) == 3 and mask.shape[0] == 1):
                raise ValueError(f"Mask at index {i} must have shape [1, width, height], got {mask.shape}")            
            
            # scale the mask to the desired width and height if necessary
            if mask.shape[-2] != width or mask.shape[-1] != height:
                #print("[pseudocomfy] ApplyDenseDiffusion\n\tscaling mask to width, height: ", width, height)
                mat_msks[i] = scale_tensor_image(mask, width, height)

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
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "ipadapter": ("IPADAPTER",),
                "mat_txts": ("STRING", {"forceInput": True}), # we expect a list of strings
                "mat_imgs": ("IMAGE", {"forceInput": True}), # we expect a list of images
                "mat_msks": ("IMAGE", {"forceInput": True}), # we expect a list of masks
                "env_scene": ("STRING", {"forceInput": True}),
                "env_style": ("STRING", {"forceInput": True}),
                "env_negative": ("STRING", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "env_cond_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mat_cond_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "ipadapter_weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
            },
        }
    
    INPUT_IS_LIST = True # All inputs of ``type`` will become ``list[type]``, regardless of how many items are passed in.

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("model", "positive", "negative",)
    OUTPUT_IS_LIST = (False, False, False,)

    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Processors"

    def func(self, model, clip, ipadapter, mat_txts, mat_imgs, mat_msks, env_scene, env_style, env_negative, width, height, env_cond_strength, mat_cond_strength, ipadapter_weight):
        
        # unwrap all expected singletons
        #
        
        # if model or clip or ipadapter is a list, use the first element
        if isinstance(model, list) and len(model)>0: model = model[0]
        if isinstance(clip, list) and len(clip)>0: clip = clip[0]
        if isinstance(ipadapter, list) and len(ipadapter)>0: ipadapter = ipadapter[0]

        # If env_x are a list, concatenate into a single string
        if isinstance(env_scene, list): env_scene = ", ".join(env_scene)
        if isinstance(env_style, list): env_style = ", ".join(env_style)
        if isinstance(env_negative, list): env_negative = ", ".join(env_negative)

        # if width or height are a list, use the first element
        if isinstance(width, list) and len(width)>0: width = width[0]
        if isinstance(height, list) and len(height)>0: height = height[0]     
        
        # if any strength is a list, use the first element
        if isinstance(env_cond_strength, list) and len(env_cond_strength)>0: env_cond_strength = env_cond_strength[0]
        if isinstance(mat_cond_strength, list) and len(mat_cond_strength)>0: mat_cond_strength = mat_cond_strength[0]
        if isinstance(ipadapter_weight, list) and len(ipadapter_weight)>0: ipadapter_weight = ipadapter_weight[0]


        # report inputs
        #

        print(f"[pseudocomfy] ApplyIPAdaper\n\tenv_scene: '{env_scene[:20]}...'\n\tenv_style: '{env_style[:20]}...'\n\tenv_negative: '{env_negative[:20]}...'")
        print(f"\twidth: {width}, height: {height}\n\tenv_cond_strength: {env_cond_strength}\n\tmat_cond_strength: {mat_cond_strength}\n\tipadapter_weight: {ipadapter_weight}")
        # Create a report string pairing each mat_txt with its corresponding mat_msk shape
        mat_report = "\n".join(
            f"\tmat[{i}]: {tuple(msk.shape)} '{txt[:20]}...'"
            for i, (txt, msk) in enumerate(zip(mat_txts, mat_msks))
        )
        print(f"{mat_report}")

        # Print relevant info about ipadapter contents
        if isinstance(ipadapter, dict):
            # Print clipvision info
            clipvision = ipadapter.get("clipvision", {})
            print(f"\tclipvision.model type: {type(clipvision.get('model', None))}")

            clipvision_file = clipvision.get('file', 'N/A')
            try: print(f"\tclipvision.file: {os.path.basename(clipvision_file)}")
            except Exception: print(f"\tclipvision.file: {clipvision_file}")

            # Print ipadapter file info
            ipadapter_dict = ipadapter.get("ipadapter", {})
            ipadapter_file = ipadapter_dict.get('file', 'N/A')
            try: print(f"\tipadapter.file: {os.path.basename(ipadapter_file)}")
            except Exception: print(f"\tipadapter.file: {ipadapter_file}")

            # Print keys in ipadapter.model
            ipadapter_model = ipadapter_dict.get("model", {})
            print(f"\tipadapter.model keys: {list(ipadapter_model.keys())}")


        # main
        #

        # add styles to each object prompt
        styled_mat_txts = [ (env_style if m is None or m == '' else m + ", " + env_style) for m in mat_txts]
        
        # turn the list of strings into a list of conditionings:
        cond_mat_txts = [clip_text_encode(clip, s) for s in styled_mat_txts] # appending as a list - format of comfy when returning CONDITIONING type

        env_positive = "; ".join([env_scene, env_style]) # combine scene + style prompts into a single string

        positive_prompt_cond = clip_text_encode(clip, env_positive)
        negative_prompt_cond = clip_text_encode(clip, env_negative)

        empty_mask = create_solid_mask(1.0, width, height)
        positive_prompt_cond = conditioning_set_mask(positive_prompt_cond, empty_mask, strength=env_cond_strength) # first combining, using default vals for other params

        # apply text conditioning to positive_prompt_cond
        for i in range(len(cond_mat_txts)):
            temp = conditioning_set_mask( cond_mat_txts[i], mat_msks[i], strength=mat_cond_strength)
            positive_prompt_cond = conditioning_combine(positive_prompt_cond, temp)

        # apply image conditioning to the model
        for i in range(len(cond_mat_txts)):
            if mat_imgs[i] is not None:
                # model, ipadapter, image, weight, start_at, end_at, weight_type, attn_mask=None
                weight_type = 'standard' # 'standard' or 'style transfer' or 'prompt_is_more_important'
                start_at = 0.0
                end_at = 1.0
                model, _ = apply_ipadapter(model, ipadapter, mat_imgs[i], ipadapter_weight, start_at, end_at, weight_type, mat_msks[i])

        
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
