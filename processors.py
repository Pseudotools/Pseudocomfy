from .helpers.helpers import *
from .helpers.dense_diffusion import combine, apply
from .helpers.ipadapter import apply_ipadapter


class ProcessJSON:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_data": ("DICT", ),
                "scale_img_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.5})
            },
        }
    
    RETURN_TYPES = ("STRING_LIST",
                    "IMAGE_LIST",
                    "MASK_LIST",
                    "STRING",
                    "STRING",
                    "STRING",
                    "INT",
                    "INT",
                    "IMAGE",
                    "IMAGE",)
    
    RETURN_NAMES = ("object_txts",
                    "object_imgs",
                    "masks",
                    "pmt_scene",
                    "pmt_style",
                    "pmt_negative",
                    "width",
                    "height",
                    "img_depth",
                    "img_edge",)

    FUNCTION = "process_json"

    CATEGORY = "Pseudocomfy/Processors"

    def process_json(self, json_data, scale_img_by):
        map_semantic = json_data['map_semantic']
        object_txts = [entry['pmt_txt'] for entry in map_semantic]
        object_imgs_base64 = [entry['pmt_img'] for entry in map_semantic]
        
        masks_base64 = [entry['mask'] for entry in map_semantic]

        if len(object_txts) != len(masks_base64):
            raise ValueError("Number of prompts and masks must be equal.")        

        
        # base prompts (pos/neg):
        pmts_environment = json_data['pmts_environment']
        pmt_scene = pmts_environment['pmt_scene']
        pmt_style = pmts_environment['pmt_style']
        pmt_negative = pmts_environment['pmt_negative']

        width = make_multiple_of_64(json_data['width'])
        height = make_multiple_of_64(json_data['height'])


        # depth image:
        img_depth = json_data['img_depth']
        depth_tensor = decode_and_scale_depth(img_depth, scale_img_by, width, height)

        masks = []
        for img in masks_base64:
            scaled_mask = decode_and_scale_mask(img, scale_img_by, width, height)
            masks.append(scaled_mask)

        object_imgs = []
        for img in object_imgs_base64:
            if img is not None:
                img = decode_image_prompt(img)
            
            object_imgs.append(img)


        width = int(width * scale_img_by)
        height = int(height * scale_img_by) # wrapping in int cuz that's the format for empty mask and latent
        return (
            object_txts,
            object_imgs,
            masks,
            pmt_scene,
            pmt_style,
            pmt_negative,
            width,
            height,
            depth_tensor,
            [],
        )



class Combiner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "object_txts": ("STRING_LIST",),
                "masks": ("MASK_LIST",),
                "pmt_scene": ("STRING", {"forceInput": True}),
                "pmt_style": ("STRING", {"forceInput": True}),
                "pmt_negative": ("STRING", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True})
            },
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")

    RETURN_NAMES = ("model", "positive", "negative")

    FUNCTION = "combiner"

    CATEGORY = "Pseudocomfy/Processors"

    def combiner(self, model, clip, object_txts, masks, pmt_scene, pmt_style, pmt_negative, width, height):
        styled_object_prompts = [prompt + ", " + pmt_style for prompt in object_txts] # adding styles to each object prompt
        # turning the list of strings into a list of conditionings:
        obj_pmts_cond = [clip_text_encode(clip, prompt) for prompt in styled_object_prompts] # appending as a list - format of comfy when returning CONDITIONING type

        combined_pmt_list = [pmt_scene, pmt_style] + object_txts # list containing all scene, style and object prompts 
        pmt_positive = "; ".join(combined_pmt_list) # combine all prompts into a single string

        positive_prompt_cond = clip_text_encode(clip, pmt_positive) # wrapping in a list - format of comfy when returning CONDITIONING type
        negative_prompt_cond = clip_text_encode(clip, pmt_negative)

        empty_mask = create_solid_mask(1.0, width, height)
        model = combine(model, positive_prompt_cond, empty_mask, 1.0) # first combining with dense diffusion

        for i in range(len(obj_pmts_cond)):
            model = combine(model, obj_pmts_cond[i], masks[i], 1.0)

        work_model, cond = apply(model)

        return (work_model, cond, negative_prompt_cond)
    

class MixedBuiltinCombinerIPAdaper:
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