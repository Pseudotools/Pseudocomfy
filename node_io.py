
import os, json, io, base64

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args

from .helpers.imgutil import scale_tensor_image


# ==============================================================================
# adpated from ComfyUI SaveImagne node
# ==============================================================================
class SaveImageWithEmbeddedMasks:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "imgs": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "pseudocomfy", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
            "optional": {   
                "msks": ("MASK", {"tooltip": "The masks to embed in the image."}), # List of mask tensors ([1, H, W])
            },      
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    INPUT_IS_LIST = True 

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Pseudocomfy/IO"
    DESCRIPTION = "Saves the input images with embedded masks as extra channels."

    def save_images(self, imgs, filename_prefix="pseudocomfy", msks=[], prompt=None, extra_pnginfo=None):
        # given that INPUT_IS_LIST, msks is a list of masks that should be embedded in each image in each batch
        # given that INPUT_IS_LIST, imgs is a list of image batches
        image_batches = imgs
        typical_image_shape = image_batches[0][0].shape

        # if inputs that are meant to be singletons are a list, use the first element
        if isinstance(filename_prefix, list) and len(filename_prefix)>0: filename_prefix = filename_prefix[0]
        if isinstance(prompt, list) and len(prompt)>0: prompt = prompt[0]
        if isinstance(extra_pnginfo, list) and len(extra_pnginfo)>0: extra_pnginfo = extra_pnginfo[0]

        print(f"[pseudocomfy] SaveImageWithEmbeddedMasks")
        print(f"\tfilename_prefix = {filename_prefix}")
        print(f"\ttypical_image_shape = {tuple(typical_image_shape)}")
        print(f"\timage_batches count = {len(image_batches)}")
        for i, img_batch in enumerate(image_batches):
            print(f"\timg_batch[{i}] shape = {tuple(img_batch.shape)}")        
        print(f"\tmsks count = {len(msks)}")
        for i, msk in enumerate(msks):
            print(f"\tmsks[{i}] shape = {tuple(msk.shape)}")
        

        # Ensure all masks have shape [1, width, height]
        
        for i in range(len(msks)):
            mask = msks[i]
            # Check mask has three dimensions and the first dimension is 1
            if not (isinstance(mask.shape, tuple) and len(mask.shape) == 3 and mask.shape[0] == 1):
                raise ValueError(f"Mask at index {i} must have shape [1, width, height], got {mask.shape}")            
            
            # scale the mask to the desired width and height if necessary
            if mask.shape[-2] != typical_image_shape[0] or mask.shape[-1] != typical_image_shape[1]:
                msks[i] = scale_tensor_image(mask, typical_image_shape[1], typical_image_shape[0]) # note that for scale_tensor_image, the order of width and height is reversed from order of tensor shape
        

        print(f"\tmsks count = {len(msks)}")
        for i, msk in enumerate(msks):
            print(f"\tmsks[{i}] shape = {tuple(msk.shape)}")
        
        
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, typical_image_shape[1], typical_image_shape[0])
        results = list()

        for img_batch in image_batches:
            for (batch_number, image) in enumerate(img_batch):
                i = 255. * image.cpu().numpy()
                image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                    # Embed ALL masks as base64 PNGs in metadata
                    for mask_idx, mask in enumerate(msks):
                        mask_np = mask.squeeze().cpu().numpy()
                        mask_img = Image.fromarray(np.clip(255 * mask_np, 0, 255).astype(np.uint8), mode="L")
                        buf = io.BytesIO()
                        mask_img.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        mask_json = {
                            "format": "PNG",
                            "encoding": "base64",
                            "size": list(mask_img.size),
                            "data": b64
                        }
                        metadata.add_text(f"pseudocomfy_mask_{mask_idx:02d}", json.dumps(mask_json))

                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                file = f"{filename_with_batch_num}_{counter:05}_.png"
                image.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
                counter += 1

        return { "ui": { "images": results } }