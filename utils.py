import torch

from .helpers import mask_to_image


class MakeMaskBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK_LIST",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "append"

    CATEGORY = "Pseudocomfy/Utils"

    def append(self, masks):
        result = mask_to_image(masks[0])

        if len(masks) > 1:
            for i in range(1, len(masks)):
                result = torch.cat((result,  mask_to_image(masks[i])), 0)


        return (result,)