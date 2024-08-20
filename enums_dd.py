# From https://github.com/Mikubill/sd-webui-controlnet/blob/main/scripts/enums.py

from enum import Enum
from typing import List, NamedTuple
from functools import lru_cache


class UnetBlockType(Enum):
    INPUT = "input"
    OUTPUT = "output"
    MIDDLE = "middle"


class TransformerID(NamedTuple):
    block_type: UnetBlockType
    # The id of the block the transformer is in. Not all blocks have cross attn.
    block_id: int
    # The index of transformer within the block.
    # A block can have multiple transformers in SDXL.
    block_index: int
    # The call index of transformer if in a single step of diffusion.
    transformer_index: int


class TransformerIDResult(NamedTuple):
    input_ids: List[TransformerID]
    output_ids: List[TransformerID]
    middle_ids: List[TransformerID]

    def get(self, idx: int) -> TransformerID:
        return self.to_list()[idx]

    def to_list(self) -> List[TransformerID]:
        return sorted(
            self.input_ids + self.output_ids + self.middle_ids,
            key=lambda i: i.transformer_index,
        )


class StableDiffusionVersion(Enum):
    """The version family of stable diffusion model."""

    UNKNOWN = 0
    SD1x = 1
    SD2x = 2
    SDXL = 3

    @staticmethod
    def detect_from_model_name(model_name: str) -> "StableDiffusionVersion":
        """Based on the model name provided, guess what stable diffusion version it is.
        This might not be accurate without actually inspect the file content.
        """
        if any(f"sd{v}" in model_name.lower() for v in ("14", "15", "16")):
            return StableDiffusionVersion.SD1x

        if "sd21" in model_name or "2.1" in model_name:
            return StableDiffusionVersion.SD2x

        if "xl" in model_name.lower():
            return StableDiffusionVersion.SDXL

        return StableDiffusionVersion.UNKNOWN

    def encoder_block_num(self) -> int:
        if self in (
            StableDiffusionVersion.SD1x,
            StableDiffusionVersion.SD2x,
            StableDiffusionVersion.UNKNOWN,
        ):
            return 12
        else:
            return 9  # SDXL

    def controlnet_layer_num(self) -> int:
        return self.encoder_block_num() + 1

    @property
    def transformer_block_num(self) -> int:
        """Number of blocks that has cross attn transformers in unet."""
        if self in (
            StableDiffusionVersion.SD1x,
            StableDiffusionVersion.SD2x,
            StableDiffusionVersion.UNKNOWN,
        ):
            return 16
        else:
            return 11  # SDXL

    @property
    @lru_cache(maxsize=None)
    def transformer_ids(self) -> List[TransformerID]:
        """id of blocks that have cross attention"""
        if self in (
            StableDiffusionVersion.SD1x,
            StableDiffusionVersion.SD2x,
            StableDiffusionVersion.UNKNOWN,
        ):
            transformer_index = 0
            input_ids = []
            for block_id in [1, 2, 4, 5, 7, 8]:
                input_ids.append(
                    TransformerID(UnetBlockType.INPUT, block_id, 0, transformer_index)
                )
                transformer_index += 1
            middle_id = TransformerID(UnetBlockType.MIDDLE, 0, 0, transformer_index)
            transformer_index += 1
            output_ids = []
            for block_id in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
                input_ids.append(
                    TransformerID(UnetBlockType.OUTPUT, block_id, 0, transformer_index)
                )
                transformer_index += 1
            return TransformerIDResult(input_ids, output_ids, [middle_id])
        else:
            # SDXL
            transformer_index = 0
            input_ids = []
            for block_id in [4, 5, 7, 8]:
                block_indices = (
                    range(2) if block_id in [4, 5] else range(10)
                )  # transformer_depth
                for index in block_indices:
                    input_ids.append(
                        TransformerID(
                            UnetBlockType.INPUT, block_id, index, transformer_index
                        )
                    )
                transformer_index += 1

            middle_ids = [
                TransformerID(UnetBlockType.MIDDLE, 0, index, transformer_index)
                for index in range(10)
            ]
            transformer_index += 1

            output_ids = []
            for block_id in range(6):
                block_indices = (
                    range(2) if block_id in [3, 4, 5] else range(10)
                )  # transformer_depth
                for index in block_indices:
                    output_ids.append(
                        TransformerID(
                            UnetBlockType.OUTPUT, block_id, index, transformer_index
                        )
                    )
                transformer_index += 1
            return TransformerIDResult(input_ids, output_ids, middle_ids)

    def is_compatible_with(self, other: "StableDiffusionVersion") -> bool:
        """Incompatible only when one of version is SDXL and other is not."""
        return (
            any(v == StableDiffusionVersion.UNKNOWN for v in [self, other])
            or sum(v == StableDiffusionVersion.SDXL for v in [self, other]) != 1
        )
