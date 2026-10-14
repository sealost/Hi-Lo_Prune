"""Utilities for producing one low-level patch feature per visual token."""

from typing import Sequence

import numpy as np
import torch
from PIL import Image


def image_to_merged_patch_features(
    image: Image.Image,
    grid_thw: Sequence[int] | torch.Tensor,
    *,
    patch_size: int = 16,
    merge_size: int = 2,
) -> torch.Tensor:
    """Convert one image into flattened features aligned with merged visual tokens."""
    temporal, grid_height, grid_width = [int(value) for value in grid_thw]
    if temporal != 1:
        raise ValueError("This helper currently supports a single still image")
    if grid_height % merge_size or grid_width % merge_size:
        raise ValueError("grid height and width must be divisible by merge_size")

    resized = image.convert("RGB").resize(
        (grid_width * patch_size, grid_height * patch_size),
        Image.Resampling.BICUBIC,
    )
    pixels = torch.from_numpy(np.asarray(resized, dtype=np.float32).copy())
    pixels = pixels.permute(2, 0, 1).div_(255.0).unsqueeze(0)

    merged_patch_size = patch_size * merge_size
    patches = pixels.unfold(2, merged_patch_size, merged_patch_size).unfold(
        3, merged_patch_size, merged_patch_size
    )
    return (
        patches.permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .view(-1, 3 * merged_patch_size * merged_patch_size)
    )
