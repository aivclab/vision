import math
from typing import Any

import torch

__all__ = ["round_filters", "round_repeats", "drop_connect"]

from warg import Number


def round_filters(filters: Number, global_params: Any) -> int:
    """Calculate and round number of filters based on depth multiplier."""
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats: Number, global_params: Any) -> int:
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Drop connect."""
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    random_tensor = torch.rand(
        [batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
    )
    random_tensor += keep_prob
    binary_tensor = torch.floor(random_tensor)
    return inputs / keep_prob * binary_tensor


if __name__ == "__main__":
    print(drop_connect(torch.ones(2, 2), 0.5, True))
