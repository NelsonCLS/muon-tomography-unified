"""Shared reconstruction post-processing operations."""

from __future__ import annotations

import numpy as np


def fill_empty_with_six_neighbor_mean(
    volume: np.ndarray,
    empty_mask: np.ndarray,
) -> np.ndarray:
    """Fill empty voxels from one synchronous six-face-neighbor mean pass.

    Every in-bounds face neighbor contributes, including neighbors that are
    themselves empty in the input. Non-empty voxels are left unchanged.
    """

    values = np.asarray(volume, dtype=np.float64)
    empty = np.asarray(empty_mask, dtype=bool)
    if values.ndim != 3 or empty.shape != values.shape:
        raise ValueError("volume and empty_mask must be matching 3D arrays")

    neighbor_sum = np.zeros_like(values)
    neighbor_count = np.zeros(values.shape, dtype=np.uint8)
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        lower_key = tuple(lower)
        upper_key = tuple(upper)
        neighbor_sum[lower_key] += values[upper_key]
        neighbor_count[lower_key] += 1
        neighbor_sum[upper_key] += values[lower_key]
        neighbor_count[upper_key] += 1

    filled = values.copy()
    filled[empty] = neighbor_sum[empty] / neighbor_count[empty]
    return filled
