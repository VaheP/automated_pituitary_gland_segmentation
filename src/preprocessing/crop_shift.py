from __future__ import annotations

import random

import numpy as np


def mask_bbox(mask: np.ndarray):
    indices = np.argwhere(mask > 0)
    if indices.size == 0:
        return None
    return tuple(int(v) for v in indices.min(0)), tuple(int(v) for v in indices.max(0))


def sample_containing_starts(
    bbox_min: tuple[int, int, int],
    bbox_max: tuple[int, int, int],
    crop_size: tuple[int, int, int],
    full_shape: tuple[int, int, int],
    n_samples: int,
    rng: random.Random,
):
    """Sample crop origins while keeping the complete nonzero mask in-frame."""
    ranges = []
    for axis in range(3):
        low = max(bbox_max[axis] - crop_size[axis] + 1, 0)
        high = min(bbox_min[axis], full_shape[axis] - crop_size[axis])
        ranges.append((low, high))

    if any(low > high for low, high in ranges):
        center = tuple((bbox_min[a] + bbox_max[a]) // 2 for a in range(3))
        start = [
            max(0, min(int(center[a] - crop_size[a] // 2), full_shape[a] - crop_size[a]))
            for a in range(3)
        ]
        return [tuple(start)] * n_samples

    seen: set[tuple[int, int, int]] = set()
    result: list[tuple[int, int, int]] = []
    attempts = 0
    while len(result) < n_samples and attempts < 5000:
        attempts += 1
        start = tuple(rng.randint(low, high) for low, high in ranges)
        if start not in seen:
            seen.add(start)
            result.append(start)
    while len(result) < n_samples:
        result.append(result[len(result) % len(result)])
    return result
