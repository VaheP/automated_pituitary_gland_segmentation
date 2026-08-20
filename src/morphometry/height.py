from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as label_3d
from skimage.measure import label as label_2d
from skimage.measure import regionprops

FIXED_INPLANE_SPACING_MM = 0.94


def _largest_component_3d(mask: np.ndarray):
    labels, count = label_3d(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def pc2_lrg_cmp(mask: np.ndarray, affine: np.ndarray, keep_largest_component: bool = True):
    """PC2LrgCmp in physical millimetres for a mask stored in (Z,Y,X) order."""
    mask = np.asanyarray(mask) > 0
    if not mask.any():
        raise ValueError("mask is empty")
    if keep_largest_component:
        mask = _largest_component_3d(mask)

    _, _, size_x = mask.shape
    areas = np.array([mask[:, :, x].sum() for x in range(size_x)])
    x_index = int(np.argmax(areas))
    z_index, y_index = np.nonzero(mask[:, :, x_index])
    if z_index.size < 2:
        return 0.0
    ijk = np.column_stack([z_index, y_index, np.full_like(z_index, x_index)])
    xyz = (affine @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
    yz = xyz[:, [1, 2]]
    centered = yz - yz.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    pc2 = axes[1].copy()
    if pc2[1] < 0:
        pc2 *= -1
    projections = centered @ pc2
    return float(projections.max() - projections.min())


def _longest_horizontal_run(row: np.ndarray):
    best_length = 0
    best_start = best_end = None
    in_run = False
    run_start = None
    for column, value in enumerate(row):
        if value and not in_run:
            in_run = True
            run_start = column
        elif not value and in_run:
            in_run = False
            run_end = column - 1
            length = run_end - run_start + 1
            if length > best_length:
                best_length, best_start, best_end = length, run_start, run_end
    if in_run:
        run_end = len(row) - 1
        length = run_end - run_start + 1
        if length > best_length:
            best_length, best_start, best_end = length, run_start, run_end
    return best_start, best_end, best_length


def _largest_sagittal_slice(mask: np.ndarray):
    counts = np.array([np.count_nonzero(mask[:, :, x]) for x in range(mask.shape[2])], dtype=np.int64)
    if counts.max(initial=0) == 0:
        raise ValueError("mask is empty")
    candidates = np.flatnonzero(counts == counts.max())
    return int(candidates[np.argmin(np.abs(candidates - mask.shape[2] // 2))])


def avg_col_lrg_cmp(mask: np.ndarray, spacing_mm: float = FIXED_INPLANE_SPACING_MM):
    """AvgColLrgCmp using a fixed 0.94 mm in-plane spacing."""
    mask = np.asanyarray(mask) > 0
    if mask.ndim != 3:
        raise ValueError("expected a 3D mask in (Z,Y,X) array order")
    sagittal = mask[:, :, _largest_sagittal_slice(mask)]
    occupied_rows = np.where(sagittal.any(axis=1))[0]
    middle = (int(occupied_rows.min()) + int(occupied_rows.max())) // 2
    rows = [r for r in (middle - 1, middle, middle + 1) if 0 <= r < mask.shape[0]]
    k = 2
    while len(rows) < 3 and (middle - k >= occupied_rows.min() or middle + k <= occupied_rows.max()):
        if middle - k >= occupied_rows.min() and middle - k not in rows:
            rows.insert(0, middle - k)
        if len(rows) < 3 and middle + k <= occupied_rows.max() and middle + k not in rows:
            rows.append(middle + k)
        k += 1
    lengths = [_longest_horizontal_run(sagittal[row])[2] for row in rows[:3]]
    valid = [length for length in lengths if length > 0]
    return float(np.mean(valid) if valid else 0.0) * spacing_mm


def long_vert_lrg_cmp(mask: np.ndarray, spacing_mm: float = FIXED_INPLANE_SPACING_MM):
    """LongVertLrgCmp using the largest 2-D sagittal component and fixed 0.94 mm."""
    mask = np.asanyarray(mask) > 0
    if mask.ndim != 3:
        raise ValueError("expected a 3D mask in (Z,Y,X) array order")
    best_area = 0
    best_component = None
    for x_index in range(mask.shape[2]):
        labels, count = label_2d(mask[:, :, x_index], return_num=True)
        if count == 0:
            continue
        largest = max(regionprops(labels), key=lambda region: region.area)
        if largest.area > best_area:
            best_area = largest.area
            best_component = labels == largest.label
    if best_component is None:
        raise ValueError("mask is empty")
    length = max(_longest_horizontal_run(row)[2] for row in best_component)
    return float(length) * spacing_mm


def measure_file(path: Path):
    image = nib.load(str(path))
    mask = image.get_fdata() > 0
    return {
        "PC2LrgCmp_mm": pc2_lrg_cmp(mask, image.affine),
        "AvgColLrgCmp_mm": avg_col_lrg_cmp(mask),
        "LongVertLrgCmp_mm": long_vert_lrg_cmp(mask),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure pituitary height from a binary NIfTI mask")
    parser.add_argument("mask", type=Path)
    args = parser.parse_args()
    print(json.dumps(measure_file(args.mask), indent=2))


if __name__ == "__main__":
    main()
