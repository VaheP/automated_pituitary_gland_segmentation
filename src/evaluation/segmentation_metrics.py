from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes, apply_orientation, io_orientation, ornt_transform
from surface_distance import (
    compute_average_surface_distance,
    compute_dice_coefficient,
    compute_robust_hausdorff,
    compute_surface_distances,
)


def _load_3d(image: nib.Nifti1Image):
    data = np.asanyarray(image.dataobj)
    return data[..., 0] if data.ndim == 4 else data


def metrics_for_pair(prediction_path: Path, reference_path: Path):
    prediction = nib.load(str(prediction_path))
    reference = nib.load(str(reference_path))
    pred = _load_3d(prediction)
    ref = _load_3d(reference)
    if aff2axcodes(prediction.affine) != aff2axcodes(reference.affine):
        transform = ornt_transform(io_orientation(reference.affine), io_orientation(prediction.affine))
        ref = apply_orientation(ref, transform)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: prediction {pred.shape}, reference {ref.shape}")

    pred = pred > 0
    ref = ref > 0
    spacing = tuple(float(v) for v in prediction.header.get_zooms()[:3])
    if not pred.any() or not ref.any():
        dice = compute_dice_coefficient(ref, pred)
        dice = float(dice) if dice == dice else 1.0
        return {"dice": dice, "hd95_mm": float("nan"), "assd_mm": float("nan"), "spacing_mm": spacing, "note": "empty mask"}

    distances = compute_surface_distances(ref, pred, spacing_mm=spacing)
    ref_to_pred, pred_to_ref = compute_average_surface_distance(distances)
    return {
        "dice": float(compute_dice_coefficient(ref, pred)),
        "hd95_mm": float(compute_robust_hausdorff(distances, 95)),
        "assd_mm": float((ref_to_pred + pred_to_ref) / 2.0),
        "spacing_mm": spacing,
        "note": "",
    }


def main():
    parser = argparse.ArgumentParser(description="DSC, HD95, and ASSD for two NIfTI masks")
    parser.add_argument("prediction", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    print(json.dumps(metrics_for_pair(args.prediction, args.reference), indent=2))


if __name__ == "__main__":
    main()
