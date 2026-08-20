from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import (
    aff2axcodes,
    apply_orientation,
    axcodes2ornt,
    inv_ornt_aff,
    io_orientation,
    ornt_transform,
)

CANONICAL_AXCODES = ("R", "A", "S")
MODEL_AXCODES = ("P", "S", "R")


def reorient_nifti(in_path: Path, out_path: Path, target_axcodes: tuple[str, str, str]):
    """Losslessly permute and flip a NIfTI to the requested storage orientation."""
    image = nib.load(str(in_path))
    data = np.asanyarray(image.dataobj)
    current = tuple(aff2axcodes(image.affine))
    if current == target_axcodes:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(image, str(out_path))
        return out_path

    transform = ornt_transform(io_orientation(image.affine), axcodes2ornt(target_axcodes))
    new_data = apply_orientation(data, transform)
    new_affine = image.affine @ inv_ornt_aff(transform, image.shape[:3])
    header = image.header.copy()
    header.set_data_shape(new_data.shape)
    header.set_data_dtype(data.dtype)
    output = nib.Nifti1Image(new_data, new_affine, header)
    output.set_sform(new_affine, code=1)
    output.set_qform(new_affine, code=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(out_path))
    return out_path


def save_mask_like_reference(
    mask_data: np.ndarray,
    mask_geometry_path: Path,
    reference_path: Path,
    out_path: Path,
):
    """Return a mask from working geometry to the reference NIfTI geometry."""
    mask_image = nib.load(str(mask_geometry_path))
    reference = nib.load(str(reference_path))
    transform = ornt_transform(io_orientation(mask_image.affine), io_orientation(reference.affine))
    output_data = (apply_orientation(np.asanyarray(mask_data), transform) > 0).astype(np.uint8)
    expected = tuple(int(v) for v in reference.shape[:3])
    if output_data.shape[:3] != expected:
        raise ValueError(f"restored mask shape {output_data.shape[:3]} does not match {expected}")

    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_data_shape(output_data.shape)
    output = nib.Nifti1Image(output_data, reference.affine, header)
    _, s_code = reference.get_sform(coded=True)
    _, q_code = reference.get_qform(coded=True)
    output.set_sform(reference.affine, code=int(s_code) if s_code else 1)
    output.set_qform(reference.affine, code=int(q_code) if q_code else 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(out_path))
    return out_path
