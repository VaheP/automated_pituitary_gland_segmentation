from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from morphometry.height import measure_file
from preprocessing.geometry import CANONICAL_AXCODES, MODEL_AXCODES, reorient_nifti, save_mask_like_reference
from preprocessing.roi_crop import crop_nifti

TRAIN_P005 = 60.0
TRAIN_P995 = 3461.0


def _rescale_intensities(in_path: Path, out_path: Path):
    image = nib.load(str(in_path))
    data = image.get_fdata(dtype=np.float32)
    foreground = data[data > 0]
    if foreground.size < 10:
        foreground = data.ravel()
    p005 = float(np.percentile(foreground, 0.5))
    p995 = float(np.percentile(foreground, 99.5))
    span = max(p995 - p005, 1e-6)
    data = (data - p005) * (TRAIN_P995 - TRAIN_P005) / span + TRAIN_P005
    data = np.clip(data, 0.0, TRAIN_P995)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data.astype(np.float32), image.affine, header), str(out_path))


def _restore_crop(crop_mask: Path, metadata: dict):
    crop = np.asanyarray(nib.load(str(crop_mask)).dataobj)
    if crop.ndim == 4:
        crop = crop[..., 0]
    output = np.zeros(tuple(metadata["full_shape_xyz"]), dtype=np.uint8)
    x0, y0, z0 = metadata["starts_xyz"]
    sx, sy, sz = metadata["sizes_xyz"]
    output[x0:x0 + sx, y0:y0 + sy, z0:z0 + sz] = (crop > 0).astype(np.uint8)
    return output


def predict_case(
    input_path: Path,
    output_path: Path,
    trainer_dir: Path,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    device: str = "cuda",
    use_mirroring: bool = True,
):
    """Run the located final crop, five-fold nnU-Net ensemble, and reconstruction."""
    with tempfile.TemporaryDirectory(prefix="pituitary_predict_") as temporary:
        work = Path(temporary)
        canonical = reorient_nifti(input_path, work / "input_ras.nii.gz", CANONICAL_AXCODES)
        metadata_path = work / "crop.json"
        metadata = crop_nifti(canonical, work / "crop_ras.nii.gz", metadata_path)
        model_crop = reorient_nifti(work / "crop_ras.nii.gz", work / "crop_psr.nii.gz", MODEL_AXCODES)
        normalized = work / "crop_norm.nii.gz"
        _rescale_intensities(model_crop, normalized)

        nnunet_input, nnunet_output = work / "nnunet_input", work / "nnunet_output"
        nnunet_input.mkdir()
        nnunet_output.mkdir()
        shutil.copyfile(normalized, nnunet_input / "case_0000_0000.nii.gz")
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=use_mirroring,
            perform_everything_on_device=True,
            device=torch.device(device),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(trainer_dir), use_folds=folds, checkpoint_name="checkpoint_final.pth"
        )
        predictor.predict_from_files_sequential(
            str(nnunet_input), str(nnunet_output), save_probabilities=False, overwrite=True
        )

        prediction_ras = reorient_nifti(nnunet_output / "case_0000.nii.gz", work / "prediction_ras.nii.gz", CANONICAL_AXCODES)
        restored = _restore_crop(prediction_ras, asdict(metadata))
        save_mask_like_reference(restored, canonical, input_path, output_path)
    return {"mask": str(output_path), "height": measure_file(output_path)}


def main():
    parser = argparse.ArgumentParser(description="Pituitary segmentation and three mask-based height estimates")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trainer-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--disable-mirroring", action="store_true")
    args = parser.parse_args()
    result = predict_case(
        args.input,
        args.output,
        args.trainer_dir,
        tuple(args.folds),
        args.device,
        not args.disable_mirroring,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
