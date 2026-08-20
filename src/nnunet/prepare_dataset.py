from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from preprocessing.crop_shift import mask_bbox, sample_containing_starts
from preprocessing.geometry import CANONICAL_AXCODES, MODEL_AXCODES, reorient_nifti
from preprocessing.roi_crop import crop_nifti, crop_with_window


def _read_subjects(path: Path):
    return [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def _save_crop(source: Path, starts: tuple[int, int, int], sizes: tuple[int, int, int], output: Path):
    nib.save(crop_with_window(nib.load(str(source)), starts, sizes), str(output))


def _write_dataset_json(dataset: Path, count: int, description: str):
    payload = {
        "name": "PituitaryGland",
        "description": description,
        "channel_names": {"0": "T1"},
        "labels": {"background": 0, "pituitary_gland": 1},
        "file_ending": ".nii.gz",
        "numTraining": count,
    }
    (dataset / "dataset.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_full_volume(images: Path, labels: Path, subjects: list[str], output: Path):
    images_tr, labels_tr = output / "imagesTr", output / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for index, subject in enumerate(subjects):
        case = f"case_{index:04d}"
        shutil.copyfile(images / f"{subject}.nii.gz", images_tr / f"{case}_0000.nii.gz")
        shutil.copyfile(labels / f"{subject}.nii.gz", labels_tr / f"{case}.nii.gz")
        mapping[subject] = [case]
    _write_dataset_json(output, len(subjects), "Pituitary gland segmentation (full volume)")
    (output / "subject_to_cases.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping


def prepare_roi(
    images: Path,
    labels: Path,
    subjects: list[str],
    output: Path,
    shifted_crops: int,
    seed: int,
    include_base: bool,
):
    images_tr, labels_tr = output / "imagesTr", output / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, list[str]] = {}
    case_index = 0
    with tempfile.TemporaryDirectory(prefix="pituitary_prepare_") as temporary:
        work_root = Path(temporary)
        for subject_index, subject in enumerate(subjects):
            work = work_root / subject
            work.mkdir()
            image_ras = reorient_nifti(images / f"{subject}.nii.gz", work / "image_ras.nii.gz", CANONICAL_AXCODES)
            label_ras = reorient_nifti(labels / f"{subject}.nii.gz", work / "label_ras.nii.gz", CANONICAL_AXCODES)
            metadata = crop_nifti(image_ras, work / "base.nii.gz")
            label_data = np.asanyarray(nib.load(str(label_ras)).dataobj)
            bbox = mask_bbox(label_data)
            starts = [metadata.starts_xyz] if include_base else []
            if shifted_crops:
                if bbox is None:
                    starts.extend([metadata.starts_xyz] * shifted_crops)
                else:
                    starts.extend(sample_containing_starts(
                        bbox[0], bbox[1], metadata.sizes_xyz, metadata.full_shape_xyz,
                        shifted_crops, random.Random(seed + subject_index),
                    ))
            if not starts:
                starts = [metadata.starts_xyz]

            mapping[subject] = []
            for variant, start in enumerate(starts):
                image_crop_ras, label_crop_ras = work / f"image_{variant}.nii.gz", work / f"label_{variant}.nii.gz"
                _save_crop(image_ras, start, metadata.sizes_xyz, image_crop_ras)
                _save_crop(label_ras, start, metadata.sizes_xyz, label_crop_ras)
                case = f"case_{case_index:04d}"
                reorient_nifti(image_crop_ras, images_tr / f"{case}_0000.nii.gz", MODEL_AXCODES)
                reorient_nifti(label_crop_ras, labels_tr / f"{case}.nii.gz", MODEL_AXCODES)
                if np.count_nonzero(nib.load(str(labels_tr / f"{case}.nii.gz")).get_fdata()) != np.count_nonzero(label_data):
                    raise RuntimeError(f"crop clipped the mask for subject {subject}, variant {variant}")
                mapping[subject].append(case)
                case_index += 1
    _write_dataset_json(output, case_index, "Pituitary gland segmentation (deterministic ROI crops)")
    (output / "subject_to_cases.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Prepare full-volume or ROI-cropped nnU-Net data")
    parser.add_argument("mode", choices=("full", "roi"))
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shifted-crops", type=int, default=0)
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    subjects = _read_subjects(args.subjects)
    if args.mode == "full":
        prepare_full_volume(args.images, args.labels, subjects, args.output)
    else:
        prepare_roi(args.images, args.labels, subjects, args.output, args.shifted_crops, args.seed, args.include_base)


if __name__ == "__main__":
    main()
