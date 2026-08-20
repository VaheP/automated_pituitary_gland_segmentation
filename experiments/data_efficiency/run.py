from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from nnunet.prepare_dataset import prepare_roi


def grouped_validation(mapping: dict[str, list[str]], folds: int, seed: int = 42):
    subjects = sorted(mapping)
    random.Random(seed).shuffle(subjects)
    groups = [subjects[index::folds] for index in range(folds)]
    return [sorted(case for subject in group for case in mapping[subject]) for group in groups]


def main():
    parser = argparse.ArgumentParser(description="Run one nested data-efficiency training cell")
    parser.add_argument("--size", type=int, required=True, choices=[35, 30, 25, 20, 15, 10, 8, 6, 4, 2])
    parser.add_argument("--arm", choices=("with-shift", "without-shift"), required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--nnunet-results", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    metadata = json.loads((Path(__file__).with_name("subsets.json")).read_text(encoding="utf-8"))
    subjects = metadata[str(args.size)]
    dataset = args.nnunet_raw / f"Dataset{args.dataset_id:03d}_PituitaryMorphometry"
    mapping = prepare_roi(
        args.images,
        args.labels,
        subjects,
        dataset,
        shifted_crops=5 if args.arm == "with-shift" else 0,
        seed=42,
        include_base=False,
    )
    folds = min(5, args.size)
    validation = dataset / "validation_cases.json"
    validation.write_text(json.dumps(grouped_validation(mapping, folds), indent=2), encoding="utf-8")
    if args.prepare_only:
        return
    release_root = Path(__file__).resolve().parents[2]
    subprocess.run([
        sys.executable, str(release_root / "src/nnunet/train.py"), str(args.dataset_id),
        "--raw", str(args.nnunet_raw), "--preprocessed", str(args.nnunet_preprocessed),
        "--results", str(args.nnunet_results), "--folds", *map(str, range(folds)),
        "--validation-cases", str(validation), "--case-count", str(sum(map(len, mapping.values()))),
        "--device", args.device,
    ], check=True)


if __name__ == "__main__":
    main()
