from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def _environment(raw: Path, preprocessed: Path, results: Path):
    environment = dict(os.environ)
    environment.update(nnUNet_raw=str(raw), nnUNet_preprocessed=str(preprocessed), nnUNet_results=str(results))
    return environment


def install_validation_splits(validation_path: Path, case_count: int, output: Path):
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    all_cases = {f"case_{index:04d}" for index in range(case_count)}
    splits = []
    for fold in validation:
        val = set(fold)
        splits.append({"train": sorted(all_cases - val), "val": sorted(val)})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(splits, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Plan, preprocess, and train nnU-Net v2 folds")
    parser.add_argument("dataset_id", type=int)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--validation-cases", type=Path)
    parser.add_argument("--case-count", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    environment = _environment(args.raw, args.preprocessed, args.results)
    subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", str(args.dataset_id), "--verify_dataset_integrity"], env=environment, check=True)
    if args.validation_cases:
        if args.case_count is None:
            parser.error("--case-count is required with --validation-cases")
        dataset_dirs = list(args.preprocessed.glob(f"Dataset{args.dataset_id:03d}_*"))
        if len(dataset_dirs) != 1:
            raise RuntimeError(f"could not resolve preprocessed dataset {args.dataset_id}: {dataset_dirs}")
        install_validation_splits(args.validation_cases, args.case_count, dataset_dirs[0] / "splits_final.json")
    for fold in args.folds:
        subprocess.run([
            "nnUNetv2_train", str(args.dataset_id), "3d_fullres", str(fold),
            "-tr", "nnUNetTrainer", "-p", "nnUNetPlans", "-device", args.device,
        ], env=environment, check=True)


if __name__ == "__main__":
    main()
