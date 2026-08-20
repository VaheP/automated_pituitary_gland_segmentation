from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluation.segmentation_metrics import metrics_for_pair
from inference.predict import predict_case


def main():
    parser = argparse.ArgumentParser(description="Run inference and evaluate a prepared test set")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--trainer-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu", "mps"))
    args = parser.parse_args()

    args.predictions.mkdir(parents=True, exist_ok=True)
    rows = []
    for image in sorted(args.images.glob("*.nii.gz")):
        subject = image.name[:-7]
        prediction = args.predictions / f"{subject}_mask.nii.gz"
        predict_case(image, prediction, args.trainer_dir, device=args.device)
        rows.append({"image": image.name, **metrics_for_pair(prediction, args.labels / image.name)})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["image", "dice", "hd95_mm", "assd_mm", "spacing_mm", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
