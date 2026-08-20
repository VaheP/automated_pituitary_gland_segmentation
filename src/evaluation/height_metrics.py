from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def height_errors(reference_mm, estimate_mm):
    reference = np.asarray(reference_mm, dtype=float)
    estimate = np.asarray(estimate_mm, dtype=float)
    if reference.shape != estimate.shape or reference.size == 0:
        raise ValueError("reference and estimate must be nonempty arrays with identical shape")
    errors = estimate - reference
    return {
        "mae_mm": float(np.mean(np.abs(errors))),
        "rmse_mm": float(np.sqrt(np.mean(errors ** 2))),
    }


def main():
    parser = argparse.ArgumentParser(description="MAE and RMSE for height measurements")
    parser.add_argument("csv_file", type=Path, help="CSV with reference_mm and estimate_mm columns")
    args = parser.parse_args()
    with args.csv_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    print(json.dumps(height_errors([r["reference_mm"] for r in rows], [r["estimate_mm"] for r in rows]), indent=2))


if __name__ == "__main__":
    main()
