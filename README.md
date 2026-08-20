# Automated Segmentation and Height Measurement of Pituitary Gland

Code and configuration accompanying the paper **“Automated Segmentation and Height Measurement of Pituitary Gland.”**

```text
MRI -> deterministic ROI crop -> nnU-Net segmentation -> five-fold ensemble
    -> pituitary mask -> mask-based height estimation -> evaluation
```

The repository includes the cropped model workflow, the full-volume comparison, and the data-efficiency experiment.

## Model weights

Trained model weights will be released soon.

## Installation

The experiment environment uses Python 3.12.11, PyTorch 2.8.0, nnU-Net 2.6.2, and CUDA 12.8 for GPU training.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

## Data

The expert binary pituitary masks are included under `data/masks/`: 35 training masks and 21 held-out test masks. Download the corresponding IXI T1-weighted scans and prepare them as described in [data/README.md](data/README.md).

Prepare the full-volume comparison dataset:

```bash
python src/nnunet/prepare_dataset.py full \
  --images <IXI_TRAIN_IMAGES> --labels data/masks/train \
  --subjects data/splits/train_indices.txt \
  --output "$nnUNet_raw/Dataset002_PituitaryMorphometryFullVolume"
```

Prepare the cropped training dataset with five label-containing crop shifts per subject:

```bash
python src/nnunet/prepare_dataset.py roi \
  --images <IXI_TRAIN_IMAGES> --labels data/masks/train \
  --subjects data/splits/train_indices.txt --shifted-crops 5 --seed 42 \
  --output "$nnUNet_raw/Dataset001_PituitaryMorphometryROI"
```

## Training

Cropped five-fold training:

```bash
python src/nnunet/train.py 1 \
  --raw "$nnUNet_raw" --preprocessed "$nnUNet_preprocessed" --results "$nnUNet_results" \
  --validation-cases configs/validation_folds.json --case-count 175 \
  --folds 0 1 2 3 4 --device cuda
```

Full-volume comparison:

```bash
python src/nnunet/train.py 2 \
  --raw "$nnUNet_raw" --preprocessed "$nnUNet_preprocessed" --results "$nnUNet_results" \
  --folds 0 1 2 3 4 --device cuda
```

Run a data-efficiency experiment cell with or without crop-shift augmentation:

```bash
python experiments/data_efficiency/run.py --size 35 --arm with-shift \
  --dataset-id 101 --images <IXI_TRAIN_IMAGES> --labels data/masks/train \
  --nnunet-raw "$nnUNet_raw" --nnunet-preprocessed "$nnUNet_preprocessed" \
  --nnunet-results "$nnUNet_results"

python experiments/data_efficiency/run.py --size 35 --arm without-shift \
  --dataset-id 201 --images <IXI_TRAIN_IMAGES> --labels data/masks/train \
  --nnunet-raw "$nnUNet_raw" --nnunet-preprocessed "$nnUNet_preprocessed" \
  --nnunet-results "$nnUNet_results"
```

Repeat for training sizes `35 30 25 20 15 10 8 6 4 2`, assigning a distinct nnU-Net dataset ID to each run. Membership is fixed in `experiments/data_efficiency/subsets.json`.

## Inference and morphometry

Produce a full-resolution segmentation and all three height estimates:

```bash
python src/inference/predict.py <INPUT_SCAN> outputs/case_mask.nii.gz \
  --trainer-dir "$nnUNet_results/Dataset001_PituitaryMorphometryROI/nnUNetTrainer__nnUNetPlans__3d_fullres" \
  --folds 0 1 2 3 4 --device cuda
```

Measure an existing mask:

```bash
python src/morphometry/height.py outputs/case_mask.nii.gz
```

The reported estimators are PC2LrgCmp, AvgColLrgCmp, and LongVertLrgCmp.

## Evaluation

Evaluate the held-out test set:

```bash
python src/evaluation/evaluate_dataset.py \
  --images <IXI_TEST_IMAGES> --labels data/masks/test \
  --predictions outputs/test \
  --trainer-dir "$nnUNet_results/Dataset001_PituitaryMorphometryROI/nnUNetTrainer__nnUNetPlans__3d_fullres" \
  --output-csv outputs/test_metrics.csv --device cuda
```

Evaluate one segmentation pair:

```bash
python src/evaluation/segmentation_metrics.py prediction.nii.gz reference.nii.gz
```

Evaluate paired height estimates:

```bash
python src/evaluation/height_metrics.py height_pairs.csv
```

Paired estimates use `reference_mm,estimate_mm`.
