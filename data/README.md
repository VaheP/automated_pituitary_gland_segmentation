# IXI data layout

The release includes the expert binary pituitary masks used in the study:

```text
data/
  masks/
    train/   # 35 masks
    test/    # 21 masks
  splits/
    train_indices.txt
    test_indices.txt
    ixi_index_mapping.csv
```

Each mask is a compressed NIfTI file named with its study index. Background voxels have value `0`; pituitary voxels have value `1`.

IXI T1-weighted MRI volumes are distributed separately. Prepare the corresponding scans with matching numeric filenames:

```text
<IXI_IMAGES>/
  train/
    <subject_id>.nii.gz
  test/
    <subject_id>.nii.gz
```

Use `ixi_index_mapping.csv` to associate the study indices with IXI filenames and age groups. Image and mask filenames must match within each split.

The training masks are used for full-volume and ROI-cropped dataset preparation. The test masks are used only for held-out evaluation.
