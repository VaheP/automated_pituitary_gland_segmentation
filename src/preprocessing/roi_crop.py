from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from scipy import ndimage


@dataclass(frozen=True)
class CropMetadata:
    starts_xyz: tuple[int, int, int]
    sizes_xyz: tuple[int, int, int]
    full_shape_xyz: tuple[int, int, int]
    slice_axis: int
    inplane_axes: tuple[int, int]
    tilt_angle_deg: float


def _as_3d(data: np.ndarray, volume: int = 0):
    data = np.asanyarray(data)
    if data.ndim == 3:
        return data
    if data.ndim >= 4:
        return data[(slice(None), slice(None), slice(None), volume) + (0,) * (data.ndim - 4)]
    raise ValueError(f"unsupported NIfTI dimensionality: {data.ndim}")


def _plane_axes(affine: np.ndarray, plane: str):
    targets = {"sagittal": {"L", "R"}, "coronal": {"A", "P"}, "axial": {"S", "I"}}
    if plane not in targets:
        raise ValueError(f"unknown plane: {plane}")
    codes = aff2axcodes(affine)
    slice_axis = next(i for i, code in enumerate(codes) if code in targets[plane])
    inplane = tuple(i for i in range(3) if i != slice_axis)
    return slice_axis, (inplane[0], inplane[1])


def _uint8(slice_2d: np.ndarray):
    values = np.asanyarray(slice_2d).astype(np.float32, copy=False)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(values, dtype=np.uint8)
    return ((values - low) * (255.0 / (high - low))).clip(0, 255).astype(np.uint8)


def _largest_component(mask: np.ndarray):
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    return labels == int(np.argmax(sizes) + 1)


def _tilt_angle(mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size < 100:
        return 0.0
    x, y = xs.astype(np.float32), ys.astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    _, eigenvectors = np.linalg.eigh(np.cov(np.vstack([x, y])))
    vx, vy = eigenvectors[:, -1]
    return float(np.degrees(np.arctan2(vy, vx)))


def _clamp_square(x0: int, y0: int, side: int, height: int, width: int):
    side = min(max(int(side), 1), height, width)
    return int(np.clip(x0, 0, width - side)), int(np.clip(y0, 0, height - side)), side


def _rotate_points(points: np.ndarray, center: np.ndarray, angle_deg: float):
    angle = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    return (points - center) @ rotation.T + center


def _map_roi_back(rx: int, ry: int, side: int, shape: tuple[int, int], angle: float):
    height, width = shape
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
    corners = np.array([[rx, ry], [rx + side, ry], [rx + side, ry + side], [rx, ry + side]], dtype=np.float32)
    back = _rotate_points(corners, center, angle)
    min_x, max_x = int(np.floor(back[:, 0].min())), int(np.ceil(back[:, 0].max()))
    min_y, max_y = int(np.floor(back[:, 1].min())), int(np.ceil(back[:, 1].max()))
    min_x, max_x = max(0, min(min_x, width - 1)), max(0, min(max_x, width))
    min_y, max_y = max(0, min(min_y, height - 1)), max(0, min(max_y, height))
    side_out = max(1, max_x - min_x, max_y - min_y)
    x0 = (min_x + max_x) // 2 - side_out // 2
    y0 = (min_y + max_y) // 2 - side_out // 2
    return _clamp_square(x0, y0, side_out, height, width)


def find_roi(
    image: nib.Nifti1Image,
    *,
    plane: str = "sagittal",
    threshold_min: int = 30,
    threshold_max: int = 255,
    crop_divisor: float = 3.3,
    border_crop: int = 5,
    depth_radius: int = 15,
    enable_tilt: bool = True,
):
    data = _as_3d(image.dataobj)
    slice_axis, inplane_axes = _plane_axes(image.affine, plane)
    n_slices = int(data.shape[slice_axis])
    middle = ((n_slices + 1) // 2) - 1
    mid = _uint8(np.take(data, middle, axis=slice_axis))[::-1, ::-1]

    if border_crop:
        if min(mid.shape) <= 2 * border_crop:
            raise ValueError(f"border crop {border_crop} is too large for {mid.shape}")
        working = mid[border_crop:-border_crop, border_crop:-border_crop]
    else:
        working = mid
    mask = (working >= threshold_min) & (working <= threshold_max)
    mask = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7), dtype=bool))
    mask = _largest_component(mask)
    if not mask.any():
        raise RuntimeError("could not find skull proxy")

    angle = _tilt_angle(mask) if enable_tilt else 0.0
    rotated = (
        ndimage.rotate(mask.astype(np.uint8), -angle, reshape=False, order=0, mode="constant", cval=0).astype(bool)
        if abs(angle) > 1e-3
        else mask
    )
    ys, xs = np.where(rotated)
    min_x, max_x, min_y, max_y = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    if min_x >= max_x or min_y >= max_y:
        raise RuntimeError("could not find skull bounds")
    center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
    side = max(1, int(np.floor(max(max_x - min_x, max_y - min_y) / crop_divisor)))
    rx, ry, side = _clamp_square(center_x - side // 2, center_y - side // 2, side, *working.shape)
    if enable_tilt and abs(angle) > 1e-3:
        rx, ry, side = _map_roi_back(rx, ry, side, working.shape, angle)

    start_1based = max(1, (n_slices + 1) // 2 - depth_radius)
    end_1based = min(n_slices, (n_slices + 1) // 2 + depth_radius)
    depth = end_1based - start_1based + 1
    in0, in1 = inplane_axes
    n0, n1 = int(data.shape[in0]), int(data.shape[in1])
    roi0 = max(0, min(n0 - (rx + border_crop + side), n0 - side))
    roi1 = max(0, min(n1 - (ry + border_crop + side), n1 - side))
    s0 = max(0, min(start_1based - 1, n_slices - depth))
    depth = max(1, min(depth, n_slices - s0))

    starts, sizes = [0, 0, 0], [0, 0, 0]
    starts[in0], starts[in1], starts[slice_axis] = roi0, roi1, s0
    sizes[in0], sizes[in1], sizes[slice_axis] = side, side, depth
    return CropMetadata(tuple(starts), tuple(sizes), tuple(int(v) for v in data.shape[:3]), slice_axis, inplane_axes, angle)


def crop_with_window(image: nib.Nifti1Image, starts: tuple[int, int, int], sizes: tuple[int, int, int]):
    data = _as_3d(image.dataobj)
    x0, y0, z0 = starts
    sx, sy, sz = sizes
    if min(starts) < 0 or any(starts[i] + sizes[i] > data.shape[i] for i in range(3)):
        raise ValueError(f"crop is out of bounds: starts={starts}, sizes={sizes}, shape={data.shape}")
    cropped = np.asanyarray(data[x0:x0 + sx, y0:y0 + sy, z0:z0 + sz])
    translation = np.eye(4)
    translation[:3, 3] = starts
    affine = image.affine @ translation
    header = image.header.copy()
    header.set_data_shape(cropped.shape)
    output = nib.Nifti1Image(cropped, affine, header)
    _, s_code = image.get_sform(coded=True)
    _, q_code = image.get_qform(coded=True)
    if s_code:
        output.set_sform(affine, code=int(s_code))
    if q_code:
        output.set_qform(affine, code=int(q_code))
    return output


def crop_nifti(in_path: Path, out_path: Path, metadata_path: Path | None = None, **kwargs):
    image = nib.load(str(in_path))
    metadata = find_roi(image, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(crop_with_window(image, metadata.starts_xyz, metadata.sizes_xyz), str(out_path))
    if metadata_path:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Deterministic pituitary ROI crop")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--crop-divisor", type=float, default=3.3)
    parser.add_argument("--depth-radius", type=int, default=15)
    args = parser.parse_args()
    crop_nifti(args.input, args.output, args.metadata, crop_divisor=args.crop_divisor, depth_radius=args.depth_radius)


if __name__ == "__main__":
    main()
