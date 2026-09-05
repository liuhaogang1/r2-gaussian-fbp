"""Shared TIFF preprocessing for TomoPy center finding and TIGRE FBP."""

from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, zoom


def parse_config(path):
    values = {}
    if path is not None:
        path = Path(path)
    if path is None or not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def build_angles(args, n_views):
    config = parse_config(args.config)
    start = float(config.get("AngleFirst", args.angle_start))
    interval = float(config.get("AngleInterval", args.angle_interval))
    configured_count = int(config.get("NumberImages", n_views))
    if configured_count != n_views:
        raise ValueError(
            f"config declares {configured_count} views, but found {n_views} TIFF files"
        )

    angles_deg = start + np.arange(n_views, dtype=np.float32) * interval
    if (
        not args.keep_duplicate_endpoint
        and len(angles_deg) > 1
        and np.isclose(angles_deg[-1] - angles_deg[0], 180.0, atol=1e-4)
    ):
        angles_deg = angles_deg[:-1]
    return np.deg2rad(angles_deg).astype(np.float32)


def fill_nearest(image, bad):
    if not bad.any():
        return image
    if not (~bad).any():
        raise ValueError("A projection contains no valid pixels")
    _, indices = distance_transform_edt(
        bad, return_distances=True, return_indices=True
    )
    image[bad] = image[tuple(indices[:, bad])]
    return image


def process_projection(image, args):
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("A TIFF projection contains no finite pixels")

    finite_values = image[finite]
    cap = float(np.percentile(finite_values, args.clip_percentile))
    image = np.nan_to_num(image, nan=0.0, posinf=cap, neginf=0.0)

    if args.input_type in {"transmission", "intensity"}:
        bad = image <= 0
        if args.zero_policy == "nearest":
            image = fill_nearest(image, bad)
        elif args.zero_policy == "clip":
            positive = image[image > 0]
            if positive.size == 0:
                raise ValueError("A projection contains no positive intensity")
            image = np.maximum(image, np.percentile(positive, 0.1))
        image = np.maximum(image, args.log_eps)
        i0 = args.i0 if args.i0 > 0 else float(
            np.percentile(image[image > 0], args.i0_percentile)
        )
        if i0 <= 0 or not np.isfinite(i0):
            raise ValueError(f"Invalid I0 estimate: {i0}")
        image = np.maximum(-np.log(image / i0), 0.0)
    else:
        image = np.maximum(image, 0.0)

    if args.shift_v != 0:
        shifted = np.zeros_like(image)
        if args.shift_v > 0:
            shifted[:-args.shift_v] = image[args.shift_v:]
        else:
            shift = -args.shift_v
            shifted[shift:] = image[:-shift]
        image = shifted

    if args.pixel_subsample != 1:
        height, width = image.shape
        new_height = max(1, height // args.pixel_subsample)
        new_width = max(1, width // args.pixel_subsample)
        image = zoom(
            image,
            (new_height / height, new_width / width),
            order=args.resize_order,
        )
        if image.shape[0] > image.shape[1]:
            offset = (image.shape[0] - image.shape[1]) // 2
            image = image[offset : image.shape[0] - offset]
        elif image.shape[1] > image.shape[0]:
            offset = (image.shape[1] - image.shape[0]) // 2
            image = image[:, offset : image.shape[1] - offset]

    image *= np.float32(args.projection_scale)
    if not np.isfinite(image).all():
        raise ValueError("Processed projection contains NaN or Inf")
    return image.astype(np.float32)
