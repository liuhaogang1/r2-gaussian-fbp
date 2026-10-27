"""Opt-in TomoPy correction at native detector resolution, before minus-log.

This module is imported only by explicitly enabled processing paths. TomoPy
and plotting dependencies are loaded only when they are actually needed.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.fbp_preprocess import (
    correct_flat_field, estimate_free_ray_i0, fill_nearest, parse_free_ray,
    process_projection,
)


DEFAULTS = {"snr": 3.0, "la_size": 61, "sm_size": 21, "dim": 1}


def normalize_options(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("stripe_correction must be an object or null")
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown stripe_correction keys: " + ", ".join(sorted(unknown)))
    options = dict(DEFAULTS, **value)
    snr = options["snr"]
    if isinstance(snr, bool) or not isinstance(snr, (int, float)) or not np.isfinite(snr) or snr <= 0:
        raise ValueError("stripe_correction.snr must be positive and finite")
    options["snr"] = float(snr)
    for key in ("la_size", "sm_size"):
        size = options[key]
        if isinstance(size, bool) or not isinstance(size, int) or size < 3 or size % 2 != 1:
            raise ValueError(f"stripe_correction.{key} must be an odd integer >= 3")
    if isinstance(options["dim"], bool) or not isinstance(options["dim"], int) or options["dim"] not in (1, 2):
        raise ValueError("stripe_correction.dim must be 1 or 2")
    return options


def _prepare_native(image, args, flat):
    """Mirror existing intensity repair and I0 selection without taking log."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).any():
        raise ValueError("A TIFF projection must be 2D and contain finite pixels")
    cap = float(np.percentile(image[np.isfinite(image)], args.clip_percentile))
    image = np.nan_to_num(image, nan=0.0, posinf=cap, neginf=0.0)
    if getattr(args, "flat_field", False):
        if flat is None:
            raise ValueError("--flat_field requires a loaded flat-field image")
        image = correct_flat_field(image, flat, args.dark_value, args.flat_eps)
    if args.input_type == "line_integral":
        return np.maximum(image, 0.0)
    if not np.isfinite(args.log_eps) or args.log_eps <= 0:
        raise ValueError("--log_eps must be positive and finite")
    if args.zero_policy == "nearest":
        image = fill_nearest(image, image <= 0)
    elif args.zero_policy == "clip":
        positive = image[image > 0]
        if positive.size == 0:
            raise ValueError("A projection contains no positive intensity")
        image = np.maximum(image, np.percentile(positive, 0.1))
    image = np.maximum(image, args.log_eps)
    mode = getattr(args, "i0_mode", "percentile")
    if mode == "free_ray":
        i0 = estimate_free_ray_i0(image, parse_free_ray(args.free_ray),
                                 float(getattr(args, "i0_hi_percentile", 99.9)))
    elif mode == "fixed" or args.i0 > 0:
        i0 = float(args.i0)
    else:
        i0 = float(np.percentile(image[image > 0], args.i0_percentile))
    if not np.isfinite(i0) or i0 <= 0:
        raise ValueError(f"Invalid I0 estimate: {i0}")
    return np.asarray(image / i0, dtype=np.float32)


def finish_projections(native_integrals, args):
    """Use the original spatial processing and scale on corrected integrals."""
    options = argparse.Namespace(**vars(args))
    options.input_type = "line_integral"
    options.flat_field = False
    return np.stack([process_projection(image, options) for image in native_integrals], axis=0)


def save_sinograms(before, after, angles, directory, row, domain):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    low = float(min(before.min(), after.min()))
    high = float(max(before.max(), after.max()))
    if high <= low:
        high = low + max(1e-6, abs(low) * 1e-6)
    angle_deg = np.rad2deg(np.asarray(angles))
    files = []
    for name, sino in (("before", before), ("after", after)):
        filename = f"stripe_sinogram_{name}.png"
        fig = Figure(figsize=(9, 6))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        # Every acquisition angle gets one equally spaced image row. Label
        # those rows with their actual angles rather than assuming a range.
        artist = ax.imshow(sino, cmap="gray", vmin=low, vmax=high,
                           aspect="auto", origin="upper", interpolation="nearest")
        ticks = np.unique(np.linspace(0, len(angle_deg) - 1, min(7, len(angle_deg))).round().astype(int))
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{angle_deg[i]:.3g}" for i in ticks])
        ax.set_xlabel("Detector column (native pixels)")
        ax.set_ylabel("Acquisition angle (deg)")
        ax.set_title(f"Stripe correction: {name}; native detector row {row}")
        fig.colorbar(artist, ax=ax, label=domain.replace("_", " "))
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=150)
        files.append(filename)
    return {"before": files[0], "after": files[1], "vmin": low, "vmax": high}


def prepare_projections(images, args, flat=None, angles=None, preview_dir=None):
    options = normalize_options(getattr(args, "stripe_correction", None))
    if options is None:
        raise ValueError("Stripe processing requires explicit stripe_correction options")
    ncore = getattr(args, "stripe_ncore", 8)
    chunk = getattr(args, "stripe_row_chunk", 32)
    if not isinstance(ncore, int) or ncore < 1 or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("stripe_ncore and stripe_row_chunk must be positive integers")
    try:
        import tomopy
    except ImportError as exc:
        raise RuntimeError("Stripe correction requires TomoPy in the active environment; "
                           "install it or omit --stripe_correction") from exc
    native = np.stack([_prepare_native(image, args, flat) for image in images], axis=0)
    if native.shape[0] < 2 or native.shape[2] < 8 or not np.isfinite(native).all():
        raise ValueError("Stripe correction needs >= 2 views, >= 8 detector columns and finite data")
    if angles is None or np.asarray(angles).shape != (native.shape[0],) or not np.isfinite(angles).all():
        raise ValueError("Stripe preview angles must match all input views and be finite")
    row = native.shape[1] // 2
    before = native[:, row, :].copy()
    domain = "line_integral" if args.input_type == "line_integral" else "normalized_transmission"
    for start in range(0, native.shape[1], chunk):
        block = np.ascontiguousarray(native[:, start:start + chunk, :])
        corrected = np.asarray(tomopy.remove_all_stripe(block, **options, ncore=ncore), dtype=np.float32)
        if corrected.shape != block.shape or not np.isfinite(corrected).all():
            raise ValueError("TomoPy stripe correction returned a wrong shape or NaN/Inf")
        if domain == "normalized_transmission" and np.any(corrected <= 0):
            raise ValueError("TomoPy returned nonpositive transmission; inspect the input/stripe settings")
        native[:, start:start + chunk, :] = corrected
    after = native[:, row, :].copy()
    info = {"algorithm": "tomopy.remove_all_stripe", "parameters": options,
            "tomopy_version": str(getattr(tomopy, "__version__", "unknown")),
            "domain": domain, "native_shape": list(native.shape),
            "native_detector_row": row, "ncore": ncore, "row_chunk": chunk,
            "stage": "before_spatial_resampling_and_projection_scale"}
    if preview_dir is not None:
        info["sinogram_preview"] = save_sinograms(before, after, angles, preview_dir, row, domain)
    if domain == "normalized_transmission":
        # Positivity was checked above; no synthetic epsilon/fallback pixels.
        np.log(native, out=native)
        np.negative(native, out=native)
    np.maximum(native, 0.0, out=native)
    return finish_projections(native, args), native, info


def processed_metadata(path, options):
    """Require provenance instead of applying correction twice to saved NPY."""
    path = Path(path)
    metadata_path = path.parent / ("preprocess_meta.json" if path.name == "proj_preprocessed_all.npy" else "meta_data.json")
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Corrected --processed_npy requires its preprocessing metadata") from exc
    info = data.get("stripe_correction") or data.get("source", {}).get("stripe_correction")
    if not isinstance(info, dict) or info.get("algorithm") != "tomopy.remove_all_stripe" or info.get("parameters") != normalize_options(options):
        raise ValueError("--processed_npy stripe provenance does not match the requested correction")
    return info


def dataset_metadata(dataset, options):
    """Validate requested settings before reusing an existing dataset."""
    data = json.loads((Path(dataset) / "meta_data.json").read_text(encoding="utf-8"))
    info = data.get("source", {}).get("stripe_correction")
    if not isinstance(info, dict) or info.get("algorithm") != "tomopy.remove_all_stripe" or info.get("parameters") != normalize_options(options):
        raise ValueError("Dataset stripe settings do not match; regenerate it in a new output directory")
    return data


def load_train_volume(args, reconstruct_train, expected_shape):
    """Correction-aware physical FBP cache, enabled only by an explicit flag."""
    dataset = args.dataset.resolve()
    data = dataset_metadata(dataset, args.stripe_correction)
    output = args.fbp_output.resolve() if args.fbp_output is not None else dataset / "vol_train_fbp.npy"
    signature = {
        "version": 1, "scanner": data["scanner"],
        "stripe_correction": data["source"]["stripe_correction"],
        "proj_train": data["proj_train"],
        "projection_stats": [
            [str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns]
            for record in data["proj_train"]
            for path in [dataset / str(record["file_path"]).replace("\\", "/")]
        ],
    }
    sidecar = output.with_suffix(".stripe.json")
    cached = None
    try:
        cached = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if output.exists() and cached == signature:
        volume = np.asarray(np.load(output), dtype=np.float32)
    else:
        volume = np.asarray(reconstruct_train(dataset), dtype=np.float32)
        if volume.ndim != 3 or not np.isfinite(volume).all():
            raise ValueError("Stripe-aware train FBP must be a finite 3D volume")
        shape = tuple(int(n) for n in expected_shape(dataset))
        if shape and volume.shape != shape:
            raise ValueError("Stripe-aware train FBP shape does not match scanner nVoxel")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, volume)
        sidecar.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    shape = tuple(int(n) for n in expected_shape(dataset))
    if volume.ndim != 3 or not np.isfinite(volume).all() or (shape and volume.shape != shape):
        raise ValueError("Invalid cached stripe-aware train FBP volume")
    return volume, output
