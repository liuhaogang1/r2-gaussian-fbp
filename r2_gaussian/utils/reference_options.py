"""Shared options for reference supervision and the pipeline CLI."""
import argparse
import math
import warnings


REFERENCE_DEFAULTS = {
    "reference_enabled": False,
    "reference_method": "slice",
    "reference_path": None,
    "reference_loss": "l1_grad",
    "reference_weight": 0.05,
    "reference_grad_weight": 0.1,
    "reference_start_iter": 500,
    "reference_end_iter": 15000,
    "reference_interval": 50,
    "reference_seed": 0,
    "reference_slice_axis": "z",
    "reference_slice_sample_ratio": 1.0,
    "reference_block_size": (32, 32, 32),
    "reference_block_stride": None,  # Half the block size on each axis.
    "reference_block_sampling": "density_mix",
    "reference_block_density_mix": 0.7,
}

REFERENCE_ALIASES = {
    "slice_loss_enabled": "reference_enabled",
    "slice_reference_path": "reference_path",
    "slice_axis": "reference_slice_axis",
    "slice_start_iter": "reference_start_iter",
    "slice_end_iter": "reference_end_iter",
    "slice_interval": "reference_interval",
    "slice_filter_ratio": "reference_slice_sample_ratio",
    "lambda_slice": "reference_weight",
    "slice_seed": "reference_seed",
}
RETIRED_REFERENCE_OPTIONS = {
    "slice_filter_threshold": float,
    "slice_max_points": int,
    "slice_query_chunk_size": int,
}


def add_reference_arguments(parser, prefix=""):
    for name, default in REFERENCE_DEFAULTS.items():
        options = {"type": type(default) if default is not None else str}
        if name == "reference_enabled":
            options = {"action": argparse.BooleanOptionalAction}
        elif name in ("reference_block_size", "reference_block_stride"):
            options = {"type": int, "nargs": 3, "metavar": ("X", "Y", "Z")}
        parser.add_argument("--" + prefix + name, default=argparse.SUPPRESS, **options)
    for name, replacement in REFERENCE_ALIASES.items():
        default = REFERENCE_DEFAULTS[replacement]
        options = {"type": type(default) if default is not None else str}
        if isinstance(default, bool):
            options = {"action": argparse.BooleanOptionalAction}
        parser.add_argument("--" + prefix + name, default=argparse.SUPPRESS,
                            help=argparse.SUPPRESS, **options)
    for name, option_type in RETIRED_REFERENCE_OPTIONS.items():
        parser.add_argument("--" + prefix + name, type=option_type,
                            default=argparse.SUPPRESS, help=argparse.SUPPRESS)


def migrate_reference_options(options, prefix=""):
    """Resolve explicit options before inserting defaults."""
    values = dict(options)
    path_key = prefix + "reference_path"
    if values.get(path_key) == "":
        values[path_key] = None
    for old, new in REFERENCE_ALIASES.items():
        old, new = prefix + old, prefix + new
        if old not in values:
            continue
        value = values.pop(old)
        if old == prefix + "slice_filter_ratio":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{old} must be a numeric percentage")
            value /= 100
        if new == path_key and value == "":
            value = None
        if new in values and values[new] != value:
            raise ValueError(f"Conflicting reference options: {old} and {new}")
        values[new] = value
        warnings.warn(f"{old} is deprecated; use {new}", FutureWarning, stacklevel=2)
    for name in RETIRED_REFERENCE_OPTIONS:
        key = prefix + name
        if key in values:
            values.pop(key)
            warnings.warn(f"{key} is retired and has no effect on reference supervision",
                          FutureWarning, stacklevel=2)
    return values


def validate_reference_options(options, prefix=""):
    values = {name: options.get(prefix + name, default)
              for name, default in REFERENCE_DEFAULTS.items()}
    if not isinstance(values["reference_enabled"], bool):
        raise ValueError("reference_enabled must be a boolean")
    if not values["reference_enabled"]:
        return values
    if options.get(prefix + "pose_enabled", False):
        raise ValueError("reference_enabled cannot be combined with pose_enabled")
    if not values["reference_path"]:
        raise ValueError("reference_path is required when reference_enabled=true")
    if values["reference_method"] not in ("slice", "block"):
        raise ValueError("reference_method must be slice or block")
    if values["reference_loss"] not in ("l1", "l1_grad"):
        raise ValueError("reference_loss must be l1 or l1_grad")

    for name, minimum in (("reference_start_iter", 1), ("reference_end_iter", 1),
                          ("reference_interval", 1), ("reference_seed", 0)):
        value = values[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if values["reference_end_iter"] < values["reference_start_iter"]:
        raise ValueError("reference_end_iter must be >= reference_start_iter")
    for name in ("reference_weight", "reference_grad_weight",
                 "reference_slice_sample_ratio", "reference_block_density_mix"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and numeric")
    if min(values["reference_weight"], values["reference_grad_weight"]) < 0:
        raise ValueError("reference weights must be non-negative")

    if values["reference_slice_axis"] not in ("x", "y", "z"):
        raise ValueError("reference_slice_axis must be x, y or z")
    if not 0 < values["reference_slice_sample_ratio"] <= 1:
        raise ValueError("reference_slice_sample_ratio must be in (0, 1]")
    if values["reference_block_sampling"] not in ("uniform", "density_mix"):
        raise ValueError("reference_block_sampling must be uniform or density_mix")
    if not 0 <= values["reference_block_density_mix"] < 1:
        raise ValueError("reference_block_density_mix must be in [0, 1)")

    for name in ("reference_block_size", "reference_block_stride"):
        value = values[name]
        if name == "reference_block_stride" and value is None:
            continue
        if (not isinstance(value, (list, tuple)) or len(value) != 3
                or any(type(item) is not int or item < 1 for item in value)):
            raise ValueError(f"{name} must contain three positive integers")
    stride = values["reference_block_stride"]
    if stride is not None and any(step > size for step, size
                                  in zip(stride, values["reference_block_size"])):
        raise ValueError("reference_block_stride cannot exceed reference_block_size")
    return values
