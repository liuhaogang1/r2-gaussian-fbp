"""Validate completed stages before allowing pipeline resume."""
from pathlib import Path
import json

from r2_gaussian.utils.artifact_utils import file_identity, identity


def init_metadata_path(args):
    if args.init_source == "custom":
        if args.init_volume is None:
            raise ValueError("--init_volume is required when --init_source=custom")
        adjacent = Path(args.init_volume).resolve().with_name("meta_data.json")
        if adjacent.is_file():
            return adjacent
    return Path(args.output_dir) / "meta_data.json"


def _files(paths):
    result = []
    for value in paths:
        if value is None:
            continue
        path = Path(value).resolve()
        if path.is_dir():
            result.extend(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            result.append(path)
        else:
            raise FileNotFoundError(f"Required stage input/output is missing: {path}")
    return sorted(set(result))


def signature(args, stage, command=None):
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir)
    source = Path(args.train_source_path or output)
    model = Path(args.train_model_path or args.model_path)
    backend = None
    prefixes = {"center": ("center_", "fbp_"), "fbp": ("fbp_",),
                "init": ("init_",), "train": ("train_",), "test": ("test_",)}[stage]
    parameters = {key: value for key, value in vars(args).items()
                  if key.startswith(prefixes)}
    parameters.update(seed=args.seed, output=str(output), model=str(model))
    paths = []
    selected_test_iteration = None
    selected_test_pose = None
    if stage in {"center", "fbp"}:
        from scripts.fbp_preprocess import select_tiff_paths
        raw = Path(args.input_dir)
        prefix = 'center' if stage == 'center' else 'fbp'
        flat = getattr(args, prefix + '_flat_field')
        pattern = getattr(args, prefix + '_projection_glob') or ('Tomo_*.tiff' if flat else None)
        paths += select_tiff_paths(raw, pattern)
        if flat:
            paths += select_tiff_paths(raw, getattr(args, prefix + '_flat_glob') or 'Flat_*.tiff')
        # Hash acquisition inputs, not generated outputs nested below input_dir.
        paths += [args.fbp_config]
        candidates = sorted(raw.glob('*.txt'))
        if len(candidates) == 1:
            paths += candidates
        if stage == "center":
            paths += [args.center_config, args.center_processed_npy, args.center_angles_npy,
                      args.center_endpoint_pair_npy]
        else:
            center = args.center_json or args.center_output
            if center and Path(center).exists():
                paths.append(center)
    elif stage == "init":
        paths.append(init_metadata_path(args))
        if args.init_source == "custom":
            paths.append(args.init_volume)
        elif args.init_source == "full_fbp":
            paths.append(output / "vol_fbp.npy")
        else:
            paths += dataset_files(output)
    elif stage == "train":
        paths += dataset_files(source)
        paths.append(args.train_start_checkpoint)
        if args.train_reference_enabled:
            paths.append(args.train_reference_path)
        if not args.train_start_checkpoint:
            paths.append(args.train_ply_path or args.init_output)
    else:
        source = Path(args.test_source_path or source)
        model = Path(args.test_model_path or model)
        from r2_gaussian.utils.config_io import saved_config_path
        paths += dataset_files(source) + [model / "point_cloud", saved_config_path(model)]
        if (model / "pose_calibration").exists():
            paths.append(model / "pose_calibration")
        from r2_gaussian.pose.selection import resolve_test_selection
        selected_test_iteration, selected_test_pose = resolve_test_selection(
            model, getattr(args, "test_iteration", -1), getattr(args, "test_pose_result", None)
        )
        paths.append(selected_test_pose)
    implementations = {"center": ["scripts/scan_fbp_center.py", "scripts/scan_fbp_center_tomopy.py"],
                       "fbp": ["prepare_fbp_tiff.py", "scripts/fbp_preprocess.py"],
                       "init": ["init_from_fbp.py", "scripts/init_denoise.py"],
                       "train": ["train.py", "r2_gaussian/gaussian/gaussian_model.py"],
                       "test": ["test.py"]}[stage]
    paths += [root / name for name in implementations]
    # All stages depend on project utilities.
    paths += [p for folder in (root / "r2_gaussian", root / "scripts")
              for p in folder.rglob("*.py") if "submodules" not in p.parts]
    if stage in {"train", "test"}:
        from r2_gaussian.gaussian.render_query import raster_backend_provenance
        pose_enabled = bool(args.train_pose_enabled)
        if stage == "test":
            pose_enabled = selected_test_pose is not None
            parameters["resolved_test_iteration"] = selected_test_iteration
            parameters["resolved_test_pose_result"] = (
                str(selected_test_pose) if selected_test_pose is not None else None
            )
        backend = raster_backend_provenance(pose_enabled)
        # The runtime loader enforces this checkout for both package and _C.
        # Hash source interfaces as well as built binaries; omit build copies.
        native = root / "r2_gaussian" / "submodules"
        for backend_dir in ("simple-knn", "xray-gaussian-rasterization-voxelization",
                            "self-xray-gaussian-rasterization-voxelization"):
            paths += [p for p in (native / backend_dir).rglob("*")
                      if p.is_file() and not {"build", "__pycache__"}.intersection(p.parts)
                      and p.suffix.lower() in {".py", ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh",
                                               ".h", ".hpp", ".inl", ".pyd", ".so", ".dll"}]
        paths.append(backend["build_record_path"])
    return identity({"version": 3, "parameters": parameters, "command": command, "backend": backend,
                     "files": [file_identity(path) for path in _files(paths)]})


def dataset_files(source):
    if source.is_file():
        return [source]
    metadata = source / "meta_data.json"
    data = json.loads(metadata.read_text(encoding="utf-8"))
    files = [metadata]
    files += [source / str(record["file_path"]).replace("\\", "/")
              for key in ("proj_train", "proj_test") for record in data.get(key, [])]
    if data.get("vol"):
        files.append(source / data["vol"])
    return files


def capture(args, stage, signature_value):
    output = Path(args.output_dir)
    model = Path(args.train_model_path or args.model_path)
    if stage == "center":
        paths = [args.center_output]
    elif stage == "fbp":
        names = ["proj_preprocessed_all.npy", "angles_unique.npy", "preprocess_meta.json"] if args.fbp_preprocess_only else [
            "meta_data.json", "vol_fbp.npy", "vol_fbp_raw.npy", "proj_all.npy", "proj_train", "proj_test"]
        paths = [output / name for name in names]
        if (output / 'proj_endpoint_pair.npy').exists():
            paths.append(output / 'proj_endpoint_pair.npy')
        if not args.fbp_preprocess_only:
            data = json.loads((output / 'meta_data.json').read_text(encoding='utf-8'))
            if data.get('source', {}).get('sparse_fbp'):
                paths.append(data['source']['sparse_fbp'])
    elif stage == "init":
        paths = [args.init_output]
        if getattr(args, "init_denoise", False):
            paths.append(args.init_denoise_output_dir)
    elif stage == "train":
        from r2_gaussian.utils.config_io import saved_config_path
        paths = [model / "point_cloud", saved_config_path(model)]
        if args.train_pose_enabled:
            paths.append(model / "pose_calibration")
        paths += list((model / "ckpt").glob("chkpnt*.pth"))
    else:
        paths = [Path(args.test_model_path or model) / "test"]
    files = [p for p in _files(paths) if p.name != "pipeline_manifest.json"]
    if not files:
        raise ValueError(f"{stage} returned without required artifacts")
    return {"signature": signature_value, "artifacts": [file_identity(p) for p in files]}


def validate(record, expected, stage):
    message = f"Cannot resume {stage}: inputs/configuration or artifacts changed; use a new output/model directory."
    if not record or record.get("signature") != expected or not record.get("artifacts"):
        raise ValueError(message)
    for artifact in record["artifacts"]:
        try:
            matches = file_identity(artifact["path"]) == artifact
        except OSError:
            matches = False
        if not matches:
            raise ValueError(message + f" Invalid artifact: {artifact['path']}")
