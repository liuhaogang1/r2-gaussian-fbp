"""Validate completed stages before allowing pipeline resume."""
from pathlib import Path
import json

from r2_gaussian.utils.artifact_utils import file_identity, identity


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
    prefixes = {"center": ("center_", "fbp_"), "fbp": ("fbp_",),
                "init": ("init_",), "train": ("train_",), "test": ("test_",)}[stage]
    parameters = {key: value for key, value in vars(args).items()
                  if key.startswith(prefixes)}
    parameters.update(seed=args.seed, output=str(output), model=str(model))
    paths = []
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
        if args.init_source == "custom":
            paths.append(args.init_volume)
            if (output / "meta_data.json").exists():
                paths.append(output / "meta_data.json")
        else:
            paths.append(output / "meta_data.json")
            if args.init_source == "full_fbp":
                paths.append(output / "vol_fbp.npy")
            else:
                paths += dataset_files(output)
    elif stage == "train":
        paths += dataset_files(source)
        paths += [args.train_ply_path or args.init_output,
                  args.train_slice_reference_path, args.train_start_checkpoint]
    else:
        source = Path(args.test_source_path or source)
        model = Path(args.test_model_path or model)
        paths += dataset_files(source) + [model / "point_cloud", model / "cfg_args"]
        if (model / "pose_calibration").exists():
            paths.append(model / "pose_calibration")
        paths.append(args.test_pose_result)
    implementations = {"center": ["scripts/scan_fbp_center.py", "scripts/scan_fbp_center_tomopy.py"],
                       "fbp": ["prepare_fbp_tiff.py", "scripts/fbp_preprocess.py"],
                       "init": ["init_from_fbp.py", "scripts/init_denoise.py"],
                       "train": ["train.py", "r2_gaussian/gaussian/gaussian_model.py"],
                       "test": ["test.py"]}[stage]
    paths += [root / name for name in implementations]
    # All stages depend on project utilities; include extension Python sources
    # without hashing large inputs or generated artifacts a second time.
    paths += [p for folder in (root / "r2_gaussian", root / "scripts")
              for p in folder.rglob("*.py") if "submodules" not in p.parts]
    return identity({"version": 1, "parameters": parameters, "command": command,
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
        paths = [model / "point_cloud", model / "cfg_args"]
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
