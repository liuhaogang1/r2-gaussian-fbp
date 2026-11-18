"""Forward-project a saved XYZ volume using nominal parallel-beam dataset geometry."""

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2_gaussian.utils.geometry_utils import validate_physical_dataset


def sampling_angles(step_text):
    try:
        step = Decimal(step_text)
    except InvalidOperation as exc:
        raise ValueError("--step-deg must be a positive finite number") from exc
    if not step.is_finite() or step <= 0:
        raise ValueError("--step-deg must be a positive finite number")
    if step > 180:
        return np.zeros(1, dtype=np.float64)
    if step < Decimal(180) / int(np.iinfo(np.intp).max // 8):
        raise ValueError("--step-deg produces too many frames for a NumPy array")
    numerator, denominator = step.as_integer_ratio()
    last_index = 180 * denominator // numerator
    angles = np.arange(last_index + 1, dtype=np.float64) * float(step)
    if last_index * numerator == 180 * denominator:
        angles[-1] = 180.0
    return angles


def read_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def load_scanner(volume_path, shape, meta_path):
    sidecar = volume_path.with_suffix(".json")
    export = read_json(sidecar) if sidecar.is_file() else {}
    highres = export.get("nifti_geometry")
    if meta_path is None:
        meta_path = volume_path.parent / "meta_data.json"
        if not meta_path.is_file() and highres and export.get("source_path"):
            meta_path = Path(export["source_path"]) / "meta_data.json"
    if not meta_path.is_file():
        raise ValueError(f"Metadata file missing at {meta_path}. Supply --meta PATH.")
    metadata = read_json(meta_path)
    validate_physical_dataset(metadata)
    scanner = metadata["scanner"].copy()
    if scanner.get("mode") != "parallel":
        raise ValueError(f"Expected parallel scanner geometry, got {scanner.get('mode')!r}")
    if highres is not None:
        if (export.get("volume") != volume_path.name
                or list(shape) != export.get("shape_xyz")
                or list(shape) != highres["shape_xyz"]):
            raise ValueError(f"High-resolution export {sidecar} does not match the volume")
        scanner.update(nVoxel=list(shape), sVoxel=highres["size_xyz"],
                       offOrigin=highres["center_xyz"])
    elif list(shape) != scanner["nVoxel"]:
        raise ValueError(f"Volume shape {shape} differs from scanner.nVoxel {scanner['nVoxel']}. "
                         "Supply matching geometry or the high-resolution export sidecar.")
    for key, length in (("sVoxel", 3), ("offOrigin", 3), ("nDetector", 2),
                        ("sDetector", 2), ("offDetector", 2)):
        values = np.asarray(scanner[key])
        if values.shape != (length,) or not np.isfinite(values).all():
            raise ValueError(f"scanner.{key} must have {length} finite values")
        if key in ("sVoxel", "sDetector", "nDetector") and np.any(values <= 0):
            raise ValueError(f"scanner.{key} must contain positive values")
        if key == "nDetector" and values.dtype.kind not in "iu":
            raise ValueError("scanner.nDetector must contain integer pixel counts")
    for key in ("DSO", "DSD", "accuracy"):
        value = scanner[key]
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"scanner.{key} must be positive and finite")
    scanner.setdefault("filter", None)
    return scanner, meta_path.resolve(), sidecar if highres is not None else None


def write_projections(folder, volume_zyx, scanner, angles, write_tiff):
    from tigre.utilities.Ax import Ax
    from r2_gaussian.utils.ct_utils import get_geometry_tigre

    geo = get_geometry_tigre(scanner)
    shape = (len(angles), *scanner["nDetector"])
    stack = np.lib.format.open_memmap(folder / "projections.npy", mode="w+",
                                      dtype=np.float32, shape=shape)
    try:
        for start in range(0, len(angles), 32):
            radians = np.deg2rad(angles[start:start + 32]).astype(np.float32)
            batch = Ax(volume_zyx, geo, radians)
            if batch.shape != (len(radians), *shape[1:]) or not np.isfinite(batch).all():
                raise ValueError(f"Invalid TIGRE projection output at frame {start}")
            stack[start:start + len(batch)] = batch[:, ::-1, :]
        stack.flush()
        if write_tiff:
            import tifffile
            tifffile.imwrite(folder / "projections.tif", stack, photometric="minisblack",
                             metadata={"axes": "QYX"})
    finally:
        del stack


def run(args):
    path = args.volume.resolve()
    angles = sampling_angles(args.step_deg)
    output = args.out or path.with_name(f"{path.stem}_reprojection_{args.step_deg}deg")
    if output.exists():
        raise FileExistsError(f"Output already exists at {output}. Choose a new --out.")
    if path.suffix.lower() != ".npy":
        raise ValueError("--volume must be a 3D XYZ .npy file")
    volume = np.load(path, mmap_mode="r", allow_pickle=False)
    if volume.ndim != 3 or not volume.size or volume.dtype.kind not in "iuf":
        raise ValueError("--volume must be a nonempty real numeric 3D XYZ array")
    scanner, meta_path, sidecar = load_scanner(path, volume.shape, args.meta)
    volume_zyx = np.array(volume.transpose(2, 1, 0), dtype=np.float32, order="C", copy=True)
    del volume
    if any(not np.isfinite(plane).all() for plane in volume_zyx):
        raise ValueError("Input volume contains non-finite float32 values")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        folder = Path(temporary) / "result"
        folder.mkdir()
        write_projections(folder, volume_zyx, scanner, angles, args.tiff)
        np.save(folder / "angles_deg.npy", angles)
        report = {
            "volume": str(path), "meta": str(meta_path),
            "volume_sidecar": str(sidecar) if sidecar else None,
            "scanner": scanner, "geometry": "nominal parallel beam in dataset units",
            "volume_axes": "XYZ", "projection_axes": ["angle", "V", "U"],
            "projection_values": "float32 line integrals",
            "shape": [len(angles), *scanner["nDetector"]],
            "step_deg": args.step_deg, "first_deg": float(angles[0]),
            "last_deg": float(angles[-1]), "frame_count": len(angles),
            "angles_file": "angles_deg.npy",
        }
        (folder / "projection_meta.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        folder.rename(output)
    print(f"Saved {len(angles)} projections ({angles[0]:g} to {angles[-1]:g} deg) to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", required=True, type=Path, help="Project XYZ volume (.npy)")
    parser.add_argument("--step-deg", default="0.5", help="Angle interval in degrees (default: 0.5)")
    parser.add_argument("--meta", type=Path, help="Dataset meta_data.json")
    parser.add_argument("--out", type=Path, help="New output directory")
    parser.add_argument("--tiff", action="store_true", help="Also save a float32 multipage TIFF")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        parser.exit(1, f"Forward projection failed: {exc}\n")


if __name__ == "__main__":
    main()
