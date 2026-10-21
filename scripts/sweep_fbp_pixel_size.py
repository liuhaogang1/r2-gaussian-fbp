"""Optional batch runner for prepare_fbp_tiff.py; never called by the pipeline.

Every candidate starts the unchanged original script in its own output folder.
This wrapper does not preprocess projections, estimate a center, or reconstruct.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = ROOT / "prepare_fbp_tiff.py"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Optional pixel_size sweep. Runs the unchanged prepare_fbp_tiff.py "
            "once per candidate; does not scan/correct rotation center."
        ),
        epilog=(
            "Put prepare_fbp_tiff.py arguments after --. Its --pixel_size and "
            "--output_dir are controlled by this wrapper. Output must be new/empty."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--pixel_sizes", nargs="+", type=float, required=True)
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Sweep root; each candidate gets a separate subdirectory")
    parser.add_argument("--slice_fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only; do not create files or run FBP")
    parser.add_argument("prepare_args", nargs=argparse.REMAINDER, help="Original prepare_fbp_tiff.py options after --")
    return parser


def split_prepare_args(values):
    values = list(values or [])
    if values and values[0] == "--":
        values.pop(0)
    if not values:
        raise ValueError("Pass prepare_fbp_tiff.py options after -- (at least --input_dir)")
    forbidden = {"--pixel_size", "--output_dir", "--center_json", "--offDetector"}
    found = [item for item in values if item.split("=", 1)[0] in forbidden]
    if found:
        raise ValueError(
            "Do not pass " + ", ".join(found) +
            ": pixel_size/output are per-candidate; center correction is intentionally deferred"
        )
    if "--input_dir" not in values:
        raise ValueError("--input_dir is required among prepare_fbp_tiff.py options")
    input_index = values.index("--input_dir")
    if input_index + 1 >= len(values):
        raise ValueError("--input_dir needs a path")
    input_dir = Path(values[input_index + 1]).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
    values[input_index + 1] = str(input_dir)
    return values, input_dir


def validate_args(args):
    args.output_dir = args.output_dir.resolve()
    args.pixel_sizes = list(dict.fromkeys(args.pixel_sizes))
    if not args.pixel_sizes or any(not math.isfinite(x) or x <= 0 for x in args.pixel_sizes):
        raise ValueError("pixel_sizes must be finite positive numbers")
    if not args.slice_fractions or any(not math.isfinite(x) or x < 0 or x > 1 for x in args.slice_fractions):
        raise ValueError("slice_fractions must be in [0, 1]")
    args.slice_fractions = list(dict.fromkeys(args.slice_fractions))
    prepare_args, input_dir = split_prepare_args(args.prepare_args)
    if (args.output_dir == input_dir or args.output_dir in input_dir.parents
            or input_dir in args.output_dir.parents):
        raise ValueError("Sweep output must be separate from the input directory tree")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("Sweep output must be new or empty; existing results are never overwritten")
    args.prepare_args = prepare_args
    args.input_dir = input_dir


def build_command(pixel_size, case_output, prepare_args):
    return [
        sys.executable, str(PREPARE_SCRIPT), *prepare_args,
        "--pixel_size", repr(pixel_size),
        "--output_dir", str(case_output),
    ]


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def write_summary(output, rows):
    fields = ["case", "pixel_size", "status", "return_code", "elapsed_seconds", "output_dir", "error"]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def comparison_images(output, rows, fractions):
    """Make paginated XYZ contact sheets from original-script output volumes."""
    from PIL import Image, ImageDraw

    successful = [row for row in rows if row["status"] == "success"]
    if not successful:
        return
    stats = []
    for row in successful:
        volume = np.load(Path(row["output_dir"]) / "vol_fbp.npy", mmap_mode="r")
        stride = max(1, volume.size // 1_000_000)
        sample = np.asarray(volume.reshape(-1)[::stride])
        positive = sample[sample > 0]
        if positive.size:
            low, high = np.percentile(positive, [1.0, 99.5])
            stats.append((float(low), float(high)))
        del volume
    if not stats:
        return
    shared = (min(pair[0] for pair in stats), max(pair[1] for pair in stats))
    (output / "comparison_windows.json").write_text(
        json.dumps({"shared_low_high": shared, "slice_fractions": fractions,
                    "method": "min per-case positive p1, max per-case positive p99.5; auto uses each case p1/p99.5"},
                   indent=2), encoding="utf-8"
    )
    cell, label = 256, 52
    for page_start in range(0, len(successful), 8):
        page = successful[page_start:page_start + 8]
        for col_start in range(0, len(fractions), 6):
            columns = fractions[col_start:col_start + 6]
            for axis, axis_name in enumerate("xyz"):
                for window_name in ("shared", "auto"):
                    sheet = Image.new("RGB", (cell * len(columns), (cell + label) * len(page)), "white")
                    draw = ImageDraw.Draw(sheet)
                    for r, row in enumerate(page):
                        volume = np.load(Path(row["output_dir"]) / "vol_fbp.npy", mmap_mode="r")
                        stride = max(1, volume.size // 1_000_000)
                        sample = np.asarray(volume.reshape(-1)[::stride])
                        positive = sample[sample > 0]
                        low, high = shared if window_name == "shared" else tuple(np.percentile(positive, [1.0, 99.5]))
                        for c, fraction in enumerate(columns):
                            x, y = c * cell, r * (cell + label)
                            draw.text((x + 6, y + 4), f"pixel_size={row['pixel_size']:.8g}  {axis_name}={fraction:g} ({window_name})", fill="black")
                            index = min(int(fraction * volume.shape[axis]), volume.shape[axis] - 1)
                            plane = np.take(volume, index, axis=axis)
                            scale = high - low if high > low else 1.0
                            gray = np.round(np.clip((plane - low) / scale, 0, 1) * 255).astype(np.uint8)
                            tile = Image.fromarray(gray).convert("RGB")
                            tile.thumbnail((cell, cell))
                            sheet.paste(tile, (x + (cell - tile.width) // 2, y + label + (cell - tile.height) // 2))
                        del volume
                    suffix = f"p{page_start // 8 + 1:02d}_c{col_start // 6 + 1:02d}"
                    sheet.save(output / f"comparison_{axis_name}_{window_name}_{suffix}.png")


def main(args):
    validate_args(args)
    rows = []
    commands = []
    for index, pixel_size in enumerate(args.pixel_sizes, start=1):
        case = f"{index:03d}_ps_{pixel_size:.12g}"
        case_root = args.output_dir / case
        case_output = case_root / "fbp"
        command = build_command(pixel_size, case_output, args.prepare_args)
        commands.append({"case": case, "pixel_size": pixel_size, "command": command,
                         "output_dir": str(case_output)})
        rows.append({"case": case, "pixel_size": pixel_size, "status": "pending",
                     "return_code": "", "elapsed_seconds": "",
                     "output_dir": str(case_output), "error": ""})
    if args.dry_run:
        for item in commands:
            print(shlex.join(item["command"]))
        print("Dry run: commands only; no files were created and prepare_fbp_tiff.py was not run.")
        return 0

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "commands.json").write_text(
        json.dumps({"wrapper": str(Path(__file__).resolve()), "prepare_script": str(PREPARE_SCRIPT),
                    "input_dir": str(args.input_dir), "commands": commands},
                   indent=2, ensure_ascii=False, default=json_default), encoding="utf-8"
    )
    write_summary(output, rows)
    try:
        for row, item in zip(rows, commands):
            case_root = output / row["case"]
            case_root.mkdir()
            started = time.perf_counter()
            try:
                with (case_root / "prepare.log").open("w", encoding="utf-8") as log:
                    completed = subprocess.run(item["command"], cwd=ROOT, stdout=log,
                                               stderr=subprocess.STDOUT, check=False)
                row["return_code"] = completed.returncode
                row["status"] = "success" if completed.returncode == 0 else "failed"
                if completed.returncode != 0:
                    row["error"] = f"prepare_fbp_tiff.py exited {completed.returncode}; see prepare.log"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            write_summary(output, rows)
            print(f"{row['case']}: {row['status']} {row['error']}", flush=True)
            if row["status"] == "failed" and args.fail_fast:
                for pending in rows:
                    if pending["status"] == "pending":
                        pending.update(status="skipped", error="Stopped after failure (--fail_fast)")
                break
    except KeyboardInterrupt:
        for pending in rows:
            if pending["status"] == "pending":
                pending.update(status="interrupted", error="Stopped by user")
        raise
    finally:
        write_summary(output, rows)
        comparison_images(output, rows, args.slice_fractions)
    successes = sum(row["status"] == "success" for row in rows)
    print(f"Finished: {successes}/{len(rows)} successful. Results: {output}", flush=True)
    return 0 if successes == len(rows) else 1


if __name__ == "__main__":
    try:
        sys.exit(main(build_parser().parse_args()))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
