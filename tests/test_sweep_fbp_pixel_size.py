"""CPU tests for the optional subprocess wrapper; no TIGRE required."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import sweep_fbp_pixel_size as sweep


class SweepWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fbp_sweep_wrapper_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input"
        self.input.mkdir()

    def parse(self, *extra):
        return sweep.build_parser().parse_args([
            "--pixel_sizes", "0.1", "0.01", "0.001",
            "--output_dir", str(self.root / "sweep"),
            *extra,
            "--", "--input_dir", str(self.input), "--flat_field",
            "--input_type", "line_integral", "--shift_v", "0",
            "--nVoxel", "4", "4", "4",
        ])

    def test_candidates_are_validated_and_deduplicated(self):
        args = self.parse("--pixel_sizes", "0.1", "0.1", "0.01")
        sweep.validate_args(args)
        self.assertEqual(args.pixel_sizes, [0.1, 0.01])
        for bad in ("0", "-1", "nan", "inf"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sweep.validate_args(self.parse("--pixel_sizes", bad))
        with self.assertRaises(ValueError):
            sweep.validate_args(self.parse("--slice_fractions", "-0.1"))

    def test_output_input_and_forbidden_overrides_are_protected(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            sweep.validate_args(self.parse("--output_dir", str(self.input)))
        with self.assertRaisesRegex(ValueError, "center correction"):
            args = self.parse()
            args.prepare_args.extend(["--center_json", "center.json"])
            sweep.validate_args(args)
        out = self.root / "sweep"
        out.mkdir()
        (out / "keep").touch()
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            sweep.validate_args(self.parse())
        self.assertTrue((out / "keep").is_file())

    def test_build_command_runs_original_prepare_with_per_case_values(self):
        args = self.parse()
        sweep.validate_args(args)
        case = args.output_dir / "001" / "fbp"
        cmd = sweep.build_command(0.01, case, args.prepare_args)
        self.assertEqual(Path(cmd[1]), sweep.PREPARE_SCRIPT)
        self.assertIn("--flat_field", cmd)
        self.assertIn("--input_type", cmd)
        self.assertEqual(cmd[cmd.index("--pixel_size") + 1], "0.01")
        self.assertEqual(Path(cmd[cmd.index("--output_dir") + 1]), case)
        self.assertNotIn("scan_fbp_center_tomopy.py", " ".join(cmd))

    def test_dry_run_prints_commands_and_writes_nothing(self):
        args = self.parse("--dry_run")
        with mock.patch.object(sweep.subprocess, "run") as run:
            self.assertEqual(sweep.main(args), 0)
        run.assert_not_called()
        self.assertFalse(args.output_dir.exists())

    def fake_prepare(self, command, **kwargs):
        output = Path(command[command.index("--output_dir") + 1])
        output.mkdir(parents=True)
        np.save(output / "vol_fbp.npy", np.arange(64, dtype=np.float32).reshape(4, 4, 4) + 1)
        return subprocess.CompletedProcess(command, self.return_codes.pop(0))

    def test_each_candidate_runs_prepare_independently_and_continues_failure(self):
        args = self.parse()
        with mock.patch.object(sweep.subprocess, "run", side_effect=self.fake_prepare) as run:
            self.return_codes = [0, 7, 0]
            self.assertEqual(sweep.main(args), 1)
        self.assertEqual(run.call_count, 3)
        commands = json.loads((args.output_dir / "commands.json").read_text())["commands"]
        self.assertEqual(len(commands), 3)
        self.assertTrue(all(Path(item["command"][1]) == sweep.PREPARE_SCRIPT for item in commands))
        self.assertTrue(all("--flat_field" in item["command"] for item in commands))
        summary = (args.output_dir / "summary.csv").read_text(encoding="utf-8-sig")
        self.assertIn("success", summary)
        self.assertIn("failed", summary)
        self.assertEqual(len(list(args.output_dir.glob("comparison_*.png"))), 6)
        self.assertTrue((args.output_dir / "001_ps_0.1" / "prepare.log").is_file())

    def test_fail_fast_skips_later_candidates(self):
        args = self.parse("--fail_fast")
        with mock.patch.object(sweep.subprocess, "run", side_effect=self.fake_prepare) as run:
            self.return_codes = [4]
            self.assertEqual(sweep.main(args), 1)
        self.assertEqual(run.call_count, 1)
        summary = (args.output_dir / "summary.csv").read_text(encoding="utf-8-sig")
        self.assertIn("skipped", summary)

    def test_cli_rejects_center_and_pixel_size_forwarding(self):
        command = [
            sys.executable, str(ROOT / "scripts" / "sweep_fbp_pixel_size.py"),
            "--pixel_sizes", "0.1", "--output_dir", str(self.root / "out"),
            "--", "--input_dir", str(self.input), "--center_json", "center.json",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("center correction is intentionally deferred", result.stderr)


if __name__ == "__main__":
    unittest.main()
