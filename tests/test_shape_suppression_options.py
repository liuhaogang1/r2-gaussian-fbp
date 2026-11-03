"""CLI/config wiring tests for opt-in Gaussian shape suppression."""
from pathlib import Path
import tempfile
import unittest

import run_pipeline as pipeline


class ShapeSuppressionOptionTests(unittest.TestCase):
    def args(self, *extra):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        return pipeline.resolve_args(pipeline.build_parser().parse_args([
            "--input_dir", str(root / "input"),
            "--output_dir", str(root / "output"),
            "--model_path", str(root / "model"),
            *extra,
        ]))

    def test_shape_options_are_absent_from_legacy_resolved_arguments(self):
        args = self.args()
        self.assertNotIn("train_shape_enabled", vars(args))

    def test_shape_options_resolve_and_reach_train_command(self):
        args = self.args(
            "--train_shape_enabled",
            "--train_shape_hard_action", "repair",
            "--train_shape_needle_threshold", "12",
            "--train_lambda_shape_needle", "0.02",
        )
        self.assertTrue(args.train_shape_enabled)
        command = pipeline.build_train_command(
            args, Path("init.npy"), Path("train.yml")
        )
        self.assertIn("--shape_enabled", command)
        self.assertIn("--shape_hard_action", command)
        self.assertIn("repair", command)
        self.assertIn("--lambda_shape_needle", command)
        self.assertIn("0.02", command)

    def test_invalid_hard_action_is_rejected(self):
        args = self.args(
            "--train_shape_enabled", "--train_shape_needle_enabled",
            "--train_shape_hard_action", "delete_all",
        )
        with self.assertRaisesRegex(ValueError, "shape_hard_action"):
            pipeline.validate_args(args)


if __name__ == "__main__":
    unittest.main()
