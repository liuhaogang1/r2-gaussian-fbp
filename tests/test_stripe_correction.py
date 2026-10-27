"""CPU tests for the explicit TomoPy stripe-correction path."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

from scripts import stripe_correction as stripe


def args_for(input_type="line_integral"):
    return SimpleNamespace(
        stripe_correction={"snr": 3.0, "la_size": 61, "sm_size": 21, "dim": 1},
        stripe_ncore=1, stripe_row_chunk=2, clip_percentile=99.9,
        flat_field=False, dark_value=0.0, flat_eps=1e-6,
        input_type=input_type, zero_policy="nearest", log_eps=1e-6,
        i0_mode="fixed", free_ray=[0, 2, 0, 2], i0_hi_percentile=99.9,
        i0=1.0, i0_percentile=99.5, shift_v=0, pixel_subsample=1,
        resize_order=1, projection_scale=1.0,
    )


class StripeCorrectionTests(unittest.TestCase):
    def test_options_are_strict_and_defaulted(self):
        self.assertEqual(stripe.normalize_options({}), stripe.DEFAULTS)
        with self.assertRaises(ValueError):
            stripe.normalize_options({"la_size": 60})
        with self.assertRaises(ValueError):
            stripe.normalize_options({"unknown": 1})

    def test_native_correction_saves_same_window_before_and_after(self):
        fake = ModuleType("tomopy")
        fake.__version__ = "test"
        fake.remove_all_stripe = lambda block, **kwargs: block + 0.25
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(sys.modules, {"tomopy": fake}):
            output, native, info = stripe.prepare_projections(
                np.ones((4, 6, 8), dtype=np.float32), args_for(),
                angles=np.linspace(0.0, 1.0, 4), preview_dir=Path(temporary),
            )
            self.assertEqual(output.shape, (4, 6, 8))
            self.assertEqual(native.shape, (4, 6, 8))
            self.assertEqual(info["parameters"], stripe.DEFAULTS)
            self.assertTrue((Path(temporary) / "stripe_sinogram_before.png").exists())
            self.assertTrue((Path(temporary) / "stripe_sinogram_after.png").exists())

    def test_metadata_provenance_is_required_for_processed_npy(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            processed = directory / "proj_preprocessed_all.npy"
            np.save(processed, np.zeros((2, 3, 4), dtype=np.float32))
            with self.assertRaises(ValueError):
                stripe.processed_metadata(processed, stripe.DEFAULTS)
            (directory / "preprocess_meta.json").write_text(json.dumps({
                "stripe_correction": {
                    "algorithm": "tomopy.remove_all_stripe",
                    "parameters": stripe.DEFAULTS,
                }
            }), encoding="utf-8")
            self.assertEqual(
                stripe.processed_metadata(processed, stripe.DEFAULTS)["parameters"],
                stripe.DEFAULTS,
            )


if __name__ == "__main__":
    unittest.main()
