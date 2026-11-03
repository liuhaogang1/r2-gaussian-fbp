"""CPU numerical regressions; load production expressions without CUDA imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class Array(np.ndarray):
    """Small torch facade for the split scale calculation."""

    def repeat(self, *sizes):
        return np.tile(np.asarray(self), sizes).view(Array)

    def clamp(self, lower, upper):
        return np.clip(self, lower, upper).view(Array)


class GaussianNumericsTests(unittest.TestCase):
    def split_scales(self, scales, bounds):
        path = ROOT / "r2_gaussian/gaussian/gaussian_model.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GaussianModel")
        split = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "densify_and_split")
        start = next(i for i, n in enumerate(split.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "child_scales" for t in n.targets))
        end = next(i for i, n in enumerate(split.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "new_rotation" for t in n.targets))
        def inverse(x):
            if bounds is None:
                return np.log(x)
            p = np.maximum((x - bounds[0]) / (bounds[1] - bounds[0]), 0)
            return np.log(p / (1 - p))
        namespace = dict(
            self=SimpleNamespace(get_scaling=np.asarray(scales, dtype=np.float32).view(Array),
                                 scale_bound=bounds, scaling_inverse_activation=inverse),
            selected_pts_mask=np.ones(len(scales), dtype=bool), N=2,
            torch=SimpleNamespace(finfo=np.finfo),
        )
        exec(compile(ast.Module(body=split.body[start:end], type_ignores=[]), str(path), "exec"), namespace)
        return namespace["new_scaling"], namespace["child_scales"]

    def test_anisotropic_split_at_and_below_minimum_stays_finite(self):
        # Only the longest axis controls selection; shorter axes shrink below min.
        with np.errstate(all="raise"):
            raw, scales = self.split_scales([[0.0005, 0.0006, 0.2]], [0.0005, 0.5])
        self.assertTrue(np.isfinite(raw).all())
        self.assertTrue((scales > 0.0005).all())
        self.assertTrue((scales < 0.5).all())
        np.testing.assert_allclose(scales[:, 2], 0.2 / 1.6)

    def test_ordinary_and_unbounded_split_keep_expected_scales(self):
        for bounds in ([0.0005, 0.5], None):
            raw, scales = self.split_scales([[0.02, 0.04, 0.2]], bounds)
            self.assertTrue(np.isfinite(raw).all())
            np.testing.assert_allclose(scales, np.tile([[0.02, 0.04, 0.2]], (2, 1)) / 1.6)

    def test_inverse_softplus_roundtrip_small_and_large_densities(self):
        path = ROOT / "r2_gaussian/utils/gaussian_utils.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "inverse_softplus")
        namespace = {"torch": SimpleNamespace(log=np.log, expm1=np.expm1)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        values = np.array([1e-12, 1e-6, 0.05, 1., 100., 1000.], dtype=np.float32)
        for beta in (0.5, 1., 2.):
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                raw = namespace["inverse_softplus"](values, beta)
            self.assertTrue(np.isfinite(raw).all())
            np.testing.assert_allclose(np.logaddexp(0, beta * raw) / beta, values, rtol=3e-6)


if __name__ == "__main__":
    unittest.main()
