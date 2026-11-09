"""Check CUDA derivatives against their own forward finite differences.

Run after building both rasterization/voxelization extensions in this checkout:
python -m unittest discover -s tests -p test_voxelizer_gradients_cuda.py -v
"""
import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


class _VoxelizerGradientChecks:
    pose_enabled = False

    @classmethod
    def setUpClass(cls):
        if torch is None or not torch.cuda.is_available():
            raise unittest.SkipTest("PyTorch with CUDA is required")
        from r2_gaussian.gaussian import GaussianModel, query
        from r2_gaussian.gaussian.render_query import _load_raster_backend
        from r2_gaussian.utils.loss_utils import tv_3d_loss

        try:
            _load_raster_backend(cls.pose_enabled)
        except (ImportError, RuntimeError) as error:
            raise unittest.SkipTest(str(error)) from error
        cls.model_class = GaussianModel
        cls.query = staticmethod(query)
        cls.tv_loss = staticmethod(tv_3d_loss)

    def make_model(self):
        model = self.model_class()
        device = "cuda"
        model._xyz = torch.nn.Parameter(torch.tensor([[.08, -.04, .06]], device=device))
        model._scaling = torch.nn.Parameter(torch.log(torch.tensor([[.11, .09, .12]], device=device)))
        model._rotation = torch.nn.Parameter(torch.tensor([[.97, .07, -.12, .14]], device=device))
        model._density = torch.nn.Parameter(model.density_inverse_activation(
            torch.tensor([[1.3]], device=device)))
        return model

    def volume(self, model, counts, spacing, precomputed=False):
        pipe = SimpleNamespace(pose_enabled=self.pose_enabled,
                               compute_cov3D_python=precomputed, debug=False)
        return self.query(model, [.03, -.02, .01], counts,
                          np.asarray(counts) * spacing, pipe)["vol"]

    def loss(self, model, counts, spacing, precomputed=False, tv=False):
        vol = self.volume(model, counts, spacing, precomputed)
        if tv:
            return self.tv_loss(vol, reduction="mean")
        axes = [torch.linspace(-.3, .7, count, device="cuda") for count in counts]
        x, y, z = torch.meshgrid(*axes, indexing="ij")
        weights = 1 + .2 * x - .1 * y + .3 * z
        return (vol * weights).mean()

    def check_parameter(self, model, name, counts, spacing, precomputed=False, tv=False):
        for item in [model._xyz, model._scaling, model._rotation, model._density]:
            item.grad = None
        loss_fn = lambda: self.loss(model, counts, spacing, precomputed, tv)
        loss_fn().backward()
        param = getattr(model, name)
        actual = param.grad.detach().cpu().numpy().copy().reshape(-1)
        original = param.detach().clone()
        expected = []
        epsilon = .001 if not tv else .0003
        try:
            with torch.no_grad():
                for index in range(param.numel()):
                    probes = []
                    for sign in [-1, 1]:
                        param.copy_(original)
                        param.reshape(-1)[index] += sign * epsilon
                        probes.append(float(loss_fn()))
                    expected.append((probes[1] - probes[0]) / (2 * epsilon))
        finally:
            with torch.no_grad():
                param.copy_(original)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=.025, atol=2e-4,
                                   err_msg=f"{name}, counts={counts}, spacing={spacing}, "
                                           f"pose={self.pose_enabled}, covariance_python={precomputed}, tv={tv}")

    def test_position_gradients_in_all_slice_axes_and_blocks(self):
        for counts in [(1, 11, 7), (9, 1, 7), (9, 11, 1), (9, 11, 7)]:
            for spacing in [np.array([.04, .03, .05]), np.array([.015, .02, .025])]:
                for precomputed in [False, True]:
                    with self.subTest(counts=counts, spacing=spacing.tolist(), precomputed=precomputed):
                        self.check_parameter(self.make_model(), "_xyz", counts, spacing, precomputed)

    def test_density_scale_and_rotation_derivatives(self):
        for precomputed in [False, True]:
            for name in ["_density", "_scaling", "_rotation"]:
                with self.subTest(parameter=name, precomputed=precomputed):
                    self.check_parameter(self.make_model(), name, (9, 11, 7),
                                         np.array([.04, .03, .05]), precomputed)

    def test_tv_position_derivative(self):
        self.check_parameter(self.make_model(), "_xyz", (9, 11, 7),
                             np.array([.04, .03, .05]), tv=True)

    def test_extension_is_loaded_from_this_checkout(self):
        name = ("self_xray_gaussian_rasterization_voxelization" if self.pose_enabled
                else "xray_gaussian_rasterization_voxelization")
        module = importlib.import_module(name + "._C")
        expected = Path(__file__).resolve().parents[1] / "r2_gaussian" / "submodules"
        self.assertIn(expected.resolve(), Path(module.__file__).resolve().parents,
                      f"Stale extension from another checkout: {module.__file__}")


class DefaultVoxelizerGradientTests(_VoxelizerGradientChecks, unittest.TestCase):
    pose_enabled = False


class PoseVoxelizerGradientTests(_VoxelizerGradientChecks, unittest.TestCase):
    pose_enabled = True


if __name__ == "__main__":
    unittest.main()
