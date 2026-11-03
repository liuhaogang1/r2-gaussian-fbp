"""Small CUDA smoke tests for anomalous Gaussian-shape operations."""
from types import SimpleNamespace
import unittest

import torch

from r2_gaussian.gaussian.gaussian_model import GaussianModel
from train import _shape_regularization, _shape_hard_process


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Gaussian smoke tests")
class ShapeSuppressionCudaTests(unittest.TestCase):
    def make_model(self):
        model = GaussianModel()
        device = torch.device("cuda")
        scales = torch.tensor([[1.0, 2.0, 30.0], [1.0, 2.0, 4.0]], device=device)
        model._xyz = torch.nn.Parameter(torch.zeros((2, 3), device=device))
        model._scaling = torch.nn.Parameter(torch.log(scales))
        model._rotation = torch.nn.Parameter(torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device
        ))
        model._density = torch.nn.Parameter(model.density_inverse_activation(
            torch.ones((2, 1), device=device)
        ))
        model.max_radii2D = torch.zeros((2,), device=device)
        model.spatial_lr_scale = 1.0
        model.training_setup(SimpleNamespace(
            position_lr_init=1e-3, position_lr_final=1e-4, position_lr_max_steps=10,
            density_lr_init=1e-3, density_lr_final=1e-4, density_lr_max_steps=10,
            scaling_lr_init=1e-3, scaling_lr_final=1e-4, scaling_lr_max_steps=10,
            rotation_lr_init=1e-3, rotation_lr_final=1e-4, rotation_lr_max_steps=10,
        ))
        return model

    def test_shape_penalty_has_gradient_only_for_over_threshold_gaussian(self):
        model = self.make_model()
        opt = SimpleNamespace(
            shape_enabled=True, shape_start_iter=1, shape_end_iter=5,
            shape_ramp_iters=1, shape_needle_threshold=10.0,
            shape_sheet_threshold=10.0, shape_needle_enabled=True,
            shape_sheet_enabled=False, lambda_shape_needle=1.0,
            lambda_shape_sheet=0.0,
        )
        loss, diagnostics = _shape_regularization(model, opt, 1)
        self.assertEqual(diagnostics["shape_needle_count"], 1)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        grad = model._scaling.grad
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(grad[0].abs().sum().item(), 0.0)
        self.assertEqual(grad[1].abs().sum().item(), 0.0)

    def test_repair_and_split_keep_optimizer_tensors_aligned(self):
        model = self.make_model()
        # Populate Adam state as the real training loop does before hard actions.
        sum(parameter.square().sum() for parameter in (
            model._xyz, model._density, model._scaling, model._rotation
        )).backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        model.enable_shape_tracking()
        mask = torch.tensor([True, False], device="cuda")
        untouched_scaling_momentum = model.optimizer.state[model._scaling]["exp_avg"][1].clone()
        repaired = model.shape_repair(mask, 10.0, 10.0, 4.0, True, False)
        self.assertEqual(repaired, 1)
        ratios = torch.sort(model.get_scaling.detach(), dim=1).values
        self.assertLessEqual((ratios[0, 2] / ratios[0, 1]).item(), 10.001)
        self.assertTrue(torch.isfinite(model.get_density).all().item())
        self.assertEqual(model.shape_bad_count[0].item(), 0)
        torch.testing.assert_close(
            model.optimizer.state[model._scaling]["exp_avg"][1],
            untouched_scaling_momentum,
        )
        for group in model.optimizer.param_groups:
            parameter = group["params"][0]
            state = model.optimizer.state[parameter]
            self.assertEqual(state["exp_avg"].shape, parameter.shape, group["name"])
            self.assertEqual(state["exp_avg_sq"].shape, parameter.shape, group["name"])

        split_count = model.shape_split(mask, children=2)
        self.assertEqual(split_count, 1)
        self.assertEqual(model.get_xyz.shape, (3, 3))
        self.assertEqual(model.get_scaling.shape, (3, 3))
        self.assertEqual(model.shape_bad_count.shape, (3,))
        for group in model.optimizer.param_groups:
            parameter = group["params"][0]
            state = model.optimizer.state[parameter]
            self.assertEqual(state["exp_avg"].shape, parameter.shape)
            self.assertEqual(state["exp_avg_sq"].shape, parameter.shape)
            self.assertTrue(torch.isfinite(parameter).all().item())

    def test_simultaneous_sheet_needle_repair_with_large_density(self):
        model = self.make_model()
        with torch.no_grad():
            model._scaling[0] = torch.log(torch.tensor([1., 30., 900.], device='cuda'))
            model._density[0] = 1000.
        model.enable_shape_tracking()
        repaired = model.shape_repair(torch.tensor([True, False], device='cuda'), 10., 10.)
        self.assertEqual(repaired, 1)
        scales = torch.sort(model.get_scaling[0]).values
        self.assertLessEqual(float(scales[1]/scales[0]), 10.001)
        self.assertLessEqual(float(scales[2]/scales[1]), 10.001)
        self.assertTrue(torch.isfinite(model._density).all())

    def test_shape_pruning_retains_one_gaussian_when_all_candidates_are_low(self):
        model = self.make_model()
        opt = SimpleNamespace(shape_enabled=True, shape_check_interval=1, shape_start_iter=1,
            shape_end_iter=5, shape_hard_needle_ratio=1.1, shape_hard_sheet_ratio=1.1,
            shape_patience=1, shape_hard_action='prune_low', shape_low_density_threshold=100.)
        result = _shape_hard_process(model, opt, 1)
        self.assertEqual(result['shape_pruned'], 1)
        self.assertEqual(len(model.get_xyz), 1)
        self.assertTrue(torch.isfinite(model.get_density).all())

    def test_bounded_split_rejects_unrepresentable_children(self):
        model = self.make_model()
        model.scale_bound = [1., 40.]
        model.setup_functions()
        with torch.no_grad():
            model._scaling.copy_(model.scaling_inverse_activation(
                torch.tensor([[1.1, 1.2, 1.3], [2., 3., 4.]], device='cuda')))
        before = model._xyz.clone()
        self.assertEqual(model.shape_split(torch.tensor([True, False], device='cuda'), 2), 0)
        torch.testing.assert_close(model._xyz, before)
        self.assertTrue(torch.isfinite(model._scaling).all())

    def test_split_multiple_parents_has_correct_child_positions(self):
        model = self.make_model()
        original = model.get_xyz.detach().clone()
        scales = model.get_scaling.detach().clone()
        self.assertEqual(model.shape_split(torch.ones(2, dtype=torch.bool, device='cuda'), 3), 2)
        self.assertEqual(len(model.get_xyz), 6)
        for parent in range(2):
            children = model.get_xyz[parent*3:(parent+1)*3]
            torch.testing.assert_close(children.mean(0), original[parent])
            torch.testing.assert_close(children[:, 2] - original[parent, 2],
                                      torch.tensor([-.35, 0., .35], device='cuda') * scales[parent, 2])
        self.assertTrue(torch.isfinite(model._scaling).all())


if __name__ == "__main__":
    unittest.main()
