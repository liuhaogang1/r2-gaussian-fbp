"""Small real CUDA training/evaluation runs; writes only a new output folder."""
from __future__ import annotations
import argparse
import __future__
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def load_evaluation():
    # Test harness only: preserve production Python compatibility choices.
    module = ModuleType('evaluation_smoke')
    module.__file__ = str(ROOT / 'test.py')
    code = compile((ROOT / 'test.py').read_text(), module.__file__, 'exec',
                   flags=__future__.annotations.compiler_flag)
    exec(code, module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['parallel', 'tv', 'densify', 'sparse', 'shape_repair', 'shape_split', 'shape_prune', 'resume'])
    parser.add_argument('--init', type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import train
    from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
    from r2_gaussian.gaussian import GaussianModel
    from r2_gaussian.gaussian.render_query import _load_raster_backend
    from r2_gaussian.utils.general_utils import safe_state
    source = args.source.resolve()
    init = args.init.resolve() if args.init else source / 'init_denoise/init_points.npy'
    evaluation = load_evaluation()
    results = []
    try:
        _load_raster_backend(True)
        pose_backend = {'available': True}
    except Exception as error:
        pose_backend = {'available': False, 'reason': str(error)}
    for name in args.cases:
        safe_state(False)
        cli = argparse.ArgumentParser()
        lp, op, pp = ModelParams(cli), OptimizationParams(cli), PipelineParams(cli)
        model_path = out / name
        opts = cli.parse_args(['-s', str(source), '-m', str(model_path),
                              '--ply_path', str(init), '--iterations', '6',
                              '--lambda_tv', '0', '--lambda_dssim', '0.05'])
        dataset, opt, pipe = lp.extract(opts), op.extract(opts), pp.extract(opts)
        opt.densify_until_iter = 0
        if name == 'tv':
            opt.lambda_tv, opt.tv_vol_size = .01, 8
        if name == 'densify':
            opt.densify_from_iter, opt.densify_until_iter = 1, 5
            opt.densification_interval, opt.densify_grad_threshold = 2, 0.
            opt.max_num_gaussians = 1200
        if name == 'sparse':
            opt.slice_loss_enabled = True
            opt.slice_reference_path = str(source / 'vol_fbp.npy')
            opt.slice_start_iter, opt.slice_end_iter, opt.slice_interval = 1, 6, 1
            opt.slice_max_points, opt.slice_filter_ratio = 60, 10.
            opt.slice_filter_threshold = None
        if name.startswith('shape_'):
            opt.shape_enabled = opt.shape_needle_enabled = opt.shape_sheet_enabled = True
            opt.shape_start_iter, opt.shape_end_iter, opt.shape_check_interval = 1, 6, 1
            opt.shape_patience, opt.shape_ramp_iters = 1, 1
            opt.shape_needle_threshold = opt.shape_sheet_threshold = 2.
            opt.shape_hard_needle_ratio = opt.shape_hard_sheet_ratio = 3.
            opt.shape_hard_action = {'shape_repair':'repair','shape_split':'split','shape_prune':'prune_low'}[name]
            opt.shape_low_density_threshold = 100. if name == 'shape_prune' else 0.
            opt.lambda_shape_needle = opt.lambda_shape_sheet = .001
        counts = {'adam_steps':0, 'densify_calls':0, 'sparse_queries':0, 'shape_repaired':0, 'shape_split':0, 'shape_pruned':0}
        original_step, original_densify = torch.optim.Adam.step, GaussianModel.densify_and_prune
        original_sparse, original_hard = train.query_points, train._shape_hard_process
        original_init = train.initialize_gaussian
        def step(self, *a, **kw):
            counts['adam_steps'] += 1
            return original_step(self, *a, **kw)
        def densify(self, *a, **kw):
            counts['densify_calls'] += 1
            return original_densify(self, *a, **kw)
        def sparse(*a, **kw):
            counts['sparse_queries'] += 1
            return original_sparse(*a, **kw)
        def hard(*a, **kw):
            result = original_hard(*a, **kw)
            for key in ('shape_repaired','shape_split','shape_pruned'):
                counts[key] += result.get(key, 0)
            return result
        def initialize(model, *a, **kw):
            result = original_init(model, *a, **kw)
            if name.startswith('shape_'):
                with torch.no_grad():
                    scales = model.get_scaling.detach().clone()
                    scales[:3] = torch.tensor([.002, .01, .1], device='cuda')
                    model._scaling.copy_(model.scaling_inverse_activation(scales))
            return result
        with mock.patch.object(torch.optim.Adam, 'step', step), \
             mock.patch.object(GaussianModel, 'densify_and_prune', densify), \
             mock.patch.object(train, 'query_points', sparse), \
             mock.patch.object(train, '_shape_hard_process', hard), \
             mock.patch.object(train, 'initialize_gaussian', initialize):
            if name == 'resume':
                opt.iterations = 3
                train.training(dataset, opt, pipe, None, [], [3], [3], None)
                opt.iterations = 6
                train.training(dataset, opt, pipe, None, [], [6], [], str(model_path / 'ckpt/chkpnt3.pth'))
            else:
                train.training(dataset, opt, pipe, None, [6] if name == 'parallel' else [], [6], [], None)
        expected_steps = 4 if name == 'resume' else 5
        assert counts['adam_steps'] == expected_steps, (name, counts)
        if name == 'densify': assert counts['densify_calls'] > 0
        if name == 'sparse': assert counts['sparse_queries'] == 6
        if name.startswith('shape_'):
            key = {'shape_repair':'shape_repaired','shape_split':'shape_split','shape_prune':'shape_pruned'}[name]
            assert counts[key] > 0, counts
        volume = np.load(model_path / 'point_cloud/iteration_6/vol_pred.npy')
        assert np.isfinite(volume).all()
        with torch.no_grad():
            evaluation.testing(dataset, pipe, 6, False, False, False)
        test = model_path / 'test/iter_6'
        prediction = np.load(test / 'vol_pred.npy')
        assert np.isfinite(prediction).all()
        np.testing.assert_allclose(prediction, volume, rtol=1e-5, atol=1e-6)
        plane = np.load(test / 'reconstruction/00003_pred.npy')
        np.testing.assert_array_equal(plane, prediction[:, :, 3])
        results.append(dict(name=name, counts=counts, finite_volume=True, evaluation=True))
        (out / 'results.json').write_text(json.dumps(dict(runs=results, pose_backend=pose_backend), indent=2))
        print('PASS_RUN', name, counts, flush=True)
    print(json.dumps(dict(runs=results, pose_backend=pose_backend), indent=2))


if __name__ == '__main__':
    main()
