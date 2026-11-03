"""Regressions for dataset, cache, resume and pose artifact integrity."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml

import init_from_fbp as init
import run_pipeline as pipeline
from r2_gaussian.pose.artifacts import (
    pose_metadata, validate_pose_file, validate_checkpoint_binding,
)
from r2_gaussian.utils.artifact_utils import atomic_npz, dataset_identity
from scripts import pipeline_artifacts as artifacts
from scripts.fbp_preprocess import select_tiff_paths, process_projection, build_angles
from fbp_test_support import import_prepare


class ArtifactIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'data'
        self.source.mkdir()
        for i in range(2):
            np.save(self.source / f'p{i}.npy', np.full((4, 6), i, np.float32))
        np.save(self.source / 'truth.npy', np.ones((5, 7, 9), np.float32))
        self.metadata = dict(scanner=dict(mode='parallel', nVoxel=[5, 7, 9],
                             sVoxel=[1, 2, 3], offOrigin=[0, 0, 0]),
                             vol='truth.npy', proj_train=[dict(file_path=f'p{i}.npy', angle=i*.3)
                                                        for i in range(2)], proj_test=[])
        self.write_metadata()
        self.scene = SimpleNamespace(source_path=str(self.source),
                     scanner_cfg=self.metadata['scanner'], train_cameras=[
                         SimpleNamespace(image_name=f'p{i}', angle=i*.3) for i in range(2)])

    def write_metadata(self):
        (self.source / 'meta_data.json').write_text(json.dumps(self.metadata))

    def args(self, extra=()):
        return pipeline.resolve_args(pipeline.build_parser().parse_args([
            '--input_dir', str(self.root / 'raw'), '--output_dir', str(self.source),
            '--model_path', str(self.root / 'model'), *extra]))

    def test_pose_file_reorders_views_and_rejects_wrong_iteration_and_content(self):
        path = self.root / 'pose.npz'
        atomic_npz(path, **pose_metadata(self.scene, 8), delta_angle=np.array([.1, .2]))
        self.scene.train_cameras.reverse()
        self.assertEqual(validate_pose_file(path, self.scene, 8), [1, 0])
        with self.assertRaises(ValueError):
            validate_pose_file(path, self.scene, 7)
        np.save(self.source / 'p0.npy', np.ones((4, 6), np.float32))
        with self.assertRaises(ValueError):
            validate_pose_file(path, self.scene, 8)

    def test_pose_checkpoint_requires_identical_order(self):
        binding = pose_metadata(self.scene, 8)
        validate_checkpoint_binding(binding, self.scene, 8)
        self.scene.train_cameras.reverse()
        with self.assertRaisesRegex(ValueError, 'view order'):
            validate_checkpoint_binding(binding, self.scene, 8)

    def test_real_pose_snapshot_loads_parallel_and_cone_with_unicode_binding(self):
        import torch
        from test_parallel_pose_geometry import production_save_pose_result
        from r2_gaussian.pose import ParallelPose, ConePose, StaticParallelPose, StaticConePose
        save = production_save_pose_result()
        self.scene.model_path = str(self.root / 'model')
        for mode in ('parallel', 'cone'):
            self.metadata['scanner']['mode'] = mode
            self.write_metadata()
            if mode == 'parallel':
                pose = ParallelPose(2, dict(optimize_angle=True, optimize_global_detector_shift=True), device='cpu')
                loader = StaticParallelPose
            else:
                pose = ConePose(2, {}, device='cpu')
                loader = StaticConePose
            save(self.scene, pose, 8)
            path = self.root / 'model/pose_calibration/iteration_8/pose_offsets.npz'
            self.assertEqual(validate_pose_file(path, self.scene, 8), [0, 1])
            restored = loader.from_npz(path, device='cpu')
            for key, value in pose.values().items():
                if value is not None:
                    torch.testing.assert_close(value, restored.values()[key])

    def test_pose_rejects_legacy_shapes_nan_and_duplicate_ids(self):
        path = self.root / 'pose.npz'
        for field in (dict(delta_angle=np.zeros((2, 1))), dict(global_shift=np.zeros(3)),
                      dict(delta_quaternion=np.zeros((2, 3))), dict(beam_tilt=np.array([0., np.nan])),
                      dict(image_ids=np.array(['p0', 'p0']))):
            with self.subTest(field=list(field)):
                values = dict(pose_metadata(self.scene, 8), **field)
                atomic_npz(path, **values)
                with self.assertRaises(ValueError):
                    validate_pose_file(path, self.scene, 8)
        atomic_npz(path, delta_angle=np.zeros(2))
        with self.assertRaisesRegex(ValueError, 'Unbound'):
            validate_pose_file(path, self.scene, 8)

    def test_dataset_identity_is_order_independent_but_content_sensitive(self):
        original = dataset_identity(self.source)
        self.metadata['proj_train'].reverse()
        self.write_metadata()
        self.assertEqual(dataset_identity(self.source), original)
        self.metadata['scanner']['offOrigin'][0] = .2
        self.write_metadata()
        self.assertNotEqual(dataset_identity(self.source), original)

    def test_train_fbp_cache_content_and_output_tampering_force_reconstruction(self):
        args = SimpleNamespace(use_train_fbp=True, dataset=self.source, volume=None,
                               fbp_output=self.root / 'cache', stripe_correction=None)
        volume = np.ones((5, 7, 9), np.float32)
        with mock.patch.object(init, 'reconstruct_from_train_views', return_value=volume) as recon, \
             contextlib.redirect_stdout(io.StringIO()):
            _, path = init.load_initial_volume(args)
            self.assertEqual(path.suffix, '.npy')
            init.load_initial_volume(args)
            self.assertEqual(recon.call_count, 1)
            np.save(self.source / 'p0.npy', np.ones((4, 6), np.float32))
            init.load_initial_volume(args)
            self.assertEqual(recon.call_count, 2)
            np.save(path, volume * 2)
            init.load_initial_volume(args)
            self.assertEqual(recon.call_count, 3)

    def test_stripe_train_cache_reuse_tampering_and_provenance(self):
        from scripts.stripe_correction import normalize_options
        self.metadata['source'] = {'stripe_correction': {
            'algorithm': 'tomopy.remove_all_stripe',
            'parameters': normalize_options({})}}
        self.write_metadata()
        args = SimpleNamespace(use_train_fbp=True, dataset=self.source, volume=None,
                               fbp_output=self.root / 'stripe_cache', stripe_correction={})
        volume = np.ones((5, 7, 9), np.float32)
        with mock.patch.object(init, 'reconstruct_from_train_views', return_value=volume) as recon, \
             contextlib.redirect_stdout(io.StringIO()):
            _, path = init.load_initial_volume(args)
            self.assertEqual(path.suffix, '.npy')
            init.load_initial_volume(args)
            self.assertEqual(recon.call_count, 1)
            np.save(self.source / 'p0.npy', np.ones((4, 6), np.float32))
            init.load_initial_volume(args)
            self.assertEqual(recon.call_count, 2)
            np.save(path, volume * 2)
            init.load_initial_volume(args)
            self.assertEqual(recon.call_count, 3)
            args.stripe_correction = {'snr': 4.0}
            with self.assertRaisesRegex(ValueError, 'stripe settings do not match'):
                init.load_initial_volume(args)
            self.assertEqual(recon.call_count, 3)

    def test_resume_validates_repeated_resume_config_inputs_and_output(self):
        initfile = self.root / 'init.npy'
        np.save(initfile, np.zeros((5, 4), np.float32))
        base = ['--start_from', 'train', '--stop_after', 'train',
                '--train_ply_path', str(initfile), '--train_iterations', '4']
        calls = []
        def train_stub(command, **kw):
            calls.append(command)
            model = self.root / 'model'
            folder = model / 'point_cloud' / 'iteration_4'
            folder.mkdir(parents=True)
            (folder / 'point_cloud.pickle').write_bytes(b'model')
            (model / 'cfg_args').write_text('Namespace()')
        with mock.patch.object(pipeline.subprocess, 'run', side_effect=train_stub), \
             contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(self.args(base))
            pipeline.main(self.args([*base, '--resume']))
            pipeline.main(self.args([*base, '--resume']))
            self.assertEqual(len(calls), 1)
            changed = self.args([*base, '--resume', '--train_iterations', '5'])
            with self.assertRaisesRegex(ValueError, 'Cannot resume'):
                pipeline.main(changed)
            model = self.root / 'model' / 'point_cloud' / 'iteration_4' / 'point_cloud.pickle'
            model.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Invalid artifact'):
                pipeline.main(self.args([*base, '--resume']))

    def test_dry_run_entire_pipeline_writes_nothing(self):
        args = self.args(['--dry_run'])
        args.output_dir = self.root / 'newout'
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(args)
        self.assertFalse(args.output_dir.exists())
        self.assertFalse(args.model_path.exists())

    def test_top_level_pipeline_paths_and_legacy_train_paths_are_distinct(self):
        values = dict(model_path='model', source_path='data', iterations=12)
        current = pipeline.flatten_config(values)
        legacy = pipeline.flatten_config(values, legacy_train=True)
        self.assertEqual(current['model_path'], 'model')
        self.assertEqual(current['train_source_path'], 'data')
        self.assertEqual(legacy['train_model_path'], 'model')
        self.assertEqual(legacy['train_iterations'], 12)

    def test_source_signature_ignores_generated_nested_outputs_but_hashes_tiffs(self):
        raw = self.root / 'raw'
        raw.mkdir()
        (raw / 'view0.tiff').write_bytes(b'raw')
        args = self.args()
        before = artifacts.signature(args, 'fbp')
        generated = raw / 'outputs'
        generated.mkdir()
        (generated / 'manifest.json').write_text('{}')
        self.assertEqual(artifacts.signature(args, 'fbp'), before)
        (raw / 'view0.tiff').write_bytes(b'changed')
        self.assertNotEqual(artifacts.signature(args, 'fbp'), before)

    def test_config_inheritance_none_typed_values_and_test_path_fallback(self):
        parent = self.root / 'parent.yml'
        parent.write_text('iterations: 19\nmax_scale: 0.4\nmax_screen_size: 23\n')
        child = self.root / 'child.yml'
        child.write_text(yaml.safe_dump(dict(inherit_from=str(parent), lambda_tv=.1)))
        args = self.args(['--train_config', str(child), '--train_source_path', str(self.source),
                          '--train_model_path', str(self.root / 'special_model')])
        command = pipeline.build_train_command(args, self.root / 'init.npy', child)
        self.assertEqual(args.train_iterations, 19)
        self.assertNotIn('--max_scale', command)
        self.assertNotIn('--max_screen_size', command)
        path = self.root / 'generated.yml'
        pipeline.write_train_config(args, path, self.source, args.model_path, self.root / 'init.npy')
        values = yaml.safe_load(path.read_text())
        self.assertEqual((values['max_scale'], values['max_screen_size']), (.4, 23))
        command = pipeline.build_test_command(args, None)
        self.assertEqual(command[command.index('-m')+1], str(self.root / 'special_model'))
        self.assertEqual(command[command.index('-s')+1], str(self.source))

    def test_natural_filename_sort_and_cone_keeps_180_degree_endpoint(self):
        for name in ['Tomo_10.tif', 'Tomo_2.tiff', 'Tomo_1.tiff']:
            (self.root / name).touch()
        self.assertEqual([p.stem for p in select_tiff_paths(self.root)], ['Tomo_1','Tomo_2','Tomo_10'])
        args = SimpleNamespace(config=None, angle_start=0, angle_interval=90,
                               mode='cone', keep_duplicate_endpoint=False)
        self.assertEqual(len(build_angles(args, 3)), 3)
        args.mode = 'parallel'
        self.assertEqual(len(build_angles(args, 3)), 2)

    def test_cone_pitch_is_explicit_and_axis_detector_options_equivalent(self):
        prepare = import_prepare()
        args = prepare.build_parser().parse_args(['--input_dir', str(self.root), '--output_dir', str(self.root / 'out')])
        args.mode, args.DSO, args.DSD = 'cone', 5., 10.
        args.pixel_size = .1
        with self.assertRaisesRegex(ValueError, 'pixel_size_kind'):
            prepare.build_geometry(args, (24, 32))
        args.pixel_size_kind = 'axis'
        axis = prepare.build_geometry(args, (24, 32))
        args.pixel_size_kind, args.pixel_size = 'detector', .2
        detector = prepare.build_geometry(args, (24, 32))
        np.testing.assert_allclose(axis.dDetector, detector.dDetector)
        np.testing.assert_allclose(axis.sDetector, detector.sDetector)

    def test_odd_detector_sampling_retains_center_and_declared_pitch(self):
        for shape in [(51, 73), (51, 74), (50, 75)]:
            yy, xx = np.indices(shape)
            # A linear ramp's central derivatives expose any spacing/center drift.
            image = (100 + 2*yy + 3*xx).astype(np.float32)
            args = SimpleNamespace(input_type='line_integral', clip_percentile=100, shift_v=0,
                   pixel_subsample=3, resize_order=1, projection_scale=1., flat_field=False)
            result = process_projection(image, args)
            expected_center = 100 + 2*(shape[0]-1)/2 + 3*(shape[1]-1)/2
            self.assertAlmostEqual(float(result.mean()), expected_center, places=3)
            n = result.shape[0]//2
            self.assertAlmostEqual(float(result[n,n+1]-result[n,n]), 9., places=3)
            self.assertAlmostEqual(float(result[n+1,n]-result[n,n]), 6., places=3)


if __name__ == '__main__':
    unittest.main()
