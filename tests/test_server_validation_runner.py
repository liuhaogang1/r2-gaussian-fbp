"""Server launcher regressions: failure propagation, settings and physical coordinates."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scripts import server_parallel_validation as server


class ServerRunnerTests(unittest.TestCase):
    def test_plan_creates_no_output_and_config_resolves_real_flags(self):
        import run_pipeline
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'new'
            result=subprocess.run([sys.executable,str(server.ROOT/'scripts/server_parallel_validation.py'),
                '--plan','--run-root',str(root)],capture_output=True,text=True,check=True)
            self.assertFalse(root.exists())
            config=json.loads(result.stdout)['pipeline']
            path=Path(temp)/'pipeline.json'
            path.write_text(json.dumps(config))
            args=run_pipeline.resolve_args(run_pipeline.build_parser().parse_args([
                '--pipeline_config',str(path),'--train_config',str(server.ROOT/'configs/pose_parallel.yaml')]))
            self.assertTrue(args.train_pose_enabled)
            self.assertTrue(args.train_shape_enabled)
            self.assertFalse(args.train_slice_loss_enabled)
            self.assertEqual(args.train_iterations,200)
            self.assertEqual(args.fbp_n_train,360)
            self.assertEqual(args.fbp_n_test,0)
            self.assertEqual(args.stripe_correction['dim'],1)
            self.assertTrue(args.train_parallel_optimize_per_view_detector_shift)

    def test_child_failure_is_logged_and_summary_never_reports_success(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            runner=server.Runner(root)
            with self.assertRaisesRegex(RuntimeError,'exited 7'):
                runner.command('failure',[sys.executable,'-c',"print('diagnostic marker',flush=True); raise SystemExit(7)"])
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['status'],'failed')
            self.assertEqual(summary['stages'][0]['status'],'failed')
            self.assertIn('diagnostic marker',(root/'logs/failure.log').read_text())
            self.assertIn('exited 7',(root/'logs/failure.log').read_text())

    def test_init_coordinate_validation_uses_normalized_scene_and_xyz_nifti(self):
        from scipy.ndimage import gaussian_filter
        from r2_gaussian.utils.geometry_utils import initialization_positions
        from r2_gaussian.utils.volume_export_utils import save_nifti_xyz
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            data=root/'real/dataset'
            folder=root/'real/model/init_denoise'
            data.mkdir(parents=True)
            folder.mkdir(parents=True)
            scanner=dict(nVoxel=[9,11,13],sVoxel=[.004,.006,.005],offOrigin=[.001,-.002,.003])
            (data/'meta_data.json').write_text(json.dumps(dict(scanner=scanner)))
            v=np.random.default_rng(0).uniform(.1,1,(9,11,13)).astype(np.float32)
            np.save(data/'vol_train_fbp.npy',v)
            filtered=gaussian_filter(v,.8,mode='nearest',truncate=4.)
            np.save(folder/'vol_init_denoised.npy',filtered)
            save_nifti_xyz(filtered,scanner,folder/'vol_init_denoised.nii.gz')
            indices=np.array([[0,0,0],[4,5,6],[8,10,12]])
            points=np.column_stack([initialization_positions(indices,v.shape,scanner),filtered[tuple(indices.T)]*.2]).astype(np.float32)
            np.save(folder/'init_points.npy',points)
            (folder/'denoise_report.json').write_text(json.dumps(dict(input=dict(volume_path=str(data/'vol_train_fbp.npy')),
                    normalization_scale=1.,candidate_count=v.size)))
            result=server.check_real(root,'init',6,root/'check.log')
            self.assertTrue(result['xyz_roundtrip'])
            points[:,0]*=-1
            np.save(folder/'init_points.npy',points)
            with self.assertRaises((AssertionError,ValueError)):
                server.check_real(root,'init',6,root/'bad.log')


if __name__=='__main__':
    unittest.main()
