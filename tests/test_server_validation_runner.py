"""Server launcher regressions: failure propagation, settings and physical coordinates."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

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
            self.assertFalse(runner.command('failure',[sys.executable,'-c',"print('diagnostic marker',flush=True); raise SystemExit(7)"]))
            self.assertFalse(runner.stage('dependent',lambda log:self.fail('Dependency should not execute'),requires=('failure',)))
            self.assertFalse(runner.stage('transitive',lambda log:self.fail('Transitive dependency should not execute'),requires=('dependent',)))
            self.assertTrue(runner.command('next_independent',[sys.executable,'-c',"print('continued')"]))
            self.assertEqual(runner.finish(),1)
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['status'],'failed')
            self.assertEqual(summary['stages'][0]['status'],'failed')
            self.assertEqual(summary['stages'][-1]['status'],'passed')
            self.assertEqual(summary['counts'],dict(passed=1,failed=1,skipped=2,interrupted=0))
            self.assertEqual(summary['failures'][0]['name'],'failure')
            self.assertEqual(summary['failures'][0]['log'],str(root/'logs/failure.log'))
            self.assertIn('diagnostic marker',(root/'logs/failure.log').read_text())
            self.assertIn('exited 7',(root/'logs/failure.log').read_text())
            self.assertIn('failure=failed',(root/'logs/dependent.log').read_text())
            self.assertIn('logs',(root/'report.txt').read_text())
            self.assertIn('next_independent\tpassed',(root/'summary.tsv').read_text())

    def test_clean_finish_and_interrupt_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            runner=server.Runner(Path(temp))
            runner.stage('passed',lambda log:{'ok':True})
            self.assertEqual(runner.finish(),0)
            self.assertEqual(runner.results['status'],'completed')
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            runner=server.Runner(Path(temp))
            def interrupt(log):
                raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                runner.stage('interrupted',interrupt)
            self.assertEqual(runner.finish(),130)
            self.assertEqual(runner.results['status'],'interrupted')

    def test_suite_dependencies_preserve_independent_real_branches(self):
        # Exercise the actual stage graph without CUDA, TIFFs or training.
        for failures in ({'03_physical_parallel','07_synthetic_pipeline','20_center_fbp_init'},{'22_train'}):
            with self.subTest(failures=failures),tempfile.TemporaryDirectory() as temp, \
                 contextlib.redirect_stdout(io.StringIO()),contextlib.ExitStack() as stack:
                root=Path(temp)
                runner=server.Runner(root)
                args=server.parser().parse_args([])
                def command(runner,name,command,requires=()):
                    def execute(log):
                        log.write_text(name)
                        if name in failures: raise RuntimeError('Simulated failure')
                        return {'returncode':0}
                    return runner.stage(name,execute,requires=requires)
                stack.enter_context(mock.patch.object(server.Runner,'command',command))
                for name in ('static_checks','check_real','check_projection_metrics','check_pose_artifacts',
                             'check_cache_resume','prepare_real_branches','check_highres_mask'):
                    stack.enter_context(mock.patch.object(server,name,return_value={}))
                server.run_validation(runner,args,root,[sys.executable],root/'pipeline.json')
                self.assertEqual(runner.finish(),1)
                states={s['name']:s['status'] for s in runner.results['stages']}
                self.assertEqual(states['30_highres'],'skipped')
                self.assertEqual(states['32_real_chunk_consistency'],'skipped')
                if '20_center_fbp_init' in failures:
                    self.assertEqual(states['10_cone_geometry'],'passed')
                    self.assertEqual(states['08_synthetic_branches'],'skipped')
                    self.assertEqual(states['09_auxiliary'],'skipped')
                    self.assertEqual(states['22_train'],'skipped')
                    self.assertEqual(states['26_real_ordinary'],'skipped')
                else:
                    self.assertEqual(states['23_test'],'skipped')
                    self.assertEqual(states['26_real_ordinary'],'passed')
                    self.assertEqual(states['26_real_sparse_evaluation_artifacts'],'passed')
                    self.assertEqual(states['27_real_checkpoint_resume'],'skipped')

    def test_real_branch_configs_do_not_require_a_started_main_train(self):
        import yaml
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args=server.parser().parse_args([])
            (root/'pipeline.json').write_text(json.dumps(server.settings(args,root)))
            server.prepare_real_branches(args,root,root/'prepare.log')
            ordinary=yaml.safe_load((root/'ordinary_train.yaml').read_text())
            sparse=yaml.safe_load((root/'sparse_train.yaml').read_text())
            resume=yaml.safe_load((root/'checkpoint_resume.yaml').read_text())
            self.assertFalse(ordinary['pose_enabled'])
            self.assertTrue(sparse['slice_loss_enabled'])
            self.assertEqual(ordinary['ply_path'],str(root/'real/model/init_denoise/init_points.npy'))
            self.assertEqual(resume['start_checkpoint'],str(root/f'real/model/ckpt/chkpnt{args.iterations}.pth'))
            self.assertFalse((root/'real/model/init_denoise/.pipeline_train_config.yml').exists())

    def test_main_returns_failure_exit_code_and_finalizes_unexpected_errors(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)/'run'
            with mock.patch.object(sys,'argv',['server','--run-root',str(root)]), \
                 mock.patch.object(Path,'rglob',return_value=[]), \
                 mock.patch.object(server.os,'chdir'), \
                 mock.patch.object(server,'run_validation',side_effect=ValueError('orchestration failed')):
                self.assertEqual(server.main(),1)
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['status'],'failed')
            self.assertEqual(summary['failures'][0]['name'],'runner_error')
            self.assertIn('orchestration failed',(root/'report.txt').read_text())

    def test_projection_metrics_require_all_views_and_finite_values(self):
        import yaml
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            data=root/'real/dataset'
            evaluation=root/'real/model/test/iter_6'
            data.mkdir(parents=True)
            evaluation.mkdir(parents=True)
            (data/'meta_data.json').write_text(json.dumps(dict(proj_train=[{},{}],proj_test=[{}])))
            path=evaluation/'eval2d_render_test.yml'
            metrics=dict(psnr_2d=20.,ssim_2d=.8,psnr_2d_projs=[20.],ssim_2d_projs=[.8])
            path.write_text(yaml.safe_dump(metrics))
            result=server.check_projection_metrics(root,'test',6,1,root/'metrics.log')
            self.assertEqual(result['n_views'],1)
            for broken in (dict(metrics,ssim_2d_projs=[]),dict(metrics,psnr_2d=float('nan')),
                           dict(metrics,psnr_2d_projs=[float('inf')]),dict(status='skipped')):
                path.write_text(yaml.safe_dump(broken))
                with self.assertRaises(ValueError):
                    server.check_projection_metrics(root,'test',6,1,root/'bad.log')

    def test_real_pose_audit_rejects_corruption_bounds_and_broken_csv(self):
        import csv
        from types import SimpleNamespace
        from r2_gaussian.pose.artifacts import pose_metadata
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            data=root/'real/dataset'
            model=root/'real/model'
            folder=model/'pose_calibration'
            data.mkdir(parents=True)
            (model/'init_denoise').mkdir(parents=True)
            records=[dict(file_path=f'{i:04d}.npy',angle=i*.5) for i in range(2)]
            for record in records: np.save(data/record['file_path'],np.ones((4,4),np.float32))
            scanner=dict(mode='parallel')
            (data/'meta_data.json').write_text(json.dumps(dict(scanner=scanner,proj_train=records,proj_test=[])))
            config={f'parallel_optimize_{k}':True for k in ('angle','global_detector_shift',
                    'per_view_detector_shift','beam_tilt','detector_roll')}
            (model/'init_denoise/.pipeline_train_config.yml').write_text(json.dumps(config))
            scene=SimpleNamespace(source_path=str(data),scanner_cfg=scanner,
                                  train_cameras=[SimpleNamespace(image_name=f'{i:04d}',angle=i*.5) for i in range(2)])
            values=dict(pose_format_version=np.asarray(2),delta_angle=np.array([.001,-.001]),
                        global_shift=np.array([.5,.25]),view_shift=np.array([[.1,-.2],[-.1,.2]]),
                        beam_tilt=np.array([.001,.002]),detector_roll=np.array([.001]))
            for iteration in (3,6):
                snapshot=folder/f'iteration_{iteration}'
                snapshot.mkdir(parents=True)
                np.savez(snapshot/'pose_offsets.npz',**dict(values,**pose_metadata(scene,iteration)))
            final=dict(values,**pose_metadata(scene,6))
            np.savez(folder/'pose_offsets.npz',**final)
            with (folder/'pose_report.csv').open('w',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['view','image','angle_delta_rad','shift_u_px','shift_v_px',
                                 'beam_tilt_u_rad','beam_tilt_v_rad','detector_roll_rad'])
                for i in range(2):
                    writer.writerow([i,f'{i:04d}',values['delta_angle'][i],
                                     *(values['global_shift']+values['view_shift'][i]),
                                     *values['beam_tilt'],values['detector_roll'][0]])
            result=server.check_pose_artifacts(root,[3,6],root/'pose.log')
            self.assertTrue(result['csv_matches_snapshot'])
            for changes in (dict(detector_roll=np.array([.1])),dict(global_shift=np.array([np.nan,0.])),
                            dict(view_shift=np.zeros((3,2))),dict(delta_angle=np.array([.001,.001])),
                            dict(dataset_identity=np.asarray('wrong-dataset'))):
                broken=dict(final,**changes)
                np.savez(folder/'iteration_6/pose_offsets.npz',**broken)
                np.savez(folder/'pose_offsets.npz',**broken)
                with self.assertRaises(ValueError):
                    server.check_pose_artifacts(root,[3,6],root/'bad.log')
            np.savez(folder/'iteration_6/pose_offsets.npz',**final)
            np.savez(folder/'pose_offsets.npz',**final)
            report=folder/'pose_report.csv'
            report.write_text(report.read_text().replace('0000','wrong-view'))
            with self.assertRaisesRegex(ValueError,'CSV view identifiers'):
                server.check_pose_artifacts(root,[3,6],root/'bad_csv.log')

    def test_real_chunk_audit_detects_seam_error_without_cuda(self):
        import torch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            folder=root/'real/highres'
            folder.mkdir(parents=True)
            shape=np.array([33,41,49])
            size=np.array([2.,3.,4.])
            center=np.array([.2,-.3,.4])
            def field(local_center,count,local_size):
                coordinates=np.indices(count).transpose(1,2,3,0)
                xyz=np.asarray(local_center)-np.asarray(local_size)/2+(coordinates+.5)*np.asarray(local_size)/count
                return (100+np.sum(xyz*[1,2,3],axis=-1)).astype(np.float32)
            volume=field(center,shape,size)
            np.save(folder/'vol_pred.npy',volume)
            (folder/'vol_pred.json').write_text(json.dumps(dict(iteration=6,pose_enabled=True,
                    shape_xyz=shape.tolist(),query_chunk_size_xyz=[16,24,32])))
            def query(model,local_center,count,local_size,pipe):
                self.assertTrue(pipe.pose_enabled)
                return dict(vol=torch.from_numpy(field(local_center,count,local_size)))
            exporter=SimpleNamespace(load_scanner_config=lambda path:(dict(sVoxel=size,offOrigin=center),1.))
            gaussian=SimpleNamespace(GaussianModel=lambda unused:None,initialize_gaussian=lambda *args:6,query=query)
            cfg=dict(output_dir='unused',model_path='unused',train_iterations=6,train_pose_enabled=True)
            with mock.patch.dict(sys.modules,{'export_highres':exporter,'r2_gaussian.gaussian':gaussian}):
                result=server.check_chunked_export(cfg,root)
                self.assertEqual(len(result['windows']),3)
                self.assertTrue(all(w['mismatched_voxels']==0 for w in result['windows']))
                volume[16,24,32]+=1
                np.save(folder/'vol_pred.npy',volume)
                with self.assertRaisesRegex(AssertionError,'differs from direct queries'):
                    server.check_chunked_export(cfg,root)
                report=json.loads((folder/'chunk_consistency.json').read_text())
                self.assertGreater(sum(w['mismatched_voxels'] for w in report['windows']),0)

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
