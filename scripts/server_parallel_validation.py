"""Reproduce regression checks, then exercise parallel reconstruction on real TIFFs.

No package installation or production compatibility workaround is performed.
Run --plan to inspect the exact pipeline settings without importing CUDA packages.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=Path('/root/fnvme/data_10_8/1/cata2_8.4kev_400ms_1'))
    p.add_argument('--scan-config', type=Path, help='Defaults to INPUT/config.txt')
    p.add_argument('--run-root', type=Path, help='Must not exist; defaults to a timestamped validation directory')
    p.add_argument('--train-config', type=Path, default=ROOT/'configs/pose_parallel.yaml')
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--branch-iterations', type=int, default=12)
    p.add_argument('--n-train', type=int, default=360)
    p.add_argument('--n-test', type=int, default=0, help='0 preserves the supplied empty held-out split')
    p.add_argument('--n-points', type=int, default=100000)
    p.add_argument('--density-thresh', type=float, default=.25)
    p.add_argument('--n-voxel', nargs=3, type=int, default=[256,256,256])
    p.add_argument('--highres', nargs=3, type=int, default=[320,320,320])
    p.add_argument('--stripe-ncore', type=int, default=8)
    p.add_argument('--stripe-row-chunk', type=int, default=32)
    p.add_argument('--upstream', type=Path, help='Optional server copy of the supplied official code for comparison')
    p.add_argument('--no-pose', action='store_true', help='Explicit ordinary-training run; no automatic fallback')
    p.add_argument('--skip-synthetic', action='store_true', help='Record synthetic/regression suite as skipped')
    p.add_argument('--skip-cone', action='store_true', help='Skip separate cone geometry smoke test only')
    p.add_argument('--skip-real-branches', action='store_true', help='Skip real-data ordinary/sparse/resume side runs')
    p.add_argument('--plan', action='store_true', help='Print settings only; create nothing')
    p.add_argument('--_preflight-config', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--_chunk-check-config', type=Path, help=argparse.SUPPRESS)
    return p


def settings(a, out):
    total = a.iterations
    half = max(2, total//2)
    return dict(input_dir=str(a.input_dir.resolve()), output_dir=str(out/'real/dataset'),
        model_path=str(out/'real/model'), fbp_config=str((a.scan_config or a.input_dir/'config.txt').resolve()),
        train_config=str(a.train_config.resolve()), seed=0,
        fbp_projection_glob='Tomo_*.tiff', fbp_flat_glob='Flat_*.tiff', fbp_flat_field=True,
        fbp_dark_value=0, fbp_flat_eps=1e-6,
        stripe_correction=dict(snr=3.,la_size=61,sm_size=21,dim=1),
        stripe_ncore=a.stripe_ncore, stripe_row_chunk=a.stripe_row_chunk,
        fbp_input_type='transmission', fbp_log_eps=1e-6, fbp_mode='parallel',
        fbp_shift_v=0, fbp_pixel_subsample=1, fbp_recon_pixel_subsample=1,
        fbp_resize_order=1, fbp_projection_scale=.125, fbp_pixel_size=.00004,
        fbp_nVoxel=a.n_voxel, fbp_sVoxel='auto', fbp_offOrigin=[0,0,0],
        fbp_DSD=7, fbp_DSO=5, fbp_accuracy=.5, fbp_filter='hann',
        center_method='pair', center_pixel_size=.00004, center_output=str(out/'real/center.json'),
        fbp_n_train=a.n_train, fbp_n_test=a.n_test,
        fbp_sparse_views=min(75,a.n_train), fbp_sparse_fbp_output='vol_sparse_fbp.npy',
        init_source='train_fbp', init_denoise=True, init_denoise_sigma=.8, init_denoise_save_nii=True,
        init_n_points=a.n_points, init_density_thresh=a.density_thresh, init_density_rescale=.2,
        init_normalize_percentile=99.5, init_visualize=True, init_visualize_slices=True,
        init_visualize_fbp_nii=True, train_pose_enabled=not a.no_pose, train_pose_mode='parallel',
        train_parallel_optimize_angle=True, train_parallel_optimize_global_detector_shift=True,
        train_parallel_optimize_per_view_detector_shift=True, train_parallel_optimize_beam_tilt=True,
        train_parallel_optimize_detector_roll=True, train_max_num_gaussians=300000,
        train_iterations=total, train_lambda_tv=.01, train_tv_vol_size=16,
        train_densify_from_iter=2, train_densify_until_iter=max(3,total-1),
        train_densification_interval=max(2,total//10),
        train_shape_enabled=True, train_shape_needle_enabled=True, train_shape_sheet_enabled=True,
        train_lambda_shape_needle=.001, train_lambda_shape_sheet=.001,
        train_shape_start_iter=1, train_shape_end_iter=total, train_shape_ramp_iters=max(1,total//5),
        train_shape_check_interval=max(1,total//10), train_shape_patience=2,
        train_shape_hard_action='repair',
        train_test_iterations=[half,total], train_save_iterations=[half,total],
        train_checkpoint_iterations=[half,total], test_iteration=total,
        test_skip_render_train=False, test_skip_render_test=False, test_skip_recon=False)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


class Runner:
    def __init__(self, out):
        self.out = out
        (out/'logs').mkdir(parents=True)
        self.results = dict(started=dt.datetime.now().astimezone().isoformat(),
                            python=sys.executable, stages=[], status='running')
        self.save()

    def save(self):
        temp = self.out/'summary.tmp'
        temp.write_text(json.dumps(self.results,indent=2,ensure_ascii=False),encoding='utf-8')
        temp.replace(self.out/'summary.json')
        lines=['stage	status	seconds	log']
        for s in self.results['stages']:
            lines.append(f"{s['name']}\t{s['status']}\t{s.get('seconds',0):.2f}\t{s.get('log','')}")
        (self.out/'summary.tsv').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    def skip(self, name, reason):
        print(f'[SKIP] {name}: {reason}',flush=True)
        log=self.out/'logs'/f'{name}.log'
        log.write_text(f'[SKIP] {name}: {reason}\n',encoding='utf-8')
        self.results['stages'].append(dict(name=name,status='skipped',reason=reason,log=str(log)))
        self.save()
        return False

    def stage(self, name, action, requires=()):
        states={s['name']:s['status'] for s in self.results['stages']}
        blocked=[f'{dependency}={states.get(dependency,"not run")}' for dependency in requires
                 if states.get(dependency)!='passed']
        if blocked:
            return self.skip(name,'Prerequisite not passed: '+', '.join(blocked))
        record=dict(name=name,status='running',log=str(self.out/'logs'/f'{name}.log'),requires=list(requires))
        self.results['stages'].append(record)
        self.save()
        print(f'\n[START] {name} | {dt.datetime.now().astimezone().isoformat()}',flush=True)
        start=time.monotonic()
        try:
            result=action(Path(record['log']))
            record.update(status='passed',details=result)
        except BaseException as exc:
            import traceback
            with Path(record['log']).open('a',encoding='utf-8') as f:
                f.write('\n'+traceback.format_exc())
            interrupted=not isinstance(exc,Exception)
            record.update(status='interrupted' if interrupted else 'failed',error=str(exc) or type(exc).__name__)
            self.results['status']=record['status']
            if interrupted:
                raise
        finally:
            record['seconds']=time.monotonic()-start
            self.save()
            print(f"[{record['status'].upper()}] {name} ({record['seconds']:.1f}s) | {record['log']}",flush=True)
        return record['status']=='passed'

    def command(self, name, command, requires=()):
        command=list(map(str,command))
        def execute(log):
            print('+ '+shlex.join(command),flush=True)
            with log.open('w',encoding='utf-8') as f:
                f.write('+ '+shlex.join(command)+'\n')
                f.flush()
                with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                      text=True,encoding='utf-8',errors='replace') as process:
                    try:
                        for line in process.stdout:
                            print(line,end='',flush=True)
                            f.write(line)
                            f.flush()
                        code=process.wait()
                    except BaseException:
                        process.terminate()
                        try: process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise
            if code:
                raise RuntimeError(f'{name}: command exited {code}; see {log}')
            return dict(command=command,returncode=code)
        return self.stage(name,execute,requires=requires)

    def finish(self, interrupted=False):
        stages=self.results['stages']
        failures=[dict(name=s['name'],log=s['log'],error=s['error']) for s in stages if s['status']=='failed']
        interrupted=interrupted or any(s['status']=='interrupted' for s in stages)
        counts={status:sum(s['status']==status for s in stages)
                for status in ('passed','failed','skipped','interrupted')}
        self.results.update(status='interrupted' if interrupted else ('failed' if failures else 'completed'),
                            finished=dt.datetime.now().astimezone().isoformat(),failures=failures,counts=counts,
                            note='Smoke validation only; metrics against FBP are not independent ground truth.')
        self.save()
        lines=[f"RESULT: {self.results['status'].upper()} | "+', '.join(f'{k}={v}' for k,v in counts.items())]
        for status in ('failed','interrupted','skipped'):
            for stage in stages:
                if stage['status']==status:
                    lines.extend([f"[{status.upper()}] {stage['name']}: {stage.get('error',stage.get('reason',''))}",
                                  f"  log: {stage['log']}"])
        lines.extend([f'Summary: {self.out / "summary.tsv"}',f'JSON: {self.out / "summary.json"}',
                      f'Main model (if produced): {self.out / "real/model"}'])
        report='\n'.join(lines)+'\n'
        (self.out/'report.txt').write_text(report,encoding='utf-8')
        print('\n'+report,flush=True)
        return 130 if interrupted else int(bool(failures))


def preflight(a, out, cfg, log):
    if sys.version_info < (3,10):
        raise RuntimeError('Server entrypoints require Python >=3.10; no compatibility changes/installations are made.')
    import numpy as np
    import torch
    import tomopy
    import SimpleITK
    import tifffile
    from r2_gaussian.gaussian.render_query import _load_raster_backend
    from scripts.fbp_preprocess import select_tiff_paths, build_angles
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in the active interpreter')
    _load_raster_backend(False)
    if not a.no_pose:
        _load_raster_backend(True)
    # Import TIGRE here, but all real Ax/reconstruction CUDA work is in children.
    import tigre
    files=select_tiff_paths(a.input_dir,'Tomo_*.tiff')
    flats=select_tiff_paths(a.input_dir,'Flat_*.tiff')
    if not files or not flats:
        raise RuntimeError('Tomo_*.tiff or Flat_*.tiff missing')
    if not Path(cfg['fbp_config']).is_file() or not a.train_config.is_file():
        raise FileNotFoundError('Acquisition/training config missing')
    from types import SimpleNamespace
    angles=build_angles(SimpleNamespace(config=cfg['fbp_config'],angle_start=0.,angle_interval=.5,
                        mode='parallel',keep_duplicate_endpoint=False),len(files))
    if a.n_train+a.n_test>len(angles):
        raise ValueError(f'Requested {a.n_train} train + {a.n_test} test views; only {len(angles)} unique views')
    sample=tifffile.imread(files[0])
    if sample.ndim!=2 or tifffile.imread(flats[0]).shape!=sample.shape:
        raise ValueError('Projection/flat must be matching 2D arrays')
    # Execute the installed stripe operator, rather than merely importing it.
    stripe=tomopy.remove_all_stripe(np.ones((8,4,64),np.float32),snr=3.,la_size=61,sm_size=21,dim=1,ncore=1)
    if stripe.shape!=(8,4,64) or not np.isfinite(stripe).all():
        raise ValueError('TomoPy remove_all_stripe failed the execution probe')
    data=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
              gpu=torch.cuda.get_device_name(0),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
              projections=len(files),flats=len(flats),detector_shape=list(sample.shape),
              selected_input_files=[str(p) for p in files],
              pose_enabled=not a.no_pose,config_sha256=digest(cfg['fbp_config']))
    import shutil
    shutil.copy2(cfg['fbp_config'],out/'acquisition_config.txt')
    log.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(data,indent=2),flush=True)
    return data


def check_real(out, part, iteration, log, model=None):
    import numpy as np
    data=out/'real/dataset'
    model=model or out/'real/model'
    meta=json.loads((data/'meta_data.json').read_text())
    def array(path,shape=None):
        v=np.load(path,mmap_mode='r')
        if shape is not None and tuple(v.shape)!=tuple(shape):
            raise ValueError(f'Wrong shape {v.shape}: {path}')
        for block in v:
            if not np.isfinite(block).all(): raise ValueError(f'Nonfinite values: {path}')
        return v
    result=dict(part=part,scanner=meta['scanner'])
    if part=='fbp':
        center=json.loads((out/'real/center.json').read_text())
        if not center['center_valid']: raise ValueError(f'Invalid rotation center: {center}')
        result['center']=center
        shape=meta['scanner']['nVoxel']
        array(data/'vol_fbp.npy',shape)
        array(data/'vol_sparse_fbp.npy',shape)
        if meta['scanner']['mode']!='parallel': raise ValueError('Expected parallel data')
        paths=[r['file_path'] for k in ('proj_train','proj_test') for r in meta[k]]
        if len(set(paths))!=len(paths): raise ValueError('Train/test overlap or duplicate projections')
        for path in paths: array(data/path)
        result.update(n_train=len(meta['proj_train']),n_test=len(meta['proj_test']))
    elif part=='init':
        import SimpleITK as sitk
        from scipy.ndimage import gaussian_filter
        folder=model/'init_denoise'
        report=json.loads((folder/'denoise_report.json').read_text())
        original=array(report['input']['volume_path'])
        denoised=array(folder/'vol_init_denoised.npy',original.shape)
        expected=np.maximum(gaussian_filter(np.maximum(original,0),sigma=.8,mode='nearest',truncate=4.),0)
        np.testing.assert_array_equal(denoised,expected)
        image=sitk.ReadImage(str(folder/'vol_init_denoised.nii.gz'))
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image).transpose(2,1,0),denoised)
        points=array(folder/'init_points.npy')
        n=np.array(original.shape)
        size=np.array(meta['scanner']['sVoxel'])
        center=np.array(meta['scanner']['offOrigin'])
        physical_points=points[:,:3]*(float(np.max(size))/2)
        indices=np.rint((physical_points-center+size/2)/(size/n)-.5).astype(int)
        if not ((indices>=0).all() and (indices<n).all()): raise ValueError('Point outside voxel grid')
        positions=(indices+.5)*size/n-size/2+center
        np.testing.assert_allclose(physical_points,positions,rtol=2e-5,atol=float(np.max(size/n))*1e-3)
        normalized=np.clip(denoised/report['normalization_scale'],0,1)
        np.testing.assert_allclose(points[:,3],normalized[tuple(indices.T)]*.2,rtol=2e-5,atol=1e-7)
        result.update(points=len(points),candidate_count=report['candidate_count'],xyz_roundtrip=True)
    elif part=='train':
        array(model/f'point_cloud/iteration_{iteration}/vol_pred.npy',meta['scanner']['nVoxel'])
        if not (model/f'point_cloud/iteration_{iteration}/point_cloud.pickle').is_file():
            raise FileNotFoundError('Final Gaussian model missing')
        if not (model/f'ckpt/chkpnt{iteration}.pth').is_file(): raise FileNotFoundError('Final checkpoint missing')
    elif part=='test':
        status=json.loads((model/f'test/iter_{iteration}/evaluation_status.json').read_text())
        if status['status']!='completed' or not status['reconstruction']: raise ValueError(status)
        for key,expected in dict(iteration=iteration,n_train=len(meta['proj_train']),n_test=len(meta['proj_test']),
                                 render_train=True,render_test=True).items():
            if status.get(key)!=expected: raise ValueError(f'Evaluation {key}: {status.get(key)} != {expected}')
        array(model/f'test/iter_{iteration}/vol_pred.npy',meta['scanner']['nVoxel'])
        result['evaluation']=status
    log.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return result


def check_projection_metrics(out, split, iteration, expected_views, log, model=None):
    """Audit the real renderer output without imposing a convergence threshold."""
    import numpy as np
    import yaml
    model=model or out/'real/model'
    meta=json.loads((out/'real/dataset/meta_data.json').read_text())
    count=len(meta[f'proj_{split}'])
    if count!=expected_views or count==0:
        raise ValueError(f'{split} view count: {count}, expected {expected_views} nonempty views')
    path=model/f'test/iter_{iteration}/eval2d_render_{split}.yml'
    values=yaml.safe_load(path.read_text())
    if not isinstance(values,dict) or values.get('status')=='skipped':
        raise ValueError(f'Missing rendered metrics: {path}')
    result=dict(split=split,n_views=count,metrics=str(path))
    for metric in ('psnr','ssim'):
        scalar=np.asarray(values[f'{metric}_2d'],dtype=float)
        per_view=np.asarray(values[f'{metric}_2d_projs'],dtype=float)
        if scalar.shape!=() or per_view.shape!=(count,) or not np.isfinite(scalar).all() or not np.isfinite(per_view).all():
            raise ValueError(f'{metric}: expected finite scalar and ({count},) per-view metrics in {path}')
        result[metric]=dict(mean=float(scalar),min=float(per_view.min()),max=float(per_view.max()))
    result['note']='Execution/integrity check only; no universal PSNR/SSIM quality threshold.'
    log.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return result


def check_pose_artifacts(out, iterations, log):
    """Check real learned corrections, dataset binding, gauges and CSV export."""
    import csv
    import numpy as np
    import yaml
    from types import SimpleNamespace
    from r2_gaussian.pose.artifacts import validate_pose_file
    from r2_gaussian.pose.parallel_pose import PARALLEL_POSE_FORMAT_VERSION
    data=out/'real/dataset'
    model=out/'real/model'
    meta=json.loads((data/'meta_data.json').read_text())
    config=yaml.safe_load((model/'init_denoise/.pipeline_train_config.yml').read_text())
    cameras=[SimpleNamespace(image_name=Path(r['file_path']).name.split('.')[0],angle=r['angle'])
             for r in meta['proj_train']]
    scene=SimpleNamespace(source_path=str(data),scanner_cfg=meta['scanner'],train_cameras=cameras)
    specs={
        'delta_angle':('angle','angle_max_deg',.5,True),
        'global_shift':('global_detector_shift','detector_shift_max_pixel',3.,False),
        'view_shift':('per_view_detector_shift','detector_shift_max_pixel',3.,False),
        'beam_tilt':('beam_tilt','beam_tilt_max_deg',.5,True),
        'detector_roll':('detector_roll','detector_roll_max_deg',.5,True),
    }
    result=dict(snapshots={},note='Near-bound parameters are diagnostic, not proof of failed convergence.')
    for iteration in iterations:
        path=model/f'pose_calibration/iteration_{iteration}/pose_offsets.npz'
        order=validate_pose_file(path,scene,iteration)
        if order!=list(range(len(cameras))): raise ValueError(f'Pose snapshot view order changed: {path}')
        with np.load(path,allow_pickle=False) as saved:
            if int(saved['pose_format_version'])!=PARALLEL_POSE_FORMAT_VERSION:
                raise ValueError(f'Unsupported pose convention: {path}')
            stats={}
            for key,(switch,limit,default,radians) in specs.items():
                if config.get(f'parallel_optimize_{switch}',False) and key not in saved:
                    raise ValueError(f'Enabled pose parameter {key} missing: {path}')
                if key not in saved: continue
                value=saved[key]
                bound=float(config.get(f'parallel_{limit}',default))*(np.pi/180 if radians else 1.)
                tolerance=max(1e-7,abs(bound)*1e-5)
                if not np.isfinite(bound) or bound<0 or np.max(np.abs(value))>bound+tolerance:
                    raise ValueError(f'{key} exceeds configured bound {bound}: {path}')
                if key in ('delta_angle','view_shift') and np.max(np.abs(value.mean(axis=0)))>tolerance:
                    raise ValueError(f'{key} is not zero-mean: {path}')
                stats[key]=dict(max_abs=float(np.abs(value).max()),rms=float(np.sqrt(np.mean(value**2))),
                                bound=bound,near_bound=int(np.count_nonzero(np.abs(value)>=.95*bound)) if bound else 0)
            result['snapshots'][str(iteration)]=stats
    final=iterations[-1]
    folder=model/'pose_calibration'
    with np.load(folder/f'iteration_{final}/pose_offsets.npz',allow_pickle=False) as snapshot, \
         np.load(folder/'pose_offsets.npz',allow_pickle=False) as latest:
        if set(snapshot.files)!=set(latest.files): raise ValueError('Latest pose keys differ from final snapshot')
        for key in snapshot.files: np.testing.assert_array_equal(snapshot[key],latest[key],err_msg=key)
        with (folder/'pose_report.csv').open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        if len(rows)!=len(cameras): raise ValueError('Pose CSV view count mismatch')
        global_shift=snapshot['global_shift'] if 'global_shift' in snapshot else np.zeros(2)
        view_shift=snapshot['view_shift'] if 'view_shift' in snapshot else np.zeros((len(cameras),2))
        angles=snapshot['delta_angle'] if 'delta_angle' in snapshot else np.zeros(len(cameras))
        beam=snapshot['beam_tilt'] if 'beam_tilt' in snapshot else np.zeros(2)
        roll=snapshot['detector_roll'] if 'detector_roll' in snapshot else np.zeros(1)
        for index,row in enumerate(rows):
            if int(row['view'])!=index or row['image']!=cameras[index].image_name:
                raise ValueError('Pose CSV view identifiers/order mismatch')
            actual=[float(row[k]) for k in ('angle_delta_rad','shift_u_px','shift_v_px',
                                            'beam_tilt_u_rad','beam_tilt_v_rad','detector_roll_rad')]
            expected=[angles[index],*(global_shift+view_shift[index]),*beam,roll[0]]
            if not np.isfinite(actual).all(): raise ValueError('Nonfinite pose CSV values')
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-7)
    result.update(n_views=len(cameras),csv_matches_snapshot=True,latest_matches_snapshot=True)
    log.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    a=parser().parse_args()
    if a._preflight_config:
        cfg=json.loads(a._preflight_config.read_text())
        a.input_dir=Path(cfg['input_dir'])
        a.train_config=Path(cfg['train_config'])
        a.n_train,a.n_test=cfg['fbp_n_train'],cfg['fbp_n_test']
        a.no_pose=not cfg['train_pose_enabled']
        preflight(a,a._preflight_config.parent,cfg,a._preflight_config.parent/'preflight.json')
        return
    if a._chunk_check_config:
        cfg=json.loads(a._chunk_check_config.read_text())
        check_chunked_export(cfg,a._chunk_check_config.parent)
        return
    if a.iterations<6 or a.branch_iterations<6 or min(a.n_voxel+a.highres)<8:
        raise ValueError('iterations/branch-iterations >=6; voxel/highres dimensions >=8 required')
    if min(a.n_train,a.n_points,a.stripe_ncore,a.stripe_row_chunk)<1 or a.n_test<0:
        raise ValueError('Invalid counts')
    tag=dt.datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}'
    out=(a.run_root or Path('/root/fnvme/TEST_10.27/stripe_correct')/('parallel_validation_'+tag)).resolve()
    cfg=settings(a,out)
    if a.plan:
        print(json.dumps(dict(run_root=str(out),pipeline=cfg,synthetic=not a.skip_synthetic,
              real_side_branches=not a.skip_real_branches,highres=a.highres),indent=2))
        return
    out.mkdir(parents=True,exist_ok=False)
    os.chdir(ROOT)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    os.environ.setdefault('MPLBACKEND','Agg')
    os.environ.setdefault('PYTHONIOENCODING','utf-8')
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    r=Runner(out)
    print(f'RUN ROOT: {out}',flush=True)
    config=out/'pipeline.json'
    config.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    # Snapshot all first-party sources and the launch settings for later diagnosis.
    import shutil
    for path in list(ROOT.rglob('*')):
        relative=path.relative_to(ROOT)
        if out not in path.parents and path.is_file() and path.suffix in ('.py','.sh','.yaml','.yml') and not any(
            p in relative.parts for p in ('analysis','submodules','.git','__pycache__','.agents','.codex')):
            destination=out/'source_snapshot'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
    (out/'invocation.json').write_text(json.dumps(dict(argv=sys.argv,settings=vars(a)),default=str,indent=2))
    py=[sys.executable,'-u']
    try:
        run_validation(r,a,out,py,config)
    except KeyboardInterrupt:
        return r.finish(interrupted=True)
    except Exception as exc:
        # Orchestration errors must also leave a final failure report.
        def failed(log):
            raise exc
        r.stage('runner_error',failed)
    return r.finish()


def run_validation(r,a,out,py,config):
    r.command('00_preflight',py+['scripts/server_parallel_validation.py','--_preflight-config',config])
    r.command('01_real_cli_help',py+['test.py','--help'])
    r.stage('01_source_and_cli_checks',lambda log:static_checks(py,log))
    if a.skip_synthetic:
        r.skip('synthetic_regression_suite','--skip-synthetic explicitly selected')
    else:
        r.command('02_unittest',py+['-m','unittest','discover','-s','tests','-v'])
        r.command('03_physical_parallel',py+['scripts/validate_physical_projection.py']+([] if a.no_pose else ['--pose']))
        r.command('04_parallel_reconstruction',py+['scripts/probe_parallel_reconstruction.py'])
        r.command('05_tigre_offset_diagnostic',py+['scripts/probe_tigre_parallel_offsets.py'])
        r.command('06_orientation',py+['scripts/validate_data_flow_orientation.py','--output_dir',out/'synthetic/orientation'])
        r.command('07_synthetic_pipeline',py+['scripts/smoke_pipeline_integration.py','--output_dir',out/'synthetic/pipeline'])
        r.command('08_synthetic_branches',py+['scripts/smoke_training_integration.py','--source',out/'synthetic/pipeline/parallel/dataset',
                    '--init',out/'synthetic/pipeline/parallel/model/init_points.npy','--output_dir',out/'synthetic/training'],
                    requires=('07_synthetic_pipeline',))
        r.command('09_auxiliary',py+['scripts/smoke_auxiliary_integration.py','--source_run',out/'synthetic/pipeline','--output_dir',out/'synthetic/auxiliary'],
                    requires=('07_synthetic_pipeline',))
        if a.skip_cone: r.skip('10_cone_geometry','--skip-cone selected; synthetic pipeline still exercises its cone branch')
        else: r.command('10_cone_geometry',py+['scripts/smoke_cone_geometry.py','--output_dir',out/'synthetic/cone'])
    if a.upstream:
        r.command('11_official_comparison',py+['scripts/smoke_repository_checks.py','--upstream',a.upstream,'--output',out/'repository.json'])
    else:
        r.skip('11_official_comparison','Supply --upstream for an actual official source comparison on server')
    pipeline=py+['run_pipeline.py','--pipeline_config',config,'--train_config',a.train_config]
    # Keep start_from=center so successful earlier stages remain validated.
    # Denoised initialization requires init inside the execution window, so
    # center/FBP/init run together; their artifacts are then checked separately.
    r.command('20_center_fbp_init',pipeline+['--stop_after','init'])
    for index,part in ((20,'fbp'),(21,'init')):
        r.stage(f'{index}_{part}_artifacts',lambda log,part=part:check_real(out,part,a.iterations,log),
                requires=('20_center_fbp_init',))
    for index,part in ((22,'train'),(23,'test')):
        requires=('20_fbp_artifacts','21_init_artifacts') if part=='train' else ('22_train_artifacts',)
        r.command(f'{index}_{part}',pipeline+['--stop_after',part,'--resume'],requires=requires)
        r.stage(f'{index}_{part}_artifacts',lambda log,part=part:check_real(out,part,a.iterations,log),
                requires=(f'{index}_{part}',))
    r.stage('23_train_projection_metrics',lambda log:check_projection_metrics(out,'train',a.iterations,a.n_train,log),
            requires=('23_test',))
    if a.n_test:
        r.stage('23_heldout_projection_metrics',lambda log:check_projection_metrics(out,'test',a.iterations,a.n_test,log),
                requires=('23_test',))
    else:
        r.skip('23_heldout_projection_metrics','--n-test=0: no held-out projection quality evidence')
    if a.no_pose:
        r.skip('23_pose_artifacts','--no-pose explicitly selected')
    else:
        r.stage('23_pose_artifacts',lambda log:check_pose_artifacts(out,[max(2,a.iterations//2),a.iterations],log),
                requires=('22_train',))
    r.command('24_resume_once',pipeline+['--resume'],requires=('23_test',))
    r.command('25_resume_twice',pipeline+['--resume'],requires=('24_resume_once',))
    r.stage('25_cache_and_resume_safety',lambda log:check_cache_resume(a,out,config,log),requires=('25_resume_twice',))
    if a.skip_real_branches:
        r.skip('real_side_branches','--skip-real-branches explicitly selected')
    else:
        run_real_branches(r,a,out,py)
    r.command('30_highres',py+['export_highres.py','-s',out/'real/dataset','-m',out/'real/model',
               '--iteration',a.iterations,'--resolution',*a.highres,'--chunk_size',64,64,64,
               '--output',out/'real/highres/vol_pred.npy']+([] if a.no_pose else ['--pose_enabled']),
               requires=('22_train_artifacts',))
    r.stage('31_highres_and_mask',lambda log:check_highres_mask(out,py,log),requires=('30_highres',))
    r.command('32_real_chunk_consistency',py+['scripts/server_parallel_validation.py','--_chunk-check-config',config],
              requires=('30_highres',))


def prepare_real_branches(a,out,log):
    """Keep preparation inside a stage so missing config cannot abort the suite."""
    import yaml
    base_path=out/'real/model/init_denoise/.pipeline_train_config.yml'
    if not base_path.is_file():
        # The pipeline writes this file only when entering train. Independent
        # branches must also work when that stage was skipped or failed early.
        import run_pipeline as pipeline
        args=pipeline.resolve_args(pipeline.build_parser().parse_args([
            '--pipeline_config',str(out/'pipeline.json'),'--train_config',str(a.train_config)]))
        base_path=out/'real_branches_base.yaml'
        pipeline.write_train_config(args,base_path,out/'real/dataset',out/'real/model',
                                    out/'real/model/init_denoise/init_points.npy')
    base=yaml.safe_load(base_path.read_text())
    n=a.branch_iterations
    for name in ('ordinary','sparse'):
        config=dict(base,model_path=str(out/'real_branches'/name),iterations=n,pose_enabled=False,
                    slice_loss_enabled=name=='sparse',densify_until_iter=0,shape_enabled=False,
                    test_iterations=[n],save_iterations=[n],checkpoint_iterations=[n//2,n])
        if name=='sparse':
            config.update(slice_reference_path=str(out/'real/dataset/vol_sparse_fbp.npy'),slice_start_iter=1,
                slice_end_iter=n,slice_interval=1,slice_max_points=1024,slice_filter_ratio=10.,lambda_slice=.01)
        path=out/f'{name}_train.yaml'
        path.write_text(yaml.safe_dump(config),encoding='utf-8')
    # Resume the main pose/ordinary checkpoint in an independent model directory.
    resume=dict(base,model_path=str(out/'real_branches/resume'),iterations=a.iterations+2,
                test_iterations=[a.iterations+2],save_iterations=[a.iterations+2],
                checkpoint_iterations=[a.iterations+2],start_checkpoint=str(out/f'real/model/ckpt/chkpnt{a.iterations}.pth'))
    path=out/'checkpoint_resume.yaml'
    path.write_text(yaml.safe_dump(resume),encoding='utf-8')
    result=dict(ordinary=str(out/'ordinary_train.yaml'),sparse=str(out/'sparse_train.yaml'),resume=str(path))
    log.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def run_real_branches(r,a,out,py):
    """Independent real-data branches can still run when the main train fails."""
    r.stage('26_prepare_real_branches',lambda log:prepare_real_branches(a,out,log),requires=('21_init_artifacts',))
    for name in ('ordinary','sparse','resume'):
        n=a.iterations+2 if name=='resume' else a.branch_iterations
        train='27_real_checkpoint_resume' if name=='resume' else f'26_real_{name}'
        evaluation='28_real_resume_evaluation' if name=='resume' else f'26_real_{name}_evaluation'
        path=out/('checkpoint_resume.yaml' if name=='resume' else f'{name}_train.yaml')
        model=out/'real_branches'/name
        requires=('26_prepare_real_branches','22_train_artifacts') if name=='resume' else ('26_prepare_real_branches',)
        r.command(train,py+['train.py','--config',path],requires=requires)
        r.stage(train+'_artifacts',lambda log,n=n,model=model:check_real(out,'train',n,log,model=model),requires=(train,))
        r.command(evaluation,py+['test.py','-s',out/'real/dataset','-m',model,'--iteration',n],
                  requires=(train+'_artifacts',))
        r.stage(evaluation+'_artifacts',lambda log,n=n,model=model:check_real(out,'test',n,log,model=model),requires=(evaluation,))
        for split,count in (('train',a.n_train),('test',a.n_test)):
            stage=evaluation+f'_{split}_metrics'
            if count:
                r.stage(stage,lambda log,split=split,count=count,n=n,model=model:
                        check_projection_metrics(out,split,n,count,log,model=model),requires=(evaluation,))
            else:
                r.skip(stage,'--n-test=0: no held-out projections')


def static_checks(py,log):
    import shutil
    files=[p for p in ROOT.rglob('*.py') if not any(
        x in p.relative_to(ROOT).parts for x in ('analysis','submodules','.git','__pycache__'))]
    for p in files: compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
    entries=['run_pipeline.py','prepare_fbp_tiff.py','init_from_fbp.py','train.py','export_highres.py',
        'apply_mask_to_volume.py','run_sio2_experiment.py','scripts/prepare_sio2_experiments.py',
        'scripts/local_fbp_preview.py','scripts/validate_raw_orientation.py','scripts/scan_fbp_center.py',
        'scripts/scan_fbp_center_tomopy.py','scripts/sweep_fbp_pixel_size.py']
    with log.open('w',encoding='utf-8') as f:
        for entry in entries:
            subprocess.run(py+[str(ROOT/entry),'--help'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        bash=shutil.which('bash')
        if bash:
            for script in (ROOT/'scripts').glob('*.sh'):
                subprocess.run([bash,'-n',str(script)],stdout=f,stderr=subprocess.STDOUT,check=True)
    return dict(compiled=len(files),cli_help=len(entries),bash_syntax=bool(bash))


def check_cache_resume(a,out,config,log):
    import contextlib
    from types import SimpleNamespace
    from unittest import mock
    import init_from_fbp as initializer
    import run_pipeline as pipeline
    report=json.loads((out/'real/model/init_denoise/denoise_report.json').read_text())
    cached=Path(report['input']['volume_path'])
    before=(digest(cached),cached.stat().st_mtime_ns)
    args=SimpleNamespace(use_train_fbp=True,dataset=out/'real/dataset',volume=None,
                         fbp_output=cached,stripe_correction={})
    args.stripe_correction=json.loads(config.read_text())['stripe_correction']
    cli=['--pipeline_config',str(config),'--train_config',str(a.train_config),'--resume']
    with log.open('w',encoding='utf-8') as f,contextlib.redirect_stdout(f):
        with mock.patch.object(initializer,'reconstruct_from_train_views',side_effect=AssertionError('Valid cache recomputed')):
            initializer.load_initial_volume(args)
        if before!=(digest(cached),cached.stat().st_mtime_ns): raise AssertionError('Valid cache rewritten')
        with mock.patch.object(pipeline.subprocess,'run',side_effect=AssertionError('Resume relaunched a stage')):
            pipeline.main(pipeline.resolve_args(pipeline.build_parser().parse_args(cli)))
            changed=pipeline.resolve_args(pipeline.build_parser().parse_args(cli))
            changed.train_lambda_dssim=.123456789
            try: pipeline.main(changed)
            except ValueError as error:
                if 'Cannot resume train' not in str(error): raise
                print('Expected rejection:',error)
            else: raise AssertionError('Changed training config accepted')
            pipeline.main(pipeline.resolve_args(pipeline.build_parser().parse_args(cli)))
    return dict(cache_not_rewritten=True,resume_no_subprocess=True,changed_config_rejected=True,
                valid_resume_after_rejection=True)


def check_highres_mask(out,py,log):
    import numpy as np
    import SimpleITK as sitk
    folder=out/'real/highres'
    volume=np.load(folder/'vol_pred.npy',mmap_mode='r')
    if not all(np.isfinite(plane).all() for plane in volume):
        raise ValueError('High-resolution output contains NaN/Inf')
    image=sitk.ReadImage(str(folder/'vol_pred.nii.gz'))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image).transpose(2,1,0),volume)
    scanner=json.loads((out/'real/dataset/meta_data.json').read_text())['scanner']
    spacing=np.array(scanner['sVoxel'])/np.array(volume.shape)
    origin=np.array(scanner['offOrigin'])-np.array(scanner['sVoxel'])/2+spacing/2
    np.testing.assert_allclose(image.GetSpacing(),spacing,rtol=2e-5)
    np.testing.assert_allclose(image.GetOrigin(),origin,rtol=2e-5,atol=1e-8)
    np.testing.assert_allclose(np.array(image.GetDirection()).reshape(3,3),np.eye(3),atol=1e-6)
    mask=np.zeros(volume.shape,np.uint8)
    slices=tuple(slice(n//4,3*n//4) for n in volume.shape)
    mask[slices]=1
    mask_image=sitk.GetImageFromArray(mask.transpose(2,1,0))
    mask_image.CopyInformation(image)
    sitk.WriteImage(mask_image,str(folder/'test_box_mask.nii.gz'))
    command=list(map(str,py+['apply_mask_to_volume.py','--volume',folder/'vol_pred.npy',
          '--mask',folder/'test_box_mask.nii.gz','--reference',folder/'vol_pred.nii.gz',
          '--mask_axes','xyz','--output',folder/'masked.npy']))
    with log.open('w',encoding='utf-8') as f:
        subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    np.testing.assert_array_equal(np.load(folder/'masked.npy'),volume*mask)
    return dict(xyz_nifti_exact=True,physical_geometry=True,box_mask_exact=True,
                note='Generated box is a coordinate test, not a sample segmentation.')


def check_chunked_export(cfg,out):
    """Requery small windows of the real model on the exported global grid.

    A separate process isolates CUDA state. Windows cross chunk boundaries and
    include partial final tiles without repeating a full high-resolution export.
    """
    import numpy as np
    import torch
    from types import SimpleNamespace
    from export_highres import load_scanner_config
    from r2_gaussian.gaussian import GaussianModel, initialize_gaussian, query
    folder=out/'real/highres'
    metadata=json.loads((folder/'vol_pred.json').read_text())
    if metadata['iteration']!=cfg['train_iterations'] or metadata['pose_enabled']!=cfg['train_pose_enabled']:
        raise ValueError('High-resolution export iteration/backend does not match this run')
    volume=np.load(folder/'vol_pred.npy',mmap_mode='r')
    shape=np.asarray(volume.shape,dtype=int)
    if list(volume.shape)!=metadata['shape_xyz']: raise ValueError('Export shape differs from sidecar')
    scanner,_=load_scanner_config(cfg['output_dir'])
    size=np.asarray(scanner['sVoxel'],dtype=float)
    center=np.asarray(scanner['offOrigin'],dtype=float)
    spacing=size/shape
    chunks=np.asarray(metadata['query_chunk_size_xyz'],dtype=int)
    boundary=(shape//2//chunks)*chunks
    boundary=np.where((boundary==0)&(chunks<shape),chunks,boundary)
    starts=[np.zeros(3,dtype=int),np.maximum(0,boundary-8),np.maximum(0,((shape-17)//8)*8)]
    model=GaussianModel(None)
    initialize_gaussian(model,SimpleNamespace(model_path=cfg['model_path']),cfg['train_iterations'])
    pipe=SimpleNamespace(pose_enabled=cfg['train_pose_enabled'],compute_cov3D_python=False,debug=False)
    result=dict(rtol=5e-5,atol=5e-6,windows=[])
    for label,lower in zip(('first_tiles','chunk_seams','last_partial_tiles'),starts):
        upper=np.minimum(shape,lower+24) if label!='last_partial_tiles' else shape
        count=upper-lower
        local_center=center-size/2+(lower+count/2)*spacing
        with torch.inference_mode():
            reference=query(model,local_center.tolist(),count.tolist(),(count*spacing).tolist(),pipe)['vol'].cpu().numpy()
        actual=np.asarray(volume[tuple(slice(int(lo),int(hi)) for lo,hi in zip(lower,upper))])
        if reference.shape!=actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
            raise ValueError(f'Nonfinite values or wrong query shape in {label}')
        difference=np.abs(actual-reference)
        bad=int(np.count_nonzero(~np.isclose(actual,reference,rtol=result['rtol'],atol=result['atol'])))
        report=dict(window=label,start=lower.tolist(),stop=upper.tolist(),max_abs_error=float(difference.max()),
                    relative_l1_error=float(difference.sum()/max(float(np.abs(reference).sum()),1e-30)),
                    mismatched_voxels=bad,voxels=int(actual.size))
        result['windows'].append(report)
        print(json.dumps(report),flush=True)
    (folder/'chunk_consistency.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if any(w['mismatched_voxels'] for w in result['windows']):
        raise AssertionError('Real model chunk export differs from direct queries; see chunk_consistency.json')
    return result


if __name__=='__main__':
    sys.exit(main())
