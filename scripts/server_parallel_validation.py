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
        self.results['stages'].append(dict(name=name,status='skipped',reason=reason))
        self.save()

    def stage(self, name, action):
        record=dict(name=name,status='running',log=str(self.out/'logs'/f'{name}.log'))
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
            record.update(status='failed',error=str(exc))
            self.results['status']='failed'
            raise
        finally:
            record['seconds']=time.monotonic()-start
            self.save()
            print(f"[{record['status'].upper()}] {name} ({record['seconds']:.1f}s) | {record['log']}",flush=True)

    def command(self, name, command):
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
        self.stage(name,execute)


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


def check_real(out, part, iteration, log):
    import numpy as np
    data=out/'real/dataset'
    model=out/'real/model'
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
        if not (model/f'ckpt/chkpnt{iteration}.pth').is_file(): raise FileNotFoundError('Final checkpoint missing')
    elif part=='test':
        status=json.loads((model/f'test/iter_{iteration}/evaluation_status.json').read_text())
        if status['status']!='completed' or not status['reconstruction']: raise ValueError(status)
        array(model/f'test/iter_{iteration}/vol_pred.npy',meta['scanner']['nVoxel'])
        result['evaluation']=status
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
                    '--init',out/'synthetic/pipeline/parallel/model/init_points.npy','--output_dir',out/'synthetic/training'])
        r.command('09_auxiliary',py+['scripts/smoke_auxiliary_integration.py','--source_run',out/'synthetic/pipeline','--output_dir',out/'synthetic/auxiliary'])
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
        r.stage(f'{index}_{part}_artifacts',lambda log,part=part:check_real(out,part,a.iterations,log))
    for index,part in ((22,'train'),(23,'test')):
        r.command(f'{index}_{part}',pipeline+['--stop_after',part,'--resume'])
        r.stage(f'{index}_{part}_artifacts',lambda log,part=part:check_real(out,part,a.iterations,log))
    r.command('24_resume_once',pipeline+['--resume'])
    r.command('25_resume_twice',pipeline+['--resume'])
    r.stage('25_cache_and_resume_safety',lambda log:check_cache_resume(a,out,config,log))
    if a.skip_real_branches:
        r.skip('real_side_branches','--skip-real-branches explicitly selected')
    else:
        run_real_branches(r,a,out,py)
    r.command('30_highres',py+['export_highres.py','-s',out/'real/dataset','-m',out/'real/model',
               '--iteration',a.iterations,'--resolution',*a.highres,'--chunk_size',64,64,64,
               '--output',out/'real/highres/vol_pred.npy']+([] if a.no_pose else ['--pose_enabled']))
    r.stage('31_highres_and_mask',lambda log:check_highres_mask(out,py,log))
    r.results.update(status='completed',finished=dt.datetime.now().astimezone().isoformat(),
                     note='Smoke validation only; metrics against FBP are not independent ground truth.')
    r.save()
    print(f'\nCOMPLETE: {out}/summary.tsv\nMain model: {out}/real/model\nHighres: {out}/real/highres',flush=True)


def run_real_branches(r,a,out,py):
    """Use real projections, while keeping incompatible options in separate models."""
    import yaml
    # Fully resolved training config records physical scale and model/data paths.
    base=yaml.safe_load((out/'real/model/init_denoise/.pipeline_train_config.yml').read_text())
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
        r.command(f'26_real_{name}',py+['train.py','--config',path])
        r.command(f'26_real_{name}_evaluation',py+['test.py','-s',out/'real/dataset','-m',out/'real_branches'/name,'--iteration',n])
    # Resume the main pose/ordinary checkpoint in an independent model directory.
    resume=dict(base,model_path=str(out/'real_branches/resume'),iterations=a.iterations+2,
                test_iterations=[a.iterations+2],save_iterations=[a.iterations+2],
                checkpoint_iterations=[a.iterations+2],start_checkpoint=str(out/f'real/model/ckpt/chkpnt{a.iterations}.pth'))
    path=out/'checkpoint_resume.yaml'
    path.write_text(yaml.safe_dump(resume),encoding='utf-8')
    r.command('27_real_checkpoint_resume',py+['train.py','--config',path])
    r.command('28_real_resume_evaluation',py+['test.py','-s',out/'real/dataset','-m',out/'real_branches/resume','--iteration',a.iterations+2])


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


if __name__=='__main__':
    main()
