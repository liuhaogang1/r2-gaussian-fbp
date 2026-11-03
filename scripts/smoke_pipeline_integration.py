"""Real small-data pipeline, FDK and experiment integration checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock
import numpy as np
import tifffile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--generate_only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not args.generate_only:
        out.mkdir(parents=True, exist_ok=False)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output_dir', str(out), '--generate_only'], check=True)
        return run_pipeline_checks(out)
    import prepare_fbp_tiff as prepare
    from tigre.utilities.Ax import Ax
    results = {}
    coords = (np.indices((33,35,37)).transpose(1,2,3,0)+.5)/[33,35,37]*2-1
    volume = np.zeros((33,35,37), np.float32)
    for center, sigma, density in [([-.32,.2,.12],.16,1),([.35,-.3,-.28],.12,.8),([-.05,.1,.42],.13,.6)]:
        volume += density*np.exp(-.5*np.sum(((coords-center)/sigma)**2,axis=-1))
    # Finish TIGRE work before any PyTorch CUDA use; stages run in subprocesses.
    for mode in ('parallel','cone'):
        raw = out / (mode+'_raw')
        raw.mkdir()
        options = prepare.build_parser().parse_args(['--input_dir',str(raw),'--output_dir',str(out/'unused'),
            '--mode',mode,'--pixel_size_kind','detector','--pixel_subsample','1','--pixel_size','.05',
            '--DSO','5','--DSD','8','--nVoxel','33','35','37','--sVoxel','2','2','2',
            '--offDetector','.1', '0' if mode=='parallel' else '-.05'])
        angles = np.deg2rad(np.arange(19)*10 if mode=='parallel' else np.arange(24)*15).astype(np.float32)
        geometry = prepare.build_geometry(options,(65,65))
        if mode == 'cone':
            projections = Ax(np.ascontiguousarray(volume.transpose(2,1,0)),geometry,angles)[:,::-1,:]
        else:
            # Independent parallel line integrals: TIGRE 2.2 Ax applies
            # version-specific distance factors to nonzero detector offsets.
            # Do not use that forward projector as a physical truth source.
            from r2_gaussian.dataset.dataset_readers import angle2pose
            vv, uu = np.indices((65,65))
            projections = []
            for angle in angles:
                w2c = np.linalg.inv(angle2pose(5., angle))
                image = np.zeros((65,65), np.float32)
                for center, sigma, density in [([-.32,.2,.12],.16,1),([.35,-.3,-.28],.12,.8),([-.05,.1,.42],.13,.6)]:
                    local = w2c @ np.r_[center,1.]
                    image += density*np.sqrt(2*np.pi)*sigma*np.exp(-.5*(((uu-32)*.05+.1-local[0])**2+((vv-32)*.05-local[1])**2)/sigma**2)
                projections.append(image)
            projections = np.array(projections, np.float32)
        for i, image in enumerate(projections):
            tifffile.imwrite(raw / f'view_{i}.tiff',image)
        np.save(out/(mode+'_projections.npy'),projections)


def run_pipeline_checks(out):
    import run_pipeline as pipeline
    results = {}
    launcher = out/'evaluation_launcher.py'
    launcher.write_text("import __future__,sys\nfrom pathlib import Path\np=Path(sys.argv.pop(1)).resolve()\nsys.path.insert(0,str(p.parent))\nsys.argv[0]=str(p)\nexec(compile(p.read_text(),str(p),'exec',flags=__future__.annotations.compiler_flag),{'__name__':'__main__','__file__':str(p)})\n")
    real_run = subprocess.run
    launched = []
    def run(command, **kwargs):
        launched.append(list(map(str,command)))
        if Path(command[1]).name == 'test.py':
            command = [command[0], str(launcher), *command[1:]]
        return real_run(command, **kwargs)
    for mode in ('parallel','cone'):
        config = dict(input_dir=str(out/(mode+'_raw')),output_dir=str(out/mode/'dataset'),
            model_path=str(out/mode/'model'),fbp_input_type='line_integral',fbp_clip_percentile=100,
            fbp_projection_scale=1,fbp_pixel_subsample=1,fbp_pixel_size=.05,fbp_nVoxel=[33,35,37],
            fbp_sVoxel=[2,2,2],fbp_n_train=12,fbp_n_test=6 if mode=='parallel' else 0,
            fbp_DSO=5,fbp_DSD=8,fbp_mode=mode,fbp_angle_interval=10 if mode=='parallel' else 15,
            init_n_points=200,init_density_thresh=.03,init_visualize=False,init_visualize_slices=True,
            train_iterations=6,train_lambda_tv=0,train_densify_until_iter=0,train_test_iterations=[6],
            train_checkpoint_iterations=[3,6],test_iteration=6,
            init_output=str(out/mode/'model'/'init_points'))
        if mode=='parallel':
            config.update(center_slice_margin=3,center_slice_step=4,center_pair_min_offset_px=-5,
                center_pair_max_offset_px=5,center_pair_min_peak_prominence=.00001,
                fbp_sparse_views=6,fbp_sparse_fbp_output='sparse_volume',fbp_recon_pixel_subsample=2,
                init_denoise=True,init_denoise_save_nii=True)
        else:
            config.update(manual_off_detector=[.1,-.05],fbp_pixel_size_kind='detector')
        file = out/(mode+'_config.json')
        file.write_text(json.dumps(config))
        def resolved(resume=False):
            return pipeline.resolve_args(pipeline.build_parser().parse_args(['--pipeline_config',str(file)]+(['--resume'] if resume else [])))
        with mock.patch.object(pipeline.subprocess,'run',run):
            pipeline.main(resolved())
            before=len(launched)
            pipeline.main(resolved(True))
            pipeline.main(resolved(True))
            assert len(launched)==before, 'Resume launched a subprocess'
            altered=resolved(True)
            altered.train_lambda_dssim=.123
            try:
                pipeline.main(altered)
            except ValueError as error:
                assert 'Cannot resume train' in str(error), str(error)
            else:
                raise AssertionError('Changed training configuration accepted')
            pipeline.main(resolved(True))
        dataset=out/mode/'dataset'
        metadata=json.loads((dataset/'meta_data.json').read_text())
        evaluation=json.loads((out/mode/'model/test/iter_6/evaluation_status.json').read_text())
        assert evaluation['status']=='completed' and evaluation['n_test']==len(metadata['proj_test'])
        assert metadata['source']['geometry_version']==2
        assert len(metadata['source']['projection_files'])==len(metadata['source']['angles_all'])
        assert (out/mode/'model/init_points.npy').is_file()
        if mode=='parallel':
            center=json.loads((out/mode/'dataset_center.json').read_text())
            assert center['center_valid'], center
            assert abs(center['offset_px']-2)<.1, center['offset_px']
            assert (dataset/'sparse_volume.npy').is_file()
            results['center_offset_px']=center['offset_px']
        results[mode]=dict(real_pipeline=True,repeated_resume=True,changed_config_rejected=True,
            n_train=len(metadata['proj_train']),n_test=len(metadata['proj_test']),
            eval_annotation_launcher_only=True)
        (out/'results.json').write_text(json.dumps(results,indent=2))
    print('PASS real center/FBP/FDK/init/train/test and repeated resume',flush=True)


if __name__=='__main__':
    main()
