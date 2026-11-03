"""Real small-data experiment, cache and command-line utility checks."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source_run',type=Path,required=True)
    parser.add_argument('--output_dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    source=args.source_run.resolve()
    results={}
    def run(script, arguments):
        command=[sys.executable,str(ROOT/script),*map(str,arguments)]
        print('+',*command,flush=True)
        return subprocess.run(command,cwd=ROOT,check=True,capture_output=True,text=True)
    projections=np.load(source/'parallel_projections.npy')
    raw=out/'intensity_raw'
    raw.mkdir()
    flat=np.full(projections.shape[1:],1000,np.float32)
    tifffile.imwrite(raw/'Flat_0.tiff',flat)
    for i,image in enumerate(projections):
        tifffile.imwrite(raw/f'Tomo_{i}.tiff',10+990*np.exp(-image))
    (raw/'scan.txt').write_text('NumberImages=19\nAngleInterval=10\n')
    common=['--input_dir',raw,'--pixel_subsample','1','--pixel_size','.05',
        '--projection_scale','1','--nVoxel','33','35','37','--sVoxel','2','2','2',
        '--offDetector','.1','0','--DSO','5','--DSD','8','--n_test','6','--train_views','12','--sparse_views','6',
        '--dark_value','10']
    experiment=out/'experiment'
    run('scripts/prepare_sio2_experiments.py',[*common,'--output_dir',experiment,'--preprocess_only'])
    corrected=np.load(experiment/'corrected_all.npy')
    np.testing.assert_allclose(corrected,projections,rtol=2e-4,atol=3e-7)
    cached=(experiment/'corrected_all.npy').stat().st_mtime_ns
    run('scripts/prepare_sio2_experiments.py',[*common,'--output_dir',experiment])
    assert (experiment/'corrected_all.npy').stat().st_mtime_ns==cached
    for name in ['vol_full_fbp.npy','vol_sparse75_fbp.npy','vol_fbp.npy']:
        assert np.load(experiment/name).shape==(33,35,37)
        assert np.isfinite(np.load(experiment/name)).all()
    results['experiment_full_sparse_and_intensity']=True
    ordinary=out/'preprocessed'
    prep=['--input_dir',raw,'--output_dir',ordinary,'--projection_glob','Tomo_*.tiff',
        '--flat_glob','Flat_*.tiff','--flat_field','--dark_value','10','--input_type','transmission',
        '--i0_mode','fixed','--i0','1','--clip_percentile','100','--shift_v','0',
        '--pixel_subsample','1','--projection_scale','1','--preprocess_only']
    run('prepare_fbp_tiff.py',prep)
    np.testing.assert_allclose(np.load(ordinary/'proj_preprocessed_all.npy'),projections,rtol=2e-4,atol=3e-7)
    assert len(np.load(ordinary/'angles_unique.npy'))==18
    results['shared_flat_preprocess_and_duplicate_endpoint']=True
    dataset=source/'parallel/dataset'
    init=['--dataset',dataset,'--use_train_fbp','--fbp_output',out/'train_cache',
        '--n_points','200','--density_thresh','.03','--output',out/'train_points.npy','--visualize_fbp_nii']
    first=run('init_from_fbp.py',init)
    assert 'Reconstruct train-view FBP' in first.stdout
    cache=(out/'train_cache.npy').stat().st_mtime_ns
    second=run('init_from_fbp.py',init)
    assert 'Load cached' in second.stdout
    assert (out/'train_cache.npy').stat().st_mtime_ns==cache
    run('init_from_fbp.py',['--volume',experiment/'vol_full_fbp.npy','--output',out/'custom_points',
        '--n_points','100','--density_thresh','.03','--visualize','--visualize_slices'])
    assert np.load(out/'custom_points.npy').shape==(100,4)
    results['train_fbp_cache_and_custom_init']=True
    run('scripts/local_fbp_preview.py',[dataset/'vol_fbp.npy',out/'local_preview','--min_component','2'])
    results['local_preview_cli']=True
    rng=np.random.default_rng(7)
    raw_values=rng.integers(0,256,(13,13,13),dtype=np.uint8)
    reference=np.flip(raw_values.transpose(2,0,1),axis=1).astype(np.float32)
    raw_values.tofile(out/'volume.raw')
    np.save(out/'reference.npy',reference)
    run('scripts/validate_raw_orientation.py',['--raw',out/'volume.raw','--raw_shape','13','13','13',
        '--reference',out/'reference.npy','--output',out/'oriented.npy','--score_shape','13','13','13'])
    orientation=json.loads((out/'oriented_orientation.json').read_text())
    assert orientation['score']>.99999,orientation
    results['raw_96_orientation_candidates']=orientation
    # Execute the legacy converter only after AST-rebinding its hardcoded paths.
    tree=ast.parse((ROOT/'tif2png8bit.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ('input_dir','output_dir'):
                node.value=ast.Constant(str(raw if node.targets[0].id=='input_dir' else out/'png8'))
    exec(compile(ast.fix_missing_locations(tree),'tif2png8bit.py','exec'),{'__name__':'__main__'})
    assert len(list((out/'png8').glob('*.png')))==1
    results['legacy_converter_isolated_one_image']=True
    import run_sio2_experiment as wrapper
    commands=[]
    for experiment_name in ('full_init','sparse_fbp_init'):
        argv=['wrapper','--experiment',experiment_name,'--dataset',str(experiment),'--model_path',str(out/'unused_model')]
        if experiment_name=='full_init': argv+=['--init_volume',str(experiment/'vol_full_fbp.npy')]
        with mock.patch.object(sys,'argv',argv),mock.patch.object(wrapper,'run',side_effect=lambda c:commands.append(c)):
            wrapper.main(wrapper.parse_args())
    assert len(commands)==6
    results['experiment_dispatch_both_choices_mocked']=True
    launcher=source/'evaluation_launcher.py'
    import shutil
    skip_model=out/'skip_model'
    shutil.copytree(source/'parallel/model/point_cloud',skip_model/'point_cloud')
    shutil.copy2(source/'parallel/model/cfg_args',skip_model/'cfg_args')
    subprocess.run([sys.executable,str(launcher),str(ROOT/'test.py'),'-s',str(dataset),
        '-m',str(skip_model),'--iteration','6',
        '--skip_render_train','--skip_render_test','--skip_recon'],cwd=ROOT,check=True)
    status=json.loads((skip_model/'test/iter_6/evaluation_status.json').read_text())
    assert status['status']=='completed' and not any(status[k] for k in ('render_train','render_test','reconstruction'))
    results['evaluation_all_skip_status']=True
    (out/'results.json').write_text(json.dumps(results,indent=2))
    print('PASS auxiliary integrations',json.dumps(results),flush=True)


if __name__=='__main__':
    main()
