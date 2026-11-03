"""Static/CLI coverage and preservation checks against supplied upstream."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,required=True)
    args=parser.parse_args()
    files=[p for p in ROOT.rglob('*.py') if not any(x in p.parts for x in ('analysis','submodules','.git','__pycache__'))]
    for path in files:
        compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
    results=dict(compiled_python_files=len(files),cli={},preserved_upstream={})
    entries=['run_pipeline.py','prepare_fbp_tiff.py','init_from_fbp.py','train.py','export_highres.py',
        'apply_mask_to_volume.py','run_sio2_experiment.py','scripts/prepare_sio2_experiments.py',
        'scripts/local_fbp_preview.py','scripts/validate_raw_orientation.py','scripts/scan_fbp_center.py',
        'scripts/scan_fbp_center_tomopy.py','scripts/sweep_fbp_pixel_size.py']
    for entry in entries:
        process=subprocess.run([sys.executable,str(ROOT/entry),'--help'],cwd=ROOT,capture_output=True,text=True)
        assert process.returncode==0,(entry,process.stderr)
        assert 'usage:' in process.stdout.lower(),entry
        results['cli'][entry]='help passed'
    # Keep production Python compatibility unchanged, but record the real CLI limit.
    process=subprocess.run([sys.executable,str(ROOT/'test.py'),'--help'],cwd=ROOT,capture_output=True,text=True)
    results['cli']['test.py']=dict(returncode=process.returncode,stderr=process.stderr.strip())
    if process.returncode:
        assert 'unsupported operand type(s) for |' in process.stderr,process.stderr
    upstream=args.upstream.resolve()
    relative=Path('r2_gaussian/gaussian/gaussian_model.py')
    def methods(path):
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GaussianModel')
        return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    original,current=methods(upstream/relative),methods(ROOT/relative)
    names=['setup_functions','densify_and_prune','densify_and_split','densify_and_clone']
    for name in names:
        assert original[name]==current[name],name
    results['preserved_upstream']['gaussian_methods']=names
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in ['r2_gaussian/utils/image_utils.py','r2_gaussian/utils/loss_utils.py']:
        assert ast.dump(ast.parse((upstream/relative).read_text()))==ast.dump(ast.parse((ROOT/relative).read_text())),relative
        results['preserved_upstream'][relative]='AST identical (line endings normalized)'
    cuda_files=[p for p in (ROOT/'r2_gaussian/submodules').rglob('*') if p.is_file() and p.suffix in ('.cu','.cuh') and 'third_party' not in p.parts]
    missing=[]
    same=0
    for path in cuda_files:
        relative=path.relative_to(ROOT)
        baseline=upstream/relative
        if not baseline.is_file():
            missing.append(str(relative))
            continue
        assert path.read_text()==baseline.read_text(),str(relative)
        same+=1
    results['preserved_upstream']['cuda']=dict(identical=same,not_in_official=missing)
    bash=Path(shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe')
    results['shell_scripts']={}
    if bash.exists():
        for path in (ROOT/'scripts').glob('*.sh'):
            process=subprocess.run([str(bash),'-n',str(path)],capture_output=True,text=True)
            assert process.returncode==0,(str(path),process.stderr)
            results['shell_scripts'][path.name]='syntax passed; external Linux data paths not executed'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
