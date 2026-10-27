from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
old = (root / 'scripts/fbp_geometry.py').read_text(encoding='utf-8')
start = old.index('def parse_svoxel')
end = old.index('def reconstruction_scanner')
(root / 'scripts/auto_svoxel.py').write_text(
    '"""Optional per-dataset FOV sizing, enabled only by sVoxel auto."""\n'
    'import argparse\nfrom pathlib import Path\nimport numpy as np\n\n\n'
    + old[start:end], encoding='utf-8')
gaussian_start = old.index('def reconstruction_scanner')
(root / 'scripts/fbp_geometry.py').write_text(
    '"""Optional geometry for the explicitly enabled Gaussian FBP mode."""\n'
    'from copy import deepcopy\nimport numpy as np\n\n'
    'COORDINATE_MODES = ("physical", "gaussian")\n\n\n'
    + old[gaussian_start:], encoding='utf-8')
for name in ('prepare_fbp_tiff.py', 'init_from_fbp.py', 'run_pipeline.py'):
    (root / name).write_bytes(subprocess.check_output(['git', 'show', '964783a:' + name]))
