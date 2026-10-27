"""CPU-only analysis of the supplied FBP/Gaussian NIfTI pairs."""
import gzip
import json
import struct
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SOURCE = Path('D:/TEM/data/output')
OUT = Path(__file__).resolve().parent

def read_volume(path):
    with gzip.open(path, 'rb') as stream:
        header = stream.read(352)
        shape = struct.unpack_from('<8h', header, 40)[1:4]
        offset = int(struct.unpack_from('<f', header, 108)[0])
        datatype = struct.unpack_from('<h', header, 70)[0]
        if datatype != 16:
            raise ValueError(f'Expected float32: {path}')
        if offset > 352:
            stream.read(offset - 352)
        data = np.frombuffer(stream.read(int(np.prod(shape)) * 4), dtype='<f4')
    # NIfTI I is fastest; use file I/J/K for both paired volumes.
    volume = data.reshape(shape, order='F')
    return volume, {
        'path': str(path), 'shape_ijk': list(shape),
        'spacing_ijk': list(struct.unpack_from('<8f', header, 76)[1:4]),
        'sform': [list(struct.unpack_from('<4f', header, off)) for off in (280, 296, 312)],
    }

def normalize(plane):
    values = plane[np.isfinite(plane) & (plane > 0)]
    high = np.percentile(values, 99) if values.size else 1
    return np.clip(plane / max(high, 1e-10), 0, 1)

def main():
    pairs = []
    for folder in sorted(SOURCE.glob('cata*')):
        gt = sorted(folder.glob('vol_gt*.nii.gz'))
        pred = sorted(folder.glob('vol_pred*.nii.gz'))
        if gt and pred:
            pairs.append((folder.name, gt[0], pred[0]))
    fig, axes = plt.subplots(len(pairs), 6, figsize=(17, 3 * len(pairs)), layout='constrained')
    report = []
    for row, (name, gt_path, pred_path) in enumerate(pairs):
        gt, gt_meta = read_volume(gt_path)
        pred, pred_meta = read_volume(pred_path)
        small = {}
        for axis in range(3):
            for offset, (tag, volume) in enumerate((('FBP', gt), ('Gaussian', pred))):
                plane = np.take(volume, volume.shape[axis] // 2, axis=axis)
                ax = axes[row, axis * 2 + offset]
                ax.imshow(normalize(plane).T, origin='lower', cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'{name} | {tag} | IJK[{axis}] mid', fontsize=10)
                ax.set_xticks([0, 64, 128, 192, 255])
                ax.set_yticks([])
                small[f'{tag}_{axis}'] = plane.copy()
        np.savez_compressed(OUT / f'{name}_planes.npz', **small)
        report.append({'case': name, 'fbp': gt_meta, 'gaussian': pred_meta})
        print(name, flush=True)
    fig.savefig(OUT / 'paired_center_slices.png', dpi=120)
    plt.close(fig)
    (OUT / 'headers.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
