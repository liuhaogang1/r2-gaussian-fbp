"""Estimate relative slice scales; diagnostic registration, not ground truth."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import differential_evolution

OUT = Path(__file__).resolve().parent

def prepare(a):
    a = np.maximum(np.asarray(a, dtype=np.float64), 0)
    positive = a[a > 0]
    high = np.percentile(positive, 99) if positive.size else 1
    return gaussian_filter(np.clip(a / max(high, 1e-12), 0, 1), 1.5)[::2, ::2]

def fit(fbp, pred):
    a, b = prepare(fbp), prepare(pred)
    grid = np.indices(b.shape, dtype=float)
    center = (np.asarray(b.shape) - 1) / 2
    target = b.ravel() - b.mean()
    target_norm = np.linalg.norm(target)
    def warp(parameters):
        scales = np.asarray(parameters[:2])
        shift = np.asarray(parameters[2:])
        coordinates = (grid - center[:, None, None]) * scales[:, None, None] + center[:, None, None] + shift[:, None, None]
        return map_coordinates(a, coordinates, order=1, mode='constant', cval=0)
    def cost(parameters):
        values = warp(parameters).ravel()
        values = values - values.mean()
        return -float(values @ target / max(np.linalg.norm(values) * target_norm, 1e-12))
    result = differential_evolution(cost, [(0.3, 2.8), (0.3, 2.8), (-20, 20), (-20, 20)], seed=7, popsize=9, maxiter=100, tol=1e-5, polish=True)
    base = differential_evolution(lambda t: cost([1, 1, *t]), [(-20, 20), (-20, 20)], seed=7, popsize=7, maxiter=45)
    return {'fbp_to_gaussian_size_ratio_axes': result.x[:2].tolist(), 'sample_shift_fullres_pixels': (result.x[2:] * 2).tolist(), 'ncc_after_scale': -result.fun, 'ncc_translation_only': -base.fun}, warp(result.x), b

def main():
    paths = sorted(OUT.glob('cata*_planes.npz'))
    fig, axes = plt.subplots(len(paths), 3, figsize=(11, 3 * len(paths)), layout='constrained')
    report = []
    for row, path in enumerate(paths):
        planes = np.load(path)
        item = {'case': path.stem.removesuffix('_planes'), 'planes': []}
        for axis in range(3):
            result, aligned, pred = fit(planes[f'FBP_{axis}'], planes[f'Gaussian_{axis}'])
            result['normal_ijk_axis'] = axis
            item['planes'].append(result)
            rgb = np.stack([aligned, pred, pred], axis=-1)
            axes[row, axis].imshow(rgb.transpose(1, 0, 2), origin='lower')
            sx, sy = result['fbp_to_gaussian_size_ratio_axes']
            axes[row, axis].set_title(f"{item['case']} axis {axis} | scale {sx:.3f}, {sy:.3f}\nNCC {result['ncc_translation_only']:.3f} -> {result['ncc_after_scale']:.3f}", fontsize=9)
            axes[row, axis].axis('off')
        report.append(item)
        print(json.dumps(item), flush=True)
    fig.savefig(OUT / 'registered_overlays.png', dpi=110)
    plt.close(fig)
    (OUT / 'slice_scale_fits.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
