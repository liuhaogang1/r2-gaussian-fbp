"""Compare real cone CUDA rendering, TIGRE projection and physical rays."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def centroid(image):
    v, u = np.indices(image.shape)
    return np.array([(image*u).sum(), (image*v).sum()]) / image.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from tigre.utilities.Ax import Ax
    import prepare_fbp_tiff as prepare
    from r2_gaussian.dataset.cameras import Camera
    from r2_gaussian.dataset.dataset_readers import angle2pose
    from r2_gaussian.gaussian import GaussianModel, render
    results = []
    prepared = []
    n = 65
    coords = (np.indices((n,n,n)).transpose(1,2,3,0)+.5)/n*2-1
    for offset in [[0.,0.], [.12,-.08]]:
        options = prepare.build_parser().parse_args(['--input_dir', str(out), '--output_dir', str(out/'unused'),
             '--mode', 'cone', '--pixel_size_kind', 'detector', '--pixel_subsample', '1',
             '--pixel_size', '.04', '--DSO', '5', '--DSD', '8', '--nVoxel', str(n),str(n),str(n),
             '--sVoxel', '2','2','2', '--offDetector', *map(str,offset)])
        for center in [[-.35,.16,.18],[.35,.16,.18]]:
            center = np.array(center, dtype=np.float32)
            sigma = .055
            volume = np.exp(-.5*np.sum(((coords-center)/sigma)**2,axis=-1)).astype(np.float32)
            geometry = prepare.build_geometry(options,(96,112))
            angles = np.array([0.,.8,1.6],np.float32)
            projections = Ax(np.ascontiguousarray(volume.transpose(2,1,0)),geometry,angles)[:,::-1,:]
            prepared.append((center, offset, geometry, angles, projections))
    # TIGRE manages its own CUDA context. Finish all Ax calls before creating
    # PyTorch CUDA tensors, as in the end-to-end preprocessing workflow.
    for center, offset, geometry, angles, projections in prepared:
            sigma = .055
            model = GaussianModel()
            model._xyz = torch.nn.Parameter(torch.tensor(center[None],device='cuda'))
            model._scaling = torch.nn.Parameter(torch.full((1,3),np.log(sigma),device='cuda'))
            model._rotation = torch.nn.Parameter(torch.tensor([[1.,0.,0.,0.]],device='cuda'))
            model._density = torch.nn.Parameter(torch.tensor([[np.log(np.expm1(1.))]],device='cuda',dtype=torch.float32))
            for angle, projection in zip(angles, projections):
                scanner = dict(sDetector=geometry.sDetector.tolist(),nDetector=[96,112],offDetector=offset)
                c2w = angle2pose(5.,angle)
                w2c = np.linalg.inv(c2w)
                camera = Camera(0,scanner,w2c[:3,:3].T,w2c[:3,3],angle,1,
                       2*np.arctan(geometry.sDetector[1]/16),2*np.arctan(geometry.sDetector[0]/16),
                       torch.zeros((1,96,112)), 'cone',0)
                np.testing.assert_allclose(camera.camera_center.detach().cpu(), c2w[:3,3],atol=1e-5)
                pipe = SimpleNamespace(pose_enabled=False,compute_cov3D_python=False,debug=False)
                with torch.no_grad():
                    image = render(camera,model,pipe)['render'][0].cpu().numpy()
                camera_point = w2c @ np.r_[center,1.]
                ray = camera_point[:2]*8/camera_point[2]-offset
                expected = ray/.04 + (np.array([112,96])-1)/2
                cuda_center, tigre_center = centroid(image), centroid(projection)
                print(dict(offset=offset, center=center.tolist(), angle=float(angle), cuda=cuda_center.tolist(), tigre=tigre_center.tolist(), expected=expected.tolist()), flush=True)
                np.testing.assert_allclose(cuda_center,expected,atol=.08)
                np.testing.assert_allclose(tigre_center,expected,atol=.35)
                np.testing.assert_allclose(cuda_center,tigre_center,atol=.35)
                results.append(dict(offset=offset,center=center.tolist(),angle=float(angle),
                                    cuda=cuda_center.tolist(),tigre=tigre_center.tolist(),analytic=expected.tolist()))
    (out/'results.json').write_text(json.dumps(results,indent=2))
    print('PASS',len(results),'cone cases; two depths, three angles, detector offsets')


if __name__ == '__main__':
    main()
