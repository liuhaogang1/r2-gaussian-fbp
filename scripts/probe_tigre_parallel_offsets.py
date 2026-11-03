"""Report the installed TIGRE projector's parallel detector-offset behavior."""
import inspect
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def projector_cases(projector):
    """Only select methods through a parameter the installed Ax declares.

    Older installations expose krylov='ray-voxel'; its method names need not
    match projection_type. Probe their installed default without guessing.
    """
    if 'projection_type' in inspect.signature(projector).parameters:
        return [(method, {'projection_type': method}) for method in ('Siddon', 'interpolated')]
    return [('installed_default', {})]


def main():
    import prepare_fbp_tiff as prepare
    from tigre.utilities.Ax import Ax

    cases=projector_cases(Ax)
    print(f'Ax signature: {inspect.signature(Ax)}',flush=True)
    if cases[0][0]=='installed_default':
        print('No projection_type parameter: probing installed default only; '
              'Siddon/interpolated selector comparison is unavailable.',flush=True)
    n=65
    xyz=(np.indices((n,n,n)).transpose(1,2,3,0)+.5)/n*2-1
    volume=np.exp(-.5*np.sum(((xyz-[0,.16,.18])/.08)**2,axis=-1)).astype(np.float32)
    volume=np.ascontiguousarray(volume.transpose(2,1,0))
    angles=np.array([0],np.float32)
    for dsd in (5,8,10):
        for offset in ([0,0],[.12,0],[0,-.08],[.24,0],[0,-.16]):
            args=prepare.build_parser().parse_args([
                '--input_dir','.','--output_dir','unused','--mode','parallel',
                '--nVoxel',str(n),str(n),str(n),'--sVoxel','2','2','2',
                '--pixel_subsample','1','--pixel_size','.04','--DSO','5',
                '--DSD',str(dsd),'--offDetector',*map(str,offset)])
            geo=prepare.build_geometry(args,(96,112))
            for method,kwargs in cases:
                image=Ax(volume,geo,angles,**kwargs)[0,::-1,:]
                mass=float(image.sum())
                if image.shape!=(96,112) or not np.isfinite(image).all() or not np.isfinite(mass) or mass<=0:
                    raise ValueError(f'Invalid TIGRE image: DSD={dsd}, offset={offset}, method={method}')
                v,u=np.indices(image.shape)
                centroid=[float((image*u).sum()/mass),float((image*v).sum()/mass)]
                print(f'DSD={dsd} offDetector={offset} method={method} '
                      f'TIGRE_geo_offset={np.asarray(geo.offDetector).tolist()} '
                      f'centroid_uv={centroid}',flush=True)


if __name__=='__main__':
    main()
