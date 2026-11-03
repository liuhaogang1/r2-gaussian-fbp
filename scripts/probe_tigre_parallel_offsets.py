from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare_fbp_tiff as p
from tigre.utilities.Ax import Ax
n=65
xyz=(np.indices((n,n,n)).transpose(1,2,3,0)+.5)/n*2-1
volume=np.exp(-.5*np.sum(((xyz-[0,.16,.18])/.08)**2,axis=-1)).astype(np.float32)
for dsd in [5,8,10]:
 for offset in [[0,0],[.12,0],[0,-.08],[.24,0],[0,-.16]]:
  args=p.build_parser().parse_args(['--input_dir','.','--output_dir','unused','--mode','parallel','--nVoxel',str(n),str(n),str(n),'--sVoxel','2','2','2','--pixel_subsample','1','--pixel_size','.04','--DSO','5','--DSD',str(dsd),'--offDetector',*map(str,offset)])
  geo=p.build_geometry(args,(96,112))
  for method in ['Siddon','interpolated']:
   image=Ax(np.ascontiguousarray(volume.transpose(2,1,0)),geo,np.array([0],np.float32),projection_type=method)[0,::-1,:]
   v,u=np.indices(image.shape)
   print(dsd,offset,method,[(image*u).sum()/image.sum(),(image*v).sum()/image.sum()],flush=True)
