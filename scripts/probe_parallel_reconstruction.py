from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare_fbp_tiff as p
from r2_gaussian.dataset.dataset_readers import angle2pose
n=65
center=np.array([.2,.15,.18])
angles=np.linspace(0,np.pi,90,endpoint=False).astype(np.float32)
yy,xx=np.indices((96,112))
for dsd in [5,8,10]:
 for offset in [[0,0],[.12,-.08]]:
  args=p.build_parser().parse_args(['--input_dir','.','--output_dir','unused','--mode','parallel','--nVoxel',str(n),str(n),str(n),'--sVoxel','2','2','2','--pixel_subsample','1','--pixel_size','.04','--DSO','5','--DSD',str(dsd),'--offDetector',*map(str,offset)])
  geo=p.build_geometry(args,(96,112))
  images=[]
  for a in angles:
   w2c=np.linalg.inv(angle2pose(5,a))
   local=w2c@np.r_[center,1]
   u=(xx-55.5)*.04+offset[0]
   v=(yy-47.5)*.04+offset[1]
   images.append(np.exp(-.5*((u-local[0])**2+(v-local[1])**2)/.1**2))
  result=p.reconstruct_volume(np.array(images,np.float32),angles,args,geo)
  xyz=(np.indices(result.shape).transpose(1,2,3,0)+.5)/n*2-1
  weight=np.maximum(result,0)*np.exp(-np.sum((xyz-center)**2,axis=-1)/.3**2)
  centroid=np.sum(xyz*weight[...,None],axis=(0,1,2))/weight.sum()
  np.testing.assert_allclose(centroid,center,atol=.002)
  print(dsd,offset,'centroid',centroid,'peak',xyz[np.unravel_index(result.argmax(),result.shape)],flush=True)
