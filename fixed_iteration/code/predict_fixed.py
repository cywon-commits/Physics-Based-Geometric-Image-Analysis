"""Frozen single-image deployment CLI. Input .npy is a boolean/0-1/+/-1 mask."""
import argparse
from study_core import *
def predict(mask,config):
 f=np.where(mask,1.,-1.).astype('float32');dim=f.ndim
 if dim not in [2,3]:raise ValueError('Only 2D or 3D arrays are supported')
 tic=time.perf_counter();n=initialize(f);initial=time.perf_counter()-tic;tic=time.perf_counter()
 for _ in range(config['selected_N']):n=one_step(n,f,**config['parameters'])
 field=time.perf_counter()-tic;tic=time.perf_counter();T=tensor(n,config['kernel_by_dimension'][str(dim)]);measure=time.perf_counter()-tic;value=float(np.trace(T)*config['calibration_by_dimension'][str(dim)])
 eig,vec=np.linalg.eigh(T)
 return n,dict(N=config['selected_N'],dimension=dim,geometry=value,units='pixels' if dim==2 else 'voxel^2',tensor=T.tolist(),tensor_axis=vec[:,0].tolist(),eigengap=float((eig[1]-eig[0])/max(abs(eig[-1]),1e-12)),axis_warning='Interpret minimum-eigenvalue axis only for connected elongated shapes; eigenvector sign is arbitrary.',initialization_seconds=initial,field_seconds=field,measurement_seconds=measure,total_seconds=initial+field+measure)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--config',type=Path,default=OUT/'selected_configuration.json');p.add_argument('--out',type=Path,required=True);a=p.parse_args();mask=np.load(a.input)>0;config=json.load(open(a.config));n,result=predict(mask,config);a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(result,indent=2));np.savez_compressed(a.out.with_suffix('.npz'),n=n,f=np.where(mask,1.,-1.).astype('float32'));print(json.dumps(result,indent=2))
