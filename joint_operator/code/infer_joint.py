"""Frozen deployment. Default refuses to imply a validated upgrade when constraints failed."""
from joint_study_core import *
import argparse

def predict(mask,frozen,experimental=False):
 dim=mask.ndim;candidate=frozen['selection'][str(dim)]
 gate=json.load(open(OUT/'deployment_gate.json')) if (OUT/'deployment_gate.json').exists() else {}
 accepted=bool(candidate['recommended'] and gate.get('by_dimension',{}).get(str(dim),{}).get('production_accepted',False))
 if experimental or accepted:
  cfg=candidate['generator'];N=candidate['N'];o=candidate['observer'];coef=candidate['calibration'];mode='joint_candidate'
 else:
  cfg=parameter_config();N=30;o=dict(p=BASE['kernel_by_dimension'][str(dim)],observer='gradient',mix=[1,0,0],which='min');coef=BASE['calibration_by_dimension'][str(dim)];mode='PBGIA30_fallback'
 f=np.where(mask,1.,-1.).astype('float32');tic=time.perf_counter();n=initialize(f);init=time.perf_counter()-tic;tic=time.perf_counter()
 for _ in range(N):n=step(n,f,cfg)
 field=time.perf_counter()-tic;tic=time.perf_counter();T=tensor(n,o['p']);B,V,G=orientation_tensors(n,cfg,T);Q=sum(w*normalized_tensor(a) for w,a in zip(o['mix'],[G,B,V])) if o['observer']=='ensemble' else dict(gradient=G,bond=B,direct=V)[o['observer']];eig,vec=np.linalg.eigh(Q);idx=0 if o['which']=='min' else -1;gap=(eig[1]-eig[0] if idx==0 else eig[-1]-eig[-2])/max(np.linalg.norm(eig),1e-14);measure=time.perf_counter()-tic
 segmentation=n[...,-1]>0;inputdice=2*(segmentation&mask).sum()/max(segmentation.sum()+mask.sum(),1);border=np.ones(mask.shape,bool);border[(slice(4,-4),)*dim]=False
 warnings=['Tensor axis is a director (sign arbitrary); spheres, rings, tori and separated spheres have no prescribed global axis here.','Input-mask fidelity is not a certificate of clean ground-truth shape preservation.']
 if not accepted:warnings.append('The strict clean-Dice/topology validation target was not met by the learned candidate; no production upgrade is recommended.')
 if n[...,-1][border].max()>=0:warnings.append('Insufficient negative exterior collar; topology compactification is not justified.')
 if gap<1e-3:warnings.append('Orientation eigenaxis is ill-conditioned/undefined at the declared eigengap threshold.')
 return n,dict(mode=mode,N=N,dimension=dim,geometry=float(np.trace(T)*coef),unit='pixels' if dim==2 else 'voxel^2',axis=vec[:,idx].tolist(),eigengap=float(gap),orientation_defined=bool(gap>=1e-3),input_dice=float(inputdice),initialization_seconds=init,field_seconds=field,measurement_seconds=measure,total_seconds=init+field+measure,warnings=warnings,generator=cfg,measurement_kernel=o['p'],observer=o['observer'],which=o['which'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mask',type=Path);p.add_argument('--selection',type=Path,default=OUT/'pretest_selection.json');p.add_argument('--experimental',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args();mask=np.load(a.mask)>0;frozen=json.load(open(a.selection));n,result=predict(mask,frozen,a.experimental);a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(result,indent=2));np.savez_compressed(a.out.with_suffix('.npz'),n=n,f=np.where(mask,1.,-1.).astype('float32'));print(json.dumps(result,indent=2))
