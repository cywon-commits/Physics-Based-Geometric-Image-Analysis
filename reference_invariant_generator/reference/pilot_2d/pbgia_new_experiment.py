"""PBGIA generator learning pilot. Original notebook plus exact 4th-order baseline.
Uses a nonseparable odd-in-x/even-in-y 5x5 kernel, unnormalized; fixed NN exchange.
No Dice or topology objectives. Train-only reference circle for geometric calibration.
"""
import os,time,json,csv
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.special import ellipe
from scipy.optimize import minimize
import torch
import torch.nn.functional as F
out=Path('/mnt/data/pbgia_fresh_results');out.mkdir(exist_ok=True)
torch.set_num_threads(2);np.random.seed(51117);torch.manual_seed(51117)
N=10;S=72;X,Y=np.meshgrid(np.arange(S)-(S-1)/2,np.arange(S)-(S-1)/2)
# coefficient ordering (|row displacement| = 0,1,2) x (column displacement =1,2)
basis=[]
for ry in range(3):
 for rx in (1,2):
  k=np.zeros((5,5),np.float32)
  for iy in ([0] if ry==0 else [-ry,ry]): k[2+iy,2+rx]=1;k[2+iy,2-rx]=-1
  basis.append(k)
BT=torch.tensor(np.stack(basis));FOURTH=np.array([2/3,-1/12,0,0,0,0],np.float32)
# in uploaded TextureFilterTrain5, convolve2d kernel sign is reversed vs correlation
LEGACY=np.array([1,.57063894,.30414468,.33693168,.09205035,.08973684],np.float32)

def sample(shape,values,theta,shift,noise=0,seed=0):
 x,y=X-shift[0],Y-shift[1];u=np.cos(theta)*x+np.sin(theta)*y;v=-np.sin(theta)*x+np.cos(theta)*y
 if shape=='circle':
  r,=values;mask=x*x+y*y<=r*r;per=2*np.pi*r;axis=None
 elif shape=='ellipse':
  a,b=values;mask=(u/a)**2+(v/b)**2<=1;per=4*a*ellipe(1-(b/a)**2);axis=theta
 elif shape=='stadium':
  length,r=values;mask=np.maximum(abs(u)-length/2,0)**2+v*v<=r*r;per=2*length+2*np.pi*r;axis=theta
 elif shape=='rounded_box':
  a,b,r=values;qx=abs(u)-(a-r);qy=abs(v)-(b-r);dist=np.hypot(np.maximum(qx,0),np.maximum(qy,0))+np.minimum(np.maximum(qx,qy),0)-r;mask=dist<=0;per=4*(a+b-2*r)+2*np.pi*r;axis=theta
 elif shape=='ring':
  ro,ri=values;d=np.hypot(x,y);mask=(d<=ro)&(d>=ri);per=2*np.pi*(ro+ri);axis=None
 elif shape=='multi':
  r1,r2,sep=values;mask=((u-sep/2)**2+v*v<=r1*r1)|((u+sep/2)**2+v*v<=r2*r2);per=2*np.pi*(r1+r2);axis=None
 f=np.where(mask,1,-1).astype(np.float32)
 if noise>0:
  rng=np.random.default_rng(seed);f[rng.random(f.shape)<noise]*=-1
 ins=f>0;d=ndi.distance_transform_edt(ins)-ndi.distance_transform_edt(~ins);s=np.tanh(d/2.3).astype(np.float32)
 gx=(np.roll(s,-1,1)-np.roll(s,1,1))/2;gy=(np.roll(s,-1,0)-np.roll(s,1,0))/2
 vmag=np.sqrt(np.maximum(0,1-s*s))/np.maximum(np.hypot(gx,gy),1e-7)
 n=np.stack([gx*vmag,gy*vmag,s],0)
 n/=np.maximum(np.sqrt((n*n).sum(0,keepdims=True)),1e-8)
 return f,n,float(per),axis

def dataset(spec,ang_count,offset,noise=0):
 rows=[]
 for i,(shape,values) in enumerate(spec):
  for k,a in enumerate((np.arange(ang_count)+offset)*np.pi/ang_count):
   shift=(.16+.06*((i+k)%3),-.31+.13*((i+2*k)%4))
   f,n,per,axis=sample(shape,values,a,shift,noise,seed=51117+1234*i+79*k)
   rows.append((f,n,per,axis,shape,i))
 f=torch.tensor(np.stack([r[0] for r in rows])[:,None]);n=torch.tensor(np.stack([r[1] for r in rows]));gt=torch.tensor([r[2] for r in rows]);ax=torch.tensor([r[3] if r[3] is not None else 0 for r in rows]);has=torch.tensor([r[3] is not None for r in rows]);groups=np.array([r[5] for r in rows]);fam=[r[4] for r in rows]
 return dict(f=f,n=n,gt=gt,ax=ax,has=has,groups=groups,families=fam)
train_specs=[('circle',(7.5,)),('circle',(11.5,)),('ellipse',(15.,7.)),('ellipse',(12.5,8.5)),('stadium',(17.,4.8)),('stadium',(11.5,6.5))]
val_specs=[('circle',(9.2,)),('ellipse',(13.8,6.7)),('stadium',(15.5,5.8)),('rounded_box',(14.,8.,2.8))]
test_specs=[('circle',(6.3,)),('circle',(13.2,)),('ellipse',(16.2,6.2)),('stadium',(22.,4.6)),('rounded_box',(16.,8.,3.)),('ring',(13.,6.)),('multi',(5.5,7.,21.))]
TR=dataset(train_specs,6,0);VA=dataset(val_specs,8,.2);TE=dataset(test_specs,12,.12);TN=dataset(test_specs,12,.12,.01);REF=dataset([('circle',(10.3,))],1,0)

def deriv4(n,axis):return (2/3)*(n.roll(-1,axis)-n.roll(1,axis))-(1/12)*(n.roll(-2,axis)-n.roll(2,axis))

def observe(p,coef,N=10):
 theta=torch.as_tensor(coef,dtype=torch.float32)
 k=(theta[:,None,None]*BT).sum(0);kern=torch.stack([k,k.T],0)[:,None]
 n=p['n']
 for _ in range(N):
  s=n[:,2:3];g=F.conv2d(F.pad(s,(2,2,2,2),mode='circular'),kern)
  e=n.roll(1,2)+n.roll(-1,2)+n.roll(1,3)+n.roll(-1,3)
  h=torch.cat((e[:,:2]+g,e[:,2:3]+2*p['f']),dim=1)
  n=h/torch.linalg.vector_norm(h,dim=1,keepdim=True).clamp_min(1e-8)
 gx=deriv4(n,3);gy=deriv4(n,2)
 xx=(gx*gx).sum((1,2,3));yy=(gy*gy).sum((1,2,3));xy=(gx*gy).sum((1,2,3))
 raw=xx+yy;angle=.5*torch.atan2(torch.sin(2*(.5*torch.atan2(2*xy,xx-yy)+np.pi/2-p['ax'])),torch.cos(2*(.5*torch.atan2(2*xy,xx-yy)+np.pi/2-p['ax'])))
 return raw,angle

def lossval(params,p,reference=REF,deriv=False):
 param=torch.as_tensor(params,dtype=torch.float32).clone().detach().requires_grad_(deriv)
 ref,_=observe(reference,param);m,angle=observe(p,param)
 c=reference['gt'][0]/ref[0];gerr=c*m/p['gt']-1
 lgeo=(gerr*gerr).mean();lang=(angle[p['has']]**2).mean()
 cvs=[]
 for k in np.unique(p['groups']):
  z=m[p['groups']==k];cvs.append((z.std(unbiased=False)/z.mean())**2)
 lrot=torch.stack(cvs).mean();loss=lgeo+lang+3*lrot
 if deriv:
  loss.backward();return float(loss.detach()),param.grad.numpy().astype('float64')
 return float(loss.detach())

def metrics(p,params):
 with torch.no_grad():
  ref,_=observe(REF,params);m,ang=observe(p,params)
  c=float((REF['gt'][0]/ref[0]).item());e=(100*(c*m/p['gt']-1)).numpy();a=(ang.abs()*180/np.pi).numpy(); has=p['has'].numpy()
  fams={k:round(float(np.abs(e[[v==k for v in p['families']]]).mean()),4) for k in sorted(set(p['families']))}
  return dict(MARE_pct=float(np.abs(e).mean()),RMSE_pct=float(np.sqrt((e*e).mean())),bias_pct=float(e.mean()),angle_MAE_deg=float(a[has].mean()),calibration=c,by_family=fams)

if __name__=='__main__':
 print('Running two train-only L-BFGS-B optimizations; evaluation is separate.',flush=True)
 optim={};bounds=[(-1.5,1.5),(-.8,.8)]+[(-.75,.75)]*4
 for name,startcoef in [('fourth_start',FOURTH),('legacy_start',LEGACY)]:
  t=time.time()
  res=minimize(lambda z:lossval(z,TR,deriv=True),np.asarray(startcoef,dtype=float),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=16,ftol=1e-8,gtol=1e-6,maxls=12))
  optim[name]=dict(coefficients=res.x.tolist(),train_objective=float(res.fun),val_objective=lossval(res.x,VA),success=bool(res.success),termination=str(res.message),nit=int(res.nit),nfev=int(res.nfev),seconds=time.time()-t)
  (out/'optimizers.json').write_text(json.dumps(optim,indent=2))
  print(name, 'train_loss',res.fun,'validation_loss',optim[name]['val_objective'],'nfev',res.nfev,flush=True)
 print('Next run: python /mnt/data/pbgia_evaluate_results.py',flush=True)
