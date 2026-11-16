"""Joint derivative/exchange/observer operators. Main initialization matches pinned PBGIA-30."""
from pathlib import Path
import numpy as np,pandas as pd,scipy.ndimage as ndi
import itertools,json,time,hashlib,ast
from baseline_utils import tensor,features,feature_tensor,derivative,metrics,conventional
from fixed_iteration_checkpoints import initialize,one_step as old_step,CHECKPOINTS
from joint_core import derivative_filters
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';OUT.mkdir(exist_ok=True)
SNAPS=OUT/'checkpoints';SNAPS.mkdir(exist_ok=True)
MAIN={'J':1.,'h':2.,'D':1.}
BASE=json.load(open(ROOT/'vendor/pbgia30/selected_configuration.json'))

def parameter_config(family='axial',gp=(0,0,0),ep=(0,0,0),normalized=True,init='edt'):
 return dict(family=family,gp=list(map(float,gp)),ep=list(map(float,ep)),normalized=normalized,init=init,**MAIN)

def exchange_kernel(dim,c):
 G=derivative_filters(dim,*c['gp']);k=np.zeros((5,)*dim);center=(2,)*dim
 if c['family']=='axial':
  for i in range(dim):
   for sign in [-1,1]:idx=list(center);idx[i]+=sign;k[tuple(idx)]=1
 elif c['family']=='tied':
  k=(G**2).sum(0)
  if c['normalized']:k*=2*dim/k.sum()
 elif c['family']=='free':
  selfw,second,diagonal=c['ep'];axial=1-selfw-second-diagonal
  assert min(selfw,second,diagonal,axial)>=-1e-12
  k[center]=2*dim*selfw
  for i in range(dim):
   for sign in [-1,1]:
    idx=list(center);idx[i]+=sign;k[tuple(idx)]+=axial
    idx=list(center);idx[i]+=2*sign;k[tuple(idx)]+=second
  for i,j in itertools.combinations(range(dim),2):
   for si,sj in itertools.product([-1,1],repeat=2):
    idx=list(center);idx[i]+=si;idx[j]+=sj;k[tuple(idx)]+=2*dim*diagonal/(4*(dim*(dim-1)/2))
 else:raise ValueError(c['family'])
 return k

def generated_gradient(s,p):
 b,t,u=p;d=[-b,-(.5-2*b),0,.5-2*b,b];sm=[u,t,1-2*t-2*u,t,u];out=[]
 for i in range(s.ndim):
  a=ndi.correlate1d(s,d,axis=i,mode='wrap')
  for j in range(s.ndim):
   if j!=i and (t or u):a=ndi.correlate1d(a,sm,axis=j,mode='wrap')
  out.append(a)
 return np.stack(out,-1)

def exchange(n,c):
 dim=n.ndim-1
 if c['family']=='axial':return sum(np.roll(n,1,k)+np.roll(n,-1,k) for k in range(dim))
 if c['family']=='tied':
  b,t,u=c['gp'];d=np.array([-b,-(.5-2*b),0,.5-2*b,b]);sm=np.array([u,t,1-2*t-2*u,t,u]);out=np.zeros_like(n)
  for i in range(dim):
   a=ndi.correlate1d(n,d*d,axis=i,mode='wrap')
   for j in range(dim):
    if j!=i:a=ndi.correlate1d(a,sm*sm,axis=j,mode='wrap')
   out+=a
  if c['normalized']:out*=2*dim/(dim*np.sum(d*d)*np.sum(sm*sm)**(dim-1))
  return out
 selfw,second,diagonal=c['ep'];axial=1-selfw-second-diagonal
 out=(2*dim*selfw)*n+axial*sum(np.roll(n,1,k)+np.roll(n,-1,k) for k in range(dim))
 if second:out+=second*sum(np.roll(n,2,k)+np.roll(n,-2,k) for k in range(dim))
 if diagonal:
  coef=2*dim*diagonal/(4*(dim*(dim-1)/2))
  for i,j in itertools.combinations(range(dim),2):
   for si,sj in itertools.product([-1,1],repeat=2):out+=coef*np.roll(np.roll(n,si,i),sj,j)
 return out

def step(n,f,c):
 if c['family']=='axial' and c['gp']==[0.,0.,0.]:return old_step(n,f,J=c['J'],h=c['h'],D=c['D'])
 H=c['J']*exchange(n,c);H[...,:f.ndim]+=c['D']*generated_gradient(n[...,-1],c['gp']);H[...,-1]+=c['h']*f;norm=np.linalg.norm(H,axis=-1,keepdims=True)
 return np.where(norm>1e-12,H/np.maximum(norm,1e-12),n).astype('float32')

def normalized_tensor(T):return T/max(np.linalg.norm(T),1e-14)
def orientation_tensors(n,c,T):
 v=n[...,:-1].reshape(-1,n.shape[-1]-1).astype(float);hj=exchange(n,c)[...,:-1].reshape(v.shape).astype(float);d=v.shape[1]
 B=(hj.T@v+v.T@hj)/2;B-=np.eye(d)*np.trace(B)/d
 V=v.T@v;V-=np.eye(d)*np.trace(V)/d
 G=T-np.eye(d)*np.trace(T)/d
 return B,V,G

def read_axis(Q,axis,which='min',threshold=1e-3):
 if axis is None:return dict(angle_error=np.nan,eigengap=np.nan,orientation_defined=False)
 vals,vec=np.linalg.eigh(Q);idx=0 if which=='min' else -1;gap=(vals[1]-vals[0] if idx==0 else vals[-1]-vals[-2])/max(np.linalg.norm(vals),1e-14)
 err=float(np.degrees(np.arccos(np.clip(abs(vec[:,idx]@axis),0,1))))
 return dict(angle_error=err,eigengap=float(gap),orientation_defined=bool(gap>=threshold))

def observe(n,c,p,axis,obs='gradient',mix=(1,0,0),which='min',F=None):
 tic=time.perf_counter();d=n.ndim-1;T=tensor(n,p) if F is None else feature_tensor(F,p,d);B,V,G=orientation_tensors(n,c,T)
 maps={'bond':B,'direct':V,'gradient':G};Q=sum(w*normalized_tensor(a) for w,a in zip(mix,[G,B,V])) if obs=='ensemble' else maps[obs]
 result=dict(raw_measurement=float(np.trace(T)),**read_axis(Q,axis,which),measurement_seconds=time.perf_counter()-tic)
 return result,dict(gradient=G,bond=B,direct=V)

# Analytical synthetic fixtures and group split; former GitHub test prototypes are excluded.
def make_manifest():
 path=OUT/'split_manifest.csv'
 if path.exists():return pd.read_csv(path)
 rng=np.random.default_rng(261116);rows=[];unit=0
 for dim,families in [(2,['circle','stadium','ellipse','ring','multi']),(3,['sphere','capsule','torus','ellipsoid','multi','small'])]:
  for fam in families:
   held=fam in ['ring','ellipsoid']
   if dim==2:rs=[9.1,12.1] if held else [4.4,5.6,6.8,7.9,8.7] if fam=='multi' else [6.4,8.1,9.7,11.3,12.8]
   else:rs=[5.7,7.7] if held else [9.4,10.4,11.4,12.4,13.4] if fam=='torus' else [1.8,2.3,2.8,3.3,3.8] if fam=='small' else [3.4,4.1,4.8,5.3,5.8] if fam=='multi' else [4.4,5.4,6.4,7.4,8.4]
   perm=rng.permutation(len(rs));splits={i:('test' if held else 'train' if j<2 else 'validation' if j==2 else 'test') for j,i in enumerate(perm)}
   for i,R in enumerate(rs):
    uid=f'j{unit:03d}';unit+=1;phase=float(rng.uniform(.03,.6))
    for rep in range(2):
     for ni,noise in enumerate([0.,.002,.01]):rows.append(dict(case=f'{uid}_r{rep}_n{ni}',unit=uid,split=splits[i],role='geometry',family=fam,heldout_family=held,dim=dim,R=R,size=80 if dim==2 else 48,rep=rep,phase=phase,noise=noise,seed=261116+unit*113+rep*17,noise_seed=461116+unit*137+rep*23+ni,spacing=1.))
 for dim,fam,R in [(2,'circle',6.85),(3,'sphere',5.85)]:rows.append(dict(case=f'joint_cal{dim}',unit=f'joint_cal{dim}',split='train',role='calibration',family=fam,heldout_family=False,dim=dim,R=R,size=80 if dim==2 else 48,rep=0,phase=.23,noise=0.,seed=261116,noise_seed=461116,spacing=1.))
 df=pd.DataFrame(rows);df.to_csv(path,index=False)
 old=pd.read_csv(ROOT/'vendor/pbgia30/split_manifest.csv');a=set(df[['dim','family','R']].itertuples(index=False,name=None));b=set(old[['dim','family','R']].itertuples(index=False,name=None));assert a.isdisjoint(b),a&b
 return df

def fixture(row):
 # Reuse tested analytical formulas; modify rotations/aspect ratios for the new study.
 import baseline_utils as bu
 return new_geometry(row)

# Standalone modification of prior generator: rotations not restricted to a fixed planar tilt.
from baseline_utils import geometry as _previous_geometry
_src=__import__('inspect').getsource(_previous_geometry).replace('def geometry(row):','def new_geometry(row):')
_src=_src.replace("theta=(.17+.64*rep+float(row['phase']))","theta=(.11+.93*rep+float(row['phase']))")
_src=_src.replace("axis=np.array([np.cos(theta)*.82,np.sin(theta)*.82,.572]);axis/=np.linalg.norm(axis);v=np.cross([0,0,1],axis);v/=np.linalg.norm(v);Q=np.stack([axis,v,np.cross(axis,v)],1)","axis=np.array([np.cos(theta)*np.cos(.29+.41*rep),np.sin(theta)*np.cos(.29+.41*rep),np.sin(.29+.41*rep)]);axis/=np.linalg.norm(axis);v=np.cross([0,0,1],axis);v/=np.linalg.norm(v);Q=np.stack([axis,v,np.cross(axis,v)],1)")
_src=_src.replace('[1.6*R,.75*R]+([R] if dim==3 else [])','[1.72*R,.69*R]+([1.07*R] if dim==3 else [])').replace('slice(7,-7)','slice(7,-7)')
from scipy.special import ellipe
exec(_src,globals())

def rollout(row,c,save=False,features_needed=False):
 f,clean,normal,dist,truth,axis=fixture(row);tic=time.perf_counter()
 n=initialize(f) if c['init']=='edt' else np.concatenate([np.zeros(f.shape+(f.ndim,),dtype='float32'),f[...,None]],-1)
 init_sec=time.perf_counter()-tic;elapsed=0.;states={};ms={};prev=None
 digest=hashlib.sha256(f.tobytes()+json.dumps(c,sort_keys=True).encode()+b'joint_v1').hexdigest()
 for t in range(1,41):
  tic=time.perf_counter();proposal=step(n,f,c);elapsed+=time.perf_counter()-tic;prev,n=n,proposal
  if t in CHECKPOINTS:
   states[t]=n.copy();n1=step(n,f,c);delta=np.linalg.norm(n1-n,axis=-1);two=np.linalg.norm(n1-prev,axis=-1)
   m=metrics(n,f,clean,normal,dist);m.update(N=t,one_step_rms=float(np.sqrt(np.mean(delta**2))),one_step_max=float(delta.max()),two_step_rms=float(np.sqrt(np.mean(two**2))),finite=bool(np.isfinite(n).all()),initialization_seconds=init_sec,field_seconds=elapsed,clean_iou=float(((n[...,-1]>0)&clean).sum()/max(1,((n[...,-1]>0)|clean).sum())))
   ms[t]=m
 if save:np.savez_compressed(SNAPS/(digest+'.npz'),f=f,**{'n'+str(t):s for t,s in states.items()})
 return dict(row=row,config=c,key=digest,truth=truth,axis=axis,f=f,clean=clean,normal=normal,dist=dist,states=states,metrics=ms,features={N:features(s) for N,s in states.items()} if features_needed else {})

def source_legacy_function(name='texture'):
 from scipy.signal import convolve2d
 path=ROOT/'reference'/('texturefiltertrain5 (1).py' if name=='texture' else 'lengthorientationapplication (1).py');module=ast.parse(path.read_text());fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='run_spin_simulation')
 if name=='length':
  defaults=next(n for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='default_kernel_dmx' for t in n.targets));env={'np':np,'convolve2d':convolve2d,'default_parameters':(.1,1,10),'default_conversion':2/37};exec(compile(ast.Module(body=[defaults],type_ignores=[]),str(path),'exec'),env)
 else:env={'np':np,'convolve2d':convolve2d}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env);return env['run_spin_simulation']

def old_kernel():
 module=ast.parse((ROOT/'reference/lengthorientationapplication (1).py').read_text());assign=next(n for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='default_kernel_dmx' for t in n.targets));env={'np':np};exec(compile(ast.Module(body=[assign],type_ignores=[]),'legacy-kernel','exec'),env);return env['default_kernel_dmx']

def historical(f,variant='texture'):
 from scipy.signal import convolve2d
 K=old_kernel();KJ=K*K+K.T*K.T;ns=5 if variant=='texture' else 10;D=1 if variant=='texture' else .1;mode='wrap' if variant=='texture' else 'symm'
 vx=np.zeros_like(f,dtype=float);vy=vx.copy();s=f.astype(float)
 for _ in range(ns):
  hx=convolve2d(vx,KJ,'same',boundary=mode);hy=convolve2d(vy,KJ,'same',boundary=mode);hs=convolve2d(s,KJ,'same',boundary=mode)
  H=np.stack([hx+D*convolve2d(s,K,'same',boundary=mode),hy+D*convolve2d(s,K.T,'same',boundary=mode),hs+f],-1);norm=np.linalg.norm(H,axis=-1);v=H/np.maximum(norm[...,None],1e-12);vx,vy,s=np.moveaxis(v,-1,0)
 tx=np.sum(hx*vx-hy*vy);ty=np.sum(hx*vy+hy*vx);Q=np.array([[-tx/2,ty/2],[ty/2,tx/2]])
 return np.stack([vy,vx,s],-1),Q,dict(N=ns,D=D,h=1,exchange_sum=float(KJ.sum()),boundary=mode,init='binary',observer='historical stale HJ(t-1) and v(t)')
