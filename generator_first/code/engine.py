"""Measurement-only engine: immutable input, two independent heads, finite-state checks."""
from operators import *
from data import ROOT,OUT,fixture
import json,time,hashlib
CHECKPOINTS=(5,10,20,30,40)

def kernel_array(dim,c):
 if c['family']!='signed':return exchange_kernel(dim,c)
 q=c['q'];k=np.zeros((5,)*dim);k[(2,)*dim]=2*dim*.6
 for i in range(dim):
  for sign in [-1,1]:
   pos=[2]*dim;pos[i]+=sign;k[tuple(pos)]+=.4+q
   pos=[2]*dim;pos[i]+=2*sign;k[tuple(pos)]-=q
 return k

def apply_exchange(n,c):
 if c['family']!='signed':return exchange(n,c)
 d=n.ndim-1;q=c['q'];return (2*d*.6)*n+(.4+q)*sum(np.roll(n,1,i)+np.roll(n,-1,i) for i in range(d))-q*sum(np.roll(n,2,i)+np.roll(n,-2,i) for i in range(d))

def update(n,f,c):
 H=c['J']*apply_exchange(n,c);H[...,:f.ndim]+=c['D']*generated_gradient(n[...,-1],c['gp'])/c.get('spacing',1.);H[...,-1]+=c['h']*f;norm=np.linalg.norm(H,axis=-1,keepdims=True)
 return np.where(norm>1e-12,H/np.maximum(norm,1e-12),n).astype('float32')

def key(f,c):return hashlib.sha256(f.tobytes()+json.dumps(dict(shape=f.shape,dtype=str(f.dtype),generator=c,initialization='binary_EDT_2.3',boundary='periodic'),sort_keys=True).encode()).hexdigest()

def rollout(row,c,checkpoints=CHECKPOINTS,need_features=False):
 f,truth,axis,aux=fixture(row);before=f.copy();tic=time.perf_counter();n=initialize(f) if c.get('init','edt')=='edt' else np.concatenate([np.zeros(f.shape+(f.ndim,),dtype='float32'),f[...,None]],-1);initial=time.perf_counter()-tic;elapsed=0.;states={};diagnostics={};previous=None
 for it in range(1,max(checkpoints)+1):
  tic=time.perf_counter();proposal=update(n,f,c);elapsed+=time.perf_counter()-tic;previous,n=n,proposal
  if not np.isfinite(n).all():raise FloatingPointError('Nonfinite field state')
  if it in checkpoints:
   unit_error=float(abs(np.linalg.norm(n,axis=-1)-1).max())
   if unit_error>2e-5:raise FloatingPointError('Normalization failure')
   states[it]=n.copy();n1=update(n,f,c);delta=np.linalg.norm(n1-n,axis=-1);two=np.linalg.norm(n1-previous,axis=-1)
   diagnostics[it]=dict(N=it,field_seconds=elapsed,initialization_seconds=initial,unit_error=unit_error,finite=True,one_step_rms=float(np.sqrt(np.mean(delta**2))),one_step_max=float(delta.max()),two_step_rms=float(np.sqrt(np.mean(two**2))))
 assert np.array_equal(f,before)
 return dict(row=row,f=f,truth=truth,axis=axis,aux=aux,c=c,key=key(f,c),states=states,diagnostics=diagnostics,F={N:features(a) for N,a in states.items()} if need_features else {})

def axis_result(Q,axis,which):
 vals,vec=np.linalg.eigh(Q);i=0 if which=='min' else -1;gap=(vals[1]-vals[0] if i==0 else vals[-1]-vals[-2])/max(np.linalg.norm(vals),1e-14)
 return dict(angle_error=float(np.degrees(np.arccos(np.clip(abs(vec[:,i]@axis),0,1)))) if axis is not None else np.nan,eigengap=float(gap),predicted_ambiguous=bool(gap<.001),axis=vec[:,i])

def orientation_matrices(n,c,p,F=None):
 T=tensor(n,p) if F is None else feature_tensor(F,p,n.ndim-1);d=n.ndim-1;v=n[...,:d].reshape(-1,d).astype(float);hj=apply_exchange(n,c)[...,:d].reshape(-1,d).astype(float);B=(v.T@hj+hj.T@v)/2;V=v.T@v
 return [T-np.eye(d)*np.trace(T)/d,B-np.eye(d)*np.trace(B)/d,V-np.eye(d)*np.trace(V)/d]

def geometry_measure(n,head,F=None):
 kind=head.get('type','learned_tensor')
 if kind=='learned_tensor':return float(np.trace(tensor(n,head['p']) if F is None else feature_tensor(F,head['p'],n.ndim-1)))
 s=n[...,-1];g=np.stack([(np.roll(s,-1,k)-np.roll(s,1,k))*.5 for k in range(n.ndim-1)],-1);gn=np.linalg.norm(g,axis=-1)
 if kind=='scalar_L1':return float(gn.sum())
 if kind=='coarea':
  eps=head.get('epsilon',.35);return float((np.exp(-s*s/(2*eps*eps))/(np.sqrt(2*np.pi)*eps)*gn).sum())
 raise ValueError(kind)

def readout(n,c,heads,coefficient,axis=None,spacing=1.,F=None):
 tic=time.perf_counter();d=n.ndim-1;pg=heads['geometry']['p'];po=heads['orientation']['p'];M=geometry_measure(n,heads['geometry'],F);mat=orientation_matrices(n,c,po,F);o=heads['orientation'];Q=sum(w*normalized_tensor(T) for w,T in zip(o['mix'],mat)) if o['type']=='ensemble' else mat[{'gradient':0,'bond':1,'direct':2}[o['type']]];a=axis_result(Q,axis,o['which']);measurement=time.perf_counter()-tic
 return dict(raw_measurement=M,prediction=coefficient*M*spacing**(d-2),**{k:v for k,v in a.items() if k!='axis'},measurement_seconds=measurement),a['axis']

def scalar_readout(n):
 g=np.stack([(np.roll(n[...,-1],-1,k)-np.roll(n[...,-1],1,k))*.5 for k in range(n.ndim-1)],-1);return float(np.linalg.norm(g,axis=-1).sum())

def calibrate(ref,heads,N):
 M=geometry_measure(ref['states'][N],heads[str(N)]['geometry']);return ref['truth']/(M*float(ref['row']['spacing'])**(ref['row']['dim']-2))
