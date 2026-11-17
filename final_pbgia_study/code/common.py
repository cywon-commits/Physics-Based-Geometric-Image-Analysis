from pathlib import Path
import sys,json,hashlib,time
import numpy as np,pandas as pd
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';OUT.mkdir(exist_ok=True);sys.path.insert(0,str(ROOT/'vendor'))
from baseline import initialize,gradient,old_step,derivative,tensor
BASE=json.load(open(ROOT/'reference/PBGIA30_frozen.json'));SEED=27111743

def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=True))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def axis_error(M,axis,which='min'):
 if axis is None:return np.nan,np.nan
 if not np.isfinite(M).all():return np.nan,np.nan
 vals,vec=np.linalg.eigh(M);j=0 if which=='min' else -1;gap=(vals[1]-vals[0] if j==0 else vals[-1]-vals[-2])/max(np.linalg.norm(vals),1e-30);return float(np.degrees(np.arccos(np.clip(abs(vec[:,j]@axis),0,1)))),float(gap)
def g4(a,j,spacing=1.):return ((2/3)*(np.roll(a,-1,j)-np.roll(a,1,j))-(1/12)*(np.roll(a,-2,j)-np.roll(a,2,j)))/spacing

def physical_energies(n,f,D=1.,h=2.,J=1.):
 # Positive NN exchange; no subtraction from NN in base evolution.
 nn=sum(np.roll(n,k,j) for j in range(f.ndim) for k in [-1,1]);g=gradient(n[...,-1]);ex=-.5*J*np.sum(n*nn,dtype=float);ext=-h*np.sum(f*n[...,-1],dtype=float);cross=-D*np.sum(n[...,:f.ndim]*g,dtype=float)
 return dict(exchange=ex,external=ext,cross=cross,total=ex+ext+cross)

def effective(n,f,reciprocal=False):
 H=sum(np.roll(n,k,j)+np.roll(n,-k,j) for j in range(f.ndim) for k in [1]);H[...,:f.ndim]+=gradient(n[...,-1]);H[...,-1]+=2*f
 if reciprocal:H[...,-1]-=sum(.5*(np.roll(n[...,j],-1,j)-np.roll(n[...,j],1,j)) for j in range(f.ndim))
 return H

def variational_step(n,f):
 H=effective(n,f,True);tangent=H-n*np.sum(n*H,axis=-1,keepdims=True);e0=physical_energies(n,f)['total'];norm2=np.sum(tangent*tangent,dtype=float);step=.1
 for _ in range(25):
  candidate=n+step*tangent;candidate/=np.linalg.norm(candidate,axis=-1,keepdims=True);energy=physical_energies(candidate,f)['total']
  if energy<=e0-1e-4*step*norm2+1e-10:return candidate,step
  step*=.5
 return n.copy(),0.

def measure(n,axis=None,axis_role='object_major',spacing=1.):
 d=n.ndim-1;t0=time.perf_counter();T=tensor(n,BASE['kernel_by_dimension'][str(d)])*spacing**(d-2);s=n[...,-1];gs=np.stack([g4(s,j,spacing) for j in range(d)],-1);gn=np.linalg.norm(gs,axis=-1);density=.5*gn;tv=float(density.sum()*spacing**d);E=float(np.trace(T));normal=gs/np.maximum(gn[...,None],1e-30);M=normal.reshape(-1,d).T@(density.reshape(-1,1)*normal.reshape(-1,d))*spacing**d
 # Coordinate covariance from the same phase-boundary density.
 coords=np.indices(s.shape,dtype=float).reshape(d,-1).T*spacing;mass=density.sum();centroid=(density.reshape(-1,1)*coords).sum(0)/max(mass,1e-30);z=coords-centroid;C=z.T@(density.reshape(-1,1)*z)/max(mass,1e-30)
 which='max' if axis_role in ['symmetry_axis','oblate_normal'] else 'min';sp='min' if which=='max' else 'max';ae,eg=axis_error(T,axis,which);an,ng=axis_error(M,axis,which);ac,cg=axis_error(C,axis,sp)
 return dict(I_E=E,I_TV=tv,energy_axis_error=ae,normal_axis_error=an,spatial_axis_error=ac,energy_gap=eg,normal_gap=ng,spatial_gap=cg,measurement_seconds=time.perf_counter()-t0),dict(energy=T,normal=M,spatial=C)
