"""One frozen generator for all measurements; exact fixed N=10."""
import numpy as np,time
from scipy import ndimage as ndi

def gradient4(s):return np.stack([(2/3)*(np.roll(s,-1,j)-np.roll(s,1,j))-(1/12)*(np.roll(s,-2,j)-np.roll(s,2,j)) for j in range(s.ndim)],-1)
def initialize(f,width=2.3):
 signed=ndi.distance_transform_edt(f>0)-ndi.distance_transform_edt(f<0);s=np.tanh(signed/width);g=np.stack([.5*(np.roll(s,-1,j)-np.roll(s,1,j)) for j in range(f.ndim)],-1);v=np.sqrt(np.maximum(0,1-s*s))[...,None]*g/np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-7);n=np.concatenate([v,s[...,None]],-1);return n/np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-12)
def step(n,f,spacing=1.):
 H=sum(np.roll(n,k,j) for j in range(f.ndim) for k in [-1,1]);H[...,:f.ndim]+=gradient4(n[...,-1])/spacing;H[...,-1]+=2*f;norm=np.linalg.norm(H,axis=-1,keepdims=True);return np.where(norm>1e-12,H/np.maximum(norm,1e-12),n)
def generate(f,width=2.3,spacing=1.,N=10):
 before=f.copy();tic=time.perf_counter();n=initialize(f,width);initial=time.perf_counter()-tic;tic=time.perf_counter()
 for _ in range(N):n=step(n,f,spacing)
 elapsed=time.perf_counter()-tic;assert np.array_equal(f,before);assert np.isfinite(n).all();assert abs(np.linalg.norm(n,axis=-1)-1).max()<1e-12;return n,dict(field_seconds=elapsed,initialization_seconds=initial)
