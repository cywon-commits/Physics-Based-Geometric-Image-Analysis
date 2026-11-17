"""Geometry-only numerical operators. No reconstruction or topological screening."""
from pathlib import Path
import numpy as np, scipy.ndimage as ndi
import itertools,ast
ROOT=Path(__file__).resolve().parents[1]
MAIN={'J':1.,'h':2.,'D':1.}


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


def derivative_filters(dim, b=-1/12, t1=0.0, t2=0.0):
    """5^d derivative tensors with d_i linear response = +1 (correlation sign).
    arr coordinate derivatives; axis permutations equivariant, not full SO(d).
    """
    assert dim in (2,3)
    d = np.array([-b, -(.5-2*b), 0., (.5-2*b), b],dtype=float)
    smooth = np.array([t2,t1,1-2*t1-2*t2,t1,t2],dtype=float)
    kernels=[]
    for axis in range(dim):
        k=np.ones((1,)*dim,dtype=float)
        for j in range(dim):
            vec=d if axis==j else smooth
            shape=[1]*dim;shape[j]=5
            k=k*vec.reshape(shape)
        kernels.append(k)
    return np.stack(kernels)


def initialize(f, width=2.3):
    signed=ndi.distance_transform_edt(f>0)-ndi.distance_transform_edt(f<0)
    s=np.tanh(signed/width).astype(np.float32)
    g=gradient(s); norm=np.linalg.norm(g,axis=-1,keepdims=True)
    v=np.sqrt(np.maximum(0.,1.-s*s))[...,None]*g/np.maximum(norm,1e-7)
    n=np.concatenate([v,s[...,None]],axis=-1)
    n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-12)
    return n.astype(np.float32)


def gradient(s):
    return np.stack([(np.roll(s,-1,axis=k)-np.roll(s,1,axis=k))*.5 for k in range(s.ndim)],axis=-1)


def old_step(n,f,J=1.,h=2.,D=1.,alpha=1.):
    dim=f.ndim
    H=sum(np.roll(n,1,axis=k)+np.roll(n,-1,axis=k) for k in range(dim))*J
    H[...,:dim]+=D*gradient(n[...,dim])
    H[...,dim]+=h*f
    hn=np.linalg.norm(H,axis=-1,keepdims=True)
    proposal=np.divide(H, np.maximum(hn,1e-12),out=np.zeros_like(H),where=hn>1e-12)
    proposal=np.where(hn>1e-12,proposal,n)
    if alpha!=1.:
        proposal=(1.-alpha)*n+alpha*proposal
        proposal/=np.maximum(np.linalg.norm(proposal,axis=-1,keepdims=True),1e-12)
    return proposal.astype(np.float32)


def derivative(n,axis,p):
 d=n.ndim-1;b,t,u=p;a=(.5-2*b)*(np.roll(n,-1,axis)-np.roll(n,1,axis))+b*(np.roll(n,-2,axis)-np.roll(n,2,axis))
 for j in range(d):
  if j!=axis:a=(1-2*t-2*u)*a+t*(np.roll(a,1,j)+np.roll(a,-1,j))+u*(np.roll(a,2,j)+np.roll(a,-2,j))
 return a


def tensor(n,p):
 g=[derivative(n,k,p) for k in range(n.ndim-1)]
 return np.array([[np.sum(a*b,dtype=np.float64) for b in g] for a in g])


def features(n):
 # Exact linear derivative basis. Tensor under any admissible kernel is a small quadratic form.
 d=n.ndim-1;fields=[]
 for axis in range(d):
  diffs=[(np.roll(n,-i,axis)-np.roll(n,i,axis)) for i in (1,2)]
  for diff in diffs:
   for choice in itertools.product(range(3),repeat=d-1):
    a=diff
    for j,kind in zip([j for j in range(d) if j!=axis],choice):
     if kind:a=np.roll(a,kind,j)+np.roll(a,-kind,j)-2*a
    fields.append(a.reshape(-1))
 B=np.stack(fields).astype('float64');return B@B.T


def weights(p,d):
 b,t,u=p;s=[1,t,u];return np.array([a*np.prod([s[c] for c in choice]) for a in [.5-2*b,b] for choice in itertools.product(range(3),repeat=d-1)])


def feature_tensor(F,p,d):
 w=weights(p,d);m=len(w);return np.einsum('a,iajb,b->ij',w,F.reshape(d,m,d,m),w,optimize=True)
