"""Minimal auditable 2D/3D joint-operator PBGIA prototype.
Axis components follow NumPy array axes, not screen x/y.
This is a smoke-test kernel and not the finished trained benchmark.
"""
import numpy as np
from scipy import ndimage as ndi


def binary_edt_init(f, width=2.3):
    f = np.asarray(f)
    assert f.ndim in (2,3) and np.all(np.isin(np.unique(f), [-1,1]))
    inside = f > 0
    sd = ndi.distance_transform_edt(inside) - ndi.distance_transform_edt(~inside)
    s = np.tanh(sd / width)
    g = np.stack(np.gradient(s), axis=-1)
    norm = np.linalg.norm(g, axis=-1, keepdims=True)
    # Fallback direction at flat points; in saturated regions |v| is nearly zero.
    direction = np.divide(g, norm, out=np.zeros_like(g), where=norm>1.e-12)
    direction[..., 0] = np.where(norm[...,0]>1.e-12, direction[...,0], 1.)
    v = direction * np.sqrt(np.maximum(0, 1-s*s))[...,None]
    n = np.concatenate((v, s[...,None]), axis=-1).astype('float32')
    n /= np.linalg.norm(n,axis=-1,keepdims=True)
    return n


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


def exchange_filter(dim, derivative_kernels=None, kind='axial', radius=1., spread=.12):
    """Return non-negative inversion-symmetric KJ. Normalize to sum=2d for comparability.
    'tied': historical pointwise sum_i G_i**2 (NOT G^T G).
    'free': independent, positive, axis-symmetric 5^d separable Gaussian-family weights.
    """
    shp=(5,)*dim; k=np.zeros(shp,float);c=(2,)*dim
    if kind=='axial':
        for i in range(dim):
            for s in (-1,1):
                idx=list(c);idx[i]+=s;k[tuple(idx)]+=1.
    elif kind=='tied':
        if derivative_kernels is None:raise ValueError('need G kernels')
        k=np.square(derivative_kernels).sum(axis=0)
    elif kind=='free':
        q=np.array([spread, radius, 1., radius, spread],float)
        # positive separable exchange, no negative antialignment
        for i in range(dim):
            qshape=[1]*dim;qshape[i]=5
            k+=np.broadcast_to(q.reshape(qshape),shp)
    else:raise ValueError(kind)
    if k.sum()<=0:raise ValueError('zero exchange')
    return (k* (2*dim/k.sum())).astype(float)


def one_step(n,f,kder,kexc,J=1.,h=2.,D=1.):
    dim=f.ndim
    s=n[...,-1]; v=n[...,:dim]
    hj=np.stack([ndi.correlate(n[...,a],kexc,mode='wrap') for a in range(dim+1)],axis=-1)
    grad=np.stack([ndi.correlate(s,kder[i],mode='wrap') for i in range(dim)],axis=-1)
    H=np.concatenate((J*hj[...,:dim]+D*grad,(J*hj[...,dim]+h*f)[...,None]),axis=-1)
    hn=np.linalg.norm(H,axis=-1,keepdims=True)
    nxt=np.divide(H,hn,out=np.array(n,dtype=float,copy=True),where=hn>1e-12)
    return nxt.astype('float32')


def run(f,N,kder,kexc,J=1,h=2,D=1,init='edt'):
    f=np.asarray(f,dtype='float32')
    if init=='edt':n=binary_edt_init(f)
    elif init=='binary':
        n=np.zeros(f.shape+(f.ndim+1,),dtype='float32');n[...,-1]=f
    else:raise ValueError('initialization')
    for _ in range(N):n=one_step(n,f,kder,kexc,J,h,D)
    return n


def observables(n,kexc,derivatives=None):
    """Returns the two 2D/3D nematic-tensor variants and gradient tensor.
    Qbond := sym(HJ⊗v) - tr(HJ·v)/d I (historical 2D generalization).
    Qv := v⊗v - |v|²/d I.
    Note Qbond/Qv eigenvalue ordering must be calibrated against ground-truth axes.
    """
    dim=n.ndim-1;v=n[...,:dim].astype(float)
    HJ=np.stack([ndi.correlate(v[...,a],kexc,mode='wrap') for a in range(dim)],axis=-1)
    I=np.eye(dim)
    bond=np.einsum('...i,...j->ij',HJ,v,optimize=True)
    bond=(bond+bond.T)/2-I*np.trace(bond)/dim
    direct=np.einsum('...i,...j->ij',v,v,optimize=True)
    direct=direct-I*np.trace(direct)/dim
    if derivatives is None:return bond,direct,None
    grad=np.stack([np.stack([ndi.correlate(n[...,a],derivatives[i],mode='wrap') for a in range(dim+1)],axis=-1) for i in range(dim)])
    G=np.einsum('i...a,j...a->ij',grad,grad,optimize=True)
    return bond,direct,G


def axis_mae(Q,gt,which='min'):
    vals,vecs=np.linalg.eigh(Q)
    chosen=vecs[:,0 if which=='min' else -1]
    err=np.degrees(np.arccos(np.clip(abs(chosen @ gt),0.,1.)))
    gap=np.min(np.abs(np.diff(vals)))/max(np.max(np.abs(vals)),1e-12)
    return float(err),float(gap)


def shape(dim=2, kind='stadium', radius=6., size=None, angle_deg=24., shift=.2):
    """Synthetic binary fixture. Exact signed shape only creates input + GT, never initializes state."""
    if size is None:size=64 if dim==2 else 32
    coords=np.indices((size,)*dim,dtype=float)
    x=coords-(size-1)/2
    x[0]-=shift
    a=np.radians(angle_deg); ca,sa=np.cos(a),np.sin(a)
    # coordinate-axis 0/1 principal axis (same convention as n)
    u=ca*x[0]+sa*x[1]; w=-sa*x[0]+ca*x[1]
    gt=np.array([ca,sa]+([0] if dim==3 else []),float)
    rad2=w*w + (x[2]*x[2] if dim==3 else 0)
    if kind in ('stadium','capsule'):
        d=radius-np.sqrt(np.maximum(abs(u)-radius,0)**2+rad2)
    elif kind in ('ellipse','ellipsoid'):
        d=1-((u/(radius*1.6))**2+rad2/(radius*.75)**2)
    else:raise ValueError(kind)
    return np.where(d>=0,1.,-1.).astype('float32'),gt
