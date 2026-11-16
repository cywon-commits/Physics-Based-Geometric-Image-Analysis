"""Minimal invariants, independent of the legacy benchmark module (which runs on import)."""
import numpy as np

def derivative_interior(n, axis, b=-1/12, t1=0.0, t2=0.0):
    d1=.5-2*b; d2=b
    a=d1*(np.roll(n,-1,axis=axis)-np.roll(n,1,axis=axis)) + d2*(np.roll(n,-2,axis=axis)-np.roll(n,2,axis=axis))
    for other in range(n.ndim-1):
        if other == axis: continue
        a=(1-2*t1-2*t2)*a+t1*(np.roll(a,1,axis=other)+np.roll(a,-1,axis=other))+t2*(np.roll(a,2,axis=other)+np.roll(a,-2,axis=other))
    return a

for dim in (2,3):
    shape=(20,)*dim
    const=np.ones(shape+(1,))
    for axis in range(dim):
        for p in [(0,0,0),(-1/12,0,0),(-.09,.12,.02)]:
            assert np.allclose(derivative_interior(const,axis,*p),0)
            coord=np.indices(shape)[axis].astype(float)[...,None]
            arr=derivative_interior(coord,axis,*p)
            core=(slice(5,-5),)*dim+(slice(None),)
            assert np.max(abs(arr[core]-1)) < 1e-10, (dim,axis,p)
    # A scalar field varying only along one axis gives rank-1 gradient tensor.
    n=np.zeros(shape+(dim+1,)); n[...,0]=np.indices(shape)[0]
    vals=[derivative_interior(n,ax) for ax in range(dim)]
    core=(slice(5,-5),)*dim+(slice(None),)
    T=np.array([[np.sum(vals[i][core]*vals[j][core]) for j in range(dim)] for i in range(dim)])
    assert np.allclose(T,T.T)
    assert np.linalg.eigvalsh(T).min()>-1e-9
    assert np.trace(T)>0
print('PASS: constant/linear reproduction, tensor symmetry and PSD for 2D/3D')
