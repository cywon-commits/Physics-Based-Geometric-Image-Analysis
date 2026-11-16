import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from run_research import *

def test_update_matches_specification_and_scalar_has_no_divergence():
 rng=np.random.default_rng(9);n=rng.normal(size=(8,8,8,4)).astype('float32');n/=np.linalg.norm(n,axis=-1,keepdims=True);f=np.where(rng.random((8,8,8))>.5,1.,-1.).astype('float32');before=f.copy()
 H=sum(np.roll(n,1,k)+np.roll(n,-1,k) for k in range(3));H[...,:3]+=np.stack([deriv(n[...,3],k) for k in range(3)],-1);H[...,3]+=2*f;H/=np.linalg.norm(H,axis=-1,keepdims=True)
 assert np.allclose(update(n,f),H,atol=2e-7)
 assert np.array_equal(f,before)

def test_zero_effective_field_preserves_unit_vector():
 n=np.zeros((6,6,6,4),dtype='float32');n[...,3]=1
 out=update(n,np.ones((6,6,6)),h=0,D=0,J=0)
 assert np.array_equal(n,out)

def test_constant_linear_and_axis_covariance():
 for dim in [2,3]:
  x=np.indices((15,)*dim,dtype=float)
  for p in [(0,0,0),(0,3/16,0),(-1/12,0,0),(-.07,.12,.02)]:
   for k in range(dim):
    assert np.allclose(deriv(np.ones((15,)*dim),k,p),0)
    assert np.allclose(deriv(x[k],k,p)[(slice(3,-3),)*dim],1)
  n=initialize(x[0]<7);f=np.where(x[0]<7,1.,-1.)
  perm=list(reversed(range(dim)));nt=np.transpose(n,perm+[dim])[...,perm+[dim]]
  lhs=update(nt,np.transpose(f,perm));rhs=np.transpose(update(n,f),perm+[dim])[...,perm+[dim]]
  assert np.max(abs(lhs-rhs))<5e-7

def test_analytic_normals_and_mask_only_init():
 d,m,nu,_,_=fixture('sphere',N=40,R=9)
 x=np.indices(m.shape,dtype=float)-(40-1)/2
 for k,sh in enumerate((.15,-.3,.2)):x[k]-=sh
 truth=np.moveaxis(x,0,-1);truth/=np.linalg.norm(truth,axis=-1,keepdims=True)
 assert np.max(abs(truth-nu))<2e-6
 assert np.max(abs(np.linalg.norm(initialize(m),axis=-1)-1))<3e-7
