import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from joint_core import binary_edt_init,derivative_filters,exchange_filter,run,one_step,observables,shape,axis_mae

def test_derivative_moments():
 for d in (2,3):
  G=derivative_filters(d,-.06,.12,.04)
  assert G.shape==(d,)+(5,)*d
  for i in range(d):
   assert abs(G[i].sum())<1e-12
   for j in range(d):
    m=np.arange(-2,3).reshape(tuple(5 if k==j else 1 for k in range(d)))
    assert abs((G[i]*m).sum()-(1 if i==j else 0))<1e-12

def test_exchange_positive_and_normalized():
 for d in (2,3):
  gd=derivative_filters(d)
  for kind in ('axial','tied','free'):
   k=exchange_filter(d,gd,kind=kind)
   assert np.min(k)>=0
   assert abs(k.sum()-2*d)<1e-10

def test_unit_frozen_and_observables():
 for d,k in [(2,'stadium'),(3,'capsule')]:
  f,axis=shape(d,k,size=34 if d==2 else 20,radius=3)
  unchanged=f.copy();gd=derivative_filters(d,0,0,0);kj=exchange_filter(d,gd,kind='tied')
  n=run(f,3,gd,kj)
  assert np.array_equal(f,unchanged)
  assert np.max(abs(np.linalg.norm(n,axis=-1)-1))<1e-6
  Q1,Q2,T=observables(n,kj,gd)
  for q in (Q1,Q2,T):
   assert q.shape==(d,d) and np.allclose(q,q.T)
  assert abs(np.trace(Q1))<1e-6 and abs(np.trace(Q2))<1e-6
  assert np.isfinite(axis_mae(Q1,axis,'min')[0])

def test_axis_swap_covariance():
 f,axis=shape(3,'ellipsoid',size=22,radius=3,angle_deg=0)
 g=derivative_filters(3,-1/12,.12,.01);k=exchange_filter(3,g,kind='tied')
 n=run(f,3,g,k)
 fs=np.swapaxes(f,0,1);ns=run(fs,3,g,k)
 expected=np.swapaxes(n,0,1)[...,[1,0,2,3]]
 assert np.max(abs(expected-ns))<2e-5

def test_zero_field_safe():
 f=np.ones((8,8),dtype='float32');n=binary_edt_init(f)
 g=derivative_filters(2,0,0,0);k=exchange_filter(2,g)
 y=one_step(n,f,g,k,J=0,h=0,D=0)
 assert np.array_equal(n,y)
