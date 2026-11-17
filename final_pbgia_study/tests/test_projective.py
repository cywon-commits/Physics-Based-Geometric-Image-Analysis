import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from projective import *

def test_pointwise_sign_invariance_rank_one_and_mixed_tensor():
 rng=np.random.default_rng(7);n=rng.normal(size=(6,7,8,4));n/=np.linalg.norm(n,axis=-1,keepdims=True);sign=rng.choice([-1,1],n.shape[:-1]);Q=n[..., :,None]*n[...,None,:];assert np.allclose(Q,(n*sign[...,None])[..., :,None]*(n*sign[...,None])[...,None,:]);assert np.allclose(Q@Q,Q);assert np.allclose(np.trace(Q,axis1=-2,axis2=-1),1)
 for weight in ['uniform','wall']:
  a,_=fiber_tensor(n,2,'full',weight);b,_=fiber_tensor(n*sign[...,None],2,'full',weight);assert np.array_equal(a,b);assert not np.allclose(a@a,a)
 mixed=np.diag([.5,.5,0.]);a=eigendirector(mixed);assert not a['defined'];assert np.isclose(a['purity'],.5)

def test_RP1_integer_winding_vs_RP2_Z2_controls_and_gauge():
 c=controls();assert c.correct.all();assert len(c)==10
 phi=np.arange(128)*2*np.pi/128;d=np.stack([np.cos(phi/2),np.sin(phi/2),0*phi],-1);Q=d[..., :,None]*d[...,None,:];r=loop_invariant(Q);rng=np.random.default_rng(9);d*=rng.choice([-1,1],(128,1));assert np.array_equal(Q,d[..., :,None]*d[...,None,:]);assert r['z2_holonomy']==-1
 # No signed vector can be recovered uniquely from projector data.
 assert np.array_equal(d[..., :,None]*d[...,None,:],(-d)[..., :,None]*(-d)[...,None,:])

def test_fiber_axis_projection_and_numerical_rank():
 n=np.zeros((4,5,6,4));n[...,3]=1;Q,m=fiber_tensor(n,0,'full','uniform');a=eigendirector(Q);assert Q.shape==(5,6,4,4);assert a['defined'].all();assert np.allclose(a['purity'],1)
 q,_=fiber_tensor(n,1,'vector','wall');b=eigendirector(q);assert not b['defined'].any();assert np.isnan(b['purity']).all()
