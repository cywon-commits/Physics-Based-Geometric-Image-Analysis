import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from joint_study_core import *

def test_e0_baseline_exact():
 for dim in [2,3]:
  from joint_core import shape
  f,_=shape(dim,'stadium' if dim==2 else 'capsule',radius=3,size=22)
  a=initialize(f);b=a.copy();c=parameter_config()
  for _ in range(30):a=step(a,f,c);b=old_step(b,f)
  assert np.array_equal(a,b)

def test_separable_exchange_and_generation_match_dense():
 rng=np.random.default_rng(17)
 for d in [2,3]:
  f=np.where(rng.random((9,)*d)>.5,1.,-1.).astype('float32');n=initialize(f);gp=[-.07,.12,.03];G=derivative_filters(d,*gp)
  for c in [parameter_config('axial',gp),parameter_config('tied',gp),parameter_config('tied',gp,normalized=False),parameter_config('free',gp,[.21,.1,.11])]:
   K=exchange_kernel(d,c);dense=np.stack([ndi.correlate(n[...,i],K,mode='wrap') for i in range(d+1)],-1)
   assert np.allclose(dense,exchange(n,c),atol=3e-6)
   if c['normalized'] or c['family']=='axial':assert abs(K.sum()-2*d)<1e-10
   assert K.min()>=0
  g=np.stack([ndi.correlate(n[...,-1],G[i],mode='wrap') for i in range(d)],-1);assert np.allclose(g,generated_gradient(n[...,-1],gp),atol=4e-7)

def test_bond_half_angle_identity_and_sign_invariance():
 rng=np.random.default_rng(18);n=rng.normal(size=(13,13,3));n/=np.linalg.norm(n,axis=-1,keepdims=True);c=parameter_config('free',[-.07,.1,.01],[.2,.1,.1]);v=n[...,:2];H=exchange(n,c)[...,:2];T0=np.sum(H[...,0]*v[...,0]-H[...,1]*v[...,1]);T1=np.sum(H[...,0]*v[...,1]+H[...,1]*v[...,0]);B,V,G=orientation_tensors(n,c,tensor(n,[0,0,0]));assert np.allclose(B,[[T0/2,T1/2],[T1/2,-T0/2]])
 u=np.array([np.cos(.5*np.arctan2(T1,T0)),np.sin(.5*np.arctan2(T1,T0))]);_,vec=np.linalg.eigh(B);assert abs(u@vec[:,-1])>1-1e-10
 flipped=n.copy();flipped[...,:2]*=-1;Bs,Vs,Gs=orientation_tensors(flipped,c,tensor(flipped,[0,0,0]));assert np.allclose(B,Bs);assert np.allclose(V,Vs);assert np.allclose(G,Gs)

def test_covariance_reflection_and_synchronous_frozen():
 rng=np.random.default_rng(19)
 for d in [2,3]:
  f=np.where(rng.random((9,)*d)>.5,1.,-1.).astype('float32');old=f.copy();n=initialize(f);c=parameter_config('free',[-.07,.1,.01],[.2,.1,.1]);perm=list(reversed(range(d)));nt=np.transpose(n,perm+[d])[...,perm+[d]];ft=np.transpose(f,perm);expected=np.transpose(step(n,f,c),perm+[d])[...,perm+[d]];assert np.max(abs(step(nt,ft,c)-expected))<2e-6
  reflected=np.flip(n,0).copy();reflected[...,0]*=-1;want=np.flip(step(n,f,c),0).copy();want[...,0]*=-1;assert np.max(abs(step(reflected,np.flip(f,0),c)-want))<2e-6
  out=step(n,f,c);assert np.array_equal(f,old);assert np.max(abs(np.linalg.norm(out,axis=-1)-1))<3e-7
  c0=dict(c,J=0,h=0,D=0);assert np.array_equal(step(n,f,c0),n)

def test_new_split_not_prior_geometries():
 df=make_manifest();prior=pd.read_csv(ROOT/'vendor/pbgia30/split_manifest.csv');assert set(df[['dim','family','R']].itertuples(index=False,name=None)).isdisjoint(set(prior[['dim','family','R']].itertuples(index=False,name=None)));assert df.groupby('unit').split.nunique().max()==1

def test_legacy_exact_source():
 f,_=__import__('joint_core').shape(2,'stadium',radius=4,size=36,angle_deg=27)
 for name in ['texture','length']:
  ours,Q,info=historical(f,name);func=source_legacy_function(name);kwargs=dict(kernel_dmx=old_kernel(),parameters=(info['D'],1,info['N']));raw=func(f.astype(float),**kwargs);want=raw[3][...,[1,0,2]];assert np.allclose(ours,want,atol=2e-12)

def test_two_cycle_not_convergence():
 grid=np.indices((8,8));n=np.zeros((8,8,3),dtype='float32');n[...,-1]=(-1.)**grid.sum(0);f=np.ones((8,8),dtype='float32');c=dict(parameter_config(),h=0.,D=0.)
 n1=step(n,f,c);n2=step(n1,f,c);assert np.array_equal(n2,n);assert np.linalg.norm(n1-n)>1

def test_calibration_and_freeze_if_present():
 df=make_manifest();assert set(df[df.role=='calibration'].split)=={'train'}
 p=OUT/'pretest_selection.json'
 if p.exists() and (OUT/'pretest_selection.sha256').exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==(OUT/'pretest_selection.sha256').read_text()
