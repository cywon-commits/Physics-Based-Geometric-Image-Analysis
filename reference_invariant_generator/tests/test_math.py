import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from core import *
import torch
from scipy.signal import convolve2d,correlate2d

def test_kernel_embedding_and_sign():
 for d in [2,3]:
  p=G4 if d==2 else np.r_[G4[:2],np.zeros(10)];K=kernels(p,d);coords=np.indices((5,)*d)-2
  for j in range(d):
   assert abs(K[j].sum())<1e-14;assert abs((K[j]*coords[j]).sum()-1)<1e-14
   for a in range(d):assert np.array_equal(np.flip(K[j],a),(-1 if a==j else 1)*K[j])
  perm=tuple(reversed(range(d)))
  for j in range(d):assert np.allclose(K[j].transpose(perm),K[perm[j]])
 s=np.random.default_rng(1).normal(size=(11,11));K=kernels(LEGACY);a=derivative(s[None],LEGACY)[0]
 for j in range(2):
  assert np.allclose(a[...,j],correlate2d(s,K[j],mode='same',boundary='wrap'));assert np.allclose(a[...,j],convolve2d(s,-K[j],mode='same',boundary='wrap'))
 ramp=np.indices((11,11))[1].astype(float);assert np.allclose(derivative(ramp[None],G4)[0,2:-2,2:-2,1],1.)
 assert not np.isclose(np.sum(K[1]*np.indices((5,5))[1]),1)

def torch_rollout(f,n,p,N):
 ns=torch.tensor(n);ff=torch.tensor(f);pp=torch.as_tensor(p,dtype=torch.float64);basis=torch.tensor(np.stack([kernels(np.eye(6)[i]) for i in range(6)]));K=torch.einsum('p,pijk->ijk',pp,basis)
 for _ in range(N):
  H=sum(torch.roll(ns,k,a) for a in [1,2] for k in [-1,1]);ss=ns[...,-1];gs=[]
  for ax in range(2):gs.append(sum(K[ax,i,j]*torch.roll(ss,(-(i-2),-(j-2)),(1,2)) for i in range(5) for j in range(5)))
  H=H+torch.cat((torch.stack(gs,-1),(2*ff)[...,None]),-1);norm=torch.linalg.vector_norm(H,dim=-1,keepdim=True);ns=torch.where(norm>1e-12,H/norm.clamp_min(1e-12),ns)
 gs=[(2/3)*(torch.roll(ns,-1,a)-torch.roll(ns,1,a))-(1/12)*(torch.roll(ns,-2,a)-torch.roll(ns,2,a)) for a in [1,2]];I=sum((g*g).sum((1,2,3)) for g in gs);return I,ns

def test_exact_jacobian_autograd_and_fd():
 rng=np.random.default_rng(2);f=np.where(rng.random((1,12,12))>.7,1.,-1.);n=np.stack([init(f[0])]);p=G4+np.array([.03,.01,.02,-.01,.01,.02]);before=f.copy();a=forward(f,n,p,3,True);pt=torch.tensor(p,requires_grad=True);I,ns=torch_rollout(f,n,pt,3);I.sum().backward();assert np.allclose(a['I'],I.detach().numpy(),rtol=1e-12);assert np.allclose(a['dI'][0],pt.grad.numpy(),rtol=1e-10,atol=1e-10);assert np.array_equal(f,before);assert a['unit_error']<1e-12
 for j in range(6):
  delta=np.eye(6)[j]*1e-5;fd=(forward(f,n,p+delta,3)['I']-forward(f,n,p-delta,3)['I'])/(2e-5);assert np.allclose(fd,a['dI'][:,j],rtol=2e-6,atol=1e-6)

def test_reference_free_logic_and_split():
 rows=pd.DataFrame(dict(unit=['a','a','b','b'],noise=[0]*4,rep=[0,1,0,1],truth=[2,2,4,4]));ax=[None]*4;theta=torch.zeros(4,dtype=torch.float64)
 values=[]
 for scale in [1.,7.3]:
  x=torch.tensor([3,3,6,6],dtype=torch.float64)*scale;_,v=metrics_torch(x,theta,rows,ax,0);values.append(v['prop'])
 assert max(values)<1e-24
 x=torch.tensor([3,3,7,7],dtype=torch.float64);_,a=metrics_torch(x,theta,rows,ax,0);_,b=metrics_torch(x*7,theta,rows,ax,0);assert abs(a['prop']-b['prop'])<1e-15
 df=manifest();assert df.groupby('unit').split.nunique().max()==1;assert (df[df.role=='reference'].split=='train').all()
 old=pd.read_csv(ROOT/'reference/RADIUS_SENSITIVITY_HISTORICAL.csv');assert set(df.query("split=='test' and family=='circle'").R).isdisjoint(set(old.ref_radius))
 for p in (ROOT/'reference/old_manifests').glob('*.csv'):
  if p.exists():
   old=pd.read_csv(p);pairs=set(old[['family','R']].itertuples(index=False,name=None));assert set(df.query("split=='test'")[['family','R']].itertuples(index=False,name=None)).isdisjoint(pairs)

def test_baseline_previous_path():
 import importlib.util
 spec=importlib.util.spec_from_file_location('old',ROOT/'reference/previous_operators.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 f=np.where(np.random.default_rng(1).random((16,16))>.8,1.,-1.).astype('float32');n=old.initialize(f);previous=n.copy();c=old.parameter_config('axial',[-1/12,0,0])
 for _ in range(10):previous=old.step(previous,f,c)
 current=forward(f[None],n[None].astype(float),G4,10)['state'][0];assert np.allclose(current,previous,atol=2e-6)

def test_chunked_reverse_gradient_matches_exact_forward():
 from torch_engine import objective_autograd
 df=manifest();r=pd.concat([df.query("family=='circle' and split=='train' and role=='geometry'").head(4),df.query("family=='ellipse' and split=='train' and role=='geometry'").head(4)]);r=r.copy();r['size']=32;ds=load(r);a=objective(ds,G4,1);b=objective_autograd(ds,G4,1);assert np.allclose(a[0],b[0],atol=1e-12);assert np.allclose(a[1],b[1],rtol=1e-9,atol=1e-9)

def test_loss_exact_gradient_and_frozen_guard():
 from torch_engine import objective_autograd
 r=manifest().query("family=='ellipse' and split=='train' and role=='geometry' and rotation_id==0 and shift_id==0").iloc[:4].copy();r['size']=32;ds=load(r);p=G4+np.array([.03,.01,.02,.01,.01,.02]);value,grad,_=objective_autograd(ds,p,1)
 for j in [0,3,5]:
  dp=np.eye(6)[j]*1e-5;v1=objective_autograd(ds,p+dp,1)[0];v2=objective_autograd(ds,p-dp,1)[0];assert np.isclose((v1-v2)/2e-5,grad[j],rtol=1e-4,atol=1e-7)
 if (OUT/'model_freeze_2d.json').exists():
  from evaluation import check_lock
  f=check_lock();assert not f['test_seen']
  if (OUT/'test_audit_2d.json').exists():assert json.load(open(OUT/'test_audit_2d.json'))['selection_sha256']==hashlib.sha256((OUT/'model_freeze_2d.json').read_bytes()).hexdigest()

def test_original_notebook_mapping_and_zero_field_branch():
 p=LEGACY;old=np.array([[p[5],p[4],0,-p[4],-p[5]],[p[3],p[2],0,-p[2],-p[3]],[p[1],p[0],0,-p[0],-p[1]],[p[3],p[2],0,-p[2],-p[3]],[p[5],p[4],0,-p[4],-p[5]]]);assert np.array_equal(old,-kernels(p)[1])
 f=np.ones((1,9,9));n=np.zeros((1,9,9,3));n[...,-1]=1.;n[0,3,4,-1]=-1;n[0,5,4,-1]=-1;n[0,4,3,-1]=-1;n[0,4,5,-1]=1;n[0,4,4,-1]=-1
 a=forward(f,n,np.zeros(6),1);assert np.array_equal(a['state'][0,4,4],n[0,4,4])

def test_all_three_d_orbit_covariance():
 rng=np.random.default_rng(33);p=rng.normal(size=12);K=kernels(p,3)
 for perm in itertools.permutations(range(3)):
  for j in range(3):assert np.allclose(K[perm[j]].transpose(perm),K[j])
 for j in range(3):
  for axis in range(3):assert np.allclose(np.flip(K[j],axis),(-1 if axis==j else 1)*K[j])

def test_frozen_coefficients_cases_and_no_repeat(monkeypatch):
 if not (OUT/'model_freeze_2d.json').exists():return
 import evaluation
 f=evaluation.check_lock();df=manifest()
 if (OUT/'test_raw_2d.csv').exists():
  r=pd.read_csv(OUT/'test_raw_2d.csv')
  for name,g in r.groupby('model'):
   assert len(g)==408;assert not g.case.duplicated().any();assert set(g.case)==set(df.query("split=='test'")['case']);assert np.allclose(g.q,g.I/g.truth);assert g.unit_error.max()<1e-10
  import pytest
  with pytest.raises(AssertionError):evaluation.test()
  with pytest.raises(AssertionError):evaluation.validate()
  with pytest.raises(AssertionError):evaluation.freeze()
 import importlib
 cli=importlib.import_module('infer');row=df.query("role=='reference'").iloc[0].to_dict();mask,_,_=fixture(row);a=cli.predict(mask);assert a['N']==10;assert np.isfinite(a['I']);assert a['I']>0
