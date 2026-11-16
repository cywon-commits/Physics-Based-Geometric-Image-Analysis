import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from engine import *
from data import make_manifest,rotation

def test_moments():
 for d in [2,3]:
  for p in [[0,0,0],[-1/12,0,0],[-.07,.13,.02]]:
   G=derivative_filters(d,*p);x=np.indices((5,)*d)-2
   assert np.max(abs(G.sum(tuple(range(1,d+1)))))<1e-12
   for i in range(d):
    for j in range(d):assert abs((G[i]*x[j]).sum()-(i==j))<1e-12

def test_baseline_exact_and_frozen_f():
 for d in [2,3]:
  rng=np.random.default_rng(d);f=np.where(rng.random((12,)*d)>.5,1.,-1.).astype('float32');old=f.copy();a=initialize(f);b=a.copy()
  for i in range(5):a=update(a,f,parameter_config());b=old_step(b,f)
  assert np.max(abs(a-b))<2e-6
  assert np.array_equal(f,old);assert np.max(abs(np.linalg.norm(a,axis=-1)-1))<2e-6
  c=parameter_config();c.update(J=0.,D=0.,h=0.);assert np.array_equal(update(a,f,c),a)

def test_covariance():
 for d in [2,3]:
  f=np.random.default_rng(d).normal(size=(9,)*d).astype('float32');p=[-.07,.13,.02];g=generated_gradient(f,p);perm=list(reversed(range(d)))
  gp=generated_gradient(f.transpose(perm),p)
  for i in range(d):assert np.allclose(gp[...,i],g[...,perm[i]].transpose(perm),atol=2e-6)
  gr=generated_gradient(np.flip(f,0),p)
  for i in range(d):assert np.allclose(gr[...,i],np.flip(g[...,i],0)*(-1 if i==0 else 1),atol=2e-6)
  for family in ['axial','tied','free']:
   c=parameter_config(family,p,[.2,.03,.04]);k=exchange_kernel(d,c);assert np.min(k)>=0;assert abs(k.sum()-2*d)<1e-12
  Q=rotation(d,911);assert np.allclose(Q.T@Q,np.eye(d));assert abs(np.linalg.det(Q)-1)<1e-10

def test_split():
 df=make_manifest();assert df.groupby('unit').split.nunique().max()==1
 for fn in ['previous_fixed_split.csv','previous_joint_split.csv']:
  old=__import__('pandas').read_csv(ROOT/'reference'/fn)
  assert set(df[df.role=='geometry'][['dim','family','R']].itertuples(index=False,name=None)).isdisjoint(set(old[['dim','family','R']].itertuples(index=False,name=None)))
 assert hashlib.sha256((OUT/'manifest.csv').read_bytes()).hexdigest()==(OUT/'split_lock.sha256').read_text()

def test_hadamard_scale_identity_and_zero_h():
 for d in [2,3]:
  rng=np.random.default_rng(d);n=rng.normal(size=(10,)*d+(d+1,)).astype('float32');n/=np.linalg.norm(n,axis=-1,keepdims=True);f=np.where(rng.random((10,)*d)>.5,1.,-1.).astype('float32')
  raw=parameter_config('tied',[-.08,.12,.03],normalized=False);scale=exchange_kernel(d,raw).sum()/(2*d);normalized=parameter_config('tied',raw['gp']);normalized.update(h=2/scale,D=1/scale)
  assert np.allclose(update(n,f,raw),update(n,f,normalized),atol=2e-6)
  assert np.allclose(exchange_kernel(d,raw),(derivative_filters(d,*raw['gp'])**2).sum(0))

def test_independent_heads_and_axis_conventions():
 d=2;f=np.where(np.indices((20,20))[0]<8,1.,-1.).astype('float32');n=initialize(f);c=parameter_config();h=dict(geometry=dict(p=[0,0,0]),orientation=dict(p=[-.08,.1,.02],type='gradient',which='min',mix=[1,0,0]));a,_=readout(n,c,h,1,np.array([0,1.]));h2=json.loads(json.dumps(h));h2['orientation']['p']=[0,.25,0];b,_=readout(n,c,h2,1,np.array([0,1.]));assert a['prediction']==b['prediction']
 T=np.array([[1,.3],[.3,2.]]);v=np.linalg.eigh(T)[1][:,0];angle=.5*np.arctan2(2*T[0,1],T[0,0]-T[1,1])+np.pi/2;assert abs(abs(np.array([np.cos(angle),np.sin(angle)])@v)-1)<1e-10
 for d in [2,3]:
  row=make_manifest().query('dim==@d').iloc[0].to_dict();case=rollout(row,c,[5]);assert set(case['states'])=={5}

def test_no_old_screening_and_new_split():
 import ast
 banned={'dice','iou','euler','components','degree','wall_width','clean_iou','chi'}
 for fn in ['study.py','evaluate.py','engine.py','observer.py']:
  tree=ast.parse((ROOT/'code'/fn).read_text());names={n.id.lower() for n in ast.walk(tree) if isinstance(n,ast.Name)};assert not names&banned
 df=make_manifest();old=__import__('pandas').read_csv(ROOT/'reference/previous_geometry_only_split.csv');assert set(df[df.role=='geometry'][['dim','family','R']].itertuples(index=False,name=None)).isdisjoint(set(old[['dim','family','R']].itertuples(index=False,name=None)))

def test_frozen_artifacts_and_train_calibration():
 import pandas as pd
 p=OUT/'pretest_selection.json'
 if not p.exists():return # pre-training test stage
 lock=hashlib.sha256(p.read_bytes()).hexdigest();assert lock==(OUT/'pretest_selection.sha256').read_text();frozen=json.load(open(p))
 for fn,sha in frozen['hashes'].items():assert hashlib.sha256((OUT/fn).read_bytes()).hexdigest()==sha
 cal=pd.read_csv(OUT/'calibration.csv');refs=set(make_manifest().query("role=='reference' and split=='train'")['case']);assert set(cal.reference_case)<=refs
 for d,configs in frozen['configurations'].items():
  for name,c in configs.items():
   if 'fixed_B' in name:
    assert c['N']==configs['B']['N'];assert c['heads']==configs['B']['heads']
   if c['model'] not in ['A']:
    mode=name.split('_',1)[1] if 'fixed_B' in name else 'retuned';r=cal[(cal.dim==int(d))&(cal.model==c['model'])&(cal.N==c['N'])&(cal['mode']==mode)];assert len(r)==1;assert np.isclose(r.coefficient.iloc[0],c['coefficient'],rtol=1e-13)
 if (OUT/'test_audit.json').exists():
  assert json.load(open(OUT/'test_audit.json'))['selection_sha256']==lock
  r=pd.read_csv(OUT/'test_results.csv');manifest=make_manifest()
  for (d,name),g in r.groupby(['dim','model']):
   assert set(g.case)==set(manifest.query("dim==@d and split=='test'")['case']);assert len(g)==96;assert g.finite.all();assert g.unit_error.max()<2e-5

def test_cli_exact_fixed_N(monkeypatch):
 if not (OUT/'pretest_selection.json').exists():return
 import diagnostics,importlib
 cli=importlib.import_module('infer');original=diagnostics.update;calls=[]
 def counted(n,f,c):calls.append(1);return original(n,f,c)
 monkeypatch.setattr(diagnostics,'update',counted)
 for d in [2,3]:
  row=make_manifest().query("dim==@d and role=='reference'").iloc[0].to_dict();f,_,_,_=fixture(row);before=f.copy();calls.clear();prediction=cli.predict(f,preset='default');assert len(calls)==prediction['fixed_N'];assert np.array_equal(f,before);assert np.isfinite(prediction['geometry'])
