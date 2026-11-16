import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from study_core import *

def test_synchronous_dynamic_map_and_immutable_input():
 rng=np.random.default_rng(27)
 for dim in [2,3]:
  f=np.where(rng.random((9,)*dim)>.5,1,-1).astype('float32');before=f.copy();n=initialize(f);H=sum(np.roll(n,1,k)+np.roll(n,-1,k) for k in range(dim));H[...,:dim]+=gradient(n[...,-1]);H[...,-1]+=2*f;H/=np.linalg.norm(H,axis=-1,keepdims=True)
  assert np.allclose(one_step(n,f),H,atol=2e-7);assert np.array_equal(f,before)

def test_feature_tensor_matches_full_measurement():
 rng=np.random.default_rng(28)
 for dim in [2,3]:
  n=rng.normal(size=(7,)*dim+(dim+1,)).astype('float32');n/=np.linalg.norm(n,axis=-1,keepdims=True);F=features(n)
  for p in [[0,0,0],[-1/12,0,0],[-.07,.12,.01]]:assert np.allclose(feature_tensor(F,p,dim),tensor(n,p),rtol=3e-6,atol=2e-5)

def test_moments_and_axes():
 for dim in [2,3]:
  grid=np.indices((17,)*dim,dtype='float32')
  for p in [[0,0,0],[-1/12,0,0],[-.07,.12,.02]]:
   for k in range(dim):
    assert np.allclose(derivative(np.ones((17,)*dim+(1,),dtype='float32'),k,p),0)
    got=derivative(grid[k][...,None],k,p)[(slice(4,-4),)*dim];assert np.allclose(got,1)
  f=np.where(grid[0]<8,1.,-1.).astype('float32');n=initialize(f);perm=list(reversed(range(dim)));nt=np.transpose(n,perm+[dim])[...,perm+[dim]];ft=np.transpose(f,perm);expected=np.transpose(one_step(n,f),perm+[dim])[...,perm+[dim]]
  assert np.max(abs(one_step(nt,ft)-expected))<4e-7

def test_two_cycle_smoke():
 idx=np.indices((8,8));n=np.zeros((8,8,3),dtype='float32');n[...,2]=(-1.)**idx.sum(0);f=np.ones((8,8),dtype='float32');n1=one_step(n,f,h=0,D=0);n2=one_step(n1,f,h=0,D=0)
 assert np.array_equal(n2,n);assert np.linalg.norm(n1-n)>1

def test_split_and_train_only_calibration():
 from run_fixed_study import manifest
 df=manifest();assert df.groupby('unit').split.nunique().max()==1;assert set(df[df.role=='calibration'].split)=={'train'}
 for family,g in df.groupby(['dim','family']):
  assert g.groupby('R').split.nunique().max()==1
 assert set(df[df.family.isin(['ring','ellipsoid'])].split)=={'test'}

def test_smoke_restore_and_exact_reproduction(tmp_path):
 from fixed_iteration_checkpoints import generate,synthetic_mask
 for dim,size in [(2,40),(3,24)]:
  f=synthetic_mask(dim,size);snaps,_=generate(f)
  for N in CHECKPOINTS:
   saved=np.load(ROOT/f'smoke_{dim}d/checkpoint_{N:02d}.npz');assert np.array_equal(snaps[N],saved['n'])
  path=tmp_path/f'{dim}.npz';np.savez_compressed(path,n=snaps[10],f=f);z=np.load(path);n=z['n']
  for _ in range(10):n=one_step(n,z['f'])
  assert np.array_equal(n,snaps[20])

def test_cache_key_and_reuse():
 row=dict(case='test_cache',unit='test_cache',split='diagnostic',role='test',family='circle',dim=2,R=3.,size=24,rep=0,phase=.1,noise=0.,seed=1,noise_seed=2,spacing=1.)
 a=checkpoints(row,PARAMS);b=checkpoints(row,PARAMS)
 for N in CHECKPOINTS:assert np.array_equal(a['snaps'][N],b['snaps'][N])
 assert a['key']==b['key'];assert key_for(a['f'],dict(J=1.,h=4.,D=1.))!=a['key']

def test_frozen_calibration_train_only_if_available():
 p=OUT/'kernel_by_iteration.json'
 if not p.exists():return
 data=json.load(open(p));df=pd.read_csv(OUT/'split_manifest.csv')
 for r in data.get('calibration',[]):assert df.set_index('case').loc[r['reference_case'],'split']=='train'
 cfg=OUT/'selected_configuration.json'
 if cfg.exists() and (OUT/'selection_lock.sha256').exists():assert hashlib.sha256(cfg.read_bytes()).hexdigest()==(OUT/'selection_lock.sha256').read_text()
