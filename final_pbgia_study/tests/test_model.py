import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from data import *

def test_exact_pinned_reproduction():
 cases=json.load(open(ROOT/'reference/historical_reproduction_cases.json'));results=[]
 for row in cases:
  d=int(row['dim']);fixture_file=np.load(ROOT/'reference'/f'historical_regression_{d}d.npz');f=fixture_file['f'];expected=fixture_file['n'];n=initialize(f)
  for _ in range(30):n=old_step(n,f)
  assert np.array_equal(n,expected)
  value=np.trace(tensor(n,BASE['kernel_by_dimension'][str(d)]))*BASE['calibration_by_dimension'][str(d)];assert np.isclose(value,row['measured'],rtol=1e-12)
  results.append(dict(dim=d,case=row['case'],bitwise_state_match=True,prediction=float(value),historical_prediction=row['measured']))
 (OUT/'historical_reproduction.json').write_text(json.dumps(results,indent=2))

def test_unit_input_current_gradient_and_adjoint():
 rng=np.random.default_rng(7)
 for d in [2,3]:
  f=np.where(rng.random((10,)*d)>.5,1.,-1.).astype('float32');before=f.copy();n=initialize(f);H=effective(n,f,False);expected=H/np.linalg.norm(H,axis=-1,keepdims=True);a=old_step(n,f);assert np.array_equal(f,before);assert np.allclose(a,expected,atol=2e-7);assert abs(np.linalg.norm(a,axis=-1)-1).max()<2e-7
  s=rng.normal(size=f.shape);v=rng.normal(size=f.shape+(d,));lhs=np.sum(v*gradient(s));adj=-sum(.5*(np.roll(v[...,j],-1,j)-np.roll(v[...,j],1,j)) for j in range(d));assert abs(lhs-np.sum(s*adj))<1e-10

def test_variational_energy_gradient_and_monotonicity():
 rng=np.random.default_rng(2);f=np.where(rng.random((8,8))>.5,1.,-1.);n=initialize(f).astype(float);direction=rng.normal(size=n.shape);e=1e-6;fd=(physical_energies(n+e*direction,f)['total']-physical_energies(n-e*direction,f)['total'])/(2*e);expected=-np.sum(effective(n,f,True)*direction);assert np.isclose(fd,expected,atol=2e-5)
 # Base H lacks a nonzero reciprocal term and is not that full-energy gradient.
 assert np.linalg.norm(effective(n,f,False)-effective(n,f,True))>.1
 values=[]
 for _ in range(10):values.append(physical_energies(n,f)['total']);n,step=variational_step(n,f)
 values.append(physical_energies(n,f)['total']);assert np.max(np.diff(values))<=1e-8;assert np.max(abs(np.linalg.norm(n,axis=-1)-1))<1e-12

def test_split_nonoverlap_and_fresh_prototypes():
 df=manifest();assert df.groupby('unit').split.nunique().max()==1;assert (df[df.role=='reference'].split=='train').all();new=df[df.split=='test'][['dim','family','R']]
 for p in (ROOT/'reference').glob('*.csv'):
  old=pd.read_csv(p)
  if {'dim','family','R'}<=set(old):assert set(new.itertuples(index=False,name=None)).isdisjoint(set(old[['dim','family','R']].itertuples(index=False,name=None)))
 assert sha(OUT/'manifest.csv')==(OUT/'manifest.sha256').read_text()
