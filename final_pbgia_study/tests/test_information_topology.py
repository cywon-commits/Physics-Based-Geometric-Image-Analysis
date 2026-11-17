import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from pixel_information import radial_grid,raster_ball,radius_interval,shape_size
from common import *
from projective import loop_invariant,eigendirector
from topology_study import synthetic_degree_field,audit_degree

def test_same_raster_different_truth_and_minimax_bound():
 for d in [2,3]:
  x,dist,c=radial_grid(d,.75,extent=12);R=2.641;mask=dist<=R;lo,hi=radius_interval(dist,R)
  assert hi>lo and np.array_equal(mask,raster_ball(x,c,lo)) and np.array_equal(mask,raster_ball(x,c,hi))
  margin=float(abs(dist-R).min());delta=.2*margin;e0=np.eye(d)[0]
  assert np.array_equal(mask,raster_ball(x,c-delta*e0,R-delta)) and np.array_equal(mask,raster_ball(x,c+delta*e0,R+delta))
  a,b=shape_size(d,lo),shape_size(d,hi)
  for estimate in [a,b,(a+b)/2,0.]:assert max(abs(estimate-a),abs(estimate-b))>=abs(a-b)/2-1e-12

def test_null_tensor_is_unidentified_and_loop_is_not_a_signed_flow():
 z=np.zeros((128,3,3));a=eigendirector(z);assert not a['defined'].any();assert np.isnan(a['purity']).all();assert loop_invariant(z)['status']=='INCONCLUSIVE'

def test_s3_degree_sign_and_constant_controls():
 for mirror,expected in [(False,1),(True,-1)]:
  r=audit_degree(synthetic_degree_field(size=17,radius=4.31,mirror=mirror),'test');assert r['status']=='NUMERICALLY_SUPPORTED';assert r['degree']==expected

def test_zero_effective_field_retains_existing_spin():
 f=np.ones((7,7),np.float32);n=np.zeros((7,7,3),np.float32);n[...,2]=-1
 for a,b,sign in [(2,3,1),(4,3,-1),(3,2,1),(3,4,-1)]:n[a,b]=[sign*np.sqrt(.75),0,-.5]
 assert np.linalg.norm(effective(n,f)[3,3])<1e-7
 assert np.array_equal(old_step(n,f)[3,3],n[3,3])

def test_parallel_alignment_is_not_guaranteed_energy_descent():
 sigma=np.where(np.indices((8,8)).sum(0)%2,1.,-1.);n=np.zeros((8,8,3));n[...,0]=np.sqrt(1-.7**2)*sigma;n[...,-1]=.7*sigma
 assert physical_energies(old_step(n,sigma),sigma)['total']-physical_energies(n,sigma)['total']>sigma.size
 assert physical_energies(variational_step(n,sigma)[0],sigma)['total']<physical_energies(n,sigma)['total']

def test_freeze_manifest_and_heldout_once_audit():
 from geometry_study import check_lock
 frozen=check_lock();a=json.load(open(OUT/'test_audit.json'));assert a['freeze_sha256']==sha(OUT/'model_freeze.json');assert a['no_retuning'];assert a['images']==144
