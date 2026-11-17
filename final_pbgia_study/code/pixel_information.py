"""Bounded raster non-identifiability experiments; no fitting to pair truths."""
from common import *
from geometry_study import classical,scalar_zero_surface,check_lock
from scipy.integrate import quad
import resource

def radial_grid(dim,spacing,extent=48.,center=None):
 size=int(np.ceil(extent/spacing));size+=size%2
 center=np.asarray(center if center is not None else [.173,-.291,.113][:dim])
 x=(np.indices((size,)*dim,dtype=float)-(size-1)/2)*spacing
 dist=np.linalg.norm(x-center.reshape((dim,)+(1,)*dim),axis=0)
 return x,dist,center

def raster_ball(x,center,R):
 return np.linalg.norm(x-np.asarray(center).reshape((x.shape[0],)+(1,)*(x.ndim-1)),axis=0)<=R

def radius_interval(dist,R):
 low=float(dist[dist<=R].max());high=float(dist[dist>R].min());delta=high-low
 return low+1e-5*delta,high-1e-5*delta

def shape_size(dim,R):return 2*np.pi*R if dim==2 else 4*np.pi*R*R

def mask_hash(mask):return hashlib.sha256(np.asarray(mask,np.uint8).tobytes()).hexdigest()

def fixed_estimators(mask,spacing,selection,axis=None):
 d=mask.ndim;f=np.where(mask,1.,-1.).astype('float32');tic=time.perf_counter();n=initialize(f)
 for _ in range(30):n=old_step(n,f)
 field_time=time.perf_counter()-tic;r,_=measure(n,axis);factor=spacing**(d-1);q=selection[str(d)];s0,s0time=scalar_zero_surface(n[...,-1],spacing)
 result={'energy_original_c':BASE['calibration_by_dimension'][str(d)]*r['I_E']*factor,'energy_train_c':q['c_E']*r['I_E']*factor,'TV_raw':r['I_TV']*factor,'TV_train_c':q['c_TV']*r['I_TV']*factor,'selected':(q['alpha']*q['c_E']*r['I_E']+(1-q['alpha'])*q['c_TV']*r['I_TV'])*factor,'generated_s0':s0}
 result.update({c['method']:c['value'] for c in classical(f,axis,spacing) if c['method']!='binary_PCA'})
 coords=np.array(np.nonzero(mask),float).T*spacing-(np.array(mask.shape)-1)*spacing/2
 centroid=coords.mean(0);return result,dict(field_seconds=field_time,measurement_seconds=r['measurement_seconds']+s0time,energy_axis_error=r['energy_axis_error'],spatial_axis_error=r['spatial_axis_error'],binary_centroid=centroid)

def ellipse_mask(x,R,theta,center):
 y=x-np.asarray(center).reshape((2,1,1));u=np.cos(theta)*y[0]+np.sin(theta)*y[1];v=-np.sin(theta)*y[0]+np.cos(theta)*y[1]
 return (u/(1.8*R))**2+(v/(.65*R))**2<=1

def same_angle_endpoint(x,R,theta,center,direction):
 original=ellipse_mask(x,R,theta,center);lo=0.;hi=1e-5
 while hi<.5 and np.array_equal(ellipse_mask(x,R,theta+direction*hi,center),original):lo=hi;hi*=1.6
 for _ in range(45):
  mid=(lo+hi)/2
  if np.array_equal(ellipse_mask(x,R,theta+direction*mid,center),original):lo=mid
  else:hi=mid
 return theta+direction*lo*.999

def run():
 assert not (OUT/'pixel_pairs.csv').exists(),'Diagnostic already computed'
 frozen=check_lock();selection=frozen['selection'];tic=time.perf_counter();records=[];orientation=[];corrugation=[];checks=[]
 for d in [2,3]:
  for a in [1.,.75,.5]:
   x,dist,c=radial_grid(d,a)
   for R in [2.641,4.641,8.641,12.641]:
    mask=dist<=R;lo,hi=radius_interval(dist,R);margin=float(abs(dist-R).min());delta=.2*margin
    estimates,cost=fixed_estimators(mask,a,selection)
    for kind,ca,cb,ra,rb in [('radius_interval',c,c,lo,hi),('center_and_radius',c-np.eye(d)[0]*delta,c+np.eye(d)[0]*delta,R-delta,R+delta)]:
     ma=raster_ball(x,ca,ra);mb=raster_ball(x,cb,rb);assert np.array_equal(mask,ma) and np.array_equal(mask,mb)
     pa,pb=shape_size(d,ra),shape_size(d,rb);lower=abs(pa-pb)/2;relative=100*lower/((pa+pb)/2)
     for method,value in estimates.items():records.append(dict(dim=d,spacing=a,R=R,R_pixels=R/a,pair_kind=kind,R_low=ra,R_high=rb,center_low=json.dumps(ca.tolist()),center_high=json.dumps(cb.tolist()),truth_low=pa,truth_high=pb,min_curvature_radius=min(ra,rb),mask_sha256=mask_hash(mask),identical_mask=True,identical_prediction=True,method=method,estimate=value,error_low_percent=100*(value/pa-1),error_high_percent=100*(value/pb-1),unavoidable_absolute_error=lower,unavoidable_relative_percent=relative,worst_absolute_error=max(abs(value-pa),abs(value-pb)),excess_over_two_shape_bound=max(abs(value-pa),abs(value-pb))-lower,centroid_low_error=float(np.linalg.norm(cost['binary_centroid']-ca)),centroid_high_error=float(np.linalg.norm(cost['binary_centroid']-cb)),field_seconds=cost['field_seconds'],measurement_seconds=cost['measurement_seconds']))
    # The same binary array is the actual deterministic input. Check a second
    # complete field calculation for one pair per dimension, rather than pretend
    # that duplicate evaluations are independent experimental samples.
    if a==1 and R==4.641:
     again,_=fixed_estimators(raster_ball(x,c,hi),a,selection);assert all(value==again[k] for k,value in estimates.items());checks.append(dict(dim=d,bitwise_equal_estimator_outputs=True,mask_sha256=mask_hash(mask)))
    if d==2 and a==1 and R==8.641:
     eps=.25*margin;angle=np.arctan2(x[1]-c[1],x[0]-c[0])
     for k in [4,32,256,2048,16384]:
      corr=dist<=R+eps*np.cos(k*angle);assert np.array_equal(corr,mask)
      # k equal periods give this one-period exact integral after substitution.
      perimeter=quad(lambda z:np.hypot(R+eps*np.cos(z),eps*k*np.sin(z)),0,2*np.pi,epsabs=1e-10,limit=1000)[0]
      corrugation.append(dict(R=R,epsilon=eps,frequency=k,truth_perimeter=perimeter,base_perimeter=2*np.pi*R,ratio_to_circle=perimeter/(2*np.pi*R),identical_mask=True,mask_sha256=mask_hash(mask),selected_estimate=estimates['selected'],curvature_constraint='NOT_BOUNDED; pathological high frequency family'))
    print('PIXEL',d,a,R,flush=True)
 # Orientation ambiguity is independent of the perimeter ambiguity.
 for R in [4.641,8.641]:
  x,_,c=radial_grid(2,1);theta=.437;mask=ellipse_mask(x,R,theta,c);lo=same_angle_endpoint(x,R,theta,c,-1);hi=same_angle_endpoint(x,R,theta,c,1);assert np.array_equal(mask,ellipse_mask(x,R,lo,c));assert np.array_equal(mask,ellipse_mask(x,R,hi,c))
  axis=np.array([np.cos(theta),np.sin(theta)]);est,cost=fixed_estimators(mask,1.,selection,axis);pts=np.array(np.nonzero(mask),float).T;_,vec=np.linalg.eigh(np.cov(pts.T));pca=vec[:,-1];f=np.where(mask,1.,-1.).astype('float32');n=initialize(f)
  for _ in range(30):n=old_step(n,f)
  _,matrices=measure(n,axis)
  for method,M,which in [('binary_PCA',np.cov(pts.T),'max'),('energy_normal_tensor',matrices['energy'],'min'),('field_spatial_covariance',matrices['spatial'],'max')]:
   values,vec=np.linalg.eigh(M);v=vec[:,0 if which=='min' else -1];err=lambda t:np.degrees(np.arccos(np.clip(abs(v@np.array([np.cos(t),np.sin(t)])),0,1)))
   orientation.append(dict(R=R,theta_low=lo,theta_high=hi,unavoidable_angle_degrees=np.degrees(hi-lo)/2,method=method,error_low_degrees=err(lo),error_high_degrees=err(hi),estimated_axis=json.dumps(v.tolist()),identical_mask=True,mask_sha256=mask_hash(mask)))
 pd.DataFrame(records).to_csv(OUT/'pixel_pairs.csv',index=False);pd.DataFrame(orientation).to_csv(OUT/'pixel_orientation_pairs.csv',index=False);pd.DataFrame(corrugation).to_csv(OUT/'pixel_pathological.csv',index=False)
 dump('pixel_audit.json',dict(mask_pairs=len({(r['dim'],r['spacing'],r['R'],r['pair_kind']) for r in records}),unique_base_fields=24,duplicate_determinism_checks=checks,wall_seconds=time.perf_counter()-tic,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,policy='Fixed dimensionless pixel model at each spacing; perimeter/area rescaled by a^(d-1); no pair truth calibration, no claim of full error decomposition'))
 print('PIXEL COMPLETE',flush=True)

if __name__=='__main__':run()
