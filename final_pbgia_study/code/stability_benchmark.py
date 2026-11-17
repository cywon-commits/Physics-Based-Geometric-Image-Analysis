"""Fixed descriptive controls, never inputs to alpha/model selection."""
from data import *
from geometry_study import classical,scalar_zero_surface
from physics import normal_alignment

def run():
 # Isolated shift/rotation factors supplement the jointly perturbed main set.
 rows=[]
 for d in [2,3]:
  S=80 if d==2 else 64;R=7.431;axes=R*np.array([1.8,.65] if d==2 else [1.8,.81,.59]);x=np.indices((S,)*d,dtype=float)-(S-1)/2
  for k,angle in enumerate([0.,.3,.7,1.1]):
   Q=np.eye(d);Q[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
   if d==3:
    t=.6*angle;P=np.array([[np.cos(t),0,np.sin(t)],[0,1,0],[-np.sin(t),0,np.cos(t)]]);Q=P@Q
   axis=Q[:,0]
   for j,shift in enumerate([np.zeros(d),np.full(d,.23),np.full(d,-.31),np.array([.41,-.17,.13][:d])]):
    y=np.einsum('ji,j...->i...',Q,x-shift.reshape((d,)+(1,)*d));level=1-np.sum((y/axes.reshape((d,)+(1,)*d))**2,axis=0);f=np.where(level>=0,1.,-1.).astype('float32');n=initialize(f)
    for _ in range(30):n=old_step(n,f)
    r,_=measure(n,axis);rows.append(dict(dim=d,R=R,rotation_id=k,angle=angle,shift_id=j,shift=json.dumps(shift.tolist()),**r))
 pd.DataFrame(rows).to_csv(OUT/'controlled_rotation_shift.csv',index=False)
 # One a-priori sigma=1 pixel Gaussian baseline, raw measurements only.
 # This is a conventional descriptive comparator, not a selected PBGIA candidate.
 traditional=[]
 for row in manifest().query("split=='test'").to_dict('records'):
  f,truth,axis,aux=fixture(row);t=time.perf_counter();smoothed=ndi.gaussian_filter((f>0).astype(float),1.,mode='constant')>=.5;smoothing=time.perf_counter()-t
  for value in classical(np.where(smoothed,1.,-1.).astype('float32'),axis,axis_role=aux['axis_role']):traditional.append(dict(row,truth=truth,**value,smoothing_seconds=smoothing,preprocessing='Gaussian sigma=1 pixel, threshold0.5, fixed without fitting'))
 pd.DataFrame(traditional).to_csv(OUT/'test_conventional_gaussian.csv',index=False)
 # Time each full call in isolation; initialization, evolution and readouts separate.
 costs=[]
 for d in [2,3]:
  f,truth,axis,aux=fixture(diagnostic_row(d,'ellipse' if d==2 else 'ellipsoid',7.431,dict(axes=[1.8,.65] if d==2 else [1.8,.81,.59]),size=80 if d==2 else 64))
  for rep in range(6):
   t=time.perf_counter();n=initialize(f);ini=time.perf_counter()-t;t=time.perf_counter()
   for _ in range(30):n=old_step(n,f)
   field=time.perf_counter()-t;r,_=measure(n,axis);s0,st=scalar_zero_surface(n[...,-1])
   if rep>0:
    costs.append(dict(dim=d,rep=rep,method='PBGIA30 common field + all tensor/TV/energy readouts + s0',initialization_seconds=ini,field_seconds=field,measurement_seconds=r['measurement_seconds']+st,total_seconds=ini+field+r['measurement_seconds']+st))
    for v in classical(f,axis):costs.append(dict(dim=d,rep=rep,method=v['method'],initialization_seconds=0.,field_seconds=0.,measurement_seconds=v['seconds'],total_seconds=v['seconds']))
 pd.DataFrame(costs).to_csv(OUT/'isolated_runtime.csv',index=False)
 print('STABILITY/BENCHMARK COMPLETE',flush=True)

if __name__=='__main__':run()
