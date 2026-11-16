from run_research import *
from topology_audit import certificate,cap,q3

def run():
 rows=[];hist=[]
 cases=[('torus',48,2.,1.,1),('sphere',48,2.,1.,1),('capsule',48,2.,1.,1),('torus',64,2.,1.,1),('sphere',64,2.,1.,1),('torus',48,.1,3.,1),('torus',48,.1,3.,.5),('sphere',48,.1,.16,1),('torus',48,8.,.16,1)]
 for i,(kind,N,h,D,alpha) in enumerate(cases):
  d,mask,nu,truth,ax=fixture(kind,N=N,R=12 if kind=='torus' else 7 if kind=='capsule' else 9)
  name=f'supp_{i:02d}_{kind}';f=np.where(mask,1.,-1.).astype('float32');n=initialize(mask);prev=None;tic=time.perf_counter()
  for it in range(2001):
   nx=update(n,f,h=h,D=D,alpha=alpha);res=float(np.sqrt(np.mean(np.sum((nx-n)**2,-1))));cy=float(np.sqrt(np.mean(np.sum((nx-prev)**2,-1)))) if prev is not None else np.nan
   hist.append(dict(run=name,iteration=it,residual=res,two_step_residual=cy))
   if it in [0,100,300,600,1200,2000]:
    undamped=update(n,f,h=h,D=D);rr=float(np.sqrt(np.mean(np.sum((undamped-n)**2,-1))))
    m=dict(run=name,shape=kind,N=N,h=h,D=D,J=1,alpha=alpha,iteration=it,residual=res,undamped_residual=rr,two_step_residual=cy,**metrics(n,mask,nu,d),Q3_fd=q3(n),**certificate(n));dg=degree(cap(n));m.update(dg);rows.append(m)
    np.savez_compressed(OUT/(name+f'_t{it:04d}.npz'),n=n,mask=mask)
    pd.DataFrame(rows).to_csv(OUT/'supplemental_metrics.csv',index=False);pd.DataFrame(hist).to_csv(OUT/'supplemental_history.csv',index=False)
   if it<2000:prev,n=n,nx
  print(name,time.perf_counter()-tic,rows[-1],flush=True)
 # Axis permutation covariance, same discrete input and spin transformation.
 _,mask,_,_,_=fixture('capsule',N=40,R=7,angle=.61)
 n=initialize(mask);f=np.where(mask,1.,-1.).astype('float32');trans=np.transpose(n,(2,0,1,3))[...,[2,0,1,3]];ft=np.transpose(f,(2,0,1))
 lhs=update(trans,ft);rhs=np.transpose(update(n,f),(2,0,1,3))[...,[2,0,1,3]]
 json.dump({'axis_cycle_covariance_max_error':float(abs(lhs-rhs).max())},open(OUT/'covariance.json','w'),indent=2)
if __name__=='__main__':run()
