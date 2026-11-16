"""Independent train-only geometry and orientation head fitting."""
from engine import *
from scipy.optimize import differential_evolution

def fit_heads(cases,seed=271116):
 ref=next(c for c in cases if c['row']['role']=='reference');train=[c for c in cases if c['row']['role']!='reference'];dim=ref['row']['dim'];out={}
 for N in CHECKPOINTS:
  for c in cases:
   if N not in c['F']:c['F'][N]=features(c['states'][N])
  def geometry_loss(p):
   coeff=ref['truth']/np.trace(feature_tensor(ref['F'][N],p,dim));errors={};rot={}
   for c in train:
    val=coeff*np.trace(feature_tensor(c['F'][N],p,dim));errors.setdefault(c['row']['unit'],[]).append(abs(val/c['truth']-1));rot.setdefault((c['row']['unit'],c['row']['noise']),[]).append(val)
   mare=np.mean([np.mean(v) for v in errors.values()]);cv=np.mean([np.std(v)/max(abs(np.mean(v)),1e-12) for v in rot.values()]);return float(mare)
  angular=[c for c in train if c['axis'] is not None]
  def orient_loss(p,which='min'):
   byunit={}
   for c in angular:
    T=feature_tensor(c['F'][N],p,dim);err=axis_result(T,c['axis'],which)['angle_error'];byunit.setdefault(c['row']['unit'],[]).append(err)
   return float(np.mean([np.mean(v) for v in byunit.values()])/90)
  tic=time.perf_counter();kwargs=dict(bounds=[(-.14,.03),(0,.3),(0,.08)],popsize=4,maxiter=9,polish=False,tol=0,atol=0,workers=1)
  geom=differential_evolution(geometry_loss,seed=seed+dim,**kwargs)
  # Both eigenaxis conventions are trained and compared without test labels.
  omin=differential_evolution(lambda p:orient_loss(p,'min'),seed=seed+dim+31,**kwargs)
  # Maximum axis is a no-extra-optimization control on the same learned kernel.
  porient=omin.x.tolist();options=[]
  mats=[]
  for c in angular:
   T=feature_tensor(c['F'][N],porient,dim);d=dim;v=c['states'][N][...,:d].reshape(-1,d).astype(float);hj=apply_exchange(c['states'][N],c['c'])[...,:d].reshape(-1,d).astype(float);B=(v.T@hj+hj.T@v)/2;V=v.T@v;mat=[T-np.eye(d)*np.trace(T)/d,B-np.eye(d)*np.trace(B)/d,V-np.eye(d)*np.trace(V)/d];mats.append((c['row']['unit'],c['axis'],mat))
  for kind in ['gradient','bond','direct']:
   for which in ['min','max']:
    groups={}
    for unit,axis,mat in mats:groups.setdefault(unit,[]).append(axis_result(mat[{'gradient':0,'bond':1,'direct':2}[kind]],axis,which)['angle_error'])
    options.append(dict(type=kind,mix=[1,0,0],which=which,train_angle_MAE=float(np.mean([np.mean(v) for v in groups.values()]))))
  for a in range(5):
   for b in range(5-a):
    weights_=np.array([a,b,4-a-b])/4
    for which in ['min','max']:
     groups={}
     for unit,axis,mat in mats:groups.setdefault(unit,[]).append(axis_result(sum(w*normalized_tensor(T) for w,T in zip(weights_,mat)),axis,which)['angle_error'])
     options.append(dict(type='ensemble',mix=weights_.tolist(),which=which,train_angle_MAE=float(np.mean([np.mean(v) for v in groups.values()]))))
  chosen=min(options,key=lambda a:a['train_angle_MAE']);chosen.update(p=porient,evaluations=omin.nfev,optimizer_status=str(omin.message),gradient_objective=omin.fun)
  coeff=ref['truth']/np.trace(tensor(ref['states'][N],geom.x));out[str(N)]=dict(geometry=dict(p=geom.x.tolist(),coefficient=float(coeff),objective=geom.fun,evaluations=geom.nfev,optimizer_status=str(geom.message)),orientation=chosen,orientation_candidates=options,fit_seconds=time.perf_counter()-tic)
 return out

