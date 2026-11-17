from torch_engine import *
from scipy.optimize import minimize
import platform,resource,sys
TRACE=[]
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=True))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def protocol():
 manifest();dump('protocol.json',dict(seed=SEED,primary_N=10,N_validation=[5,10,20,30],exchange='fixed NN sum4',observer='fixed exact G4 tensor, min eigenaxis',init='binary EDT tanh width2.3 lattice',physics=dict(J=1,D=1,h=2),normalization='H only, NEVER normalize derivative coefficients',family='nonseparable odd/even symmetric5x5 six unconstrained-gain coefficients',starts=['G4','legacy','pilot','perturbation'],bounds=[-1.5,1.5],noise_weights=[0,1,4],lambda_rotation=1,lambda_angle=.0001,theta_scale_degrees=1,loss='prototype variance log mean(I/P), within-unit CV squared, clean angle squared, paired noise log-I change',proxy='all17train prototypes; two crossed rotation/shift poses; paired noise0,.002,.005,.01 (clean-only excludes noisy inputs)',optimizer=dict(method='L-BFGS-B',gradient='PyTorch reverse autograd, bounded-memory exact two-pass chain rule; independently checked against exact forward Jacobian',maxiter=60,maxfun=100,maxls=12,ftol=1e-11,gtol=1e-6),separable='same maxiter/maxfun, fixed moment1, starts G4, clean and noise1',selection='N10: exclude fitted candidates worse than G4 in clean log-q variance; per-noise-weight select minimum declared objective on validation with G4 always anchored; separately select N among5/10/20/30 for the frozen noise1 arm via same validation objective',test='freeze protocol,manifest,models,reference set,N,weights before single fresh test; no test-based ruler selection',resources=dict(primary='complete2D',threeD='verified12orbit kernels and differentiation smoke only; no allotted3D optimization budget; 8GiB CPU environment, prioritize full2D multistart',wider='7x7,exchange,observer retuning deferred; no automatic budget expansion')))
 dump('environment.json',dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),threads=1,gpu_used=False,memory_limit_bytes=8589934592))

def train():
 assert not (OUT/'models.json').exists();assert not (OUT/'model_freeze_2d.json').exists();assert (OUT/'smoke_pass.json').exists();df=manifest();rows=df.query("split=='train' and role=='geometry' and rotation_id==shift_id and rotation_id<2");all_ds=load(rows);clean_ds=load(rows[rows.noise==0]);rng=np.random.default_rng(SEED);starts=dict(G4=G4,legacy=LEGACY,pilot=PILOT,perturbation=G4+rng.normal(0,.035,6));models={k:dict(p=v.tolist(),weight=None,family='anchor') for k,v in dict(G4=G4,legacy=LEGACY,pilot=PILOT).items()};trials=[];t0=time.perf_counter();cpu=time.process_time()
 for w in [0,1,4]:
  ds=clean_ds if w==0 else all_ds
  for name,x0 in starts.items():
   label=f'fit_w{w}_{name}';evaluation=0;tic=time.perf_counter();initial=None
   def fun(p):
    nonlocal evaluation,initial
    val,grad,parts=objective_autograd(ds,p,w);evaluation+=1
    if initial is None:initial=val
    TRACE.append(dict(model=label,noise_weight=w,evaluation=evaluation,objective=val,gradient_norm=float(np.linalg.norm(grad)),elapsed=time.perf_counter()-tic,parameters=json.dumps(p.tolist()),**parts));pd.DataFrame(TRACE).to_csv(OUT/'train_optimizer_trace.csv',index=False);return val,grad
   result=minimize(fun,x0,jac=True,method='L-BFGS-B',bounds=[(-1.5,1.5)]*6,options=dict(maxiter=60,maxfun=100,maxls=12,ftol=1e-11,gtol=1e-6));models[label]=dict(p=result.x.tolist(),weight=w,family='nonseparable');trials.append(dict(model=label,initial=initial,final=float(result.fun),nit=result.nit,nfev=result.nfev,njev=result.njev,success=bool(result.success),message=str(result.message),gradient_norm=float(np.linalg.norm(result.jac)),active_bounds=int(np.sum(np.isclose(abs(result.x),1.5))),wall_seconds=time.perf_counter()-tic));dump('models.json',models);pd.DataFrame(trials).to_csv(OUT/'train_trials.csv',index=False);print('FIT',label,trials[-1],flush=True)
 # Separable moment-one family with exact G4 embedding.
 def mapping(x):
  b,t,u=x;a=.5-2*b;q=1-2*t-2*u;return np.array([a*q,b*q,a*t,b*t,a*u,b*u]),np.array([[-2*q,q,-2*t,t,-2*u,u],[-2*a,-2*b,a,b,0,0],[-2*a,-2*b,0,0,a,b]]).T
 for w in [0,1]:
  label=f'separable_w{w}';tic=time.perf_counter();ds=clean_ds if w==0 else all_ds;initial=None;ev=0
  def fun_sep(x):
   nonlocal initial,ev
   p,J=mapping(x);v,g,parts=objective_autograd(ds,p,w);ev+=1
   if initial is None:initial=v
   TRACE.append(dict(model=label,noise_weight=w,evaluation=ev,objective=v,gradient_norm=float(np.linalg.norm(J.T@g)),elapsed=time.perf_counter()-tic,parameters=json.dumps(x.tolist()),**parts));pd.DataFrame(TRACE).to_csv(OUT/'train_optimizer_trace.csv',index=False);return v,J.T@g
  res=minimize(fun_sep,[-1/12,0,0],jac=True,method='L-BFGS-B',bounds=[(-.3,.2),(-.4,.5),(-.4,.5)],options=dict(maxiter=60,maxfun=100,maxls=12,ftol=1e-11,gtol=1e-6));p,_=mapping(res.x);models[label]=dict(p=p.tolist(),weight=w,family='separable',separable_parameters=res.x.tolist());trials.append(dict(model=label,initial=initial,final=float(res.fun),nit=res.nit,nfev=res.nfev,njev=res.njev,success=bool(res.success),message=str(res.message),gradient_norm=float(np.linalg.norm(res.jac)),active_bounds=int(np.sum(np.isclose(res.x,[-.3,-.4,-.4])|np.isclose(res.x,[.2,.5,.5]))),wall_seconds=time.perf_counter()-tic));dump('models.json',models);pd.DataFrame(trials).to_csv(OUT/'train_trials.csv',index=False);print('FIT',label,flush=True)
 dump('training_cost.json',dict(wall_seconds=time.perf_counter()-t0,CPU_seconds=time.process_time()-cpu,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));print('TRAIN2D COMPLETE',flush=True)
if __name__=='__main__':train()
