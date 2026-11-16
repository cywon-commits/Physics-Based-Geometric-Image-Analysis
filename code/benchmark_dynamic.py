"""Independent training, calibration and held-out geometry benchmark for dynamic grad(s)."""
from run_research import *
from scipy.optimize import minimize
from scipy.special import ellipe

def tensor(n,p):
 d=n.ndim-1;g=[deriv(n,i,p,d) for i in range(d)]
 return np.array([[np.sum(a*b,dtype=np.float64) for b in g] for a in g])
def angle_error(T,axis,small=True):
 if axis is None:return np.nan
 vals,vec=np.linalg.eigh(T);v=vec[:,0 if small else -1]
 return float(np.degrees(np.arccos(np.clip(abs(v@axis),0,1))))
def conventional(mask,axis):
 d=mask.ndim;tic=time.perf_counter();rows=[]
 coords=np.array(np.nonzero(mask));T=np.cov(coords);pa=angle_error(T,axis,False)
 if d==2:
  contours=find_contours(mask.astype(float),.5);length=sum(np.linalg.norm(np.diff(c,axis=0),axis=1).sum() for c in contours)
  rows.extend([('marching_squares',length,pa),('Crofton',perimeter_crofton(mask,directions=4),pa)])
 else:
  verts,faces,_,_=marching_cubes(mask.astype('float32'),.5);area=mesh_surface_area(verts,faces)
  a,b,c=verts[faces[:,0]],verts[faces[:,1]],verts[faces[:,2]];cross=np.cross(b-a,c-a);ar=np.linalg.norm(cross,axis=-1)/2;nu=cross/np.maximum(2*ar[:,None],1e-12);ST=np.einsum('ni,nj,n->ij',nu,nu,ar)
  rows.extend([('marching_cubes+surface_tensor',area,angle_error(ST,axis)),('voxel_faces+PCA',sum(np.count_nonzero(np.diff(mask.astype(int),axis=k)) for k in range(d)),pa)])
 smooth=ndi.gaussian_filter(mask.astype(float),1);g=np.stack(np.gradient(smooth),-1);ST=np.einsum('...i,...j->ij',g,g) if False else np.array([[np.sum(g[...,i]*g[...,j]) for j in range(d)] for i in range(d)])
 rows.append(('Gaussian_sigma1+structure_tensor',np.linalg.norm(g,axis=-1).sum(),angle_error(ST,axis)))
 gm=smooth>.5;gc=np.array(np.nonzero(gm));ga=angle_error(np.cov(gc),axis,False)
 if d==2:
  val=sum(np.linalg.norm(np.diff(c,axis=0),axis=1).sum() for c in find_contours(gm.astype(float),.5))
  rows.append(('Gaussian_sigma1_threshold+marching_squares_PCA',val,ga))
 else:
  vv,ff,_,_=marching_cubes(gm.astype('float32'),.5)
  rows.append(('Gaussian_sigma1_threshold+marching_cubes_PCA',mesh_surface_area(vv,ff),ga))
 return rows,time.perf_counter()-tic

def make_case(kind,R,theta,shift,dim,N,noise=0,seed=0,r=3):
 d,mask,nu,truth,axis=fixture(kind,N=N,R=R,r=r,angle=theta,shift=shift,dim=dim)
 if kind=='multi':raise ValueError()
 if noise:
  rng=np.random.default_rng(seed);core=np.zeros_like(mask);core[(slice(6,-6),)*dim]=True;mask=np.logical_xor(mask,(rng.random(mask.shape)<noise)&core)
 return dict(kind=kind,R=R,angle=theta,shift=json.dumps(list(shift)),dim=dim,N=N,noise=noise,seed=seed,d=d,mask=mask,normal=nu,truth=truth,axis=axis)

def multi_case(dim,N,seed,noise=0):
 x=np.indices((N,)*dim,dtype=float)-(N-1)/2
 R=5 if dim==2 else 4
 a=np.sqrt((x[0]-9)**2+sum(x[j]**2 for j in range(1,dim)));b=np.sqrt((x[0]+9)**2+sum(x[j]**2 for j in range(1,dim)))
 d=R-np.minimum(a,b);mask=d>=0
 if noise:
  core=np.zeros_like(mask);core[(slice(6,-6),)*dim]=True;mask^=(np.random.default_rng(seed).random(mask.shape)<noise)&core
 nu=-np.stack(np.gradient(d),-1);nu/=np.maximum(np.linalg.norm(nu,axis=-1,keepdims=True),1e-12)
 return dict(kind='multi',R=R,angle=0,shift='[0,0,0]',dim=dim,N=N,noise=noise,seed=seed,d=d,mask=mask,normal=nu,truth=2*(2*np.pi*R if dim==2 else 4*np.pi*R*R),axis=np.array([1.,0]+([0.] if dim==3 else [])))

def run():
 rng=np.random.default_rng(61116);train=[];cal=[];test=[];raw=[];candidates=[]
 for h in [2.,4.]:
  for dim,kind,R,N in [(2,'circle',9,64),(3,'sphere',6,48)]:
   c=make_case(kind,R,0,(0,0,0),dim,N);c.update(split='calibration',h=h);cal.append(c)
 for kind in ['circle','stadium','ellipse']:
  for angle in [0,.35,.8]:
   c=make_case(kind,6,angle,(.1,.2,0),2,64);c.update(split='train',h=2.);train.append(c)
 for dim,N,kinds in [(2,64,['circle','stadium','ellipse','ring','multi']),(3,48,['sphere','capsule','torus','ellipsoid','multi','small'])]:
  for kind in kinds:
   for rep in range(3):
    for noise in [0.,.002]:
     seed=61116+100*dim+rep;angle=[.0,.37,.81][rep];shift=tuple(rng.uniform(-.45,.45,3));R=(12 if kind in ('torus','ring') else 2 if kind=='small' else 7 if kind=='capsule' else 9 if dim==3 else 11)
     c=multi_case(dim,N,seed,noise) if kind=='multi' else make_case(kind,R,angle,shift,dim,N,noise,seed,r=3 if kind in ('torus','ring') else 4.5)
     c.update(split='test',h=2.);test.append(c)
 # Independent h transfer subset, noiseless reps.
 for c in test[:]:
  if c['noise']==0 and c['angle'] in [0,.37] and c['kind'] in ['stadium','ellipse','capsule','torus']:cc=c.copy();cc['h']=4.;test.append(cc)
 cases=cal+train+test
 for i,c in enumerate(cases):
  c['id']=f'b{i:03d}';c['n'],c['info'],_=solve(c['mask'],dict(h=c['h'],D=1,J=1),steps=1200)
  c['metrics']=metrics(c['n'],c['mask'],c['normal'],c['d'])
  clean=c['d']>=0;out=c['n'][...,-1]>0
  c['metrics'].update(clean_dice=2*(out&clean).sum()/max(1,out.sum()+clean.sum()),clean_chi=euler_number(clean,connectivity=c['dim']),topologically_valid_clean=euler_number(out,connectivity=c['dim'])==euler_number(clean,connectivity=c['dim']),topologically_valid_input=euler_number(out,connectivity=c['dim'])==euler_number(c['mask'],connectivity=c['dim']))
  np.savez_compressed(OUT/(c['id']+'_field.npz'),n=c['n'],mask=c['mask'])
  print('field',c['id'],c['kind'],c['info'],flush=True)
 # Each objective calibrated only on independent circle. No held-out GT in training.
 ref=next(c for c in cal if c['dim']==2 and c['h']==2)
 def score(p,detail=False):
  coeff=ref['truth']/np.trace(tensor(ref['n'],p));errors=[];angles=[];groups={}
  for c in train:
   T=tensor(c['n'],p);estimate=np.trace(T)*coeff;errors.append((estimate/c['truth']-1)**2)
   ae=angle_error(T,c['axis']);
   if np.isfinite(ae):angles.append((ae/10)**2)
   groups.setdefault(c['kind'],[]).append(estimate)
  mse=np.mean(errors);am=np.mean(angles);cv=np.mean([(np.std(a)/np.mean(a))**2 for a in groups.values()])
  return (mse,am,cv) if detail else mse+.05*am+cv
 fits=[]
 for w in [.01,.05,.2]:
  def obj(p):a,b,c=score(p,True);return a+w*b+c
  fit=minimize(obj,[-.06,.15,.01],method='Nelder-Mead',bounds=[(-.14,.03),(0,.3),(0,.08)],options={'maxiter':500,'xatol':.0005,'fatol':1e-8})
  fits.append(fit);candidates.append(dict(weight=w,p=json.dumps(fit.x.tolist()),success=fit.success,iterations=fit.nit,objective=fit.fun,geometry_mse=score(fit.x,True)[0],angle_scaled_mse=score(fit.x,True)[1],cv_squared=score(fit.x,True)[2]))
 for c in candidates:
  c['pareto_nondominated']=not any(o['geometry_mse']<=c['geometry_mse'] and o['angle_scaled_mse']<=c['angle_scaled_mse'] and o['cv_squared']<=c['cv_squared'] and (o['geometry_mse']<c['geometry_mse'] or o['angle_scaled_mse']<c['angle_scaled_mse'] or o['cv_squared']<c['cv_squared']) for o in candidates)
 chosen=fits[1];K['learned']=tuple(chosen.x);pd.DataFrame(candidates).to_csv(OUT/'kernel_training.csv',index=False)
 json.dump({'parameters':K,'objective':'geometry_relative_mse + weight*(angle_deg/10)^2 + rotation_CV_squared','selected_weight':.05,'train_ids':[c['id'] for c in train],'calibration_ids':[c['id'] for c in cal]},open(OUT/'kernels.json','w'),indent=2)
 b,t,u=chosen.x;np.savetxt(OUT/'learned_5x5_cross_correlation.csv',np.outer([u,t,1-2*t-2*u,t,u],[-b,-(.5-2*b),0,.5-2*b,b]),delimiter=',')
 coeff={}
 for c in cal:
  for method,p in K.items():coeff[(c['dim'],c['h'],method)]=c['truth']/np.trace(tensor(c['n'],p))
  conv,_=conventional(c['mask'],c['axis'])
  for method,val,ang in conv:
   coeff[(c['dim'],c['h'],method)]=c['truth']/val
 for c in cases:
  base={k:c[k] for k in ['id','kind','R','angle','shift','dim','N','noise','seed','split','h','truth']};base.update(c['info'],**c['metrics'],D=1,J=1,dynamic_kernel='central',spacing=1,units='pixel' if c['dim']==2 else 'voxel^2',model='dynamic_grad_s_normalized')
  for method,p in K.items():
   tic=time.perf_counter();T=tensor(c['n'],p);value=np.trace(T)*coeff[(c['dim'],c['h'],method)];ang=angle_error(T,c['axis']);sec=time.perf_counter()-tic
   raw.append(dict(base,method='PBGIA_'+method,measurement_kernel=json.dumps(p),calibration_coefficient=coeff[(c['dim'],c['h'],method)],raw_measurement=float(np.trace(T)),measured=value,axis_error_deg=ang,relative_error_pct=100*(value/c['truth']-1),measurement_seconds=sec,total_seconds=c['info']['seconds']+sec))
  conv,sec=conventional(c['mask'],c['axis'])
  for method,value,ang in conv:
   if method=='Gaussian_sigma1+structure_tensor':value*=coeff[(c['dim'],c['h'],method)]
   raw.append(dict(base,method=method,measurement_kernel='',calibration_coefficient=coeff[(c['dim'],c['h'],method)] if method=='Gaussian_sigma1+structure_tensor' else 1.,raw_measurement=value/coeff[(c['dim'],c['h'],method)] if method=='Gaussian_sigma1+structure_tensor' else value,measured=value,axis_error_deg=ang,relative_error_pct=100*(value/c['truth']-1),measurement_seconds=sec,total_seconds=sec))
   if method!='Gaussian_sigma1+structure_tensor':
    calibrated=value*coeff[(c['dim'],c['h'],method)];raw.append(dict(base,method=method+'+reference_calibrated',measurement_kernel='',calibration_coefficient=coeff[(c['dim'],c['h'],method)],raw_measurement=value,measured=calibrated,axis_error_deg=ang,relative_error_pct=100*(calibrated/c['truth']-1),measurement_seconds=sec,total_seconds=sec))
  pd.DataFrame(raw).to_csv(OUT/'benchmark_raw.csv',index=False)
 df=pd.DataFrame(raw);summ=[]
 for keys,g in df[df.split=='test'].groupby(['dim','h','noise','method']):
  vals=abs(g.relative_error_pct.to_numpy());boots=np.mean(rng.choice(vals,(3000,len(vals)),replace=True),axis=1)
  summ.append(dict(zip(['dim','h','noise','method'],keys),n=len(g),MARE_pct=vals.mean(),CI95_low=np.quantile(boots,.025),CI95_high=np.quantile(boots,.975),angle_MAE_deg=g.axis_error_deg.mean(),max_residual=g.residual.max(),converged_fraction=g.converged.mean(),mean_total_seconds=g.total_seconds.mean()))
 pd.DataFrame(summ).to_csv(OUT/'benchmark_summary.csv',index=False)
 cvs=[]
 for keys,g in df[(df.split=='test')&(df.noise==0)].groupby(['dim','h','method','kind']):
  cvs.append(dict(zip(['dim','h','method','kind'],keys),n=len(g),rotation_shift_CV_pct=100*g.measured.std(ddof=0)/g.measured.mean()))
 pd.DataFrame(cvs).to_csv(OUT/'rotation_shift_cv.csv',index=False)
 # Dynamic generator changes fields; keep measurement central and refit no test coefficient.
 gr=[]
 for dim,kind,R,N in [(2,'stadium',11,64),(3,'sphere',9,48),(3,'capsule',7,48),(3,'torus',12,48)]:
  for h in [2.,4.]:
   for method,p in K.items():
    d,mask,nu,truth,ax=fixture(kind,N=N,R=R,angle=.61,dim=dim)
    n,info,_=solve(mask,dict(h=h,D=1,J=1,p=p),steps=1200)
    gr.append(dict(dim=dim,shape=kind,h=h,D=1,J=1,generator_kernel=method,measurement_kernel='central',**info,**metrics(n,mask,nu,d),central_trace=float(np.trace(tensor(n,K['central'])))))
    pd.DataFrame(gr).to_csv(OUT/'generator_kernel_sweep.csv',index=False)
 print('DONE benchmark',len(cases),len(raw),flush=True)
if __name__=='__main__':run()
