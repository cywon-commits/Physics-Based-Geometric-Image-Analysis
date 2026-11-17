from data import *
from skimage.measure import find_contours,perimeter_crofton,marching_cubes,mesh_surface_area
from physics import normal_alignment

def scalar_zero_surface(s,spacing=1.):
 t=time.perf_counter()
 if not (s.min()<0<s.max()):return np.nan,time.perf_counter()-t
 if s.ndim==2:
  value=sum(np.linalg.norm(np.diff(c,axis=0),axis=1).sum() for c in find_contours(s,0))*spacing
 else:
  v,faces,_,_=marching_cubes(s,0,spacing=(spacing,)*s.ndim);value=mesh_surface_area(v,faces)
 return float(value),time.perf_counter()-t

def classical(f,axis,spacing=1.,axis_role='object_major'):
 mask=f>0;d=f.ndim;t=time.perf_counter();coords=np.array(np.nonzero(mask),float).T*spacing;C=np.cov(coords.T);pca,gap=axis_error(C,axis,'min' if axis_role in ['symmetry_axis','oblate_normal'] else 'max');pca_time=time.perf_counter()-t;out=[]
 if d==2:
  t=time.perf_counter();value=float(perimeter_crofton(mask,4)*spacing);out.append(dict(method='Crofton',value=value,angle_error=pca,seconds=time.perf_counter()-t+pca_time))
  t=time.perf_counter();value=sum(np.linalg.norm(np.diff(c,axis=0),axis=1).sum() for c in find_contours(mask.astype(float),.5))*spacing;out.append(dict(method='marching_squares',value=float(value),angle_error=pca,seconds=time.perf_counter()-t+pca_time))
 else:
  t=time.perf_counter();v,faces,_,_=marching_cubes(mask.astype('float32'),.5,spacing=(spacing,)*d);value=float(mesh_surface_area(v,faces));out.append(dict(method='marching_cubes',value=value,angle_error=pca,seconds=time.perf_counter()-t+pca_time))
 out.append(dict(method='binary_PCA',value=np.nan,angle_error=pca,seconds=pca_time));return out

def run_cases(rows):
 measurements=[];conventional=[];times=[]
 for i,row in enumerate(rows.to_dict('records')):
  f,truth,axis,aux=fixture(row);tic=time.perf_counter();n=initialize(f);ini=time.perf_counter()-tic;tic=time.perf_counter()
  for _ in range(30):n=old_step(n,f)
  sec=time.perf_counter()-tic;r,_=measure(n,axis,aux['axis_role']);s0,t0=scalar_zero_surface(n[...,-1]);local=normal_alignment(n,aux['inward_normal'],abs(aux['distance_for_band'])<1.5);measurements.append(dict(row,truth=truth,**r,**local,I_s0=s0,surface0_seconds=t0,energy_frozen=BASE['calibration_by_dimension'][str(f.ndim)]*r['I_E'],field_seconds=sec,initialization_seconds=ini,axis_role=aux['axis_role'],normal_target_defined=aux['axis_role'] not in ['undefined','pair_separation'],spatial_target_defined=axis is not None,unit_error=float(abs(np.linalg.norm(n,axis=-1)-1).max())))
  for c in classical(f,axis,axis_role=aux['axis_role']):conventional.append(dict(row,truth=truth,**c))
  if i%24==0:print('GEOMETRY',row['split'],row['dim'],i,'/',len(rows),flush=True)
 return pd.DataFrame(measurements),pd.DataFrame(conventional)

def train_validate():
 assert not (OUT/'model_freeze.json').exists();assert not (OUT/'test_raw.csv').exists();df=manifest();part=df[(df.split!='test')];r,c=run_cases(part);r.to_csv(OUT/'train_validation_raw.csv',index=False);c.to_csv(OUT/'train_validation_conventional.csv',index=False);coeff=[]
 for d in [2,3]:
  refs=r[(r.dim==d)&(r.role=='reference')]
  for source in ['I_E','I_TV']:
   q=refs.assign(q=refs[source]/refs.truth).groupby('unit').q.mean();coef=float(q.sum()/(q*q).sum());coeff.append(dict(dim=d,source=source,coefficient=coef,reference_units=q.index.tolist()))
  cc=c[(c.dim==d)&(c.role=='reference')]
  for method,g in cc[cc.method!='binary_PCA'].groupby('method'):
   q=g.assign(q=g.value/g.truth).groupby('unit').q.mean();coeff.append(dict(dim=d,source=method,coefficient=float(q.sum()/(q*q).sum()),reference_units=q.index.tolist()))
 dump('calibration.json',coeff);trials=[];selection={}
 for d in [2,3]:
  ce=next(x['coefficient'] for x in coeff if x['dim']==d and x['source']=='I_E');ct=next(x['coefficient'] for x in coeff if x['dim']==d and x['source']=='I_TV');val=r[(r.dim==d)&(r.split=='validation')];base={}
  for alpha in [0.,.25,.5,.75,1.]:
   e=100*((alpha*ce*val.I_E+(1-alpha)*ct*val.I_TV)/val.truth-1);v=val.assign(error=e);by_noise={str(n):float(g.groupby('unit').error.apply(lambda x:abs(x).mean()).mean()) for n,g in v.groupby('noise')};overall=float(v.groupby('unit').error.apply(lambda x:abs(x).mean()).mean());trials.append(dict(dim=d,alpha=alpha,overall_MARE=overall,by_noise=json.dumps(by_noise)));base[alpha]=(overall,by_noise)
  candidates=[]
  for alpha,(overall,by_noise) in base.items():
   if alpha==1.:continue
   if overall<=base[1.][0]-.1 and all(by_noise[n]<=base[1.][1][n]+.1 for n in by_noise):candidates.append(alpha)
  alpha=min(candidates,key=lambda x:base[x][0]) if candidates else 1.;selection[str(d)]=dict(alpha=alpha,status='ACCEPTED_VALIDATION' if candidates else 'REJECTED_KEEP_ENERGY',c_E=ce,c_TV=ct,validation_baseline_MARE=base[1.][0],validation_blend_MARE=base[alpha][0],N=30)
 pd.DataFrame(trials).to_csv(OUT/'alpha_validation.csv',index=False);frozen=dict(selection=selection,freeze_utc=pd.Timestamp.now(tz='UTC').isoformat(),test_seen=False,hashes={x:sha(OUT/x) for x in ['protocol.json','manifest.csv','calibration.json','alpha_validation.csv']},baseline_configuration_sha256=sha(ROOT/'reference/PBGIA30_frozen.json'));dump('model_freeze.json',frozen);(OUT/'model_freeze.sha256').write_text(sha(OUT/'model_freeze.json'));print('FROZEN',selection,flush=True)

def check_lock():
 f=json.load(open(OUT/'model_freeze.json'));assert sha(OUT/'model_freeze.json')==(OUT/'model_freeze.sha256').read_text()
 for p,h in f['hashes'].items():assert sha(OUT/p)==h
 assert sha(ROOT/'reference/PBGIA30_frozen.json')==f['baseline_configuration_sha256']
 return f

def final_test():
 import resource
 assert not (OUT/'test_raw.csv').exists(),'Final test already scored; no new candidates';frozen=check_lock();start=pd.Timestamp.now(tz='UTC').isoformat();tic=time.perf_counter();r,c=run_cases(manifest().query("split=='test'"));r.to_csv(OUT/'test_raw.csv',index=False);c.to_csv(OUT/'test_conventional.csv',index=False);dump('test_audit.json',dict(freeze_sha256=sha(OUT/'model_freeze.json'),no_retuning=True,images=len(r),units=r.unit.nunique(),start_utc=start,end_utc=pd.Timestamp.now(tz='UTC').isoformat(),wall_seconds=time.perf_counter()-tic,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'code').glob('*.py'))+list((ROOT/'vendor').glob('*.py'))}));print('TEST COMPLETE',len(r),flush=True)

def supplement_observers():
 # Fixed, unselected diagnostics; never change calibration/selection or original timings.
 path=OUT/'train_validation_raw.csv';r=pd.read_csv(path);c=pd.read_csv(OUT/'train_validation_conventional.csv');updates=0
 for row in manifest().query("split!='test'").to_dict('records'):
  f,truth,axis,aux=fixture(row);n=initialize(f)
  for _ in range(30):n=old_step(n,f)
  old=r.loc[r.case==row['case']].iloc[0];I=float(np.trace(tensor(n,BASE['kernel_by_dimension'][str(f.ndim)])));assert np.isclose(I,old.I_E,rtol=1e-13)
  s0,t0=scalar_zero_surface(n[...,-1]);local=normal_alignment(n,aux['inward_normal'],abs(aux['distance_for_band'])<1.5)
  for key,value in dict(I_s0=s0,surface0_seconds=t0,**local).items():r.loc[r.case==row['case'],key]=value
  if aux['axis_role']=='symmetry_axis':
   coords=np.array(np.nonzero(f>0),float).T;err,_=axis_error(np.cov(coords.T),axis,'min');c.loc[c.case==row['case'],'angle_error']=err;updates+=1
 r.to_csv(path,index=False);c.to_csv(OUT/'train_validation_conventional.csv',index=False)
 dump('supplemental_observer_audit.json',dict(fixed_observers=['generated_s0','local_inward_normal','v_grad_s_alignment'],baseline_energy_verified=True,selection_unchanged=True,torus_PCA_target_correction_cases=updates))

if __name__=='__main__':
 import sys
 {'validate':train_validate,'test':final_test,'supplement':supplement_observers}[sys.argv[1]]()
