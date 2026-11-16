"""Predeclared split -> train -> validation -> freeze -> test. No adaptive deployment stop."""
from study_core import *
from scipy.optimize import differential_evolution
import argparse,platform,os,importlib.metadata as md
sys.path.insert(0,str(ROOT/'reference'))
from pbgia_s3_integer_degree import degree

def manifest():
 path=OUT/'split_manifest.csv'
 if path.exists():return pd.read_csv(path)
 rng=np.random.default_rng(61116);rows=[];proto=0
 for dim,families in [(2,['circle','stadium','ellipse','ring','multi']),(3,['sphere','capsule','torus','ellipsoid','multi','small'])]:
  for family in families:
   held=family in ['ring','ellipsoid'];sizes=([8.2,11.2] if held else [6.,7.5,9.,10.5,12.]) if dim==2 else ([6.2,8.2] if held else [9.,10.,11.,12.,13.] if family=='torus' else [1.5,2.,2.5,3.,3.5] if family=='small' else [3.,3.5,4.,4.5,5.] if family=='multi' else [4.5,5.5,6.5,7.5,8.5])
   if dim==2 and family=='multi':sizes=[4.,5.,6.,7.,8.]
   order=rng.permutation(len(sizes));assign={i:('test' if held else 'train' if j<2 else 'validation' if j==2 else 'test') for j,i in enumerate(order)}
   for i,R in enumerate(sizes):
    unit=f'g{proto:03d}';phase=float(rng.uniform(0,.3));proto+=1
    for rep in range(2):
     for ni,noise in enumerate([0.,.002,.01]):
      rows.append(dict(case=f'{unit}_r{rep}_n{ni}',unit=unit,split=assign[i],role='geometry',family=family,heldout_family=held,dim=dim,R=R,size=80 if dim==2 else 56,rep=rep,phase=phase,noise=noise,seed=61116+proto*100+rep*10,noise_seed=91116+proto*100+rep*10+ni,spacing=1.))
 for dim,family,R in [(2,'circle',5.25),(3,'sphere',4.25)]:rows.append(dict(case=f'cal{dim}',unit=f'cal{dim}',split='train',role='calibration',family=family,heldout_family=False,dim=dim,R=R,size=80 if dim==2 else 56,rep=0,phase=0.,noise=0.,seed=61116,noise_seed=91116,spacing=1.))
 df=pd.DataFrame(rows);df.to_csv(path,index=False)
 design=dict(seed=61116,checkpoints=list(CHECKPOINTS),main=PARAMS,transfer=[dict(J=1.,h=4.,D=1.),dict(J=1.,h=2.,D=.5)],generation='central fixed',kernel_dimension=3,bounds=[[-.14,.03],[0,.3],[0,.08]],optimizer='differential_evolution',optimizer_popsize=6,optimizer_maxiter=14,optimizer_evaluations=270,optimizer_polish=False,validation_tolerance_geometry_pp=.1,validation_tolerance_orientation_deg=.1,selection='smallest N passes every dimension/noise stratum, no more clean-mask/representative-degree failures, faster deployment estimate; h fixed a priori at 2',orientation='minimum eigenvalue only for connected elongated fixtures; multi/circle/sphere/ring/torus undefined',mask_failure='clean Dice < .95 OR Euler mismatch OR component mismatch',bootstrap='resample independent geometry units; all derived variants kept together')
 (OUT/'design.json').write_text(json.dumps(design,indent=2));return df

def cap(n):
 shape=n.shape[:3];idx=np.indices(shape);edge=np.minimum(idx,np.array(shape)[:,None,None,None]-1-idx).min(0);w=np.clip(edge/3,0,1)[...,None];pole=np.zeros(4);pole[-1]=-1;out=w*n+(1-w)*pole;out/=np.linalg.norm(out,axis=-1,keepdims=True);return out

def topology(n):
 # Sufficient global hemisphere test; failed certificate is explicitly NOT a singularity proof.
 minmargin=np.inf
 for perm in itertools.permutations(range(3)):
  offsets=np.zeros((4,3),int)
  for j,k in enumerate(perm):offsets[j+1]=offsets[j];offsets[j+1,k]+=1
  vs=np.stack([n[tuple(slice(o,sz-1+o) for sz,o in zip(n.shape[:3],off))] for off in offsets],-2);dots=np.sum(vs*vs.sum(-2)[...,None,:],-1);minmargin=min(minmargin,float(dots.min()))
 idx=np.indices(n.shape[:3]);edge=np.minimum(idx,np.array(n.shape[:3])[:,None,None,None]-1-idx).min(0);collar=float(n[...,-1][edge<=3].max());nc=cap(n)
 dgs=[degree(nc,t) for t in [( .37139,-.52177,.64829,.40981),(-.6,.2,.3,.7),(.2,.8,-.5,-.2)]]
 return dict(degree=dgs[0]['integer_degree'],degree_target_agreement=len(set(x['integer_degree'] for x in dgs))==1,hemisphere_min_dot=minmargin,negative_collar_max_s=collar,degree_certified=minmargin>1e-7 and collar<0 and len(set(x['integer_degree'] for x in dgs))==1,degree_diagnostics=json.dumps(dgs))

def save_metrics(cases,stage):
 rows=[]
 for c in cases:
  base={k:v for k,v in c['row'].items()};base.update(h=c.get('params',PARAMS)['h'],D=c.get('params',PARAMS)['D'],J=1,cache_key=c['key'])
  for m in c['metrics']:rows.append(dict(base,**m))
 path=OUT/'checkpoint_metrics.csv'
 if path.exists():
  old=pd.read_csv(path);old=old[~old.case.isin([c['row']['case'] for c in cases])];df=pd.concat([old,pd.DataFrame(rows)],ignore_index=True)
 else:df=pd.DataFrame(rows)
 df.to_csv(path,index=False)

def load_cases(df,stage,features_needed=False,params=PARAMS):
 cases=[]
 for i,row in enumerate(df.to_dict('records')):
  c=checkpoints(row,params,with_features=features_needed);c['params']=params;cases.append(c)
  if (i+1)%10==0:print(stage,i+1,'/',len(df),flush=True)
 save_metrics(cases,stage);return cases

def calibrations(refs,kernels,params=PARAMS):
 out={}
 for c in refs:
  dim=c['row']['dim']
  for N in CHECKPOINTS:
   for method,p in {**KERNELS,'learned':kernels[str(dim)][str(N)]['p']}.items():out[(dim,N,method)]=c['truth']/np.trace(tensor(c['snaps'][N],p))
 return out

def fit(train):
 kernel={};training=[]
 for dim in [2,3]:
  cases=[c for c in train if c['row']['dim']==dim and c['row']['role']=='geometry'];ref=next(c for c in train if c['row']['dim']==dim and c['row']['role']=='calibration');kernel[str(dim)]={}
  truth=np.array([c['truth'] for c in cases]);axis=[c['axis'] for c in cases]
  for N in CHECKPOINTS:
   Fs=[c['features'][N] for c in cases];Fr=ref['features'][N]
   def objective(p,details=False):
    coef=ref['truth']/np.trace(feature_tensor(Fr,p,dim));pred=[];angs=[];groups={}
    for c,F,a in zip(cases,Fs,axis):
     T=feature_tensor(F,p,dim);val=coef*np.trace(T);pred.append(val);ang,_=orientation(T,a)
     if np.isfinite(ang):angs.append((ang/10)**2)
     # Rotation/shift CV within same independent prototype and noise rate.
     groups.setdefault((c['row']['unit'],c['row']['noise']),[]).append(val)
    mse=np.mean((np.array(pred)/truth-1)**2);ae=np.mean(angs);cv=np.mean([(np.std(v)/np.mean(v))**2 for v in groups.values()]);score=mse+.05*ae+cv
    return dict(geometry_mse=float(mse),angle_scaled_mse=float(ae),cv_squared=float(cv),objective=float(score)) if details else score
   tic=time.perf_counter();res=differential_evolution(objective,[(-.14,.03),(0,.3),(0,.08)],seed=61116+dim,popsize=6,maxiter=14,tol=0,atol=0,polish=False,workers=1);sec=time.perf_counter()-tic
   kernel[str(dim)][str(N)]=dict(p=res.x.tolist(),evaluations=res.nfev,optimizer_success=bool(res.success),optimizer_message=res.message,optimizer_seconds=sec,**objective(res.x,True))
   print('FIT',dim,N,kernel[str(dim)][str(N)],flush=True)
 (OUT/'kernel_by_iteration.json').write_text(json.dumps(dict(kernels=kernel,normalization={'constant_response':0,'linear_gradient':1,'cross_correlation_derivative':'[-b,-(.5-2b),0,.5-2b,b]','transverse':'[t2,t1,1-2t1-2t2,t1,t2]','bounds':[[-.14,.03],[0,.3],[0,.08]]},train_ids=[c['row']['case'] for c in train],calibration=[],validation_scores=[]),indent=2));return kernel

def evaluate(cases,kernels,coeff,methods=None,selected_N=None):
 rows=[]
 for c in cases:
  dim=c['row']['dim'];base=dict(c['row'],truth=c['truth'],cache_key=c['key'],h=c.get('params',PARAMS)['h'],D=c.get('params',PARAMS)['D'],J=1)
  for N in (CHECKPOINTS if selected_N is None else [selected_N,40] if selected_N!=40 else [40]):
   choices={**KERNELS,'learned':kernels[str(dim)][str(N)]['p']}
   for method,p in choices.items():
    if methods and method not in methods:continue
    tic=time.perf_counter();T=tensor(c['snaps'][N],p);angle,gap=orientation(T,c['axis']);measurement=time.perf_counter()-tic;val=np.trace(T)*coeff[(dim,N,method)];m=next(m for m in c['metrics'] if m['N']==N)
    rows.append(dict(base,method=method,kernel=json.dumps(p),calibration_coefficient=coeff[(dim,N,method)],raw_measurement=np.trace(T),measured=val,error_pct=100*(val/c['truth']-1),angle_error=angle,eigengap=gap,measurement_seconds=measurement,total_seconds=m['field_seconds']+m['initialization_seconds']+measurement,**m))
 return pd.DataFrame(rows)

def summarize(df):
 rows=[]
 for keys,g in df.groupby(['dim','noise','N','method']):
  units=g.groupby('unit').agg(abs_error=('error_pct',lambda x:np.mean(abs(x))),angle=('angle_error','mean'),total=('total_seconds','mean'),mask_failure=('clean_dice',lambda x:float((x<.95).mean())))
  vals=units.abs_error.to_numpy();rng=np.random.default_rng(61116);boot=rng.choice(vals,(2000,len(vals)),replace=True).mean(1)
  cv=g.groupby('unit').measured.agg(lambda x:100*x.std(ddof=0)/abs(x.mean()))
  rows.append(dict(zip(['dim','noise','N','method'],keys),n_cases=len(g),n_units=len(units),geometry_MARE=vals.mean(),error_p90=np.quantile(abs(g.error_pct),.9),error_p95=np.quantile(abs(g.error_pct),.95),CI95_low=np.quantile(boot,.025),CI95_high=np.quantile(boot,.975),orientation_MAE=units.angle.mean(),rotation_shift_CV=cv.mean(),mask_failures=int(((g.clean_dice<.95)|(g.chi!=g.clean_chi)|(g.components!=g.clean_components)).sum()),unresolved_cases=int(g.unresolved_wall.sum()),median_seconds=g.total_seconds.median(),max_residual=g.one_step_rms.max()))
 return pd.DataFrame(rows)

def degrees_for(cases,stage):
 rows=[]
 for family in ['sphere','capsule','torus']:
  c=next(c for c in cases if c['row']['family']==family and c['row']['noise']==0 and c['row']['rep']==0)
  for N in CHECKPOINTS:rows.append(dict(stage=stage,case=c['row']['case'],family=family,N=N,**topology(c['snaps'][N])))
 path=OUT/'degree_results.csv';old=pd.read_csv(path) if path.exists() else pd.DataFrame();pd.concat([old[old.stage!=stage] if len(old) else old,pd.DataFrame(rows)],ignore_index=True).to_csv(path,index=False);return pd.DataFrame(rows)

def selection(summary,degrees):
 s=summary[summary.method=='learned'];rows=[]
 for tol in [.05,.1,.2,.5]:
  for N in CHECKPOINTS:
   details=[];passed=True
   for keys,g in s.groupby(['dim','noise']):
    a=g[g.N==N].iloc[0];b=g[g.N==40].iloc[0];geo=a.geometry_MARE-b.geometry_MARE;ang=a.orientation_MAE-b.orientation_MAE;valid=a.mask_failures<=b.mask_failures;fast=a.median_seconds<b.median_seconds or N==40
    ok=geo<=tol+1e-12 and (not np.isfinite(ang) or ang<=tol+1e-12) and valid and fast
    details.append(dict(dim=int(keys[0]),noise=float(keys[1]),geometry_delta_pp=geo,orientation_delta_deg=ang,mask_failures=int(a.mask_failures),mask_failures_40=int(b.mask_failures),faster=bool(fast),passed=bool(ok)));passed &= ok
   dn=degrees[degrees.N==N].set_index('family');d40=degrees[degrees.N==40].set_index('family');degree_ok=all(bool(dn.loc[f,'degree_certified']) and dn.loc[f,'degree']==d40.loc[f,'degree'] for f in dn.index);passed &= degree_ok
   rows.append(dict(tolerance=tol,N=N,passed=bool(passed),degree_ok=degree_ok,details=json.dumps(details)))
 df=pd.DataFrame(rows);df.to_csv(OUT/'selection_sensitivity.csv',index=False);eligible=df[(df.tolerance==.1)&df.passed];N=int(eligible.N.min()) if len(eligible) else 40
 return N,df

def baselines(cases,refs):
 coefficients={};rows=[]
 for c in refs:
  for r in conventional(c['f'],c['axis']):
   if np.isfinite(r['raw_measurement']) and r['raw_measurement']>0:coefficients[(c['row']['dim'],r['method'])]=c['truth']/r['raw_measurement']
 for c in cases:
  for m in conventional(c['f'],c['axis']):
   for calibrated in ([False,True] if np.isfinite(m['raw_measurement']) else [False]):
    coef=coefficients.get((c['row']['dim'],m['method']),1) if calibrated else 1;val=m['raw_measurement']*coef
    rows.append(dict(c['row'],**m,calibrated=calibrated,calibration_coefficient=coef,measured=val,truth=c['truth'],error_pct=100*(val/c['truth']-1)))
 return pd.DataFrame(rows)

def train_validate():
 assert not (OUT/'test_results.csv').exists(),'Test already scored: cannot overwrite the locked selection.'
 df=manifest();train=load_cases(df[df.split=='train'],'train',True);kernel=json.load(open(OUT/'kernel_by_iteration.json'))['kernels'] if (OUT/'kernel_by_iteration.json').exists() else fit(train);refs=[c for c in train if c['row']['role']=='calibration'];coeff=calibrations(refs,kernel)
 tr=evaluate(train,kernel,coeff);tr.to_csv(OUT/'train_results.csv',index=False)
 val=load_cases(df[df.split=='validation'],'validation');vr=evaluate(val,kernel,coeff);vr.to_csv(OUT/'validation_results.csv',index=False);summary=summarize(vr);summary.to_csv(OUT/'validation_summary.csv',index=False);dg=degrees_for(val,'validation');N,sens=selection(summary,dg)
 frozen=dict(selected_N=N,kernel_by_dimension={dim:kernel[dim][str(N)]['p'] for dim in ['2','3']},parameters=PARAMS,generation_kernel='central',dtype='float32',alpha=1,boundary='periodic with seven-voxel clean input margin',calibration_by_dimension={str(dim):coeff[(dim,N,'learned')] for dim in [2,3]},normalization='direct H/|H|; retain n if |H|<=1e-12',decision_rule='smallest N passing <=.1 pp geometry and <=.1 degree orientation increases in every validation dim/noise stratum; no additional mask/degree failures; measured speed improvement',validation_only=True,test_evaluated=False,freeze_time_utc=pd.Timestamp.now(tz='UTC').isoformat(),split_manifest_sha256=hashlib.sha256((OUT/'split_manifest.csv').read_bytes()).hexdigest(),kernel_file_sha256=hashlib.sha256((OUT/'kernel_by_iteration.json').read_bytes()).hexdigest())
 (OUT/'selected_configuration.json').write_text(json.dumps(frozen,indent=2));print('FROZEN SELECTED N',N,flush=True)
 # Append train-only coefficients and validation metrics in separate metadata file without changing frozen kernel identity.
 coeffrows=[dict(dim=k[0],N=k[1],method=k[2],coefficient=v,reference_case=f'cal{k[0]}',h=2.,D=1.) for k,v in coeff.items()];pd.DataFrame(coeffrows).to_csv(OUT/'calibration_coefficients.csv',index=False)
 kf=json.load(open(OUT/'kernel_by_iteration.json'));kf['calibration']=coeffrows;kf['validation_scores']=json.loads(summary.to_json(orient='records'));(OUT/'kernel_by_iteration.json').write_text(json.dumps(kf,indent=2))
 # Freeze the hash of the complete kernel metadata before test too.
 frozen['kernel_file_sha256']=hashlib.sha256((OUT/'kernel_by_iteration.json').read_bytes()).hexdigest();(OUT/'selected_configuration.json').write_text(json.dumps(frozen,indent=2));(OUT/'selection_lock.sha256').write_text(hashlib.sha256((OUT/'selected_configuration.json').read_bytes()).hexdigest())
 # Persist validation baselines only, preserving locked test chronology.
 baselines(val,refs).to_csv(OUT/'baseline_validation_results.csv',index=False)

def final_test():
 locked=hashlib.sha256((OUT/'selected_configuration.json').read_bytes()).hexdigest();assert locked==(OUT/'selection_lock.sha256').read_text();assert not (OUT/'test_results.csv').exists(),'Test scoring already exists; do not silently retune/retest.'
 frozen=json.load(open(OUT/'selected_configuration.json'));kernel=json.load(open(OUT/'kernel_by_iteration.json'))['kernels'];df=manifest();refs=load_cases(df[df.role=='calibration'],'references');coeff=calibrations(refs,kernel);cases=load_cases(df[df.split=='test'],'test');results=evaluate(cases,kernel,coeff,selected_N=frozen['selected_N']);results.to_csv(OUT/'test_results.csv',index=False);summarize(results).to_csv(OUT/'test_summary.csv',index=False);degrees_for(cases,'test');baselines(cases,refs).to_csv(OUT/'baseline_results.csv',index=False)
 # Paired per-case differences, then cluster bootstrap over independent prototypes.
 paired=[]
 for label,data in [('validation',pd.read_csv(OUT/'validation_results.csv')),('test',results)]:
  learned=data[data.method=='learned'];ref=learned[learned.N==40].set_index('case')
  for _,r in learned.iterrows():
   b=ref.loc[r['case']];paired.append(dict(split=label,case=r['case'],unit=r.unit,dim=r.dim,noise=r.noise,N=r.N,heldout_family=r.heldout_family,geometry_abs_error_delta_pp=abs(r.error_pct)-abs(b.error_pct),orientation_error_delta_deg=r.angle_error-b.angle_error,measurement_delta_vs40_pct=100*(r.measured/b.measured-1),total_seconds=r.total_seconds,seconds40=b.total_seconds))
 pd.DataFrame(paired).to_csv(OUT/'paired_differences.csv',index=False)
 assert locked==hashlib.sha256((OUT/'selected_configuration.json').read_bytes()).hexdigest()
 json.dump(dict(selection_sha256=locked,test_finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),test_rows=len(results),no_retuning=True),open(OUT/'test_audit.json','w'),indent=2);print('TEST DONE',len(results),flush=True)

def transfer():
 assert (OUT/'test_audit.json').exists();frozen=json.load(open(OUT/'selected_configuration.json'));kernel=json.load(open(OUT/'kernel_by_iteration.json'))['kernels'];df=manifest();sub=df[(df.split=='test')&(df.rep==0)&(df.noise==0)];allr=[]
 for params in [dict(J=1.,h=4.,D=1.),dict(J=1.,h=2.,D=.5)]:
  rf=df[df.role=='calibration'].copy();rf['case']=rf.case+f"_h{params['h']}_D{params['D']}";refs=load_cases(rf,'transfer_refs',params=params);coef=calibrations(refs,kernel,params)
  ss=sub.copy();ss['case']=ss.case+f"_h{params['h']}_D{params['D']}";cases=load_cases(ss,'transfer',params=params);r=evaluate(cases,kernel,coef,methods=['learned'],selected_N=frozen['selected_N']);allr.append(r)
 pd.concat(allr,ignore_index=True).to_csv(OUT/'transfer_results.csv',index=False);print('TRANSFER DONE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['design','train','test','transfer']);a=p.parse_args()
 if a.stage=='design':manifest()
 elif a.stage=='train':train_validate()
 elif a.stage=='test':final_test()
 else:transfer()
