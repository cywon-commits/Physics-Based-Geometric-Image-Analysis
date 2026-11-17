from study import *

def raw(ds,p,N,model,split):
 a=evaluate(ds,p,N);r=ds['rows'].copy();r['model']=model;r['N']=N;r['scored_split']=split;r['truth']=ds['truth'];r['I']=a['I'];r['I_scalar']=a['I_scalar'];r['I_vector']=a['I_vector'];r['q']=a['I']/ds['truth'];r['gap']=a['gap'];r['theta']=a['theta'];r['angle_error']=[np.degrees(abs(.5*np.arctan2(np.sin(2*(t-np.arctan2(ax[1],ax[0]))),np.cos(2*(t-np.arctan2(ax[1],ax[0])))))) if ax is not None else np.nan for t,ax in zip(a['theta'],ds['axes'])];r['unit_error']=a['unit_error'];r['field_seconds_per_image']=a['field_seconds']/len(r);r['observer_seconds_per_image']=a['observer_seconds']/len(r);return r

def summary(raw):
 rows=[]
 for (split,model,N),r in raw.groupby(['scored_split','model','N']):
  clean=r[r.noise==0];q=clean.groupby('unit').q.mean();l=np.log(q);rot=clean.groupby('unit').I.agg(lambda x:np.var(x)/np.mean(x)**2).mean();angle=clean.groupby('unit').angle_error.agg(lambda x:np.nanmean((x/1)**2) if x.notna().any() else np.nan).mean();join=r.merge(clean[['unit','rep','I']],on=['unit','rep'],suffixes=('','_clean'));nz=join[join.noise>0];noise=nz.assign(err=lambda z:np.log(z.I/z.I_clean)**2).groupby('unit').err.mean().mean()
  parts=dict(prop=float(np.var(l)),rotation=float(rot),angle=float(angle),noise=float(noise));rows.append(dict(split=split,model=model,N=N,q_CV_pct=100*float(np.std(q)/np.mean(q)),log_q_std=float(np.std(l)),**parts,**{f'objective_w{w}':parts['prop']+parts['rotation']+.0001*parts['angle']+w*parts['noise'] for w in [0,1,4]}))
 return pd.DataFrame(rows)

def validate():
 assert (OUT/'training_cost.json').exists(),'Training is incomplete';assert not (OUT/'model_freeze_2d.json').exists();assert not (OUT/'test_raw_2d.csv').exists();models=json.load(open(OUT/'models.json'));df=manifest();allrows=[]
 for split in ['train','validation']:
  ds=load(df[(df.split==split)&(df.role=='geometry')])
  for name,m in models.items():
   for N in [5,10,20,30] if name=='G4' or name.startswith('fit_') else [10]:allrows.append(raw(ds,m['p'],N,name,split))
   print('VALIDATE',split,name,flush=True)
  pd.concat(allrows).to_csv(OUT/'train_validation_raw.csv',index=False)
 sm=summary(pd.concat(allrows));sm.to_csv(OUT/'validation_summary.csv',index=False);v=sm[(sm.split=='validation')&(sm.N==10)].copy();p=v[['prop','rotation','angle','noise']].to_numpy();v['pareto']=[not np.any(np.all(p<=z,axis=1)&np.any(p<z,axis=1)) for z in p];v.to_csv(OUT/'validation_pareto.csv',index=False);print('VALIDATE2D COMPLETE',flush=True)

def freeze():
 assert not (OUT/'model_freeze_2d.json').exists();assert not (OUT/'test_raw_2d.csv').exists();sm=pd.read_csv(OUT/'validation_summary.csv');models=json.load(open(OUT/'models.json'));selected={name:dict(model=name,p=models[name]['p'],N=10) for name in ['G4','legacy','pilot']}
 for w in [0,1,4]:
  g=sm[(sm.split=='validation')&(sm.N==10)];base=g[g.model=='G4'].iloc[0];tr=sm[(sm.split=='train')&(sm.N==10)].set_index('model');allowed=[]
  for _,a in g.iterrows():
   name=a.model
   if name=='G4' or (models[name]['weight']==w and a.prop<=base.prop+1e-14 and a[f'objective_w{w}']<=base[f'objective_w{w}']+1e-14 and tr.loc[name,f'objective_w{w}']<=tr.loc['G4',f'objective_w{w}']+1e-14):allowed.append(a)
  best=min(allowed,key=lambda x:x[f'objective_w{w}']);selected[f'selected_w{w}']=dict(model=best.model,p=models[best.model]['p'],N=10,weight=w,validation_objective=float(best[f'objective_w{w}']))
 # N sensitivity is validation-only on the selected robust model, with a matching G4 arm.
 name=selected['selected_w1']['model'];v=sm[(sm.split=='validation')&(sm.model==name)];best=v.sort_values('objective_w1').iloc[0];N=int(best.N);selected['selected_N']=dict(model=name,p=models[name]['p'],N=N,weight=1);selected['G4_matched_N']=dict(model='G4',p=G4.tolist(),N=N)
 fixed=dict(configurations=selected,test_seen=False,freeze_utc=pd.Timestamp.now(tz='UTC').isoformat(),hashes={name:sha(OUT/name) for name in ['protocol.json','manifest_2d.csv','models.json','validation_summary.csv']});dump('model_freeze_2d.json',fixed);(OUT/'model_freeze_2d.sha256').write_text(sha(OUT/'model_freeze_2d.json'));print('FROZEN',selected,flush=True)

def check_lock():
 assert sha(OUT/'model_freeze_2d.json')==(OUT/'model_freeze_2d.sha256').read_text();f=json.load(open(OUT/'model_freeze_2d.json'))
 for name,h in f['hashes'].items():assert sha(OUT/name)==h
 return f

def stats(r):
 q=r.groupby('unit').q.mean();l=np.log(q);a=r.angle_error.dropna();rng=np.random.default_rng(SEED);u=q.to_numpy();boot=rng.choice(u,(2000,len(u)));bootcv=100*np.std(boot,axis=1)/np.mean(boot,axis=1);cv=r.groupby('unit').I.agg(lambda x:100*np.std(x)/np.mean(x)).mean()
 return dict(images=len(r),units=len(q),q_mean=float(q.mean()),q_CV_pct=float(100*np.std(q)/q.mean()),log_q_std=float(np.std(l)),q_CV_CI_low=float(np.percentile(bootcv,2.5)),q_CV_CI_high=float(np.percentile(bootcv,97.5)),log_q_std_CI_low=float(np.percentile(np.std(np.log(boot),axis=1),2.5)),log_q_std_CI_high=float(np.percentile(np.std(np.log(boot),axis=1),97.5)),q_median=float(q.median()),q_p90=float(q.quantile(.9)),q_p95=float(q.quantile(.95)),q_min=float(q.min()),q_max=float(q.max()),rotation_shift_CV_pct=float(cv),angle_count=len(a),angle_MAE=float(a.mean()),angle_RMSE=float(np.sqrt(np.mean(a*a))),angle_p95=float(a.quantile(.95)),angle_max=float(a.max()))

def test():
 assert not (OUT/'test_raw_2d.csv').exists(),'No repeated test or retuning';frozen=check_lock();df=manifest();ds=load(df.query("split=='test'"));refs=load(df.query("role=='reference'"));rows=[];rr=[];cache={}
 for preset,c in frozen['configurations'].items():
  k=json.dumps(dict(p=c['p'],N=c['N']))
  if k not in cache:cache[k]=(raw(ds,c['p'],c['N'],preset,'test'),raw(refs,c['p'],c['N'],preset,'train_reference'))
  a,b=cache[k];rows.append(a.assign(model=preset,source_model=c['model']));rr.append(b.assign(model=preset));print('TEST',preset,flush=True)
 r=pd.concat(rows);r.to_csv(OUT/'test_raw_2d.csv',index=False);refsraw=pd.concat(rr);refsraw.to_csv(OUT/'reference_raw.csv',index=False);generate_diagnostics(r,refsraw);dump('test_audit_2d.json',dict(selection_sha256=sha(OUT/'model_freeze_2d.json'),test_images=408,independent_prototypes=17,no_retuning=True,all_cases_retained=True));print('TEST2D COMPLETE',flush=True)

def generate_diagnostics(r,ref):
 summaries=[];family=[];curv=[];rot=[];sweep=[];pairwise=[];paired=[]
 for (name,noise),g in r.groupby(['model','noise']):
  summaries.append(dict(model=name,N=int(g.N.iloc[0]),noise=noise,**stats(g)))
  for fam,h in g.groupby('family'):family.append(dict(model=name,noise=noise,family=fam,**stats(h)))
  for unit,h in g.groupby('unit'):
   row=h.iloc[0];curv.append(dict(model=name,noise=noise,unit=unit,R=row.R,curvature_proxy=1/row.R,family=row.family,q_mean=h.q.mean(),q_CV_pct=100*h.q.std(ddof=0)/h.q.mean(),angle_MAE=h.angle_error.mean()))
  for key,axis in [('rotation', 'shift_id'),('shift','rotation_id')]:
   z=g.groupby(['unit',axis]).I.agg(lambda x:100*x.std(ddof=0)/x.mean());rot.append(dict(model=name,noise=noise,variation=key,CV_pct=float(z.mean()),groups=len(z)))
  q=g.groupby('unit').q.mean()
  for u,v in itertools.combinations(q.index,2):pairwise.append(dict(model=name,noise=noise,unit1=u,unit2=v,ratio=q[u]/q[v],log_ratio=np.log(q[u]/q[v])))
  refg=ref[ref.model==name]
  for _,rr in refg.iterrows():
   c=rr.truth/rr.I;e=100*(c*g.q-1);unit_e=pd.DataFrame(dict(unit=g.unit.to_numpy(),err=abs(e.to_numpy()))).groupby('unit').err.mean();sweep.append(dict(model=name,noise=noise,reference_unit=rr.unit,reference_family=rr.family,reference_R=rr.R,c=c,MARE=float(unit_e.mean()),bias=float(e.mean()),RMSE=float(np.sqrt(np.mean(e*e))),p95=float(np.percentile(abs(e),95)),worst=float(abs(e).max())))
  clean=r[(r.model==name)&(r.noise==0)].set_index(['unit','rep']);x=g.set_index(['unit','rep']).I/clean.I;paired.append(dict(model=name,noise=noise,mean_log_I_shift=float(np.log(x).mean()),RMS_log_I_shift=float(np.sqrt(np.mean(np.log(x)**2))),max_abs_log_I_shift=float(abs(np.log(x)).max())))
 pd.DataFrame(summaries).to_csv(OUT/'calibration_free_summary.csv',index=False);pd.DataFrame(summaries).to_csv(OUT/'test_by_noise.csv',index=False);pd.DataFrame(family).to_csv(OUT/'test_by_family.csv',index=False);pd.DataFrame(curv).to_csv(OUT/'test_by_curvature.csv',index=False);pd.DataFrame(rot).to_csv(OUT/'rotation_shift_summary.csv',index=False);pd.DataFrame(sweep).to_csv(OUT/'reference_ruler_sweep.csv',index=False);pd.DataFrame(pairwise).to_csv(OUT/'pairwise_q_ratios.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'paired_noise_drift.csv',index=False)
 # This is the primary frozen-observer ablation, not observer retuning.
 r.to_csv(OUT/'ablation_fixed_observer.csv',index=False)
if __name__=='__main__':
 import sys
 {'validate2d':validate,'freeze2d':freeze,'test2d':test}[sys.argv[1]]()
