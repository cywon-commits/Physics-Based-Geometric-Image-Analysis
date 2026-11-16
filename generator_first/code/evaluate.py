from study import *
from classical import conventional
BASE=json.load(open(ROOT/'reference/PBGIA30_frozen.json'))

def old_heads(d):
 p=BASE['kernel_by_dimension'][str(d)];return {str(N):dict(geometry=dict(p=p),orientation=dict(p=p,type='gradient',mix=[1,0,0],which='min')) for N in CHECKPOINTS}

def measure(cases,heads,model,mode,coeffs=None,Ns=CHECKPOINTS):
 ref=next(c for c in cases if c['row']['role']=='reference');out=[]
 for N in Ns:
  coef=calibrate(ref,heads,N) if coeffs is None else coeffs[str(N)]
  for c in cases:
   if c is ref:continue
   r,_=readout(c['states'][N],c['c'],heads[str(N)],coef,c['axis'],float(c['row']['spacing']));diag=c['diagnostics'][N];out.append(dict(c['row'],model=model,mode=mode,truth=c['truth'],error_pct=100*(r['prediction']/c['truth']-1),coefficient=coef,**diag,**r,total_seconds=diag['initialization_seconds']+diag['field_seconds']+r['measurement_seconds']))
 return pd.DataFrame(out)

def summary(df,keys=['dim','model','mode','N']):
 out=[]
 for key,g in df.groupby(keys,dropna=False):
  unit=g.groupby('unit').error_pct.apply(lambda a:np.mean(abs(a)));e=g.error_pct.dropna().to_numpy();a=g.angle_error.dropna().to_numpy();cv=g.groupby(['unit','noise']).prediction.agg(lambda x:100*x.std(ddof=0)/max(abs(x.mean()),1e-12));rng=np.random.default_rng(SEED);boot=np.mean(rng.choice(unit.to_numpy(),(2000,len(unit))),axis=1) if len(unit) else [np.nan]
  out.append(dict(zip(keys,key if isinstance(key,tuple) else [key]))|dict(images=len(g),units=g.unit.nunique(),oriented_images=len(a),undefined_orientation=len(g)-len(a),geometry_MARE=float(unit.mean()),geometry_median=float(np.median(abs(e))) if len(e) else np.nan,geometry_RMSE=float(np.sqrt(np.mean(e*e))) if len(e) else np.nan,p90=float(np.percentile(abs(e),90)) if len(e) else np.nan,p95=float(np.percentile(abs(e),95)) if len(e) else np.nan,worst_unit=float(unit.max()),worst_image=float(np.max(abs(e))) if len(e) else np.nan,signed_bias=float(np.mean(e)) if len(e) else np.nan,CI_low=float(np.percentile(boot,2.5)),CI_high=float(np.percentile(boot,97.5)),angle_MAE=float(np.mean(a)) if len(a) else np.nan,angle_RMSE=float(np.sqrt(np.mean(a*a))) if len(a) else np.nan,angle_p90=float(np.percentile(a,90)) if len(a) else np.nan,angle_max=float(np.max(a)) if len(a) else np.nan,rotation_CV=float(cv.mean()),mean_seconds=float(g.total_seconds.mean()),median_seconds=float(g.total_seconds.median())))
 return pd.DataFrame(out)

def validate():
 assert not (OUT/'pretest_selection.json').exists();models=json.load(open(OUT/'models.json'));raw=[];cal=[]
 for d in [2,3]:
  base=models[str(d)]['B'];refrow=rows_for(d,'train').query("role=='reference'");base_ref=fields(refrow,base['generator']);bc={str(N):calibrate(base_ref[0],base['heads'],N) for N in CHECKPOINTS}
  for name,m in models[str(d)].items():
   for split in ['train','validation']:
    cc=fields(rows_for(d,split),m['generator']);rf=next(c for c in cc if c['row']['role']=='reference')
    variants=[('fixed_B_frozen_c',base['heads'],bc),('fixed_B_train_c',base['heads'],None),('retuned',m['heads'],None)]
    for mode,h,co in variants:
     r=measure(cc,h,name,mode); # fixed coefficient explicitly reapplied
     if co is not None:
      r['coefficient']=r.N.astype(str).map(co);r['prediction']=r.raw_measurement*r.coefficient;r['error_pct']=100*(r.prediction/r.truth-1)
     raw.append(r.assign(split_scored=split))
     if split=='train':
      for N in CHECKPOINTS:cal.append(dict(dim=d,model=name,mode=mode,N=N,coefficient=co[str(N)] if co else calibrate(rf,h,N),reference_case=rf['row']['case']))
    if name=='B':
     r=measure(cc,old_heads(d),'A','original_frozen',Ns=[30]);r['coefficient']=BASE['calibration_by_dimension'][str(d)];r['prediction']=r.raw_measurement*r.coefficient;r['error_pct']=100*(r.prediction/r.truth-1);raw.append(r.assign(split_scored=split));raw.append(measure(cc,old_heads(d),'A','train_recalibrated',Ns=[30]).assign(split_scored=split))
    pd.concat(raw).to_csv(OUT/'train_validation_results.csv',index=False);print('EVAL',d,name,split,flush=True)
 pd.DataFrame(cal).to_csv(OUT/'calibration.csv',index=False);df=pd.concat(raw,ignore_index=True)
 for split in ['train','validation']:summary(df[df.split_scored==split]).to_csv(OUT/f'{split}_summary.csv',index=False)
 df[df['mode'].str.startswith('fixed')].to_csv(OUT/'ablation_fixed_observer.csv',index=False);df[df['mode']=='retuned'].to_csv(OUT/'ablation_retrained_observer.csv',index=False)
 sm=summary(df[(df.split_scored=='validation')&(df['mode']=='retuned')&~df.model.isin(['binary_init','raw_tied','raw_scale_matched'])]);selection={};paretos=[]
 for d,g in sm.groupby('dim'):
  g=g.copy();p=g[['geometry_MARE','angle_MAE','median_seconds']].to_numpy();g['pareto']=[not np.any(np.all(p<=a,axis=1)&np.any(p<a,axis=1)) for a in p];paretos.append(g);gp=g.sort_values(['geometry_MARE','angle_MAE','median_seconds']).iloc[0];op=g.sort_values(['angle_MAE','geometry_MARE','median_seconds']).iloc[0];eligible=g[(g.geometry_MARE<=g.geometry_MARE.min()+.25)&(g.angle_MAE<=g.angle_MAE.min()+.25)];chosen=eligible.sort_values('median_seconds').iloc[0] if len(eligible) else None
  sel={}
  # Primary comparisons frozen per arm, not test-best N selection.
  specs={name:g[g.model==name].sort_values(['geometry_MARE','angle_MAE','median_seconds']).iloc[0] for name in ['B','D','E','F','G']}
  specs.update(geometry_priority=gp,orientation_priority=op)
  if chosen is not None:specs['default']=chosen
  for preset,s in specs.items():
   name=s['model'];N=int(s['N']);m=models[str(d)][name];coeff=next(a['coefficient'] for a in cal if a['dim']==d and a['model']==name and a['mode']=='retuned' and a['N']==N);sel[preset]=dict(model=name,N=N,generator=m['generator'],heads=m['heads'][str(N)],coefficient=coeff,validation_geometry_MARE=s.geometry_MARE,validation_angle_MAE=s.angle_MAE)
  # Fixed observer C/E/F at the same per-arm chosen N; both frozen and train-recalibrated c.
  for name in ['D','E','F','G']:
   N=sel['B']['N'];m=models[str(d)][name]
   for mode in ['fixed_B_frozen_c','fixed_B_train_c']:
    coef=next(a['coefficient'] for a in cal if a['dim']==d and a['model']==name and a['mode']==mode and a['N']==N);sel[name+'_'+mode]=dict(model=name,N=N,generator=m['generator'],heads=models[str(d)]['B']['heads'][str(N)],coefficient=coef)
  sel['A']=dict(model='A',N=30,generator=parameter_config(),heads=old_heads(d)['30'],coefficient=BASE['calibration_by_dimension'][str(d)])
  sel['A_train_c']=dict(sel['A'],coefficient=calibrate(fields(rows_for(d,'train').query("role=='reference'"),parameter_config())[0],old_heads(d),30))
  selection[str(d)]=sel
 pd.concat(paretos).to_csv(OUT/'validation_pareto.csv',index=False)
 frozen=dict(configurations=selection,freeze_utc=pd.Timestamp.now(tz='UTC').isoformat(),test_seen=False,hashes={fn:hashlib.sha256((OUT/fn).read_bytes()).hexdigest() for fn in ['protocol.json','manifest.csv','models.json','calibration.csv']});dump('pretest_selection.json',frozen);(OUT/'pretest_selection.sha256').write_text(hashlib.sha256((OUT/'pretest_selection.json').read_bytes()).hexdigest());print('FROZEN',flush=True)

def classical_rows(rows):
 ref=rows[rows.role=='reference'].iloc[0].to_dict();f,t,_,_=fixture(ref);coef={r['method']:t/r['raw_measurement'] for r in conventional(f,None) if np.isfinite(r['raw_measurement']) and r['raw_measurement']>0};out=[]
 for row in rows[rows.role!='reference'].to_dict('records'):
  f,t,axis,_=fixture(row)
  for r in conventional(f,axis):
   if axis is not None and not np.isfinite(r['angle_error']):r['angle_error']=90.
   for calibrated in ([False,True] if r['method'] in coef else [False]):
    v=r['raw_measurement']*(coef[r['method']] if calibrated else 1)*row['spacing']**(row['dim']-1);out.append(dict(row,**r,model=r['method'],mode='calibrated' if calibrated else 'raw',N=0,truth=t,prediction=v,error_pct=100*(v/t-1),total_seconds=r['measurement_seconds']))
 return pd.DataFrame(out)

def test():
 assert not (OUT/'test_results.csv').exists();frozen=json.load(open(OUT/'pretest_selection.json'));lock=hashlib.sha256((OUT/'pretest_selection.json').read_bytes()).hexdigest();assert lock==(OUT/'pretest_selection.sha256').read_text()
 for fn,sha in frozen['hashes'].items():assert hashlib.sha256((OUT/fn).read_bytes()).hexdigest()==sha
 out=[];classical=[];hist=[]
 for d in [2,3]:
  configs=frozen['configurations'][str(d)];rows=rows_for(d,'test');groups={}
  for preset,c in configs.items():groups.setdefault(json.dumps(c['generator'],sort_keys=True),[]).append((preset,c))
  for key,items in groups.items():
   Ns=sorted(set(c['N'] for _,c in items));cc=fields(rows,json.loads(key),Ns)
   for preset,c in items:
    N=c['N'];r=measure(cc,{str(N):c['heads']},preset,'frozen_test',{str(N):c['coefficient']},[N]);r['source_model']=c['model'];out.append(r)
   print('TEST',d,[p for p,_ in items],flush=True)
  classical.append(classical_rows(rows))
  if d==2:
   ref=rows.query("role=='reference'").iloc[0].to_dict();fr,tr,_,_=fixture(ref)
   for variant,params in [('texture',(1,1,5)),('length',(.1,1,10))]:
    fn=source_legacy_function(variant);K=old_kernel()
    refval=fn(fr.astype(float),K,params)[0];coef=tr/refval
    for row in rows.query("role!='reference'").to_dict('records'):
     f,t,a,_=fixture(row);tic=time.perf_counter();result=fn(f.astype(float),K,params);elapsed=time.perf_counter()-tic;v=result[0];theta=np.radians(result[2]);pred_axis=np.array([np.sin(theta),np.cos(theta)]);err=float(np.degrees(np.arccos(np.clip(abs(pred_axis@a),0,1)))) if a is not None else np.nan
     for mode,co in [('original_raw',1.),('train_calibrated',coef)]:hist.append(dict(row,model='H_'+variant,mode=mode,N=params[2],truth=t,prediction=v*co,error_pct=100*(v*co/t-1),angle_error=err,total_seconds=elapsed,D=params[0],h=params[1],init='binary',boundary='wrap' if variant=='texture' else 'symm',exchange_sum=float((K*K+K.T*K.T).sum()),observer='original totalAmount and tiltingAngle'))

 df=pd.concat(out,ignore_index=True);df.to_csv(OUT/'test_results.csv',index=False);summary(df).to_csv(OUT/'test_summary.csv',index=False);summary(df,['dim','model','mode','N','noise']).to_csv(OUT/'test_noise_summary.csv',index=False);summary(df,['dim','model','mode','N','family','noise']).to_csv(OUT/'test_family_summary.csv',index=False);cl=pd.concat(classical,ignore_index=True);cl.to_csv(OUT/'test_conventional.csv',index=False);summary(cl).to_csv(OUT/'test_conventional_summary.csv',index=False);pd.DataFrame(hist).to_csv(OUT/'test_legacy.csv',index=False)
 # Paired, unit-resampled C/D/E/F minus B, all samples remain clustered.
 pairs=[]
 for d,g in df.groupby('dim'):
  b=g[g.model=='B'].set_index('case')
  for name,h in g.groupby('model'):
   if name=='B':continue
   z=h.set_index('case').join(b[['error_pct','angle_error']],rsuffix='_B');z['geom_delta']=abs(z.error_pct)-abs(z.error_pct_B);z['angle_delta']=z.angle_error-z.angle_error_B
   for metric in ['geom_delta','angle_delta']:
    a=z.groupby('unit')[metric].mean().dropna().to_numpy();rng=np.random.default_rng(SEED);bb=np.mean(rng.choice(a,(3000,len(a))),axis=1);pairs.append(dict(dim=d,model=name,metric=metric,units=len(a),mean=float(a.mean()),CI_low=float(np.percentile(bb,2.5)),CI_high=float(np.percentile(bb,97.5))))
 pd.DataFrame(pairs).to_csv(OUT/'paired_bootstrap.csv',index=False);dump('test_audit.json',dict(selection_sha256=lock,completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),no_retuning=True,all_cases_retained=True));print('TEST COMPLETE',len(df),flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--validate',action='store_true');p.add_argument('--test',action='store_true');a=p.parse_args()
 if a.validate:validate()
 if a.test:test()
