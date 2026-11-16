from train_joint import *
from pbgia_s3_integer_degree import degree

def cap(n):
 idx=np.indices(n.shape[:3]);edge=np.minimum(idx,np.array(n.shape[:3])[:,None,None,None]-1-idx).min(0);w=np.clip(edge/3,0,1)[...,None];pole=np.array([0,0,0,-1.]);out=w*n+(1-w)*pole;return out/np.linalg.norm(out,axis=-1,keepdims=True)

def topology(n):
 margin=np.inf
 for perm in itertools.permutations(range(3)):
  offs=np.zeros((4,3),int)
  for j,a in enumerate(perm):offs[j+1]=offs[j];offs[j+1,a]+=1
  vs=np.stack([n[tuple(slice(o,sz-1+o) for sz,o in zip(n.shape[:3],off))] for off in offs],-2);dots=np.sum(vs*vs.sum(-2)[...,None,:],-1);margin=min(margin,float(dots.min()))
 idx=np.indices(n.shape[:3]);edge=np.minimum(idx,np.array(n.shape[:3])[:,None,None,None]-1-idx).min(0);collar=float(n[...,-1][edge<=3].max());nc=cap(n);targets=[(.37139,-.52177,.64829,.40981),(-.6,.2,.3,.7),(.2,.8,-.5,-.2)];ds=[degree(nc,t) for t in targets];agree=len(set(d['integer_degree'] for d in ds))==1
 return dict(degree=ds[0]['integer_degree'],target_agreement=agree,hemisphere_min_dot=margin,negative_collar_max_s=collar,nonvanishing_certified=bool(margin>1e-7 and collar<0),degree_certified=bool(margin>1e-7 and collar<0 and agree),targets=json.dumps(ds),boundary_spread=max(d['boundary_spread'] for d in ds))

def base_result(cases,tag='PBGIA30'):
 ref=next(c for c in cases if c['row']['role']=='calibration');dim=ref['row']['dim'];p=BASE['kernel_by_dimension'][str(dim)];refc=ref['truth']/np.trace(tensor(ref['states'][30],p));out=[]
 for c in cases:
  if c['row']['role']=='calibration':continue
  r,_=observe(c['states'][30],c['config'],p,c['axis'],'gradient');m=c['metrics'][30]
  for suffix,coef in [('frozen',BASE['calibration_by_dimension'][str(dim)]),('train_recalibrated',refc)]:
   val=r['raw_measurement']*coef;out.append(dict(c['row'],model=tag+'_'+suffix,N=30,truth=c['truth'],measured=val,error_pct=100*(val/c['truth']-1),calibration_coefficient=coef,observer='gradient',orientation_variant='selected',total_seconds=m['field_seconds']+m['initialization_seconds']+r['measurement_seconds'],**{k:v for k,v in m.items() if k!='N'},**r))
 return pd.DataFrame(out)

def summary(data):
 rows=[]
 for keys,g in data.groupby(['dim','model','N','noise']):
  u=g.groupby('unit').agg(error=('error_pct',lambda a:np.mean(abs(a))),angle=('angle_error','mean'))
  rows.append(dict(zip(['dim','model','N','noise'],keys),n_units=len(u),n_cases=len(g),geometry_MARE=u.error.mean(),p90=np.quantile(abs(g.error_pct),.9),p95=np.quantile(abs(g.error_pct),.95),angle_MAE=u.angle.mean(),angle_RMSE=float(np.sqrt(np.nanmean(g.angle_error**2))) if g.angle_error.notna().any() else np.nan,angle_max=g.angle_error.max(),orientation_miss_rate=float((~g.orientation_defined[g.angle_error.notna()]).mean()) if g.angle_error.notna().any() else np.nan,min_clean_dice=g[g.noise==0].clean_dice.min() if (g.noise==0).any() else np.nan,min_dice=g.clean_dice.min(),mask_failures=int(((g.clean_dice<.995)|(g.chi!=g.clean_chi)|(g.components!=g.clean_components)).sum()),rotation_shift_CV=float(g.groupby('unit').measured.agg(lambda a:100*a.std(ddof=0)/abs(a.mean())).mean()),median_seconds=g.total_seconds.median(),max_residual=g.one_step_rms.max(),unresolved=int(g.unresolved_wall.sum())))
 return pd.DataFrame(rows)

def baseline_conventional(cases):
 ref=next(c for c in cases if c['row']['role']=='calibration');coefs={r['method']:ref['truth']/r['raw_measurement'] for r in conventional(ref['f'],ref['axis']) if np.isfinite(r['raw_measurement']) and r['raw_measurement']>0};rows=[]
 for c in cases:
  if c['row']['role']=='calibration':continue
  for r in conventional(c['f'],c['axis']):
   for calibrated in ([False,True] if r['method'] in coefs else [False]):
    coef=coefs[r['method']] if calibrated else 1;value=r['raw_measurement']*coef;rows.append(dict(c['row'],**r,calibrated=calibrated,calibration_coefficient=coef,truth=c['truth'],measured=value,error_pct=100*(value/c['truth']-1)))
 return pd.DataFrame(rows)

def legacy_results(rows):
 refrow=make_manifest().query("dim==2 and role=='calibration'").iloc[0].to_dict();f,*_=fixture(refrow);n,Q,info=historical(f);reftruth=fixture(refrow)[4];coef=reftruth/np.trace(tensor(n,[-1/12,0,0]));out=[]
 for row in rows.to_dict('records'):
  f,clean,normal,dist,truth,axis=fixture(row);tic=time.perf_counter();n,Q,info=historical(f);sec=time.perf_counter()-tic;T=tensor(n,[-1/12,0,0]);val=coef*np.trace(T);m=metrics(n,f,clean,normal,dist);out.append(dict(row,model='historical_texture_5',**info,**read_axis(Q,axis,'min'),truth=truth,measured=val,error_pct=100*(val/truth-1),calibration_coefficient=coef,total_seconds=sec,**m))
 return pd.DataFrame(out)

def validate():
 assert not (OUT/'test_results.csv').exists();models=json.load(open(OUT/'trained_models.json'));df=make_manifest();data=[];base=[];top=[];train_all=[];con=[];newcoeff={};caldiag=[]
 for dim in [2,3]:
  newcoeff[str(dim)]={}
  ref=df[(df.dim==dim)&(df.role=='calibration')]
  for model,cfg in models[str(dim)].items():
   for split in ['train','validation']:
    subset=df[(df.dim==dim)&(df.split==split)&(df.role=='geometry')];rows=pd.concat([subset,ref],ignore_index=True)
    cc=fields(rows,cfg['generator'],stage=f'full_{split}_{dim}_{model}');result=scored_rows(cc,cfg['observers'],model)
    if split=='train':train_all.append(result)
    else:data.append(result)
    refcase=next(c for c in cc if c['row']['role']=='calibration')
    if split=='train':
     for it,m in refcase['metrics'].items():caldiag.append(dict(dim=dim,model=model,case=refcase['row']['case'],**m))
    newcoeff[str(dim)][model]={str(N):refcase['truth']/np.trace(tensor(refcase['states'][N],cfg['observers'][str(N)]['p'])) for N in CHECKPOINTS}
    if model=='observer_only':
     if split=='validation':base.append(base_result(cc));con.append(baseline_conventional(cc))
    if dim==3 and split=='validation':
     for family in ['sphere','capsule','torus']:
      c=next(c for c in cc if c['row']['family']==family and c['row']['noise']==0 and c['row']['rep']==0)
      for N in CHECKPOINTS:top.append(dict(stage='validation',model=model,N=N,case=c['row']['case'],family=family,**topology(c['states'][N])))
    # Save only representative small checkpoint bundles; other cases are reproducible via keys.
    if split=='validation':
     c=next(c for c in cc if c['row']['family']==('stadium' if dim==2 else 'capsule') and c['row']['noise']==0 and c['row']['rep']==0);np.savez_compressed(SNAPS/f'validation_{dim}_{model}.npz',f=c['f'],**{'n'+str(N):s for N,s in c['states'].items()})
    print('FULL',split,dim,model,flush=True)
    pd.concat(train_all).to_csv(OUT/'train_results.csv',index=False)
    if data:pd.concat(data+base).to_csv(OUT/'validation_results.csv',index=False)
 raw=pd.concat(data+base,ignore_index=True);tr=pd.concat(train_all,ignore_index=True);s=summary(raw);s.to_csv(OUT/'validation_summary.csv',index=False);pd.DataFrame(top).to_csv(OUT/'topology_audit.csv',index=False);pd.concat(con).to_csv(OUT/'baseline_validation.csv',index=False)
 pd.DataFrame(caldiag).to_csv(OUT/'calibration_diagnostics.csv',index=False)
 selection={};pareto=[]
 for dim in [2,3]:
  bs=raw[(raw.dim==dim)&(raw.model=='PBGIA30_frozen')];candidates=[]
  for model in models[str(dim)]:
   for N in CHECKPOINTS:
    g=raw[(raw.dim==dim)&(raw.model==model)&(raw.N==N)];tg=tr[(tr.dim==dim)&(tr.model==model)&(tr.N==N)];clean=g[g.noise==0];trainclean=tg[tg.noise==0];calm=next(a for a in caldiag if a['dim']==dim and a['model']==model and a['N']==N);minclean=min(clean.clean_dice.min(),trainclean.clean_dice.min(),calm['clean_dice']);calok=calm['chi']==calm['clean_chi'] and calm['components']==calm['clean_components'];cleanok=calok and minclean>=.995 and (clean.chi==clean.clean_chi).all() and (clean.components==clean.clean_components).all() and (trainclean.chi==trainclean.clean_chi).all() and (trainclean.components==trainclean.clean_components).all()
    fails=lambda a:int(((a.clean_dice<.995)|(a.chi!=a.clean_chi)|(a.components!=a.clean_components)).sum());noisyok=all(fails(g[g.noise==noise])<=fails(bs[bs.noise==noise]) for noise in [.002,.01]);dg=pd.DataFrame(top);dg=dg[(dg.model==model)&(dg.N==N)] if dim==3 else pd.DataFrame();degreeok=True if dim==2 else bool(dg.degree_certified.all() and (dg.degree==dg.family.map({'sphere':1,'capsule':1,'torus':0})).all())
    scores=s[(s.dim==dim)&(s.model==model)&(s.N==N)];geom=scores.geometry_MARE.mean();angle=scores.angle_MAE.mean();timev=scores.median_seconds.mean();feasible=bool(cleanok and noisyok and degreeok);loss=geom+angle
    entry=dict(dim=dim,model=model,N=N,strict_feasible=feasible,min_clean_dice=minclean,clean_ok=bool(cleanok),noise_no_worse=bool(noisyok),degree_ok=bool(degreeok),geometry_MARE=geom,orientation_MAE=angle,seconds=timev,loss=loss);candidates.append(entry)
  feasible=[c for c in candidates if c['strict_feasible']];pool=feasible if feasible else candidates;best=min(pool,key=lambda a:a['loss']);allowed=[a for a in pool if a['geometry_MARE']<=best['geometry_MARE']+.1 and a['orientation_MAE']<=best['orientation_MAE']+.1];chosen=min(allowed,key=lambda a:a['seconds']);model=chosen['model'];N=chosen['N'];selection[str(dim)]=dict(**chosen,generator=models[str(dim)][model]['generator'],observer=models[str(dim)][model]['observers'][str(N)],calibration=newcoeff[str(dim)][model][str(N)],recommended=bool(chosen['strict_feasible']),production_policy='joint candidate' if chosen['strict_feasible'] else 'no validated upgrade; retain PBGIA-30, candidate experimental only')
  for a in candidates:a['pareto_nondominated']=not any(b['geometry_MARE']<=a['geometry_MARE'] and b['orientation_MAE']<=a['orientation_MAE'] and b['seconds']<=a['seconds'] and (b['geometry_MARE']<a['geometry_MARE'] or b['orientation_MAE']<a['orientation_MAE'] or b['seconds']<a['seconds']) for b in candidates)
  pareto+=candidates
 pd.DataFrame(pareto).to_csv(OUT/'joint_pareto.csv',index=False)
 frozen=dict(selection=selection,protocol_hash=hashlib.sha256((OUT/'experiment_config.json').read_bytes()).hexdigest(),split_hash=hashlib.sha256((OUT/'split_manifest.csv').read_bytes()).hexdigest(),trained_models_hash=hashlib.sha256((OUT/'trained_models.json').read_bytes()).hexdigest(),frozen_utc=pd.Timestamp.now(tz='UTC').isoformat(),test_evaluated=False,baseline=BASE)
 (OUT/'pretest_selection.json').write_text(json.dumps(frozen,indent=2));(OUT/'pretest_selection.sha256').write_text(hashlib.sha256((OUT/'pretest_selection.json').read_bytes()).hexdigest());print('FROZEN',[(d,c['model'],c['N'],c['strict_feasible']) for d,c in selection.items()],flush=True)


def final_test():
 assert not (OUT/'test_results.csv').exists();lock=(OUT/'pretest_selection.sha256').read_text();assert lock==hashlib.sha256((OUT/'pretest_selection.json').read_bytes()).hexdigest();frozen=json.load(open(OUT/'pretest_selection.json'));models=json.load(open(OUT/'trained_models.json'));df=make_manifest();data=[];controls=[];top=[];con=[];initabl=[];legacy=[];observer_test=[]
 for dim in [2,3]:
  sel=frozen['selection'][str(dim)];rows=pd.concat([df[(df.dim==dim)&(df.split=='test')],df[(df.dim==dim)&(df.role=='calibration')]],ignore_index=True)
  for label,cfg in [('joint_selected',models[str(dim)][sel['model']]),('E0_observer_only',models[str(dim)]['observer_only'])]:
   cc=fields(rows,cfg['generator'],stage=f'test_{dim}_{label}');r=scored_rows(cc,cfg['observers'],label,extra_observers=(label=='E0_observer_only'))
   if label=='joint_selected':r=r[r.N==sel['N']].copy();data.append(r)
   else:
    # Independent observer-only comparison at fixed N=30, all observers predeclared.
    observer_test.append(r[r.N==30]);controls.append(r[(r.N==30)&(r.orientation_variant=='selected')]);controls.append(base_result(cc));con.append(baseline_conventional(cc))
   for family in (['sphere','capsule','torus'] if dim==3 else []):
    c=next(c for c in cc if c['row']['family']==family and c['row']['noise']==0 and c['row']['rep']==0);N=sel['N'] if label=='joint_selected' else 30;top.append(dict(stage='test',model=label,N=N,case=c['row']['case'],family=family,**topology(c['states'][N])))
   c=next(c for c in cc if c['row']['family']==('stadium' if dim==2 else 'capsule') and c['row']['noise']==0 and c['row']['rep']==0);np.savez_compressed(SNAPS/f'test_{dim}_{label}.npz',f=c['f'],**{'n'+str(N):n for N,n in c['states'].items()})
  if dim==2:legacy.append(legacy_results(rows[rows.role!='calibration']))
 pd.concat(observer_test).to_csv(OUT/'orientation_observer_comparison_test.csv',index=False)
 allr=pd.concat(data+controls,ignore_index=True);allr.to_csv(OUT/'test_results.csv',index=False);summary(allr).to_csv(OUT/'test_summary.csv',index=False);pd.concat(con).to_csv(OUT/'baseline_comparison.csv',index=False);pd.concat(legacy).to_csv(OUT/'historical_test.csv',index=False);old=pd.read_csv(OUT/'topology_audit.csv');pd.concat([old,pd.DataFrame(top)],ignore_index=True).to_csv(OUT/'topology_audit.csv',index=False)
 # Historical control is separate: binary init, h=1, original unnormalized coefficient scale, stale HJ tensor.
 gate={}
 for dim in [2,3]:
  g=allr[(allr.dim==dim)&(allr.model=='joint_selected')];clean=g[g.noise==0];baseg=allr[(allr.dim==dim)&(allr.model=='PBGIA30_frozen')]
  fails=lambda a:int(((a.clean_dice<.995)|(a.chi!=a.clean_chi)|(a.components!=a.clean_components)).sum())
  clean_ok=bool((clean.clean_dice>=.995).all() and (clean.chi==clean.clean_chi).all() and (clean.components==clean.clean_components).all())
  noisy_ok=all(fails(g[g.noise==noise])<=fails(baseg[baseg.noise==noise]) for noise in [.002,.01])
  dt=pd.DataFrame(top);dt=dt[(dt.model=='joint_selected')] if dim==3 else pd.DataFrame();degree_ok=True if dim==2 else bool(dt.degree_certified.all() and (dt.degree==dt.family.map({'sphere':1,'capsule':1,'torus':0})).all())
  gate[str(dim)]=dict(validation_recommended=frozen['selection'][str(dim)]['recommended'],test_clean_ok=clean_ok,test_noisy_ok=bool(noisy_ok),test_degree_ok=bool(degree_ok),production_accepted=bool(frozen['selection'][str(dim)]['recommended'] and clean_ok and noisy_ok and degree_ok))
 (OUT/'deployment_gate.json').write_text(json.dumps(dict(selection_sha256=lock,by_dimension=gate,policy='held-out rejection allowed, no retuning'),indent=2))
 assert lock==hashlib.sha256((OUT/'pretest_selection.json').read_bytes()).hexdigest();(OUT/'test_audit.json').write_text(json.dumps(dict(selection_sha256=lock,completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),no_retuning=True),indent=2));print('TEST DONE',len(allr),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['validate','test']);a=p.parse_args();validate() if a.stage=='validate' else final_test()
