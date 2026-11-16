from joint_study_core import *
from scipy.optimize import differential_evolution
import argparse,resource
KOBS_FIXED={'central':[0,0,0],'fourth':[-1/12,0,0],'Scharr':[0,3/16,0]}
LEDGER=pd.read_csv(OUT/'training_expense.csv').to_dict('records') if (OUT/'training_expense.csv').exists() else []
def proxy_rows(dim):
 df=make_manifest();tr=df[(df.split=='train')&(df.dim==dim)];clean=tr[(tr.noise==0)&(tr.rep==0)&(tr.role=='geometry')].groupby('family').head(1);elong='stadium' if dim==2 else 'capsule';extra=tr[(tr.family==elong)&(tr.unit==clean[clean.family==elong].unit.iloc[0])&(((tr.rep==1)&(tr.noise==0))|((tr.rep==0)&(tr.noise==.01)))];return pd.concat([clean,extra,tr[tr.role=='calibration']],ignore_index=True)

def fields(rows,c,features_needed=False,save=False,stage='train'):
 tic=time.perf_counter();cases=[rollout(row,c,save=save,features_needed=features_needed) for row in rows.to_dict('records')];LEDGER.append(dict(stage=stage,dim=int(rows.iloc[0].dim),config=json.dumps(c),images=len(rows),steps_per_image=40,seconds=time.perf_counter()-tic,features=features_needed));pd.DataFrame(LEDGER).to_csv(OUT/'training_expense.csv',index=False);return cases

def learn_observers(cases,seed=261116):
 dim=cases[0]['row']['dim'];ref=next(c for c in cases if c['row']['role']=='calibration');train=[c for c in cases if c['row']['role']!='calibration'];result={}
 for N in CHECKPOINTS:
  for c in cases:
   if N not in c['features']:c['features'][N]=features(c['states'][N])
  def loss(p):
   coeff=ref['truth']/np.trace(feature_tensor(ref['features'][N],p,dim));errors=[];angles=[];groups={}
   for c in train:
    T=feature_tensor(c['features'][N],p,dim);val=np.trace(T)*coeff;errors.append((val/c['truth']-1)**2)
    e=read_axis(T,c['axis'])['angle_error']
    if np.isfinite(e):angles.append((min(e,10)/10)**2)
    groups.setdefault((c['row']['unit'],c['row']['noise']),[]).append(val)
   cv=np.mean([(np.std(v)/np.mean(v))**2 for v in groups.values()]);return float(np.mean(errors)+.05*np.mean(angles)+cv)
  tic=time.perf_counter();res=differential_evolution(loss,[(-.14,.03),(0,.3),(0,.08)],seed=seed+dim,popsize=5,maxiter=11,polish=False,tol=0,atol=0,workers=1);p=res.x.tolist();coeff=ref['truth']/np.trace(feature_tensor(ref['features'][N],p,dim));tensors=[]
  for c in train:
   if c['axis'] is not None:
    T=feature_tensor(c['features'][N],p,dim);B,V,G=orientation_tensors(c['states'][N],c['config'],T);tensors.append((c['axis'],G,B,V))
  choices=[]
  for obs in ['gradient','bond','direct']:
   for which in ['min','max']:
    err=[read_axis({'gradient':G,'bond':B,'direct':V}[obs],a,which)['angle_error'] for a,G,B,V in tensors];choices.append(dict(observer=obs,mix=[1,0,0],which=which,MAE=float(np.mean(err)),objective=float(np.mean(np.minimum(err,10)**2))))
  for a in range(5):
   for b in range(5-a):
    w=np.array([a,b,4-a-b])/4
    for which in ['min','max']:
     err=[read_axis(sum(v*normalized_tensor(T) for v,T in zip(w,[G,B,V])),axis,which)['angle_error'] for axis,G,B,V in tensors];choices.append(dict(observer='ensemble',mix=w.tolist(),which=which,MAE=float(np.mean(err)),objective=float(np.mean(np.minimum(err,10)**2))))
  chosen=min(choices,key=lambda a:a['objective']);result[str(N)]=dict(p=p,c=float(coeff),observer=chosen['observer'],mix=chosen['mix'],which=chosen['which'],observer_candidates=choices,evaluations=res.nfev,optimizer_status=str(res.message),objective=res.fun,training_seconds=time.perf_counter()-tic)
 return result

def scored_rows(cases,obs,model,extra_observers=False):
 ref=next(c for c in cases if c['row']['role']=='calibration');dim=ref['row']['dim'];coeff={N:ref['truth']/np.trace(tensor(ref['states'][N],obs[str(N)]['p'])) for N in CHECKPOINTS};rows=[]
 for c in cases:
  if c['row']['role']=='calibration':continue
  for N in CHECKPOINTS:
   o=obs[str(N)];choices=[(o['observer'],o.get('which','min'),o.get('mix',[1,0,0]),'selected')]
   if extra_observers:choices += [(name,'min',[1,0,0],name+'_min') for name in ['gradient','bond','direct']]+[(name,'max',[1,0,0],name+'_max') for name in ['bond','direct']]
   for observer,which,mix,label in choices:
    result,_=observe(c['states'][N],c['config'],o['p'],c['axis'],observer,mix,which);value=result['raw_measurement']*coeff[N];m=c['metrics'][N];rows.append(dict(c['row'],model=model,observer=observer,orientation_variant=label,which=which,mix=json.dumps(mix),gp=json.dumps(c['config']['gp']),ep=json.dumps(c['config']['ep']),exchange=c['config']['family'],exchange_sum=exchange_kernel(dim,c['config']).sum(),normalized=c['config']['normalized'],initialization=c['config']['init'],kernel_obs=json.dumps(o['p']),calibration_coefficient=coeff[N],truth=c['truth'],measured=value,error_pct=100*(value/c['truth']-1),cache_key=c['key'],total_seconds=m['field_seconds']+m['initialization_seconds']+result['measurement_seconds'],**m,**result))
 return pd.DataFrame(rows)

def candidate_score(cases,observers,baseline):
 ref=next(c for c in cases if c['row']['role']=='calibration');train=[c for c in cases if c['row']['role']!='calibration'];rows=[]
 for N in CHECKPOINTS:
  o=observers[str(N)];coef=ref['truth']/np.trace(tensor(ref['states'][N],o['p']));errs=[];ang=[];clean_fail=0;noisy_fail=0;base_noisy=0;min_dice=1
  for c in train:
   m=c['metrics'][N];r,_=observe(c['states'][N],c['config'],o['p'],c['axis'],o['observer'],o['mix'],o['which']);errs.append((r['raw_measurement']*coef/c['truth']-1)**2)
   if np.isfinite(r['angle_error']):ang.append((min(r['angle_error'],10)/10)**2)
   fail=m['clean_dice']<.995 or m['chi']!=m['clean_chi'] or m['components']!=m['clean_components']
   if c['row']['noise']==0:clean_fail+=int(fail);min_dice=min(min_dice,m['clean_dice'])
   else:
    noisy_fail+=int(fail);bm=baseline[c['row']['case']];base_noisy+=int(bm['clean_dice']<.995 or bm['chi']!=bm['clean_chi'] or bm['components']!=bm['clean_components'])
  objective=np.mean(errs)+.05*np.mean(ang);feasible=clean_fail==0 and noisy_fail<=base_noisy;penalty=1000*clean_fail+1000*max(0,noisy_fail-base_noisy)+100*(1-min_dice)
  rows.append(dict(N=N,objective=float(objective),penalized=float(objective+penalty),feasible=feasible,clean_failures=clean_fail,noisy_failures=noisy_fail,min_clean_dice=min_dice,geometry_RMSE=float(np.sqrt(np.mean(errs))*100),orientation_scaled_mse=float(np.mean(ang))))
 return rows

def sample_candidates(family,rng,count,center=None):
 from scipy.stats import qmc
 dims=6 if family=='free' else 3;sampler=qmc.LatinHypercube(d=dims,seed=int(rng.integers(1,1000000)));u=sampler.random(count);bounds=np.array([[-.12,.03],[0,.23],[0,.06]]+([[0,.65],[0,.15],[0,.15]] if family=='free' else []))
 values=qmc.scale(u,bounds[:,0],bounds[:,1])
 if center is not None:
  v=np.array(center['gp']+(center['ep'] if family=='free' else []));values=np.clip(v+(values-(bounds[:,0]+bounds[:,1])/2)*.5,bounds[:,0],bounds[:,1]);values[0]=v
 configs=[parameter_config(family,v[:3],v[3:] if family=='free' else [0,0,0]) for v in values]
 if center is None:
  configs[0]=parameter_config(family,[0,0,0],[.25,.02,.03] if family=='free' else [0,0,0]);configs[1]=parameter_config(family,[-1/12,.1,0],[.32,0,0] if family=='free' else [0,0,0])
 return configs

def train():
 assert not (OUT/'test_results.csv').exists();all_models={};search=[];stage1=[];abl=[];generators={};measurements={}
 for dim in [2,3]:
  rows=proxy_rows(dim);base=fields(rows,parameter_config(),features_needed=True,stage='observer_only');baseline={c['row']['case']:c['metrics'][30] for c in base};obs0=learn_observers(base);all_models[str(dim)]={};stage1.append(scored_rows(base,obs0,'E0_observer_only',True));all_models[str(dim)]['observer_only']=dict(generator=parameter_config(),observers=obs0)
  # Controlled ablations: generation/exchange amplitude, binary initialization, observer fixed.
  abconfigs=[('E0_fourth',parameter_config('axial',[-1/12,0,0])),('E1_central_normalized',parameter_config('tied')),('E1_central_raw',parameter_config('tied',normalized=False)),('E1_fourth_normalized',parameter_config('tied',[-1/12,0,0])),('E1_fourth_raw',parameter_config('tied',[-1/12,0,0],normalized=False)),('E2_control',parameter_config('free',[0,0,0],[.25,.02,.03])),('E0_binary',parameter_config(init='binary'))]
  abl.append(scored_rows(base,obs0,'E0_EDT',True))
  for tag,cfg in abconfigs:
   cc=fields(rows,cfg,stage='controlled_ablation');abl.append(scored_rows(cc,obs0,tag,True));print('ABLATION',dim,tag,flush=True)
  pd.concat(abl).to_csv(OUT/'factorial_ablation_raw.csv',index=False)
  for family in ['axial','tied','free']:
   rng=np.random.default_rng(261116+dim*100+len(family));centers=[None,None];working_obs=obs0;roundbest=None
   for roundid in range(2):
    winner=None
    for start in range(2):
     configs=sample_candidates(family,rng,6,centers[start]);bestlocal=None
     for k,cfg in enumerate(configs):
      cc=fields(rows,cfg,stage=f'joint_{dim}_{family}_r{roundid}_s{start}_c{k}');scores=candidate_score(cc,working_obs,baseline);best=min(scores,key=lambda a:a['penalized']);entry=dict(dim=dim,family=family,round=roundid,start=start,candidate=k,config=json.dumps(cfg),**best);search.append(entry)
      if bestlocal is None or best['penalized']<bestlocal[0]:bestlocal=(best['penalized'],cfg)
      if winner is None or best['penalized']<winner[0]:winner=(best['penalized'],cfg,cc)
     centers[start]=bestlocal[1]
    roundbest=winner;working_obs=learn_observers(winner[2],seed=261116);print('ROUND',dim,family,roundid,'score',winner[0],winner[1],flush=True)
    pd.DataFrame(search).to_csv(OUT/'generator_search.csv',index=False)
   cfg=roundbest[1];all_models[str(dim)][family]=dict(generator=cfg,observers=working_obs)
   generators.setdefault(str(dim),{})[family]=cfg;measurements.setdefault(str(dim),{})[family]=working_obs
   (OUT/'trained_generator_kernels.json').write_text(json.dumps(generators,indent=2));(OUT/'trained_measurement_kernels.json').write_text(json.dumps(measurements,indent=2));(OUT/'trained_models.json').write_text(json.dumps(all_models,indent=2))
 pd.concat(stage1).to_csv(OUT/'orientation_observer_comparison.csv',index=False);pd.concat(abl).to_csv(OUT/'factorial_ablation_raw.csv',index=False);print('JOINT TRAINING DONE',flush=True)

if __name__=='__main__':train()
