"""Post-freeze deployment timing, paired cluster CIs, margin/refinement diagnostics."""
from run_fixed_study import *
from predict_fixed import predict

def run():
 frozen=json.load(open(OUT/'selected_configuration.json'));locked=hashlib.sha256((OUT/'selected_configuration.json').read_bytes()).hexdigest();assert locked==(OUT/'selection_lock.sha256').read_text();df=manifest();N=frozen['selected_N'];kernels=json.load(open(OUT/'kernel_by_iteration.json'))['kernels'];times=[]
 # Warmed deployment timing is separate from one-pass checkpoint experiment costs.
 subset=df[(df.split=='test')&(df.noise==0)&(df.rep==0)].groupby(['dim','family']).head(1)
 for row in subset.to_dict('records'):
  f,clean,nu,d,truth,axis=geometry(row)
  for iteration in sorted(set([N,40])):
   cfg=dict(frozen,selected_N=iteration,kernel_by_dimension={str(dim):kernels[str(dim)][str(iteration)]['p'] for dim in [2,3]})
   cfg['calibration_by_dimension']={str(dim):float(pd.read_csv(OUT/'calibration_coefficients.csv').query("dim==@dim and N==@iteration and method=='learned'").coefficient.iloc[0]) for dim in [2,3]}
   predict(f>0,cfg) # unrecorded warmup, explicit timing-only repeat
   for repeat in range(5):
    nn,result=predict(f>0,cfg);times.append(dict(case=row['case'],unit=row['unit'],family=row['family'],dim=row['dim'],noise=0,repeat=repeat,kind='single_N_deployment',**{k:v for k,v in result.items() if k.endswith('seconds') or k=='N'}))
  # Baseline timings are their own measured operations (not charged PBGIA solve).
  for repeat in range(5):
   for b in conventional(f,axis):times.append(dict(case=row['case'],unit=row['unit'],family=row['family'],dim=row['dim'],noise=0,repeat=repeat,kind='conventional',N=0,method=b['method'],measurement_seconds=b['measurement_seconds'],total_seconds=b['measurement_seconds']))
 pd.DataFrame(times).to_csv(OUT/'timing.csv',index=False)
 # Explicit independent-geometry cluster bootstrap for paired changes and family holdouts.
 pairs=pd.read_csv(OUT/'paired_differences.csv');cis=[]
 for keys,g in pairs.groupby(['split','dim','noise','N','heldout_family']):
  unit=g.groupby('unit')[['geometry_abs_error_delta_pp','orientation_error_delta_deg','measurement_delta_vs40_pct']].mean()
  for metric in unit:
   vals=unit[metric].dropna().to_numpy()
   if not len(vals):continue
   rng=np.random.default_rng(61116);boots=rng.choice(vals,(3000,len(vals)),replace=True).mean(1);cis.append(dict(zip(['split','dim','noise','N','heldout_family'],keys),metric=metric,n_units=len(vals),mean=vals.mean(),CI95_low=np.quantile(boots,.025),CI95_high=np.quantile(boots,.975)))
 pd.DataFrame(cis).to_csv(OUT/'paired_bootstrap_ci.csv',index=False)
 # More detailed test stratification (held-out shape families reported separately).
 raw=pd.read_csv(OUT/'test_results.csv');summary=[]
 for keys,g in raw.groupby(['dim','noise','family','heldout_family','N','method']):
  units=g.groupby('unit').error_pct.agg(lambda v:np.mean(abs(v)));summary.append(dict(zip(['dim','noise','family','heldout_family','N','method'],keys),n_units=len(units),n_cases=len(g),MARE=units.mean(),p95=np.quantile(abs(g.error_pct),.95),orientation_MAE=g.angle_error.mean()))
 pd.DataFrame(summary).to_csv(OUT/'test_family_summary.csv',index=False)
 # Same physical sphere, varied spacing. Fixed lattice dynamics -> wall's physical width changes.
 checks=[]
 for spacing,size in [(1.,48),(.75,64),(.5,96)]:
  for role,R in [('calibration',4.25),('test',7.3)]:
   row=dict(case=f'refine_{role}_{spacing}',unit=f'refine_{role}',split='diagnostic',role=role,family='sphere',heldout_family=False,dim=3,R=R/spacing,size=size,rep=0,phase=.21,noise=0.,seed=71717,noise_seed=81818,spacing=spacing)
   c=checkpoints(row,PARAMS);p=kernels['3'][str(N)]['p'];M=np.trace(tensor(c['snaps'][N],p));m=next(x for x in c['metrics'] if x['N']==N);checks.append(dict(experiment='physical_grid_refinement',spacing=spacing,size=size,role=role,R_physical=R,truth=c['truth'],raw_measurement=M,**m))
 for size in [48,64,80]:
  row=dict(case=f'margin_{size}',unit='margin_torus',split='diagnostic',role='test',family='torus',heldout_family=False,dim=3,R=12,size=size,rep=0,phase=.21,noise=0.,seed=71717,noise_seed=81818,spacing=1.)
  c=checkpoints(row,PARAMS);M=np.trace(tensor(c['snaps'][N],kernels['3'][str(N)]['p']));m=next(x for x in c['metrics'] if x['N']==N);checks.append(dict(experiment='domain_margin_only',spacing=1.,size=size,role='test',R_physical=12,truth=c['truth'],raw_measurement=M,**m))
 ck=pd.DataFrame(checks)
 for spacing,g in ck[ck.experiment=='physical_grid_refinement'].groupby('spacing'):
  ref=g[g.role=='calibration'].iloc[0];idx=g.index;ck.loc[idx,'calibration_coefficient']=ref.truth/ref.raw_measurement;ck.loc[idx,'error_pct']=100*(ck.loc[idx,'raw_measurement']*ref.truth/ref.raw_measurement/ck.loc[idx,'truth']-1)
 coeff=frozen['calibration_by_dimension']['3'];idx=ck.experiment=='domain_margin_only';ck.loc[idx,'calibration_coefficient']=coeff;ck.loc[idx,'error_pct']=100*(ck.loc[idx,'raw_measurement']*coeff/ck.loc[idx,'truth']-1);ck.to_csv(OUT/'refinement_margin_checks.csv',index=False)
 # Completed measured profiles and failures, no analytic distances enter evolution.
 profiles=[]
 for row in subset[subset.family.isin(['sphere','capsule','torus','stadium'])].to_dict('records'):
  c=checkpoints(row,PARAMS)
  for iteration in CHECKPOINTS:
   n=c['snaps'][iteration];s=n[...,-1];bins=np.linspace(-4,4,41)
   for lo,hi in zip(bins[:-1],bins[1:]):
    w=(c['d']>=lo)&(c['d']<hi)
    if w.any():profiles.append(dict(case=row['case'],family=row['family'],dim=row['dim'],N=iteration,distance=(lo+hi)/2,mean_s=float(s[w].mean()),count=int(w.sum())))
 pd.DataFrame(profiles).to_csv(OUT/'wall_profiles.csv',index=False)
 assert hashlib.sha256((OUT/'selected_configuration.json').read_bytes()).hexdigest()==locked
 print('ADDITIONAL CHECKS DONE',flush=True)
if __name__=='__main__':run()
