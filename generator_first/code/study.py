"""Generator-first CPU study; no reconstruction metrics or selection policies."""
from observer import fit_heads
from engine import *
from data import make_manifest,SEED
import pandas as pd,resource,platform,sys,os
from scipy.stats import qmc

TRIALS=[]
def dump(name,obj):
 (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=True))
def rows_for(dim,split,proxy=False):
 df=make_manifest();r=df[(df.dim==dim)&((df.split==split)|(df.role=='reference'))]
 if proxy:r=r[(r.role=='reference')|((r.rep<2)&r.noise.isin([0.,.01]))]
 return r

def fields(rows,c,checkpoints=CHECKPOINTS,feat=False):return [rollout(r,c,checkpoints,feat) for r in rows.to_dict('records')]
def score(cases,heads,Ns=CHECKPOINTS):
 ref=next(c for c in cases if c['row']['role']=='reference');out=[]
 for N in Ns:
  errors={};angles={};coef=calibrate(ref,heads,N)
  for c in cases:
   if c is ref:continue
   r,_=readout(c['states'][N],c['c'],heads[str(N)],coef,c['axis']);u=c['row']['unit'];errors.setdefault(u,[]).append(abs(r['prediction']/c['truth']-1))
   if c['axis'] is not None:angles.setdefault(u,[]).append(r['angle_error']/90)
  out.append(dict(N=N,geometry=float(np.mean([np.mean(v) for v in errors.values()])),angle=float(np.mean([np.mean(v) for v in angles.values()]))))
 return out

def search(dim,family,heads,rows,Ns=CHECKPOINTS,budget=12,fixed_gp=None):
 rng=np.random.default_rng(SEED+dim*100+len(TRIALS));n=6 if family=='free' else 3;bounds=np.array([[-.14,.04],[0,.27],[0,.08]]+([[0,.6],[0,.17],[0,.17]] if n==6 else []));v=qmc.scale(qmc.LatinHypercube(n,seed=int(rng.integers(1000000))).random(budget),bounds[:,0],bounds[:,1]);best=None;start=time.perf_counter()
 for i,x in enumerate(v):
  # Two independent starts, local proposal refinement in the second half.
  if i>=budget//2 and best is not None:
   center=np.array(best[1]['gp']+(best[1]['ep'] if n==6 else []));x=np.clip(center+.45*(x-bounds.mean(1)),bounds[:,0],bounds[:,1])
  if i in [0,budget//2]:x[:3]=[0,0,0] if i==0 else [-1/12,0,0]
  c=parameter_config(family,x[:3] if fixed_gp is None else fixed_gp,x[3:] if n==6 else [0,0,0]);t=time.perf_counter();cases=fields(rows,c,Ns);s=score(cases,heads,Ns)
  losses={str(w):float(np.mean([a['geometry']+w*a['angle'] for a in s])) for w in [0.,.05,.2]};loss=losses['0.05'];elapsed=time.perf_counter()-t
  TRIALS.append(dict(dim=dim,family=family,scope='shared' if len(Ns)>1 else 'N_specific',N=0 if len(Ns)>1 else Ns[0],fixed_gp=fixed_gp is not None,index=i,configuration=json.dumps(c),loss=loss,loss_geometry_only=losses['0.0'],loss_angle_weight_02=losses['0.2'],seconds=elapsed,cumulative_family_seconds=time.perf_counter()-start,per_N=json.dumps(s),status='finite_budget_search_no_convergence_claim'))
  pd.DataFrame(TRIALS).to_csv(OUT/'generator_trials.csv',index=False)
  if best is None or loss<best[0]:best=(loss,c)
  print('GEN',dim,family,list(Ns),i,round(loss,5),flush=True)
 return best[1]

def train():
 assert not (OUT/'pretest_selection.json').exists();models={};obs=[];timings=[]
 for d in [2,3]:
  rows=rows_for(d,'train',True);t=time.perf_counter();base=fields(rows,parameter_config(),feat=True);heads=fit_heads(base,SEED);models[str(d)]={'B':dict(generator=parameter_config(),heads=heads)}
  generators={'G':parameter_config('axial',[-1/12,0,0])}
  for fam,name in [('axial','D'),('tied','E'),('free','F')]:generators[name]=search(d,fam,heads,rows)
  # Separate optimization for N=5/10/30; same frozen B observer per N.
  for N in [5,10,30]:generators[f'D_N{N}']=search(d,'axial',heads,rows,[N],6)
  # Hold optimized G fixed while changing only exchange.
  gp=generators['D']['gp']
  generators['E_fixedG']=parameter_config('tied',gp)
  generators['F_fixedG']=search(d,'free',heads,rows,CHECKPOINTS,6,fixed_gp=gp)
  for name,c in generators.items():
   cases=fields(rows,c,feat=True);h=fit_heads(cases,SEED);models[str(d)][name]=dict(generator=c,heads=h)
   for N,hh in h.items():
    for kind in ['geometry','orientation']:obs.append(dict(dim=d,model=name,N=N,head=kind,parameters=json.dumps(hh[kind])))
   dump('models.json',models);pd.DataFrame(obs).to_csv(OUT/'observer_trials.csv',index=False)
  # Binary and raw-tied scaling controls with exactly B's observer.
  binary=parameter_config();binary['init']='binary';raw=parameter_config('tied',gp,normalized=False);alpha=exchange_kernel(d,raw).sum()/(2*d);equiv=parameter_config('tied',gp);equiv.update(h=2/alpha,D=1/alpha)
  for name,c in [('binary_init',binary),('raw_tied',raw),('raw_scale_matched',equiv)]:models[str(d)][name]=dict(generator=c,heads=heads,control=True)
  timings.append(dict(stage=f'train_{d}',wall_seconds=time.perf_counter()-t,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump('models.json',models);pd.DataFrame(timings).to_csv(OUT/'training_timings.csv',index=False)
 print('TRAIN COMPLETE',flush=True)

def protocol():
 make_manifest();dump('protocol.json',dict(seed=SEED,model='dynamic grad(s), periodic, synchronous normalized H, no div(v)',physics=dict(J=1,h=2,D=1),Ns=list(CHECKPOINTS),primary='generator changes under B frozen observer; independently retuned observer secondary',generator_budget=dict(shared_evaluations_per_family_per_dim=12,N_specific_axial_evaluations_each=6,N_specific=[5,10,30],fixedG_free_evaluations=6,starts=2),proxy='4 units x 2 rotations x clean/1% + train reference per dimension',loss='unit mean relative absolute geometry + .05*unit mean angle/90; also record weights0,.2',observer='separate geometry MARE and orientation angle only, DE120 each head/N, train reference coefficient',selection='validation Pareto geometry/angle/single-pass component latency in warmed process; fastest within .25 percentage points best geometry and .25deg best angle; else geometry and orientation priority separately',test_policy='fresh grouped prototypes, freeze before single test, no test retuning',exclusions='No reconstruction/topology/residual/width loss, candidate rejection, selection or fallback',extension='G3 only if validation fixed-observer G2 improvement; not automatically expanded',cost_sensitivity='equal eval primary; common measured-walltime prefix, not separate equal-time optimizer',spacing='generated gradient divided by physical a; observer physical derivatives and quadrature equivalent factor a^(d-2) with train reference calibration',limitations='finite-budget local/random search, no global optimum certificate',comparisons='fixed B observer comparisons at B-selected N, coefficient frozen and separately train-only recalibrated; per-arm retuned N reported separately'))
 dump('environment.json',dict(python=sys.version,platform=platform.platform(),cpu=os.cpu_count(),numpy=np.__version__,dtype='float32 state,float64 reductions',gpu_used=False,threads=os.environ.get('OPENBLAS_NUM_THREADS'),source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'code').glob('*.py')}))

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.smoke:
  protocol();r=[]
  for d in [2,3]:
   for fam in ['axial','tied','free']:
    c=rollout(rows_for(d,'train').iloc[0].to_dict(),parameter_config(fam,[-1/12,.1,.01],[.2,.03,.04]),[5]);r.append(dict(dim=d,family=fam,unit_error=c['diagnostics'][5]['unit_error'],finite=c['diagnostics'][5]['finite']))
  dump('smoke.json',r);print(r)
 if a.train:train()
