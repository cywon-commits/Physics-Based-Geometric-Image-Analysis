"""Replay only train B field and head fitting; no new test evaluation."""
from study import *
results=[];models=json.load(open(OUT/'models.json'));tic=time.perf_counter();cpu=time.process_time()
for d in [2,3]:
 cases=fields(rows_for(d,'train',True),parameter_config(),feat=True);heads=fit_heads(cases,SEED);a=models[str(d)]['B']['heads'];matches=[]
 for N,h in heads.items():
  for name in ['geometry','orientation']:
   assert h[name]['p']==a[N][name]['p'],(d,N,name)
  assert all(h['orientation'][k]==a[N]['orientation'][k] for k in ['type','mix','which'])
  matches.append(int(N))
 results.append(dict(dim=d,matched_N=matches,exact_coefficients=True))
dump('reproduction_check.json',dict(train_only=True,results=results,wall_seconds=time.perf_counter()-tic,process_CPU_seconds=time.process_time()-cpu));print('REPRODUCTION PASSED')
