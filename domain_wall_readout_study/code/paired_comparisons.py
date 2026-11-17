"""Paired prototype bootstrap on frozen outputs; no fitting or selection."""
from data import *
def run():
 r=pd.read_csv(OUT/'analysis_raw.csv');r=r[~r.is_reference];rows=[];keys=['suite','dim','width','spacing','noise','truth_domain']
 for key,g in r.groupby(keys,dropna=False):
  base=g[g.method=='energy'][['case','unit','error_global_c_pct']].rename(columns={'error_global_c_pct':'base_error'})
  if not len(base):continue # marching plane uses a different footprint
  for method,h in g.groupby('method'):
   if method=='energy':continue
   z=h.merge(base,on=['case','unit'],validate='one_to_one');delta=abs(z.error_global_c_pct)-abs(z.base_error);unit=delta.groupby(z.unit).mean().to_numpy();result=dict(zip(keys,key));result.update(method=method,baseline='energy',units=len(unit),MARE_difference_pp=float(unit.mean()))
   if len(unit)>=2:
    digest=hashlib.sha256(str((key,method)).encode()).digest();rng=np.random.default_rng(int.from_bytes(digest[:4],'little'));boot=rng.choice(unit,(3000,len(unit))).mean(axis=1);result.update(CI_low=float(np.percentile(boot,2.5)),CI_high=float(np.percentile(boot,97.5)))
   else:result.update(CI_low=np.nan,CI_high=np.nan)
   rows.append(result)
 pd.DataFrame(rows).to_csv(OUT/'paired_readout_differences.csv',index=False);print('PAIRED COMPARISONS COMPLETE')
if __name__=='__main__':run()
