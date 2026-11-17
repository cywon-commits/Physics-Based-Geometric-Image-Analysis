"""Fixed-spacing domain-padding probe; no fit/selection or raw benchmark rewrite."""
from run_study import *
def run():
 assert (OUT/'raw.csv').exists();rows=[]
 df=make_manifest();probes=df[(df.suite=='digital_planar')&(df.noise==0)&(df.shift_id==1)&df.width.isin([2.3,4.6])].groupby(['dim','width']).tail(1)
 for row in probes.to_dict('records'):
  cached={}
  for padding in [0,32]:
   rr=dict(row,size=row['size']+padding);f,n,t,axis,roi,meta=fixture(rr);n,timing=generate(f,float(row['width']),1.,10);values=measure(n,1.,roi,include_orientation=False)
   for v in values:
    if padding==0:cached[v['method']]=v['value']
    rows.append(dict(dim=row['dim'],width=row['width'],rotation_id=row['rotation_id'],shift_id=row['shift_id'],domain_size=rr['size'],padding=padding,spacing=1.,method=v['method'],value=v['value'],baseline_value=cached[v['method']],change_pct=100*(v['value']/cached[v['method']]-1),roi_side=meta['roi_side']))
 pd.DataFrame(rows).to_csv(OUT/'boundary_padding.csv',index=False);print('PADDING CHECK COMPLETE',len(rows))
if __name__=='__main__':run()
