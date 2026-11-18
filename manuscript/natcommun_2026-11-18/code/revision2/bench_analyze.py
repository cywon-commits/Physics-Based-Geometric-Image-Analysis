import numpy as np,pandas as pd,json
d=pd.read_csv('data/bench_raw.csv')
res={}
def cal_const(tr,col): return float(np.median(tr.truth/tr[col]))
rows=[]
for dim in (2,3):
    D=d[d.dim==dim]; tr=D[(D.split=='train')&(D.noise==0)&(D.family!='near_touch')]; te=D[D.split=='test']
    base=('crofton','ms') if dim==2 else ('mc',)
    meths={'PBGIA':('PBGIA',1.0)}
    for b in base: meths[f'{b}_cal']=(b,cal_const(tr,b))
    gcols=[c for c in D.columns if c.startswith('gauss_ms_' if dim==2 else 'gauss_mc_')]
    best=min(gcols,key=lambda c: np.mean(np.abs(tr[c]*cal_const(tr,c)/tr.truth-1)))
    meths['gauss_cal']=(best,cal_const(tr,best))
    for mcol in (('median_crofton','median_ms') if dim==2 else ('median_mc',)):
        meths[mcol+'_cal']=(mcol,cal_const(tr,mcol))   # calibrated on clean train median-filtered? use same clean train
    res[dim]=dict(gauss_sigma=best,constants={k:v[1] for k,v in meths.items()})
    for k,(col,c) in meths.items():
        e=100*(te[col]*c/te.truth-1)
        for (fam,nz),g in te.assign(e=e).groupby(['family','noise']):
            rows.append(dict(dim=dim,method=k,family=fam,noise=nz,MARE=g.e.abs().mean(),bias=g.e.mean(),n=len(g)))
R=pd.DataFrame(rows); R.to_csv('data/bench_summary_family.csv',index=False)
def overall(dim,nz,excl=True):
    q=R[(R.dim==dim)&(R.noise==nz)]
    if excl: q=q[q.family!='near_touch']
    return q.groupby('method').MARE.mean().round(3)
for dim in (2,3):
    print('dim',dim,res[dim]['gauss_sigma'])
    print(pd.DataFrame({nz:overall(dim,nz) for nz in (0,0.002,0.01)}).to_string())
    print(R[(R.dim==dim)&(R.noise==0)].pivot(index='family',columns='method',values='MARE').round(2).to_string())
json.dump(res,open('data/bench_constants.json','w'),indent=1)
