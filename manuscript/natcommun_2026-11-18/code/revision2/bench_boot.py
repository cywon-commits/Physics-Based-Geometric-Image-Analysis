import numpy as np,pandas as pd,json
d=pd.read_csv('data/bench_raw.csv'); C=json.load(open('data/bench_constants.json'))
rng=np.random.default_rng(1); out=[]
for dim in (2,3):
    c=C[str(dim)]['constants']; g=C[str(dim)]['gauss_sigma']
    comps={'conv_cal':('crofton' if dim==2 else 'mc',c['crofton_cal' if dim==2 else 'mc_cal']),'gauss_cal':(g,c['gauss_cal']),'median_cal':('median_crofton' if dim==2 else 'median_mc',c['median_crofton_cal' if dim==2 else 'median_mc_cal'])}
    for nz in (0,0.002,0.01):
        te=d[(d.dim==dim)&(d.split=='test')&(d.noise==nz)&(d.family!='near_touch')]
        ep=(te.PBGIA/te.truth-1).abs()*100
        for name,(col,k) in comps.items():
            ec=(te[col]*k/te.truth-1).abs()*100; diff=(ep-ec).values; fam=te.family.values
            # family-stratified bootstrap of the family-averaged MARE difference
            fams=np.unique(fam); bs=[]
            for _ in range(4000):
                bs.append(np.mean([np.mean(rng.choice(diff[fam==f],size=(fam==f).sum())) for f in fams]))
            est=np.mean([diff[fam==f].mean() for f in fams])
            out.append(dict(dim=dim,noise=nz,comparator=name,diff=round(est,3),lo=round(np.percentile(bs,2.5),3),hi=round(np.percentile(bs,97.5),3)))
r=pd.DataFrame(out); r.to_csv('data/bench_paired.csv',index=False); print(r.to_string())
