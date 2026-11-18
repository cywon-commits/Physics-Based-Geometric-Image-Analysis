"""Paired, family-stratified bootstrap of MARE differences on the gray test set (non-contact families)."""
import numpy as np,pandas as pd
import sys
SRC=sys.argv[1] if len(sys.argv)>1 else 'data/t36_gray_test_v2.csv'
d=pd.read_csv(SRC); d=d[d.family!='near_touch']
M=['PBGIA_gray','PBGIA_bin','ms_gray','gms05_gray','gms10_gray']
for m in M: d['e_'+m]=100*(d[m]/d.truth-1).abs()
rng=np.random.default_rng(1); rows=[]
for (k,n),g in d.groupby(['kind','noise']):
    fams=[x for _,x in g.groupby('family')]
    def stat(a,b,idx=None):
        return np.mean([ (f['e_'+a].values-f['e_'+b].values)[ix].mean() for f,ix in zip(fams,idx)])
    for b in ['PBGIA_bin','ms_gray','gms05_gray','gms10_gray']:
        pt=stat('PBGIA_gray',b,[np.arange(len(f)) for f in fams])
        bs=[stat('PBGIA_gray',b,[rng.integers(0,len(f),len(f)) for f in fams]) for _ in range(2000)]
        rows.append(dict(kind=k,noise=n,vs=b,diff=pt,lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5)))
r=pd.DataFrame(rows); r.to_csv(SRC.replace('.csv','_paired.csv'),index=False); print(r.round(3).to_string())
