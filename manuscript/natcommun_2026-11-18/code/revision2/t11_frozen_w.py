import sys,json,numpy as np,pandas as pd
FZ='/home/claude/frozen/final_pbgia_study'; sys.path[:0]=[FZ+'/code',FZ+'/vendor']
import data as D0
from baseline import initialize,old_step
from curv import site_curvature
from wfun import G,wts
C=json.load(open('data/t10_constants.json'))
def m_of_k(k,d,g):
    b=C['coef'][f'{d}_{g}']; k=np.clip(np.abs(k),0,0.45 if d==2 else 0.85); return C[f'm{d}_{g}']+b[0]*k+b[1]*k*k+b[2]*k**3
man=D0.manifest(); rows=[]
for _,r in man[man.role=='geometry'].iterrows():
    f,truth,axis,aux=D0.fixture(r); n=initialize(f)
    for _ in range(30): n=old_step(n,f)
    s=n[...,-1].astype(np.float64); d=f.ndim; k=site_curvature(s)
    row=dict(case=r.case,unit=r.unit,split=r.split,dim=d,family=r.family,noise=r.noise,truth=truth)
    for g in G:
        w=wts(s,g); row[f'{g}_planar']=w.sum()/C[f'm{d}_{g}']; row[f'{g}_curv']=np.sum(w/m_of_k(k,d,g))
    rows.append(row)
pd.DataFrame(rows).to_csv('frozen_eval/weighted_readouts_all.csv',index=False); print('done')
