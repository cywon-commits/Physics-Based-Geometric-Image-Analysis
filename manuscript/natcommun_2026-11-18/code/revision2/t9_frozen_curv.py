import sys,numpy as np,pandas as pd
FZ='/home/claude/frozen/final_pbgia_study'; sys.path[:0]=[FZ+'/code',FZ+'/vendor']
import data as D0
from baseline import initialize,old_step
from curvcorr import boundary_estimate
man=D0.manifest(); rows=[]
for split in ('train','validation','test'):
    for _,r in man[man.split==split].iterrows():
        f,truth,axis,aux=D0.fixture(r); n=initialize(f)
        for _ in range(30): n=old_step(n,f)
        s=n[...,-1]
        rows.append(dict(case=r.case,unit=r.unit,split=split,role=r.role,dim=f.ndim,family=r.family,noise=r.noise,truth=truth,
            TV_refree=boundary_estimate(s,'planar'),TV_curv=boundary_estimate(s,'site')))
    print(split,flush=True)
pd.DataFrame(rows).to_csv('frozen_eval/curv_readouts_all.csv',index=False)
