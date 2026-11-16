"""Very small deterministic factorial smoke; full optimization delegated to Codex."""
import csv,time
from pathlib import Path
import numpy as np
from joint_core import derivative_filters,exchange_filter,run,observables,shape,axis_mae
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'smoke_results';OUT.mkdir(exist_ok=True)
rows=[]
for dim in (2,3):
 for kind in (('stadium','ellipse') if dim==2 else ('capsule','ellipsoid')):
  for angle in (13.,34.,66.):
   f,axis=shape(dim,kind,angle_deg=angle,size=(56 if dim==2 else 28),radius=(6 if dim==2 else 4))
   for exch in ('axial','tied','free'):
    for diff in ('central','fourth'):
     gp=(0.,0.,0.) if diff=='central' else (-1/12,0.,0.)
     gd=derivative_filters(dim,*gp)
     kj=exchange_filter(dim,gd,kind=exch)
     for N in (5,10):
      start=time.perf_counter();n=run(f,N,gd,kj);seconds=time.perf_counter()-start
      bond,direct,G=observables(n,kj,gd)
      for obs,T,select in [('gradient',G,'min'),('bond_min',bond,'min'),('bond_max',bond,'max'),('direct_min',direct,'min'),('direct_max',direct,'max')]:
       err,gap=axis_mae(T,axis,select)
       seg=n[...,-1]>0;truth=f>0;dice=2*(seg&truth).sum()/max(seg.sum()+truth.sum(),1)
       rows.append(dict(dim=dim,kind=kind,angle_deg=angle,exchange=exch,derivative=diff,N=N,observer=obs,MAE_deg=err,eigengap=gap,dice=dice,seconds=seconds,unit_error=np.max(abs(np.linalg.norm(n,axis=-1)-1))))
with open(OUT/'smoke_raw.csv','w',newline='') as out:
 w=csv.DictWriter(out,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
from collections import defaultdict
agg=defaultdict(list)
for r in rows:agg[(r['dim'],r['exchange'],r['observer'],r['N'])].append(r['MAE_deg'])
with open(OUT/'smoke_summary.csv','w',newline='') as out:
 w=csv.writer(out);w.writerow(['dim','exchange','observer','N','MAE_deg','n'])
 for k,v in sorted(agg.items()):w.writerow([*k,np.mean(v),len(v)])
print('PASS smoke:',len(rows),'measurements; outputs:',OUT)
print('NOTE: small synthetic fixtures, NO training/selection and NO held-out test.')
