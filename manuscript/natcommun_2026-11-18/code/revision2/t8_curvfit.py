import numpy as np,csv
from common import generate_field, generate3
from curv import site_curvature,tv_weights
rng=np.random.default_rng(11); rows=[]
for d in (2,3):
    radii=[2.5,3,3.5,4,5,6,7,8,10,12,16,20,28,40] if d==2 else [2.5,3,3.5,4,5,6,7,8,10,13,16,20]
    for R in radii:
        for rep in range(6 if d==2 else 3):
            n=int(2*R+24); c=n/2+rng.uniform(-.5,.5,d)
            r=np.sqrt(sum((np.indices((n,)*d)[a]-c[a])**2 for a in range(d))); mask=r<=R
            s=(generate_field(mask,30)[2] if d==2 else generate3(mask,30)[1]).astype(np.float64)
            w=tv_weights(s); k=site_curvature(s)
            kw=(w*k).sum()/w.sum()
            truth=2*np.pi*R if d==2 else 4*np.pi*R*R
            rows.append(dict(d=d,R=R,rep=rep,m=w.sum()/truth,k_weighted=kw,k_true=(1/R if d==2 else 2/R)))
        q=[x for x in rows if x['d']==d and x['R']==R]; print(d,R,np.mean([x['m'] for x in q]),np.mean([x['k_weighted'] for x in q]),q[0]['k_true'],flush=True)
with open('data/t8_curvfit.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
