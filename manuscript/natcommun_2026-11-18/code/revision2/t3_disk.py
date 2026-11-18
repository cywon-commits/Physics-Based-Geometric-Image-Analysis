import numpy as np, csv
from skimage.measure import perimeter_crofton
from common import generate_field, tv_per, pgrad
rng=np.random.default_rng(20261117); rows=[]
for R in [1.5,2,2.5,3,4,5,6,8,10,12,16,20,25,32,40]:
    n=int(2*R+30)
    for rep in range(8):
        c=(n/2+rng.uniform(-.5,.5), n/2+rng.uniform(-.5,.5))
        yy,xx=np.mgrid[:n,:n]; r=np.hypot(yy-c[0],xx-c[1]); mask=r<=R
        for N in (0,30):
            vr,vc,s=generate_field(mask,nsteps=N)
            g=pgrad(s); w=0.5*np.sqrt(g[0]**2+g[1]**2); TV=w.sum()
            rw=(w*r).sum()/max(TV,1e-12)
            area_pos=(s>0).sum()
            rows.append(dict(R=R,rep=rep,N=N,P_true=2*np.pi*R,TV=TV,two_pi_rw=2*np.pi*rw,mean_u=rw-R,
                crofton=perimeter_crofton(mask,4),r_eq_spos=np.sqrt(area_pos/np.pi),
                dice=2*np.sum((s>0)&mask)/(np.sum(s>0)+mask.sum()),mask_px=int(mask.sum())))
    sel=[q for q in rows if q['R']==R and q['N']==30]
    print(R,np.mean([q['TV']/q['P_true'] for q in sel]),np.mean([q['TV']/q['two_pi_rw'] for q in sel]),np.mean([q['mean_u'] for q in sel]),np.mean([q['dice'] for q in sel]),np.mean([q['crofton']/q['P_true'] for q in sel]),flush=True)
with open('data/t3_disk.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
