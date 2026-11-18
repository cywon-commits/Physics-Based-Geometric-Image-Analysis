import numpy as np, csv, itertools, math
from common import generate3, tv_per, pgrad
# (a) planar walls with primitive normals in fundamental wedge p>=q>=r>=0
L=60; idx=np.indices((L,L,L)).astype(np.float64)+0.5
cand=[(p,q,r) for p in range(0,8) for q in range(0,p+1) for r in range(0,q+1) if p>0 and math.gcd(math.gcd(p,q),r)==1 and p*p+q*q+r*r<=50]
rows=[]
for p,q,r in cand:
    mask=((p*idx[0]+q*idx[1]+r*idx[2])%L)<L/2
    v,s=generate3(mask,30)
    area=2*L*L*math.sqrt(p*p+q*q+r*r)
    nv=np.array([p,q,r])/math.sqrt(p*p+q*q+r*r)
    rows.append(dict(p=p,q=q,r=r,nz=nv[0],TV_per_area=tv_per(s)/area))
    print(p,q,r,round(rows[-1]['TV_per_area'],4),flush=True)
with open('data/t5_planes3d.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# (b) spheres
rng=np.random.default_rng(3); srows=[]
for R in [2,3,4,6,8,10,13,16,20]:
    n=int(2*R+20)
    for rep in range(4):
        c=n/2+rng.uniform(-.5,.5,3)
        rr=np.sqrt(sum((np.indices((n,n,n))[a]-c[a])**2 for a in range(3))); mask=rr<=R
        v,s=generate3(mask,30)
        g=pgrad(s); w=0.5*np.sqrt(sum(x*x for x in g)); TV=w.sum()
        mu=(w*rr).sum()/TV; m2=(w*rr*rr).sum()/TV
        srows.append(dict(R=R,rep=rep,A_true=4*np.pi*R*R,TV=TV,four_pi_r2w=4*np.pi*m2,mean_u=mu-R,mean_u2=m2-2*R*mu+R*R,
                          dice=2*np.sum((s>0)&mask)/(np.sum(s>0)+mask.sum())))
    q=[x for x in srows if x['R']==R]
    print('sphere',R,np.mean([x['TV']/x['A_true'] for x in q]),np.mean([x['TV']/x['four_pi_r2w'] for x in q]),np.mean([x['mean_u'] for x in q]),np.mean([x['mean_u2'] for x in q]),np.mean([x['dice'] for x in q]),flush=True)
with open('data/t5_spheres.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(srows[0]));w.writeheader();w.writerows(srows)
