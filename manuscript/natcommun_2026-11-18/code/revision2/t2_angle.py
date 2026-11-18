import numpy as np, csv
from common import generate_field, tv_per as tv2
from common import ie_per
ie2=lambda a,b,c: ie_per((a,b,c))
L=120; rows=[]
pairs=[(0,1),(1,8),(1,5),(1,4),(1,3),(2,5),(1,2),(3,5),(2,3),(3,4),(4,5),(1,1)]
y,x=np.mgrid[:L,:L]+0.5
for p,q in pairs:
    phi=np.degrees(np.arctan2(p,q)); Ltot=2*L*np.hypot(p,q)
    mask=((-p*x+q*y)%L)<L/2
    for N in (0,30,200):
        for D in (0.0,1.0):
            vr,vc,s=generate_field(mask,nsteps=N,D=D)
            rows.append(dict(p=p,q=q,phi_deg=phi,N=N,D=D,TV_per_len=tv2(s)/Ltot,IE_per_len=ie2(vr,vc,s)/Ltot))
    print(p,q,round(phi,2),[round(r['TV_per_len'],4) for r in rows[-6:]],flush=True)
with open('data/t2_angle.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
