import sys,math,numpy as np,pandas as pd
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from real_image_examples import generate_field
from common import generate3
from kern import tv
ALS=np.round(np.arange(0,0.301,0.025),3); rows=[]
L=120; y,x=np.mgrid[:L,:L]+0.5
pairs=[(0,1),(1,8),(1,5),(1,4),(1,3),(2,5),(1,2),(3,5),(2,3),(3,4),(4,5),(1,1)]
for D in (0.5,1.0,2.0,3.0):
    for p,q in pairs:
        s=generate_field(((-p*x+q*y)%L)<L/2,30,D=D)[2].astype(float); phi=math.degrees(math.atan2(p,q)); Lt=2*L*math.hypot(p,q)
        for al in ALS: rows.append(dict(dim=2,D=D,phi=phi,alpha=al,m=tv(s,al).sum()/Lt))
L3=60; idx=np.indices((L3,)*3).astype(float)+0.5
cand=[(p,q,r) for p in range(0,8) for q in range(0,p+1) for r in range(0,q+1) if p>0 and math.gcd(math.gcd(p,q),r)==1 and p*p+q*q+r*r<=50]
for D in (1.0,2.0):
    for (p,q,r) in cand:
        s=generate3(((p*idx[0]+q*idx[1]+r*idx[2])%L3)<L3/2,30,D=D)[1].astype(float); A=2*L3*L3*math.sqrt(p*p+q*q+r*r)
        for al in ALS: rows.append(dict(dim=3,D=D,phi=f'{p}{q}{r}',alpha=al,m=tv(s,al).sum()/A))
df=pd.DataFrame(rows); df.to_csv('data/t20_kernel.csv',index=False)
g=df.groupby(['dim','D','alpha']).m.agg(lambda v: v.max()-v.min()).unstack('alpha').round(4); print(g.to_string())
