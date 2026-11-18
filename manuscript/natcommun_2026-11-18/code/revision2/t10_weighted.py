"""Level-weighted coarea readout: B_g = sum |grad s| g(s), with int_{-1}^{1} g = 1.
g_uniform = 1/2 (plain TV/2), g_wall = (3/4)(1-s^2).  Planar metrication and curvature curve recomputed per weight."""
import sys,math,json,numpy as np,pandas as pd
from common import generate_field, generate3
from curv import site_curvature
G={'uniform':lambda s:0.5+0*s,'wall':lambda s:0.75*(1-s*s)}
def wts(s,g):
    gr=[(np.roll(s,-1,a)-np.roll(s,1,a))*0.5 for a in range(s.ndim)]; return np.sqrt(sum(x*x for x in gr))*G[g](s)
out={}
# planar metrication
L=120; y,x=np.mgrid[:L,:L]+0.5
pairs=[(0,1),(1,8),(1,5),(1,4),(1,3),(2,5),(1,2),(3,5),(2,3),(3,4),(4,5),(1,1)]
pl={g:[] for g in G}
for p,q in pairs:
    s=generate_field(((-p*x+q*y)%L)<L/2,30)[2].astype(np.float64)
    for g in G: pl[g].append((math.degrees(math.atan2(p,q)),wts(s,g).sum()/(2*L*math.hypot(p,q))))
for g in G:
    phi,m=map(np.array,zip(*pl[g])); out[f'm2_{g}']=float(np.trapezoid(m,phi)/45)
L3=60; idx=np.indices((L3,)*3).astype(np.float64)+0.5
cand=[(p,q,r) for p in range(0,8) for q in range(0,p+1) for r in range(0,q+1) if p>0 and math.gcd(math.gcd(p,q),r)==1 and p*p+q*q+r*r<=50]
N=np.array(cand,float);N/=np.linalg.norm(N,axis=1)[:,None]
u=np.random.default_rng(0).normal(size=(400000,3));u=np.sort(np.abs(u/np.linalg.norm(u,axis=1)[:,None]),axis=1)[:,::-1]
vw=np.bincount(np.argmax(u@N.T,axis=1),minlength=len(cand))/len(u)
m3={g:[] for g in G}
for (p,q,r) in cand:
    s=generate3(((p*idx[0]+q*idx[1]+r*idx[2])%L3)<L3/2,30)[1].astype(np.float64)
    for g in G: m3[g].append(wts(s,g).sum()/(2*L3*L3*math.sqrt(p*p+q*q+r*r)))
for g in G: out[f'm3_{g}']=float((vw*np.array(m3[g])).sum())
print(out,flush=True)
# curvature curves from disks/spheres
rng=np.random.default_rng(11); rows=[]
for d in (2,3):
    radii=[2.5,3,3.5,4,5,6,7,8,10,12,16,20,28,40] if d==2 else [2.5,3,3.5,4,5,6,7,8,10,13,16,20]
    for R in radii:
        for rep in range(6 if d==2 else 3):
            n=int(2*R+24); c=n/2+rng.uniform(-.5,.5,d)
            r=np.sqrt(sum((np.indices((n,)*d)[a]-c[a])**2 for a in range(d))); mask=r<=R
            s=(generate_field(mask,30)[2] if d==2 else generate3(mask,30)[1]).astype(np.float64)
            truth=2*np.pi*R if d==2 else 4*np.pi*R*R
            rows.append(dict(d=d,R=R,k=(1/R if d==2 else 2/R),**{g:wts(s,g).sum()/truth for g in G}))
cf=pd.DataFrame(rows); cf.to_csv('data/t10_curvfit.csv',index=False)
coef={}
for d in (2,3):
    q=cf[cf.d==d]; k=q.k.values
    for g in G:
        A=np.c_[k,k*k,k**3]; coef[f'{d}_{g}']=np.linalg.lstsq(A,q[g].values-out[f'm{d}_{g}'],rcond=None)[0].tolist()
out['coef']=coef; json.dump(out,open('data/t10_constants.json','w'),indent=1); print(out)
