import sys,json,csv,numpy as np
sys.path.insert(0,'/home/claude/repo/Physics-Based-Geometric-Image-Analysis/manuscript/integrated_2026-11-17/code')
import new_quantitative_studies as nq
from common import generate_field
from curv import site_curvature
from wfun import wts
from shapes2d import stadium
C=json.load(open('data/t10_constants.json'))
def est(s,d,curv=True):
    s=s.astype(np.float64); w=wts(s,'uniform')
    if not curv: return w.sum()/C[f'm{d}_uniform']
    b=C['coef'][f'{d}_uniform']; k=np.clip(np.abs(site_curvature(s)),0,0.45 if d==2 else 0.85)
    return np.sum(w/(C[f'm{d}_uniform']+b[0]*k+b[1]*k*k+b[2]*k**3))
rng=np.random.default_rng(7); rows=[]
for R in (11,15,19,23):
    for th in (0,27,54):
        m0=stadium(160,R,th); P=2*np.pi*R+4*R
        for p in (0.0,0.002,0.01):
            for rep in range(3 if p>0 else 1):
                m=m0^(rng.random(m0.shape)<p); s=generate_field(m,30)[2]
                rows.append(dict(kind='stadium',R=R,noise=p,truth=P,planar=est(s,2,False),curv=est(s,2,True)))
for kind in ['sphere','capsule','torus','pair','fibers']:
    mask,sd,area=nq.shape3(68,kind); s=nq.generate3(mask,30)[...,3]
    rows.append(dict(kind=kind,R=0,noise=0,truth=area,planar=est(s,3,False),curv=est(s,3,True)))
for r in rows: r['e_planar']=100*(r['planar']/r['truth']-1); r['e_curv']=100*(r['curv']/r['truth']-1)
with open('data/t12_check.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
import pandas as pd; d=pd.DataFrame(rows)
st=d[d.kind=='stadium']; print(st.assign(a=st.e_planar.abs(),b=st.e_curv.abs()).groupby('noise')[['a','b']].mean().round(3))
print(d[d.kind!='stadium'][['kind','e_planar','e_curv']].round(2)); x=d[d.kind!='stadium']; print('3D MARE planar',x.e_planar.abs().mean().round(2),'curv',x.e_curv.abs().mean().round(2),'4 smooth curv',x[x.kind!='fibers'].e_curv.abs().mean().round(2))
