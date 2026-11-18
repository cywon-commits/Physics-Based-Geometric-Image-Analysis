"""Re-score frozen test fixtures (codex/final-pbgia-20261117) with new readouts.
Fixtures and generator are imported unchanged from the frozen package; no selection is altered."""
import sys,json,time,numpy as np,pandas as pd
FZ='/home/claude/frozen/final_pbgia_study'
sys.path[:0]=[FZ+'/code',FZ+'/vendor']
import data as D0
from baseline import initialize,old_step
from skimage.measure import find_contours,marching_cubes,mesh_surface_area
M={2:1.0189,3:1.0160}
GL_t,GL_w=np.polynomial.legendre.leggauss(8)   # nodes on (-1,1)
def tv_c(s):
    g=[(np.roll(s,-1,a)-np.roll(s,1,a))*0.5 for a in range(s.ndim)]; return 0.5*np.sum(np.sqrt(sum(x*x for x in g)))
def levelset_measure(s,t):
    if s.ndim==2:
        L=0.0
        for c in find_contours(s,t):
            d=np.diff(c,axis=0); L+=np.sum(np.hypot(d[:,0],d[:,1]))
        return L
    try:
        v,f,_,_=marching_cubes(s,t); return mesh_surface_area(v,f)
    except Exception: return 0.0
man=D0.manifest(); test=man[(man.split=='test')]
rows=[]
for _,r in test.iterrows():
    f,truth,axis,aux=D0.fixture(r)
    n=initialize(f)
    for _ in range(30): n=old_step(n,f)
    s=n[...,-1].astype(np.float64); s=np.pad(s,2,mode='wrap')  # contours stay closed (objects are padded away from edges)
    d=f.ndim
    cq=0.5*sum(w*levelset_measure(s,t) for t,w in zip(GL_t,GL_w))
    rows.append(dict(case=r.case,unit=r.unit,dim=d,family=r.family,noise=r.noise,truth=truth,
                     TV_refree=tv_c(n[...,-1].astype(np.float64))/M[d],coarea_levelsets=cq,s0_contour=levelset_measure(s,0.0)))
    print(r.case,d,round(rows[-1]['TV_refree']/truth-1,4),round(cq/truth-1,4),flush=True)
pd.DataFrame(rows).to_csv('frozen_eval/new_readouts_test.csv',index=False)
