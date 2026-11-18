"""Accuracy vs number of synchronous updates N (2D selected readout h=1.5; 3D readout h=2).
Two readouts: constants frozen at N=30 (as published) and constants re-derived at each N from the same
primitives (planar walls + disks/spheres). Test set identical to the benchmark (bench_run / t27)."""
import sys,json,math,numpy as np,pandas as pd,scipy.ndimage as ndi
from multiprocessing import Pool
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code/revision2','/home/claude/work/manuscript_v2/code']
import t23_aniso as A, t22_select as T22
from t22_select import split_shapes
from bench_shapes import make2d,make3d
from readout import boundary,tv_density,curvature,KMAX,MBAR,COEF
L2=json.load(open('data/t25_law_h1.5.json')); A.NH=6
NS2=[0,1,2,3,5,8,12,20,30,50,100,200]; NS3=[0,1,2,3,5,8,12,20,30,60]
_orig=T22.gen
def setN(N):
    def g(dim,m,D,h,NN=None): return _orig(dim,m,D,h,N)
    A.gen=g; T22.gen=g
def fit2(N):
    setN(N); c,_=A.planar_data(2,1.0,1.5,0); sw=T22.sweep(2,1.0,1.5,N); b=A.fit_delta(2,sw,0,c); return c,b
def est3(s,mb,cf):
    k=np.clip(np.abs(curvature(s)),0,KMAX[3]); return float(np.sum(tv_density(s)/(mb+cf[0]*k+cf[1]*k*k+cf[2]*k**3)))
def tv0(s,al=0): return tv_density(s)
def job2(N):
    c,b=fit2(N); rows=[]
    for fam,i,seed in split_shapes(2,'test'):
        rng=np.random.default_rng(seed); m0,T,size=make2d(fam,rng)
        for p in (0.0,0.002,0.01):
            m=m0^(rng.random(m0.shape)<p) if p>0 else m0
            s=_orig(2,m,1.0,1.5,N)
            rows.append(dict(dim=2,N=N,family=fam,inst=i,noise=p,truth=T,frozen=A.estimate(2,s,0,L2['c'],L2['b']),refit=A.estimate(2,s,0,c,b)))
    return rows,dict(dim=2,N=N,c=c,b=b)
def job3(N):
    setN(N)
    # re-derive 3D constants: planar mean (direction-weighted) and cubic curvature law on spheres, isotropic as in readout.py
    import t22_select as TT
    pl=TT.planar(3,1.0,2.0,[0],N); mb=float(pl[0]) if isinstance(pl,dict) else float(pl)
    sw=TT.sweep(3,1.0,2.0,N); kk=[];rr=[]
    for e in sw:
        w=tv_density(e['s']); kc=np.clip(e['kk'],0,KMAX[3])
        kk.append(np.sum(w*kc)/w.sum()); rr.append(w.sum()/e['truth']-mb)
    kk=np.array(kk); cf=np.linalg.lstsq(np.c_[kk,kk**2,kk**3],np.array(rr),rcond=None)[0]
    rows=[]
    for fam,i,seed in split_shapes(3,'test'):
        rng=np.random.default_rng(seed); m0,T,size=make3d(fam,rng)
        for p in (0.0,0.002,0.01):
            m=m0^(rng.random(m0.shape)<p) if p>0 else m0
            s=_orig(3,m,1.0,2.0,N)
            rows.append(dict(dim=3,N=N,family=fam,inst=i,noise=p,truth=T,frozen=boundary(s),refit=est3(s,mb,cf)))
    return rows,dict(dim=3,N=N,mb=mb,cf=cf.tolist())
if __name__=='__main__':
    with Pool(2) as pool:
        r2=pool.map(job2,NS2[::-1]); r3=pool.map(job3,NS3[::-1])
    rows=[r for rr,_ in r2+r3 for r in rr]; df=pd.DataFrame(rows); df.to_csv('data/t55_nsteps.csv',index=False)
    json.dump([c for _,c in r2+r3],open('data/t55_consts.json','w'),default=float,indent=1)
    for col in ('frozen','refit'): df['e_'+col]=100*(df[col]/df.truth-1).abs()
    nc=df[df.family!='near_touch']
    print(nc.groupby(['dim','noise','N','family'])[['e_frozen','e_refit']].mean().groupby(['dim','noise','N']).mean().round(3).to_string())
