"""3D: constants re-derived at each N, anchored to the published N=30 constants:
m_bar(N)=MBAR*mb(N)/mb(30), coef(N)=COEF+cf(N)-cf(30), with mb, cf from t55 (planar walls, spheres)."""
import sys,json,numpy as np,pandas as pd
from multiprocessing import Pool
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code/revision2','/home/claude/work/manuscript_v2/code']
from t22_select import gen,split_shapes
from bench_shapes import make3d
from readout import tv_density,curvature,KMAX,MBAR,COEF
C=[c for c in json.load(open('data/t55_consts.json')) if c['dim']==3]; C={c['N']:c for c in C}
def job(N):
    mb=MBAR[3]*C[N]['mb']/C[30]['mb']; cf=np.array(COEF[3])+np.array(C[N]['cf'])-np.array(C[30]['cf']); rows=[]
    for fam,i,seed in split_shapes(3,'test'):
        rng=np.random.default_rng(seed); m0,T,size=make3d(fam,rng)
        for p in (0.0,0.002,0.01):
            m=m0^(rng.random(m0.shape)<p) if p>0 else m0; s=gen(3,m,1.0,2.0,N)
            k=np.clip(np.abs(curvature(s)),0,KMAX[3]); est=float(np.sum(tv_density(s)/(mb+cf[0]*k+cf[1]*k*k+cf[2]*k**3)))
            rows.append(dict(N=N,family=fam,inst=i,noise=p,truth=T,anchored=est))
    return rows
if __name__=='__main__':
    with Pool(2) as pool: r=pool.map(job,sorted(C)[::-1])
    df=pd.DataFrame([x for rr in r for x in rr]); df.to_csv('data/t55b_nsteps3d.csv',index=False)
    df['e']=100*(df.anchored/df.truth-1).abs(); nc=df[df.family!='near_touch']
    print(nc.groupby(['noise','N','family']).e.mean().groupby(['noise','N']).mean().round(3).to_string())
