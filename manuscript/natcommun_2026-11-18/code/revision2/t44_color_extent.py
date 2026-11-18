"""Color perimeters: which scalar (or set of scalars) should drive the field?
Two-phase color image I = c_b + g (c_o - c_b) + iid noise (std sd per channel).
Scenarios: 'lum' = generic colours with luminance contrast; 'iso' = isoluminant (Delta c orthogonal to luminance weights).
Scalars: luminance Y; first principal axis of the pixel colours (PCA); each channel separately, combined with
inverse-variance weights (estimated contrast-to-noise squared; channels with CNR<3 dropped).
Each scalar is read with (i) a single mid-level cut (binary readout) and (ii) the continuous source."""
import sys,numpy as np,pandas as pd,scipy.ndimage as ndi
from multiprocessing import Pool
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from t43_gray_multicut import binP,levels,est_gray,G
from gray import params2d,render,gen_gray
from t32_gray import S
from bench_shapes import FAM2
W=np.array([.299,.587,.114])
def read(Ic):
    lo,hi,sig=levels(Ic); cnr=abs(hi-lo)/max(sig,1e-6)
    if hi<lo: Ic=-Ic; lo,hi=-lo,-hi
    g=np.clip((Ic-lo)/(hi-lo),0,1)
    return dict(cut=binP(Ic>(lo+hi)/2),cont=est_gray(gen_gray(2*g-1,h=G['h']),G['c'],G['b'],G['km']),cnr=cnr)
def colours(rng,kind):
    while True:
        cb=rng.uniform(.25,.75,3); d=rng.normal(size=3)
        if kind=='iso': d-=W*(W@d)/(W@W)
        d*=0.35/np.linalg.norm(d); co=cb+d
        if np.all((co>0.05)&(co<0.95)) and (kind=='iso' or abs(W@d)>0.08): return cb,co
def job(a):
    fam,i,seed=a; rng=np.random.default_rng(seed); p=params2d(fam,rng); g0=render(p,S=S); out=[]
    for kind in ('lum','iso'):
        cb,co=colours(np.random.default_rng(seed+5+(kind=='iso')),kind)
        for sd in (0.0,0.03,0.06):
            I=cb[:,None,None]+g0[None]*(co-cb)[:,None,None]+np.random.default_rng(seed+11).normal(0,sd,(3,)+g0.shape)
            r=dict(family=fam,inst=i,kind=kind,noise=sd,truth=p['P'])
            Y=np.tensordot(W,I,1); rY=read(Y if np.ptp(Y)>1e-6 else Y+1e-9*np.random.default_rng(0).normal(size=Y.shape))
            X=I.reshape(3,-1); X=X-X.mean(1,keepdims=True); u=np.linalg.eigh(X@X.T)[1][:,-1]; rP=read(np.tensordot(u,I,1))
            rc=[read(I[c]) for c in range(3)]; w=np.array([x['cnr']**2 if x['cnr']>=3 else 0 for x in rc])
            for k in ('cut','cont'):
                r[f'Y_{k}']=rY[k]; r[f'PCA_{k}']=rP[k]
                r[f'chan_{k}']=np.sum(w*np.array([x[k] for x in rc]))/w.sum() if w.sum()>0 else np.nan
            r['cnr_Y']=rY['cnr']; r['cnr_PCA']=rP['cnr']; r['cnr_best_chan']=max(x['cnr'] for x in rc)
            out.append(r)
    return out
if __name__=='__main__':
    jobs=[]; seed=40261118
    for fam in FAM2:
        if fam=='near_touch': continue
        for i in range(12): seed+=1; jobs.append((fam,i,seed))
    with Pool(2) as pool: res=pool.map(job,jobs,chunksize=2)
    d=pd.DataFrame([r for rr in res for r in rr]); d.to_csv('data/t44_color_extent.csv',index=False)
    M=[f'{s}_{k}' for s in ('Y','PCA','chan') for k in ('cut','cont')]
    for m in M: d['e_'+m]=100*(d[m]/d.truth-1).abs()
    print(d.groupby(['kind','noise','family'])[['e_'+m for m in M]].mean().groupby(['kind','noise']).mean().round(3).to_string())
    print(d.groupby(['kind','noise'])[['cnr_Y','cnr_PCA','cnr_best_chan']].median().round(1).to_string())
