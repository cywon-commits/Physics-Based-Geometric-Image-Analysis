"""Train-split selection under noise for the grayscale source (beta=0), and calibration of gray comparators."""
import sys,json,numpy as np,pandas as pd,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from t32_gray import calib,est,S
from gray import params2d,render
from t22_select import split_shapes
from bench_run import ms_len
from multiprocessing import Pool
SIG=[0.5,0.75,1.0,1.5]
def noisy(g,rng,p,kind):
    if p==0: return g
    if kind=='flip': fl=rng.random(g.shape)<p; return np.where(fl,1-g,g)
    return g+rng.normal(0,p,g.shape)            # additive gaussian, std p (gray units)
def _cal(h): return calib(h,0.0)
if __name__=='__main__':
    H=[1.5,2.0,2.5,3.0]
    with Pool(2) as pool: cals=pool.map(_cal,H)
    json.dump(cals,open('data/t33_gray_consts.json','w'),indent=1)
    rows=[]
    for fam,i,seed in split_shapes(2,'train'):
        rng=np.random.default_rng(seed); p=params2d(fam,rng); g0=render(p,S=S)
        for kind,p_ in (('flip',0.0),('flip',0.01),('gauss',0.05),('gauss',0.1)):
            g=noisy(g0,rng,p_,kind); r=dict(family=fam,inst=i,kind=kind,noise=p_,truth=p['P'],ms_raw=ms_len(g))
            for cal in cals: r[f"P_h{cal['h']}"]=est(np.clip(g,0,1) if kind=='gauss' else g,cal)
            for s in SIG: r[f'gms_{s}']=ms_len(ndi.gaussian_filter(g,s))
            rows.append(r)
    df=pd.DataFrame(rows); df.to_csv('data/t33_gray_train.csv',index=False)
    cl=df[df.noise==0]; cols=[c for c in df.columns if c.startswith(('P_','gms_','ms_'))]
    cst={c:(1.0 if c.startswith('P_') else float(np.median(cl.truth/cl[c]))) for c in cols}
    nc=df.family!='near_touch'
    for c in cols:
        e=100*(df[c]*cst[c]/df.truth-1).abs(); print(c,round(cst[c],4),df[nc].assign(e=e[nc]).groupby(['kind','noise']).e.mean().round(3).to_dict())
    json.dump(cst,open('data/t33_gray_calconst.json','w'),indent=1)
