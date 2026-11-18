"""Gray perimeters: single cut vs multi-cut level stack vs continuous source, on the fresh test geometries (seeds of t42).
Contrast levels lo/hi are estimated from the image itself (medians of the two Otsu classes); no parameter is tuned here:
binary readout = selected 2D readout (h=1.5, Eq. mphikappa); continuous source = t38 selection."""
import sys,json,numpy as np,pandas as pd,scipy.ndimage as ndi
from multiprocessing import Pool
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from gray import params2d,render,gen_gray
from t38_corek import est as est_gray
from t32_gray import S
from t22_select import gen
from bench_run import ms_len
from bench_shapes import FAM2
from skimage.filters import threshold_otsu
import t23_aniso as A
A.NH=6; L=json.load(open('data/t25_law_h1.5.json')); G=json.load(open('data/t38_gray_selected.json')); CC=json.load(open('data/t33_gray_calconst.json'))
def binP(m): return A.estimate(2,gen(2,m,1.0,1.5),0,L['c'],L['b']) if 0<m.mean()<1 else np.nan
def levels(I):
    sm=ndi.gaussian_filter(I,1.0); t=threshold_otsu(sm); b=I[sm<=t]; lo=np.median(b); hi=np.median(I[sm>t]); sig=1.4826*np.median(np.abs(b-lo)); return lo,hi,sig   # Otsu on a lightly smoothed copy: robust for small objects
def methods(I):
    lo,hi,sig=levels(I); g=np.clip((I-lo)/(hi-lo),0,1); r={}
    r['cut1']=binP(I>lo+0.5*(hi-lo))
    for K in (3,5,9): r[f'cut{K}']=np.nanmean([binP(I>lo+(k+.5)/K*(hi-lo)) for k in range(K)])
    a=min(0.5,3*sig/(hi-lo)); r['cutA5']=np.nanmean([binP(I>lo+(hi-lo)*(a+(1-2*a)*(k+.5)/5)) for k in range(5)])   # levels kept 3 noise std from both plateaus
    sm=ndi.gaussian_filter(I,0.8); r['quant5']=np.nanmean([binP(sm>q) for q in np.quantile(sm,(.15,.325,.5,.675,.85))])
    r['cont']=est_gray(gen_gray(2*g-1,h=G['h']),G['c'],G['b'],G['km']); r['ms']=ms_len(g)*CC['ms_raw']; return r
def job(a):
    fam,i,seed=a; rng=np.random.default_rng(seed); p=params2d(fam,rng); g0=render(p,S=S); rng2=np.random.default_rng(seed+99); out=[]
    for sd in (0.0,0.05,0.1):
        I=0.2+0.6*g0+rng2.normal(0,sd,g0.shape)          # object 0.8, background 0.2: noise std in units of contrast is sd/0.6
        out.append(dict(family=fam,inst=i,noise=sd,truth=p['P'],**methods(I)))
    return out
if __name__=='__main__':
    jobs=[]; seed=30261118
    for fam in FAM2:
        for i in range(30): seed+=1; jobs.append((fam,i,seed))
    with Pool(2) as pool: res=pool.map(job,jobs,chunksize=3)
    d=pd.DataFrame([r for rr in res for r in rr]); d.to_csv('data/t43_gray_multicut.csv',index=False)
    M=['cut1','cut3','cut5','cut9','cutA5','quant5','cont','ms']
    for m in M: d['e_'+m]=100*(d[m]/d.truth-1).abs()
    nc=d[d.family!='near_touch']; print(nc.groupby(['noise','family'])[['e_'+m for m in M]].mean().groupby('noise').mean().round(3).to_string())
    print(d[d.noise==0].groupby('family')[['e_'+m for m in M]].mean().round(3).to_string())
