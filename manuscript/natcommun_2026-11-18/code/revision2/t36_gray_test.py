"""Single TEST evaluation of the selected gray readout and comparators (same flips as the binary benchmark)."""
import sys,json,numpy as np,pandas as pd,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from t38_corek import est
from t32_gray import phi,S
from gray import params2d,render,gen_gray
from t22_select import split_shapes,gen
from bench_run import ms_len
import t23_aniso as A
from multiprocessing import Pool
G=json.load(open('data/t38_gray_selected.json')); CC=json.load(open('data/t33_gray_calconst.json')); L=json.load(open('data/t25_law_h1.5.json')); A.NH=6
def job(a):
    fam,i,seed=a; rng=np.random.default_rng(seed); p=params2d(fam,rng); g0=render(p,S=S); m0=g0>0.5; out=[]
    for kind,p_ in (('flip',0.0),('flip',0.002),('flip',0.01),('gauss',0.05),('gauss',0.1)):
        if kind=='flip':
            fl=rng.random(g0.shape)<p_ if p_>0 else np.zeros(g0.shape,bool); g=np.where(fl,1-g0,g0); m=m0^fl
        else: g=np.clip(g0+rng.normal(0,p_,g0.shape),0,1); m=g>0.5
        out.append(dict(family=fam,inst=i,kind=kind,noise=p_,truth=p['P'],size=p['size'],
            PBGIA_gray=est(gen_gray(phi(g,0),h=G['h']),G['c'],G['b'],G['km']),
            PBGIA_bin=A.estimate(2,gen(2,m,1.0,1.5),0,L['c'],L['b']),
            ms_gray=ms_len(g)*CC['ms_raw'],gms05_gray=ms_len(ndi.gaussian_filter(g,0.5))*CC['gms_0.5'],gms10_gray=ms_len(ndi.gaussian_filter(g,1.0))*CC['gms_1.0']))
    return out
if __name__=='__main__':
    with Pool(2) as pool: res=pool.map(job,split_shapes(2,'test'),chunksize=3)
    d=pd.DataFrame([r for rr in res for r in rr]); d.to_csv('data/t36_gray_test_v2.csv',index=False)
