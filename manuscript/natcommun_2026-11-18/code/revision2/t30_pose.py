"""Rasterization floor, part 1: pose ensembles of fixed test geometries (binary input).
For each test geometry: K random sub-pixel poses (rotation + shift); record estimator outputs.
Pose spread is error caused only by where the lattice falls (truth fixed)."""
import sys,json,numpy as np,pandas as pd,scipy.ndimage as ndi
from multiprocessing import Pool
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from gray import params2d,render,kappa2_integral
from t22_select import gen,split_shapes
import t23_aniso as A
from bench_run import ms_len
from skimage.measure import perimeter_crofton
A.NH=6; L=json.load(open('data/t25_law_h1.5.json')); C=json.load(open('manuscript_v2/data/revision2/benchmark/bench_constants.json'))['2']['constants']
K=24
def job(a):
    fam,i,seed=a; p=params2d(fam,np.random.default_rng(seed)); rng=np.random.default_rng(seed+777); out=[]
    for k in range(K+1):
        t,sh=(p['t'],p['sh']) if k==0 else (rng.uniform(0,np.pi),rng.uniform(-.5,.5,2))
        m=render(p,t=t,sh=sh)>0.5
        out.append(dict(family=fam,inst=i,pose=k,truth=p['P'],size=p['size'],k2=kappa2_integral(p),
            PBGIA=A.estimate(2,gen(2,m,1.0,1.5),0,L['c'],L['b']),
            gauss=ms_len(ndi.gaussian_filter(m.astype(float),0.75))*C['gauss_cal'],
            crofton=perimeter_crofton(m,4)*C['crofton_cal']))
    return out
if __name__=='__main__':
    jobs=[j for j in split_shapes(2,'test') if j[0] !='near_touch']
    with Pool(2) as pool: res=pool.map(job,jobs,chunksize=3)
    pd.DataFrame([r for rr in res for r in rr]).to_csv('data/t30_pose.csv',index=False)
