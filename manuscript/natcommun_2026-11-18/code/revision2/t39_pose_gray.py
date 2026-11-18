"""Pose ensembles (24 poses) for gray input on the same test geometries as t30, plus a disk-radius sweep for the floor figure."""
import sys,json,numpy as np,pandas as pd,scipy.ndimage as ndi
from multiprocessing import Pool
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from gray import params2d,render,gen_gray,kappa2_integral
from t38_corek import est
from t32_gray import phi,S
from t22_select import gen,split_shapes
from bench_run import ms_len
from skimage.measure import perimeter_crofton
import t23_aniso as A
G=json.load(open('data/t38_gray_selected.json')); CC=json.load(open('data/t33_gray_calconst.json')); L=json.load(open('data/t25_law_h1.5.json')); A.NH=6
C=json.load(open('manuscript_v2/data/revision2/benchmark/bench_constants.json'))['2']['constants']
K=24
def row(p,t,sh):
    g=render(p,S=S,t=t,sh=sh); m=render(p,t=t,sh=sh)>0.5
    return dict(PBGIA_gray=est(gen_gray(phi(g,0),h=2.0),G['c'],G['b'],G['km']),ms_gray=ms_len(g)*CC['ms_raw'],
                PBGIA_bin=A.estimate(2,gen(2,m,1.0,1.5),0,L['c'],L['b']),gauss_bin=ms_len(ndi.gaussian_filter(m.astype(float),0.75))*C['gauss_cal'],
                crofton_bin=perimeter_crofton(m,4)*C['crofton_cal'])
def job(a):
    fam,i,seed=a; p=params2d(fam,np.random.default_rng(seed)); rng=np.random.default_rng(seed+777); out=[]
    for k in range(K+1):
        t,sh=(p['t'],p['sh']) if k==0 else (rng.uniform(0,np.pi),rng.uniform(-.5,.5,2))
        out.append(dict(family=fam,inst=i,pose=k,truth=p['P'],size=p['size'],k2=kappa2_integral(p),**row(p,t,sh)))
    return out
def disk(R):
    rng=np.random.default_rng(int(R*100)); out=[]
    for k in range(48):
        p=dict(fam='disk',R=R,P=2*np.pi*R,t=0.0,sh=np.zeros(2)); out.append(dict(R=R,truth=2*np.pi*R,**row(p,rng.uniform(0,np.pi),rng.uniform(-.5,.5,2))))
    return out
if __name__=='__main__':
    jobs=[j for j in split_shapes(2,'test') if j[0]!='near_touch']
    with Pool(2) as pool:
        res=pool.map(job,jobs,chunksize=3); rd=pool.map(disk,[3,4,5,6,8,10,13,16,20,26,34,45])
    pd.DataFrame([r for rr in res for r in rr]).to_csv('data/t39_pose_all.csv',index=False)
    pd.DataFrame([r for rr in rd for r in rr]).to_csv('data/t39_disk_sweep.csv',index=False)
