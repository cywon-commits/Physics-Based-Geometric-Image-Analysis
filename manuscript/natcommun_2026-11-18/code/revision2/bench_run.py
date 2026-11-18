import sys,numpy as np,pandas as pd,scipy.ndimage as ndi
from multiprocessing import Pool
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code/revision2','/home/claude/work/manuscript_v2/code']
from bench_shapes import *
from readout import boundary
from real_image_examples import generate_field
import new_quantitative_studies as nq
from skimage.measure import perimeter_crofton,find_contours,marching_cubes,mesh_surface_area
SIG2=[0.5,0.75,1.0,1.5,2.0]; SIG3=[0.5,0.75,1.0,1.5]
def ms_len(a,level=0.5):
    L=0.0
    for c in find_contours(np.pad(a,2),level):
        d=np.diff(c,axis=0); L+=np.hypot(d[:,0],d[:,1]).sum()
    return L
def mc_area(a,level=0.5):
    try: v,f,_,_=marching_cubes(np.pad(a,2).astype(np.float32),level); return mesh_surface_area(v,f)
    except Exception: return np.nan
def job(args):
    dim,split,fam,i,seed=args; rng=np.random.default_rng(seed); out=[]
    m0,T,size=(make2d if dim==2 else make3d)(fam,rng)
    for p in ((0.0,0.002,0.01) if split=='test' else (0.0,)):
        m=m0^(rng.random(m0.shape)<p) if p>0 else m0
        r=dict(dim=dim,split=split,family=fam,inst=i,noise=p,truth=T,size=size)
        if dim==2:
            s=generate_field(m,nsteps=30)[2]; r['PBGIA']=boundary(s)
            r['crofton']=perimeter_crofton(m,4); r['ms']=ms_len(m.astype(float))
            for sg in SIG2: r[f'gauss_ms_{sg}']=ms_len(ndi.gaussian_filter(m.astype(float),sg))
            md=ndi.median_filter(m.astype(np.uint8),3).astype(bool); r['median_crofton']=perimeter_crofton(md,4); r['median_ms']=ms_len(md.astype(float))
        else:
            s=nq.generate3(m,30)[...,3]; r['PBGIA']=boundary(s)
            r['mc']=mc_area(m.astype(float))
            for sg in SIG3: r[f'gauss_mc_{sg}']=mc_area(ndi.gaussian_filter(m.astype(float),sg))
            md=ndi.median_filter(m.astype(np.uint8),3).astype(bool); r['median_mc']=mc_area(md.astype(float))
        out.append(r)
    return out
if __name__=='__main__':
    jobs=[]; seed=20261118
    for dim,fams,ntr,nte in ((2,FAM2,8,30),(3,FAM3,3,12)):
        for fam in fams:
            for split,nn in (('train',ntr),('test',nte)):
                for i in range(nn): seed+=1; jobs.append((dim,split,fam,i,seed))
    with Pool(2) as pool: res=pool.map(job,jobs,chunksize=4)
    df=pd.DataFrame([r for rr in res for r in rr]); df.to_csv('data/bench_raw.csv',index=False); print(len(df))
