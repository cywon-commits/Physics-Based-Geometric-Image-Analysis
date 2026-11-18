"""Cell nuclei (BBBC038 / DSB2018, StarDist subset; CC0): object counts against expert annotation.
Comparators are tuned on the 447 training images; PBGIA uses the fixed settings of the paper (h/J=1.5, N=30)."""
import zlib,sys,glob,json,itertools,numpy as np,scipy.ndimage as ndi,pandas as pd,tifffile
from multiprocessing import Pool
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code/revision2','/home/claude/work/manuscript_v2/code']
from gray import gen_gray
from real_image_examples import generate_field
from skimage.filters import threshold_otsu
import new_quantitative_studies as nq
ROOT='/home/claude/celldata/raw/dsb2018'
SIGS=[0,0.5,1,1.5,2,3]; AMINS=[0,5,10,15,20,30,40,60,80]
def load(f):
    I=tifffile.imread(f).astype(float); L=tifffile.imread(f.replace('images','masks'))
    return (I.mean(-1) if I.ndim==3 else I),L
def levels(I):
    sm=ndi.gaussian_filter(I,1.0); t=threshold_otsu(sm); b=I[sm<=t]; lo=np.median(b); hi=np.median(I[sm>t]); return lo,hi,t
def objs(mask,L,amin=0):
    lab,n=ndi.label(mask)
    if amin>0 and n:
        sz=ndi.sum(np.ones_like(lab),lab,np.arange(1,n+1)); keep=np.r_[False,sz>=amin]; lab,n=ndi.label(keep[lab])
    gt=L>0; ids=np.arange(1,n+1)
    hit=ndi.maximum(gt,lab,ids) if n else np.array([])      # object overlaps some nucleus
    fp=int(np.sum(np.asarray(hit)==0)) if n else 0
    gl=np.unique(L)[1:]; cov=ndi.maximum(mask_of(lab),L,gl) if len(gl) else np.array([]); miss=int(np.sum(np.asarray(cov)==0))
    return n,fp,miss
def mask_of(lab): return (lab>0).astype(np.uint8)
def base(f):
    I,L=load(f); ngt=len(np.unique(L))-1; merged=ndi.label(ndi.binary_dilation(L>0,iterations=1))[1]
    return I,L,ngt,merged
def job_train(f):
    I,L,ngt,merged=base(f); out=[]
    for sg in SIGS:
        J=ndi.gaussian_filter(I,sg) if sg>0 else I; m=J>threshold_otsu(J)
        for am in AMINS: n,fp,miss=objs(m,L,am); out.append(dict(img=f,sig=sg,amin=am,ngt=ngt,n=n))
    return out
PAD=8
def chi(m):   # Euler characteristic, 8-connected foreground / 4-connected background (consistent pair)
    lab,nl=ndi.label(m,structure=np.ones((3,3))); _,nh=ndi.label(np.pad(~m,1,constant_values=True)); return nl-(nh-1)
def job_test(a):
    f,cfg,noise=a; I0,L,ngt,merged=base(f); lo0,hi0,_=levels(I0)
    I=I0+np.random.default_rng(zlib.crc32(f.encode())).normal(0,noise*(hi0-lo0),I0.shape) if noise>0 else I0
    lo,hi,t=levels(I); r=dict(img=f.split('/')[-1][:8],noise=noise,ngt=ngt,merged=merged,H=I.shape[0],W=I.shape[1])
    m0=I>threshold_otsu(I); r['otsu_n'],r['otsu_fp'],r['otsu_miss']=objs(m0,L)
    J=ndi.gaussian_filter(I,cfg['sig']) if cfg['sig']>0 else I; m1=J>threshold_otsu(J); r['tuned_n'],r['tuned_fp'],r['tuned_miss']=objs(m1,L,cfg['amin'])
    P=lambda z,v: np.pad(z,PAD,constant_values=v); crop=lambda z: z[PAD:-PAD,PAD:-PAD]
    s=generate_field(P(m0,False),30,h=1.5)[2].astype(float); r['pbb_n'],r['pbb_fp'],r['pbb_miss']=objs(crop(s)>0,L)
    g=np.clip((P(I,lo)-lo)/(hi-lo),0,1); vr2,vc2,s2=gen_gray(2*g-1,h=1.5,full=True); r['pbc_n'],r['pbc_fp'],r['pbc_miss']=objs(crop(s2)>0,L)
    r['pbc_Q']=nq.q2_solid_angle(vr2.astype(np.float32),vc2.astype(np.float32),s2.astype(np.float32)); r['pbc_chi']=chi(s2>0)
    return r
if __name__=='__main__':
    tr=sorted(glob.glob(f'{ROOT}/train/images/*.tif')); te=sorted(glob.glob(f'{ROOT}/test/images/*.tif'))
    import os
    if os.path.exists('data/t60_cells_cfg.json'): cfg=json.load(open('data/t60_cells_cfg.json'))
    else:
        with Pool(2) as pool: T=pd.DataFrame([x for xx in pool.map(job_train,tr) for x in xx])
        T['e']=np.abs(T.n/T.ngt-1); sel=T.groupby(['sig','amin']).e.mean(); best=sel.idxmin(); cfg=dict(sig=float(best[0]),amin=int(best[1])); T.to_csv('data/t60_cells_train.csv',index=False)
        json.dump(cfg,open('data/t60_cells_cfg.json','w'))
    print('train selection',cfg)
    with Pool(2) as pool: R=pd.DataFrame(pool.map(job_test,[(f,cfg,nz) for nz in (0.0,0.1,0.2) for f in te]))
    R.to_csv('data/t60_cells.csv',index=False)
    for nz,Rn in R.groupby('noise'):
        for k in ('otsu','tuned','pbb','pbc'):
            e=np.abs(Rn[k+'_n']/Rn.ngt-1); em=np.abs(Rn[k+'_n']/Rn.merged-1)
            print(f'noise {nz} {k:6s} err vs GT: mean {e.mean():.3f} median {e.median():.3f} | vs merged: mean {em.mean():.3f} median {em.median():.3f} | FP/img {Rn[k+"_fp"].mean():.1f} | missed {Rn[k+"_miss"].sum()/Rn.ngt.sum():.3f}')
    print('degree vs chi max dev', np.abs(R.pbc_Q-R.pbc_chi).max(), 'n images',len(R))
