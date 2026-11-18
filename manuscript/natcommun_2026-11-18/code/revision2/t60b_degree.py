"""Degree of the generated field on the cell images vs Euler characteristics of {s>0} under three digital conventions."""
import sys,glob,zlib,numpy as np,pandas as pd,scipy.ndimage as ndi
from multiprocessing import Pool
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code/revision2','/home/claude/work/manuscript_v2/code']
import t60_cells as T
from gray import gen_gray
import new_quantitative_studies as nq
def chi_conn(m,fg8):
    sf=np.ones((3,3)) if fg8 else None; sb=None if fg8 else np.ones((3,3))
    _,nl=ndi.label(m,structure=sf); _,nh=ndi.label(np.pad(~m,1,constant_values=True),structure=sb); return nl-(nh-1)
def chi_bilinear(s,up=4):
    z=ndi.zoom(s,up,order=1); m=z>0; return chi_conn(m,False),chi_conn(m,True)
def job(a):
    f,noise=a; I0,L,ngt,merged=T.base(f); lo0,hi0,_=T.levels(I0)
    I=I0+np.random.default_rng(zlib.crc32(f.encode())).normal(0,noise*(hi0-lo0),I0.shape) if noise>0 else I0
    lo,hi,t=T.levels(I); P=8; g=np.clip((np.pad(I,P,constant_values=lo)-lo)/(hi-lo),0,1)
    vr,vc,s=gen_gray(2*g-1,h=1.5,full=True); Q=nq.q2_solid_angle(vr,vc,s)
    b4,b8=chi_bilinear(s)
    return dict(img=f.split('/')[-1][:8],noise=noise,Q=Q,chi4=chi_conn(s>0,False),chi8=chi_conn(s>0,True),chib4=b4,chib8=b8)
if __name__=='__main__':
    te=sorted(glob.glob(T.ROOT+'/test/images/*.tif'))
    with Pool(2) as pool: R=pd.DataFrame(pool.map(job,[(f,nz) for nz in (0.0,0.1,0.2) for f in te]))
    R.to_csv('data/t60b_degree.csv',index=False)
    R['Qi']=R.Q.round(); print('max |Q-int|',(R.Q-R.Qi).abs().max())
    for c in ('chi4','chi8','chib4','chib8'): print(c,'equal',(R.Qi==R[c]).mean().round(3),' max dev',(R.Qi-R[c]).abs().max())
    print('Q within [chi4,chi8]',((R.Qi>=R[['chi4','chi8']].min(1))&(R.Qi<=R[['chi4','chi8']].max(1))).mean())
