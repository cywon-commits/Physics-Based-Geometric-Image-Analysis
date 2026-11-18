"""Degree vs chi of the generated domain at the wall scale: merge gaps <= 2 px (dilate by 1) before counting."""
import sys,glob,numpy as np,pandas as pd,scipy.ndimage as ndi
from multiprocessing import Pool
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code/revision2','/home/claude/work/manuscript_v2/code']
import t60_cells as T, t60b_degree as B
from real_image_examples import generate_field
from gray import gen_gray
from skimage.filters import threshold_otsu
import new_quantitative_studies as nq
def job(f):
    I,L,ngt,merged=T.base(f); lo,hi,t=T.levels(I); P=8; out={'img':f.split('/')[-1][:8]}
    m=np.pad(I>threshold_otsu(I),P,constant_values=False)
    g=np.clip((np.pad(I,P,constant_values=lo)-lo)/(hi-lo),0,1)
    for tag,(vr,vc,s) in (('bin',[z.astype(np.float64) for z in generate_field(m,30,h=1.5)]),('cont',gen_gray(2*g-1,h=1.5,full=True))):
        Q=round(nq.q2_solid_angle(vr,vc,s)); d=s>0
        out[tag+'_Q']=Q; out[tag+'_chi8']=B.chi_conn(d,True); out[tag+'_chi4']=B.chi_conn(d,False)
        dd=ndi.binary_dilation(d,iterations=1); out[tag+'_chiW']=B.chi_conn(dd,True)     # wall-scale domain
        sat=s>0.5; out[tag+'_chiS']=B.chi_conn(ndi.binary_dilation(sat,iterations=1),True)
    return out
if __name__=='__main__':
    te=sorted(glob.glob(T.ROOT+'/test/images/*.tif'))
    with Pool(2) as pool: R=pd.DataFrame(pool.map(job,te))
    R.to_csv('data/t60d_degree_rule.csv',index=False)
    for tag in ('bin','cont'):
        for c in ('chi8','chi4','chiW','chiS'): print(tag,c,'Q==',(R[tag+'_Q']==R[tag+'_'+c]).mean(),'mean|d|',(R[tag+'_Q']-R[tag+'_'+c]).abs().mean().round(2))
