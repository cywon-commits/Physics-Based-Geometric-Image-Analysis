import sys,glob,zlib,numpy as np,pandas as pd,scipy.ndimage as ndi
from multiprocessing import Pool
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code/revision2','/home/claude/work/manuscript_v2/code']
import t60_cells as T, t60b_degree as B
from real_image_examples import generate_field
from gray import gen_gray
import new_quantitative_studies as nq
from skimage.filters import threshold_otsu
def job(f):
    I,L,ngt,merged=T.base(f); lo,hi,t=T.levels(I); P=8
    m=np.pad(I>threshold_otsu(I),P,constant_values=False)
    vr,vc,s=[z.astype(np.float64) for z in generate_field(m,30,h=1.5)]; Q=nq.q2_solid_angle(vr,vc,s)
    g=np.clip((np.pad(I,P,constant_values=lo)-lo)/(hi-lo),0,1); vr2,vc2,s2=gen_gray(2*g-1,h=1.5,full=True); Q2=nq.q2_solid_angle(vr2,vc2,s2)
    unsat=np.mean(np.abs(s2)<0.5)
    lab,n=ndi.label(s2>0); _,nh=ndi.label(np.pad(s2<=0,1,constant_values=True),structure=np.ones((3,3)))
    return dict(img=f.split('/')[-1][:8],Qb=Q,chib4=B.chi_conn(s>0,False),chib8=B.chi_conn(s>0,True),Qc=Q2,chic4=B.chi_conn(s2>0,False),chic8=B.chi_conn(s2>0,True),unsat=unsat,
                frac_mid=np.mean((g>0.3)&(g<0.7)))
if __name__=='__main__':
    te=sorted(glob.glob(T.ROOT+'/test/images/*.tif'))
    with Pool(2) as pool: R=pd.DataFrame(pool.map(job,te))
    R.to_csv('data/t60c_degree_bin.csv',index=False)
    R['Qb']=R.Qb.round(); R['Qc']=R.Qc.round()
    R['okb']=(R.Qb==R.chib4)|(R.Qb==R.chib8); R['okc']=(R.Qc==R.chic4)|(R.Qc==R.chic8)
    R['inb']=(R.Qb>=R[['chib4','chib8']].min(axis=1))&(R.Qb<=R[['chib4','chib8']].max(axis=1))
    print('binary: Q equals chi4 or chi8:',R.okb.mean(),' within:',R.inb.mean(),' chi4==chi8:',(R.chib4==R.chib8).mean())
    print('cont: Q equals chi4 or chi8:',R.okc.mean())
    print(R[~R.okb][['img','Qb','chib4','chib8']].to_string())
    print(R[['unsat','frac_mid']].corrwith((R.Qc-R.chic8).abs()))
