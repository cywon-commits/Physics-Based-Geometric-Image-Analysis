"""Coarea-weighted continuous director: by the coarea formula, a stack of K->inf cuts weights each pixel by the density of
level crossings, i.e. by |grad s|. Weight the continuous-source tensor by w=|grad s|^p (p=1: stack limit; p=2: structure-tensor-like).
Evaluated on the t48 suite."""
import sys,csv,numpy as np,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code','/home/claude/work/manuscript_v2/code/revision2']
import t48_hybrid as H
from gray import gen_gray
from multiprocessing import Pool
def raw_w(img,p,smooth=.8,h=2.0):
    sm=ndi.gaussian_filter(img,smooth); lo,hi=np.quantile(sm,(.02,.98)); g=np.clip((sm-lo)/(hi-lo),0,1)
    vr,vc,s=gen_gray(2*g-1,h=h,full=True); gs=np.hypot((np.roll(s,-1,0)-np.roll(s,1,0))/2,(np.roll(s,-1,1)-np.roll(s,1,1))/2)
    w=gs**p; n2=vr*vr+vc*vc+1e-12; return np.stack([w*vr*vr/n2,w*vc*vc/n2,w*vr*vc/n2])
H.RAW.clear(); H.RAW.update({'cont_w1':lambda im:raw_w(im,1),'cont_w2':lambda im:raw_w(im,2)})
if __name__=='__main__':
    M=list(H.RAW)
    with Pool(2) as pool:
        lic=[r for x in pool.map(H.lic_job,M) for r in x]
        real=[r for x in pool.map(H.real_job,[(i,m) for i in range(9) for m in M]) for r in x]
    for rows,fn in ((lic,'t50_lic.csv'),(real,'t50_real.csv')):
        with open('/home/claude/work/data/'+fn,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
