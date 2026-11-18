"""Hybrids of the threshold stack and the continuous source.
(1) 'sat_b': partially saturated source phi_b(g)=tanh(b(2g-1))/tanh b  (an interpolation in the source).
(2) 'filtstack_K': threshold decomposition -> filter every level with the binary generator (sign rule) ->
    recombine g~ = (1/K) sum_k [s_k>0] (layer-cake after filtering, a 'stack filter') -> continuous source h(2g~-1).
(3) 'cont_eq': continuous source on the rank-equalized image (controls for the equalization implicit in (2)).
Evaluated on the texture tests of t47 (LIC disclinations, strata, nine-texture rotation consistency, defect counts)."""
import sys,csv,numpy as np,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code','/home/claude/work/manuscript_v2/code/revision2']
from dirscale import qfield,defect_list,tangent_deg
from gray import gen_gray
from real_image_examples import generate_field,img_rotate,roi_center
import multithreshold_directors as mt
from t47_cont_textures import raw_cont,rot_q
from multiprocessing import Pool
def outer(vr,vc): return np.stack([vr*vr,vc*vc,vr*vc])
def raw_sat(img,b,smooth=.8,h=2.0):
    sm=ndi.gaussian_filter(img,smooth); lo,hi=np.quantile(sm,(.02,.98)); g=np.clip((sm-lo)/(hi-lo),0,1)
    vr,vc,_=gen_gray(np.tanh(b*(2*g-1))/np.tanh(b),h=h,full=True); return outer(vr,vc)
def filtered_levels(img,K,smooth=.8):
    sm=ndi.gaussian_filter(img,smooth); acc=np.zeros_like(sm); n=0
    for q in np.quantile(sm,(np.arange(K)+.5)/K):
        m=sm>q
        if m.mean()<.002 or m.mean()>.998: acc+=m; n+=1; continue
        acc+=generate_field(m,nsteps=30)[2]>0; n+=1
    return acc/n
def raw_filtstack(img,K,h=2.0):
    g=filtered_levels(img,K); vr,vc,_=gen_gray(2*g-1,h=h,full=True); return outer(vr,vc)
def raw_eq(img,K=None,smooth=.8,h=2.0):
    sm=ndi.gaussian_filter(img,smooth); g=(np.argsort(np.argsort(sm.ravel())).reshape(sm.shape)+.5)/sm.size
    vr,vc,_=gen_gray(2*g-1,h=h,full=True); return outer(vr,vc)
from dirscale import raw_tensor
RAW={'cuts5':raw_tensor,'cont':raw_cont,'sat_b1':lambda im:raw_sat(im,1.0),'sat_b3':lambda im:raw_sat(im,3.0),
     'cont_eq':raw_eq,'filtstack_8':lambda im:raw_filtstack(im,8),'filtstack_16':lambda im:raw_filtstack(im,16),'filtstack_32':lambda im:raw_filtstack(im,32)}
def real_job(a):
    idx,m=a; nm,img,_,truth=mt.construct_cases()[idx]; rimg=img_rotate(img,10); roi=roi_center(img.shape,.6); rows=[]
    R0=RAW[m](img); R1=RAW[m](rimg)
    for s in [2,4,8,16]:
        q0,_=qfield(R0,s); q1b=rot_q(qfield(R1,s)[0],10); w=np.abs(q0)*np.abs(q1b)*roi
        d=np.abs((np.angle(q1b)-np.angle(q0)+np.pi)%(2*np.pi)-np.pi)/2; gt=np.nan
        if truth is not None: tan=tangent_deg(q0); gt=float(np.mean(np.abs((tan[roi]-truth[roi]+90)%180-90)))
        rows.append(dict(image=nm,method=m,sigma=s,local_rot_err=float(np.degrees((w*d).sum()/w.sum())),gt_err=gt,n_defects=len(defect_list(q0,margin=24))))
    return rows
def lic_job(m):
    n=320; y,x=np.mgrid[:n,:n]+0.0; truth=[(110,100,+.5),(110,140,-.5),(220,110,+.5),(220,220,-.5)]
    img=np.load('/home/claude/work/manuscript_v2/data/revision2/t15_lic_img.npy'); R=RAW[m](img); rows=[]
    for s in [2,3,4,6,8,10,12,14,16,18]:
        q,_=qfield(R,s); d=defect_list(q,margin=24); used=set(); hit=0
        for r0,c0,k0 in truth:
            for i,(r,c,k) in enumerate(d):
                if i not in used and k==k0 and np.hypot(r-r0,c-c0)<=max(6,s): used.add(i); hit+=1; break
        rows.append(dict(method=m,sigma=s,n=len(d),matched=hit))
    return rows
if __name__=='__main__':
    M=list(RAW)
    with Pool(2) as pool:
        lic=[r for x in pool.map(lic_job,M) for r in x]
        real=[r for x in pool.map(real_job,[(i,m) for i in range(9) for m in M]) for r in x]
    for rows,fn in ((lic,'t48_lic.csv'),(real,'t48_real.csv')):
        with open('/home/claude/work/data/'+fn,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
