import sys,csv,numpy as np,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from dirscale import *
import multithreshold_directors as mt
from real_image_examples import img_rotate,roi_center
cases=mt.construct_cases(); names=[c[0] for c in cases]
sigmas=[2,3,4,6,8,12,16]; rows=[]
def rot_q(q,theta):
    """rotate a q-field back by -theta (spatially) and remove the 2theta phase"""
    re=ndi.rotate(q.real,-theta,reshape=False,order=1,mode='reflect'); im=ndi.rotate(q.imag,-theta,reshape=False,order=1,mode='reflect')
    return (re+1j*im)*np.exp(-2j*np.deg2rad(theta))
for idx in range(len(cases)):
    nm,img,_,truth=cases[idx]
    R0={'PBGIA':raw_tensor(img),'ST':st_raw(img)}; rimg=img_rotate(img,10); R1={'PBGIA':raw_tensor(rimg),'ST':st_raw(rimg)}
    roi=roi_center(img.shape,.6)
    for s in sigmas:
        for m in ('PBGIA','ST'):
            q0,_=qfield(R0[m],s); q1,_=qfield(R1[m],s); q1b=rot_q(q1,10)
            w=np.abs(q0)*np.abs(q1b)*roi
            d=np.abs((np.angle(q1b)-np.angle(q0)+np.pi)%(2*np.pi)-np.pi)/2   # axial difference (rad)
            loc=float(np.degrees((w*d).sum()/w.sum()))
            gt=np.nan
            if truth is not None:
                tan=tangent_deg(q0); valid=roi; gt=float(np.mean(np.abs((tan[valid]-truth[valid]+90)%180-90)))
            nd=len(defect_list(q0,margin=24))
            rows.append(dict(image=nm,method=m,sigma=s,local_rot_err=loc,gt_err=gt,n_defects=nd))
    print(nm,[(r['sigma'],r['method'],round(r['local_rot_err'],2),round(r['gt_err'],2) if r['gt_err']==r['gt_err'] else '',r['n_defects']) for r in rows if r['image']==nm and r['sigma'] in (2,4,8,16)],flush=True)
with open('data/t16_real_sigma.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
