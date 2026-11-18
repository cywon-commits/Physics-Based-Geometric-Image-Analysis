"""Chance controls for the labyrinth defect matching: smaller matching radii, randomly placed defects with the detected
sign counts, and detected defects with their signs exchanged."""
import sys,numpy as np,pandas as pd
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code/revision2']
from dirscale import qfield,defect_list
import scipy.ndimage as ndi
D=np.load('data/t53_labyrinth.npz'); truth=[tuple(t) for t in D['truth']]; s,vr,vc,gs,raw=D['s'],D['vr'],D['vc'],D['gs'],D['raw']; n=s.shape[0]; M=12
nrm=vr*vr+vc*vc+1e-12
gr,gc=np.gradient(ndi.gaussian_filter(raw,1.0))
RAWS={'PBGIA':np.stack([vr*vr,vc*vc,vr*vc]),'ST':np.stack([gr*gr,gc*gc,gr*gc])}
def match(defs,rad):
    used=set(); tp=0
    for r0,c0,k0 in truth:
        best=None
        for i,(r,c_,k_) in enumerate(defs):
            if i in used or k_!=k0: continue
            d=np.hypot(r-r0,c_-c0)
            if d<=rad and (best is None or d<best[0]): best=(d,i)
        if best: used.add(best[1]); tp+=1
    return tp
rng=np.random.default_rng(0); rows=[]
for m,R in RAWS.items():
    for sig in (1.0,1.5,2.0):
        d=defect_list(qfield(R,sig)[0],margin=M+4); swap=[(r,c,-k) for r,c,k in d]
        for rad in (2.0,3.0,5.3):
            tp=match(d,rad); tps=match(swap,rad)
            rnd=[]
            for _ in range(5):
                rr=rng.uniform(M+4,n-M-4,(len(d),2)); rnd.append(match([(a,b,k) for (a,b),(_,_,k) in zip(rr,d)],rad))
            rows.append(dict(method=m,sigma=sig,radius=rad,n_det=len(d),n_truth=len(truth),precision=tp/len(d),recall=tp/len(truth),
                             precision_sign_swapped=tps/len(d),precision_random=np.mean(rnd)/len(d)))
R=pd.DataFrame(rows); print(R.round(3).to_string()); R.to_csv('data/t54_lab_controls.csv',index=False)
