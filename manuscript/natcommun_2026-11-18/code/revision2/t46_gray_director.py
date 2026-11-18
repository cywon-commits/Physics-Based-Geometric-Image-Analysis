"""Gray textures (the nine of Table multithresh): threshold stack vs continuous source vs structure tensor.
(i) rotation consistency of the global axis (7 rotations); (ii) synthetic strata: error to the analytic tangent vs sigma."""
import sys,numpy as np,pandas as pd
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from t45_color_director import tens_cuts,tens_cont,st,director
from real_image_examples import axis_mean,angle_delta,img_rotate,roi_center,synthetic_strata
from multithreshold_directors import construct_cases
from multiprocessing import Pool
def M(a): return {'cuts5':tens_cuts(a)[0],'cont':tens_cont(a)[0],'ST':st([a])}
def rot(item):
    name,img=item; res={}
    for r in (-15,-10,-5,0,5,10,15):
        a=img_rotate(img,r); roi=roi_center(a.shape,.67)
        for k,T in M(a).items(): tang,gap,_=director(T); res.setdefault(k,[]).append((r,axis_mean(tang,gap,roi)[0]))
    out=[]
    for k,v in res.items():
        a0=[x for x in v if x[0]==0][0][1]; out.append(dict(image=name,method=k,rot_err=float(np.mean([abs(angle_delta(angle_delta(a_,a0),r)) for r,a_ in v]))))
    return out
if __name__=='__main__':
    cases=[(n,img) for n,img,_,_ in construct_cases()]
    with Pool(2) as pool: rr=[r for x in pool.map(rot,cases) for r in x]
    e=pd.DataFrame(rr); e.to_csv('data/t46_gray_director_rot.csv',index=False); print(e.pivot(index='image',columns='method',values='rot_err').round(2).to_string())
    img,truth=synthetic_strata(size=320,amplitude=22,period=190.,stripe=17,seed=19); T=M(img); y,x=np.mgrid[:320,:320]; band=(x>24)&(x<296)&(y>24)&(y<296); rows=[]
    for s in (2,4,8,16):
        for k,t in T.items():
            tang,gap,_=director(t,s); w=gap*band; rows.append(dict(sigma=s,method=k,err=float(np.sum(w*np.abs(angle_delta(tang,truth)))/np.sum(w))))
    f=pd.DataFrame(rows); f.to_csv('data/t46_strata_sigma.csv',index=False); print(f.pivot(index='sigma',columns='method',values='err').round(2).to_string())
