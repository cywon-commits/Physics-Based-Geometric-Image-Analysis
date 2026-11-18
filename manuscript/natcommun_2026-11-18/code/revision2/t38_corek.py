"""Curvature evaluated on the wall core (|s|<s0) and carried along the normal to every pixel of that wall
(nearest-core assignment). Removes the spurious curvature at mid-lines between two close walls."""
import sys,math,json,numpy as np,pandas as pd,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
import t23_aniso as A
from t32_gray import phi,supersample,S
from gray import gen_gray,params2d,render
from t22_select import split_shapes
from t33_gray_train import noisy
from kern import tv
A.NH=6; S0=0.5
def feats(s,sk=1.0):
    n,k=A.normals(s,sk); core=np.abs(s)<S0
    if core.any():
        idx=ndi.distance_transform_edt(~core,return_distances=False,return_indices=True); k=k[tuple(idx)]
    return A.feat(2,n),k
def est(s,c,b,km):
    X,k=feats(s); mp=c[0]+X@np.array(c[1:]); kc=np.clip(k,0,km); return float(np.sum(tv(s,0)/(mp+b[0]*kc+b[1]*kc**2+b[2]*kc**3)))
def calib(h,km,gray=True):
    L=240; X=[];Y=[]
    src=(lambda fn,n: phi(supersample(fn,n),0)) if gray else (lambda fn,n: np.where(fn(*(np.mgrid[:n,:n]+0.0)),1.0,-1.0))
    for p,q in sorted({(p,q) for p in range(0,13) for q in range(1,13) if math.gcd(p,q)==1 and p<=q}):
        s=gen_gray(src(lambda y,x:((-p*(x+.5)+q*(y+.5))%L)<L/2,L),h=h); ph=math.atan2(p,q)
        X.append([math.cos(4*j*ph) for j in range(1,A.NH+1)]); Y.append(tv(s,0).sum()/(2*L*math.hypot(p,q)))
    M=np.c_[np.ones(len(X)),np.array(X)]; c=np.linalg.lstsq(M,np.array(Y),rcond=None)[0]
    rng=np.random.default_rng(11); kk=[];rr=[]
    for R in ([1.5,1.75,2,2.25] if gray else [])+[2.5,3,3.5,4,5,6,7,8,10,12,16,20,28,40]:
        for rep in range(6):
            n=int(2*R+24); c0=n/2+rng.uniform(-.5,.5,2); s=gen_gray(src(lambda y,x:np.hypot(y-c0[0],x-c0[1])<=R,n),h=h)
            Xf,_=feats(s); w=tv(s,0); rr.append(w.sum()/(2*np.pi*R)-np.sum(w*(c[0]+Xf@c[1:]))/w.sum()); kk.append(min(1/R,km))
    kk=np.array(kk); b=np.linalg.lstsq(np.c_[kk,kk**2,kk**3],np.array(rr),rcond=None)[0]
    return dict(h=h,km=km,gray=gray,c=c.tolist(),b=b.tolist())
def bands(cal):
    out={}; L=120
    for w in [3,4,5,6,8]:
        e=[]
        for p,q in [(0,1),(1,3),(1,2),(1,1)]:
            n=math.hypot(p,q); fn=lambda y,x:np.abs(((-p*(x+.5)+q*(y+.5))%L)/n-L/n/2)<=w/2
            f=phi(supersample(fn,L),0) if cal['gray'] else np.where(fn(*(np.mgrid[:L,:L]+0.0)),1.0,-1.0)
            e.append(100*(est(gen_gray(f,h=cal['h']),cal['c'],cal['b'],cal['km'])/(2*L*n)-1))
        out[w]=round(float(np.mean(e)),3)
    return out
if __name__=='__main__':
    from multiprocessing import Pool
    grid=[(h,km) for h in (1.5,2.0) for km in (0.45,0.6,0.8)]
    with Pool(2) as pool: cals=pool.starmap(calib,grid)
    tr=[]
    for fam,i,seed in split_shapes(2,'train'):
        rng=np.random.default_rng(seed); p=params2d(fam,rng); g0=render(p,S=S)
        for kind,p_ in (('flip',0.0),('flip',0.01),('gauss',0.05),('gauss',0.1)): tr.append((fam,kind,p_,p['P'],np.clip(noisy(g0,rng,p_,kind),0,1)))
    FF={h:[gen_gray(phi(g,0),h=h) for *_,g in tr] for h in (1.5,2.0)}; rows=[]
    for cal in cals:
        e=pd.DataFrame([dict(family=f,kind=k,noise=n,e=100*abs(est(s,cal['c'],cal['b'],cal['km'])/T-1)) for (f,k,n,T,_),s in zip(tr,FF[cal['h']])])
        fam=e[(e.kind=='flip')&(e.noise==0)].groupby('family').e.mean().round(2).to_dict()
        e=e[e.family!='near_touch'].groupby(['kind','noise','family']).e.mean().groupby(['kind','noise']).mean()
        rows.append(dict(h=cal['h'],km=cal['km'],**{f'{k}{n}':round(v,3) for (k,n),v in e.items()},avg=round(e.mean(),3),bands=bands(cal),clean_fam=fam))
        print(rows[-1],flush=True)
    json.dump(cals,open('data/t38_corek_consts.json','w'),indent=1); pd.DataFrame(rows).to_csv('data/t38_corek_select.csv',index=False)
