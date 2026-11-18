"""Readout whose normal and curvature come from the generated in-plane spin itself: n = v/|v|, kappa = -div n.
(no Gaussian smoothing across neighbouring walls). Constants from gray planar walls + disks (R>=1.5); check on straight bands."""
import sys,math,json,numpy as np,pandas as pd
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from t32_gray import phi,supersample,S
from gray import gen_gray,params2d,render
from t22_select import split_shapes
from t33_gray_train import noisy
from kern import tv
NH=6
def vfeat(vr,vc,sv=0.0):
    import scipy.ndimage as ndi
    if sv>0: vr=ndi.gaussian_filter(vr,sv,mode='wrap'); vc=ndi.gaussian_filter(vc,sv,mode='wrap')
    nr=np.hypot(vr,vc)+1e-12; n0=vr/nr; n1=vc/nr
    k=np.abs(-((np.roll(n0,-1,0)-np.roll(n0,1,0))*.5+(np.roll(n1,-1,1)-np.roll(n1,1,1))*.5))
    ph=np.arctan2(n0,n1); return np.stack([np.cos(4*j*ph) for j in range(1,NH+1)],-1),k
def field(g,h): vr,vc,s=gen_gray(phi(g,0),h=h,full=True); return vr,vc,s
def est(F,c,b,km,sv):
    vr,vc,s=F; X,k=vfeat(vr,vc,sv); mp=c[0]+X@np.array(c[1:]); kc=np.clip(k,0,km); return float(np.sum(tv(s,0)/(mp+b[0]*kc+b[1]*kc**2+b[2]*kc**3)))
def calib(h,km,sv):
    L=240; X=[];Y=[]
    for p,q in sorted({(p,q) for p in range(0,13) for q in range(1,13) if math.gcd(p,q)==1 and p<=q}):
        g=supersample(lambda y,x:((-p*(x+.5)+q*(y+.5))%L)<L/2,L); vr,vc,s=field(g,h); ph=math.atan2(p,q)
        X.append([math.cos(4*j*ph) for j in range(1,NH+1)]); Y.append(tv(s,0).sum()/(2*L*math.hypot(p,q)))
    A=np.c_[np.ones(len(X)),np.array(X)]; c=np.linalg.lstsq(A,np.array(Y),rcond=None)[0]
    rng=np.random.default_rng(11); kk=[];rr=[]
    for R in [1.5,1.75,2,2.25,2.5,3,3.5,4,5,6,7,8,10,12,16,20,28,40]:
        for rep in range(6):
            n=int(2*R+24); c0=n/2+rng.uniform(-.5,.5,2); vr,vc,s=field(supersample(lambda y,x:np.hypot(y-c0[0],x-c0[1])<=R,n),h)
            Xf,_=vfeat(vr,vc,sv); w=tv(s,0); mp=c[0]+Xf@c[1:]; rr.append(w.sum()/(2*np.pi*R)-np.sum(w*mp)/w.sum()); kk.append(min(1/R,km))
    kk=np.array(kk); b=np.linalg.lstsq(np.c_[kk,kk**2,kk**3],np.array(rr),rcond=None)[0]
    return dict(h=h,km=km,sv=sv,c=c.tolist(),b=b.tolist())
def bands(cal):
    out={}; L=120
    for w in [3,4,5,6,8]:
        e=[]
        for p,q in [(0,1),(1,3),(1,2),(1,1)]:
            n=math.hypot(p,q); g=supersample(lambda y,x:np.abs(((-p*(x+.5)+q*(y+.5))%L)/n-L/n/2)<=w/2,L)
            e.append(100*(est(field(g,cal['h']),cal['c'],cal['b'],cal['km'],cal['sv'])/(2*L*n)-1))
        out[w]=round(float(np.mean(e)),3)
    return out
if __name__=='__main__':
    from multiprocessing import Pool
    grid=[(h,km,sv) for h in (1.5,2.0) for km in (0.45,0.8) for sv in (0.0,0.5)]
    with Pool(2) as pool: cals=pool.starmap(calib,grid)
    tr=[]
    for fam,i,seed in split_shapes(2,'train'):
        rng=np.random.default_rng(seed); p=params2d(fam,rng); g0=render(p,S=S)
        for kind,p_ in (('flip',0.0),('flip',0.01),('gauss',0.05),('gauss',0.1)): tr.append((fam,kind,p_,p['P'],np.clip(noisy(g0,rng,p_,kind),0,1)))
    FF={h:[field(g,h) for *_,g in tr] for h in (1.5,2.0)}; rows=[]
    for cal in cals:
        e=pd.DataFrame([dict(family=f,kind=k,noise=n,e=100*abs(est(F,cal['c'],cal['b'],cal['km'],cal['sv'])/T-1)) for (f,k,n,T,_),F in zip(tr,FF[cal['h']])])
        e=e[e.family!='near_touch'].groupby(['kind','noise','family']).e.mean().groupby(['kind','noise']).mean()
        rows.append(dict(h=cal['h'],km=cal['km'],sv=cal['sv'],**{f'{k}{n}':round(v,3) for (k,n),v in e.items()},avg=round(e.mean(),3),bands=bands(cal)))
        print(rows[-1],flush=True)
    json.dump(cals,open('data/t37_vnormal_consts.json','w'),indent=1); pd.DataFrame(rows).to_csv('data/t37_vnormal_select.csv',index=False)
