"""Gray source, beta=0: extend the curvature law to sharper tips (disks down to R=1.5) and re-select the
curvature-measurement scale sigma_k and the clip kappa_max on TRAIN."""
import sys,json,numpy as np,pandas as pd
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
import t23_aniso as A
from t32_gray import phi,supersample,planar,S
from gray import params2d,render,gen_gray
from t22_select import split_shapes
from kern import tv
from multiprocessing import Pool
A.NH=6
def disks(h,seed=11):
    rng=np.random.default_rng(seed); out=[]
    for R in [1.5,1.75,2,2.25,2.5,3,3.5,4,5,6,7,8,10,12,16,20,28,40]:
        for rep in range(6):
            n=int(2*R+24); c=n/2+rng.uniform(-.5,.5,2)
            g=supersample(lambda y,x:np.hypot(y-c[0],x-c[1])<=R,n); out.append(dict(k=1/R,s=gen_gray(phi(g,0),h=h),truth=2*np.pi*R))
    return out
def est(s,c,b,sk,km):
    n,k=A.normals(s,sk); mp=c[0]+A.feat(2,n)@np.array(c[1:]); kc=np.clip(k,0,km); return float(np.sum(tv(s,0)/(mp+b[0]*kc+b[1]*kc**2+b[2]*kc**3)))
def fitb(sw,c,sk,km,meas):
    kk=[];rr=[]
    for e in sw:
        n,k=A.normals(e['s'],sk); w=tv(e['s'],0); mp=c[0]+A.feat(2,n)@np.array(c[1:])
        rr.append(w.sum()/e['truth']-np.sum(w*mp)/w.sum()); kk.append(np.sum(w*np.clip(k,0,km))/w.sum() if meas else min(e['k'],km))
    kk=np.array(kk); return np.linalg.lstsq(np.c_[kk,kk**2,kk**3],np.array(rr),rcond=None)[0].tolist()
def run(h):
    c,_=planar(h,0.0); sw=disks(h); tr=[]
    for fam,i,seed in split_shapes(2,'train'):
        p=params2d(fam,np.random.default_rng(seed)); tr.append((fam,p['P'],gen_gray(phi(render(p,S=S),0),h=h)))
    out=[]
    for sk in (1.0,0.7):
        for km in (0.45,0.6,0.8):
            for meas in (False,True):
                b=fitb(sw,c,sk,km,meas); e={}
                for fam,T,s in tr: e.setdefault(fam,[]).append(100*(est(s,c,b,sk,km)/T-1))
                out.append(dict(h=h,sk=sk,km=km,meas=meas,c=c,b=b,train_MARE=np.mean([np.mean(np.abs(v)) for f,v in e.items() if f!='near_touch']),
                                **{'f_'+f:np.mean(np.abs(v)) for f,v in e.items()}))
    return out
if __name__=='__main__':
    with Pool(2) as pool: res=[r for rr in pool.map(run,[1.5,2.0]) for r in rr]
    df=pd.DataFrame(res); print(df.drop(columns=['c','b']).round(3).to_string()); df.to_json('data/t34_gray_kappa.json',orient='records',indent=1)
