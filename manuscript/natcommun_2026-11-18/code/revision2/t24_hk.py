"""3D metrication law in both curvature invariants: m = m3 + b1 c + b2 c^2 + b3 c^3 + bK K,  c=|2H| (measured), K Gaussian (measured).
Calibration shapes: spheres (K=H^2) and periodic cylinders (K=0) only."""
import sys,json,numpy as np,pandas as pd,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code','/home/claude/work/manuscript_v2/code/revision2']
from t22_select import gen,split_shapes
from kern import tv
from bench_shapes import make3d
M3=1.0160235364993773
def curv3(s,sigma=1.0):
    ss=ndi.gaussian_filter(s,sigma,mode='wrap'); d=lambda a,i:(np.roll(a,-1,i)-np.roll(a,1,i))*0.5
    g=[d(ss,i) for i in range(3)]; gn=np.sqrt(sum(x*x for x in g))+1e-12; n=[x/gn for x in g]
    Hs=[[d(g[i],j) for j in range(3)] for i in range(3)]
    # shape operator S = -(I - n n^T) Hess (I - n n^T) / |grad|
    P=[[(1.0 if i==j else 0.0)-n[i]*n[j] for j in range(3)] for i in range(3)]
    PH=[[sum(P[i][k]*Hs[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    S=[[-sum(PH[i][k]*P[k][j] for k in range(3))/gn for j in range(3)] for i in range(3)]
    tr=S[0][0]+S[1][1]+S[2][2]                     # = kappa1+kappa2 = 2H
    tr2=sum(S[i][j]*S[j][i] for i in range(3) for j in range(3))
    K=0.5*(tr*tr-tr2)                              # = kappa1*kappa2
    return np.abs(tr),K
def cyl_sweep(rng):
    out=[]; L=48
    for r in [2.5,3,3.5,4,5,6,7,8,10]:
        for dvec in ([1,0,0],[1,1,0],[1,1,1]):
            dv=np.array(dvec,float); dv/=np.linalg.norm(dv); c=L/2+rng.uniform(-.5,.5,3)
            X=np.indices((L,)*3).astype(float)-c.reshape(3,1,1,1)
            best=np.full((L,)*3,1e9)
            for sh in [(a,b,e) for a in (-L,0,L) for b in (-L,0,L) for e in (-L,0,L)]:
                Y=X+np.array(sh,float).reshape(3,1,1,1); t=np.einsum('i,i...->...',dv,Y); perp=Y-dv.reshape(3,1,1,1)*t
                best=np.minimum(best,np.sqrt((perp**2).sum(0)))
            m=best<=r; A=2*np.pi*r*L*np.linalg.norm(dvec)
            s=gen(3,m,1.0,2.0); out.append(dict(kind='cyl',s=s,truth=A,c=1/r,K=0.0))
    return out
def sph_sweep(rng):
    out=[]
    for R in [2.5,3,3.5,4,5,6,7,8,10,13,16,20]:
        for rep in range(3):
            n=int(2*R+24); c=n/2+rng.uniform(-.5,.5,3); r=np.sqrt(sum((np.indices((n,)*3)[a]-c[a])**2 for a in range(3)))
            out.append(dict(kind='sph',s=gen(3,r<=R,1.0,2.0),truth=4*np.pi*R*R,c=2/R,K=1/R**2))
    return out
def design(e,meas,kmax=0.85):
    w=tv(e['s'],0)
    if meas:
        c,K=curv3(e['s']); c=np.clip(c,0,kmax); K=np.clip(K,-0.2,0.2); cw=np.sum(w*c)/w.sum(); Kw=np.sum(w*K)/w.sum()
    else: cw,Kw=e['c'],e['K']
    return [cw,cw**2,cw**3,Kw], w.sum()/e['truth']-M3
if __name__=='__main__':
    rng=np.random.default_rng(5); cal=sph_sweep(rng)+cyl_sweep(rng)
    for meas in (False,True):
        X=[];Y=[]
        for e in cal: x,y=design(e,meas); X.append(x); Y.append(y)
        X=np.array(X);Y=np.array(Y)
        for useK in (False,True):
            A=X if useK else X[:,:3]; b=np.linalg.lstsq(A,Y,rcond=None)[0]; resid=np.abs(A@b-Y)
            kinds=np.array([e['kind'] for e in cal])
            print('meas',meas,'useK',useK,'coef',np.round(b,4),'resid sph',resid[kinds=='sph'].mean().round(4),'cyl',resid[kinds=='cyl'].mean().round(4))
            if meas:
                errs={}
                for fam,i,seed in split_shapes(3,'train'):
                    m,T,size=make3d(fam,np.random.default_rng(seed)); s=gen(3,m,1.0,2.0); w=tv(s,0); c,K=curv3(s); c=np.clip(c,0,0.85); K=np.clip(K,-0.2,0.2)
                    mm=M3+b[0]*c+b[1]*c*c+b[2]*c**3+(b[3]*K if useK else 0)
                    errs.setdefault(fam,[]).append(100*(np.sum(w/mm)/T-1))
                nc=np.mean([np.mean(np.abs(v)) for f,v in errs.items() if f!='near_touch'])
                print('   train',round(nc,3),{f:round(np.mean(np.abs(v)),2) for f,v in errs.items()})
                json.dump(dict(m3=M3,coef=b.tolist(),useK=useK),open(f'data/t24_law_K{int(useK)}.json','w'))
