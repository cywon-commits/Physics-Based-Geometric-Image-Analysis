"""Grayscale source: h_i = h * phi_beta(g_i), g = area fraction (partial volume) in [0,1].
phi_0(g)=2g-1 (linear, preserves the zeroth moment of the step), phi_beta = tanh(beta(2g-1))/tanh(beta), phi_inf = sign.
Constants re-derived for each (h,beta) from gray-rendered planar walls (C4v harmonics) and disks (Delta(kappa));
selection on TRAIN only."""
import sys,math,json,numpy as np,pandas as pd,scipy.ndimage as ndi
from multiprocessing import Pool
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from gray import params2d,render,gen_gray
from t22_select import split_shapes
import t23_aniso as A
from kern import tv
from bench_run import ms_len
A.NH=6; S=8
def phi(g,beta):
    x=2*np.asarray(g)-1
    if beta==0: return x
    if beta==np.inf: return np.where(x>0,1.0,-1.0)
    return np.tanh(beta*x)/np.tanh(beta)
def supersample(fun,n,S=S,off=(0.0,0.0)):
    o=(np.arange(S)+0.5)/S-0.5; y0,x0=np.mgrid[:n,:n]+0.0; acc=np.zeros((n,n))
    for oy in o:
        for ox in o: acc+=fun(y0+oy+off[0],x0+ox+off[1])
    return acc/S/S
def planar(h,beta):
    L=240; X=[];Y=[]
    dirs=sorted({(p,q) for p in range(0,13) for q in range(1,13) if math.gcd(p,q)==1 and p<=q})
    for p,q in dirs:
        g=supersample(lambda y,x:((-p*(x+.5)+q*(y+.5))%L)<L/2,L); s=gen_gray(phi(g,beta),h=h); ph=math.atan2(p,q)
        X.append([math.cos(4*j*ph) for j in range(1,A.NH+1)]); Y.append(tv(s,0).sum()/(2*L*math.hypot(p,q)))
    X=np.c_[np.ones(len(X)),np.array(X)]; Y=np.array(Y); c=np.linalg.lstsq(X,Y,rcond=None)[0]; return c.tolist(),float(np.abs(X@c-Y).max())
def disks(h,beta,seed=11):
    rng=np.random.default_rng(seed); out=[]
    for R in [2.5,3,3.5,4,5,6,7,8,10,12,16,20,28,40]:
        for rep in range(6):
            n=int(2*R+24); c=n/2+rng.uniform(-.5,.5,2)
            g=supersample(lambda y,x:np.hypot(y-c[0],x-c[1])<=R,n); out.append(dict(k=1/R,s=gen_gray(phi(g,beta),h=h),truth=2*np.pi*R))
    return out
def calib(h,beta):
    c,err=planar(h,beta); b=A.fit_delta(2,disks(h,beta),0,c); return dict(h=h,beta=beta,c=c,b=b,planar_maxerr=err)
def est(g,cal): return A.estimate(2,gen_gray(phi(g,cal['beta']),h=cal['h']),0,cal['c'],cal['b'])
def _cal(a): return calib(*a)
if __name__=='__main__':
    grid=[(h,b) for h in (1.0,1.5,2.0,3.0) for b in (0.0,1.0,3.0,np.inf)]
    with Pool(2) as pool: cals=pool.map(_cal,grid)
    tr=[]
    for fam,i,seed in split_shapes(2,'train'):
        p=params2d(fam,np.random.default_rng(seed)); tr.append((fam,p['P'],render(p,S=S)))
    rows=[]
    for cal in cals:
        e={}
        for fam,T,g in tr: e.setdefault(fam,[]).append(100*abs(est(g,cal)/T-1))
        rows.append(dict(h=cal['h'],beta=cal['beta'],planar_maxerr=cal['planar_maxerr'],train_MARE=np.mean([np.mean(v) for f,v in e.items() if f!='near_touch']),
                         near_touch=np.mean(e['near_touch']),**{'f_'+f:np.mean(v) for f,v in e.items()}))
        print({k:(round(v,3) if isinstance(v,float) else v) for k,v in rows[-1].items()},flush=True)
    pd.DataFrame(rows).to_csv('data/t32_gray_select.csv',index=False)
    json.dump([{k:(str(v) if k=='beta' else v) for k,v in c.items()} for c in cals],open('data/t32_gray_consts.json','w'),indent=1)
