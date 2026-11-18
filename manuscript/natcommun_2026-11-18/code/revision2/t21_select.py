"""Theory-guided candidate readouts; constants from planar walls + disk/sphere sweeps; selection on benchmark TRAIN only."""
import sys,json,numpy as np,pandas as pd,scipy.ndimage as ndi
from multiprocessing import Pool
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code','/home/claude/work/manuscript_v2/code/revision2']
from real_image_examples import generate_field
from common import generate3
from kern import tv
from readout import curvature
from bench_shapes import make2d,make3d,FAM2,FAM3
K=pd.read_csv('data/t20_kernel.csv')
CAND={2:dict(D=[1.0,2.0,3.0],al=[0.0,0.125,0.25]),3:dict(D=[1.0,2.0],al=[0.0,0.05,0.1])}
KMAX={2:0.45,3:0.85}
def mbar(dim,D,al):
    q=K[(K.dim==dim)&(K.D==D)&(np.isclose(K.alpha,al))]
    if dim==2:
        q=q.assign(p=q.phi.astype(float)).sort_values('p'); return float(np.trapezoid(q.m,q.p)/45)
    return None
def gen(dim,m,D): return (generate_field(m,30,D=D)[2] if dim==2 else generate3(m,30,D=D)[1]).astype(np.float64)
def sweep(args):
    dim,D,seed=args; rng=np.random.default_rng(seed); out=[]
    radii=[2.5,3,3.5,4,5,6,7,8,10,12,16,20,28,40] if dim==2 else [2.5,3,3.5,4,5,6,7,8,10,13,16,20]
    for R in radii:
        for rep in range(6 if dim==2 else 3):
            n=int(2*R+24); c=n/2+rng.uniform(-.5,.5,dim)
            r=np.sqrt(sum((np.indices((n,)*dim)[a]-c[a])**2 for a in range(dim))); s=gen(dim,r<=R,D); k=np.abs(curvature(s))
            truth=2*np.pi*R if dim==2 else 4*np.pi*R*R
            out.append(dict(R=R,k=(1/R if dim==2 else 2/R),s=s,kk=k,truth=truth))
    return out
def train_shapes(dim):
    jobs=[]; seed=20261118
    for d_,fams,ntr,nte in ((2,FAM2,8,30),(3,FAM3,3,12)):
        for fam in fams:
            for split,nn in (('train',ntr),('test',nte)):
                for i in range(nn):
                    seed+=1
                    if d_==dim and split=='train': jobs.append((fam,i,seed))
    return jobs
def mk(dim,fam,seed): rng=np.random.default_rng(seed); return (make2d if dim==2 else make3d)(fam,rng)
if __name__=='__main__':
    res=[]; consts={}
    m3=json.load(open('data/t3d_mbar.json')) if False else {}
    import sys as _s
    dims=[int(a) for a in _s.argv[1:]] or [2,3]
    for dim in dims:
        for D in CAND[dim]['D']:
            sw=sweep((dim,D,11))
            tr=train_shapes(dim); fields=[]
            for fam,i,seed in tr:
                m,T,size=mk(dim,fam,seed); s=gen(dim,m,D); fields.append((fam,T,s,np.abs(curvature(s))))
            for al in CAND[dim]['al']:
                # planar constant: 2D from angle table; 3D solid-angle weighted from t20 rows
                if dim==2: mb=mbar(2,D,al)
                else:
                    q=K[(K.dim==3)&(K.D==D)&(np.isclose(K.alpha,al))]
                    import math
                    N=np.array([[int(c) for c in str(int(float(v))).zfill(3)] for v in q.phi],float); N=N/np.linalg.norm(N,axis=1)[:,None]
                    u=np.random.default_rng(0).normal(size=(200000,3)); u=np.sort(np.abs(u/np.linalg.norm(u,axis=1)[:,None]),axis=1)[:,::-1]
                    w=np.bincount(np.argmax(u@N.T,axis=1),minlength=len(N))/len(u); mb=float((w*q.m.values).sum())
                kk=np.array([e['k'] for e in sw]); mm=np.array([tv(e['s'],al).sum()/e['truth'] for e in sw])
                A=np.c_[kk,kk**2,kk**3]; b=np.linalg.lstsq(A,mm-mb,rcond=None)[0]
                f=lambda k: mb+b[0]*np.clip(k,0,KMAX[dim])+b[1]*np.clip(k,0,KMAX[dim])**2+b[2]*np.clip(k,0,KMAX[dim])**3
                errs={}
                for fam,T,s,k in fields:
                    errs.setdefault(fam,[]).append(100*(np.sum(tv(s,al)/f(k))/T-1))
                nc=np.mean([np.mean(np.abs(v)) for fm,v in errs.items() if fm!='near_touch'])
                res.append(dict(dim=dim,D=D,alpha=al,mbar=mb,train_MARE=nc,near_touch=np.mean(np.abs(errs['near_touch'])),**{f'fam_{k}':np.mean(np.abs(v)) for k,v in errs.items()}))
                consts[f'{dim}_{D}_{al}']=dict(mbar=mb,coef=b.tolist())
                print(res[-1]['dim'],D,al,round(mb,4),round(nc,3),round(res[-1]['near_touch'],3),flush=True)
    tag='_'.join(map(str,dims)); pd.DataFrame(res).to_csv(f'data/t21_select_{tag}.csv',index=False); json.dump(consts,open(f'data/t21_consts_{tag}.json','w'),indent=1)
