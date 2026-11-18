"""Candidate generator/readout configurations; all constants from planar walls + disk/sphere sweeps; selection on benchmark TRAIN only."""
import sys,math,json,numpy as np,pandas as pd
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code','/home/claude/work/manuscript_v2/code/revision2']
from real_image_examples import generate_field
from common import generate3
from kern import tv
from readout import curvature
from bench_shapes import make2d,make3d,FAM2,FAM3
KMAX={2:0.45,3:0.85}
def gen(dim,m,D,h,N=30): return (generate_field(m,N,D=D,h=h)[2] if dim==2 else generate3(m,N,D=D,h=h)[1]).astype(np.float64)
def planar(dim,D,h,als,N):
    out={a:[] for a in als}
    if dim==2:
        L=120; y,x=np.mgrid[:L,:L]+0.5; pairs=[(0,1),(1,8),(1,5),(1,4),(1,3),(2,5),(1,2),(3,5),(2,3),(3,4),(4,5),(1,1)]; ph=[]
        for p,q in pairs:
            s=gen(2,((-p*x+q*y)%L)<L/2,D,h,N); ph.append(math.degrees(math.atan2(p,q)))
            for a in als: out[a].append(tv(s,a).sum()/(2*L*math.hypot(p,q)))
        return {a:float(np.trapezoid(out[a],ph)/45) for a in als}
    L=60; idx=np.indices((L,)*3).astype(float)+0.5
    cand=[(p,q,r) for p in range(0,8) for q in range(0,p+1) for r in range(0,q+1) if p>0 and math.gcd(math.gcd(p,q),r)==1 and p*p+q*q+r*r<=50]
    Nn=np.array(cand,float); Nn/=np.linalg.norm(Nn,axis=1)[:,None]
    u=np.random.default_rng(0).normal(size=(200000,3)); u=np.sort(np.abs(u/np.linalg.norm(u,axis=1)[:,None]),axis=1)[:,::-1]
    w=np.bincount(np.argmax(u@Nn.T,axis=1),minlength=len(cand))/len(u)
    for (p,q,r) in cand:
        s=gen(3,((p*idx[0]+q*idx[1]+r*idx[2])%L)<L/2,D,h,N)
        for a in als: out[a].append(tv(s,a).sum()/(2*L*L*math.sqrt(p*p+q*q+r*r)))
    return {a:float((w*np.array(out[a])).sum()) for a in als}
def sweep(dim,D,h,N,seed=11):
    rng=np.random.default_rng(seed); out=[]
    radii=[2.5,3,3.5,4,5,6,7,8,10,12,16,20,28,40] if dim==2 else [2.5,3,3.5,4,5,6,7,8,10,13,16,20]
    for R in radii:
        for rep in range(6 if dim==2 else 3):
            n=int(2*R+24); c=n/2+rng.uniform(-.5,.5,dim)
            r=np.sqrt(sum((np.indices((n,)*dim)[a]-c[a])**2 for a in range(dim))); s=gen(dim,r<=R,D,h,N)
            out.append(dict(k=(1/R if dim==2 else 2/R),s=s,kk=np.abs(curvature(s)),truth=(2*np.pi*R if dim==2 else 4*np.pi*R*R)))
    return out
def split_shapes(dim,which):
    jobs=[]; seed=20261118
    for d_,fams,ntr,nte in ((2,FAM2,8,30),(3,FAM3,3,12)):
        for fam in fams:
            for split,nn in (('train',ntr),('test',nte)):
                for i in range(nn):
                    seed+=1
                    if d_==dim and split==which: jobs.append((fam,i,seed))
    return jobs
def fitlaw(dim,sw,al,mb):
    kk=np.array([e['k'] for e in sw]); mm=np.array([tv(e['s'],al).sum()/e['truth'] for e in sw])
    b=np.linalg.lstsq(np.c_[kk,kk**2,kk**3],mm-mb,rcond=None)[0]; return b.tolist()
def law(k,dim,mb,b):
    k=np.clip(k,0,KMAX[dim]); return mb+b[0]*k+b[1]*k*k+b[2]*k**3
if __name__=='__main__':
    dim=int(sys.argv[1]); configs=json.loads(sys.argv[2]); res=[]; consts={}
    for cfg in configs:
        D,h,N,als=cfg['D'],cfg['h'],cfg.get('N',30),cfg['al']
        mbs=planar(dim,D,h,als,N); sw=sweep(dim,D,h,N)
        fields=[]
        for fam,i,seed in split_shapes(dim,'train'):
            m,T,size=(make2d if dim==2 else make3d)(fam,np.random.default_rng(seed)); s=gen(dim,m,D,h,N); fields.append((fam,T,s,np.abs(curvature(s))))
        for al in als:
            b=fitlaw(dim,sw,al,mbs[al]); errs={}
            for fam,T,s,k in fields: errs.setdefault(fam,[]).append(100*(np.sum(tv(s,al)/law(k,dim,mbs[al],b))/T-1))
            nc=np.mean([np.mean(np.abs(v)) for f,v in errs.items() if f!='near_touch'])
            key=f'{dim}_D{D}_h{h}_N{N}_a{al}'; consts[key]=dict(D=D,h=h,N=N,alpha=al,mbar=mbs[al],coef=b)
            res.append(dict(key=key,dim=dim,D=D,h=h,N=N,alpha=al,mbar=mbs[al],train_MARE=nc,near_touch=np.mean(np.abs(errs['near_touch'])),**{'f_'+f:np.mean(np.abs(v)) for f,v in errs.items()}))
            print(key,round(mbs[al],4),round(nc,3),round(res[-1]['near_touch'],3),flush=True)
    tag=sys.argv[3]; pd.DataFrame(res).to_csv(f'data/t22_{tag}.csv',index=False); json.dump(consts,open(f'data/t22_{tag}.json','w'),indent=1)
