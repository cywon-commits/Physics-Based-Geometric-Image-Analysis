"""Direction- and curvature-dependent metrication: m(phi,kappa) = m_planar(phi) + Delta(kappa).
m_planar from planar walls, expanded in the C4v-invariant harmonics cos(4k phi) (2D) / cubic harmonics (3D)."""
import sys,math,json,numpy as np,pandas as pd,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code','/home/claude/work/manuscript_v2/code/revision2']
from t22_select import gen,split_shapes,sweep,KMAX
from kern import tv,tgrad
from bench_shapes import make2d,make3d
NH=6
KM={2:0.45,3:0.85}
def normals(s,sigma=1.0):
    ss=ndi.gaussian_filter(s,sigma,mode='wrap'); g=[(np.roll(ss,-1,a)-np.roll(ss,1,a))*0.5 for a in range(s.ndim)]
    nr=np.sqrt(sum(x*x for x in g))+1e-12; n=[x/nr for x in g]
    k=-sum((np.roll(n[a],-1,a)-np.roll(n[a],1,a))*0.5 for a in range(s.ndim)); return n,np.abs(k)
def feat(dim,n):
    if dim==2:
        phi=np.arctan2(n[0],n[1]); return np.stack([np.cos(4*j*phi) for j in range(1,NH+1)],-1)
    x,y,z=[np.abs(a) for a in n]; K4=x**4+y**4+z**4-0.6; K6=x*x*y*y*z*z-1/105.; K8=x**8+y**8+z**8-1/3.
    return np.stack([K4,K6,K8,K4*K4-np.mean(K4*K4)],-1)
def planar_data(dim,D,h,al):
    X=[];Y=[]
    if dim==2:
        L=240; y,x=np.mgrid[:L,:L]+0.5
        dirs=sorted({(p,q) for p in range(0,13) for q in range(1,13) if math.gcd(p,q)==1 and p<=q})
        for p,q in dirs:
            s=gen(2,((-p*x+q*y)%L)<L/2,D,h); phi=math.atan2(p,q)
            X.append([math.cos(4*j*phi) for j in range(1,NH+1)]); Y.append(tv(s,al).sum()/(2*L*math.hypot(p,q)))
    else:
        L=60; idx=np.indices((L,)*3).astype(float)+0.5
        cand=[(p,q,r) for p in range(0,8) for q in range(0,p+1) for r in range(0,q+1) if p>0 and math.gcd(math.gcd(p,q),r)==1 and p*p+q*q+r*r<=50]
        for (p,q,r) in cand:
            s=gen(3,((p*idx[0]+q*idx[1]+r*idx[2])%L)<L/2,D,h); nn=np.array([p,q,r],float); nn/=np.linalg.norm(nn)
            X.append(feat(3,[np.array(v) for v in nn])); Y.append(tv(s,al).sum()/(2*L*L*np.linalg.norm([p,q,r])))
    X=np.array(X); Y=np.array(Y); A=np.c_[np.ones(len(X)),X]; c=np.linalg.lstsq(A,Y,rcond=None)[0]
    return c.tolist(), float(np.abs(A@c-Y).max())
def estimate(dim,s,al,c,b):
    n,k=normals(s); F=feat(dim,n); mp=c[0]+F@np.array(c[1:]); kc=np.clip(k,0,KM[dim])
    return float(np.sum(tv(s,al)/(mp+b[0]*kc+b[1]*kc**2+b[2]*kc**3)))
def fit_delta(dim,sw,al,c):
    kk=[];rr=[]
    for e in sw:
        n,_=normals(e['s']); w=tv(e['s'],al); mp=c[0]+feat(dim,n)@np.array(c[1:])
        # ratio of TV to planar-corrected expectation
        kk.append(e['k']); rr.append(w.sum()/(np.sum(w/mp)/1.0)*0+ w.sum()/e['truth'] - np.sum(w*0)+0)
    res=[];kk=[]
    for e in sw:
        n,km=normals(e['s']); w=tv(e['s'],al); mp=c[0]+feat(dim,n)@np.array(c[1:])
        res.append(w.sum()/e['truth']-np.sum(w*mp)/w.sum()); kk.append(np.sum(w*np.clip(km,0,KM[dim]))/w.sum() if MEAS else e['k'])
    kk=np.array(kk); res=np.array(res); b=np.linalg.lstsq(np.c_[kk,kk**2,kk**3],res,rcond=None)[0]; return b.tolist()
MEAS=False
def main():
    global NH,MEAS
    dim=int(sys.argv[1]); cfgs=json.loads(sys.argv[2]); out=[]; consts={}
    for cfg in cfgs:
        D,h=cfg['D'],cfg['h']; NH=cfg.get('NH',6); MEAS=cfg.get('meas',False); KM[dim]=cfg.get('kmax',KM[dim]); sw=sweep(dim,D,h,30)
        fields=[]
        for fam,i,seed in split_shapes(dim,'train'):
            m,T,size=(make2d if dim==2 else make3d)(fam,np.random.default_rng(seed)); fields.append((fam,T,gen(dim,m,D,h)))
        for al in cfg['al']:
            c,fit_err=planar_data(dim,D,h,al); b=fit_delta(dim,sw,al,c); errs={}
            for fam,T,s in fields: errs.setdefault(fam,[]).append(100*(estimate(dim,s,al,c,b)/T-1))
            nc=np.mean([np.mean(np.abs(v)) for f,v in errs.items() if f!='near_touch'])
            key=f'{dim}_D{D}_h{h}_a{al}_NH{NH}_meas{int(MEAS)}_km{KM[dim]}'; consts[key]=dict(D=D,h=h,alpha=al,c=c,b=b,NH=NH,meas=MEAS,kmax=KM[dim],planar_fit_maxerr=fit_err)
            out.append(dict(key=key,train_MARE=nc,near_touch=np.mean(np.abs(errs['near_touch'])),**{'f_'+f:np.mean(np.abs(v)) for f,v in errs.items()}))
            print(key,'planar fit err',round(fit_err,5),'train',round(nc,3),{f:round(np.mean(np.abs(v)),2) for f,v in errs.items()},flush=True)
    tag=sys.argv[3]; pd.DataFrame(out).to_csv(f'data/t23_{tag}.csv',index=False); json.dump(consts,open(f'data/t23_{tag}.json','w'),indent=1)

if __name__=='__main__': main()
