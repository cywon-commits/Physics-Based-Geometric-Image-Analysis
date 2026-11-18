"""Orientation bias of the wall normal on the lattice: dphi = sum_j a_j sin(4 j phi) (odd about mirror axes),
fitted on planar walls only; predicted leading term (k^2/24) sin 4phi from centered differences."""
import sys,math,json,numpy as np,scipy.ndimage as ndi,pandas as pd
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code/revision2','/home/claude/work/manuscript_v2/code']
from real_image_examples import generate_field
from readout import tv_density
from scipy.spatial import cKDTree
L=240; y,x=np.mgrid[:L,:L]+0.5; P=[];B=[];K2=[]
for p in range(0,13):
    for q in range(1,13):
        if p>q or math.gcd(p,q)!=1: continue
        m=((-p*x+q*y)%L)<L/2; true=np.arctan2(q,-p)
        vr,vc,s=[z.astype(float) for z in generate_field(m,30,h=1.5)]; w=tv_density(s)
        Q=[(a*w).sum() for a in (vr*vr,vc*vc,vr*vc)]; ph=0.5*np.arctan2(2*Q[2],Q[1]-Q[0])
        bias=np.angle(np.exp(2j*(ph-true)))/2; phi=np.arctan2(p,q); P.append(phi); B.append(bias)
        # effective k^2 of the wall: spectrum of s along the normal, weighted by |FT of grad|^2
        prof=s[:, L//2] if p==0 else None
P=np.array(P);B=np.array(B)
A=np.c_[[np.sin(4*j*P) for j in (1,2,3)]].T; a=np.linalg.lstsq(A,B,rcond=None)[0]
print('a_j (deg):',np.degrees(a).round(3),' max resid (deg):',np.degrees(np.abs(A@a-B).max()).round(3))
# k^2 of a straight axis-aligned wall from the lattice profile of s (gradient-weighted spectral mean)
m=(y<L/2); vr,vc,s=[z.astype(float) for z in generate_field(m,30,h=1.5)]; pr=s[:,5]; g=np.gradient(pr); G=np.abs(np.fft.rfft(g))**2; k=np.fft.rfftfreq(L)*2*np.pi
k2=(G*k**2).sum()/G.sum(); print('wall <k^2>=%.3f  -> predicted a1=%.3f deg'%(k2,np.degrees(k2/24)))
json.dump(dict(a=a.tolist(),k2=k2),open('data/t59_bias.json','w'))
def corr(ph,a): return ph-sum(a[j]*np.sin(4*(j+1)*ph) for j in range(len(a)))
rows=[]; rng=np.random.default_rng(5901); n=128
def err_axial(ph,NR,NC,W):
    pt=np.arctan2(NR,NC); d=np.angle(np.exp(2j*(ph-pt)))/2; return np.degrees(np.sqrt(np.sum(W*d*d)/W.sum()))
def tensor_angle(vr,vc,w,sg):
    nn=vr*vr+vc*vc+1e-12; Q=[ndi.gaussian_filter(z*w,sg) for z in (vr*vr/nn,vc*vc/nn,vr*vc/nn)]; return 0.5*np.arctan2(2*Q[2],Q[1]-Q[0])
for shape in ('ellipse','line','disk'):
  for inst in range(40 if shape=='ellipse' else 16):
    yy,xx=np.mgrid[:n,:n]+0.0; sh=rng.uniform(-.5,.5,2); yy=yy-(n-1)/2-sh[0]; xx=xx-(n-1)/2-sh[1]
    if shape=='ellipse':
        t=rng.uniform(0,np.pi); R=float(np.exp(rng.uniform(np.log(6),np.log(26)))); ar=rng.uniform(1.5,3); aa=min(R*np.sqrt(ar),50); b=R/np.sqrt(ar)
    elif shape=='disk': t=0; aa=b=rng.uniform(10,40)
    else: t=rng.uniform(0,np.pi); aa=b=None
    c,s_=np.cos(t),np.sin(t)
    if shape=='line':
        m=(c*xx+s_*yy)<0; NC=-c*np.ones((n,n)); NR=-s_*np.ones((n,n)); D=np.abs(c*xx+s_*yy); D[(np.abs(xx)>45)|(np.abs(yy)>45)]=99; b=np.inf
    else:
        u=c*xx+s_*yy; v=-s_*xx+c*yy; m=(u/aa)**2+(v/b)**2<=1
        th=np.linspace(0,2*np.pi,20000,endpoint=False); bu=aa*np.cos(th); bv=b*np.sin(th); bx=c*bu-s_*bv; by=s_*bu+c*bv
        nu=-bu/aa**2; nv=-bv/b**2; nx=c*nu-s_*nv; ny=s_*nu+c*nv; nn_=np.hypot(nx,ny); nx/=nn_; ny/=nn_
        d,idx=cKDTree(np.c_[by,bx]).query(np.c_[yy.ravel(),xx.ravel()]); NR=ny[idx].reshape(n,n); NC=nx[idx].reshape(n,n); D=d.reshape(n,n)
    band=D<2
    for p in (0.0,0.01):
        mm=m^(rng.random(m.shape)<p) if p>0 else m
        vr,vc,s=[z.astype(float) for z in generate_field(mm,30,h=1.5)]; w=tv_density(s); W=w*band
        ph=np.arctan2(vr,vc)
        rows.append(dict(shape=shape,inst=inst,noise=p,b=b,method='PBGIA pixel',sig=0,err=err_axial(ph,NR,NC,W)))
        rows.append(dict(shape=shape,inst=inst,noise=p,b=b,method='PBGIA pixel, corrected',sig=0,err=err_axial(corr(ph,a),NR,NC,W)))
        for sg in (1,2,4):
            phs=tensor_angle(vr,vc,w,sg)
            rows.append(dict(shape=shape,inst=inst,noise=p,b=b,method='PBGIA director',sig=sg,err=err_axial(phs,NR,NC,W)))
            rows.append(dict(shape=shape,inst=inst,noise=p,b=b,method='PBGIA director, corrected',sig=sg,err=err_axial(corr(phs,a),NR,NC,W)))
        for sg in (1,1.5,2,4):
            g=ndi.gaussian_filter(mm.astype(float),sg); gr,gc=np.gradient(g); wg=np.hypot(gr,gc)*band
            rows.append(dict(shape=shape,inst=inst,noise=p,b=b,method='Gaussian gradient',sig=sg,err=err_axial(np.arctan2(gr,gc),NR,NC,wg)))
df=pd.DataFrame(rows); df.to_csv('data/t59_ellipse_normals.csv',index=False)
print(df.groupby(['shape','noise','method','sig']).err.mean().round(2).unstack(['shape','noise']).to_string())
