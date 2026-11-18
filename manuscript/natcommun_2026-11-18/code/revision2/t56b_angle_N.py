"""Wall-normal error vs N against exact nearest-boundary normals (band D<2), pixel level and director sigma=2."""
import sys,numpy as np,scipy.ndimage as ndi,pandas as pd
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code/revision2','/home/claude/work/manuscript_v2/code']
from real_image_examples import generate_field
from readout import tv_density
from scipy.spatial import cKDTree
NS=[0,1,2,3,5,8,12,20,30,50,100,200]
def err_axial(ph,NR,NC,W):
    pt=np.arctan2(NR,NC); d=np.angle(np.exp(2j*(ph-pt)))/2; return np.degrees(np.sqrt(np.sum(W*d*d)/W.sum()))
def tensor_angle(vr,vc,w,sg):
    nn=vr*vr+vc*vc+1e-12; Q=[ndi.gaussian_filter(z*w,sg) for z in (vr*vr/nn,vc*vc/nn,vr*vc/nn)]; return 0.5*np.arctan2(2*Q[2],Q[1]-Q[0])
rng=np.random.default_rng(55601); n=128; rows=[]
for inst in range(40):
    t=rng.uniform(0,np.pi); sh=rng.uniform(-.5,.5,2); R=float(np.exp(rng.uniform(np.log(6),np.log(26)))); ar=rng.uniform(1.5,3)
    a=min(R*np.sqrt(ar),50); b=R/np.sqrt(ar)
    y,x=np.mgrid[:n,:n]+0.0; y=y-(n-1)/2-sh[0]; x=x-(n-1)/2-sh[1]; c,s_=np.cos(t),np.sin(t); u=c*x+s_*y; v=-s_*x+c*y
    m0=(u/a)**2+(v/b)**2<=1
    th=np.linspace(0,2*np.pi,20000,endpoint=False); bu=a*np.cos(th); bv=b*np.sin(th); bx=c*bu-s_*bv; by=s_*bu+c*bv
    nu=-bu/a**2; nv=-bv/b**2; nx=c*nu-s_*nv; ny=s_*nu+c*nv; q=np.hypot(nx,ny); nx/=q; ny/=q
    d,idx=cKDTree(np.c_[by,bx]).query(np.c_[y.ravel(),x.ravel()]); NR=ny[idx].reshape(n,n); NC=nx[idx].reshape(n,n); band=d.reshape(n,n)<2
    for p in (0.0,0.01):
        m=m0^(rng.random(m0.shape)<p) if p>0 else m0
        for N in NS:
            vr,vc,s=[z.astype(float) for z in generate_field(m,N,h=1.5)]; w=tv_density(s); W=w*band
            rows.append(dict(inst=inst,noise=p,b=b,N=N,pix=err_axial(np.arctan2(vr,vc),NR,NC,W),dir2=err_axial(tensor_angle(vr,vc,w,2),NR,NC,W)))
        for sg in (1,1.5,2):
            g=ndi.gaussian_filter(m.astype(float),sg); gr,gc=np.gradient(g); wg=np.hypot(gr,gc)*band
            rows.append(dict(inst=inst,noise=p,b=b,N=-1,method=f'gauss{sg}',pix=err_axial(np.arctan2(gr,gc),NR,NC,wg)))
df=pd.DataFrame(rows); df.to_csv('data/t56b_angle_N.csv',index=False)
print(df[df.N>=0].groupby(['noise','N'])[['pix','dir2']].mean().round(2).to_string()); print(df[df.N<0].groupby(['noise','method']).pix.mean().round(2))
