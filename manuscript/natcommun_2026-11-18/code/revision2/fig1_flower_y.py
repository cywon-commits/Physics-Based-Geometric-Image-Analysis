import sys,numpy as np,scipy.ndimage as ndi
sys.path[:0]=['.','../code/revision2','../code','/home/claude/work']
import new_quantitative_studies as nq
from readout import tv_density,curvature,m_of_k
from dirscale import qfield,defect_list,charges
from scipy.integrate import quad
n=110; y,x=np.mgrid[:n,:n].astype(float); X=x-54.7; Y=y-55.3
for R,a,k in ((32,0.12,4),(32,0.15,5),(30,0.10,3),(33,0.08,5)):
    th=np.arctan2(Y,X); m=np.hypot(X,Y)<R*(1+a*np.cos(k*th+0.3))
    vr,vc,s=nq.generate_field(m,nsteps=30); s=s.astype(float)
    P=tv_density(s)/m_of_k(curvature(s),2)
    f=lambda t: np.hypot(R*(1+a*np.cos(k*t)),-R*a*k*np.sin(k*t)); Pe=quad(f,0,2*np.pi,limit=200)[0]
    print(R,a,k,P.sum(),Pe,100*(P.sum()/Pe-1))
# Y shape
def seg(px,py,qx,qy,w):
    dx,dy=qx-px,qy-py; t=np.clip(((x-px)*dx+(y-py)*dy)/(dx*dx+dy*dy),0,1); return np.hypot(x-px-t*dx,y-py-t*dy)<w
c=(55,60); L=36
Ym=np.zeros((n,n),bool)
for ang in (90,210,330):
    a_=np.deg2rad(ang); Ym|=seg(c[0],c[1],c[0]+L*np.cos(a_),c[1]-L*np.sin(a_),5.0)
vr,vc,s=nq.generate_field(Ym,nsteps=30); raw=np.stack([vr*vr,vc*vc,vr*vc]).astype(float)
for sig in (3,4,5,6,8,10):
    q,_=qfield(raw,sig); d=defect_list(q,3); print(sig,[(round(r),round(cc),kk) for r,cc,kk in d], sum(kk for *_,kk in d))
np.save('/tmp/p/ymask.npy',Ym)
