"""Shape families with exact perimeters / surface areas and random poses."""
import numpy as np
from scipy import special, integrate
def rot2(t): c,s=np.cos(t),np.sin(t); return np.array([[c,-s],[s,c]])
def rand_rot3(rng):
    q=rng.normal(size=4); q/=np.linalg.norm(q); a,b,c,d=q
    return np.array([[a*a+b*b-c*c-d*d,2*(b*c-a*d),2*(b*d+a*c)],[2*(b*c+a*d),a*a-b*b+c*c-d*d,2*(c*d-a*b)],[2*(b*d-a*c),2*(c*d+a*b),a*a-b*b-c*c+d*d]])
def ellipse_perimeter(a,b):
    return 4*a*special.ellipe(1-(b/a)**2) if a>=b else ellipse_perimeter(b,a)
def ellipsoid_area(a,b,c):
    a,b,c=sorted([a,b,c],reverse=True)
    phi=np.arccos(c/a); m=(a*a*(b*b-c*c))/(b*b*(a*a-c*c))
    return 2*np.pi*c*c+2*np.pi*a*b/np.sin(phi)*(special.ellipeinc(phi,m)*np.sin(phi)**2+special.ellipkinc(phi,m)*np.cos(phi)**2)
def flower_r(R,eps,k,th): return R*(1+eps*np.cos(k*th))
def flower_perimeter(R,eps,k):
    th=np.linspace(0,2*np.pi,200001); r=flower_r(R,eps,k,th); dr=-R*eps*k*np.sin(k*th)
    return np.trapezoid(np.sqrt(r*r+dr*dr),th)
def bumpy_r(R,eps,th,ph): return R*(1+eps*np.sin(th)**2*np.cos(3*ph))
def bumpy_area(R,eps):
    th=np.linspace(0,np.pi,1601); ph=np.linspace(0,2*np.pi,3201); T,P=np.meshgrid(th,ph,indexing='ij')
    r=bumpy_r(R,eps,T,P); rt=R*eps*2*np.sin(T)*np.cos(T)*np.cos(3*P); rp=-R*eps*np.sin(T)**2*3*np.sin(3*P)
    dA=np.sqrt(r*r*(r*r+rt*rt)*np.sin(T)**2+r*r*rp*rp)
    return np.trapezoid(np.trapezoid(dA,ph,axis=1),th)
# ---------------- 2D
def make2d(fam,rng,n=128):
    t=rng.uniform(0,np.pi); sh=rng.uniform(-.5,.5,2); y,x=np.mgrid[:n,:n]+0.0; y=y-(n-1)/2-sh[0]; x=x-(n-1)/2-sh[1]
    Q=rot2(t); u=Q[0,0]*x+Q[1,0]*y; v=Q[0,1]*x+Q[1,1]*y   # body coords
    R=float(np.exp(rng.uniform(np.log(4),np.log(26))))
    if fam=='disk': m=np.hypot(u,v)<=R; P=2*np.pi*R; size=R
    elif fam=='ellipse':
        ar=rng.uniform(1.5,3); a=R*np.sqrt(ar); b=R/np.sqrt(ar); a=min(a,50); m=(u/a)**2+(v/b)**2<=1; P=ellipse_perimeter(a,b); size=b
    elif fam=='stadium':
        L=rng.uniform(1,3)*R; R=min(R,50/(1+L/R/2)); L=min(L,100-2*R); m=np.hypot(np.maximum(np.abs(u)-L/2,0),v)<=R; P=2*np.pi*R+2*L; size=R
    elif fam=='ring':
        w=rng.uniform(4,9); Rc=max(R,w/2+4)+w/2; Rc=min(Rc,50); rr=np.hypot(u,v); m=np.abs(rr-Rc)<=w/2; P=2*np.pi*(2*Rc); size=w/2
    elif fam=='flower':
        k=int(rng.integers(3,7)); eps=rng.uniform(.08,.18); R=max(R,6); th=np.arctan2(v,u); m=np.hypot(u,v)<=flower_r(R,eps,k,th); P=flower_perimeter(R,eps,k)
        size=R/(1+eps*(k*k-1))    # min radius of curvature (approx, at inner lobes)
    elif fam in ('pair','near_touch'):
        R=min(R,22); g=rng.uniform(6,12) if fam=='pair' else rng.uniform(1,3)
        d=R+g/2; m=(np.hypot(u-d,v)<=R)|(np.hypot(u+d,v)<=R); P=4*np.pi*R; size=R
    return m,P,size
# ---------------- 3D
def make3d(fam,rng,n=64):
    Q=rand_rot3(rng); sh=rng.uniform(-.5,.5,3); g=np.indices((n,n,n)).astype(np.float64)-(n-1)/2
    g=g-sh.reshape(3,1,1,1); u,v,w=np.einsum('ij,j...->i...',Q.T,g)
    R=float(np.exp(rng.uniform(np.log(4),np.log(15))))
    if fam=='sphere': m=u*u+v*v+w*w<=R*R; A=4*np.pi*R*R; size=R
    elif fam=='ellipsoid':
        a,b,c=R*np.array([rng.uniform(1.2,1.6),1.0,rng.uniform(.6,.85)]); a=min(a,24); m=(u/a)**2+(v/b)**2+(w/c)**2<=1; A=ellipsoid_area(a,b,c); size=c
    elif fam=='capsule':
        L=rng.uniform(1,2.5)*R; R=min(R,12); L=min(L,46-2*R); m=np.sqrt(np.maximum(np.abs(u)-L/2,0)**2+v*v+w*w)<=R; A=4*np.pi*R*R+2*np.pi*R*L; size=R
    elif fam=='torus':
        r=rng.uniform(3,6); Rc=min(max(R,r+3)+r,24-r); m=(np.hypot(u,v)-Rc)**2+w*w<=r*r; A=4*np.pi**2*Rc*r; size=r
    elif fam=='bumpy':
        eps=rng.uniform(.08,.15); R=max(R,6); R=min(R,20); rr=np.sqrt(u*u+v*v+w*w); th=np.arccos(np.clip(w/np.maximum(rr,1e-9),-1,1)); ph=np.arctan2(v,u)
        m=rr<=bumpy_r(R,eps,th,ph); A=bumpy_area(R,eps); size=R/(1+8*eps)
    elif fam in ('pair','near_touch'):
        R=min(R,11); gp=rng.uniform(5,9) if fam=='pair' else rng.uniform(1,3); d=R+gp/2
        m=((u-d)**2+v*v+w*w<=R*R)|((u+d)**2+v*v+w*w<=R*R); A=8*np.pi*R*R; size=R
    return m,A,size
FAM2=['disk','ellipse','stadium','ring','flower','pair','near_touch']
FAM3=['sphere','ellipsoid','capsule','torus','bumpy','pair','near_touch']
