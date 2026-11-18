"""Grayscale (partial-volume) rendering of the 2D benchmark shapes and the PBGIA generator with a continuous source.
Shape parameters are drawn in exactly the same order as bench_shapes.make2d, so seed -> identical geometry;
pose (t, shift) can be overridden to build pose ensembles of one geometry."""
import sys,numpy as np,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work']
from bench_shapes import rot2,ellipse_perimeter,flower_r,flower_perimeter

def params2d(fam,rng):
    p=dict(fam=fam,t=rng.uniform(0,np.pi),sh=rng.uniform(-.5,.5,2)); R=float(np.exp(rng.uniform(np.log(4),np.log(26))))
    if fam=='disk': p.update(R=R,P=2*np.pi*R,size=R)
    elif fam=='ellipse':
        ar=rng.uniform(1.5,3); a=min(R*np.sqrt(ar),50); b=R/np.sqrt(ar); p.update(a=a,b=b,P=ellipse_perimeter(a,b),size=b)
    elif fam=='stadium':
        L=rng.uniform(1,3)*R; R=min(R,50/(1+L/R/2)); L=min(L,100-2*R); p.update(R=R,L=L,P=2*np.pi*R+2*L,size=R)
    elif fam=='ring':
        w=rng.uniform(4,9); Rc=min(max(R,w/2+4)+w/2,50); p.update(w=w,Rc=Rc,P=4*np.pi*Rc,size=w/2)
    elif fam=='flower':
        k=int(rng.integers(3,7)); eps=rng.uniform(.08,.18); R=max(R,6); p.update(R=R,k=k,eps=eps,P=flower_perimeter(R,eps,k),size=R/(1+eps*(k*k-1)))
    elif fam in ('pair','near_touch'):
        R=min(R,22); g=rng.uniform(6,12) if fam=='pair' else rng.uniform(1,3); p.update(R=R,d=R+g/2,P=4*np.pi*R,size=R)
    return p

def inside(p,u,v):
    f=p['fam']
    if f=='disk': return np.hypot(u,v)<=p['R']
    if f=='ellipse': return (u/p['a'])**2+(v/p['b'])**2<=1
    if f=='stadium': return np.hypot(np.maximum(np.abs(u)-p['L']/2,0),v)<=p['R']
    if f=='ring': return np.abs(np.hypot(u,v)-p['Rc'])<=p['w']/2
    if f=='flower': return np.hypot(u,v)<=flower_r(p['R'],p['eps'],p['k'],np.arctan2(v,u))
    return (np.hypot(u-p['d'],v)<=p['R'])|(np.hypot(u+p['d'],v)<=p['R'])

def render(p,n=128,S=1,t=None,sh=None):
    """Area fraction on an n x n grid using S x S point supersampling (S=1 reproduces make2d's binary mask)."""
    t=p['t'] if t is None else t; sh=p['sh'] if sh is None else sh; Q=rot2(t)
    o=(np.arange(S)+0.5)/S-0.5 if S>1 else np.zeros(1); acc=np.zeros((n,n))
    y0,x0=np.mgrid[:n,:n]+0.0
    for oy in o:
        for ox in o:
            y=y0+oy-(n-1)/2-sh[0]; x=x0+ox-(n-1)/2-sh[1]
            u=Q[0,0]*x+Q[1,0]*y; v=Q[0,1]*x+Q[1,1]*y; acc+=inside(p,u,v)
    return acc/len(o)**2

def kappa2_integral(p,M=20000):
    """Exact-geometry integral of kappa^2 ds (for the quantization-noise floor)."""
    f=p['fam']
    if f=='disk': return 2*np.pi/p['R']
    if f=='stadium': return 2*np.pi/p['R']
    if f=='ring': return 2*np.pi/(p['Rc']+p['w']/2)+2*np.pi/(p['Rc']-p['w']/2)
    if f in ('pair','near_touch'): return 4*np.pi/p['R']
    th=np.linspace(0,2*np.pi,M,endpoint=False)
    if f=='ellipse': x=p['a']*np.cos(th); y=p['b']*np.sin(th)
    else: r=flower_r(p['R'],p['eps'],p['k'],th); x=r*np.cos(th); y=r*np.sin(th)
    dt=th[1]; x1=np.gradient(x,dt); y1=np.gradient(y,dt); x2=np.gradient(x1,dt); y2=np.gradient(y1,dt)
    sp=np.hypot(x1,y1); k=(x1*y2-y1*x2)/sp**3; return float(np.sum(k*k*sp)*dt)

def gen_gray(f,nsteps=30,J=1.0,D=1.0,h=1.5,w_init=2.3,full=False):
    """2D PBGIA generator with a continuous source f in [-1,1] (identical to generate_field for f=+-1)."""
    f=np.asarray(f,np.float64); b=f>0
    xi=ndi.distance_transform_edt(b)-ndi.distance_transform_edt(~b); s=np.tanh(xi/w_init)
    dr=(np.roll(s,-1,0)-np.roll(s,1,0))*0.5; dc=(np.roll(s,-1,1)-np.roll(s,1,1))*0.5
    sc=np.sqrt(np.maximum(0,1-s*s))/np.maximum(np.sqrt(dr*dr+dc*dc),1e-12); vr=dr*sc; vc=dc*sc
    for _ in range(nsteps):
        Hr=J*(np.roll(vr,-1,0)+np.roll(vr,1,0)+np.roll(vr,-1,1)+np.roll(vr,1,1))+D*(np.roll(s,-1,0)-np.roll(s,1,0))*0.5
        Hc=J*(np.roll(vc,-1,0)+np.roll(vc,1,0)+np.roll(vc,-1,1)+np.roll(vc,1,1))+D*(np.roll(s,-1,1)-np.roll(s,1,1))*0.5
        Hs=J*(np.roll(s,-1,0)+np.roll(s,1,0)+np.roll(s,-1,1)+np.roll(s,1,1))+h*f
        H=np.sqrt(Hr*Hr+Hc*Hc+Hs*Hs); ok=H>1e-12; Hm=np.maximum(H,1e-12)
        vr=np.where(ok,Hr/Hm,vr); vc=np.where(ok,Hc/Hm,vc); s=np.where(ok,Hs/Hm,s)
    return (vr,vc,s) if full else s
