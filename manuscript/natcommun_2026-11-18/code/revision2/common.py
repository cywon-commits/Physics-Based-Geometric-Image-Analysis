import sys,numpy as np, scipy.ndimage as ndi
sys.path.insert(0,'/home/claude/repo/Physics-Based-Geometric-Image-Analysis/manuscript/integrated_2026-11-17/code')
from real_image_examples import generate_field   # 2D reference implementation (user code)
def tv2(s):
    gr,gc=np.gradient(s.astype(np.float64)); return 0.5*np.sum(np.hypot(gr,gc))
def ie2(vr,vc,s):
    return sum(np.sum(np.square(g)) for z in (vr,vc,s) for g in np.gradient(z.astype(np.float64)))
def generate3(mask,nsteps=30,J=1.0,D=1.0,h=2.0,w_init=2.3):
    b=np.asarray(mask,bool); f=np.where(b,1.0,-1.0).astype(np.float32)
    xi=ndi.distance_transform_edt(b)-ndi.distance_transform_edt(~b)
    s=np.tanh(xi/w_init).astype(np.float32)
    g=[(np.roll(s,-1,a)-np.roll(s,1,a))*0.5 for a in range(3)]
    nr=np.sqrt(sum(x*x for x in g)); sc=np.sqrt(np.maximum(0,1-s*s))/np.maximum(nr,1e-12)
    v=[x*sc for x in g]
    for _ in range(nsteps):
        Hv=[J*sum(np.roll(v[c],k,a) for a in range(3) for k in (1,-1))+D*(np.roll(s,-1,c)-np.roll(s,1,c))*0.5 for c in range(3)]
        Hs=J*sum(np.roll(s,k,a) for a in range(3) for k in (1,-1))+h*f
        H=np.sqrt(sum(x*x for x in Hv)+Hs*Hs); ok=H>1e-12; Hm=np.maximum(H,1e-12)
        v=[np.where(ok,x/Hm,y) for x,y in zip(Hv,v)]; s=np.where(ok,Hs/Hm,s)
    return v,s
def tv3(s):
    g=np.gradient(s.astype(np.float64)); return 0.5*np.sum(np.sqrt(sum(x*x for x in g)))
def pgrad(z):
    z=z.astype(np.float64); return [(np.roll(z,-1,a)-np.roll(z,1,a))*0.5 for a in range(z.ndim)]
def tv_per(s):
    return 0.5*np.sum(np.sqrt(sum(g*g for g in pgrad(s))))
def ie_per(comps):
    return sum(np.sum(g*g) for z in comps for g in pgrad(z))
