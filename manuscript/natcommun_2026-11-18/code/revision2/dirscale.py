"""Scale-resolved projective director from PBGIA fields (shared helpers)."""
import sys,numpy as np,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work/manuscript_v2/code']
from real_image_examples import generate_field
def raw_tensor(img,thresholds=(.15,.325,.5,.675,.85),smooth=.8):
    """unsmoothed threshold-pooled outer products of v (per pixel): returns (rr,cc,rc)"""
    sm=ndi.gaussian_filter(img,smooth); cuts=np.unique(np.quantile(sm,thresholds)); acc=None; n=0
    for q in cuts:
        m=sm>q
        if m.mean()<.005 or m.mean()>.995: continue
        vr,vc,s=generate_field(m,nsteps=30); t=np.stack([vr*vr,vc*vc,vr*vc]).astype(np.float64)
        acc=t if acc is None else acc+t; n+=1
    return acc/n
def st_raw(img,pre=1.2):
    a=ndi.gaussian_filter(img,pre); gr,gc=np.gradient(a); return np.stack([gr*gr,gc*gc,gr*gc])
def qfield(raw,sigma):
    """complex order parameter of the normal director, q = (Qcc-Qrr) - 2i Qrc  (=|q| e^{2i phi_normal}, phi measured from +x(col) toward -row (up))"""
    rr,cc,rc=[ndi.gaussian_filter(a,sigma,mode='reflect') for a in raw]
    return (cc-rr)-2j*rc, rr+cc
def tangent_deg(q):  # director tangent angle in degrees, image convention as in multithreshold_directors
    return ((0.5*np.degrees(np.angle(q)))+90)%180
def charges(q):
    a=np.angle(q); d=lambda u,v:(v-u+np.pi)%(2*np.pi)-np.pi
    w=d(a[:-1,:-1],a[:-1,1:])+d(a[:-1,1:],a[1:,1:])+d(a[1:,1:],a[1:,:-1])+d(a[1:,:-1],a[:-1,:-1])
    return -np.rint(w/(2*np.pi))/2.0     # loop (r,c)->(r,c+1)->(r+1,c+1)->(r+1,c) is clockwise for y up; minus sign gives the standard counterclockwise charge
def defect_list(q,margin=16):
    k=charges(q); r,c=np.nonzero(k); keep=(r>=margin)&(c>=margin)&(r<k.shape[0]-margin)&(c<k.shape[1]-margin)
    return [(rr+.5,cc+.5,k[rr,cc]) for rr,cc in zip(r[keep],c[keep])]
