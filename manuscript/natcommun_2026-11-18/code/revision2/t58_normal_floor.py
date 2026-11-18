"""Pixel-level wall-normal error: raster limit and its reduction by tensor averaging over sigma."""
import sys,numpy as np,scipy.ndimage as ndi,pandas as pd
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code/revision2','/home/claude/work/manuscript_v2/code']
from real_image_examples import generate_field
from readout import tv_density
from scipy.spatial import cKDTree
rng=np.random.default_rng(5801); n=160; rows=[]
def axial_err(ar,ac,NR,NC,W):   # director (axial) angle error
    ph=np.arctan2(ar,ac); pt=np.arctan2(NR,NC); d=np.angle(np.exp(2j*(ph-pt)))/2; return np.degrees(np.sqrt(np.sum(W*d*d)/W.sum()))
def vec_err(ar,ac,NR,NC,W):
    ang=np.arccos(np.clip((ar*NR+ac*NC)/(np.hypot(ar,ac)+1e-12),-1,1)); return np.degrees(np.sqrt(np.sum(W*ang**2)/W.sum()))
for kind in ('line','disk'):
    for inst in range(24):
        y,x=np.mgrid[:n,:n]+0.0; cy,cx=(n-1)/2+rng.uniform(-.5,.5,2)
        if kind=='line':
            t=rng.uniform(0,np.pi/4); m=(x-cx)*np.cos(t)+(y-cy)*np.sin(t)<0; NR=-np.sin(t)*np.ones((n,n)); NC=-np.cos(t)*np.ones((n,n))
            # also a second wall far away due to periodicity: restrict to central band
            core=np.abs((x-cx)*np.cos(t)+(y-cy)*np.sin(t))<6; core&=(np.abs(x-cx)<40)&(np.abs(y-cy)<40)
        else:
            R=rng.uniform(15,40); r=np.hypot(x-cx,y-cy); m=r<=R; NR=-(y-cy)/(r+1e-9); NC=-(x-cx)/(r+1e-9); core=np.abs(r-R)<6
        vr,vc,s=[z.astype(float) for z in generate_field(m,30,h=1.5)]; w=tv_density(s)*core
        rows.append(dict(kind=kind,method='PBGIA v (pixel)',sig=0,err=vec_err(vr,vc,NR,NC,w)))
        for sg in (1,2,4):
            Q=[ndi.gaussian_filter(a,sg,mode='wrap') for a in (vr*vr,vc*vc,vr*vc)]; q=(Q[1]-Q[0])+2j*Q[2]*(-1)
            # principal axis of tensor
            ph=0.5*np.arctan2(2*Q[2],Q[1]-Q[0]); ar,ac=np.sin(ph),np.cos(ph)
            rows.append(dict(kind=kind,method='PBGIA director',sig=sg,err=axial_err(ar,ac,NR,NC,w)))
        for sg in (1,1.5,2,4):
            g=ndi.gaussian_filter(m.astype(float),sg,mode='wrap'); gr,gc=np.gradient(g); wg=np.hypot(gr,gc)*core
            rows.append(dict(kind=kind,method='Gaussian gradient',sig=sg,err=vec_err(gr,gc,NR,NC,wg)))
        # ideal reference: gradient of the exact (anti-aliased, 8x supersampled) image smoothed at 1
        S=8; yy,xx=(np.mgrid[:n*S,:n*S]+0.5)/S-0.5
        if kind=='line': mm=(xx-cx)*np.cos(t)+(yy-cy)*np.sin(t)<0
        else: mm=np.hypot(xx-cx,yy-cy)<=R
        pv=mm.reshape(n,S,n,S).mean((1,3)); g=ndi.gaussian_filter(pv,1.0,mode='wrap'); gr,gc=np.gradient(g); wg=np.hypot(gr,gc)*core
        rows.append(dict(kind=kind,method='partial-volume gradient',sig=1,err=vec_err(gr,gc,NR,NC,wg)))
df=pd.DataFrame(rows); df.to_csv('data/t58_normal_floor.csv',index=False)
print(df.groupby(['kind','method','sig']).err.mean().round(2).to_string())
