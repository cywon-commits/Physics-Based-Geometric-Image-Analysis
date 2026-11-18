"""Colour texture: luminance-then-cut vs channel-wise cuts vs continuous luminance source (appendix figure)."""
import sys,numpy as np
sys.path[:0]=['.','../code','/home/claude/work','/home/claude/work/manuscript_v2/code/revision2']
from style import *
from matplotlib.collections import LineCollection
from t45_color_director import synth,scalars,tens_cuts,tens_cont,director,W
OUT='../figures'
I,truth=synth(0.05); S=scalars(I)
def segs(ax,bg,tang,gap,coh,color,step=11,Lm=11,gate=.25):
    H,Wd=bg.shape[:2]; yy,xx=np.mgrid[step//2:H:step,step//2:Wd:step]; a=tang[yy,xx]; c=coh[yy,xx]; g=gap[yy,xx]
    half=xx<160; ref=np.where(half,np.percentile(gap[:,:160],95),np.percentile(gap[:,160:],95))   # normalized within each half
    mag=np.clip(g/np.maximum(ref,1e-10),0,1); ok=(c>=gate)&(mag>=.07)
    L=1.5+Lm*mag[ok]; u=np.cos(np.deg2rad(a[ok])); v=-np.sin(np.deg2rad(a[ok])); xc=xx[ok]; yc=yy[ok]
    ax.imshow(bg,cmap='gray',vmin=0,vmax=1) if bg.ndim==2 else ax.imshow(bg)
    ax.add_collection(LineCollection(np.stack([np.stack([xc-.5*L*u,yc-.5*L*v],-1),np.stack([xc+.5*L*u,yc+.5*L*v],-1)],1),colors=color,linewidths=0.8))
    ax.axvline(159.5,color='white',lw=0.6,ls='--')
rgb=np.clip(np.moveaxis(I,0,-1),0,1); Y=S['Y'][0]; ys=np.clip((Y-Y.min())/np.ptp(Y),0,1)
fig,ax=plt.subplots(1,4,figsize=(FULL,FULL/4+0.25),layout='constrained')
ax[0].imshow(rgb); ax[0].axvline(159.5,color='white',lw=0.6,ls='--'); ax[0].set_title('color input',fontsize=7)
ax[1].imshow(ys,cmap='gray',vmin=0,vmax=1); ax[1].axvline(159.5,color='white',lw=0.6,ls='--'); ax[1].set_title('luminance $Y$',fontsize=7)
T=sum(tens_cuts(a)[0] for a in S['Y']); segs(ax[2],0.3+0.45*rgb,*director(T),'#ffe54b'); ax[2].set_title('cuts of $Y$',fontsize=7)
T=sum(tens_cuts(a)[0] for a in S['RGB'])/3; segs(ax[3],0.3+0.45*rgb,*director(T),'#ffe54b'); ax[3].set_title('cuts of R, G, B (pooled)',fontsize=7)
for a,l in zip(ax,'abcd'): noaxis(a); imlabel(a,l)
save(fig,'FigA5_color',OUT)
