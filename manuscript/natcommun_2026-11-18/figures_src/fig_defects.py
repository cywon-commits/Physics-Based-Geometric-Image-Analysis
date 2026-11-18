import sys,numpy as np,pandas as pd,scipy.ndimage as ndi
sys.path[:0]=['.','../code/revision2','../code']
from style import *
from matplotlib.collections import LineCollection
from dirscale import raw_tensor,qfield,defect_list
import scipy.special as sp_
def F(rho):
    x=rho**2/4; return np.sqrt(np.pi/8)*rho*(sp_.i0e(x)+sp_.i1e(x))
import multithreshold_directors as mt
D='../data/revision2'; OUT='../figures'
def draw(ax,img,q,step=9,L0=9,color=C['pbgia'],defs=None,truth=None,ms=4.5):
    H,W=img.shape; yy,xx=np.mgrid[step//2:H:step,step//2:W:step]
    tan=np.deg2rad(((0.5*np.degrees(np.angle(q)))+90)%180)[yy,xx]; amp=np.abs(q)[yy,xx]; mag=np.clip(amp/np.percentile(np.abs(q),95),0,1)
    L=1+L0*mag; u=np.cos(tan); v=-np.sin(tan)
    seg=np.stack([np.stack([xx-.5*L*u,yy-.5*L*v],-1),np.stack([xx+.5*L*u,yy+.5*L*v],-1)],2).reshape(-1,2,2)
    ax.imshow(img,cmap='gray',vmin=-0.3,vmax=1.3); ax.add_collection(LineCollection(seg,colors=color,linewidths=0.7))
    if truth is not None:
        for r,c,k in truth: ax.plot(c,r,'o',ms=10,mfc='none',mec='white',mew=1.6); ax.plot(c,r,'o',ms=10,mfc='none',mec='black',mew=0.6,ls='none')
    if defs is not None:
        for r,c,k in defs: ax.plot(c,r,'^' if k>0 else 'v',ms=ms,mfc='black' if k>0 else 'white',mec='black',mew=0.7)
    noaxis(ax); ax.set_xlim(0,W);ax.set_ylim(H,0)
truth=[(110,100,+.5),(110,140,-.5),(220,110,+.5),(220,220,-.5)]
img=np.load(f'{D}/t15_lic_img.npy'); raw=np.load('/home/claude/work/data/t15_lic_raw.npy')
cases=mt.construct_cases(); gimg=cases[8][1]; graw=raw_tensor(gimg)
fig=plt.figure(figsize=(FULL,6.6),layout='constrained'); gs=fig.add_gridspec(3,4,height_ratios=[1,1,0.95])
a=fig.add_subplot(gs[0,0]); q,_=qfield(raw,8); draw(a,img,q,defs=defect_list(q,24),truth=truth); a.set_title('synthetic, $\\sigma=8$'); imlabel(a,'a')
b=fig.add_subplot(gs[0,1]); q,_=qfield(raw,20); draw(b,img,q,defs=defect_list(q,24),truth=truth); b.set_title('synthetic, $\\sigma=20$'); imlabel(b,'b')
c=fig.add_subplot(gs[0,2:]); t=pd.read_csv(f'{D}/t15_lic_sigma.csv')
for m,col,mk,lb in (('PBGIA','pbgia','o','PBGIA director'),('ST','conv','^','image structure tensor')):
    q_=t[t.method==m]; c.plot(q_.sigma,(q_.n_plus+q_.n_minus).clip(lower=0.5),mk+'-',color=C[col],ms=3.5,label=lb)
for dd in (40,110): c.axvline(0.4413*dd,color=C['theory'],ls='--',lw=0.8); c.text(0.4413*dd*1.04,150,f'$0.441\\,d$\n$d={dd}$',fontsize=6.5)
c.axhline(4,color=C['prev'],lw=0.6,ls=':'); c.set(xscale='log',yscale='log',xlabel='Gaussian window $\\sigma$ (pixels)',ylabel='detected defects',ylim=(0.4,1000))
c.set_xticks([2,4,8,16,32],['2','4','8','16','32']); c.minorticks_off(); c.legend(loc='lower left'); label(c,'c')
for j,s in enumerate((2,4,8,16)):
    a=fig.add_subplot(gs[1,j]); q,_=qfield(graw,s); draw(a,gimg,q,step=8,L0=8,defs=defect_list(q,24),ms=2.6 if s<8 else 4); a.set_title(f'grass, $\\sigma={s}$'); imlabel(a,'defg'[j])
h=fig.add_subplot(gs[2,:2]); k=pd.read_csv(f'{D}/t14_core.csv'); rho=np.linspace(0,3,200)
for s,mk,col in ((4,'o','pbgia'),(8,'s','smooth'),(16,'^','init')):
    q_=k[k.sigma==s]; h.plot(q_.rho,q_.amp,mk,color=C[col],ms=3,ls='none',label=f'lattice, $\\sigma={s}$')
h.plot(rho,F(rho),color=C['theory'],label='$F(\\rho)$, exact'); h.plot(rho[:60],np.sqrt(np.pi/8)*rho[:60],'--',color=C['conv'],lw=0.9,label='$\\sqrt{\\pi/8}\\,\\rho$')
h.set(xlabel='distance from core $\\rho=r/\\sigma$',ylabel='$|q_\\sigma|/|q_0|$',ylim=(0,1.05)); h.legend(loc='lower right'); label(h,'h')
i=fig.add_subplot(gs[2,2:]); r=pd.read_csv(f'{D}/t16_real_sigma.csv'); r=r[r.method=='PBGIA']; A=271*271
show=[('CC0 grass texture','grass','#0b3d78','o'),('Textile weave (synthetic)','weave','#1f5fae','D'),('CC0 retina vessel contrast','retina','#2a78d6','s'),('CC0 brick wall','brick','#6fa3e6','^'),('Wood grain (synthetic)','wood','#a9c8f0','d')]
for nm,lb,col,mk in show:
    q_=r[r.image==nm]; i.plot(q_.sigma,q_.n_defects.clip(lower=0.5),mk+'-',color=col,mec='#0b3d78',mew=0.4,ms=3.5,label=lb)
ss=np.array([2,16]); i.plot(ss,A/(4*np.pi*ss**2),color=C['theory'],lw=1.1,label='random field $A/4\\pi\\sigma^2$')
i.set(xscale='log',yscale='log',xlabel='Gaussian window $\\sigma$ (pixels)',ylabel='defects in ROI',ylim=(0.4,3000)); i.set_xticks([2,4,8,16],['2','4','8','16']); i.minorticks_off()
i.legend(loc='lower left',ncol=2,fontsize=6.3); label(i,'i')
save(fig,'Fig7_director_defects',OUT); print('ok')
