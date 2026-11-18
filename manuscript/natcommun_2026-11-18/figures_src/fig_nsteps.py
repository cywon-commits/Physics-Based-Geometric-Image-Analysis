"""Supplementary figure: convergence with the number of synchronous updates N."""
import sys,json,numpy as np,pandas as pd
from style import *
import matplotlib as mpl
D='/home/claude/work/data'; OUT='../figures'
NOISE=[0.0,0.002,0.01]; COL=dict(zip(NOISE,['#9ec5f0','#2a78d6','#0b3d78'])); LBL={0.0:'clean',0.002:'0.2% flips',0.01:'1% flips'}
def x_(N): return np.where(np.asarray(N)==0,0.5,N)
def fam_mean(df,col):
    nc=df[df.family!='near_touch'].copy(); nc['e']=100*(nc[col]/nc.truth-1).abs()
    return nc.groupby(['noise','N','family']).e.mean().groupby(['noise','N']).mean()
t=pd.read_csv(f'{D}/t55_nsteps.csv'); a=pd.read_csv(f'{D}/t56_nsteps_aux.csv'); b3=pd.read_csv(f'{D}/t55b_nsteps3d.csv'); rt=json.load(open(f'{D}/t57_runtime.json'))
fig,ax=plt.subplots(2,3,figsize=(FULL,4.3),layout='constrained')
# a residual
A=a[a.N>=1]
for p,mk in ((0.0,'o'),(0.01,'s')):
    q=A[A.noise==p].groupby('N').res_wall.mean(); ax[0,0].plot(q.index,q.values,mk+'-',color=COL[p if p else 0.002] if p==0 else COL[0.01],ms=3,label=LBL[p])
NN=np.array([1,200]); lam=4/5.5; NNd=np.linspace(8,45,60); ax[0,0].plot(NNd,0.0056*lam**(NNd-8),'--',color=C['theory'],lw=0.9,label='$\\lambda^N$, $\\lambda=zJ/(zJ+h)$')
ax[0,0].set(xscale='log',yscale='log',xlabel='updates $N$',ylabel='$|\\mathbf{n}-\\mathbf{H}/|\\mathbf{H}||$ on walls',ylim=(1e-7,1)); ax[0,0].legend(loc='lower left'); label(ax[0,0],'a')
# b 2D
F=fam_mean(t[t.dim==2],'frozen'); R=fam_mean(t[t.dim==2],'refit'); G2={0.0:0.70,0.002:0.71,0.01:1.06}
for p in NOISE:
    f=F.loc[p]; r=R.loc[p]; ax[0,1].plot(x_(f.index),f.values,'o-',color=COL[p],ms=2.8,label=LBL[p]); ax[0,1].plot(x_(r.index),r.values,'--',color=COL[p],lw=0.9)
    ax[0,1].axhline(G2[p],color=COL[p],ls=':',lw=0.8)
ax[0,1].set(xscale='log',yscale='log',xlabel='updates $N$ ($N=0$ at 0.5)',ylabel='2D perimeter MARE (%)',ylim=(0.3,5e4)); ax[0,1].legend(loc='upper right',fontsize=6.3); label(ax[0,1],'b')
# c 3D
F3=fam_mean(t[t.dim==3],'frozen'); R3=fam_mean(b3,'anchored'); G3={0.0:0.56,0.002:0.76,0.01:10.1}
for p in NOISE:
    f=F3.loc[p]; r=R3.loc[p]; ax[0,2].plot(x_(f.index),f.values,'o-',color=COL[p],ms=2.8,label=LBL[p]); ax[0,2].plot(x_(r.index),r.values,'--',color=COL[p],lw=0.9)
    ax[0,2].axhline(G3[p],color=COL[p],ls=':',lw=0.8)
ax[0,2].set(xscale='log',yscale='log',xlabel='updates $N$ ($N=0$ at 0.5)',ylabel='3D surface-area MARE (%)',ylim=(0.2,5e4)); label(ax[0,2],'c')
from matplotlib.lines import Line2D
ax[0,2].legend(handles=[Line2D([],[],color=C['ink'],marker='o',ms=2.8,label='constants of $N=30$'),Line2D([],[],color=C['ink'],ls='--',lw=0.9,label='re-derived at $N$'),Line2D([],[],color=C['ink'],ls=':',lw=0.8,label='Gaussian + MS/MC')],loc='upper right',fontsize=6.3,handlelength=1.8)
# d angle: exact nearest-boundary normals, band of 2 px (t56b), clean ellipses split by minor semi-axis
ab=pd.read_csv(f'{D}/t56b_angle_N.csv'); a0=ab[(ab.noise==0)&(ab.N>=0)]
for cond,c,lb in ((a0.b>=8,'#0b3d78','$b\\geq8$'),(a0.b<8,'#9ec5f0','$b<8$')):
    q=a0[cond].groupby('N')[['pix','dir2']].mean()
    ax[1,0].plot(x_(q.index),q.pix.values,'o-',color=c,ms=2.8,label=f'pixel, {lb}')
    ax[1,0].plot(x_(q.index),q.dir2.values,'o--',color=c,ms=2.5,mfc='white',label=f'director $\\sigma=2$, {lb}')
ax[1,0].set(xscale='log',xlabel='updates $N$ ($N=0$ at 0.5)',ylabel='wall-normal error, rms (deg)',ylim=(0,16)); ax[1,0].legend(loc='upper right',fontsize=6.0); label(ax[1,0],'d')
# e spurious domains
q=a[(a.N>=0)&(a.noise==0.01)].groupby('N')[['ncomp','nhole']].mean(); ex=(q.ncomp-1)+q.nhole
ax[1,1].plot(x_(q.index),ex.clip(lower=0.01).values,'s-',color=COL[0.01],ms=2.8)
ax[1,1].set(xscale='log',yscale='log',xlabel='updates $N$ ($N=0$ at 0.5)',ylabel='spurious domains and holes',ylim=(0.008,300)); label(ax[1,1],'e')
ax[1,1].text(0.97,0.92,'ellipses, 1% flips\n(zero plotted at 0.01)',transform=ax[1,1].transAxes,ha='right',va='top',fontsize=6.3)
# f runtime
Ns=np.array([0,5,10,30])
g2=np.array([rt[f'2d_gen_N{n}'] for n in Ns])+rt['2d_readout']; g3=np.array([rt[f'3d_gen_N{n}'] for n in Ns])+rt['3d_readout']
ax[1,2].plot(x_(Ns),g2,'o-',color=C['pbgia'],ms=3,label='PBGIA, $128^2$'); ax[1,2].plot(x_(Ns),g3,'s-',color='#0b3d78',ms=3,label='PBGIA, $64^3$')
ax[1,2].axhline(rt['2d_gauss_ms'],color=C['pbgia'],ls=':',lw=0.8); ax[1,2].axhline(rt['3d_gauss_mc'],color='#0b3d78',ls=':',lw=0.8)
ax[1,2].text(30,rt['2d_gauss_ms']*1.3,'Gaussian + MS',fontsize=6,color='#55607a',ha='right'); ax[1,2].text(30,rt['3d_gauss_mc']*0.62,'Gaussian + MC',fontsize=6,color='#55607a',ha='right',va='top')
ax[1,2].set(xscale='log',yscale='log',xlabel='updates $N$ ($N=0$ at 0.5)',ylabel='time per image (ms, one CPU core)',ylim=(0.1,1000)); ax[1,2].legend(loc='upper left',fontsize=6.3); label(ax[1,2],'f')
for x in ax.ravel(): x.minorticks_off()
save(fig,'FigA7_nsteps',OUT); print('ok')
