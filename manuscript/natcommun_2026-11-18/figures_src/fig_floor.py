"""Rasterization floor and grayscale source (draft figure)."""
import sys,numpy as np,pandas as pd
sys.path.insert(0,'/home/claude/work/manuscript_v2/figures_src')
from style import *
from matplotlib.ticker import NullFormatter,FixedLocator
W='/home/claude/work/data/'; OUT='/home/claude/work/manuscript_v2/figures'
S=pd.read_csv(W+'t41_disk_sweep_summary.csv'); I=pd.read_csv(W+'t31_infofloor.csv').groupby('R').rel_std.mean()
P=pd.read_csv(W+'t39_pose_all.csv'); F=pd.read_csv(W+'t42_gray_fresh.csv')
fig,ax=plt.subplots(1,3,figsize=(FULL,2.35),gridspec_kw=dict(wspace=0.42,width_ratios=[1,1,1.25]))
# (a) disk pose spread vs radius
a=ax[0]; R=S.R.values; rr=np.linspace(3,50,100)
a.plot(rr,100/(2*np.pi*rr**2),color=C['theory'],lw=0.9,label=r'information bound $1/2\pi R^2$')
a.plot(I.index,I.values,'x',color=C['theory'],ms=4)
a.plot(R,S.std_crofton_bin,marker=MK['prev'],color=C['prev'],lw=0.9,label='Crofton (binary)')
a.plot(R,S.std_PBGIA_bin,marker='o',color=C['pbgia'],label='PBGIA (binary)')
a.plot(R,S.std_PBGIA_gray,marker='o',mfc='white',ls='--',color=C['pbgia'],label='PBGIA (gray)')
a.plot(R,S.std_ms_gray,marker=MK['smooth'],mfc='white',ls='--',color=C['smooth'],lw=0.9,label='marching sq. (gray)')
a.set_xscale('log');a.set_yscale('log');a.set_xlabel('disk radius $R$ (px)');a.set_ylabel('pose spread of $P$ (%)')
a.set_xticks([3,5,10,20,40]);a.set_xticklabels(['3','5','10','20','40']);a.set_ylim(0.008,400)
a.xaxis.set_minor_formatter(NullFormatter()); a.legend(loc='upper right',ncol=1,borderaxespad=0.2,handletextpad=0.3,fontsize=5.8,labelspacing=0.25); label_top(a,'a')
# (b) quantization law
b=ax[1]
for m,mk,fc,ls in (('PBGIA_bin','o',C['pbgia'],'-'),('PBGIA_gray','o','white','--')):
    g=P.groupby(['family','inst']).agg(T=('truth','first'),k2=('k2','first'),sd=(m,'std'))
    b.plot(g.k2,g.sd**2,mk,ms=2.6,mfc=fc,mec=C['pbgia'],mew=0.6,ls='none')
    c=np.sum(g.sd**2*g.k2)/np.sum(g.k2**2); x=np.array([0.1,4]); b.plot(x,c*x,ls,color=C['pbgia'],lw=0.9)
    cs=locals().setdefault('cs',[]); cs.append(c)
x=np.array([0.1,4]); b.plot(x,2/(9*np.pi)*x,':',color=C['theory'],lw=0.9); pass
b.set_xscale('log');b.set_yscale('log');b.set_xlabel(r'$\oint\kappa^2\,ds$ (px$^{-1}$)');b.set_ylabel(r'var$_{\rm pose}(P)=c\oint\kappa^2ds$ (px$^2$)')
b.set_xlim(0.1,4); b.set_ylim(2e-4,6); b.xaxis.set_major_locator(FixedLocator([0.1,0.3,1,3])); b.set_xticklabels(['0.1','0.3','1','3']); b.xaxis.set_minor_formatter(NullFormatter())
from matplotlib.lines import Line2D
b.legend(handles=[Line2D([],[],marker='o',ls='-',color=C['pbgia'],ms=3,label=f'binary, $c$={cs[0]:.3f}'),Line2D([],[],marker='o',mfc='white',ls='--',color=C['pbgia'],ms=3,label=f'gray, $c$={cs[1]:.3f}'),Line2D([],[],ls=':',color=C['theory'],label=r'indep. crossings, $c=2/9\pi$')],loc='upper left',ncol=1,borderaxespad=0.2,fontsize=5.8,handletextpad=0.3,labelspacing=0.25); label_top(b,'b')
# (c) gray benchmark on fresh geometries
c=ax[2]; nc=F[F.family!='near_touch'].copy(); cond=[('flip',0.0,'clean'),('flip',0.002,'flip\n0.2%'),('flip',0.01,'flip\n1%'),('gauss',0.05,'gray\n$\\pm$.05'),('gauss',0.1,'gray\n$\\pm$.10')]
meth=[('PBGIA_gray','PBGIA (gray)',C['pbgia'],None),('PBGIA_bin','PBGIA (binary)',C['pbgia'],'////'),('ms_gray','marching sq. (gray)',C['smooth'],None),('gms05_gray',r'Gauss $\sigma$=0.5 (gray)',C['conv'],None)]
YM=1.2; wd=0.2
for j,(m,lab,col,hat) in enumerate(meth):
    v=[ (100*(g[m]/g.truth-1).abs()).groupby(g.family).mean().mean() for k,n,_ in cond for g in [nc[(nc.kind==k)&(nc.noise==n)]]]
    xs=np.arange(len(cond))+(j-1.5)*wd
    c.bar(xs,np.minimum(v,YM),wd,color='white' if hat else col,edgecolor=col,hatch=hat,lw=0.6,label=lab)
    for x_,v_ in zip(xs,v):
        if v_>YM: c.text(x_,YM*0.985,f'{v_:.0f}',rotation=90,ha='center',va='top',fontsize=5.5,color='white')
c.set_xticks(range(len(cond)));c.set_xticklabels([t for *_,t in cond],fontsize=6.3);c.set_ylim(0,YM);c.set_ylabel('MARE (%)');c.grid(axis='x',visible=False)
c.legend(loc='lower left',bbox_to_anchor=(0.0,1.01),ncol=2,borderaxespad=0,handletextpad=0.3,columnspacing=0.7,fontsize=6.3); label_top(c,'c')
save(fig,'FigA4_floor_gray',OUT)
