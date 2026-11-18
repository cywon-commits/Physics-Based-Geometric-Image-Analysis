import csv,numpy as np,matplotlib,matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from t1_wall import ode_profile
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4']; INK='#3a3a38'; GRID='#e4e3df'
plt.rcParams.update({'font.size':8.5,'pdf.fonttype':42,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6,'lines.linewidth':1.6,'savefig.bbox':'tight','legend.frameon':False})
rd=lambda f:list(csv.DictReader(open(f)))
def lab(ax,t): ax.set_title(t,loc='left',fontsize=9,fontweight='bold')
# ---- Fig T1
r=rd('data/t1_wall_collapse.csv'); ode=[x for x in r if x['source']=='ode']; lat=[x for x in r if x['source']=='lattice']
fig,ax=plt.subplots(1,3,figsize=(10.6,3.1),layout='constrained')
for i,k in enumerate([-1.0,0.0,1.0,2.0]):
    U,S,_=ode_profile(k); ax[0].plot(U,S,color=C[i],label=f'$\\kappa={k:g}$')
ax[0].set(xlim=(-6,6),xlabel='normal coordinate $u/\\ell$',ylabel='scalar component $s$'); ax[0].legend(loc='lower right'); lab(ax[0],'(a) Fixed-point wall profiles')
k=np.array([float(x['kappa']) for x in ode]); w=np.array([float(x['width_sd_over_ell']) for x in ode])
ax[1].plot(k,w,color=INK,lw=1.4,label='continuum ODE')
mk={'0.02':'o','0.08':'s','0.3':'^','1.0':'D','2.0':'v'}
for i,h in enumerate(['0.02','0.08','0.3','1.0','2.0']):
    q=[x for x in lat if x['h']==h]
    ax[1].plot([float(x['kappa']) for x in q],[float(x['width_sd_over_ell']) for x in q],mk[h],ms=6,mfc='none' if i>1 else C[i],mec=C[i],mew=1.3,ls='none',label=f'lattice $h/J={float(h):g}$ ($\\ell/a={1/np.sqrt(float(h)):.2g}$)')
ax[1].annotate('main setting\n$h/J=2,\\ \\kappa=0.71$',xy=(0.707,1.029),xytext=(0.9,0.62),fontsize=7.5,arrowprops=dict(arrowstyle='->',color=INK,lw=.8))
ax[1].set(xlabel='$\\kappa=D/\\sqrt{Jh}$',ylabel='wall width / $\\ell$'); ax[1].legend(fontsize=6.6,loc='upper left'); lab(ax[1],'(b) Width collapse $w=\\ell F(\\kappa)$')
sl=np.array([float(x['center_slope_over']) for x in ode])
ax[2].plot(k,sl,color=INK,label='ODE (exact)'); ax[2].plot(k,1-0.18856*k,'--',color=C[1],label='first order $1-K\\kappa$, $K=0.1886$')
ax[2].set(xlabel='$\\kappa$',ylabel="center slope $|\\theta'(0)|\\,\\ell/\\sqrt{2}$"); ax[2].legend(loc='upper right'); lab(ax[2],'(c) Chiral term as nonconservative force')
fig.savefig('figs/FigT1_wall_theory.pdf'); fig.savefig('figs/FigT1_wall_theory.png',dpi=130); plt.close(fig)
# ---- Fig T2
a=[x for x in rd('data/t2_angle.csv') if x['N']=='30']
fig,ax=plt.subplots(1,3,figsize=(10.6,3.1),layout='constrained')
for i,(D,mb) in enumerate([('0.0',1.0437),('1.0',1.0189)]):
    q=sorted([(float(x['phi_deg']),float(x['TV_per_len'])) for x in a if x['D']==D]); p,m=zip(*q)
    ax[0].plot(p,m,'o-',color=C[i],ms=5,label=f'$D/J={float(D):g}$, mean $\\bar m={mb:.4f}$'); ax[0].axhline(mb,color=C[i],lw=.8,ls=':')
ax[0].axhline(1,color=INK,lw=.8); ax[0].set(xlabel='wall normal angle $\\varphi$ (deg)',ylabel='TV per unit length, $m(\\varphi)$',xlim=(-1,46)); ax[0].legend(loc='center',fontsize=7); lab(ax[0],'(a) Planar-wall metrication, 2D')
d=[x for x in rd('data/t3_disk.csv') if x['N']=='30']; Rs=sorted({float(x['R']) for x in d})
f1=[np.mean([float(x['TV'])/float(x['P_true']) for x in d if float(x['R'])==R]) for R in Rs]
f2=[np.mean([float(x['TV'])/float(x['two_pi_rw']) for x in d if float(x['R'])==R]) for R in Rs]
cr=[np.mean([float(x['crofton'])/float(x['P_true']) for x in d if float(x['R'])==R]) for R in Rs]
ax[1].plot(Rs,f1,'o-',color=C[0],ms=5,label='TV / $2\\pi R$'); ax[1].plot(Rs,f2,'s--',color=C[2],ms=5,mfc='none',label='TV / $2\\pi\\langle r\\rangle_w$ (coarea)')
ax[1].plot(Rs,cr,'^:',color=C[1],ms=5,mfc='none',label='Crofton / $2\\pi R$'); ax[1].axhline(1.0189,color=C[0],lw=.8,ls=':')
ax[1].text(41,1.0189,' $\\bar m_2$',va='center',fontsize=7.5); ax[1].set(xscale='log');ax[1].set_xticks([2,3,5,10,20,40],['2','3','5','10','20','40']);ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());ax[1].set(xlabel='disk radius $R$ (pixels)',ylabel='ratio',ylim=(0.86,1.04)); ax[1].legend(loc='lower right',fontsize=7); lab(ax[1],'(b) Disks: curvature and metrication')
s=rd('data/t5_spheres.csv'); Rs=sorted({float(x['R']) for x in s})
g1=[np.mean([float(x['TV'])/float(x['A_true']) for x in s if float(x['R'])==R]) for R in Rs]
g2=[np.mean([float(x['TV'])/float(x['four_pi_r2w']) for x in s if float(x['R'])==R]) for R in Rs]
ax[2].plot(Rs,g1,'o-',color=C[0],ms=5,label='TV / $4\\pi R^2$'); ax[2].plot(Rs,g2,'s--',color=C[2],ms=5,mfc='none',label='TV / $4\\pi\\langle r^2\\rangle_w$')
ax[2].axhline(1.0160,color=C[0],lw=.8,ls=':'); ax[2].text(20.5,1.016,' $\\bar m_3$',va='center',fontsize=7.5)
ax[2].set(xscale='log');ax[2].set_xticks([2,3,4,6,10,20],['2','3','4','6','10','20']);ax[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());ax[2].set(xlabel='sphere radius $R$ (voxels)',ylabel='ratio',ylim=(0.58,1.05)); ax[2].legend(loc='lower right',fontsize=7); lab(ax[2],'(c) Spheres, 3D')
fig.savefig('figs/FigT2_metrication.pdf'); fig.savefig('figs/FigT2_metrication.png',dpi=130); plt.close(fig)
# ---- Fig T3
rules=rd('data/t4_local_rules.csv'); feats=['single','domino','L-tromino','2x2','line1x9','plus','3x3']; hs=['1.0','2.0','3.0','4.5']
zin={'single':[0],'domino':[1,1],'L-tromino':[2,1,1],'2x2':[2]*4,'line1x9':[1]+[2]*7+[1],'plus':[4,1,1,1,1],'3x3':[2,3,2,3,4,3,2,3,2]}
fig,ax=plt.subplots(1,2,figsize=(10.2,3.3),layout='constrained',gridspec_kw={'width_ratios':[1,1.15]})
M=np.array([[float(next(x['fraction_kept'] for x in rules if x['h']==h and x['feature']==f and x['hole']=='False')) for h in hs] for f in feats])
ax[0].imshow(M,cmap='Blues',vmin=-.15,vmax=1.15,aspect='auto'); ax[0].grid(False)
for i,f in enumerate(feats):
    for j,h in enumerate(hs):
        hz=float(h); worst=min(zin[f]); pred='keep' if hz>4-2*worst else ('marginal' if hz==4-2*worst else 'erase')
        ax[0].text(j,i,f'{M[i,j]:.2f}\n{pred}',ha='center',va='center',fontsize=6.8,color='white' if M[i,j]>.6 else INK)
ax[0].set_xticks(range(4),[f'$h/J={float(h):g}$' for h in hs]); ax[0].set_yticks(range(len(feats)),feats)
lab(ax[0],'(a) Kept fraction vs Ising-limit rule (worst site)')
n=rd('data/t4_noise.csv'); ps=['0.0','0.002','0.01','0.05']; xs=[0.0005,0.002,0.01,0.05]
series=[('TV_N30','PBGIA TV, $N=30$',1.0189,C[0],'o'),('TV_N10','PBGIA TV, $N=10$',1.0189,C[2],'s'),('TV_N0','EDT init only ($N=0$)',1.0189,C[3],'D'),('crofton','Crofton, raw mask',1,C[1],'^'),('crofton_median3','Crofton after 3×3 median',1,C[4],'v')]
for key,lb,mb,c,mk_ in series:
    y=[np.mean([abs(float(x[key])/mb/float(x['P_true'])-1)*100 for x in n if x['noise']==p]) for p in ps]
    ax[1].plot(xs,y,mk_+'-',color=c,ms=5,label=lb)
ax[1].set(xscale='log',yscale='log',xlabel='salt-and-pepper flip rate (0 plotted at $5\\times10^{-4}$)',ylabel='perimeter MARE (%)'); ax[1].legend(fontsize=7,loc='upper left'); lab(ax[1],'(b) Stadium perimeter under noise (12 shapes)')
fig.savefig('figs/FigT3_filter.pdf'); fig.savefig('figs/FigT3_filter.png',dpi=130); plt.close(fig)
print('ok')
