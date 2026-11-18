import sys,json,numpy as np,pandas as pd
sys.path[:0]=['.','../code/revision2']
from style import *
from t1_wall import ode_profile
from readout import MBAR,COEF,m_of_k
OUT='../figures'; D='../data/revision2'
# ---------------- Fig 2: wall theory
r=pd.read_csv(f'{D}/t1_wall_collapse.csv'); ode=r[r.source=='ode']; lat=r[r.source=='lattice']
fig,ax=plt.subplots(1,3,figsize=(FULL,2.15),layout='constrained')
cols=[C['pbgia'],C['conv'],C['smooth'],C['init']]
for i,k in enumerate([-1.0,0.0,1.0,2.0]):
    U,S,_=ode_profile(k); ax[0].plot(U,S,color=cols[i],ls=['--','-','-.',':'][i],label=f'$\\kappa={k:g}$')
ax[0].set(xlim=(-6,6),xlabel='$u/\\ell$',ylabel='$s$'); ax[0].legend(loc='lower right',handlelength=1.6)
ax[1].plot(ode.kappa,ode.width_sd_over_ell,color=C['theory'],lw=1.2,label='continuum ODE',zorder=1)
spec=[(0.02,'o',C['pbgia'],True),(0.08,'s',C['smooth'],True),(0.3,'^',C['conv'],False),(1.0,'D',C['init'],False),(2.0,'v',C['prev'],False)]
for h,mk,c,fill in spec:
    q=lat[np.isclose(lat.h,h)]
    ax[1].plot(q.kappa,q.width_sd_over_ell,mk,ms=4,mfc=c if fill else 'white',mec=c,mew=1,ls='none',label=f'$h/J$={h:g}')
ax[1].set(xlabel='$\\kappa=cD/\\sqrt{Jh}$',ylabel='$w/\\ell$',ylim=(0.45,1.35))
ax[1].legend(loc='lower left',bbox_to_anchor=(0.0,1.01),ncol=3,borderaxespad=0,handletextpad=0.1,columnspacing=0.5,fontsize=6.5)
ax[1].annotate('main setting',xy=(0.707,1.029),xytext=(-1.9,1.2),fontsize=7,arrowprops=dict(arrowstyle='->',lw=0.6,color=C['ink']))
k=ode.kappa.values; sl=ode.center_slope_over.values
ax[2].plot(k,sl,color=C['theory'],label='ODE'); ax[2].plot(k,1-2*np.sqrt(2)/15*k,'--',color=C['conv'],label='$1-K\\kappa$')
ax[2].set(xlabel='$\\kappa$',ylabel="$|\\theta'(0)|\\,\\ell/\\sqrt{2}$"); ax[2].legend(loc='upper right')
label(ax[0],'a');label(ax[2],'c');label(ax[1],'b',dx=-0.12,dy=1.0)
save(fig,'Fig2_wall_theory',OUT)
# ---------------- Fig 3: metrication planar + curvature
a2=pd.read_csv(f'{D}/t2_angle.csv'); a2=a2[a2.N==30]
cf=pd.read_csv(f'{D}/t10_curvfit.csv')
fig,ax=plt.subplots(1,3,figsize=(FULL,2.15),layout='constrained')
for Dv,c,mk,mb in ((0.0,C['prev'],'s',1.0437),(1.0,C['pbgia'],'o',MBAR[2])):
    q=a2[a2.D==Dv].sort_values('phi_deg'); ax[0].plot(q.phi_deg,q.TV_per_len,mk,color=c,ms=4,ls='-',lw=0.8,label=f'$D/J={Dv:g}$')
    ax[0].axhline(mb,color=c,lw=0.7,ls=':')
ax[0].axhline(1,color=C['ink'],lw=0.6)
ax[0].set(xlabel='wall normal angle $\\varphi$ (deg)',ylabel='$m(\\varphi)$',xlim=(-2,47),xticks=[0,15,30,45]); ax[0].legend(loc='lower center',bbox_to_anchor=(0.5,0.08))
for j,d in enumerate((2,3)):
    q=cf[cf.d==d].groupby('k')['uniform'].agg(['mean','std']).reset_index()
    kk=np.linspace(0,q.k.max()*1.02,200)
    ax[j+1].plot(kk,m_of_k(kk,d),color=C['theory'],lw=1.1,label='fit $m(\\kappa)$')
    ax[j+1].errorbar(q.k,q['mean'],yerr=q['std'],fmt='o',color=C['pbgia'],ms=3.5,capsize=1.5,elinewidth=0.7,label='disks' if d==2 else 'spheres')
    ax[j+1].plot([0],[MBAR[d]],'*',color=C['conv'],ms=8,label='planar walls',zorder=5)
    ax[j+1].set(xlabel=('curvature $a\\kappa=a/R$' if d==2 else 'curvature $2aH=2a/R$'),ylabel='$I_{\\rm TV}/\\mathcal{A}_{\\rm true}$')
    ax[j+1].legend(loc='lower left')
for a,l in zip(ax,'abc'): label(a,l)
save(fig,'Fig3_metrication',OUT)
# ---------------- Fig 4: filtering
rules=pd.read_csv(f'{D}/t4_local_rules.csv'); rules=rules[~rules.hole]
feats=['single','domino','L-tromino','2x2','line1x9','plus','3x3']; names=['single','domino','L-tromino','2×2','1×9 line','plus','3×3']
hs=[1.0,2.0,3.0,4.5]; zmin={'single':0,'domino':1,'L-tromino':1,'2x2':2,'line1x9':1,'plus':1,'3x3':2}
fig,ax=plt.subplots(1,2,figsize=(FULL,2.45),layout='constrained',gridspec_kw={'width_ratios':[1,1.15]})
M=np.array([[rules[(rules.h==h)&(rules.feature==f)].fraction_kept.iloc[0] for h in hs] for f in feats])
ax[0].imshow(M,cmap='Blues',vmin=-0.15,vmax=1.15,aspect='auto'); noaxis(ax[0])
for i,f in enumerate(feats):
    for j,h in enumerate(hs):
        thr=4-2*zmin[f]; pred='keep' if h>thr else ('marg.' if h==thr else 'erase')
        ok=(pred=='keep' and M[i,j]==1) or (pred=='erase' and M[i,j]==0) or pred=='marg.'
        ax[0].text(j,i,f'{M[i,j]:.2f}\n{pred}'+('' if ok else ' ×'),ha='center',va='center',fontsize=6.5,color='white' if M[i,j]>.6 else C['ink'])
ax[0].set_xticks(range(4),[f'{h:g}' for h in hs]); ax[0].set_yticks(range(7),names); ax[0].set_xlabel('$h/J$')
ax[0].tick_params(length=0)
n=pd.read_csv(f'{D}/t4_noise.csv'); n['shape']=n.R.astype(str)+'_'+n.theta.astype(str)
ps=[0.0,0.002,0.01,0.05]; xs=[0.0005,0.002,0.01,0.05]
ser=[('TV_N30','PBGIA, $N=30$',MBAR[2],'pbgia'),('TV_N0','distance-transform init ($N=0$)',MBAR[2],'init'),('crofton','Crofton, raw',1,'conv'),('crofton_median3','3×3 median + Crofton',1,'smooth')]
for key,lb,mb,c in ser:
    mu=[];se=[]
    for p in ps:
        e=(n[n.noise==p].assign(e=lambda x:(x[key]/mb/x.P_true-1).abs()*100).groupby('shape').e.mean())
        mu.append(e.mean()); se.append(e.std(ddof=1)/np.sqrt(len(e)))
    ax[1].errorbar(xs,mu,yerr=se,fmt=MK[c]+'-',color=C[c],ms=4,capsize=1.5,elinewidth=0.7,label=lb)
ax[1].set(xscale='log',yscale='log',xlabel='salt-and-pepper flip rate',ylabel='perimeter MARE (%)')
ax[1].set_xticks(xs,['0','0.002','0.01','0.05']); ax[1].minorticks_off(); legend_top(ax[1],ncol=2)
label(ax[0],'a',dx=-0.25); label(ax[1],'b',dx=-0.12,dy=1.0)
save(fig,'Fig4_filter',OUT)
# ---------------- Fig 5: frozen benchmark
new=pd.read_csv(f'{D}/frozen_rescore/weighted_readouts_all.csv'); new=new[new.split=='test']
new['error_percent']=100*(new.uniform_curv/new.truth-1); new['method']='PBGIA'
old=pd.read_csv('/home/claude/frozen/final_pbgia_study/output/geometry_errors_long.csv'); old=old[old.split=='test']
mp={2:{'Crofton_train_c':'conv','Gaussian1_Crofton_raw':'smooth','selected':'prev'},3:{'marching_cubes_train_c':'conv','Gaussian1_marching_cubes_raw':'smooth','selected':'prev'}}
L0=dict(pbgia='PBGIA, curvature-corrected',conv='Crofton (2D) / marching cubes (3D), train-calibrated',smooth='Gaussian ($\\sigma$=1) + same, raw',prev='previous PBGIA readout')
lab={2:L0,3:L0}
fams=['circle','ellipse','stadium','ring','near_touch','pair']; fam3={'circle':'sphere','ellipse':'ellipsoid','stadium':'capsule','ring':'torus','near_touch':'near_touch','pair':'pair'}
fig,ax=plt.subplots(2,2,figsize=(FULL,4.5),layout='constrained',gridspec_kw={'width_ratios':[1.6,1]})
for row,d in enumerate((2,3)):
    tabs={'pbgia':new[new.dim==d][['unit','family','noise','error_percent']]}
    for m,c in mp[d].items(): tabs[c]=old[(old.dim==d)&(old.method==m)][['unit','family','noise','error_percent']]
    fl=fams if d==2 else [fam3[f] for f in fams]
    for i,c in enumerate(['pbgia','conv','smooth','prev']):
        t=tabs[c][tabs[c].noise==0]; off=(i-1.5)*0.18
        for j,f in enumerate(fl):
            e=t[t.family==f].error_percent.abs().values
            ax[row,0].plot(np.full(len(e),j+off),e,MK[c],color=C[c],ms=2.2,alpha=0.45,mew=0)
            ax[row,0].plot(j+off,e.mean(),MK[c],color=C[c],ms=4.5,mec='white',mew=0.4,label=lab[d][c] if (j==0 and d==2) else None)
        mu=[t2.assign(a=t2.error_percent.abs()).groupby('unit').a.mean().mean() for t2 in [tabs[c][tabs[c].noise==p] for p in (0,0.002,0.01)]]
        ax[row,1].plot([0.0005,0.002,0.01],mu,MK[c]+'-',color=C[c],ms=4)
    ax[row,0].set_xticks(range(6),[f.replace('near_touch','near touch') for f in fl],rotation=20,ha='right'); ax[row,0].set_yscale('log'); ax[row,0].set_ylim(0.005,12)
    ax[row,0].set_ylabel(('2D perimeter' if d==2 else '3D surface area')+'\n|error| (%)'); ax[row,0].grid(axis='x',visible=False)
    ax[row,1].set(xscale='log',yscale='log',xlabel='flip rate',ylabel='MARE (%)'); ax[row,1].set_xticks([0.0005,0.002,0.01],['0','0.002','0.01']); ax[row,1].minorticks_off()
h_,l_=ax[0,0].get_legend_handles_labels(); fig.legend(h_,l_,loc='outside upper center',ncol=2,columnspacing=1.0,handletextpad=0.2)
label(ax[0,0],'a');label(ax[0,1],'b');label(ax[1,0],'c');label(ax[1,1],'d')
save(fig,'Fig5_frozen_benchmark',OUT)
# ---------------- Fig 6: stadium (quant panels only; images in Fig 1)
st=pd.read_csv(f'{D}/t13_stadium.csv'); ori=pd.read_csv('../data/stadium_orientation.csv')
fig,ax=plt.subplots(1,2,figsize=(FULL,2.2),layout='constrained')
ax[0].plot([0,180],[0,180],color=C['theory'],lw=0.8,label='identity')
ax[0].plot(ori.theta_true_deg,ori.theta_true_deg+ori.axial_error_deg,'o',color=C['pbgia'],ms=4,label='wall-weighted covariance axis')
ax[0].set(xlabel='imposed axis angle (deg)',ylabel='measured axis (deg)',xlim=(-5,180),ylim=(-5,185),xticks=[0,45,90,135,180],yticks=[0,45,90,135,180]); ax[0].legend(loc='upper left')
for key,c,lb in (('B_curv','pbgia','PBGIA, curvature-corrected'),('crofton','conv','Crofton, raw')):
    e=100*(st[key]/st.P_true-1); ax[1].plot(st.R+(0.25 if c=='conv' else -0.25),e,MK[c],color=C[c],ms=4,ls='none',label=lb)
ax[1].axhline(0,color=C['theory'],lw=0.8)
ax[1].set(xlabel='cap radius $R$ (pixels), $L=2R$',ylabel='perimeter error (%)',xticks=range(9,26,2)); legend_top(ax[1],ncol=2)
label(ax[0],'a');label(ax[1],'b',dx=-0.12,dy=1.0)
save(fig,'Fig6_stadium',OUT)
print('done')
