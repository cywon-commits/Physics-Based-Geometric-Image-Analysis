"""Director from the continuous source vs the five-cut stack vs the structure tensor (appendix figure)."""
import sys,numpy as np,pandas as pd
sys.path.insert(0,'.')
from style import *
W='/home/claude/work/data/'; OUT='../figures'
L=pd.read_csv(W+'t47_lic_sigma.csv'); R=pd.read_csv(W+'t47_real_sigma.csv')
L2=pd.read_csv(W+'t50_lic.csv').query("method=='cont_w1'").rename(columns={'n':'n_plus'}).assign(n_minus=0); L=pd.concat([L,L2.assign(method='PBGIA_contw')])
R=pd.concat([R,pd.read_csv(W+'t50_real.csv').query("method=='cont_w1'").assign(method='PBGIA_contw')])
MS=[('PBGIA_cuts','PBGIA, five cuts',C['pbgia'],'o','-',C['pbgia']),('PBGIA_cont','PBGIA, continuous',C['pbgia'],'o','--','white'),('PBGIA_contw','PBGIA, continuous, $|\\nabla s|$-weighted',C['smooth'],'D',':','white'),('ST','structure tensor',C['conv'],'^','-',C['conv'])]
fig,ax=plt.subplots(2,2,figsize=(FULL,4.3),layout='constrained')
a=ax[0,0]
for m,lb,col,mk,ls,fc in MS:
    q=L[L.method==m]; a.plot(q.sigma,(q.n_plus+q.n_minus).clip(lower=0.5),marker=mk,ls=ls,color=col,mfc=fc,ms=3.5,label=lb)
a.axhline(4,color=C['prev'],lw=0.6,ls=':'); a.set(xscale='log',yscale='log',xlabel='Gaussian window $\\sigma$ (pixels)',ylabel='detected defects',ylim=(0.4,1000))
a.set_xticks([2,4,8,16,32],['2','4','8','16','32']); a.minorticks_off(); h_,l_=a.get_legend_handles_labels(); fig.legend(h_,l_,loc='outside lower center',ncol=4,fontsize=6.6); label(a,'a')
a.text(0.03,0.04,'LIC texture, four prescribed defects',transform=a.transAxes,fontsize=6.5)
b=ax[0,1]; g=R[R.image=='Curved strata (synthetic)']
for m,lb,col,mk,ls,fc in MS:
    q=g[g.method==m]; b.plot(q.sigma,q.gt_err,marker=mk,ls=ls,color=col,mfc=fc,ms=3.5,label=lb)
b.set(xscale='log',yscale='log',xlabel='Gaussian window $\\sigma$ (pixels)',ylabel='error to analytic tangent ($^\\circ$)'); b.set_xticks([2,4,8,16],['2','4','8','16']); b.set_yticks([0.2,0.5,1,2],['0.2','0.5','1','2']); b.minorticks_off(); label(b,'b')
b.text(0.03,0.04,'curved strata',transform=b.transAxes,fontsize=6.5)
c=ax[1,0]; s8=R[R.sigma==8].pivot(index='image',columns='method',values='local_rot_err'); s8=s8.sort_values('ST')
names={'Building array (synthetic)':'building array','CC0 brick wall':'brick (CC0)','CC0 grass texture':'grass (CC0)','CC0 retina vessel contrast':'retina (CC0)',
       'Collagen-like fibers (synthetic)':'collagen-like fibers','Curved strata (synthetic)':'curved strata','Leaf veins (synthetic)':'leaf veins','Textile weave (synthetic)':'textile weave','Wood grain (synthetic)':'wood grain'}
y=np.arange(len(s8))
for (m,lb,col,mk,ls,fc),dy in zip(MS,(-0.27,-0.09,0.09,0.27)): c.plot(s8[m],y+dy,mk,color=col,mfc=fc,ms=4,ls='none',label=lb)
c.set_yticks(y,[names[n] for n in s8.index],fontsize=6.5); c.set_xscale('log'); c.set_xlabel('local rotation error at $\\sigma=8$ ($^\\circ$)'); c.grid(axis='y',visible=False); label(c,'c')
d=ax[1,1]; A=271*271
for img,lb,shade in (('CC0 grass texture','grass',1.0),('Wood grain (synthetic)','wood',0.45)):
    for m,_,col,mk,ls,fc in MS:
        q=R[(R.image==img)&(R.method==m)]; d.plot(q.sigma,q.n_defects.clip(lower=0.5),marker=mk,ls=ls,color=col,mfc=fc,ms=3.2,alpha=shade)
d.text(9,260,'grass (dark)',fontsize=6.5); d.text(2.1,7,'wood grain (pale)',fontsize=6.5)
ss=np.array([2,16]); d.plot(ss,A/(4*np.pi*ss**2),color=C['theory'],lw=1.0,label='random field $A/4\\pi\\sigma^2$')
d.set(xscale='log',yscale='log',xlabel='Gaussian window $\\sigma$ (pixels)',ylabel='defects in ROI',ylim=(0.4,5000)); d.set_xticks([2,4,8,16],['2','4','8','16']); d.minorticks_off(); d.legend(loc='lower left',fontsize=6.3); label(d,'d')
save(fig,'FigA6_contdir',OUT)
