"""Magnetic-domain images: real LTEM skyrmion lattice and real Kerr bubble-growth sequences."""
import sys,json,glob,numpy as np,pandas as pd,scipy.ndimage as ndi
sys.path[:0]=['.','/home/claude/work','/home/claude/work/manuscript_v2/code']
from style import *
W='/home/claude/work/data/'; OUT='../figures'
L=np.load(W+'t51_ltem.npz'); J=json.load(open(W+'t51_ltem.json'))
K=pd.read_csv(W+'t52_kerr.csv'); KF=np.load(W+'t52_kerr_fields.npz')
fig=plt.figure(figsize=(FULL,6.9),layout='constrained'); gs=fig.add_gridspec(3,3)
r0,c0,w=380,380,220
a=fig.add_subplot(gs[0,0]); a.imshow(L['A'][r0:r0+w,c0:c0+w],cmap='gray'); a.set_title('LTEM, Néel skyrmion lattice',fontsize=7); noaxis(a); imlabel(a,'a')
b=fig.add_subplot(gs[0,1]); M=L['M'][r0:r0+w,c0:c0+w]; b.imshow(M,cmap='gray'); b.contour(L['s'][r0:r0+w,c0:c0+w],[0],colors=[C['pbgia']],linewidths=0.8)
b.set_title('integrated $m_z$; PBGIA wall $s=0$',fontsize=7); noaxis(b); imlabel(b,'b')
c=fig.add_subplot(gs[0,2]); Q=L['Qdom'][L['inside']]
c.hist(Q,bins=np.linspace(0.99,1.01,41),color=C['pbgia']); c.set_xlabel('degree per skyrmion domain'); c.set_ylabel('domains'); label(c,'c')
c.text(0.97,0.97,f"$N$ = {int(L['inside'].sum())}\n$Q$ = {J['Q_per_domain_mean']:.4f}({int(round(J['Q_per_domain_sd']*1e4))})\n$\\Sigma Q$ = {J['Q_total_inner']:.1f}",transform=c.transAxes,va='top',ha='right',fontsize=6.5)
ci=c.inset_axes([0.17,0.18,0.25,0.36]); pl=np.array(J['plateau']); ci.plot(pl[:,0],pl[:,1],'o-',color=C['pbgia'],ms=2.5,lw=0.8)
ci.axhline(J['expected_count_triangular'],color=C['theory'],ls=':',lw=0.7); ci.set_xlabel('cut quantile',fontsize=5.5,labelpad=0); ci.set_title('domains vs cut',fontsize=5.5,pad=1); ci.tick_params(labelsize=5); ci.grid(False)
D=KF['1_18_D']; s=KF['1_18_s']
d=fig.add_subplot(gs[2,0]); d.imshow(D[150:870,420:900],cmap='gray',vmin=np.percentile(D,1),vmax=np.percentile(D,99)); d.set_title('Kerr difference image (raw)',fontsize=7); noaxis(d); imlabel(d,'g')
e=fig.add_subplot(gs[2,1]); e.imshow(ndi.gaussian_filter(D,2.0)[150:870,420:900],cmap='gray'); e.contour(s[150:870,420:900],[0],colors=[C['pbgia']],linewidths=0.9)
e.set_title('PBGIA wall after $\\sigma$ from the sign rule',fontsize=7); noaxis(e); imlabel(e,'h')
f=fig.add_subplot(gs[2,2])
for m,lb,col,mk,fc in (('median_otsu','median + Otsu (KerrPy-type)',C['prev'],'d',C['prev']),('gauss_ms','Gaussian + marching squares',C['conv'],'^',C['conv']),('PBGIA_bin','PBGIA, binary readout',C['pbgia'],'o',C['pbgia'])):
    for sq,ls in ((0,'-'),(1,'--')):
        q=K[(K.method==m)&(K.seq==sq)]; f.plot(q.frame-9,q.perim_um,marker=mk,ls=ls,color=col,mfc=fc,ms=3,lw=0.9,label=lb if sq==0 else None)
f.set_xlabel('pulse number'); f.set_ylabel('perimeter ($\\mu$m)'); f.legend(fontsize=6,loc='upper left'); label(f,'i')
# ---- labyrinth (row 2)
sys.path.insert(0,'/home/claude/work/manuscript_v2/code/revision2')
from dirscale import qfield,defect_list
Lb=np.load(W+'t53_labyrinth.npz'); raw=Lb['raw']; tr=Lb['truth']; vr,vc=Lb['vr'],Lb['vc']
q,_=qfield(np.stack([vr*vr,vc*vc,vr*vc]),1.5); dd=np.array(defect_list(q,margin=16))
x=fig.add_subplot(gs[1,0]); x.imshow(raw,cmap='gray'); x.set_title('Kerr, labyrinth domains',fontsize=7); noaxis(x); imlabel(x,'d')
r0_,c0_,w_=100,100,64; x.add_patch(plt.Rectangle((c0_,r0_),w_,w_,fill=False,ec='#ffe54b',lw=0.8))
y=fig.add_subplot(gs[1,1]); y.imshow(raw[r0_:r0_+w_,c0_:c0_+w_],cmap='gray',extent=(c0_-.5,c0_+w_-.5,r0_+w_-.5,r0_-.5))
for rr,cc,kk in tr:
    if r0_<=rr<r0_+w_ and c0_<=cc<c0_+w_: y.plot(cc,rr,'o',ms=6,mfc='none',mec='#ffe54b' if kk>0 else '#48dce9',mew=0.9)
for rr,cc,kk in dd:
    if r0_<=rr<r0_+w_ and c0_<=cc<c0_+w_: y.plot(cc,rr,'^' if kk>0 else 'v',ms=3.2,mfc=C['pbgia'] if kk>0 else 'white',mec=C['pbgia'],mew=0.7)
y.set_xlim(c0_-.5,c0_+w_-.5); y.set_ylim(r0_+w_-.5,r0_-.5); y.set_title('stripe ends / branch points vs PBGIA $\\pm\\frac{1}{2}$',fontsize=7); noaxis(y); imlabel(y,'e')
z=fig.add_subplot(gs[1,2]); T=pd.read_csv(W+'t54_lab_controls.csv')
for m,sg,col,mk,lb in (('PBGIA',1.5,C['pbgia'],'o','PBGIA director'),('ST',1.5,C['conv'],'^','structure tensor')):
    q_=T[(T.method==m)&(T.sigma==sg)]; z.plot(q_.radius,q_.precision,mk+'-',color=col,ms=3.5,label=lb)
q_=T[(T.method=='PBGIA')&(T.sigma==1.5)]; z.plot(q_.radius,q_.precision_random,':',color=C['prev'],label='random positions'); z.plot(q_.radius,q_.precision_sign_swapped,'--',color=C['prev'],label='signs exchanged')
z.set_xlabel('matching radius (pixels)'); z.set_ylabel('fraction of detected defects matched'); z.set_ylim(0,0.8); z.legend(fontsize=6,loc='upper left'); label(z,'f')
save(fig,'Fig9_magnetic',OUT)
