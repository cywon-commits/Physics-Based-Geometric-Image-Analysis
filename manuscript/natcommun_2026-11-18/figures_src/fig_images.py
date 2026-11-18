import sys,numpy as np,pandas as pd,scipy.ndimage as ndi
sys.path[:0]=['.','../code/revision2','../code']
from style import *
from matplotlib.collections import LineCollection
from readout import boundary,tv_density,curvature,m_of_k
import new_quantitative_studies as nq
import multithreshold_directors as mt
from skimage.measure import marching_cubes
OUT='../figures'
def sfield(ax,s): return ax.imshow(s,interpolation='nearest',**SFIELD)
# ---------------- Fig 1 concept
mask,_=nq.stadium(n=110,R=15,L=30,theta=30); vr,vc,s=nq.generate_field(mask,nsteps=30); s=s.astype(float)
fig,ax=plt.subplots(1,4,figsize=(FULL,1.85),layout='constrained')
ax[0].imshow(np.where(mask,1,-1),cmap='gray',vmin=-1.6,vmax=1.2,interpolation='nearest'); ax[0].set_title('source $f=\\pm1$')
im=sfield(ax[1],s); ax[1].set_title('generated $s$')
wr,wc=np.nonzero(tv_density(s)>0.3); kk=np.argmin((wr-30)**2+(wc-85)**2); wd=18; r0,c0=int(wr[kk])-wd//2,int(wc[kk])-wd//2; ax[1].add_patch(plt.Rectangle((c0-.5,r0-.5),wd,wd,fill=False,ec='black',lw=0.8))
sfield(ax[2],s[r0:r0+wd,c0:c0+wd]); yy,xx=np.mgrid[:wd,:wd]; V=vr[r0:r0+wd,c0:c0+wd]; U=vc[r0:r0+wd,c0:c0+wd]; sel=np.hypot(U,V)>0.25
ax[2].quiver(xx[sel],yy[sel],U[sel],V[sel],angles='xy',scale_units='xy',scale=0.9,width=0.012,color='black')
ax[2].set_title('wall zoom: $\\mathbf{v}$')
w=tv_density(s)/m_of_k(curvature(s),2); ax[3].imshow(w,cmap='Greys',vmin=0,vmax=np.percentile(w,99.5),interpolation='nearest')
P=2*np.pi*15+60; ax[3].set_title('readout $w/m(\\kappa)$')
ax[3].text(0.03,0.04,f'$\\hat P$={w.sum():.1f}\nexact $P$={P:.1f}',transform=ax[3].transAxes,fontsize=7,va='bottom')
for a,l in zip(ax,'abcd'): noaxis(a); imlabel(a,l)
cb=fig.colorbar(im,ax=ax[1:3],shrink=0.8,ticks=[-1,0,1],pad=0.01); cb.set_label('$s$',labelpad=1); cb.ax.tick_params(labelsize=7)
save(fig,'Fig1_concept',OUT)
# ---------------- Fig 7 topology
n=154; y,x=np.mgrid[:n,:n]; x=x-(n-1)/2; y=y-(n-1)/2; rr=np.hypot(x,y)
cases2=[('one disk',rr<25),('two disks',((x-33)**2+y*y<18**2)|((x+33)**2+y*y<18**2)),('annulus',(rr>17)&(rr<31)),('disk + annulus',((x+35)**2+y*y<18**2)|((np.hypot(x-33,y)>10)&(np.hypot(x-33,y)<23)))]
t2=pd.read_csv('../data/2d_spin_topology.csv'); t3=pd.read_csv('../data/3d_synthetic_validation.csv')
fig,ax=plt.subplots(2,4,figsize=(FULL,3.05),layout='constrained')
for j,(nm,m) in enumerate(cases2):
    s2=nq.generate_field(m,nsteps=30)[2]; im=sfield(ax[0,j],s2[20:-20,:])
    r=t2[t2.case==nm.replace('annulus','ring') if nm=='disk + annulus' else t2.case==nm].iloc[0]
    ax[0,j].set_title(f'{nm}\n$Q_2$={r.Q2_solid_angle:.0f},  $\\chi$={int(r.input_euler)}')
chis={'sphere':1,'torus':0,'pair':2,'porous':-5}
for j,kind in enumerate(['sphere','torus','pair','porous']):
    mk,_,_=nq.shape3(68,kind); nn=nq.generate3(mk,30); s3=nn[...,3]; k=68//2
    sfield(ax[1,j],s3[k])
    if kind=='porous':
        rem=mk[k]&~(s3[k]>0); add=~mk[k]&(s3[k]>0)
        ry,rx=np.nonzero(rem); ay,ax_=np.nonzero(add)
        ax[1,j].plot(rx,ry,'s',ms=1.6,mfc='none',mec='black',mew=0.5); ax[1,j].plot(ax_,ay,'s',ms=1.6,mfc='white',mec='white',mew=0.5)
    q=int(t3[t3.kind==kind].S3_integer_degree.iloc[0])
    ax[1,j].set_title(f'{kind}\n$Q_3$={q},  $\\chi(\\{{s>0\\}})$={chis[kind]}')
for a,l in zip(ax.ravel(),'abcdefgh'): noaxis(a); imlabel(a,l)
cb=fig.colorbar(im,ax=ax,shrink=0.6,ticks=[-1,0,1],pad=0.01); cb.set_label('$s$',labelpad=1); cb.ax.tick_params(labelsize=7)
save(fig,'Fig8_topology',OUT)
# ---------------- Fig 8 directors (3 synthetic cases)
cases=mt.construct_cases()
def directors(ax,img,res,color,step=8,max_length=11,gate=.25):
    H,W=img.shape; yy,xx=np.mgrid[step//2:H:step,step//2:W:step]
    a=res['tangent'][yy,xx]; coh=res['coherence'][yy,xx]; amp=res['gap'][yy,xx]
    mag=np.clip(amp/max(float(np.percentile(res['gap'],95)),1e-10),0,1); ok=(coh>=gate)&(mag>=.07)
    L=1.4+max_length*mag[ok]; u=np.cos(np.deg2rad(a[ok])); v=-np.sin(np.deg2rad(a[ok])); xc=xx[ok]; yc=yy[ok]
    seg=np.stack([np.stack([xc-.5*L*u,yc-.5*L*v],-1),np.stack([xc+.5*L*u,yc+.5*L*v],-1)],1)
    ax.imshow(img,cmap='gray',vmin=-0.6,vmax=1.6); ax.add_collection(LineCollection(seg,colors=color,linewidths=0.8))
fig,ax=plt.subplots(3,3,figsize=(FULL*0.78,FULL*0.80),layout='constrained')
res={}
for i,idx in enumerate([0,1,2]):
    nm,im_,_,_=cases[idx]; p=mt.multi_field(im_); res[idx]=p
    ax[i,0].imshow(im_,cmap='gray',vmin=0,vmax=1)
    g=ax[i,1].imshow(p['gap']/np.percentile(p['gap'],99),cmap='magma',vmin=0,vmax=1)
    directors(ax[i,2],im_,p,C['pbgia'],step=11,max_length=12)
    ax[i,0].set_ylabel(nm.replace(' (synthetic)','\n(synthetic)'),fontsize=7)
ax[0,0].set_title('input'); ax[0,1].set_title('anisotropy $(\\lambda_1-\\lambda_2)$'); ax[0,2].set_title('PBGIA director')
for a,l in zip(ax.ravel(),'abcdefghi'):
    yl=a.get_ylabel(); noaxis(a); a.set_ylabel(yl,fontsize=7); imlabel(a,l)
cb=fig.colorbar(g,ax=ax[:,1],shrink=0.35,ticks=[0,0.5,1],pad=0.01); cb.set_label('normalized',labelpad=1); cb.ax.tick_params(labelsize=7)
save(fig,'FigA1_directors',OUT)
# ---------------- Appendix A1 deskew
im_=cases[6][1]; p=mt.multi_field(im_); rot=mt.img_rotate(im_,10); pro=mt.multi_field(rot)
roi=mt.roi_center(im_.shape,.67); a0,_=mt.axis_mean(p['tangent'],p['gap'],roi); a1,_=mt.axis_mean(pro['tangent'],pro['gap'],roi)
inf=float(mt.angle_delta(a1,a0)); cor=mt.img_rotate(rot,-inf); print('deskew inferred',inf)
fig,ax=plt.subplots(1,4,figsize=(FULL,1.75),layout='constrained')
ax[0].imshow(im_,cmap='gray',vmin=0,vmax=1);ax[0].set_title('CC0 brick texture')
ax[1].imshow(rot,cmap='gray',vmin=0,vmax=1);ax[1].set_title('rotated by $+10^\\circ$')
directors(ax[2],rot,pro,C['pbgia'],step=8,max_length=9); ax[2].set_title('PBGIA director')
ax[3].imshow(cor,cmap='gray',vmin=0,vmax=1);ax[3].set_title(f'corrected by ${-inf:+.2f}^\\circ$')
for a,l in zip(ax,'abcd'): noaxis(a); imlabel(a,l)
save(fig,'FigA2_deskew',OUT)
# ---------------- Appendix A2 gallery (two parts)
for part,idxs in ((1,range(0,5)),(2,range(5,9))):
    fig,ax=plt.subplots(len(idxs),4,figsize=(FULL,1.6*len(idxs)),layout='constrained')
    for i,idx in enumerate(idxs):
        nm,im_,_,_=cases[idx]; p=mt.multi_field(im_); st=mt.structure_tensor(im_)
        ax[i,0].imshow(im_,cmap='gray',vmin=0,vmax=1); g=ax[i,1].imshow(p['gap']/np.percentile(p['gap'],99),cmap='magma',vmin=0,vmax=1)
        directors(ax[i,2],im_,p,C['pbgia']); directors(ax[i,3],im_,st,C['conv'])
        ax[i,0].set_ylabel(nm.replace(' (synthetic)','\n(synthetic)').replace('CC0 ','CC0\n'),fontsize=7)
    ax[0,0].set_title('input');ax[0,1].set_title('PBGIA anisotropy');ax[0,2].set_title('PBGIA director');ax[0,3].set_title('image structure tensor')
    letters='abcdefghijklmnopqrstuvwxyz'
    for a,l in zip(ax.ravel(),letters):
        yl=a.get_ylabel(); noaxis(a); a.set_ylabel(yl,fontsize=7); imlabel(a,l)
    cb=fig.colorbar(g,ax=ax[:,1],shrink=0.3,ticks=[0,0.5,1],pad=0.01); cb.set_label('normalized',labelpad=1); cb.ax.tick_params(labelsize=7)
    save(fig,f'FigA3_gallery_part{part}',OUT)
# ---------------- Appendix A3 S3 field in 3D
fig=plt.figure(figsize=(FULL,2.3),layout='constrained')
for i,kind in enumerate(['capsule','torus','fibers']):
    mk,_,_=nq.shape3(68,kind); nn=nq.generate3(mk,30); n3=68
    a=fig.add_subplot(1,3,i+1,projection='3d')
    vert,faces,_,_=marching_cubes(nn[...,3].astype('float32'),level=0.0); faces=faces[::max(1,len(faces)//2500)]; xyz=vert[:,[2,1,0]]-n3/2
    a.plot_trisurf(xyz[:,0],xyz[:,1],xyz[:,2],triangles=faces,alpha=.15,color='#9db9d8',linewidth=0.05)
    sl=n3//2; yy,xx=np.mgrid[3:n3:4,3:n3:4]; vv=nn[sl,yy,xx]; sel=np.abs(vv[...,3])<.8
    a.quiver(xx[sel]-n3/2,yy[sel]-n3/2,np.zeros(sel.sum()),vv[sel,2],vv[sel,1],vv[sel,0],length=3.5,normalize=True,color='black',linewidth=0.6,arrow_length_ratio=.3)
    a.set_title(kind,pad=0); a.set_box_aspect((1,1,1)); a.tick_params(labelsize=6,pad=0)
    for f_ in (a.set_xlabel,a.set_ylabel,a.set_zlabel): f_('voxels',fontsize=6,labelpad=-6)
    a.text2D(0.0,0.95,f'({"abc"[i]})',transform=a.transAxes,fontsize=FS,fontweight='bold')
save(fig,'FigA4_S3_field',OUT)
print('done')
