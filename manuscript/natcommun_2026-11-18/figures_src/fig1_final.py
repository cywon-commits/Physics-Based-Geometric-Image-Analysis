"""Fig. 1: (a) procedure from binary image to spin texture to the three readouts (true 3D perspective rendering);
(b-e) the concrete stadium example (unchanged content)."""
import sys,numpy as np,scipy.ndimage as ndi
sys.path[:0]=['.','../code/revision2','../code']
from fig1_3d import *
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.collections import LineCollection
from readout import tv_density,curvature,m_of_k
NAVY='#1d2b4f'; GREY='#6b7287'; LINE='#9aa3b8'
W=7.1; H=5.1
fig=plt.figure(figsize=(W,H))
ov=fig.add_axes([0,0,1,1]); ov.set_xlim(0,1); ov.set_ylim(0,1); ov.axis('off'); ov.set_zorder(10); ov.patch.set_alpha(0)
bg=fig.add_axes([0,0,1,1]); bg.set_xlim(0,1); bg.set_ylim(0,1); bg.axis('off'); bg.set_zorder(-10)
def rbox(x,y,w,h,fc='#f5f7fb',ec='none',r=0.012,z=-4):
    bg.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=0.6,transform=bg.transAxes,zorder=z))
def arr(p0,p1,lw=1.1,c=NAVY,ms=8,style='-|>'):
    ov.add_patch(FancyArrowPatch(p0,p1,arrowstyle=style,mutation_scale=ms,color=c,lw=lw,transform=ov.transAxes,zorder=5,shrinkA=0,shrinkB=0))
def step(x,y,k,title,sub):
    ov.text(x,y,str(k),fontsize=6.5,fontweight='bold',color='white',ha='center',va='center',bbox=dict(boxstyle='circle,pad=0.22',fc=NAVY,ec='none'))
    ov.text(x+0.016,y,title,fontsize=7.2,fontweight='bold',color=NAVY,va='center')
    if sub: ov.text(x-0.008,y-0.045,sub,fontsize=6.3,color='#33384a',va='center')


import io
from PIL import Image as _Im
def render3d(draw,figsize=(3.2,2.4),dpi=600):
    """Render a 3D scene alone, crop to content, return RGBA array (transparent background)."""
    f3=plt.figure(figsize=figsize); ax=f3.add_axes([0,0,1,1],projection='3d'); draw(ax)
    buf=io.BytesIO(); f3.savefig(buf,format='png',dpi=dpi,transparent=True); plt.close(f3); buf.seek(0)
    im=np.asarray(_Im.open(buf)).copy(); a=im[...,3]>8; r=np.nonzero(a.any(1))[0]; c=np.nonzero(a.any(0))[0]
    return im[r[0]:r[-1]+1,c[0]:c[-1]+1]
def place(img,rect):
    ax=fig.add_axes(rect); ax.imshow(img,interpolation='lanczos'); ax.set_anchor('C'); noaxis(ax); ax.patch.set_alpha(0); return ax
# ---------------------------------------------------------------- data
m,vr,vc,s=small_scene(n=26,R=5.0,L=8.0); n=s.shape[0]
bm,_=nq.stadium(n=110,R=15,L=30,theta=30); bvr,bvc,bs=nq.generate_field(bm,nsteps=30); bs=bs.astype(float); bvr=bvr.astype(float); bvc=bvc.astype(float)

# ---------------------------------------------------------------- panel a
Y0=0.455; HA=0.535
rbox(0.008,Y0,0.984,HA)
ov.text(0.016,Y0+HA-0.012,'a',fontsize=9,fontweight='bold',va='top')
yt=Y0+HA-0.04
# 1 binary image
step(0.045,yt,1,'Binary image','pixels $f_i=\\pm1$')
a1=fig.add_axes([0.03,Y0+0.08,0.15,0.30]); a1.imshow(np.where(m,1,0),cmap=mpl.colors.ListedColormap(['#20263a','#f4f4f4']),interpolation='nearest')
for k in range(n+1): a1.axhline(k-0.5,color='#7f8699',lw=0.15); a1.axvline(k-0.5,color='#7f8699',lw=0.15)
noaxis(a1)
arr((0.185,Y0+0.23),(0.205,Y0+0.23))
# 2 field
step(0.235,yt,2,'Read as a field','Zeeman term $h f_i$')
def draw_field(ax):
    setup3d(ax,n,zoom=1.0)
    X,Y=np.meshgrid(np.arange(n+1),np.arange(n+1)); cols=np.array([[mpl.colors.to_rgba('#f4f4f4' if m[n-1-i,j] else '#20263a',0.95) for j in range(n)] for i in range(n)])
    ax.plot_surface(X,Y,np.zeros_like(X,float)+PZ,facecolors=cols,shade=False,linewidth=0.1,edgecolor=(0.5,0.5,0.6,0.5),zorder=0)
    pos=[];vec=[];col=[]
    for i in range(0,n,2):
        for j in range(0,n,2):
            ff=1.0 if m[i,j] else -1.0; pos.append((j+0.5,n-1-i+0.5,0.25*ff)); vec.append(np.array([0,0,ff])); col.append('#d9534f' if ff>0 else '#5b8bd0')
    spins(ax,vec,pos,col,L=1.5,r=0.16,lw=0.9)
place(render3d(draw_field),[0.205,Y0+0.07,0.20,0.31])
arr((0.408,Y0+0.23),(0.438,Y0+0.23)); ov.text(0.423,Y0+0.275,'$\\mathbf{n}\\leftarrow\\frac{\\mathbf{H}}{|\\mathbf{H}|}$',fontsize=7,ha='center',color=NAVY)
# 3 spin texture
step(0.45,yt,3,'Spin texture $\\mathbf{n}=(\\mathbf{v},s)$','walls rotate; $\\mathbf{v}$ follows the inward normal')
def draw_tex(ax):
    setup3d(ax,n,zoom=1.0); floor(ax,n); cone_texture(ax,vr,vc,s,n,sat=0.85,L=1.2,r=0.36)
place(render3d(draw_tex),[0.44,Y0+0.06,0.275,0.36])
ov.text(0.578,Y0+0.035,'$\\mathbf{H}_v=J\\!\\sum\\mathbf{v}_j+D\\nabla s,\\quad H_s=J\\!\\sum s_j+hf$',fontsize=6.4,ha='center',color='#33384a')
# bus to readouts
xb=0.745; ys=[Y0+0.405,Y0+0.245,Y0+0.085]
ov.plot([0.718,xb],[Y0+0.23,Y0+0.23],color=NAVY,lw=1.1,transform=ov.transAxes)
ov.plot([xb,xb],[ys[-1],ys[0]],color=NAVY,lw=1.1,transform=ov.transAxes,solid_capstyle='round')
step(0.775,yt,4,'Readouts',None)
titles=[('Extent','$\\hat P=\\sum\\frac{1}{2}|\\nabla s|/m(\\varphi,\\kappa)$'),('Orientation','$q_\\sigma=G_\\sigma\\!*\\!(\\mathbf{v}\\mathbf{v}^{\\mathsf{T}})$\ncharges $k\\in\\frac{1}{2}\\mathbb{Z}$'),('Topology','$Q=\\chi(\\{s>0\\})$')]
for y,(t_,eq) in zip(ys,titles):
    arr((xb,y),(0.77,y)); rbox(0.772,y-0.068,0.215,0.128,fc='white',ec='#d5dbe8',z=-3)
    ov.text(0.862,y+0.03,t_,fontsize=6.9,fontweight='bold',color=NAVY,va='center'); ov.text(0.862,y-0.022,eq,fontsize=5.9,va='center',linespacing=1.5)
# extent icon: TV ridge in 3D
w=tv_density(bs)/m_of_k(curvature(bs),2); w=ndi.gaussian_filter(ndi.zoom(w[15:95,15:95],0.6,order=1),1.1); nn_=w.shape[0]
RP=mpl.colors.LinearSegmentedColormap.from_list('rp',['#f3f0f7','#c9a7d8','#7b3fa0','#3a0f5c'])
def draw_ext(e):
    e.computed_zorder=False; e.set_proj_type('persp',focal_length=0.5); e.view_init(elev=34,azim=-58); e.set_axis_off(); e.set_box_aspect((1,1,0.3))
    Xe,Ye=np.meshgrid(np.arange(nn_),np.arange(nn_)); e.plot_surface(Xe,Ye,w[::-1],cmap=RP,rstride=1,cstride=1,linewidth=0,antialiased=True,vmin=0,vmax=w.max())
place(render3d(draw_ext,figsize=(2.4,2.0)),[0.776,ys[0]-0.06,0.08,0.115])
# orientation icon: normal director sticks around the stadium
o=fig.add_axes([0.776,ys[1]-0.062,0.08,0.115]); o.imshow(np.where(bm,1,0)[10:100,10:100],cmap=mpl.colors.ListedColormap(['#ffffff','#e7eaf2']))
from skimage.measure import find_contours
segs=[];cols=[]; sub=bs[10:100,10:100]; VRs=bvr[10:100,10:100]; VCs=bvc[10:100,10:100]
for c_ in find_contours(sub,0.0):
    for k in range(0,len(c_),5):
        r_,q_=c_[k]; i_,j_=int(round(r_)),int(round(q_)); ang=np.arctan2(-VRs[i_,j_],VCs[i_,j_]); Lh=3.4
        segs.append([(q_-Lh*np.cos(ang),r_+Lh*np.sin(ang)),(q_+Lh*np.cos(ang),r_-Lh*np.sin(ang))]); cols.append(mpl.colormaps['twilight']((ang%np.pi)/np.pi))
o.add_collection(LineCollection(segs,colors=cols,linewidths=1.6,capstyle='round')); noaxis(o)
# topology icon: Gauss-map sphere
t=fig.add_axes([0.778,ys[2]-0.062,0.08,0.115]); import fig1_parts as FP; FP.icon_sphere(t)

# ---------------------------------------------------------------- panels b-e: the readouts, enlarged (synthetic inputs only)
from dirscale import qfield,defect_list
yb=0.035; hb=0.335; ytit=yb+hb+0.03
xs=[0.012,0.262,0.512,0.762]; wp=0.226
def ptitle(x,l,t,sub):
    ov.text(x,ytit,l,fontsize=9,fontweight='bold',va='center')
    ov.text(x+0.022,ytit,t,fontsize=7.2,fontweight='bold',color=NAVY,va='center')
    ov.text(x+0.022,ytit-0.032,sub,fontsize=6.2,color='#33384a',va='center')
for x in xs: rbox(x,yb-0.012,wp+0.004,hb+0.075,fc='#f7f8fb',ec='#e1e5ee',z=-4)
# b extent: readout density as a ridge in perspective
from scipy.integrate import quad
yF,xF=np.mgrid[:110,:110].astype(float); XF=xF-54.7; YF=yF-55.3; RF,aF,kF=32,0.15,5
fm=np.hypot(XF,YF)<RF*(1+aF*np.cos(kF*np.arctan2(YF,XF)+0.3)); fs=nq.generate_field(fm,nsteps=30)[2].astype(float)
wv=tv_density(fs)/m_of_k(curvature(fs),2); P=quad(lambda t: np.hypot(RF*(1+aF*np.cos(kF*t)),RF*aF*kF*np.sin(kF*t)),0,2*np.pi,limit=200)[0]
wb=ndi.gaussian_filter(wv,1.7)[5:105,5:105]; nb=wb.shape[0]; bmF=fm[5:105,5:105]
def draw_b(e):
    e.computed_zorder=False; e.set_proj_type('persp',focal_length=0.45); e.view_init(elev=36,azim=-60); e.set_axis_off(); e.set_box_aspect((1,1,0.26),zoom=1.15)
    Xe,Ye=np.meshgrid(np.arange(nb),np.arange(nb))
    fl=np.where(bmF[::-1],1.0,0.0)
    e.plot_surface(Xe,Ye,np.zeros_like(wb)-0.02,facecolors=np.where(fl[...,None]>0,np.array(mpl.colors.to_rgba('#dfe4ef')),np.array(mpl.colors.to_rgba('#ffffff'))),shade=False,linewidth=0,zorder=0)
    e.plot_surface(Xe,Ye,wb[::-1],cmap=RP,rstride=1,cstride=1,linewidth=0,antialiased=True,vmin=0,vmax=np.percentile(wb,99.5),zorder=1)
place(render3d(draw_b,figsize=(3.0,2.6)),[xs[0]+0.004,yb+0.035,wp-0.004,hb-0.03])
ptitle(xs[0],'b','Extent','wall density $\\frac{1}{2}|\\nabla s|/m(\\varphi,\\kappa)$')
ov.text(xs[0]+0.012,yb+0.012,f'$\\hat P={wv.sum():.1f}$   exact $P={P:.1f}$',fontsize=6.3,color='#33384a')
# c orientation: projective director of v, angle-coded
def angimg(q,tr,gain=1.0):
    ph=(np.angle(q)/2)%np.pi; coh=np.clip(gain*np.abs(q)/np.percentile(np.abs(q),97),0,1)
    rgb=mpl.colormaps['twilight'](ph/np.pi)[...,:3]; return 1-(1-rgb)*coh[...,None]*0.8
def sticks(ax,q,step,Lh,color,lw,mask=None,thr=0.15):
    H,Wd=q.shape; yy,xx=np.mgrid[step//2:H:step,step//2:Wd:step]; qq=q[yy,xx]; amp=np.abs(qq)/np.percentile(np.abs(q),97)
    th=np.angle(qq)/2+np.pi/2; keep=amp>thr
    if mask is not None: keep&=mask[yy,xx]
    u=np.cos(th[keep]); v=-np.sin(th[keep]); X=xx[keep]; Y=yy[keep]; L=Lh*np.clip(amp[keep],0.35,1)
    ax.add_collection(LineCollection(np.stack([np.stack([X-L*u,Y-L*v],-1),np.stack([X+L*u,Y+L*v],-1)],1),colors=color,linewidths=lw,capstyle='round'))
import fig1_parts as FP
M2,VR2,VC2,S2=FP.MASK,FP.VR,FP.VC,FP.S
q2,tr2=qfield(np.stack([VR2*VR2,VC2*VC2,VR2*VC2]),8)
ac=fig.add_axes([xs[1]+0.008,yb+0.01,wp-0.012,hb-0.005])
ac.imshow(np.where(M2,1.0,0.0),cmap=mpl.colors.ListedColormap(['#ffffff','#e3e7f0']),interpolation='nearest')
ac.contour(S2,[0],colors=['#9aa3b8'],linewidths=0.6)
H_,W_=q2.shape; yy,xx=np.mgrid[3:H_:6,3:W_:6]; qq=q2[yy,xx]; amp=np.abs(qq)/np.percentile(np.abs(q2),97); keep=amp>0.12
th=np.angle(qq)/2+np.pi/2; u=np.cos(th[keep]); v=-np.sin(th[keep]); X=xx[keep]; Y=yy[keep]; Lh=2.5
cols=mpl.colormaps['twilight'](((np.angle(qq[keep])/2)%np.pi)/np.pi); cols[:,3]=np.clip(0.25+amp[keep],0,1)
ac.add_collection(LineCollection(np.stack([np.stack([X-Lh*u,Y-Lh*v],-1),np.stack([X+Lh*u,Y+Lh*v],-1)],1),colors=cols,linewidths=1.3,capstyle='round'))
noaxis(ac); ac.set_xlim(8,149); ac.set_ylim(141,0)
cw=fig.add_axes([xs[1]+0.17,yb+0.025,0.045,0.045]); tt=np.linspace(0,np.pi,64)
for k_ in range(64): cw.plot([0,np.cos(tt[k_])],[0,np.sin(tt[k_])],color=mpl.colormaps['twilight'](k_/64),lw=1.2); cw.plot([0,-np.cos(tt[k_])],[0,-np.sin(tt[k_])],color=mpl.colormaps['twilight'](k_/64),lw=1.2)
cw.set_aspect('equal'); noaxis(cw); cw.patch.set_alpha(0)
ptitle(xs[1],'c','Orientation','director field $q_\\sigma$, $\\sigma=8$')
# d defects of the projective director on a synthetic labyrinth
YM=np.load('../data/revision2/fig1_yshape.npy')
yvr,yvc,ys_=[a_.astype(float) for a_ in nq.generate_field(YM,nsteps=30)]
q3,tr3=qfield(np.stack([yvr*yvr,yvc*yvc,yvr*yvc]),3); wd=YM.shape[0]
ad=fig.add_axes([xs[2]+0.008,yb+0.01,wp-0.012,hb-0.005])
ad.imshow(np.where(YM,1.0,0.0),cmap=mpl.colors.ListedColormap(['#ffffff','#d9deea']),interpolation='nearest')
ad.contour(ys_,[0],colors=['#9aa3b8'],linewidths=0.6)
sticks(ad,q3,4,1.9,'#5c6680',0.8,thr=0.06)
DEF=[d_ for d_ in defect_list(q3,3) if tr3[int(d_[0]),int(d_[1])]>1e-6*tr3.max()]
for r,c,k in DEF:
    ad.plot(c,r,'^' if k>0 else 'v',ms=8,mfc='#d6453d' if k>0 else '#2f63c4',mec='white',mew=0.9,zorder=5)
    dx,dy,ha=((9,0,'left') if r<30 else ((0,12,'center') if k<0 else (0,10,'center')))
    ad.text(c+dx,r+dy,'$+\\frac{1}{2}$' if k>0 else '$-\\frac{1}{2}$',fontsize=7,color='#d6453d' if k>0 else '#2f63c4',ha=ha,va='center',fontweight='bold',bbox=dict(boxstyle='round,pad=0.1',fc='white',ec='none',alpha=0.85))
noaxis(ad); ad.set_xlim(8,102); ad.set_ylim(98,4)
ptitle(xs[2],'d','Defects','ends $+\\frac{1}{2}$, branch point $-\\frac{1}{2}$, $\\sigma=3$')
# e topology: degree per object in perspective
TZ=np.load('../data/revision2/fig1_topology.npz',allow_pickle=True); tm,tvr,tvc,ts=TZ['m'],TZ['vr'],TZ['vc'],TZ['s']; nt=ts.shape[0]
def draw_e(ax):
    setup3d(ax,nt,zoom=1.1,elev=40); floor(ax,nt); texture_spins(ax,tvr,tvc,ts,nt,stride=2,L=1.25,r=0.36)
    for (i,j,lab) in TZ['labels']: ax.text(float(j),nt-1-float(i)+0.5,2.4,lab,fontsize=13,fontweight='bold',color=NAVY,ha='center',va='bottom',zorder=30,bbox=dict(boxstyle='round,pad=0.18',fc='white',ec='#c9cfdd',lw=0.6,alpha=0.92))
place(render3d(draw_e,figsize=(3.2,2.8)),[xs[3]+0.012,yb+0.04,wp-0.02,hb-0.04])
ptitle(xs[3],'e','Topology','each domain contributes its $\\chi$')
ov.text(xs[3]+0.012,yb+0.012,'$Q=1+0-1=0=\\chi(\\{s>0\\})$',fontsize=6.3,color='#33384a')
fig.patch.set_facecolor('white')
fig.savefig('../figures/Fig1_concept_v2.png',dpi=300); fig.savefig('../figures/Fig1_concept_v2.pdf'); print('ok')
