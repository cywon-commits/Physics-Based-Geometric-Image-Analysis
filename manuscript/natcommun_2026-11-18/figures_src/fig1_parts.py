"""Shared drawing parts for the Fig. 1 design options (overview schematics)."""
import sys,numpy as np,scipy.ndimage as ndi
sys.path[:0]=['.','../code/revision2','../code','/home/claude/work']
from style import *
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle,Polygon
from matplotlib.collections import LineCollection
from readout import tv_density,curvature,m_of_k
import new_quantitative_studies as nq
from dirscale import qfield,defect_list
CM=mpl.colormaps['coolwarm']
NAVY='#1d2b4f'; SOFT='#f4f6fb'; ACC=C['pbgia']; WARM='#e0533d'; COOL='#3b6fd8'

# ------------------------------------------------------------------ scene
def scene(n=150):
    y,x=np.mgrid[:n,:n].astype(float); cx,cy=62,78; th=np.arctan2(y-cy,x-cx); r=np.hypot(x-cx,y-cy)
    blob=r<38*(1+0.13*np.cos(3*th+0.4)); hole=np.hypot(x-cx-6,y-cy+4)<11; disk=np.hypot(x-118,y-34)<13
    m=(blob&~hole)|disk
    vr,vc,s=nq.generate_field(m,nsteps=30); return m,vr.astype(float),vc.astype(float),s.astype(float)
MASK,VR,VC,S=scene()

# ------------------------------------------------------------------ perspective plane
def proj(x,y,z,sk=0.45,sq=0.52): return x+sk*y, sq*y+z
def plane(ax,img,ox=0,oy=0,scale=1.0,cmap=None,vmin=-1,vmax=1,alpha=0.75,edge=True,zorder=1):
    n=img.shape[0]; xs=np.arange(n+1)*scale; X,Y=np.meshgrid(xs,xs[::-1])     # row 0 at top (y large)
    PX,PY=proj(X,Y,0); img=np.asarray(img)*0.62; ax.pcolormesh(PX+ox,PY+oy,img,cmap=cmap or CM,vmin=vmin,vmax=vmax,shading='flat',alpha=alpha,rasterized=True,zorder=zorder,edgecolors='face',linewidth=0.3,antialiased=True)
    if edge:
        c=np.array([[0,0],[n,0],[n,n],[0,n],[0,0]])*scale; px,py=proj(c[:,0],c[:,1],0); ax.plot(px+ox,py+oy,color='#55607a',lw=0.6,zorder=zorder+0.1)
def _arrow3(ax,x,y,vx,vy,vz,L,ox,oy,lw,z):
    x0,y0=proj(x,y,0); x1,y1=proj(x+L*vx*0.75,y+L*vy*0.75,L*vz*0.8); col=NAVY if vz==0 else ('#b2182b' if vz>0 else '#2166ac')
    ax.annotate('',xy=(x1+ox,y1+oy),xytext=(x0+ox,y0+oy),arrowprops=dict(arrowstyle='-|>,head_length=0.22,head_width=0.11',color=col,lw=lw,shrinkA=0,shrinkB=0),zorder=z)
def spins3d(ax,vr,vc,s,ox=0,oy=0,scale=1.0,wall_step=11,bulk_step=17,L=12.0,lw=0.6,zorder=3):
    """3D-looking spins: along the s=0 contour (transverse, pointing along the inward normal) and on a sparse bulk grid (up/down)."""
    from skimage.measure import find_contours
    n=s.shape[0]; items=[]
    for c in find_contours(s,0.0):
        for k in range(0,len(c),wall_step):
            r,cc=c[k]; i,j=int(round(r)),int(round(cc)); i=min(max(i,0),n-1); j=min(max(j,0),n-1)
            vx,vy,vz=vc[i,j],-vr[i,j],0.0; nr=np.hypot(vx,vy)+1e-9; items.append((r,cc,vx/nr,vy/nr,vz))
    for i in range(bulk_step//2,n,bulk_step):
        for j in range(bulk_step//2,n,bulk_step):
            if abs(s[i,j])>0.95: items.append((i,j,0.0,0.0,np.sign(s[i,j])))
    items.sort(key=lambda t:t[0])
    for r,cc,vx,vy,vz in items:
        _arrow3(ax,(cc+0.5)*scale,(n-r-0.5)*scale,vx,vy,vz,L*scale,ox,oy,lw,zorder)
def field_arrows(ax,mask,ox=0,oy=0,scale=1.0,step=12,L=9):
    n=mask.shape[0]
    for i in range(step//2,n,step):
        for j in range(step//2,n,step):
            up=mask[i,j]; x=(j+0.5)*scale; y=(n-i-0.5)*scale; x0,y0=proj(x,y,0); x1,y1=proj(x,y,(L if up else -L)*scale*0.6)
            ax.annotate('',xy=(x1+ox,y1+oy),xytext=(x0+ox,y0+oy),arrowprops=dict(arrowstyle='-|>,head_length=0.22,head_width=0.11',color='#b2182b' if up else '#2166ac',lw=0.6),zorder=3)
def box(ax,x,y,w,h,fc=SOFT,ec='#c9cfdd',lw=0.7,r=0.02,z=0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=lw,transform=ax.transAxes,zorder=z))
def arrow(ax,p0,p1,color='#55607a',lw=1.4,style='-|>',ms=10,rad=0.0,trans=None,z=5):
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle=style,mutation_scale=ms,color=color,lw=lw,connectionstyle=f'arc3,rad={rad}',transform=trans or ax.transAxes,zorder=z))

# ------------------------------------------------------------------ icons (each draws into its own axes)
def icon_binary(ax):
    ax.imshow(np.where(MASK,1,0),cmap=mpl.colors.ListedColormap(['#0d1321','#f2f2f2']),interpolation='nearest'); noaxis(ax)
def icon_s(ax): ax.imshow(S,interpolation='nearest',**SFIELD); noaxis(ax)
def icon_tv(ax):
    w=tv_density(S)/m_of_k(curvature(S),2); ax.imshow(np.sqrt(w),cmap='magma',vmin=0,vmax=np.sqrt(np.percentile(w,99.7)),interpolation='bilinear'); noaxis(ax)
    ax.set_facecolor('black')
def icon_wall(ax):
    u=np.linspace(-4,4,400)
    for k,cl in ((0.0,'#8c8c86'),(0.7,ACC),(1.5,WARM)):
        th=2*np.arctan(np.exp(u/(1+0.25*k))); ax.plot(u,np.cos(th),color=cl,lw=1.3); ax.plot(u,np.sin(th),color=cl,lw=0.8,ls=':')
    ax.axhline(0,color='#c9cfdd',lw=0.5); ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
def icon_filter(ax):
    rng=np.random.default_rng(3); m=MASK[20:120,10:110].copy(); noisy=m^(rng.random(m.shape)<0.02)
    clean=nq.generate_field(noisy,nsteps=30)[2]>0
    both=np.concatenate([noisy,np.ones((m.shape[0],6),bool)*0,clean],1)
    ax.imshow(both,cmap=mpl.colors.ListedColormap(['#0d1321','#f2f2f2']),interpolation='nearest'); noaxis(ax)
    ax.annotate('',xy=(0.62,0.5),xytext=(0.42,0.5),xycoords='axes fraction',arrowprops=dict(arrowstyle='-|>',color=WARM,lw=1.4))
LIC=np.load('../data/revision2/t15_lic_img.npy'); LRAW=np.load('/home/claude/work/data/t15_lic_raw.npy')
def icon_director(ax,sig=8,r0=70,r1=150,c0=60,c1=180,step=7):
    q,_=qfield(LRAW,sig); img=LIC[r0:r1,c0:c1]; ax.imshow(img,cmap='gray',vmin=-0.2,vmax=1.2); qq=q[r0:r1,c0:c1]
    yy,xx=np.mgrid[step//2:r1-r0:step,step//2:c1-c0:step]; tan=np.deg2rad(((0.5*np.degrees(np.angle(qq)))+90)%180)[yy,xx]
    L=4.2; u=np.cos(tan); v=-np.sin(tan); seg=np.stack([np.stack([xx-L*u,yy-L*v],-1),np.stack([xx+L*u,yy+L*v],-1)],2).reshape(-1,2,2)
    ax.add_collection(LineCollection(seg,colors='#ffd166',linewidths=0.8))
    for r,c,k in defect_list(q,margin=10):
        if r0+3<r<r1-3 and c0+3<c<c1-3: ax.plot(c-c0,r-r0,'^' if k>0 else 'v',ms=6,mfc=WARM if k>0 else COOL,mec='white',mew=0.7)
    ax.set_xlim(0,c1-c0); ax.set_ylim(r1-r0,0); noaxis(ax)
def icon_sphere(ax,elev=0.42,nmer=28):
    """Gauss-map picture of the degree: each wall point maps the normal direction to a meridian from pole to pole."""
    t=np.linspace(0,2*np.pi,300)
    g=np.linspace(-1,1,300); X,Y=np.meshgrid(g,g); R=np.hypot(X,Y); Z=np.sqrt(np.clip(1-R*R,0,1)); shade=np.where(R<=1,0.45+0.55*(-0.35*X+0.45*Y+0.82*Z),np.nan)
    ax.imshow(shade,extent=(-1,1,-1,1),origin='lower',cmap='Blues',vmin=-0.4,vmax=1.6,alpha=0.35,zorder=1)
    ce,se=np.cos(elev),np.sin(elev)
    def P3(x,y,z): return x, z*ce-y*se, z*se+y*ce
    th=np.linspace(0,np.pi,120)
    for k,ph in enumerate(np.linspace(0,2*np.pi,nmer,endpoint=False)):
        x=np.sin(th)*np.cos(ph); y=np.sin(th)*np.sin(ph); z=np.cos(th); px,py,pz=P3(x,y,z); m=pz>-0.01
        pts=np.stack([px,py],1); segs=np.stack([pts[:-1],pts[1:]],1); vis=m[:-1]&m[1:]
        lc=LineCollection(segs[vis],colors=[CM(0.5+0.5*zz) for zz in z[:-1][vis]],linewidths=0.9,zorder=2); ax.add_collection(lc)
    x=np.cos(t); y=np.sin(t); px,py,pz=P3(x,y,0*t); m=pz>0; ax.plot(np.where(m,px,np.nan),np.where(m,py,np.nan),color=NAVY,lw=1.1,zorder=3)
    ax.plot(np.cos(t),np.sin(t),color='#55607a',lw=0.7,zorder=3); px,py,_=P3(0,0,1); ax.plot(px,py,'o',color=WARM,ms=3,zorder=4)
    ax.set_xlim(-1.08,1.08); ax.set_ylim(-1.08,1.08); ax.set_aspect('equal'); noaxis(ax)
def icon_scale(ax):
    import pandas as pd; t=pd.read_csv('../data/revision2/t15_lic_sigma.csv'); q_=t[t.method=='PBGIA']
    ax.plot(q_.sigma,(q_.n_plus+q_.n_minus).clip(lower=0.5),'o-',color=ACC,ms=2.5,lw=1)
    for dd in (40,110): ax.axvline(0.4413*dd,color=WARM,ls='--',lw=0.8)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([]); ax.set_yticks([]); ax.minorticks_off(); ax.grid(False)
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    ax.set_xlabel('window $\\sigma$',fontsize=6,labelpad=1); ax.set_ylabel('defects',fontsize=6,labelpad=1)
def wall_zoom(ax,wd=16,step=1):
    wr,wc=np.nonzero(tv_density(S)>0.3); kk=np.argmin((wr-48)**2+(wc-30)**2); r0,c0=int(wr[kk])-wd//2,int(wc[kk])-wd//2
    ax.imshow(S[r0:r0+wd,c0:c0+wd],interpolation='nearest',**SFIELD); yy,xx=np.mgrid[:wd,:wd]; V=VR[r0:r0+wd,c0:c0+wd]; U=VC[r0:r0+wd,c0:c0+wd]; sel=(np.hypot(U,V)>0.35)&((yy+xx)%2==0)
    ax.quiver(xx[sel],yy[sel],U[sel],V[sel],angles='xy',scale_units='xy',scale=0.6,width=0.011,headwidth=3.5,headlength=3.5,headaxislength=3,color='black',pivot='mid'); noaxis(ax); return r0,c0,wd
def tlabel(ax,x,y,s,size=7,weight='bold',color=NAVY,ha='left',va='center',trans=None,**kw):
    ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,ha=ha,va=va,transform=trans or ax.transAxes,**kw)
def overlay(fig): a=fig.add_axes([0,0,1,1]); a.set_xlim(0,1); a.set_ylim(0,1); a.axis('off'); a.set_zorder(-1); return a
