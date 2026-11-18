"""Fig. 7 (NC numbering): degree vs Euler characteristic, rendered in oblique perspective.
Top: 2D masks -- generated s as a tilted plate with wall and bulk spins as shaded cones.
Bottom: 3D volumes -- the generated wall {s=0} as a shaded isosurface (marching cubes)."""
import sys,io,numpy as np,pandas as pd,scipy.ndimage as ndi
sys.path[:0]=['.','../code/revision2','../code']
from fig1_3d import *
from PIL import Image as _Im
from skimage.measure import find_contours,marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
NAVY='#1d2b4f'
def render3d(draw,figsize=(3.0,2.4),dpi=500):
    f3=plt.figure(figsize=figsize); ax=f3.add_axes([0,0,1,1],projection='3d'); draw(ax)
    buf=io.BytesIO(); f3.savefig(buf,format='png',dpi=dpi,transparent=True); plt.close(f3); buf.seek(0)
    im=np.asarray(_Im.open(buf)).copy(); a=im[...,3]>8; r=np.nonzero(a.any(1))[0]; c=np.nonzero(a.any(0))[0]
    return im[r[0]:r[-1]+1,c[0]:c[-1]+1]
W=FULL+0.8; H=4.0
fig=plt.figure(figsize=(W,H)); fig.patch.set_facecolor('white')
def place(img,rect):
    ax=fig.add_axes(rect); ax.imshow(img,interpolation='lanczos'); ax.set_anchor('C'); noaxis(ax); ax.patch.set_alpha(0); return ax
# ---------------- 2D cases
n=154; y,x=np.mgrid[:n,:n]; x=x-(n-1)/2; y=y-(n-1)/2; rr=np.hypot(x,y)
cases2=[('one disk',rr<25),('two disks',((x-33)**2+y*y<18**2)|((x+33)**2+y*y<18**2)),('annulus',(rr>17)&(rr<31)),('disk + annulus',((x+35)**2+y*y<18**2)|((np.hypot(x-33,y)>10)&(np.hypot(x-33,y)<23)))]
t2=pd.read_csv('../data/2d_spin_topology.csv'); t3=pd.read_csv('../data/3d_synthetic_validation.csv')
CR=slice(27,-27)
def draw2d(m):
    vr,vc,s=[a.astype(float) for a in nq.generate_field(m,nsteps=30)]
    s=s[CR,:]; vr=vr[CR,:]; vc=vc[CR,:]; ny,nx=s.shape
    def draw(ax):
        ax.computed_zorder=False; ax.set_proj_type('persp',focal_length=0.5); ax.view_init(elev=38,azim=-62); ax.set_axis_off()
        ax.set_xlim(0,nx); ax.set_ylim(0,ny); ax.set_zlim(-14,14); ax.set_box_aspect((nx,ny,28),zoom=1.25); ax.patch.set_alpha(0)
        st=2; X,Y=np.meshgrid(np.arange(0,nx+1,st),np.arange(0,ny+1,st)); sub=s[::-1][::st,::st]
        cols=np.array([[pastel(CM(0.5+0.5*v_),0.5) for v_ in row] for row in sub])
        ax.plot_surface(X,Y,np.zeros_like(X,float)-6,facecolors=cols,shade=False,linewidth=0,antialiased=False,zorder=0)
        pos=[];vec=[];col=[]
        for c in find_contours(s,0.0):
            for k in range(0,len(c),8):
                r_,q_=c[k]; i,j=int(round(r_)),int(round(q_)); d=np.array([vc[i,j],-vr[i,j],s[i,j]]); pos.append((q_+0.5,ny-1-r_+0.5,-1.0)); vec.append(d); col.append(NAVY)
        for i in range(7,ny,14):
            for j in range(7,nx,14):
                if abs(s[i,j])>0.95: pos.append((j+0.5,ny-1-i+0.5,-1.0)); vec.append(np.array([0,0,np.sign(s[i,j])])); col.append('#c0392b' if s[i,j]>0 else '#3a6fb0')
        order=np.argsort([-p[1] for p in pos]); spins(ax,[vec[k] for k in order],[pos[k] for k in order],[col[k] for k in order],L=7.5,r=1.7,lw=0)
    return draw
xs=np.linspace(0.02,0.745,4); wpan=0.215
for j,(nm,m) in enumerate(cases2):
    place(render3d(draw2d(m)),[xs[j],0.55,wpan,0.36])
    r=t2[t2.case==nm.replace('annulus','ring') if nm=='disk + annulus' else t2.case==nm].iloc[0]
    fig.text(xs[j]+wpan/2,0.965,nm,ha='center',fontsize=7.5,fontweight='bold',color=NAVY)
    fig.text(xs[j]+wpan/2,0.925,f'$Q_2={r.Q2_solid_angle:.0f}$,   $\\chi={int(r.input_euler)}$',ha='center',fontsize=7)
    fig.text(xs[j]+0.005,0.965,'abcd'[j],fontsize=9,fontweight='bold',va='center')
# ---------------- 3D volumes
chis={'sphere':1,'torus':0,'pair':2,'porous':-5}
LIGHT2=np.array([-0.45,-0.55,0.70]); LIGHT2/=np.linalg.norm(LIGHT2)
def draw3d(kind):
    mk,_,_=nq.shape3(68,kind); nn=nq.generate3(mk,30); s3=nn[...,3]
    vt,fc,_,_=marching_cubes(ndi.gaussian_filter(s3.astype(float),0.9),0.0)            # axes z,y,x
    P=np.stack([vt[:,2],-(vt[:,1]),vt[:,0]],1)  # physical x,y,z
    tri=P[fc]; nrm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); nrm/=np.linalg.norm(nrm,axis=1,keepdims=True)+1e-12
    sh=0.42+0.58*np.clip(-(nrm@LIGHT2),0,1)       # outward normal points to decreasing s
    base=np.array(mpl.colors.to_rgb('#c8463c')); fcs=np.clip(base[None,:]*sh[:,None]+0.18*(sh[:,None]**6),0,1)
    rem=None
    if kind=='porous':
        z_,y_,x_=np.nonzero(mk&~(s3>0)); rem=np.stack([x_,-y_,z_],1)
        fcs=np.concatenate([fcs,np.full((len(fcs),1),0.30)],1)
    def draw(ax):
        ax.computed_zorder=False; ax.set_proj_type('persp',focal_length=0.6); ax.view_init(elev=24,azim=-58); ax.set_axis_off(); ax.patch.set_alpha(0)
        lo=P.min(0)-2; hi=P.max(0)+2; c=(lo+hi)/2; R=(hi-lo).max()/2
        ax.set_xlim(c[0]-R,c[0]+R); ax.set_ylim(c[1]-R,c[1]+R); ax.set_zlim(c[2]-R,c[2]+R); ax.set_box_aspect((1,1,1),zoom=1.3)
        X,Y=np.meshgrid([c[0]-R,c[0]+R],[c[1]-R,c[1]+R]); ax.plot_surface(X,Y,np.full_like(X,lo[2]-3.0),color='#eef1f7',shade=False,linewidth=0,zorder=0)
        # soft shadow: projected silhouette
        ax.add_collection3d(Poly3DCollection(np.dstack([tri[...,0],tri[...,1],np.full(tri.shape[:2],lo[2]-2.9)]),facecolors=(0.75,0.78,0.86,0.06),edgecolors='none',zorder=1))
        pc=Poly3DCollection(tri,facecolors=fcs,edgecolors='none'); pc.set_zorder(2); ax.add_collection3d(pc)
        if rem is not None: ax.scatter(rem[:,0],rem[:,1],rem[:,2],s=1.6,c=NAVY,linewidths=0,depthshade=False,zorder=4)
    return draw
for j,kind in enumerate(['sphere','torus','pair','porous']):
    place(render3d(draw3d(kind)),[xs[j]+0.02,0.03,wpan-0.04,0.36])
    q=int(t3[t3.kind==kind].S3_integer_degree.iloc[0])
    fig.text(xs[j]+wpan/2,0.465,kind,ha='center',fontsize=7.5,fontweight='bold',color=NAVY)
    fig.text(xs[j]+wpan/2,0.425,f'$Q_3={q}$,   $\\chi(\\{{s>0\\}})={chis[kind]}$',ha='center',fontsize=7)
    fig.text(xs[j]+0.005,0.465,'efgh'[j],fontsize=9,fontweight='bold',va='center')
cax=fig.add_axes([0.972,0.6,0.008,0.26]); cb=fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-1,1),cmap=CM),cax=cax,ticks=[-1,0,1])
cb.ax.tick_params(labelsize=6,length=2,pad=1); cb.outline.set_linewidth(0.4); cax.set_title('$s$',fontsize=6.5,pad=2)
fig.savefig('../figures/Fig8_topology.png',dpi=300); fig.savefig('../figures/Fig8_topology.pdf'); print('ok')
