"""True 3D (perspective, oblique view) rendering of lattice spins as shaded cones on a plane."""
import sys,numpy as np
sys.path[:0]=['.','../code/revision2','../code']
from style import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import new_quantitative_studies as nq
CM=mpl.colormaps['coolwarm']
def small_scene(n=24,R=4.6,L=7.0,theta=28):
    m,_=nq.stadium(n=n,R=R,L=L,theta=theta); vr,vc,s=nq.generate_field(m,nsteps=30); return m,vr.astype(float),vc.astype(float),s.astype(float)
def setup3d(ax,n,elev=30,azim=-58,zlim=(-1.3,1.3),zoom=1.35):
    ax.computed_zorder=False; ax.set_proj_type('persp',focal_length=0.5); ax.view_init(elev=elev,azim=azim)
    ax.set_xlim(0,n); ax.set_ylim(0,n); ax.set_zlim(*zlim); ax.set_box_aspect((1,1,(zlim[1]-zlim[0])/n),zoom=zoom); ax.set_axis_off()
    ax.patch.set_alpha(0); ax.set_facecolor((1,1,1,0))
def pastel(c,f=0.55): c=np.array(mpl.colors.to_rgba(c)); c[:3]=1-(1-c[:3])*f; return c
PZ=-0.62
def plate(ax,img,n,cmap=None,f=0.5,alpha=1.0):
    X,Y=np.meshgrid(np.arange(n+1),np.arange(n+1)); Z=np.zeros_like(X,float)+PZ
    cols=np.array([[pastel((cmap or CM)(0.5+0.5*img[n-1-i,j]),f) for j in range(n)] for i in range(n)])
    ax.plot_surface(X,Y,Z,facecolors=cols,shade=False,linewidth=0.15,edgecolor=(1,1,1,0.6),antialiased=True,alpha=alpha,zorder=0)
def _cone(p,d,L,r):
    d=d/np.linalg.norm(d); a=np.array([1,0,0]) if abs(d[0])<0.9 else np.array([0,1,0]); e1=np.cross(d,a); e1/=np.linalg.norm(e1); e2=np.cross(d,e1)
    t=np.linspace(0,2*np.pi,13)[:-1]; tail=p-0.5*L*d; base=p-0.45*L*d; tip=p+0.55*L*d
    ring=[base+r*(np.cos(a_)*e1+np.sin(a_)*e2) for a_ in t]
    faces=[[ring[k],ring[(k+1)%12],tip] for k in range(12)]+[ring]
    shaft=[tail, base]
    return faces,shaft,np.array([np.cos(a_)*e1+np.sin(a_)*e2 for a_ in t])
LIGHT=np.array([-0.4,-0.5,0.77]); LIGHT/=np.linalg.norm(LIGHT)
def spins(ax,vecs,pos,colors,L=1.25,r=0.23,lw=0.9):
    """vecs: (N,3) unit spins; pos: (N,3) bases; colors: N rgba. Draw shaft + shaded cone."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    polys=[]; fcs=[]; segs=[]; scol=[]
    for p,d,c in zip(pos,vecs,colors):
        faces,shaft,nrm=_cone(np.array(p,float),np.array(d,float),L,r); segs.append(shaft); scol.append(c)
        for k,f in enumerate(faces):
            nf=np.cross(f[1]-f[0],f[2]-f[0]); nf/=np.linalg.norm(nf)+1e-12; sh=0.55+0.45*max(0,float(nf@LIGHT))
            cc=np.array(mpl.colors.to_rgba(c)); cc[:3]=np.clip(cc[:3]*sh+0.12*(1-sh)*0,0,1); polys.append(f); fcs.append(cc)
    ax.add_collection3d(Line3DCollection(segs,colors=scol,linewidths=lw),zs=None) if False else None
    if lw>0:
        lc=Line3DCollection(segs,colors=scol,linewidths=lw); lc.set_zorder(1); ax.add_collection3d(lc)
    pc=Poly3DCollection(polys,facecolors=fcs,edgecolors='none'); pc.set_zorder(2); ax.add_collection3d(pc)
def texture_spins(ax,vr,vc,s,n,stride=2,L=1.25,r=0.23,only=None,wall=0.92):
    pos=[];vec=[];col=[]
    for i in range(0,n):
        for j in range(0,n):
            isw=abs(s[i,j])<wall or np.hypot(vr[i,j],vc[i,j])>0.35
            if not isw and (i%stride or j%stride): continue
            if only is not None and not only[i,j]: continue
            x=j+0.5; y=n-1-i+0.5; d=np.array([vc[i,j],-vr[i,j],s[i,j]]); pos.append((x,y,0.0)); vec.append(d)
            col.append('#c0392b' if s[i,j]>0.6 else ('#3a6fb0' if s[i,j]<-0.6 else '#1d2b4f'))
    spins(ax,vec,pos,col,L=L,r=r)

def cone_texture(ax,vr,vc,s,n,L=1.05,r=0.34,stride=1,sat=1.0):
    pos=[];vec=[];col=[]
    for i in range(0,n,stride):
        for j in range(0,n,stride):
            pos.append((j+0.5,n-1-i+0.5,0.0)); vec.append(np.array([vc[i,j],-vr[i,j],s[i,j]]))
            col.append(CM(0.5+0.5*sat*s[i,j]))
    spins(ax,vec,pos,col,L=L,r=r,lw=0)
def floor(ax,n,color='#eef1f7'):
    X,Y=np.meshgrid([0,n],[0,n]); ax.plot_surface(X,Y,np.zeros_like(X,float)+PZ,color=color,shade=False,linewidth=0,zorder=0)
