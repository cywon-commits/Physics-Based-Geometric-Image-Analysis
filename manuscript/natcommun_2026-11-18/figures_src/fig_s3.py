import sys,numpy as np,scipy.ndimage as ndi
sys.path[:0]=['.','../code']
from style import *
import new_quantitative_studies as nq
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
OUT='../figures'; N=68
spec=[('capsule','long-axis plane','long'),('torus','axial plane','axis'),('fibers','transverse plane','perp')]
fig=plt.figure(figsize=(FULL,4.3),layout='constrained'); gs=fig.add_gridspec(2,3,height_ratios=[1,1.05])
ls=LightSource(azdeg=315,altdeg=40)
for j,(kind,desc,mode) in enumerate(spec):
    mk,_,_=nq.shape3(N,kind); n=nq.generate3(mk,30); s=n[...,3]
    v=np.stack([n[...,2],-n[...,1],n[...,0]],-1)   # spin stored as physical (x,y,z)=(axis2,-axis1,axis0); convert to array axes
    idx=np.argwhere(mk).astype(float); cen=idx.mean(0); w,U=np.linalg.eigh(np.cov((idx-cen).T))   # ascending variances
    if mode=='long': e1,e2=U[:,2],U[:,1]
    elif mode=='axis': e1,e2=U[:,0],U[:,2]
    else:
        from skimage.measure import label as lab_
        one=np.argwhere(lab_(mk)==1).astype(float); w1,U1=np.linalg.eigh(np.cov((one-one.mean(0)).T)); ax_=U1[:,2]
        e1=np.cross(ax_,[1.,0,0]); e1/=np.linalg.norm(e1); e2=np.cross(ax_,e1)
    # --- 3D surface with cutting plane
    ax=fig.add_subplot(gs[0,j],projection='3d')
    vert,faces,_,_=marching_cubes(s.astype('float32'),0.0)
    tri=vert[faces]; xyz=tri[...,[2,1,0]]-N/2
    nrm=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]); nrm/=np.linalg.norm(nrm,axis=1,keepdims=True)+1e-12
    shade=np.clip(0.35+0.65*np.abs(nrm@np.array([0.4,-0.5,0.77])),0,1)
    col=np.array(matplotlib.colors.to_rgb('#b2182b'))[None]*shade[:,None]+0.0
    pc=Poly3DCollection(xyz,facecolors=col,edgecolors='none',alpha=1.0); ax.add_collection3d(pc)
    L=26; c=cen[[2,1,0]]-N/2; a1=e1[[2,1,0]]; a2=e2[[2,1,0]]
    corners=np.array([c+L*(p*a1+q*a2) for p,q in ((-1,-1),(1,-1),(1,1),(-1,1))])
    ax.add_collection3d(Poly3DCollection([corners],facecolors=(0.5,0.5,0.5,0.22),edgecolors=C['ink'],linewidths=0.8))
    lim=27; ax.set(xlim=(-lim,lim),ylim=(-lim,lim),zlim=(-lim,lim)); ax.set_box_aspect((1,1,1)); pn=np.cross(a1,a2); pn/=np.linalg.norm(pn); dv=0.75*pn+0.45*a1+0.3*a2
    if dv[2]<0: dv=-dv
    dv/=np.linalg.norm(dv); ax.view_init(elev=float(np.degrees(np.arcsin(np.clip(dv[2],0.25,0.8)))),azim=float(np.degrees(np.arctan2(dv[1],dv[0]))))
    ax.set_xticks([-20,0,20]);ax.set_yticks([-20,0,20]);ax.set_zticks([-20,0,20]); ax.tick_params(labelsize=5.5,pad=-2)
    ax.set_title(kind,pad=-2)
    ax.text2D(0.0,0.97,f'({"abc"[j]})',transform=ax.transAxes,fontsize=FS,fontweight='bold')
    # --- slice on the plane
    t=np.arange(-L,L+0.5,0.5); T1,T2=np.meshgrid(t,t,indexing='xy')
    P=cen[:,None,None]+e1[:,None,None]*T1+e2[:,None,None]*T2
    sp=ndi.map_coordinates(s,P,order=1,mode='nearest'); vp=np.stack([ndi.map_coordinates(v[...,k],P,order=1,mode='nearest') for k in range(3)])
    v1=np.einsum('k,kij->ij',e1,vp); v2=np.einsum('k,kij->ij',e2,vp); vperp=np.sqrt(np.maximum(0,(vp**2).sum(0)-v1**2-v2**2))
    bx=fig.add_subplot(gs[1,j]); im=bx.imshow(sp,origin='lower',extent=(-L,L,-L,L),**SFIELD)
    from skimage.measure import find_contours
    for cpath in find_contours(sp,0.0):
        seg=np.cumsum(np.r_[0,np.hypot(*np.diff(cpath,axis=0).T)])*0.5; want=np.arange(0,seg[-1],2.6)
        rr=np.interp(want,seg,cpath[:,0]); cc=np.interp(want,seg,cpath[:,1])
        a1=ndi.map_coordinates(v1,[rr,cc],order=1); a2=ndi.map_coordinates(v2,[rr,cc],order=1); nn=np.hypot(a1,a2)+1e-12
        x0=-L+cc*0.5; y0=-L+rr*0.5; Lr=3.2; ux=a1/nn*Lr; uy=a2/nn*Lr
        bx.quiver(x0-0.5*ux,y0-0.5*uy,ux,uy,angles='xy',scale_units='xy',scale=1,width=0.009,color='black',edgecolor='white',linewidth=0.4,headwidth=3.5,headlength=4,headaxislength=3.5)
    bx.contour(T1,T2,sp,levels=[0],colors='white',linewidths=0.6)
    for sp_ in bx.spines.values(): sp_.set_visible(True); sp_.set_color(C['ink'])
    bx.set(xlabel='$u_1$ (voxels)',ylabel='$u_2$ (voxels)' if j==0 else None); bx.set_title(desc,fontsize=7); bx.grid(False)
    bx.tick_params(labelsize=6); label(bx,'def'[j])
cb=fig.colorbar(im,ax=fig.axes[-1],shrink=0.8,ticks=[-1,0,1],pad=0.02); cb.set_label('$s$',labelpad=1); cb.ax.tick_params(labelsize=6)
save(fig,'FigA3_S3_field',OUT); print('ok')
