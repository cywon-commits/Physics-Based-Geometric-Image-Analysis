"""New independent illustrative experiments for PBGIA paper revision.
2D stadium: EXACT analytic perimeter for cap radius R and cap-center separation L:
    P=2*pi*R+2*L; study fixes L=2R.
2D generated-field degree: solid-angle lattice quadrature, not analytic texture.
3D generated fields and compactified regular-value S3 degrees; 3D orientation and area.
All synthetic examples. The historic frozen benchmark is unchanged.
"""
import os,sys,csv,math,time
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
from skimage.measure import marching_cubes, mesh_surface_area, perimeter_crofton, euler_number
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
OUT=Path(__file__).resolve().parents[1]; FIG=OUT/'figures'; DATA=OUT/'data'
FIG.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
sys.path.insert(0,str(OUT/'code'))
from real_image_examples import generate_field,tensor_from_vector,angle_delta
# numerical regular-value degree supplied locally in code/pbgia_s3_integer_degree.py
from pbgia_s3_integer_degree import degree
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'savefig.bbox':'tight'})

def stadium(n=160,R=17,L=None,theta=0):
    if L is None: L=2*R
    yy,xx=np.mgrid[:n,:n]
    xx=xx-(n-1)/2;yy=-(yy-(n-1)/2)
    a=np.deg2rad(theta)
    longitudinal=np.cos(a)*xx+np.sin(a)*yy
    transverse=-np.sin(a)*xx+np.cos(a)*yy
    sd=R-np.hypot(np.maximum(np.abs(longitudinal)-L/2,0),transverse)
    return sd>=0,sd

def extent2(sp):
    vr,vc,s=sp
    arr=[s,vr,vc]
    grads=[np.gradient(z,edge_order=1) for z in arr]
    I=sum(np.sum(z*z,dtype=np.float64) for g in grads for z in g)
    gr,gc=np.gradient(s)
    tv=.5*float(np.sum(np.hypot(gr,gc),dtype=np.float64))
    return I,tv

def spin_axis(mask,s):
    # use scalar-field boundary energy as positional weighted covariance (long-axis readout)
    gy,gx=np.gradient(s)
    w=np.hypot(gx,gy)
    yr,xr=np.indices(mask.shape)
    x=xr.astype(float);y=-yr.astype(float)
    w=w/(w.sum()+1e-12)
    mx=np.sum(w*x);my=np.sum(w*y)
    xx=np.sum(w*(x-mx)**2); xy=np.sum(w*(x-mx)*(y-my));yy=np.sum(w*(y-my)**2)
    eig,V=np.linalg.eigh([[xx,xy],[xy,yy]])
    v=V[:,-1]
    return (np.rad2deg(np.arctan2(v[1],v[0]))%180), eig[1]/max(eig[0],1e-9)

def q2_solid_angle(vr,vc,s):
    n=np.stack([vc,-vr,s],axis=-1).astype(np.float64)  # physical x=columns, y=-rows
    # map rows physical y up: orientation cross of x right, y up uses plaquette order (00 -> 01 -> 11 -> 10) with rows downward
    a=n[:-1,:-1];b=n[:-1,1:];c=n[1:,1:];d=n[1:,:-1]
    def omega(p,q,r):
        num=np.einsum('...i,...i->...',p,np.cross(q,r))
        den=1+np.sum(p*q,axis=-1)+np.sum(q*r,axis=-1)+np.sum(r*p,axis=-1)
        return 2*np.arctan2(num,den)
    # x right y up => triangles (a,b,c) is clockwise, so negative sign
    return float(-np.sum(omega(a,b,c)+omega(a,c,d))/(4*np.pi))

def plot1_circle():
    fig,axs=plt.subplots(1,3,figsize=(11.7,3.4),layout='constrained')
    n=260; R=61
    y,x=np.mgrid[:n,:n];x=x-(n-1)/2;y=y-(n-1)/2
    rr=np.hypot(x,y); u=R-rr
    # signed DISTANCE from circle, not elliptical radius
    s=np.tanh(u/3)
    axs[0].imshow(u>=0,cmap='gray',vmin=0,vmax=1);axs[0].set_title('(a) Binary circular domain')
    axs[1].imshow(s,cmap='coolwarm',vmin=-1,vmax=1);axs[1].set_title('(b) Scalar spin component $s$')
    levels=[-4,0,4]
    ax=axs[2];ax.contour(x,y,u,levels=levels,colors=['#e58f44','#272c35','#4b82ce'],linewidths=[2,2.2,2])
    ax.set_aspect('equal'); ax.set_xlim(-87,87);ax.set_ylim(-87,87);ax.set_title('(c) Constant normal-offset wall')
    ax.annotate('',xy=(R+4,0),xytext=(R-4,0),arrowprops={'arrowstyle':'<->','linewidth':1.5,'color':'black'})
    ax.text(R+9,3,'8 px normal band',fontsize=9)
    for ax in axs[:2]:ax.axis('off')
    ax.set_xlabel('x (pixels)');ax.set_ylabel('y (pixels)')
    fig.savefig(FIG/'Fig01_corrected_constant_normal.pdf');fig.savefig(FIG/'Fig01_corrected_constant_normal.png',dpi=140);plt.close(fig)

def plot3_stadium():
    n=160
    orient=[]
    R=17;L=34
    for theta in range(0,180,15):
        mask,_=stadium(n=n,R=R,L=L,theta=theta)
        v=generate_field(mask,nsteps=30)
        estimated,ratio=spin_axis(mask,v[2]); angdiff=float(angle_delta(estimated,theta))
        orient.append({'theta_true_deg':theta,'theta_measured_deg':estimated,'axial_error_deg':angdiff,'cov_eigen_ratio':ratio})
    with open(DATA/'stadium_orientation.csv','w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(orient[0]));wr.writeheader();wr.writerows(orient)
    radii=[9,11,13,15,17,19,21,23,25]
    sizes=[]
    for r in radii:
        for ang in (0,27,54):
            mask,_=stadium(n=n,R=r,L=2*r,theta=ang)
            sp=generate_field(mask,nsteps=30)
            I,TV=extent2(sp)
            ptrue=2*np.pi*r+4*r
            pmc=perimeter_crofton(mask,directions=4)
            axis,_=spin_axis(mask,sp[2])
            sizes.append({'radius_pix':r,'L_pix':2*r,'rotation_deg':ang,'P_true':ptrue,'TV_raw':TV,'I_E':I,
                          'crofton':float(pmc),'axis_est_deg':axis})
    # Independent reference scalar: radii=9 at theta=0; other sizes out-of-sample.
    reference=next(q for q in sizes if q['radius_pix']==9 and q['rotation_deg']==0)
    c_tv=reference['P_true']/reference['TV_raw'];c_e=reference['P_true']/reference['I_E']
    for q in sizes:
        q['P_TV']=c_tv*q['TV_raw'];q['P_energy']=c_e*q['I_E']
        q['relative_TV_pct']=100*(q['P_TV']/q['P_true']-1)
    with open(DATA/'stadium_scaling.csv','w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(sizes[0]));wr.writeheader();wr.writerows(sizes)
    sel=np.array([q['radius_pix']>9 for q in sizes]);x=np.array([q['P_true'] for q in sizes])[sel];y=np.array([q['P_TV'] for q in sizes])[sel]
    a,b=np.polyfit(x,y,1); pred=a*x+b
    residual=float(np.mean(np.abs(y/x-1))*100)
    print('STADIUM',len(orient),'orientation abs MAE',np.mean(np.abs([q['axial_error_deg'] for q in orient])),'TV c',c_tv,'fit',a,b,'rel MARE%',residual,flush=True)
    mask,sd=stadium(n=n,R=17,L=34,theta=30)
    vr,vc,s=generate_field(mask,nsteps=30)
    e0,TV=extent2((vr,vc,s))
    tang,coh,tr,*_=tensor_from_vector(vr,vc,sigma=7)
    fig,axs=plt.subplots(2,3,figsize=(12.2,7.6),layout='constrained')
    axs[0,0].imshow(mask,cmap='gray');axs[0,0].set_title('(a) Input stadium, $R=17$, $L=34$')
    axs[0,1].imshow(s,cmap='coolwarm',vmin=-1,vmax=1);axs[0,1].set_title('(b) Generated $s$, $N=30$')
    axs[0,2].imshow(s,cmap='coolwarm',vmin=-1,vmax=1)
    yx,xx=np.mgrid[3:n:4,3:n:4];sel2=(coh[yx,xx]>.1)&(tr[yx,xx]>np.percentile(tr,20))&(np.abs(s[yx,xx])<.995); theta=tang[yx,xx][sel2]
    dx=np.cos(np.deg2rad(theta))*4.3;dy=-np.sin(np.deg2rad(theta))*4.3
    axs[0,2].quiver(xx[sel2],yx[sel2],dx,dy,angles='xy',scale_units='xy',scale=1,pivot='mid',headwidth=0,headlength=0,headaxislength=0,color='yellow',width=.006)
    axs[0,2].set_title('(c) Tangent directors on wall')
    for ax in axs[0]:ax.axis('off')
    X=np.array([q['theta_true_deg'] for q in orient]);Y=np.array([q['theta_measured_deg'] for q in orient]);ax=axs[1,0]
    ax.plot([0,180],[0,180],'k--',lw=1,label='Ideal');ax.scatter(X,Y,s=24);ax.set(xlabel=r'True $\theta$ (deg)',ylabel=r'Measured $\theta$ (deg)',xlim=(0,180),ylim=(0,180));ax.set_title('(d) Rotation axis recovery');ax.grid(alpha=.2)
    ax=axs[1,1];ax.scatter(x,y,s=20,label='Held-out stadium poses');ax.plot([x.min(),x.max()],[x.min(),x.max()],'k--',label='Ideal');ax.plot(np.sort(x),a*np.sort(x)+b,'-',label=f'Fit $y={a:.3f}x{b:+.2f}$')
    ax.set(xlabel='True stadium perimeter (px)',ylabel='TV-derived perimeter (px)');ax.set_title('(e) Size scaling');ax.legend(fontsize=7);ax.grid(alpha=.2)
    ax=axs[1,2];ax.axhline(0,c='k',ls='--',lw=1);ax.scatter([q['radius_pix'] for q in sizes if q['radius_pix']>9],[q['relative_TV_pct'] for q in sizes if q['radius_pix']>9],s=23)
    ax.set(xlabel='Cap radius $R$ (pixels)',ylabel='Relative TV error (%)');ax.set_title('(f) Error vs. size');ax.grid(alpha=.2)
    fig.savefig(FIG/'Fig03_stadium_quantitative.pdf');fig.savefig(FIG/'Fig03_stadium_quantitative.png',dpi=155);plt.close(fig)
    return {'c_tv':c_tv,'slope':a,'intercept':b,'orientation_MAE':float(np.mean(np.abs([q['axial_error_deg'] for q in orient]))),'perimeter_MARE_heldout':residual,'n_sizes':len(radii),'n_poses':len(sizes)}

def q2_cases():
    n=154; y,x=np.mgrid[:n,:n]
    x=x-(n-1)/2;y=y-(n-1)/2
    rr=np.hypot(x,y)
    cases=[('one disk',rr<25,1,1),('two disks',np.hypot(x-34,y)<19,np.nan,np.nan),('annulus',(rr>17)&(rr<31),0,1)]
    # Correct two disks case
    cases[1]=('two disks',((x-33)**2+y*y<18**2)|((x+33)**2+y*y<18**2),2,2)
    cases.append(('disk + ring',((x+35)**2+y*y<18**2)|((np.hypot(x-33,y)>10)&(np.hypot(x-33,y)<23)),1,2))
    rows=[];spins=[]
    for name,mask,chi,ncomp in cases:
        vr,vc,s=generate_field(mask,nsteps=30)
        q=float(q2_solid_angle(vr,vc,s));gyr,gxc=np.gradient(s)
        dx=np.gradient(np.stack([vc,-vr,s],axis=-1),axis=1);dy=-np.gradient(np.stack([vc,-vr,s],axis=-1),axis=0)
        # derivative convention dx col, dy up
        dens=np.einsum('...i,...i->...',np.stack([vc,-vr,s],axis=-1),np.cross(dx,dy))/(4*np.pi)
        eul=euler_number(mask,connectivity=1)
        rows.append({'case':name,'Q2_solid_angle':q,'input_euler':eul,'input_components':int(ndi.label(mask)[1]),'expected_Euler':chi,'expected_components':ncomp})
        spins.append((name,mask,s,dens))
        print('Q2',name,q,eul,flush=True)
    with open(DATA/'2d_spin_topology.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axs=plt.subplots(4,3,figsize=(10.0,11.5),layout='constrained')
    for i,(name,m,s,dens) in enumerate(spins):
        axs[i,0].imshow(m,cmap='gray');axs[i,1].imshow(s,cmap='coolwarm',vmin=-1,vmax=1)
        maxval=np.percentile(np.abs(dens),99.5)+1e-12
        axs[i,2].imshow(dens,cmap='RdBu_r',vmin=-maxval,vmax=maxval)
        axs[i,0].set_title(f'({chr(97+i*3)}) {name}; $\chi={rows[i]["input_euler"]}$')
        axs[i,1].set_title(f'({chr(98+i*3)}) $s(x,y)$ generated')
        axs[i,2].set_title(f'({chr(99+i*3)}) $q_2$; $Q_2={rows[i]["Q2_solid_angle"]:.4f}$')
        for a in axs[i]:a.axis('off')
    fig.savefig(FIG/'Fig07_2D_generated_Q2.pdf');fig.savefig(FIG/'Fig07_2D_generated_Q2.png',dpi=145);plt.close(fig)
    return rows

def shape3(n=68,kind='sphere'):
    zz,yy,xx=np.indices((n,n,n),dtype=np.float32)
    x=xx-(n-1)/2;y=-(yy-(n-1)/2);z=zz-(n-1)/2
    if kind=='sphere':
        R=12;sd=R-np.sqrt(x*x+y*y+z*z);area=4*np.pi*R*R
    elif kind=='capsule':
        R=8;L=21;a=np.array([.707,.565,.424]);a=a/np.linalg.norm(a)
        t=np.clip(x*a[0]+y*a[1]+z*a[2],-L/2,L/2)
        sd=R-np.sqrt((x-t*a[0])**2+(y-t*a[1])**2+(z-t*a[2])**2)
        area=4*np.pi*R*R+2*np.pi*R*L
    elif kind=='torus':
        R=14;r=5;sd=r-np.sqrt((np.hypot(x,y)-R)**2+z*z);area=4*np.pi**2*R*r
    elif kind=='pair':
        s1=10-np.sqrt((x-13)**2+y*y+z*z)
        s2=10-np.sqrt((x+13)**2+y*y+z*z)
        sd=np.maximum(s1,s2);area=8*np.pi*10**2
    elif kind=='fibers':
        # 4 separated capped cylindrical fibers in 3D, mostly parallel to z but tilted ~25 degrees
        r=3.8; L=30;a=np.array([.31,.15,.939]);a=a/np.linalg.norm(a)
        sd=np.full_like(x,-100.0)
        centers=[(-14,-14,0),(14,-14,0),(-14,14,0),(14,14,0)]
        for xc,yc,zc in centers:
            X=x-xc;Y=y-yc;Z=z-zc
            t=np.clip(X*a[0]+Y*a[1]+Z*a[2],-L/2,L/2)
            dd=r-np.sqrt((X-t*a[0])**2+(Y-t*a[1])**2+(Z-t*a[2])**2)
            sd=np.maximum(sd,dd)
        area=4*(4*np.pi*r*r+2*np.pi*r*L)
    elif kind=='porous':
        rng=np.random.default_rng(51)
        noise=ndi.gaussian_filter(rng.standard_normal((n,n,n)).astype(np.float32),sigma=3)
        noise=noise/np.std(noise)
        sd=(np.sqrt(x*x+y*y+z*z)<23)&(noise<.4)
        return sd.astype(bool),None,None
    else:raise ValueError(kind)
    return (sd>=0),sd,area

def generate3(mask,nsteps=30,w_init=2.3):
    f=np.where(mask,1.,-1.).astype('float32')
    sdf=ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask)
    s=np.tanh(sdf/w_init).astype('float32')
    G=np.gradient(s); norm=np.sqrt(sum(z*z for z in G))
    # axes array order z,y,x; spin (vx,vy,vz,s) physical x,y,z
    v=[G[2]*np.sqrt(np.maximum(0,1-s*s))/np.maximum(norm,1e-12),
       -G[1]*np.sqrt(np.maximum(0,1-s*s))/np.maximum(norm,1e-12),
       G[0]*np.sqrt(np.maximum(0,1-s*s))/np.maximum(norm,1e-12)]
    n=np.stack(v+[s],axis=-1).astype('float32')
    # For numerical regular-value count: order of array axes z,y,x affects degree sign; report absolute unless orientation selected
    for _ in range(nsteps):
        sn=n[...,3]
        Hn=sum(np.roll(n,1,axis=k)+np.roll(n,-1,axis=k) for k in range(3))
        Gz=(np.roll(sn,-1,axis=0)-np.roll(sn,1,axis=0))*.5
        Gy=(np.roll(sn,-1,axis=1)-np.roll(sn,1,axis=1))*.5
        Gx=(np.roll(sn,-1,axis=2)-np.roll(sn,1,axis=2))*.5
        Hn[...,0]+=Gx; Hn[...,1]-=Gy;Hn[...,2]+=Gz
        Hn[...,3]+=2*f
        norm=np.linalg.norm(Hn,axis=-1,keepdims=True)
        n=np.where(norm>1e-12,Hn/np.maximum(norm,1e-12),n)
    # exactly compactify outer collar, already essentially at bulk pole
    for k in range(3):
        sl1=[slice(None)]*4;sl1[k]=slice(0,3)
        sl2=[slice(None)]*4;sl2[k]=slice(-3,None)
        n[tuple(sl1)]=np.array([0,0,0,-1],np.float32)
        n[tuple(sl2)]=np.array([0,0,0,-1],np.float32)
    return n

def extent3(n):
    # derivative Gram ? consistent only within this study; calibration on one reference sphere
    grad=np.gradient(n,axis=(0,1,2))
    return float(sum(np.sum(g*g,dtype=np.float64) for g in grad))

def q3fd(n):
    gz,gy,gx=np.gradient(n,axis=(0,1,2))
    # physical xyz axis -> dx,d(-row),dz; n components x,y,z,s
    A=np.stack([n,gx,-gy,gz],axis=-2)
    det=np.linalg.det(A)
    return float(np.sum(det,dtype=np.float64)/(2*np.pi**2))

def tensor3(n):
    v=n[...,:3].astype(np.float64)
    w=np.maximum(0,1-n[...,3]**2)
    V=v.reshape(-1,3); W=w.reshape(-1)
    return np.einsum('ni,nj,n->ij',V,V,W)/np.sum(W)

def plot7_3d():
    kinds=['sphere','capsule','torus','pair','fibers','porous'];rows=[];n=68
    fields={}
    for kind in kinds:
        mask,sd,area=shape3(n,kind)
        field=generate3(mask)
        I=extent3(field)
        mesharea=float('nan')
        try:
            vs,faces,_,_=marching_cubes(mask.astype(np.float32),level=.5)
            mesharea=float(mesh_surface_area(vs,faces))
        except Exception:pass
        deg=degree(field)
        qfd=q3fd(field)
        T=tensor3(field);vals,V=np.linalg.eigh(T)
        target_err=float('nan')
        if kind=='fibers':
            target=np.array([.31,.15,.939]);target=target/np.linalg.norm(target)
            # tensor is normal-dominated, fiber long axis=MIN eigenvector
            target_err=float(np.degrees(np.arccos(np.clip(abs(V[:,0]@target),0,1))))
        print('3D',kind,'degree',deg,'Qfd',qfd,'I',I,'mesh',mesharea,'area',area,'axis',target_err,flush=True)
        rows.append({'kind':kind,'input_voxel_euler':int(euler_number(mask,connectivity=1)),
                     'input_components':int(ndi.label(mask)[1]),'S3_integer_degree':int(deg['integer_degree']),
                     'S3_preimage_hits':int(deg['number_of_preimages']),
                     'Q3_finite_difference':qfd,'reference_area_analytic':area,
                     'surface_mesh_area_voxel':mesharea,'gradient_energy':I,
                     'fiber_min_eig_axis_err_deg':target_err,
                     'interpolation_near_singular_count':deg['near_degenerate_tetrahedra']})
        if kind!='porous':fields[kind]=(mask,field)
    c=rows[0]['reference_area_analytic']/rows[0]['gradient_energy']
    for row in rows:
        row['area_energy_calibrated']=float(c*row['gradient_energy'])
        row['relative_error_analytic_pct']=float(100*(row['area_energy_calibrated']/row['reference_area_analytic']-1)) if row['reference_area_analytic'] is not None else float('nan')
    with open(DATA/'3d_synthetic_validation.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axs=plt.subplots(2,3,figsize=(10.5,7.5),layout='constrained')
    for i,k in enumerate(['sphere','capsule','torus','pair']):
        mask,spin=fields[k]; c=rows[i]
        ax=axs.flat[i];sl=n//2
        ax.imshow(spin[sl,...,3],vmin=-1,vmax=1,cmap='coolwarm')
        ax.contour(mask[sl],levels=[.5],colors='black',linewidths=.9)
        ax.set_title(f'({chr(97+i)}) {k}: integer degree={c["S3_integer_degree"]}\nfinite-difference integral={c["Q3_finite_difference"]:.3f}')
        ax.axis('off')
    k='fibers';mask,spin=fields[k];ax=axs.flat[4]
    ax.imshow(spin[n//2,:,:,3],vmin=-1,vmax=1,cmap='coolwarm');ax.contour(mask[n//2],levels=[.5],colors='black')
    ax.set_title(f'(e) 3D fibers: axis error {rows[4]["fiber_min_eig_axis_err_deg"]:.2f} deg');ax.axis('off')
    ax=axs.flat[5];xx=np.arange(4);vals=[r['S3_integer_degree'] for r in rows[:4]];fd=[r['Q3_finite_difference'] for r in rows[:4]]
    ax.plot(xx,vals,'o-',label='Regular-value degree');ax.plot(xx,fd,'s--',label='Finite difference integral')
    ax.set_xticks(xx);ax.set_xticklabels(['Sphere','Capsule','Torus','Pair'],rotation=20);ax.set_ylabel('Compactified $S^3$ degree');ax.set_title('(f) Topological consistency');ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.savefig(FIG/'Fig07_3D_generated_Q3.pdf');fig.savefig(FIG/'Fig07_3D_generated_Q3.png',dpi=145);plt.close(fig)
    # extra 3D surface and vector visualization: render actual S3 in 3D by scalar color + vector arrows on cut planes
    fig=plt.figure(figsize=(11.3,4.3),layout='constrained'); names=['capsule','torus','fibers']
    for i,kind in enumerate(names):
        mask,spin=fields[kind]
        ax=fig.add_subplot(1,3,i+1,projection='3d')
        vertices,faces,_,_=marching_cubes(mask.astype('float32'),level=.5)
        # surface decimated face plot
        faces=faces[::max(1,len(faces)//3000)]
        xyz=vertices[:,[2,1,0]]-n/2
        ax.plot_trisurf(xyz[:,0],xyz[:,1],xyz[:,2],triangles=faces,alpha=.12,color='#6c98c4',linewidth=.05)
        # arrows at s close 0 on central x-plane / thin slice; physically x/right, y/up,z
        sl=n//2; yy,xx=np.mgrid[5:n:5,5:n:5]
        vv=spin[sl,yy,xx]
        sel=np.abs(vv[:, :,3])<.8
        xs=xx[sel]-n/2;ys=yy[sel]-n/2;zs=np.zeros_like(xs)
        vec=vv[sel,:3]
        ax.quiver(xs,ys,zs,vec[:,0],vec[:,1],vec[:,2],length=4,normalize=True,color='#c14c47',linewidth=.8,arrow_length_ratio=.25)
        ax.set_title(kind+' ($s=0$ surface + v arrows)');ax.set_box_aspect((1,1,1));ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
    fig.savefig(FIG/'FigS_3D_volume_S3_visualization.pdf');fig.savefig(FIG/'FigS_3D_volume_S3_visualization.png',dpi=145);plt.close(fig)
    return rows

if __name__=='__main__':
    t0=time.time(); plot1_circle();a=plot3_stadium();b=q2_cases();c=plot7_3d();
    import json
    (DATA/'new_study_summary.json').write_text(json.dumps({'stadium':a,'topo_2d':b,'topo_3d':c,'runtime_seconds':time.time()-t0},indent=2,default=lambda x:float(x) if hasattr(x,'__float__') else str(x)),encoding='utf-8')
    print('DONE total',time.time()-t0,flush=True)
