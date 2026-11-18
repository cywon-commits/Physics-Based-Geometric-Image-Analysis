"""Real Lorentz-TEM image of a Neel skyrmion lattice (Argonne SkyrmNet demo image, MIT licence; McCray et al.,
APL Mach. Learn. 2024, doi:10.1063/5.0197138).
Tilted-sample LTEM contrast of Neel skyrmions is approximately a directional derivative of m_z: I ~ d_u m_z.
1) tilt axis u from the second moment of the power spectrum; 2) Tikhonov-regularized Fourier integration -> m_z estimate;
3) PBGIA field from the continuous source; per-skyrmion degree, skyrmion count, sizes, and lattice (5/7) defects."""
import sys,json,numpy as np,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from gray import gen_gray
from scipy.spatial import Delaunay
from skimage.feature import blob_log
A=np.load('/home/claude/arthurmccray/skyrmnet_demo/test_data/test_image_600G_150K.npy').astype(float)
n=A.shape[0]; ky,kx=np.meshgrid(np.fft.fftfreq(n),np.fft.fftfreq(n),indexing='ij'); kr=np.hypot(ky,kx)
F=np.fft.fft2(A-A.mean()); P=np.abs(F)**2
# lattice wavenumber: peak of the radial power spectrum
rb=np.arange(0.005,0.2,0.0025); prof=[P[(kr>=a)&(kr<a+0.0025)].mean() for a in rb]; k0=float(rb[int(np.argmax(prof))]+0.00125)
band=(kr>0.5*k0)&(kr<2*k0); C=np.array([[np.sum(P*band*ky*ky),np.sum(P*band*ky*kx)],[np.sum(P*band*kx*ky),np.sum(P*band*kx*kx)]])
ev,V=np.linalg.eigh(C); u=V[:,-1]; aniso=float(ev[1]/ev[0])
ku=u[0]*ky+u[1]*kx; eps=0.25*k0
M=np.real(np.fft.ifft2(F*(-1j*ku)/(ku**2+eps**2)*(kr>0.35*k0)))       # integrated: d_u M ~ I
M=ndi.gaussian_filter(M,1.5)
def comps(b): return ndi.label(b)[1]
# polarity: skyrmion cores are the minority phase forming isolated disks
q=np.quantile(M,(.3,.7)); M=M if comps(M>q[1])>comps(M<q[0]) else -M
lo,hi=np.quantile(M,(.02,.98)); g=np.clip((M-lo)/(hi-lo),0,1)
# set the cut so that the minority phase fraction matches the integrated contrast (median level)
def field(t):
    c=np.quantile(g,t); return gen_gray(np.clip(2*(g-c)/max(c,1-c),-1,1),h=2.0,full=True)
m=48
plateau=[]
for t in [0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]:
    _,_,st=field(t); lab_,N_=ndi.label(st>0); ce=np.array(ndi.center_of_mass(np.ones_like(st),lab_,range(1,N_+1)))
    plateau.append((t,int(((ce[:,0]>m)&(ce[:,0]<n-m)&(ce[:,1]>m)&(ce[:,1]<n-m)).sum())))
TCUT=0.75
vr,vc,s=field(TCUT)
# solid-angle density per plaquette
nn=np.stack([vc,-vr,s],-1); a_=nn[:-1,:-1];b_=nn[:-1,1:];c_=nn[1:,1:];d_=nn[1:,:-1]
def om(p,q_,r):
    num=np.einsum('...i,...i->...',p,np.cross(q_,r)); den=1+np.sum(p*q_,-1)+np.sum(q_*r,-1)+np.sum(r*p,-1); return 2*np.arctan2(num,den)
qd=-(om(a_,b_,c_)+om(a_,c_,d_))/(4*np.pi)
inner=np.zeros_like(qd,bool); inner[m:-m,m:-m]=True
lab,N=ndi.label(s>0)
cent=np.array(ndi.center_of_mass(np.ones_like(s),lab,range(1,N+1)))
area=ndi.sum(np.ones_like(s),lab,range(1,N+1))
# per-domain degree: sum of the charge density over each domain's Voronoi cell (nearest-domain assignment)
dist,idx=ndi.distance_transform_edt(lab[:-1,:-1]==0,return_indices=True); owner=lab[:-1,:-1][idx[0],idx[1]]
Qdom=ndi.sum(qd,owner,range(1,N+1))
inside=(cent[:,0]>m)&(cent[:,0]<n-m)&(cent[:,1]>m)&(cent[:,1]<n-m)&(area>3)
# reference count: Laplacian-of-Gaussian blobs on the integrated map
bl=blob_log((M-M.min())/np.ptp(M),min_sigma=3,max_sigma=12,num_sigma=10,threshold=0.05)
bl_in=bl[(bl[:,0]>m)&(bl[:,0]<n-m)&(bl[:,1]>m)&(bl[:,1]<n-m)]
# lattice defects from the domain centroids (interior)
pts=cent[inside]; tri=Delaunay(pts); nb=[set() for _ in pts]
for t in tri.simplices:
    for i in range(3):
        for j in range(3):
            if i!=j: nb[t[i]].add(t[j])
z=np.array([len(x) for x in nb]); deep=(pts[:,0]>m+40)&(pts[:,0]<n-m-40)&(pts[:,1]>m+40)&(pts[:,1]<n-m-40)
R_eq=np.sqrt(area[inside]/np.pi)
res=dict(lattice_period_px=float(2/np.sqrt(3)/k0),expected_count_triangular=float((n-2*m)**2/(np.sqrt(3)/2*(2/np.sqrt(3)/k0)**2)),cut_quantile=TCUT,plateau=plateau,tilt_axis_deg=float(np.degrees(np.arctan2(-u[0],u[1]))%180),spectral_anisotropy=aniso,
         Q_total_inner=float(qd[inner].sum()),N_domains_inner=int(inside.sum()),N_LoG_inner=int(len(bl_in)),
         Q_per_domain_mean=float(Qdom[inside].mean()),Q_per_domain_sd=float(Qdom[inside].std()),
         frac_domains_Q_within_0p1_of_1=float(np.mean(np.abs(np.abs(Qdom[inside])-1)<0.1)),
         R_eq_mean=float(R_eq.mean()),R_eq_sd=float(R_eq.std()),
         coord_deep_counts={int(k):int(v) for k,v in zip(*np.unique(z[deep],return_counts=True))})
print(json.dumps(res,indent=1))
np.savez('/home/claude/work/data/t51_ltem.npz',A=A,M=M,s=s,vr=vr,vc=vc,qd=qd,cent=cent,inside=inside,Qdom=Qdom,area=area,pts=pts,z=z,deep=deep,bl=bl)
json.dump(res,open('/home/claude/work/data/t51_ltem.json','w'),indent=1)
