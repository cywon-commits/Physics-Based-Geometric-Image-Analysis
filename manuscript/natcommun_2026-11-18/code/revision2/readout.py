"""Final PBGIA boundary readout (revision 2).
B = sum_x w(x)/m(kappa(x)),  w = 1/2 |G s| (centered differences),
m(kappa) = m_bar + b1 k + b2 k^2 + b3 k^3, k = |div(grad s_sigma/|grad s_sigma|)| (sigma=1), clipped.
m_bar from planar walls (t10_weighted.py); b_i from disk/sphere sweeps (t10_curvfit.csv). No benchmark data used."""
import numpy as np, scipy.ndimage as ndi
MBAR={2:1.018862336220135,3:1.0160235364993773}
COEF={2:(0.03832279585463617,-0.7483716972129485,0.4327409658981928),3:(0.08089726637158838,-0.7269344802036861,0.4169082380177215)}
KMAX={2:0.45,3:0.85}
def grad(s): return [(np.roll(s,-1,a)-np.roll(s,1,a))*0.5 for a in range(s.ndim)]
def tv_density(s): return 0.5*np.sqrt(sum(g*g for g in grad(s)))
def curvature(s,sigma=1.0):
    ss=ndi.gaussian_filter(s,sigma,mode='wrap'); g=grad(ss); nr=np.sqrt(sum(x*x for x in g))+1e-12
    return -sum((np.roll(g[a]/nr,-1,a)-np.roll(g[a]/nr,1,a))*0.5 for a in range(s.ndim))
def m_of_k(k,d):
    k=np.clip(np.abs(k),0,KMAX[d]); b=COEF[d]; return MBAR[d]+b[0]*k+b[1]*k*k+b[2]*k**3
def boundary(s,curv=True):
    s=np.asarray(s,dtype=np.float64); d=s.ndim; w=tv_density(s)
    return float(np.sum(w/m_of_k(curvature(s),d))) if curv else float(w.sum()/MBAR[d])


# ---- Direction-resolved 2D readout (selected on the benchmark training set; generator h/J=1.5, D/J=1, N=30)
# m(phi,kappa) = c0 + sum_j c_j cos(4 j phi)  (C4v-invariant harmonics, j=1..6, fitted to 46 planar walls)
#               + b1 k + b2 k^2 + b3 k^3     (residual curvature law, fitted to disks)
C2_DIR=[1.0129386834703689, 0.002141680565239055, -0.00489617503219151, -0.0017921457701057337, -0.0031126180624257613, -0.0005470763847376691, -0.0017217197777804445]
B2_DIR=[0.05714424276442897, -1.1520158680456973, 0.9241351646791964]
def boundary2d_dir(s,sigma=1.0):
    s=np.asarray(s,dtype=np.float64); w=tv_density(s)
    ss=ndi.gaussian_filter(s,sigma,mode='wrap'); g=grad(ss); nr=np.sqrt(sum(x*x for x in g))+1e-12; n=[x/nr for x in g]
    k=np.clip(np.abs(-sum((np.roll(n[a],-1,a)-np.roll(n[a],1,a))*0.5 for a in range(2))),0,0.45)
    phi=np.arctan2(n[0],n[1]); mp=C2_DIR[0]+sum(C2_DIR[j]*np.cos(4*j*phi) for j in range(1,len(C2_DIR)))
    return float(np.sum(w/(mp+B2_DIR[0]*k+B2_DIR[1]*k*k+B2_DIR[2]*k**3)))
