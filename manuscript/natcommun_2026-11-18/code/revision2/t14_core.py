"""Gaussian scale space of a director field: core law and pair annihilation.
q0 = exp(2i psi) with psi = sum k_j arg(z - z_j);   q_sigma = G_sigma * q0 (linear).
Core law (isolated |2k|=1): |q_sigma|(r) = F(r/sigma), F(rho)= sqrt(pi/8) rho e^{-rho^2/4}[I0(rho^2/4)+I1(rho^2/4)]."""
import numpy as np, scipy.ndimage as ndi, scipy.special as sp, csv
def F(rho):
    x=rho**2/4; return np.sqrt(np.pi/8)*rho*(sp.i0e(x)+sp.i1e(x))   # i0e = e^{-x} I0
def charges(q):
    """plaquette winding of arg q (=2 psi), returns k = winding/2 on (n-1)x(n-1) plaquettes"""
    a=np.angle(q)
    def d(u,v): w=v-u; return (w+np.pi)%(2*np.pi)-np.pi
    w=d(a[:-1,:-1],a[:-1,1:])+d(a[:-1,1:],a[1:,1:])+d(a[1:,1:],a[1:,:-1])+d(a[1:,:-1],a[:-1,:-1])
    return np.rint(w/(2*np.pi)).astype(int)/2.0
n=512; y,x=np.mgrid[:n,:n]+0.0; c=n/2
# (1) core law
psi=0.5*np.arctan2(y-c,x-c); q0=np.exp(2j*psi)
rows=[]
for s in (4,8,16):
    qs=ndi.gaussian_filter(q0.real,s,mode='nearest')+1j*ndi.gaussian_filter(q0.imag,s,mode='nearest')
    r=np.hypot(x-c,y-c)
    for rr in np.arange(0.5,3.0*s,0.5):
        m=(np.abs(r-rr)<0.5); rows.append(dict(sigma=s,rho=rr/s,amp=np.abs(qs[m]).mean(),theory=F(rr/s)))
err=max(abs(q['amp']-q['theory']) for q in rows); print('core law max abs dev',err)
with open('data/t14_core.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# (2) pair annihilation: +1/2 and -1/2 at separation d
prow=[]
for d in (8,16,24,32,48):
    psi=0.5*np.arctan2(y-c,x-(c-d/2))-0.5*np.arctan2(y-c,x-(c+d/2)); q0=np.exp(2j*psi)
    sc=None
    for s in np.arange(0.5,1.5*d,0.25):
        qs=ndi.gaussian_filter(q0.real,s)+1j*ndi.gaussian_filter(q0.imag,s)
        k=charges(qs); npl=(k!=0).sum()
        if npl==0: sc=s; break
    prow.append(dict(d=d,sigma_c=sc,ratio=sc/d)); print('d',d,'sigma_c',sc,'ratio',sc/d)
with open('data/t14_pair.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(prow[0]));w.writeheader();w.writerows(prow)
