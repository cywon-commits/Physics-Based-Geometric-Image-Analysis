import numpy as np, scipy.ndimage as ndi
def site_curvature(s,sigma=1.0):
    """mean-curvature-like scalar: div(grad s/|grad s|) of a Gaussian-smoothed s; 2D -> kappa, 3D -> 2H"""
    ss=ndi.gaussian_filter(s.astype(np.float64),sigma,mode='wrap')
    g=[(np.roll(ss,-1,a)-np.roll(ss,1,a))*0.5 for a in range(s.ndim)]
    nr=np.sqrt(sum(x*x for x in g))+1e-12
    return -sum((np.roll(g[a]/nr,-1,a)-np.roll(g[a]/nr,1,a))*0.5 for a in range(s.ndim))
def tv_weights(s):
    g=[(np.roll(s,-1,a)-np.roll(s,1,a))*0.5 for a in range(s.ndim)]
    return 0.5*np.sqrt(sum(x*x for x in g))
