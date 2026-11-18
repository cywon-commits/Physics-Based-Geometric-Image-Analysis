"""Level-weighted coarea readout: B_g = sum |grad s| g(s), with int_{-1}^{1} g = 1.
g_uniform = 1/2 (plain TV/2), g_wall = (3/4)(1-s^2).  Planar metrication and curvature curve recomputed per weight."""
import numpy as np
G={'uniform':lambda s:0.5+0*s,'wall':lambda s:0.75*(1-s*s)}
def wts(s,g):
    gr=[(np.roll(s,-1,a)-np.roll(s,1,a))*0.5 for a in range(s.ndim)]; return np.sqrt(sum(x*x for x in gr))*G[g](s)
