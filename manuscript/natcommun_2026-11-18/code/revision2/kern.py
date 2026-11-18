"""Gradient stencils: central difference along a, smoothed transversally with [al,1-2al,al] along the other axes (periodic)."""
import numpy as np
def tgrad(s,al):
    out=[]
    for a in range(s.ndim):
        g=(np.roll(s,-1,a)-np.roll(s,1,a))*0.5
        for b in range(s.ndim):
            if b!=a and al!=0: g=al*np.roll(g,1,b)+(1-2*al)*g+al*np.roll(g,-1,b)
        out.append(g)
    return out
def tv(s,al): return 0.5*np.sqrt(sum(x*x for x in tgrad(s,al)))
