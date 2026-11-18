import numpy as np,csv
from curv import site_curvature,tv_weights
MBAR={2:1.0189,3:1.0160}
def fit():
    r=list(csv.DictReader(open('/home/claude/work/data/t8_curvfit.csv'))); P={}
    for d in (2,3):
        q=[x for x in r if int(x['d'])==d]
        k=np.array([float(x['k_true']) for x in q]); m=np.array([float(x['m']) for x in q])-MBAR[d]
        A=np.c_[k,k*k,k**3]; P[d]=np.linalg.lstsq(A,m,rcond=None)[0]
    return P
COEF=fit()
def mfun(k,d):
    k=np.clip(np.abs(k),0,0.45 if d==2 else 0.85); b=COEF[d]; return MBAR[d]+b[0]*k+b[1]*k*k+b[2]*k**3
def boundary_estimate(s,mode='site'):
    s=s.astype(np.float64); d=s.ndim; w=tv_weights(s); k=site_curvature(s)
    if mode=='planar': return w.sum()/MBAR[d]
    if mode=='site': return np.sum(w/mfun(k,d))
