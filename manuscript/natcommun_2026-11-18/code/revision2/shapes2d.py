import numpy as np
def stadium(n,R,th):
    yy,xx=np.mgrid[:n,:n]; xx=xx-(n-1)/2; yy=-(yy-(n-1)/2); a=np.deg2rad(th); L=2*R
    lo=np.cos(a)*xx+np.sin(a)*yy; tr=-np.sin(a)*xx+np.cos(a)*yy
    return R-np.hypot(np.maximum(np.abs(lo)-L/2,0),tr)>=0
