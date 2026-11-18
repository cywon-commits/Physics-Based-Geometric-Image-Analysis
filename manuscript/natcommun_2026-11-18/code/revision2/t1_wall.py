"""1D wall theory: continuum fixed-point ODE vs actual lattice update.
ODE (continuum, a=1):  J th'' = h f sin th + D sin th cos th th'
Scaled: Q dQ/dth + kappa sin th cos th Q = sin th, Q=|th'|/sqrt(h/J), kappa=D/sqrt(J h)
=> width = sqrt(J/h) F(kappa)."""
import numpy as np, csv, sys
from scipy.integrate import solve_ivp, quad
def lattice1d(h,D,J=1.0,L=None,tol=1e-13,itmax=400000):
    ell=np.sqrt(J/h); L=L or int(max(200,40*ell*max(1,1+abs(D)/np.sqrt(J*h))))
    L+=L%2
    x=np.arange(L)-L/2+0.5; f=np.where(((x>0)&(x<L/4))|(x<-L/4),1.0,-1.0)  # two walls? keep single via antiperiodic trick
    # use open chain with fixed ends instead
    x=np.arange(L)-L/2+0.5; f=np.sign(x)
    th=np.where(x>0,0.2,np.pi-0.2); v=np.sin(th); s=np.cos(th)
    for it in range(itmax):
        vp=np.r_[v[1:],0.0]; vm=np.r_[0.0,v[:-1]]
        sp=np.r_[s[1:],1.0]; sm=np.r_[-1.0,s[:-1]]
        gs=0.5*(sp-sm)
        Hv=J*(vp+vm)+D*gs; Hs=J*(sp+sm)+h*f
        n=np.hypot(Hv,Hs); vn,sn=Hv/n,Hs/n
        d=max(np.max(abs(sn-s)),np.max(abs(vn-v))); v,s=vn,sn
        if d<tol: break
    return x,v,s,it
def wstats(x,s):
    w=np.abs(np.gradient(s,x)); Z=np.trapezoid(w,x); m=np.trapezoid(w*x,x)/Z
    return m,np.sqrt(np.trapezoid(w*x*x,x)/Z-m*m)
def ode_profile(kappa):
    # scaled units h=J=1 -> ell=1
    def shoot(p):
        rhs=lambda u,y:[y[1],np.sin(y[0])+kappa*np.sin(y[0])*np.cos(y[0])*y[1]]
        e1=lambda u,y:y[0]; e1.terminal=True
        e2=lambda u,y:y[1]; e2.terminal=True
        return solve_ivp(rhs,[0,60],[np.pi/2,-p],events=[e1,e2],max_step=0.01,rtol=1e-11,atol=1e-13)
    lo,hi=1e-6,10
    for _ in range(90):
        p=.5*(lo+hi); s=shoot(p)
        if len(s.t_events[0]): hi=p
        else: lo=p
    s=shoot(lo); u=s.t; th=s.y[0]
    U=np.r_[-u[::-1],u[1:]]; S=np.r_[-np.cos(th[::-1]),np.cos(th[1:])]
    return U,S,lo
if __name__=='__main__':
    rows=[]
    K=quad(lambda p:np.sin(p)*np.cos(p)*np.sin(p/2),0,np.pi/2)[0]
    print('K=',K)
    kap=np.round(np.linspace(-2,2,17),3)
    for k in kap:
        U,S,slope=ode_profile(k); _,wd=wstats(U,S)
        rows.append(dict(source='ode',h=np.nan,D=np.nan,kappa=k,ell=1.0,width_sd_over_ell=wd,center_slope_over=slope/np.sqrt(2),center_shift=0.0))
        print('ode',k,wd,slope/np.sqrt(2),1-K*k,flush=True)
    for h in [0.02,0.08,0.3,1.0,2.0,4.0]:
        for k in [-1.0,0.0,0.7071,1.5]:
            D=k*np.sqrt(h); x,v,s,it=lattice1d(h,D); m,wd=wstats(x,s)
            ell=np.sqrt(1/h); i0=np.argmin(abs(x-0.5))
            rows.append(dict(source='lattice',h=h,D=D,kappa=k,ell=ell,width_sd_over_ell=wd/ell,center_slope_over=np.nan,center_shift=m))
            print('lat',h,round(D,3),k,round(wd/ell,4),it,flush=True)
    with open('data/t1_wall_collapse.csv','w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
