"""Rasterization floor, part 2: information limit for a disk of unknown centre and radius.
Mask consistency is linear in (a,b,c)=(cx,cy,R^2-cx^2-cy^2): the consistent set is a convex polytope.
Uniform prior on (cx,cy,R) restricted to it; its spread in P=2*pi*R is the error no estimator can beat
even when it is told that the object is a disk."""
import numpy as np,pandas as pd
from scipy.optimize import linprog
rng=np.random.default_rng(5); rows=[]
for R in [4,5,6,8,10,13,16,20,26]:
    for rep in range(16):
        c0=rng.uniform(-.5,.5,2); n=int(2*R+8); y,x=np.mgrid[:n,:n]-(n-1)/2.0
        d=np.hypot(x-c0[1],y-c0[0]); M=d<=R; band=np.abs(d-R)<2.0; X=x[band]; Y=y[band]; Mb=M[band]
        # variables z=(a=cy,b=cx,c): inside: Y^2+X^2-2aY-2bX-c<=0 ; outside: -(..)<=-eps
        sg=np.where(Mb,1.0,-1.0); Aub=sg[:,None]*np.c_[-2*Y,-2*X,-np.ones_like(X)]; bub=-sg*(X*X+Y*Y)-(~Mb)*1e-9
        box=[]
        for j in range(3):
            e=np.zeros(3); e[j]=1; lo=linprog(e,A_ub=Aub,b_ub=bub,bounds=[(None,None)]*3).fun; hi=-linprog(-e,A_ub=Aub,b_ub=bub,bounds=[(None,None)]*3).fun; box.append((lo,hi))
        box=np.array(box); acc=[]
        while sum(len(a) for a in acc)<3000:
            Z=rng.uniform(box[:,0],box[:,1],(20000,3)); ok=(Z@Aub.T<=bub[None]).all(1); acc.append(Z[ok])
        Z=np.concatenate(acc); Rs=np.sqrt(Z[:,2]+Z[:,0]**2+Z[:,1]**2)
        # uniform prior in (cx,cy,R): Jacobian dc/dR = 2R -> weight 1/R
        w=1/Rs; w/=w.sum(); P=2*np.pi*Rs; Pm=np.sum(w*P); sd=np.sqrt(np.sum(w*(P-Pm)**2)); T=2*np.pi*R
        rows.append(dict(R=R,rep=rep,rel_std=100*sd/T,rel_halfrange=100*(P.max()-P.min())/2/T,post_mean_abs_err=100*abs(Pm/T-1)))
    df=pd.DataFrame(rows); print(R,df[df.R==R][['rel_std','rel_halfrange','post_mean_abs_err']].mean().round(4).to_dict(),flush=True)
pd.DataFrame(rows).to_csv('data/t31_infofloor.csv',index=False)
