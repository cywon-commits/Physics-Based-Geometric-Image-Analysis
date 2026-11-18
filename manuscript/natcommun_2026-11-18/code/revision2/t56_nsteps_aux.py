"""Per-N: wall-normal angle error (ellipses), fixed-point residual, noise removal (spurious components), degree."""
import sys,json,numpy as np,pandas as pd,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code/revision2','/home/claude/work/manuscript_v2/code']
from real_image_examples import generate_field
from readout import tv_density
import new_quantitative_studies as nq
NS=[0,1,2,3,5,8,12,20,30,50,100,200]
def residual(vr,vc,s,f,J=1.0,D=1.0,h=1.5):
    R=lambda a,k,ax: np.roll(a,k,ax)
    Hr=J*(R(vr,-1,0)+R(vr,1,0)+R(vr,-1,1)+R(vr,1,1))+D*(R(s,-1,0)-R(s,1,0))*0.5
    Hc=J*(R(vc,-1,0)+R(vc,1,0)+R(vc,-1,1)+R(vc,1,1))+D*(R(s,-1,1)-R(s,1,1))*0.5
    Hs=J*(R(s,-1,0)+R(s,1,0)+R(s,-1,1)+R(s,1,1))+h*f; H=np.sqrt(Hr*Hr+Hc*Hc+Hs*Hs)
    return np.sqrt((vr-Hr/H)**2+(vc-Hc/H)**2+(s-Hs/H)**2)
rng=np.random.default_rng(55601); n=128; rows=[]
for inst in range(40):
    t=rng.uniform(0,np.pi); sh=rng.uniform(-.5,.5,2); R=float(np.exp(rng.uniform(np.log(6),np.log(26)))); ar=rng.uniform(1.5,3)
    a=min(R*np.sqrt(ar),50); b=R/np.sqrt(ar)
    y,x=np.mgrid[:n,:n]+0.0; y=y-(n-1)/2-sh[0]; x=x-(n-1)/2-sh[1]; c,s_=np.cos(t),np.sin(t); u=c*x+s_*y; v=-s_*x+c*y
    F=(u/a)**2+(v/b)**2; m0=F<=1
    # inward normal = -grad F (row, col components)
    Fu=2*u/a**2; Fv=2*v/b**2; gx=Fu*c-Fv*s_; gy=Fu*s_+Fv*c   # d/dx, d/dy (x=col, y=row)
    nr_,nc_=-gy,-gx; nn=np.hypot(nr_,nc_)+1e-12; nr_/=nn; nc_/=nn
    for p in (0.0,0.01):
        m=m0^(rng.random(m0.shape)<p) if p>0 else m0; f=np.where(m,1.0,-1.0)
        ncomp_true=1
        for N in NS:
            vr,vc,s=[z.astype(np.float64) for z in generate_field(m,N,h=1.5)]
            w=tv_density(s); band=np.abs(F-1)<0.6*(2/b)*3       # pixels near the true boundary
            vv=np.hypot(vr,vc)+1e-12; cosang=np.clip((vr*nr_+vc*nc_)/vv,-1,1); ang=np.degrees(np.arccos(cosang))
            ww=w*band; err=np.sqrt(np.sum(ww*ang**2)/ww.sum())
            res=residual(vr,vc,s,f); wall=w>0.05
            lab,nl=ndi.label(s>0); _,nh=ndi.label(s<=0)
            Q=nq.q2_solid_angle(vr.astype(np.float32),vc.astype(np.float32),s.astype(np.float32))
            rows.append(dict(inst=inst,noise=p,N=N,ang_rms=err,res_mean=float(res.mean()),res_wall=float(res[wall].mean()),res_max=float(res.max()),ncomp=nl,nhole=nh-1,Q=Q,bmin=b))
        if True:   # baseline: Gaussian gradient of the mask
            for sg in (1.0,1.5):
                g=ndi.gaussian_filter(m.astype(float),sg); gr,gc=np.gradient(g); w=np.hypot(gr,gc); band=np.abs(F-1)<0.6*(2/b)*3
                cosang=np.clip((gr*nr_+gc*nc_)/(w+1e-12),-1,1); ang=np.degrees(np.arccos(cosang)); ww=w*band
                rows.append(dict(inst=inst,noise=p,N=-1,method=f'gauss{sg}',ang_rms=np.sqrt(np.sum(ww*ang**2)/ww.sum())))
df=pd.DataFrame(rows); df.to_csv('data/t56_nsteps_aux.csv',index=False)
print(df[df.N>=0].groupby(['noise','N'])[['ang_rms','res_mean','res_wall','res_max','ncomp','nhole','Q']].mean().round(4).to_string())
print(df[df.N<0].groupby(['noise','method']).ang_rms.mean())
