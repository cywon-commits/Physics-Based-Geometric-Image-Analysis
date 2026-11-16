"""Compare dynamic grad(s) against static grad(f) using normalized fixed-point iterations.
No div(v), no claim of minimizing one Hamiltonian. All initialization mask-only.
"""
import numpy as np, scipy.ndimage as ndi, pandas as pd, argparse, time
from pathlib import Path
from skimage.measure import euler_number


def field(shape,N=56):
    xyz=np.indices((N,N,N),dtype=np.float32)-(N-1)/2
    x,y,z=xyz
    if shape=='torus':
        R,r=12.,4.5
        rho=np.sqrt((x-.15)**2+(y+.30)**2)
        d=r-np.sqrt((rho-R)**2+(z-.20)**2)
        xc,yc,zc=x-.15,y+.30,z-.20
        rr=np.sqrt((rho-R)**2+zc**2)
        normal=np.stack([((rho-R)/np.maximum(rho,1e-8))*xc,((rho-R)/np.maximum(rho,1e-8))*yc,zc],axis=-1)/np.maximum(rr[...,None],1e-8)
    elif shape=='sphere':
        xc,yc,zc=x-.15,y+.30,z-.20
        rr=np.sqrt(xc**2+yc**2+zc**2)
        d=9-rr
        normal=np.stack([xc,yc,zc],axis=-1)/np.maximum(rr[...,None],1e-8)
    else: raise ValueError(shape)
    f=np.where(d>=0,1.,-1.).astype(np.float32)
    # no analytic d in initialization
    di=ndi.distance_transform_edt(f>0)-ndi.distance_transform_edt(f<0)
    s=np.tanh(di/2.3).astype(np.float32)
    grad=np.stack(np.gradient(s),axis=-1)
    gnorm=np.linalg.norm(grad,axis=-1,keepdims=True)
    v=np.sqrt(np.maximum(0,1-s*s))[...,None]*grad/np.maximum(gnorm,1e-7)
    n=np.concatenate([v,s[...,None]],axis=-1)
    n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-8)
    return f,normal,n,np.abs(d)<1.25


def neighbors(n):
    return sum(np.roll(n,1,axis=k)+np.roll(n,-1,axis=k) for k in range(3))


def grad(s):
    return np.stack([(np.roll(s,-1,axis=k)-np.roll(s,1,axis=k))*.5 for k in range(3)],axis=-1)


def metrics(n,f,normal,band,shape):
    s=n[...,3];v=n[...,:3]
    mask=s>0; ref=f>0
    vnorm=np.linalg.norm(v,axis=-1)
    wall=band&(np.abs(s)<.7)&(vnorm>.04)
    dot=(v*normal).sum(axis=-1)/np.maximum(vnorm,1e-7)
    g=grad(s);gnorm=np.linalg.norm(g,axis=-1)
    dotvs=(v*g).sum(axis=-1)/np.maximum(vnorm*gnorm,1e-7)
    labels,num=ndi.label(mask,ndi.generate_binary_structure(3,1))
    tp=(mask&ref).sum()
    # gradient evaluation: near original surface and current wall
    out={'dice':float(2*tp/(mask.sum()+ref.sum())),'positive_voxels':int(mask.sum()),'input_positive_voxels':int(ref.sum()),'components':int(num),'euler_26':int(euler_number(mask,connectivity=3)),'abs_normal_alignment':float(np.mean(np.abs(dot[wall]))) if np.any(wall) else np.nan,'signed_inward_alignment':float(np.mean(-dot[wall])) if np.any(wall) else np.nan,'abs_v_grad_s_alignment':float(np.mean(np.abs(dotvs[wall]))) if np.any(wall) else np.nan,'wall_voxels':int(wall.sum())}
    if shape=='torus':
        N=f.shape[0];idx=np.indices(f.shape,dtype=np.float32)-(N-1)/2
        rho=np.sqrt((idx[0]-.15)**2+(idx[1]+.30)**2)
        out['hole_core_occupancy']=int((mask&(rho<4)&(np.abs(idx[2]-.2)<6)).sum())
    return out


def iterate(shape,mode,h,D,J=1.,steps=600,alpha=1.,N=56,record=(0,100,300,600),degree_enabled=False):
    f,normal,n,band=field(shape,N)
    if mode=='static':
        # this is a control for normalized fixed point NOT reproduction of old projected GD
        gfixed=np.stack([ndi.gaussian_filter(f,1.,order=[int(i==j) for i in range(3)],mode='wrap') for j in range(3)],axis=-1)
    records=[];tic=time.monotonic()
    for it in range(steps+1):
        if it in record:
            m=metrics(n,f,normal,band,shape)
            m.update(shape=shape,mode=mode,h=h,D=D,J=J,step=it,alpha=alpha,N=N)
            if degree_enabled:
                from pbgia_s3_integer_degree import degree
                dg=degree(n)
                m['degree']=dg['integer_degree'];m['preimages']=dg['number_of_preimages'];m['boundary_spread']=dg['boundary_spread']
            records.append(m)
            print('RESULT',shape,mode,'h',h,'D',D,'step',it,'dice',round(m['dice'],4),'alignment',round(m['abs_normal_alignment'],4),'chi',m['euler_26'],'degree',m.get('degree','-'),flush=True)
        if it==steps: break
        g=grad(n[...,3]) if mode=='dynamic' else gfixed
        H=J*neighbors(n)
        H[...,:3]+=D*g
        H[...,3]+=h*f
        norm=np.linalg.norm(H,axis=-1,keepdims=True)
        nh=H/np.maximum(norm,1e-12)
        if alpha!=1.:
            nh=(1-alpha)*n+alpha*nh
            nh/=np.maximum(np.linalg.norm(nh,axis=-1,keepdims=True),1e-12)
        n=nh.astype(np.float32)
    records[-1]['seconds']=round(time.monotonic()-tic,2)
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='/mnt/data/pbgia_dynamic_normal_results.csv');p.add_argument('--steps',type=int,default=600);p.add_argument('--N',type=int,default=56);p.add_argument('--degree',action='store_true')
    a=p.parse_args();allrows=[]
    for shape in ['torus','sphere']:
        for h,D in [(2.,.16),(2.,1.),(2.,3.),(4.,1.)]:
            for mode in ['static','dynamic']:
                allrows+=iterate(shape,mode,h,D,steps=a.steps,N=a.N,record=tuple(sorted(set([0,100,300,a.steps]))),degree_enabled=a.degree)
                pd.DataFrame(allrows).to_csv(a.out,index=False)
    print('saved',a.out,flush=True)
if __name__=='__main__':main()
