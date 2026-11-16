"""Minimal reproducible dynamic-grad(s) checkpoint generator; NOT a trained benchmark.

Run: python code/fixed_iteration_checkpoints.py --dim 2 --size 48 --out output/smoke
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

CHECKPOINTS=(5,10,20,30,40)

def synthetic_mask(dim, size):
    coords=np.indices((size,)*dim, dtype=np.float32)
    center=(size-1)/2
    radius=size/5
    if dim==2:
        d=np.sqrt(((coords[0]-center-0.17)**2)+((coords[1]-center+0.22)**2))
    elif dim==3:
        d=np.sqrt(sum((coords[k]-center-[.17,-.22,.11][k])**2 for k in range(3)))
    else: raise ValueError('dim must be 2 or 3')
    return np.where(d<=radius,1.,-1.).astype(np.float32)

def initialize(f, width=2.3):
    signed=ndi.distance_transform_edt(f>0)-ndi.distance_transform_edt(f<0)
    s=np.tanh(signed/width).astype(np.float32)
    g=gradient(s); norm=np.linalg.norm(g,axis=-1,keepdims=True)
    v=np.sqrt(np.maximum(0.,1.-s*s))[...,None]*g/np.maximum(norm,1e-7)
    n=np.concatenate([v,s[...,None]],axis=-1)
    n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-12)
    return n.astype(np.float32)

def gradient(s):
    return np.stack([(np.roll(s,-1,axis=k)-np.roll(s,1,axis=k))*.5 for k in range(s.ndim)],axis=-1)

def one_step(n,f,J=1.,h=2.,D=1.,alpha=1.):
    dim=f.ndim
    H=sum(np.roll(n,1,axis=k)+np.roll(n,-1,axis=k) for k in range(dim))*J
    H[...,:dim]+=D*gradient(n[...,dim])
    H[...,dim]+=h*f
    hn=np.linalg.norm(H,axis=-1,keepdims=True)
    proposal=np.divide(H, np.maximum(hn,1e-12),out=np.zeros_like(H),where=hn>1e-12)
    proposal=np.where(hn>1e-12,proposal,n)
    if alpha!=1.:
        proposal=(1.-alpha)*n+alpha*proposal
        proposal/=np.maximum(np.linalg.norm(proposal,axis=-1,keepdims=True),1e-12)
    return proposal.astype(np.float32)

def generate(f,checkpoints=CHECKPOINTS,J=1.,h=2.,D=1.,alpha=1.):
    n=initialize(f)
    snaps={}; rows=[]
    for t in range(1,max(checkpoints)+1):
        nxt=one_step(n,f,J,h,D,alpha)
        delta=np.linalg.norm(nxt-n,axis=-1)
        if t in checkpoints:
            snaps[t]=nxt.copy()
            segmentation=nxt[...,-1]>0
            reference=f>0
            dice=2*np.logical_and(segmentation,reference).sum()/(segmentation.sum()+reference.sum())
            rows.append(dict(N=t,rms_update=float(np.sqrt(np.mean(delta**2))),max_update=float(np.max(delta)),dice=float(dice),unit_max_error=float(np.max(np.abs(np.linalg.norm(nxt,axis=-1)-1.)))))
        n=nxt
    return snaps,rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--dim',type=int,choices=(2,3),default=2)
    p.add_argument('--size',type=int,default=48);p.add_argument('--out',default='output/smoke')
    p.add_argument('--J',type=float,default=1.);p.add_argument('--h',type=float,default=2.);p.add_argument('--D',type=float,default=1.)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    f=synthetic_mask(a.dim,a.size)
    snaps,rows=generate(f,J=a.J,h=a.h,D=a.D)
    for n,field in snaps.items(): np.savez_compressed(out/f'checkpoint_{n:02d}.npz',n=field,f=f)
    with (out/'checkpoint_metrics.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (out/'metadata.json').write_text(json.dumps(dict(dim=a.dim,size=a.size,J=a.J,h=a.h,D=a.D,checkpoints=list(snaps),note='smoke test only; no learned kernels or independent benchmarking'),indent=2))
    print(rows)
if __name__=='__main__':main()
