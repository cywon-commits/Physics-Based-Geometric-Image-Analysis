"""Auditable dynamic-normal replication. All lengths in unit grid spacings."""
import os,json,time,argparse,sys
from pathlib import Path
import numpy as np,pandas as pd,scipy.ndimage as ndi
from skimage.measure import euler_number,marching_cubes,mesh_surface_area,find_contours,perimeter_crofton
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pbgia_s3_integer_degree import degree
OUT=Path(__file__).resolve().parents[1]/'results';OUT.mkdir(exist_ok=True)
K={'central':(0,0,0),'Scharr':(0,3/16,0),'fourth':(-1/12,0,0)}
def deriv(a,k,p=(0,0,0),dim=None):
 dim=a.ndim if dim is None else dim;b,t,u=p
 r=(.5-2*b)*(np.roll(a,-1,k)-np.roll(a,1,k))+b*(np.roll(a,-2,k)-np.roll(a,2,k))
 for j in range(dim):
  if j!=k:r=(1-2*t-2*u)*r+t*(np.roll(r,1,j)+np.roll(r,-1,j))+u*(np.roll(r,2,j)+np.roll(r,-2,j))
 return r

def fixture(kind,N=48,R=9,r=4.5,angle=0,shift=(.15,-.3,.2),dim=3):
 x=np.indices((N,)*dim,dtype=np.float32)-(N-1)/2
 for j in range(dim):x[j]-=shift[j]
 axis=np.array([np.cos(angle),np.sin(angle),0.][:dim]);u=np.einsum('i,i...->...',axis,x);rho=np.sqrt(np.maximum(0,(x*x).sum(0)-u*u))
 if kind in ('sphere','circle','small'):d=R-np.sqrt((x*x).sum(0));truth=(4*np.pi*R*R if dim==3 else 2*np.pi*R);ax=None
 elif kind in ('capsule','stadium'):d=R-np.sqrt(rho*rho+np.maximum(abs(u)-R,0)**2);truth=(8*np.pi*R*R if dim==3 else (2*np.pi+4)*R);ax=axis
 elif kind in ('torus','ring'):
  if dim==3:d=r-np.sqrt((rho-R)**2+u*u);truth=4*np.pi**2*R*r
  else:d=r-abs(np.sqrt((x*x).sum(0))-R);truth=4*np.pi*R
  ax=None
 elif kind in ('ellipse','ellipsoid'):
  from scipy.special import ellipe
  axes=np.array([1.6*R,.75*R]+([R] if dim==3 else []));v=-np.sin(angle)*x[0]+np.cos(angle)*x[1]
  q=(u/axes[0])**2+(v/axes[1])**2
  if dim==3:q+=(x[2]/axes[2])**2
  d=(1-np.sqrt(q))*R;ax=axis
  if dim==2:truth=4*axes[0]*ellipe(1-(axes[1]/axes[0])**2)
  else:
   # Deterministic quadrature of exact ellipsoid area element, independent of voxels.
   from numpy.polynomial.legendre import leggauss
   z,w=leggauss(160);ph=np.arange(320)*2*np.pi/320
   v0=np.sqrt(1-z[:,None]**2);a,b,c=axes
   integrand=np.sqrt((b*c*v0*np.cos(ph))**2+(a*c*v0*np.sin(ph))**2+(a*b*z[:,None])**2)
   truth=float((integrand*w[:,None]).sum()*2*np.pi/320)
 else:raise ValueError(kind)
 # Continuous theoretical outward normal, evaluation only (never initialization).
 if kind in ('sphere','circle','small'):normal=np.moveaxis(x,0,-1)
 elif kind in ('capsule','stadium'):normal=np.moveaxis(x-axis.reshape((dim,)+(1,)*dim)*np.clip(u,-R,R),0,-1)
 elif kind=='torus':
  radial=x-axis.reshape((dim,)+(1,)*dim)*u
  normal=np.moveaxis(radial*(1-R/np.maximum(rho,1e-9))+axis.reshape((dim,)+(1,)*dim)*u,0,-1)
 elif kind=='ring':normal=np.moveaxis(x*np.sign(np.sqrt((x*x).sum(0))-R),0,-1)
 else:
  normal=np.zeros(d.shape+(dim,),dtype=np.float32)
  normal[...,0]=u*np.cos(angle)/axes[0]**2-v*np.sin(angle)/axes[1]**2
  normal[...,1]=u*np.sin(angle)/axes[0]**2+v*np.cos(angle)/axes[1]**2
  if dim==3:normal[...,2]=x[2]/axes[2]**2
 normal=normal/np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-9)
 return d,d>=0,normal,float(truth),ax

def initialize(mask):
 dist=ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask);s=np.tanh(dist/2.3).astype('float32')
 g=np.stack([deriv(s,k) for k in range(mask.ndim)],-1);g/=np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-12)
 n=np.concatenate([np.sqrt(1-s*s)[...,None]*g,s[...,None]],-1);return n/np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-12)
def update(n,f,h=2,D=1,J=1,p=(0,0,0),alpha=1,mode='dynamic'):
 dim=f.ndim;H=J*sum(np.roll(n,1,k)+np.roll(n,-1,k) for k in range(dim))
 g=np.stack([deriv(n[...,-1] if mode=='dynamic' else f,k,p) for k in range(dim)],-1)
 H[...,:dim]+=D*g;H[...,-1]+=h*f;norm=np.linalg.norm(H,axis=-1,keepdims=True)
 out=np.where(norm>1e-12,H/np.maximum(norm,1e-12),n)
 if alpha!=1:out=(1-alpha)*n+alpha*out;out/=np.maximum(np.linalg.norm(out,axis=-1,keepdims=True),1e-12)
 return out

def metrics(n,mask,normal,d):
 s=n[...,-1];v=n[...,:-1];vn=np.linalg.norm(v,axis=-1);wall=(abs(s)<.7)&(vn>.04);band=wall&(abs(d)<1.5)
 g=np.stack([deriv(s,k) for k in range(mask.ndim)],-1);gn=np.linalg.norm(g,axis=-1)
 cos=(v*normal).sum(-1)/np.maximum(vn,1e-12);align=(v*g).sum(-1)/np.maximum(vn*gn,1e-12)
 out={'dice':2*((s>0)&mask).sum()/max(1,(s>0).sum()+mask.sum()),'components':ndi.label(s>0)[1],'chi':euler_number(s>0,connectivity=mask.ndim),'input_chi':euler_number(mask,connectivity=mask.ndim),'wall_voxels':wall.sum(),'signed_inward_cos':np.mean(-cos[band]) if band.any() else np.nan,'abs_normal_cos':np.mean(abs(cos[band])) if band.any() else np.nan,'v_grad_s_cos':np.mean(align[wall]) if wall.any() else np.nan,'unit_error':abs(np.linalg.norm(n,axis=-1)-1).max()}
 # Coarea effective width for scalar interval [-.7,.7], actual completed wall.
 out['profile_width']=1.4*wall.sum()/max(float(gn[wall].sum()),1e-12)
 return out

def solve(mask,params,steps=1200,tol=2e-6,callback=None):
 n=initialize(mask);prev=None;f=np.where(mask,1.,-1.).astype('float32');tic=time.perf_counter();hist=[]
 for it in range(steps+1):
  nxt=update(n,f,**params);res=float(np.sqrt(np.mean(np.sum((nxt-n)**2,-1))));mx=float(np.max(np.linalg.norm(nxt-n,axis=-1)))
  cycle=float(np.sqrt(np.mean(np.sum((nxt-prev)**2,-1)))) if prev is not None else np.nan
  hist.append((it,res,mx,cycle))
  if callback and (it in (0,25,100,300,600,steps) or res<tol):callback(it,n,res,mx,cycle)
  if res<tol or it==steps:break
  prev,n=n,nxt
 return n,{'iteration':it,'residual':res,'max_residual':mx,'two_step_residual':cycle,'converged':res<tol,'seconds':time.perf_counter()-tic},hist

def p0():
 rows=[];degrees=[];histories=[]
 cases=[]
 for shape in ['torus','sphere','capsule']:
  for h,D,J in [(2,.16,1),(2,1,1),(2,3,1),(4,1,1),(2,1,.5),(2,1,2)]:cases.append((shape,h,D,J,0,1,'dynamic',12 if shape=='torus' else 7 if shape=='capsule' else 9,4.5))
 for shape in ['torus','sphere','capsule']:
  cases.extend([(shape,2,1,1,.71,1,'dynamic',12 if shape=='torus' else 7 if shape=='capsule' else 9,4.5),(shape,2,1,1,0,1,'static',12 if shape=='torus' else 7 if shape=='capsule' else 9,4.5),(shape,2,3,1,0,.5,'dynamic',12 if shape=='torus' else 7 if shape=='capsule' else 9,4.5)])
 cases.extend([('torus',2,1,1,.71,1,'dynamic',14,3),('small',2,1,1,0,1,'dynamic',2,4.5)])
 for i,(shape,h,D,J,a,alpha,mode,R,r) in enumerate(cases):
  name=f'p0_{i:02d}_{shape}';d,mask,normal,truth,ax=fixture(shape,R=R,r=r,angle=a)
  base=dict(run=name,shape=shape,h=h,D=D,J=J,angle=a,alpha=alpha,mode=mode,R=R,r=r,N=48,seed=61116,spacing=1)
  def cb(it,n,res,mx,cy):
   m=metrics(n,mask,normal,d);m.update(base,iteration=it,residual=res,max_residual=mx,two_step_residual=cy)
   if shape=='torus':
    coords=np.indices(mask.shape,dtype=float)-(48-1)/2;axis=np.array([np.cos(a),np.sin(a),0]);u=np.einsum('i,i...->...',axis,coords);rho=np.sqrt(np.maximum(0,(coords*coords).sum(0)-u*u));m['core_hole_occupancy']=int(((n[...,-1]>0)&(rho<4)&(abs(u)<6)).sum())
   rows.append(m)
   np.savez_compressed(OUT/(name+f'_t{it:04d}.npz'),n=n,mask=mask)
  n,info,hist=solve(mask,dict(h=h,D=D,J=J,alpha=alpha,mode=mode),callback=cb)
  rows[-1].update(info)
  for it,res,mx,cy in hist:histories.append(dict(run=name,iteration=it,residual=res,max_residual=mx,two_step_residual=cy))
  for stage,nn in [('initial',initialize(mask)),('final',n)]:
   for target in [( .37139,-.52177,.64829,.40981),(-.6,.2,.3,.7),(.2,.8,-.5,-.2)]:degrees.append(dict(run=name,stage=stage,target=str(target),**degree(nn,target)))
  pd.DataFrame(rows).to_csv(OUT/'p0_metrics.csv',index=False);pd.DataFrame(histories).to_csv(OUT/'p0_history.csv',index=False);pd.DataFrame(degrees).to_csv(OUT/'p0_degree.csv',index=False)
  print(name,info,flush=True)
 return rows

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['p0']);args=parser.parse_args();p0()
