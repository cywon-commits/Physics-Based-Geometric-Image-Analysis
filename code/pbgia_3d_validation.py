import numpy as np, scipy.ndimage as ndi, scipy.optimize as opt, pandas as pd
from pathlib import Path
OUT=Path('/mnt/data'); A=3.; h=1.; eps=.16; N=56
coords=np.indices((N,N,N),dtype=np.float32)
X,Y,Z=(coords[i]-(N-1)/2 for i in range(3))

def field(shape, params, shift=(0,0,0)):
 x,y,z=X-shift[0],Y-shift[1],Z-shift[2]
 if shape=='sphere':
  R=params['R']; d=R-np.sqrt(x*x+y*y+z*z); area=4*np.pi*R**2
 elif shape=='capsule':
  R=params['R']; L=params['L']; axis=np.asarray(params['axis'],dtype=np.float32);axis/=np.linalg.norm(axis)
  t=np.clip(axis[0]*x+axis[1]*y+axis[2]*z,-L/2,L/2)
  xx=x-t*axis[0];yy=y-t*axis[1];zz=z-t*axis[2]
  d=R-np.sqrt(xx*xx+yy*yy+zz*zz); area=4*np.pi*R*(R+L/1.) # capsule area = sphere 4πR² + cylinder 2πR L
  area=4*np.pi*R*R+2*np.pi*R*L
 elif shape=='torus':
  R=params['R'];r=params['r'];d=r-np.sqrt((np.sqrt(x*x+y*y)-R)**2+z*z);area=4*np.pi**2*R*r
 f=np.where(d>=0,1.,-1.).astype(np.float32)
 return d.astype('float32'), f, area

def gradf(f):
 return np.stack([ndi.gaussian_filter(f,1.0,order=[int(k==j) for k in range(3)],mode='nearest') for j in range(3)],axis=-1).astype('float32')

def init(d):
 gd=np.stack(np.gradient(d),axis=-1)
 gd/=np.maximum(np.linalg.norm(gd,axis=-1,keepdims=True),1e-9)
 s=np.tanh(d/2.3)
 v=np.sqrt(np.maximum(0,1-s*s))[...,None]*gd
 n=np.concatenate((v,s[...,None]),axis=-1).astype('float32')
 return n

def energy(n,f,g,components=False):
 ex=sum(np.square(np.roll(n,-1,axis=k)-n).sum(dtype=np.float64) for k in range(3))*A
 ze=h*np.sum(1-f*n[...,3],dtype=np.float64)
 sb=-eps*np.sum(n[...,:3]*g,dtype=np.float64)
 return (ex,ze,sb) if components else ex+ze+sb

def lap(n):
 return sum(np.roll(n,1,axis=k)+np.roll(n,-1,axis=k) for k in range(3))-6*n

def evolve(d,f,steps=250,dt=.022,stopping=False):
 g=gradf(f)
 n=init(d)
 hvec=np.zeros_like(n);hvec[...,3]=h*f;hvec[...,:3]=eps*g
 vals=[]
 for k in range(steps):
  H=2*A*lap(n)+hvec
  H-=np.sum(H*n,axis=-1,keepdims=True)*n
  n+=dt*H
  n/=np.linalg.norm(n,axis=-1,keepdims=True)
  if k in (49,99,149,199,249,349): vals.append((k+1,energy(n,f,g)))
 H=2*A*lap(n)+hvec
 H-=np.sum(H*n,axis=-1,keepdims=True)*n
 rms=np.sqrt(np.mean(np.sum(H*H,axis=-1)))
 return n,energy(n,f,g,True),rms,vals

def tensor(n):
 grads=[(np.roll(n,-1,axis=k)-np.roll(n,1,axis=k))*.5 for k in range(3)]
 T=np.array([[np.sum(grads[i]*grads[j],dtype=np.float64) for j in range(3)] for i in range(3)])
 return T

def q3(n):
 # det([n,dx n,dy n,dz n]); convention
 dx,dy,dz=[(np.roll(n,-1,axis=k)-np.roll(n,1,axis=k))*.5 for k in range(3)]
 a,b,c,d=n[...,0],n[...,1],n[...,2],n[...,3]
 # determinant via 3D Levi-Civita / minors
 vol=np.einsum('...i,...i->...',n,np.stack([
  dx[...,1]*(dy[...,2]*dz[...,3]-dy[...,3]*dz[...,2])-dx[...,2]*(dy[...,1]*dz[...,3]-dy[...,3]*dz[...,1])+dx[...,3]*(dy[...,1]*dz[...,2]-dy[...,2]*dz[...,1]),
 -(dx[...,0]*(dy[...,2]*dz[...,3]-dy[...,3]*dz[...,2])-dx[...,2]*(dy[...,0]*dz[...,3]-dy[...,3]*dz[...,0])+dx[...,3]*(dy[...,0]*dz[...,2]-dy[...,2]*dz[...,0])),
 dx[...,0]*(dy[...,1]*dz[...,3]-dy[...,3]*dz[...,1])-dx[...,1]*(dy[...,0]*dz[...,3]-dy[...,3]*dz[...,0])+dx[...,3]*(dy[...,0]*dz[...,1]-dy[...,1]*dz[...,0]),
 -(dx[...,0]*(dy[...,1]*dz[...,2]-dy[...,2]*dz[...,1])-dx[...,1]*(dy[...,0]*dz[...,2]-dy[...,2]*dz[...,0])+dx[...,2]*(dy[...,0]*dz[...,1]-dy[...,1]*dz[...,0]))
 ],axis=-1))
 return float(np.sum(vol,dtype=np.float64)/(2*np.pi**2))

def sigma1d():
 x=np.arange(-90,90,dtype=np.float64)+.5
 f=np.where(x<0,1.,-1.)
 g=ndi.gaussian_filter1d(f,1.0,order=1,mode='nearest')
 def eg(theta):
  v=np.sin(theta);s=np.cos(theta)
  ex=A*np.square(np.diff(v)).sum()+A*np.square(np.diff(s)).sum()
  return ex+h*np.sum(1-f*s)-eps*np.sum(v*g)
 theta0=np.arccos(np.tanh(-x/2.3))
 res=opt.minimize(eg,theta0,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-13})
 return res.fun,res.success,res.nit

if __name__=='__main__':
 print('sigma',sigma1d(),flush=True)
 cases=[('sphere',{'R':9},(0,0,0)),('sphere',{'R':9},(.35,-.27,.21)),('sphere',{'R':12},(.15,.31,-.22)),('capsule',{'R':7,'L':16,'axis':[1,0,0]},(0,0,0)),('capsule',{'R':7,'L':16,'axis':[np.cos(.49),np.sin(.49),0]},(.3,-.2,.2)),('capsule',{'R':7,'L':16,'axis':[.65,.6,.47]},(-.25,.19,-.32)),('torus',{'R':12,'r':4.5},(.15,-.3,.2))]
 sig=sigma1d()[0]
 rows=[]
 for shape,p,shift in cases:
  d,f,area=field(shape,p,shift)
  n,parts,rms,vals=evolve(d,f,250)
  E=sum(parts)
  T=tensor(n); w,V=np.linalg.eigh(T)
  axis_err=np.nan
  if shape=='capsule':
   v=np.asarray(p['axis']);v=v/np.linalg.norm(v)
   axis_err=np.degrees(np.arccos(np.clip(abs(np.dot(V[:,0],v)),0,1)))
  Eface=sum(np.count_nonzero(np.diff(f,axis=k)) for k in range(3))
  row={'shape':shape,'parameters':str(p),'shift':str(shift),'true_area':area,'magnetic_E':E,'magnetic_area':E/sig,'magnetic_rel_error_pct':100*(E/sig/area-1),'voxel_faces':Eface,'voxel_rel_error_pct':100*(Eface/area-1),'tensor_axis_error_deg':axis_err,'Q3_fd':q3(n),'residual_rms':rms,'E_exchange':parts[0],'E_field':parts[1],'E_SB':parts[2],'energy_trace':str(vals)}
  rows.append(row)
  print(shape,p,shift,'A',round(area,2),'Earea',round(E/sig,2),'err%',round(row['magnetic_rel_error_pct'],2),'axis',round(axis_err,3),'Q',round(row['Q3_fd'],3),'rms',round(rms,5),flush=True)
 pd.DataFrame(rows).to_csv(OUT/'pbgia_3d_validation.csv',index=False)
 print('DONE',flush=True)
