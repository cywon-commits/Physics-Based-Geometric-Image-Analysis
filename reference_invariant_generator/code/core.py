"""Exact forward-mode Jacobian through the synchronous normalized field map.
Spatial coordinates row/y, column/x; vector components follow array axes.
No generator amplitude normalization or calibration enters the map/loss.
"""
from data import *
import time,itertools
G4=np.array([2/3,-1/12,0,0,0,0.]);LEGACY=np.array([1,.57063894,.30414468,.33693168,.09205035,.08973684]);PILOT=np.array([.997721901,.262736838,.618148076,-.053713896,.265501073,-.378951811])

def kernels(p,dim=2):
 pairs=[(a,) for a in range(3)] if dim==2 else list(itertools.combinations_with_replacement(range(3),2));ks=[]
 for axis in range(dim):
  k=np.zeros((5,)*dim)
  transverse=[j for j in range(dim) if j!=axis]
  for ti,pair in enumerate(pairs):
   for di in [1,2]:
    for offs in set(itertools.permutations(pair)):
     for signs in itertools.product(*[([0] if off==0 else [-off,off]) for off in offs]):
      for sign in [-1,1]:
       pos=[2]*dim;pos[axis]+=sign*di
       for j,off in zip(transverse,signs):pos[j]+=off
       k[tuple(pos)]=sign*p[2*ti+di-1]
  ks.append(k)
 return np.stack(ks)

def derivative(s,p):
 # Batch followed by spatial axes and optional trailing tangent axis.
 result=[]
 for axis in [1,2]:
  other=3-axis;d1=np.roll(s,-1,axis)-np.roll(s,1,axis);d2=np.roll(s,-2,axis)-np.roll(s,2,axis);out=0.
  for j in range(3):
   a=p[2*j]*d1+p[2*j+1]*d2
   if j:a=np.roll(a,-j,other)+np.roll(a,j,other)
   out=out+a
  result.append(out)
 return np.stack(result,axis=-1)

def obsder(n,axis):return (2/3)*(np.roll(n,-1,axis)-np.roll(n,1,axis))-(1/12)*(np.roll(n,-2,axis)-np.roll(n,2,axis))
def nn(n):return sum(np.roll(n,k,a) for a in [1,2] for k in [-1,1])

def forward(f,n,p,N=10,jac=False,spacing=1.):
 n=n.copy();B=len(f);P=len(p);dn=np.zeros(n.shape+(P,)) if jac else None;tic=time.perf_counter()
 for _ in range(N):
  g=derivative(n[...,-1],p)/spacing;H=nn(n);H[...,:2]+=g;H[...,-1]+=2*f;norm=np.linalg.norm(H,axis=-1,keepdims=True);valid=norm>1e-12;new=np.where(valid,H/np.maximum(norm,1e-12),n)
  if jac:
   dg=np.moveaxis(derivative(dn[...,-1,:],p),-1,-2)/spacing;basis=np.stack([derivative(n[...,-1],np.eye(P)[j]) for j in range(P)],-1)/spacing;dH=nn(dn);dH[...,:2,:]+=dg+basis;dn=np.where(valid[...,None],(dH-new[...,None]*np.sum(new[...,None]*dH,axis=-2,keepdims=True))/np.maximum(norm[...,None],1e-12),dn)
  n=new
 fieldtime=time.perf_counter()-tic;tic=time.perf_counter();g=[obsder(n,a) for a in [1,2]];T=np.array([[np.sum(x*y,axis=(1,2,3)) for y in g] for x in g]).transpose(2,0,1);I=np.trace(T,axis1=1,axis2=2)*spacing**0;X=T[:,0,0]-T[:,1,1];Y=2*T[:,0,1];theta=.5*np.arctan2(Y,X)+np.pi/2;gap=np.hypot(X,Y)/np.maximum(I,1e-30)
 if not np.isfinite(n).all() or not np.isfinite(I).all() or np.any(I<=1e-14):raise FloatingPointError('Nonfinite/degenerate I; no geometry rejection')
 if jac:
  dg=[obsder(dn,a) for a in [1,2]];dT=np.array([[np.sum(dx*y[...,None]+x[...,None]*dy,axis=(1,2,3)) for y,dy in zip(g,dg)] for x,dx in zip(g,dg)]).transpose(2,0,1,3);dI=dT[:,0,0]+dT[:,1,1];dX=dT[:,0,0]-dT[:,1,1];dY=2*dT[:,0,1];dtheta=.5*(X[:,None]*dY-Y[:,None]*dX)/np.maximum((X*X+Y*Y)[:,None],1e-24)
 else:dI=dtheta=None
 return dict(I=I,I_scalar=sum(np.sum(x[...,-1]**2,axis=(1,2)) for x in g),I_vector=sum(np.sum(x[...,:-1]**2,axis=(1,2,3)) for x in g),theta=theta,gap=gap,dI=dI,dtheta=dtheta,unit_error=float(np.max(abs(np.linalg.norm(n,axis=-1)-1))),field_seconds=fieldtime,observer_seconds=time.perf_counter()-tic,state=n)

def evaluate(dataset,p,N=10,jac=False,chunk=8):
 keys=['I','I_scalar','I_vector','theta','gap','dI','dtheta'];out={k:[] for k in keys};field=obstime=0.;uniterror=0
 for i in range(0,len(dataset['f']),chunk):
  a=forward(dataset['f'][i:i+chunk],dataset['n'][i:i+chunk],p,N,jac);field+=a['field_seconds'];obstime+=a['observer_seconds'];uniterror=max(uniterror,a['unit_error'])
  for k in keys:
   if a[k] is not None:out[k].append(a[k])
 return dict({k:np.concatenate(v) if v else None for k,v in out.items()},field_seconds=field,observer_seconds=obstime,unit_error=uniterror)

def metrics_torch(I,theta,rows,axes,noise_weight):
 """Small differentiable outer loss; rollout's exact Jacobian supplied separately."""
 import torch
 clean=rows.noise.to_numpy()==0;truth=torch.as_tensor(rows.truth.to_numpy(),dtype=I.dtype);units=rows.unit.to_numpy();qmeans=[];rot=[];ang=[];noise=[]
 for u in np.unique(units):
  ix=np.flatnonzero((units==u)&clean);z=I[ix];qmeans.append((z/truth[ix]).mean());rot.append(z.var(unbiased=False)/z.mean()**2)
  ixang=[i for i in ix if axes[i] is not None]
  if ixang:
   gt=torch.as_tensor([np.arctan2(axes[i][1],axes[i][0]) for i in ixang],dtype=I.dtype);err=.5*torch.atan2(torch.sin(2*(theta[ixang]-gt)),torch.cos(2*(theta[ixang]-gt)));ang.append((err/np.radians(1)) .square().mean())
  if noise_weight:
   local=[]
   for i in np.flatnonzero((units==u)&~clean):
    pair=np.flatnonzero((units==u)&clean&(rows.rep.to_numpy()==rows.rep.iloc[i]))[0];local.append((torch.log(I[i])-torch.log(I[pair]))**2)
   if local:noise.append(torch.stack(local).mean())
 logq=torch.log(torch.stack(qmeans));prop=logq.var(unbiased=False);rotation=torch.stack(rot).mean();angle=torch.stack(ang).mean() if ang else I.sum()*0;nl=torch.stack(noise).mean() if noise else I.sum()*0
 total=prop+rotation+.0001*angle+noise_weight*nl
 return total,dict(prop=float(prop.detach()),rotation=float(rotation.detach()),angle=float(angle.detach()),noise=float(nl.detach()))

def objective(dataset,p,noise_weight,jac=True):
 import torch
 a=evaluate(dataset,p,jac=jac);I=torch.tensor(a['I'],requires_grad=jac);theta=torch.tensor(a['theta'],requires_grad=jac);rows=dataset['rows'].copy();rows['truth']=dataset['truth'];loss,parts=metrics_torch(I,theta,rows,dataset['axes'],noise_weight)
 if jac:
  loss.backward();gradient=I.grad.numpy()@a['dI']+(theta.grad.numpy() if theta.grad is not None else np.zeros(len(theta)))@a['dtheta'];return float(loss.detach()),gradient,parts
 return float(loss),parts
