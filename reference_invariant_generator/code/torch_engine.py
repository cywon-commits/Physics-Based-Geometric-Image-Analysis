"""PyTorch reverse-mode autodiff; fixed NN and fixed G4 observer."""
from core import *
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
BASIS=torch.tensor(np.stack([kernels(np.eye(6)[i]) for i in range(6)]),dtype=torch.float64)

def rollout(f,n,p,N=10):
 K=torch.einsum('p,pijk->ijk',p,BASIS)[:,None];state=n
 for _ in range(N):
  H=sum(torch.roll(state,k,a) for a in [1,2] for k in [-1,1]);s=state[...,-1][:,None];g=F.conv2d(F.pad(s,(2,2,2,2),mode='circular'),K).movedim(1,-1);H=H+torch.cat((g,(2*f)[...,None]),-1);norm=torch.linalg.vector_norm(H,dim=-1,keepdim=True);state=torch.where(norm>1e-12,H/norm.clamp_min(1e-12),state)
 gs=[(2/3)*(torch.roll(state,-1,a)-torch.roll(state,1,a))-(1/12)*(torch.roll(state,-2,a)-torch.roll(state,2,a)) for a in [1,2]];xx=(gs[0]**2).sum((1,2,3));yy=(gs[1]**2).sum((1,2,3));xy=(gs[0]*gs[1]).sum((1,2,3));return xx+yy,.5*torch.atan2(2*xy,xx-yy)+np.pi/2

def objective_autograd(dataset,p,weight):
 # Two pass, exact chain rule with small chunks: bounded memory independent of dataset size.
 params=torch.tensor(p,dtype=torch.float64);Is=[];angles=[];chunk=8
 with torch.no_grad():
  for i in range(0,len(dataset['f']),chunk):
   a,b=rollout(torch.tensor(dataset['f'][i:i+chunk]),torch.tensor(dataset['n'][i:i+chunk]),params);Is.append(a);angles.append(b)
 I=torch.cat(Is).requires_grad_()
 if not torch.isfinite(I).all() or (I<=1e-14).any():raise FloatingPointError('Nonpositive/nonfinite/degenerate integral')
 theta=torch.cat(angles).requires_grad_();rows=dataset['rows'].copy();rows['truth']=dataset['truth'];loss,parts=metrics_torch(I,theta,rows,dataset['axes'],weight);loss.backward();outerI=I.grad;outertheta=theta.grad if theta.grad is not None else torch.zeros_like(theta);grad=np.zeros(6)
 for i in range(0,len(dataset['f']),chunk):
  pp=params.clone().requires_grad_();a,b=rollout(torch.tensor(dataset['f'][i:i+chunk]),torch.tensor(dataset['n'][i:i+chunk]),pp);((a*outerI[i:i+chunk]).sum()+(b*outertheta[i:i+chunk]).sum()).backward();grad+=pp.grad.numpy()
 if not np.isfinite(grad).all():raise FloatingPointError('Nonfinite gradient')
 return float(loss.detach()),grad,parts
