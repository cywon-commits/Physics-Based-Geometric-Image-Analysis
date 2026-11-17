"""Verified 3D partial work only. This is not an area optimization or test study."""
from core import *
import torch
import torch.nn.functional as F

def run():
 torch.set_num_threads(1);rng=np.random.default_rng(SEED);f=np.where(rng.random((1,8,8,8))>.7,1.,-1.);n=np.stack([init(f[0])]);base=np.r_[G4[:2],np.zeros(10)];basis=torch.tensor(np.stack([kernels(np.eye(12)[j],3) for j in range(12)]));p=torch.tensor(base+rng.normal(0,.01,12),requires_grad=True);ff=torch.tensor(f);initial=torch.tensor(n)
 def loss(p,N=3):
  k=torch.einsum('p,paijk->aijk',p,basis)[:,None];s=initial
  for _ in range(N):
   H=sum(torch.roll(s,sign,axis) for axis in [1,2,3] for sign in [-1,1]);g=F.conv3d(F.pad(s[...,-1][:,None],(2,2,2,2,2,2),mode='circular'),k).movedim(1,-1);H=H+torch.cat([g,(2*ff)[...,None]],-1);hn=torch.linalg.vector_norm(H,dim=-1,keepdim=True);s=torch.where(hn>1e-12,H/hn.clamp_min(1e-12),s)
  I=sum(((2/3)*(torch.roll(s,-1,a)-torch.roll(s,1,a))-(1/12)*(torch.roll(s,-2,a)-torch.roll(s,2,a))).square().sum() for a in [1,2,3]);return I,s
 tic=time.perf_counter();I,s=loss(p);I.backward();errors=[]
 for j in range(12):
  delta=torch.eye(12,dtype=torch.float64)[j]*1e-5;fd=float((loss(p.detach()+delta)[0]-loss(p.detach()-delta)[0])/(2e-5));errors.append(abs(fd-p.grad[j].item()))
 assert max(errors)<1e-5;assert float(abs(torch.linalg.vector_norm(s,dim=-1)-1).max())<1e-12
 smoke_seconds=time.perf_counter()-tic
 import resource
 ff=torch.tensor(np.where(rng.random((1,32,32,32))>.7,1.,-1.));initial=torch.tensor(np.stack([init(ff.numpy()[0])]))
 pb=p.detach().requires_grad_();bench=time.perf_counter();bi,_=loss(pb,10);bi.backward();benchmark_seconds=time.perf_counter()-bench
 result=dict(benchmark_32cubed_N10_forward_backward_seconds=benchmark_seconds,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,benchmark_note='Single synthetic32^3 I gradient, not geometry accuracy or full training runtime. No throughput extrapolation claimed.',status='verified_partial_3D_NOT_trained_NOT_tested',parameters=12,G4_embedded=True,N_smoke=3,grid=[8,8,8],max_FD_gradient_absolute_error=max(errors),finite_gradients=True,unit_error=float(abs(torch.linalg.vector_norm(s,dim=-1)-1).max()),seconds=smoke_seconds,blocker='No3D optimizer budget allocated: prioritize 14 full2D L-BFGS-B runs under8GiB CPU; 3D costs grow with grid volume and12 parameters. No claim of hardware impossibility.',remaining=['fresh3D geometry manifest','multi-start3D training','validation/freeze','independent area and reference sphere sweep'])
 (OUT/'status_3d.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':run()
