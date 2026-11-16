from run_research import *
from itertools import permutations

def q3(n):
 g=[deriv(n,k) for k in range(3)]
 M=np.stack([n]+g,axis=-1).astype(float)
 return float(np.linalg.det(M).sum()/(2*np.pi**2))

def certificate(n):
 # Every tetrahedron in an open hemisphere implies its normalized interpolation never vanishes.
 worst=np.inf;failed=0;minnorm=np.inf
 for perm in permutations(range(3)):
  offsets=np.zeros((4,3),int)
  for j,a in enumerate(perm):offsets[j+1]=offsets[j];offsets[j+1,a]+=1
  vs=np.stack([n[tuple(slice(o,sz-1+o) for sz,o in zip(n.shape[:3],off))] for off in offsets],-2)
  center=vs.sum(-2);dots=np.sum(vs*center[...,None,:],-1);margin=dots.min(-1)
  worst=min(worst,float(margin.min()));bad=margin<=1e-8;failed+=int(bad.sum())
  # Lower bound for certified tetrahedra: norm >= min_i vi dot unit(sum vi).
  good=~bad
  if good.any():minnorm=min(minnorm,float((margin[good]/np.linalg.norm(center[good],axis=-1)).min()))
  # Exact closest-point active-face enumeration for uncertified tetrahedra.
  vbad=vs[bad].astype(float)
  if len(vbad):
   from itertools import combinations
   best=np.full(len(vbad),np.inf)
   for size in range(1,5):
    for inds in combinations(range(4),size):
     v=vbad[:,inds,:];G=v@v.transpose(0,2,1)
     A=np.zeros((len(v),size+1,size+1));A[:,:size,:size]=G;A[:,:size,size]=1;A[:,size,:size]=1
     rhs=np.zeros(size+1);rhs[-1]=1
     sol=(np.linalg.pinv(A,rcond=1e-12)@rhs)[:,:size]
     admissible=(sol.min(1)>=-1e-8)&(abs(sol.sum(1)-1)<1e-6)
     val=np.linalg.norm(np.einsum('ni,nij->nj',sol,v),axis=1)
     best=np.minimum(best,np.where(admissible,val,np.inf))
   minnorm=min(minnorm,float(best.min()))
 return dict(hemisphere_min_dot=worst,uncertified_tetrahedra=failed,interpolation_norm_lower_bound=minnorm)

def cap(n,width=3):
 # Collapse a negative-s outer collar continuously to south pole.
 out=n.copy();idx=np.indices(n.shape[:3]);edge=np.minimum(idx,np.array(n.shape[:3])[:,None,None,None]-1-idx).min(0);w=np.clip(edge/width,0,1)[...,None]
 pole=np.zeros(4);pole[-1]=-1;out=w*out+(1-w)*pole;out/=np.linalg.norm(out,axis=-1,keepdims=True)
 return out

def run():
 rows=[];targets=[(.37139,-.52177,.64829,.40981),(-.6,.2,.3,.7),(.2,.8,-.5,-.2)]
 for path in sorted(list(OUT.glob('p0_*_t*.npz'))+list(OUT.glob('supp_*_t*.npz'))):
  name=path.name.rsplit('_t',1)[0];arr=np.load(path);n=arr['n']
  # audit initial and last checkpoint only
  paths=sorted(OUT.glob(name+'_t*.npz'))
  if path not in [paths[0],paths[-1]]:continue
  cc=certificate(n);nc=cap(n);collar=np.ones(n.shape[:3],bool);collar[3:-3,3:-3,3:-3]=False
  for target in targets:
   rows.append(dict(run=name,file=path.name,stage='initial' if path==paths[0] else 'final',target=str(target),Q3_fd=q3(n),collar_max_s=float(n[...,-1][collar].max()),cap_max_change=float(np.linalg.norm(nc-n,axis=-1).max()),**cc,**degree(nc,target)))
  pd.DataFrame(rows).to_csv(OUT/'topology_audit.csv',index=False)
 print('AUDIT',len(rows),flush=True)

if __name__=='__main__':run()
