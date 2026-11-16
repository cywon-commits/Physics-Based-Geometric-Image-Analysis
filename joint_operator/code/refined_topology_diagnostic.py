"""Post-freeze validation-only interpolation diagnostic. Does not retune or unlock test."""
from evaluate_joint import *
from itertools import permutations
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

def run():
 lock=(OUT/'pretest_selection.sha256').read_text();models=json.load(open(OUT/'trained_models.json'));df=make_manifest();cfg=models['3']['free']['generator'];rows=[]
 for family in ['sphere','capsule','torus']:
  row=df[(df.dim==3)&(df.split=='validation')&(df.family==family)&(df.noise==0)&(df.rep==0)].iloc[0].to_dict();case=rollout(row,cfg)
  for N,n in case['states'].items():rows.append(dict(stage='postfreeze_validation_only',family=family,case=row['case'],N=N,**certificate(n)))
 out=pd.DataFrame(rows);out['refined_nonvanishing']=out.interpolation_norm_lower_bound>1e-7;out.to_csv(OUT/'refined_topology_validation_only.csv',index=False)
 assert lock==hashlib.sha256((OUT/'pretest_selection.json').read_bytes()).hexdigest();print(out[['family','N','interpolation_norm_lower_bound','refined_nonvanishing']].to_string(index=False))
if __name__=='__main__':run()
