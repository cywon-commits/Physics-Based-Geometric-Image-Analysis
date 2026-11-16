"""Continue a saved n/mask checkpoint; checkpoint does not contain parameters, supply them explicitly."""
from run_research import *
p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--h',type=float,default=2);p.add_argument('--D',type=float,default=1);p.add_argument('--J',type=float,default=1);p.add_argument('--steps',type=int,default=1000);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
z=np.load(a.checkpoint);n=z['n'];mask=z['mask'];f=np.where(mask,1.,-1.).astype('float32');rows=[]
for i in range(a.steps):
 nx=update(n,f,h=a.h,D=a.D,J=a.J);rows.append(dict(iteration=i+1,residual=float(np.sqrt(np.mean(np.sum((nx-n)**2,-1))))));n=nx
 if (i+1)%100==0:np.savez_compressed(a.out,n=n,mask=mask)
np.savez_compressed(a.out,n=n,mask=mask);pd.DataFrame(rows).to_csv(a.out.with_suffix('.csv'),index=False)
