from run_research import *

def ray_width(n,mask):
 d=ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask);g=np.stack(np.gradient(d),-1);g/=np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-12)
 if mask.ndim==3:points=marching_cubes(mask.astype('float32'),.5)[0]
 else:points=np.concatenate(find_contours(mask.astype(float),.5))
 points=points[::max(1,len(points)//500)][:500];directions=np.stack([ndi.map_coordinates(g[...,i],points.T,order=1) for i in range(mask.ndim)],1);directions/=np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-12)
 ts=np.linspace(-6,6,241);coords=points[:,None,:]+ts[None,:,None]*directions[:,None,:];vals=ndi.map_coordinates(n[...,-1],coords.reshape(-1,mask.ndim).T,order=1).reshape(len(points),-1);widths=[]
 for v in vals:
  crossings=[]
  for target in [-.7,.7]:
   inds=np.flatnonzero((v[:-1]<=target)&(v[1:]>target))
   if not len(inds):break
   ii=inds[np.argmin(abs(ts[inds]))];crossings.append(ts[ii]+(target-v[ii])/(v[ii+1]-v[ii])*(ts[ii+1]-ts[ii]))
  if len(crossings)==2 and crossings[1]>crossings[0]:widths.append(crossings[1]-crossings[0])
 return dict(ray_width_median=float(np.median(widths)) if widths else np.nan,ray_width_p10=float(np.quantile(widths,.1)) if widths else np.nan,ray_width_p90=float(np.quantile(widths,.9)) if widths else np.nan,ray_width_valid_fraction=len(widths)/len(points))

def run():
 p0=pd.read_csv(OUT/'p0_metrics.csv')
 for i,c in p0.iterrows():
  arr=np.load(OUT/(c.run+f'_t{int(c.iteration):04d}.npz'))
  for k,v in ray_width(arr['n'],arr['mask']).items():p0.loc[i,k]=v
 p0.to_csv(OUT/'p0_metrics.csv',index=False)
 raw=pd.read_csv(OUT/'benchmark_raw.csv')
 for run,g in raw.groupby('id'):
  arr=np.load(OUT/(run+'_field.npz'));m=ray_width(arr['n'],arr['mask'])
  for k,v in m.items():raw.loc[raw.id==run,k]=v
 raw.to_csv(OUT/'benchmark_raw.csv',index=False)
 print('WIDTH DONE',flush=True)
if __name__=='__main__':run()
