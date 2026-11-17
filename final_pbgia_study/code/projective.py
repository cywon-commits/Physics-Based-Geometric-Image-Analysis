"""Projective tensor geometry. A fiber average need not be a rank-one projector."""
from common import *

def eigendirector(Q,gap_threshold=.001,trace_threshold=1e-10):
 tr=np.trace(Q,axis1=-2,axis2=-1);normalized=Q/np.maximum(tr[...,None,None],1e-30);vals,vec=np.linalg.eigh(normalized);gap=vals[...,-1]-vals[...,-2];defined=(tr>trace_threshold)&(gap>gap_threshold)&np.isfinite(gap);director=vec[...,:,-1];director=np.where(defined[...,None],director,np.nan);purity=np.where(tr>trace_threshold,np.sum(vals*vals,axis=-1),np.nan)
 return dict(director=director,defined=defined,eigengap=gap,trace=tr,purity=purity,eigenvalues=vals)

def fiber_tensor(n,axis,representation='full',weight='uniform'):
 d=n.ndim-1;v=n if representation=='full' else n[...,:d];w=np.ones(n.shape[:-1]) if weight=='uniform' else np.maximum(0,1-n[...,-1]**2);outer=v[..., :,None]*v[...,None,:];mass=w.sum(axis=axis);Q=np.sum(w[...,None,None]*outer,axis=axis)/np.maximum(mass[...,None,None],1e-30)
 return Q,mass

def loop_invariant(Qs,gap_threshold=.001,trace_threshold=1e-10,min_abs_dot=.2):
 a=eigendirector(Qs,gap_threshold,trace_threshold);d=a['director'];valid=bool(a['defined'].all());result=dict(status='INCONCLUSIVE',min_gap=float(a['eigengap'].min()),min_trace=float(a['trace'].min()),z2_holonomy=np.nan,RP1_winding=np.nan,director_charge=np.nan,min_abs_neighbor_dot=np.nan)
 if not valid:return result
 dots=np.sum(d*np.roll(d,-1,axis=0),axis=-1);result['min_abs_neighbor_dot']=float(abs(dots).min())
 if abs(dots).min()<min_abs_dot:return result
 holonomy=int(np.prod(np.where(dots>=0,1,-1)));result.update(status='DEFINED',z2_holonomy=holonomy)
 if d.shape[-1]==2:
  z=(d[:,0]+1j*d[:,1])**2;w=np.angle(np.roll(z,-1)/z).sum()/(2*np.pi);result.update(RP1_winding=int(round(w)),director_charge=float(round(w)/2))
 return result

def sample_tensor_loop(Q,center,radius,samples=128):
 angles=np.arange(samples)*2*np.pi/samples;coords=np.stack([center[0]+radius*np.cos(angles),center[1]+radius*np.sin(angles)]);m=Q.shape[-1];return np.stack([ndi.map_coordinates(Q[...,i,j],coords,order=1,mode='nearest') for i in range(m) for j in range(m)],-1).reshape(samples,m,m)

def sample_plaquette(Q,i,j,subdivisions):
 t=np.arange(subdivisions)/subdivisions;coords=np.concatenate([np.stack([i+t,np.full_like(t,j)]),np.stack([np.full_like(t,i+1),j+t]),np.stack([i+1-t,np.full_like(t,j+1)]),np.stack([np.full_like(t,i),j+1-t])],axis=1);m=Q.shape[-1];return np.stack([ndi.map_coordinates(Q[...,a,b],coords,order=1,mode='nearest') for a in range(m) for b in range(m)],-1).reshape(len(coords[0]),m,m)

def plaquette_loops(Q):
 a=eigendirector(Q);d=a['director'];valid=a['defined'];rows=[];eligible=0;inconclusive=0;m=Q.shape[-1]
 for i in range(Q.shape[0]-1):
  for j in range(Q.shape[1]-1):
   corners=[(i,j),(i+1,j),(i+1,j+1),(i,j+1)]
   if not all(valid[p] for p in corners):inconclusive+=1;continue
   vectors=np.stack([d[p] for p in corners]);dot=np.sum(vectors*np.roll(vectors,-1,0),axis=-1)
   if abs(dot).min()<.2:inconclusive+=1;continue
   eligible+=1;parity=int(np.prod(np.sign(dot)));w=0
   if m==2:
    z=(vectors[:,0]+1j*vectors[:,1])**2;w=int(round(np.angle(np.roll(z,-1)/z).sum()/(2*np.pi)))
   if parity==-1 or w!=0:
    # Stable classification on two finer loops through the continuous bilinear tensor.
    b=loop_invariant(sample_plaquette(Q,i,j,16));c=loop_invariant(sample_plaquette(Q,i,j,32));stable=b['status']=='DEFINED' and c['status']=='DEFINED' and b['z2_holonomy']==c['z2_holonomy'] and (m!=2 or b['RP1_winding']==c['RP1_winding']);rows.append(dict(row=i,col=j,coarse_z2=parity,coarse_RP1_winding=w,refined_stable=stable,**{f'refined_{k}':v for k,v in c.items()}))
 return rows,dict(valid_plaquettes=eligible,inconclusive_plaquettes=inconclusive,coarse_candidates=len(rows),stable_candidates=sum(x['refined_stable'] and(x['refined_z2_holonomy']==-1 or(m==2 and x['refined_RP1_winding']!=0)) for x in rows))

def controls():
 rows=[];phi=np.arange(128)*2*np.pi/128
 for ambient in [2,3]:
  for charge in [0.,.5,-.5,1.]:
   d=np.stack([np.cos(charge*phi),np.sin(charge*phi)]+([np.zeros_like(phi)] if ambient==3 else []),-1);Q=d[..., :,None]*d[...,None,:];r=loop_invariant(Q);expected_z=-1 if abs(charge)==.5 else 1;expected_w=int(2*charge) if ambient==2 else np.nan;rows.append(dict(control=f'RP{ambient-1}_charge{charge}',ambient=ambient,expected_z2=expected_z,expected_RP1_winding=expected_w,**r,correct=r['status']=='DEFINED' and r['z2_holonomy']==expected_z and(ambient!=2 or r['RP1_winding']==expected_w)))
  Q=np.broadcast_to(np.eye(ambient)/ambient,(128,ambient,ambient));r=loop_invariant(Q);rows.append(dict(control=f'RP{ambient-1}_degenerate',ambient=ambient,expected_z2=np.nan,expected_RP1_winding=np.nan,**r,correct=r['status']=='INCONCLUSIVE'))
 return pd.DataFrame(rows)
