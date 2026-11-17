"""Predeclared diagnostic suites, no fitted kernels or data-dependent choice."""
from pathlib import Path
import numpy as np,pandas as pd,json,hashlib,itertools
from scipy.spatial import ConvexHull
from scipy.special import ellipe
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';OUT.mkdir(exist_ok=True);SEED=17111791

def rotation(d,seed):
 rng=np.random.default_rng(seed)
 if d==2:
  a=rng.uniform(0,np.pi);return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
 Q,R=np.linalg.qr(rng.normal(size=(3,3)));Q=Q@np.diag(np.where(np.diag(R)>=0,1.,-1.));Q[:,0]*=np.linalg.det(Q);return Q

def plane_area(d,side,normal,offset):
 normal=np.asarray(normal,dtype=float);norm=np.linalg.norm(normal)
 if norm<=0:raise ValueError('Plane normal must be nonzero')
 normal=normal/norm;offset=offset/norm
 vertices=np.array(list(itertools.product([-side/2,side/2],repeat=d)));points=[]
 for i,a in enumerate(vertices):
  for b in vertices[i+1:]:
   if np.count_nonzero(a!=b)!=1:continue
   va=a@normal+offset;vb=b@normal+offset
   if abs(va)<1e-12:points.append(a)
   if va*vb<0:points.append(a+(b-a)*(-va)/(vb-va))
   if abs(vb)<1e-12:points.append(b)
 if not points:return 0.
 pts=np.unique(np.round(points,12),axis=0)
 if d==2:return float(np.max(np.linalg.norm(pts[:,None]-pts[None,:],axis=-1))) if len(pts)>1 else 0.
 if len(pts)<3:return 0.
 c=pts.mean(0);u=pts[0]-c;u/=np.linalg.norm(u);v=np.cross(normal,u);xy=np.stack([(pts-c)@u,(pts-c)@v],1);return float(ConvexHull(xy).volume)

def make_manifest():
 p=OUT/'manifest.csv'
 if p.exists():return pd.read_csv(p)
 rows=[];unit=0
 def add(d,suite,family,R=np.nan,params=None,split='diagnostic',widths=(2.3,),spacings=(1.,),rotations=2,shifts=2,noises=(0.,.002,.01),size=None):
  nonlocal unit
  uid=f'rd{unit:04d}';unit+=1
  for w in widths:
   for a in spacings:
    for rot in range(rotations):
     for sh in range(shifts):
      for noise in noises:
       rows.append(dict(case=f'{uid}_w{w}_a{a}_r{rot}_s{sh}_n{noise}',unit=uid,dim=d,suite=suite,family=family,split=split,R=R,params=json.dumps(params or {}),width=w,spacing=a,rotation_id=rot,shift_id=sh,noise=noise,seed=SEED+unit*401+sh*71,rotation_seed=SEED+unit*443+rot*97,noise_seed=SEED+700000+unit*467+rot*101+sh*79,size=size or (80 if d==2 else 64)))
 for d in [2,3]:
  # Width changes are a profile diagnostic, not additional independent shapes.
  add(d,'analytic_planar','plane',params=dict(oracle=True,roi_side=48 if d==2 else 24),widths=(.8,1.6,2.3,4.6),spacings=(1.,.5),rotations=4 if d==2 else 3,noises=(0.,),size=96 if d==2 else 64)
  add(d,'digital_planar','plane',params=dict(oracle=False,roi_side=48 if d==2 else 24),widths=(1.15,2.3,4.6),rotations=4 if d==2 else 3,noises=(0.,),size=96 if d==2 else 64)
  fam='circle' if d==2 else 'sphere'
  for R in ([2.193,3.193,4.693,6.193,9.193,13.193,19.193] if d==2 else [2.293,3.793,5.293,7.793,11.293,16.293]):add(d,'curvature',fam,R,rotations=1)
  for R in ([5.893,10.893,15.893] if d==2 else [4.393,8.893,13.893]):add(d,'reference',fam,R,split='reference',rotations=1,noises=(0.,))
  for R in ([9.693,14.693] if d==2 else [7.393,12.393]):
   for thickness in [.75,1.5,3.,6.]:add(d,'interaction','shell',R,dict(thickness=thickness),rotations=1)
  for gap in [.25,.75,1.5,3.,6.]:add(d,'interaction','near_touch',4.193 if d==2 else 5.393,dict(gap=gap))
  add(d,'shape','pair_far',3.593 if d==2 else 3.493,dict(gap=12.3))
  add(d,'shape','ellipse' if d==2 else 'ellipsoid',6.493,dict(axes=[2.13,.71] if d==2 else [1.93,.83,.57]))
  add(d,'shape','stadium' if d==2 else 'capsule',4.893 if d==2 else 4.493,dict(Lratio=2.23))
  if d==3:
   add(d,'shape','oblate',6.893,dict(axes=[1.57,1.57,.43]));add(d,'shape','torus',12.793,dict(minor=2.593))
  # A true physical re-rasterization of one fixed circle/sphere, not domain padding.
  add(d,'spacing','circle' if d==2 else 'sphere',8.193 if d==2 else 7.293,dict(width_policy='physical'),spacings=(.75,1.,1.5),rotations=1,shifts=1,noises=(0.,))
 df=pd.DataFrame(rows);df.to_csv(p,index=False);(OUT/'manifest.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest());return df

def fixture(row):
 d=int(row['dim']);a=float(row['spacing']);p=json.loads(row['params']);S=int(round(float(row['size'])/a));S+=S%2;x=(np.indices((S,)*d,dtype=float)-(S-1)/2)*a;Q=rotation(d,int(row['rotation_seed']));shift=np.random.default_rng(int(row['seed'])).uniform(-.43,.43,d);family=row['family'];R=float(row['R']);axis=None;roi=None;role='undefined';normal_target=False
 if family=='plane':
  normals2=[np.array([np.cos(t),np.sin(t)]) for t in [0,np.pi/12,np.pi/6,np.pi/4]];normals3=[np.array([1.,0,0]),np.array([1.,.4,.2]),np.ones(3)];normal=(normals2 if d==2 else normals3)[int(row['rotation_id'])];normal=normal/np.linalg.norm(normal);offset=0. if int(row['shift_id'])==0 else .371;level=np.einsum('i,i...->...',normal,x)+offset;mask=level>=0;axis=normal;role='plane_normal';normal_target=True;L=p['roi_side'];m=int(round(L/a));lo=(S-m)//2;roi=(slice(lo,lo+m),)*d;truth=plane_area(d,m*a,normal,offset);oracle=p['oracle'];metadata=dict(normal=normal,offset=offset,roi_side=m*a)
  if oracle:
   z=level/row['width'];s=np.tanh(z);v=np.sqrt(np.maximum(0,1-s*s))[...,None]*normal;n=np.concatenate([v,s[...,None]],-1);return None,n,truth,axis,roi,dict(axis_role=role,normal_target_defined=True,**metadata)
 else:
  for j in range(d):x[j]-=shift[j]
  y=np.einsum('ji,j...->i...',Q,x);u=y[0];rho=np.sqrt(np.sum(y[1:]**2,axis=0));rad=np.linalg.norm(y,axis=0)
  if family in ['circle','sphere']:level=R-rad;truth=2*np.pi*R if d==2 else 4*np.pi*R*R
  elif family=='shell':
   t=p['thickness'];level=t/2-abs(rad-R);truth=4*np.pi*R if d==2 else 4*np.pi*((R+t/2)**2+(R-t/2)**2)
  elif family in ['near_touch','pair_far']:
   sep=2*R+p['gap'];level=np.maximum(R-np.sqrt((u-sep/2)**2+rho*rho),R-np.sqrt((u+sep/2)**2+rho*rho));truth=4*np.pi*R if d==2 else 8*np.pi*R*R;axis=Q[:,0];role='pair_separation'
  elif family in ['ellipse','ellipsoid','oblate']:
   axes=np.array(p['axes'])*R;level=1-np.sum((y/axes.reshape((d,)+(1,)*d))**2,axis=0)
   if d==2:truth=4*axes[0]*ellipe(1-(axes[1]/axes[0])**2)
   else:
    from numpy.polynomial.legendre import leggauss
    z,w=leggauss(120);ph=np.arange(240)*2*np.pi/240;aa,bb,cc=axes;rr=np.sqrt(1-z[:,None]**2);truth=float((w[:,None]*np.sqrt((bb*cc*rr*np.cos(ph))**2+(aa*cc*rr*np.sin(ph))**2+(aa*bb*z[:,None])**2)).sum()*2*np.pi/240)
   axis=Q[:,-1] if family=='oblate' else Q[:,0];role='oblate_normal' if family=='oblate' else 'object_major';normal_target=True
  elif family in ['stadium','capsule']:
   L=p['Lratio']*R;level=R-np.sqrt(np.maximum(abs(u)-L,0)**2+rho*rho);truth=2*np.pi*R+4*L if d==2 else 4*np.pi*R*R+4*np.pi*R*L;axis=Q[:,0];role='object_major';normal_target=True
  elif family=='torus':
   minor=p['minor'];level=minor-np.sqrt((rho-R)**2+u*u);truth=4*np.pi**2*R*minor;axis=Q[:,0];role='symmetry_axis';normal_target=True
  else:raise ValueError(family)
  mask=level>=0;metadata={}
 mask=mask.copy();allowed=np.zeros_like(mask);margin=max(1,int(round(7/a)));allowed[(slice(margin,-margin),)*d]=True;mask^=(np.random.default_rng(int(row['noise_seed'])).random(mask.shape)<float(row['noise']))&allowed;f=np.where(mask,1.,-1.)
 return f,None,float(truth),axis,roi,dict(axis_role=role,normal_target_defined=normal_target,**metadata)
