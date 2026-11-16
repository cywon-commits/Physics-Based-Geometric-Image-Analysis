"""Continuous measurement truth and predeclared geometry-level splits."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.special import ellipe
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';OUT.mkdir(exist_ok=True,parents=True)
SEED=291116

def make_manifest():
 p=OUT/'manifest.csv'
 if p.exists():return pd.read_csv(p)
 units=[]
 def add(d,f,sp,R,size,**params):units.append(dict(dim=d,family=f,split=sp,R=R,size=size,params=json.dumps(params,sort_keys=True),role='geometry',heldout_family=f in ['ring','peanut','ellipsoid'],spacing=1.))
 for sp,i in [('train',0),('validation',1),('test',2)]:
  add(2,'circle',sp,[9.23,12.63,16.83][i],64)
  add(2,'stadium',sp,[4.37,5.83,7.43][i],80,Lratio=[1.73,2.23,2.63][i])
  add(2,'ellipse',sp,[5.13,7.13,9.43][i],80,axes=[2.17,.61])
  add(2,'multi',sp,[3.13,4.73,6.13][i],72,radii=[1,.81,.43])
  add(3,'sphere',sp,[6.23,8.63,10.73][i],48)
  add(3,'capsule',sp,[3.13,3.93,4.63][i],48,Lratio=[1.43,1.83,2.13][i])
  add(3,'torus',sp,[9.13,11.73,13.43][i],48,minor=[1.83,2.63,3.13][i])
  add(3,'multi',sp,[2.63,3.23,3.93][i],48,radii=[1,.81,.43])
 for R in [8.73,14.13]:add(2,'ring','test',R,64,width_ratio=.17)
 for d in [2,3]:
  for R in ([6.73,12.43] if d==2 else [4.43,7.23]):add(d,'peanut','test',R,64 if d==2 else 48,separation_ratio=1.47)
 for R in [5.13,7.13]:add(3,'ellipsoid','test',R,48,axes=[1.93,.83,.57])
 # Small spheres are descriptive stress units: previous dense small-sphere range was viewed.
 rows=[]
 for i,u in enumerate(units):
  for rep in range(4):
   for ni,noise in enumerate([0.,.002,.01]):rows.append(dict(u,unit=f'gf{i:03d}',case=f'gf{i:03d}_r{rep}_n{ni}',rep=rep,noise=noise,noise_kind='salt_pepper',seed=SEED+i*211+rep*31,noise_seed=SEED+300000+i*227+rep*37+ni))
 for d,f,R,size in [(2,'circle',10.73,64),(3,'sphere',7.23,48)]:rows.append(dict(dim=d,family=f,split='train',R=R,size=size,params='{}',role='reference',heldout_family=False,spacing=1.,unit=f'ref{d}',case=f'ref{d}',rep=0,noise=0.,noise_kind='salt_pepper',seed=SEED+900+d,noise_seed=SEED+1900+d))
 df=pd.DataFrame(rows);df.to_csv(p,index=False);(OUT/'split_lock.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest());return df

def rotation(dim,seed):
 rng=np.random.default_rng(seed)
 if dim==2:
  a=rng.uniform(0,np.pi);return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
 Q,R=np.linalg.qr(rng.normal(size=(3,3)));Q=Q@np.diag(np.where(np.diag(R)>=0,1.,-1.));Q[:,0]*=np.linalg.det(Q);return Q

def fixture(row):
 dim=int(row['dim']);size=int(row['size']);R=float(row['R']);p=json.loads(row['params']) if isinstance(row['params'],str) else row['params'];spacing=float(row.get('spacing',1));rng=np.random.default_rng(int(row['seed']));shift=rng.uniform(-.47,.47,dim);x=np.indices((size,)*dim,dtype=np.float32)-(size-1)/2
 for k in range(dim):x[k]-=shift[k]
 x*=spacing;Q=rotation(dim,int(row['seed'])+781);y=np.einsum('ji,j...->i...',Q,x);u=y[0];rho=np.sqrt(np.sum(y[1:]**2,axis=0));family=row['family'];axis=None
 if family in ['circle','sphere','small']:
  level=R-np.linalg.norm(y,axis=0);truth=2*np.pi*R if dim==2 else 4*np.pi*R**2
 elif family in ['stadium','capsule']:
  L=p['Lratio']*R;level=R-np.sqrt(np.maximum(abs(u)-L,0)**2+rho**2);truth=2*np.pi*R+4*L if dim==2 else 4*np.pi*R**2+4*np.pi*R*L;axis=Q[:,0]
 elif family in ['ellipse','ellipsoid']:
  axes=np.array(p['axes'])*R;level=1-np.sum((y/axes.reshape((dim,)+(1,)*dim))**2,axis=0);axis=Q[:,0]
  if dim==2:truth=4*axes[0]*ellipe(1-(axes[1]/axes[0])**2)
  else:
   from numpy.polynomial.legendre import leggauss
   z,w=leggauss(120);ph=np.arange(240)*2*np.pi/240;a,b,c=axes;rr=np.sqrt(1-z[:,None]**2);truth=float((w[:,None]*np.sqrt((b*c*rr*np.cos(ph))**2+(a*c*rr*np.sin(ph))**2+(a*b*z[:,None])**2)).sum()*2*np.pi/240)
 elif family=='ring':
  level=p['width_ratio']*R-abs(np.linalg.norm(y,axis=0)-R);truth=4*np.pi*R
 elif family=='torus':
  r=p['minor'];level=r-np.sqrt((rho-R)**2+u*u);truth=4*np.pi**2*R*r
 elif family=='multi':
  radii=np.array(p['radii'])*R;centers=np.array([[-2.3,-.5],[1.7,.2],[.5,2.3]])*R
  if dim==3:centers=np.column_stack([centers,np.array([.3,-.6,.8])*R])
  level=np.maximum.reduce([r-np.sqrt(np.sum((y-c.reshape((dim,)+(1,)*dim))**2,axis=0)) for r,c in zip(radii,centers)]);truth=float(np.sum(2*np.pi*radii) if dim==2 else np.sum(4*np.pi*radii**2))
 elif family=='peanut':
  delta=p['separation_ratio']*R;level=np.maximum(R-np.sqrt((u-delta/2)**2+rho**2),R-np.sqrt((u+delta/2)**2+rho**2));truth=4*R*(np.pi-np.arccos(delta/(2*R))) if dim==2 else 4*np.pi*R**2+2*np.pi*R*delta;axis=Q[:,0]
 elif family=='nested':
  rr=np.linalg.norm(y,axis=0);outer,inner,core=p['radii'];mask=((rr<=outer)&(rr>=inner))|(rr<=core);level=np.where(mask,1.,-1.);truth=float(2*np.pi*(outer+inner+core) if dim==2 else 4*np.pi*(outer**2+inner**2+core**2))
 elif family=='near_touch':
  sep=2*R+p['gap'];level=np.maximum(R-np.sqrt((u-sep/2)**2+rho**2),R-np.sqrt((u+sep/2)**2+rho**2));truth=4*np.pi*R if dim==2 else 8*np.pi*R**2
 else:raise ValueError(family)
 clean=level>=0;mask=clean.copy();allowed=np.zeros_like(mask);allowed[(slice(7,-7),)*dim]=True;rate=float(row['noise']);noise_rng=np.random.default_rng(int(row['noise_seed']));kind=row.get('noise_kind','salt_pepper')
 if rate:
  if kind=='salt_pepper':mask^=(noise_rng.random(mask.shape)<rate)&allowed
  elif kind=='correlated':
   import scipy.ndimage as ndi
   noise=ndi.gaussian_filter(noise_rng.normal(size=mask.shape),1.2);noise/=noise.std();proposal=np.where(clean,1.,-1.)+rate*noise;mask=np.where(allowed,proposal>0,clean)
  elif kind=='gaussian_threshold':mask=np.where(allowed,np.where(clean,1.,-1.)+rate*noise_rng.normal(size=mask.shape)>0,clean)
  else:raise ValueError(kind)
 f=np.where(mask,1.,-1.).astype('float32');return f,float(truth),axis,dict(clean=clean,rotation=Q,shift=shift)
