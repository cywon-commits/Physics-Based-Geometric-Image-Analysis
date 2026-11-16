"""Fixed-iteration PBGIA: exact tensor measurement and binary-EDT initialization."""
from pathlib import Path
import sys,json,time,hashlib,itertools
import numpy as np,pandas as pd
from scipy import ndimage as ndi
from scipy.special import ellipe
from skimage.measure import euler_number,marching_cubes,mesh_surface_area,find_contours,perimeter_crofton
from fixed_iteration_checkpoints import initialize,one_step,gradient,CHECKPOINTS
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';OUT.mkdir(exist_ok=True);CACHE=OUT/'cache';CACHE.mkdir(exist_ok=True)
PARAMS={'J':1.,'h':2.,'D':1.};KERNELS={'central':[0,0,0],'fourth':[-1/12,0,0],'Scharr':[0,3/16,0],'prior_learned':[-.044430528985157536,.11723440324007278,0.]}

def derivative(n,axis,p):
 d=n.ndim-1;b,t,u=p;a=(.5-2*b)*(np.roll(n,-1,axis)-np.roll(n,1,axis))+b*(np.roll(n,-2,axis)-np.roll(n,2,axis))
 for j in range(d):
  if j!=axis:a=(1-2*t-2*u)*a+t*(np.roll(a,1,j)+np.roll(a,-1,j))+u*(np.roll(a,2,j)+np.roll(a,-2,j))
 return a

def tensor(n,p):
 g=[derivative(n,k,p) for k in range(n.ndim-1)]
 return np.array([[np.sum(a*b,dtype=np.float64) for b in g] for a in g])

def features(n):
 # Exact linear derivative basis. Tensor under any admissible kernel is a small quadratic form.
 d=n.ndim-1;fields=[]
 for axis in range(d):
  diffs=[(np.roll(n,-i,axis)-np.roll(n,i,axis)) for i in (1,2)]
  for diff in diffs:
   for choice in itertools.product(range(3),repeat=d-1):
    a=diff
    for j,kind in zip([j for j in range(d) if j!=axis],choice):
     if kind:a=np.roll(a,kind,j)+np.roll(a,-kind,j)-2*a
    fields.append(a.reshape(-1))
 B=np.stack(fields).astype('float64');return B@B.T

def weights(p,d):
 b,t,u=p;s=[1,t,u];return np.array([a*np.prod([s[c] for c in choice]) for a in [.5-2*b,b] for choice in itertools.product(range(3),repeat=d-1)])

def feature_tensor(F,p,d):
 w=weights(p,d);m=len(w);return np.einsum('a,iajb,b->ij',w,F.reshape(d,m,d,m),w,optimize=True)

def orientation(T,axis):
 if axis is None:return np.nan,np.nan
 eig,vec=np.linalg.eigh(T);gap=(eig[1]-eig[0])/max(abs(eig[-1]),1e-12)
 err=float(np.degrees(np.arccos(np.clip(abs(vec[:,0]@axis),0,1))))
 return err,float(gap)

def geometry(row):
 dim=int(row['dim']);size=int(row['size']);R=float(row['R']);kind=row['family'];rep=int(row['rep']);seed=int(row['seed']);spacing=float(row.get('spacing',1));rng=np.random.default_rng(seed)
 x=np.indices((size,)*dim,dtype=np.float32)-(size-1)/2
 shift=rng.uniform(-.45,.45,dim)
 for k in range(dim):x[k]-=shift[k]
 theta=(.17+.64*rep+float(row['phase']))
 if dim==2:
  Q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
 else:
  # Proper rotations with out-of-plane tilt, fully predetermined from the prototype seed.
  axis=np.array([np.cos(theta)*.82,np.sin(theta)*.82,.572]);axis/=np.linalg.norm(axis);v=np.cross([0,0,1],axis);v/=np.linalg.norm(v);Q=np.stack([axis,v,np.cross(axis,v)],1)
 y=np.einsum('ji,j...->i...',Q,x);rho=np.sqrt(np.sum(y[1:]**2,axis=0));axis=None
 if kind in ['circle','sphere','small']:
  d=R-np.linalg.norm(y,axis=0);nu=y;truth=2*np.pi*R if dim==2 else 4*np.pi*R**2
 elif kind in ['stadium','capsule']:
  d=R-np.sqrt(rho**2+np.maximum(abs(y[0])-R,0)**2);nu=y.copy();nu[0]-=np.clip(y[0],-R,R);truth=(2*np.pi+4)*R if dim==2 else 8*np.pi*R**2;axis=Q[:,0]
 elif kind in ['ellipse','ellipsoid']:
  axes=np.array([1.6*R,.75*R]+([R] if dim==3 else []));d=(1-np.sqrt(np.sum((y/axes.reshape((dim,)+(1,)*dim))**2,axis=0)))*R;nu=y/(axes**2).reshape((dim,)+(1,)*dim);axis=Q[:,0]
  if dim==2:truth=4*axes[0]*ellipe(1-(axes[1]/axes[0])**2)
  else:
   from numpy.polynomial.legendre import leggauss
   z,w=leggauss(120);ph=np.arange(240)*2*np.pi/240;a,b,c=axes;rr=np.sqrt(1-z[:,None]**2)
   truth=float((w[:,None]*np.sqrt((b*c*rr*np.cos(ph))**2+(a*c*rr*np.sin(ph))**2+(a*b*z[:,None])**2)).sum()*2*np.pi/240)
 elif kind=='torus':
  r=.30*R;d=r-np.sqrt((rho-R)**2+y[0]**2);nu=y.copy();nu[1:]*=1-R/np.maximum(rho,1e-9);truth=4*np.pi**2*R*r
 elif kind=='ring':
  rr=np.linalg.norm(y,axis=0);r=.28*R;d=r-abs(rr-R);nu=y*np.sign(rr-R);truth=4*np.pi*R
 elif kind=='multi':
  a=y.copy();b=y.copy();a[0]-=2*R;b[0]+=2*R;ra=np.linalg.norm(a,axis=0);rb=np.linalg.norm(b,axis=0);d=R-np.minimum(ra,rb);nu=np.where((ra<rb)[None],a,b);truth=2*(2*np.pi*R if dim==2 else 4*np.pi*R**2)
  # Orientation intentionally undefined for the local full-field tensor on disconnected spheres.
 else:raise ValueError(kind)
 normal=np.einsum('ij,j...->...i',Q,nu);normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-12)
 clean=d>=0;mask=clean.copy();noise=float(row['noise']);valid=np.zeros_like(mask);valid[(slice(7,-7),)*dim]=True
 if noise:mask^=(np.random.default_rng(int(row.get('noise_seed',seed))).random(mask.shape)<noise)&valid
 f=np.where(mask,1.,-1.).astype('float32');return f,clean,normal,d,float(truth*spacing**(dim-1)),axis

def key_for(f,params):
 sha=hashlib.sha256();sha.update(f.tobytes());sha.update(json.dumps({'shape':f.shape,'params':params,'dtype':'float32','init':'supplied_mask_edt_2.3','gradient':'central','boundary':'periodic','checkpoints':CHECKPOINTS,'code':hashlib.sha256(Path(__file__).with_name('fixed_iteration_checkpoints.py').read_bytes()).hexdigest()},sort_keys=True).encode());return sha.hexdigest()

def metrics(n,f,clean,normal,d):
 s=n[...,-1];v=n[...,:-1];vn=np.linalg.norm(v,axis=-1);mask=s>0;ref=f>0;wall=(abs(s)<.7)&(vn>.04);band=wall&(abs(d)<1.5);g=gradient(s);gn=np.linalg.norm(g,axis=-1)
 border=np.ones(f.shape,bool);border[(slice(3,-3),)*f.ndim]=False
 return dict(dice=2*(mask&ref).sum()/max(mask.sum()+ref.sum(),1),clean_dice=2*(mask&clean).sum()/max(mask.sum()+clean.sum(),1),chi=int(euler_number(mask,connectivity=f.ndim)),input_chi=int(euler_number(ref,connectivity=f.ndim)),clean_chi=int(euler_number(clean,connectivity=f.ndim)),components=ndi.label(mask)[1],clean_components=ndi.label(clean)[1],wall_fraction=wall.mean(),wall_voxels=int(wall.sum()),wall_width_coarea=1.4*wall.sum()/max(float(gn[wall].sum()),1e-12) if wall.any() else np.nan,normal_signed_cos=float(np.mean(-np.sum(v*normal,-1)[band]/vn[band])) if band.any() else np.nan,v_grad_s_cos=float(np.mean(np.sum(v*g,-1)[wall]/np.maximum(vn[wall]*gn[wall],1e-12))) if wall.any() else np.nan,unit_error=float(abs(np.linalg.norm(n,axis=-1)-1).max()),negative_collar_max_s=float(s[border].max()),boundary_vector_max=float(vn[border].max()),unresolved_wall=bool(wall.sum()<max(8,.02*np.count_nonzero(ndi.binary_dilation(clean)^ndi.binary_erosion(clean)))))

def checkpoints(row,params,with_features=False):
 f,clean,normal,d,truth,axis=geometry(row);key=key_for(f,params);path=CACHE/(key+'.npz');rows=[];snaps={};grams={}
 if path.exists():
  z=np.load(path);snaps={N:z['n'+str(N)] for N in CHECKPOINTS};rows=json.loads(str(z['metrics']));times=json.loads(str(z['times']))
 else:
  prev=None;elapsed=0.;times={'initialize':0};t0=time.perf_counter();n=initialize(f);times['initialize']=time.perf_counter()-t0
  for t in range(1,41):
   tic=time.perf_counter();nxt=one_step(n,f,**params);elapsed+=time.perf_counter()-tic;older=prev;prev,n=n,nxt
   if t in CHECKPOINTS:
    snaps[t]=n.copy();tic=time.perf_counter();proposal=one_step(n,f,**params);res=np.linalg.norm(proposal-n,axis=-1);two=np.linalg.norm(proposal-prev,axis=-1)
    # proposal-prev is n(t+1)-n(t-1), since prev stores n(t-1).
    m=metrics(n,f,clean,normal,d);m.update(N=t,one_step_rms=float(np.sqrt(np.mean(res**2))),one_step_max=float(res.max()),two_step_rms=float(np.sqrt(np.mean(two**2))),last_update_rms=float(np.sqrt(np.mean(np.sum((n-prev)**2,-1)))),finite=bool(np.isfinite(n).all()),field_seconds=elapsed,initialization_seconds=times['initialize'],diagnostic_seconds=time.perf_counter()-tic);rows.append(m);times[str(t)]=elapsed
  np.savez_compressed(path,**{'n'+str(N):snaps[N] for N in CHECKPOINTS},f=f,metrics=json.dumps(rows),times=json.dumps(times))
 if with_features:
  fp=CACHE/(key+'_features.npz')
  if fp.exists():
   z=np.load(fp);grams={N:z['F'+str(N)] for N in CHECKPOINTS}
  else:
   grams={N:features(snaps[N]) for N in CHECKPOINTS};np.savez_compressed(fp,**{'F'+str(N):grams[N] for N in CHECKPOINTS})
 return dict(row=row,key=key,f=f,clean=clean,normal=normal,d=d,truth=truth,axis=axis,snaps=snaps,metrics=rows,times=times,features=grams)

def conventional(f,axis):
 mask=f>0;dim=f.ndim;tic=time.perf_counter();coords=np.array(np.nonzero(mask));pc=np.cov(coords);ev,vec=np.linalg.eigh(pc);pca=np.nan if axis is None else float(np.degrees(np.arccos(np.clip(abs(vec[:,-1]@axis),0,1))));pca_sec=time.perf_counter()-tic;out=[]
 def entry(method,call,ang=pca):
  t=time.perf_counter();value=call();out.append(dict(method=method,raw_measurement=float(value),angle_error=ang,measurement_seconds=time.perf_counter()-t+pca_sec))
 if dim==2:
  entry('Crofton',lambda:perimeter_crofton(mask,4));entry('marching_squares',lambda:sum(np.linalg.norm(np.diff(c,axis=0),axis=1).sum() for c in find_contours(mask.astype(float),.5)))
 else:
  entry('voxel_faces',lambda:sum(np.count_nonzero(np.diff(mask.astype('int8'),axis=k)) for k in range(dim)))
  t=time.perf_counter();v,faces,_,_=marching_cubes(mask.astype('float32'),.5);area=mesh_surface_area(v,faces);a,b,c=v[faces[:,0]],v[faces[:,1]],v[faces[:,2]];cr=np.cross(b-a,c-a);ar=np.linalg.norm(cr,axis=1)/2;nu=cr/np.maximum(2*ar[:,None],1e-12);T=np.einsum('ni,nj,n->ij',nu,nu,ar);out.append(dict(method='marching_cubes_surface_tensor',raw_measurement=float(area),angle_error=orientation(T,axis)[0],measurement_seconds=time.perf_counter()-t))
 t=time.perf_counter();s=ndi.gaussian_filter(mask.astype('float32'),1);g=np.stack(np.gradient(s),-1);T=np.array([[np.sum(g[...,i]*g[...,j],dtype=float) for j in range(dim)] for i in range(dim)]);out.append(dict(method='Gaussian1_gradient_ST',raw_measurement=float(np.linalg.norm(g,axis=-1).sum()),angle_error=orientation(T,axis)[0],measurement_seconds=time.perf_counter()-t))
 t=time.perf_counter();sm=s>.5
 if dim==2:val=sum(np.linalg.norm(np.diff(c,axis=0),axis=1).sum() for c in find_contours(sm.astype(float),.5))
 else:
  if sm.any():v,fa,_,_=marching_cubes(sm.astype('float32'),.5);val=mesh_surface_area(v,fa)
  else:val=0.
 if sm.any() and axis is not None:
  _,vv=np.linalg.eigh(np.cov(np.array(np.nonzero(sm))));ang=float(np.degrees(np.arccos(np.clip(abs(vv[:,-1]@axis),0,1))))
 else:ang=np.nan
 out.append(dict(method='Gaussian1_threshold_marching_PCA',raw_measurement=float(val),angle_error=ang,measurement_seconds=time.perf_counter()-t));out.append(dict(method='binary_PCA',raw_measurement=np.nan,angle_error=pca,measurement_seconds=pca_sec));return out
