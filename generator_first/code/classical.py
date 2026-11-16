import numpy as np,time
import scipy.ndimage as ndi
from skimage.measure import find_contours,perimeter_crofton,marching_cubes,mesh_surface_area

def orientation(T,axis):
 if axis is None:return np.nan,np.nan
 eig,vec=np.linalg.eigh(T);gap=(eig[1]-eig[0])/max(abs(eig[-1]),1e-12)
 err=float(np.degrees(np.arccos(np.clip(abs(vec[:,0]@axis),0,1))))
 return err,float(gap)

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
