from common import *
import geometry_previous as geo

def manifest():
 path=OUT/'manifest.csv'
 if path.exists():return pd.read_csv(path)
 rows=[];unit=0
 for d in [2,3]:
  for split,offset in [('train',0),('validation',1.3),('test',2.6)]:
   specs=[('circle' if d==2 else 'sphere',6.583+offset if d==2 else 7.583+offset,{}),('ellipse' if d==2 else 'ellipsoid',5.583+offset if d==2 else 6.583+offset,dict(axes=[1.87+.12*(offset>1),.63] if d==2 else [1.87,.81,.57])),('stadium' if d==2 else 'capsule',3.583+offset if d==2 else 4.583+offset,dict(Lratio=2.1)),('ring' if d==2 else 'torus',9.583+offset if d==2 else 10.583+offset,dict(width_ratio=.18) if d==2 else dict(minor=2.583+offset*.23)),('near_touch',3.983+offset,dict(gap=.9+offset*.1)),('pair',3.183+offset,dict(gap=9.3+offset*.1))]
   for family,R,params in specs:
    uid=f'fi{unit:03d}';unit+=1
    for rot in range(2):
     for sh in range(2):
      for k,noise in enumerate([0.,.002,.01]):rows.append(dict(case=f'{uid}_r{rot}_s{sh}_n{k}',unit=uid,dim=d,split=split,role='geometry',family=family,R=R,params=json.dumps(params),size=80 if d==2 else 64,rep=rot*2+sh,rotation_id=rot,shift_id=sh,noise=noise,spacing=1.,seed=SEED+unit*419+rot*61+sh*79,noise_seed=SEED+500000+unit*431+rot*73+sh*83,noise_kind='salt_pepper'))
  for R in ([5.783,9.783,13.783] if d==2 else [4.783,7.783,11.783]):
   uid=f'ref{d}_{R}'
   for sh in range(2):rows.append(dict(case=f'{uid}_s{sh}',unit=uid,dim=d,split='train',role='reference',family='circle' if d==2 else 'sphere',R=R,params='{}',size=80 if d==2 else 64,rep=sh,rotation_id=0,shift_id=sh,noise=0.,spacing=1.,seed=SEED+int(R*1000)+sh*79,noise_seed=0,noise_kind='salt_pepper'))
 df=pd.DataFrame(rows);df.to_csv(path,index=False);(OUT/'manifest.sha256').write_text(sha(path));return df

def fixture(row):
 rr=dict(row);family=rr['family'];rr['dim']=int(rr['dim']);rr['size']=int(rr['size']);rr['seed']=int(rr['seed']);rr['noise_seed']=int(rr['noise_seed']);p=json.loads(rr['params'])
 if family=='pair':rr['family']='near_touch'
 if family=='shell':
  rr['family']='nested';rr['params']=json.dumps(dict(radii=[rr['R']+p['thickness']/2,rr['R']-p['thickness']/2,0]))
 f,truth,axis,aux=geo.fixture(rr);Q=aux['rotation'];d=f.ndim;size=f.shape[0];x=np.indices(f.shape,dtype=float)-(size-1)/2
 for j in range(d):x[j]-=aux['shift'][j]
 x*=float(rr['spacing']);y=np.einsum('ji,j...->i...',Q,x);u=y[0];rho=np.sqrt(np.sum(y[1:]**2,axis=0));rad=np.linalg.norm(y,axis=0);R=rr['R'];role='undefined';normal=None;distance=None
 if family in ['circle','sphere']:distance=R-rad;normal=-y/np.maximum(rad,1e-30)
 elif family in ['stadium','capsule']:
  normal=y.copy();normal[0]=np.sign(u)*np.maximum(abs(u)-p['Lratio']*R,0);distance=R-np.linalg.norm(normal,axis=0);normal=-normal/np.maximum(np.linalg.norm(normal,axis=0),1e-30);role='object_major'
 elif family in ['ellipse','ellipsoid']:
  axes=np.array(p['axes'])*R;g=y/axes.reshape((d,)+(1,)*d)**2;normal=-g/np.maximum(np.linalg.norm(g,axis=0),1e-30);distance=(1-np.sum((y/axes.reshape((d,)+(1,)*d))**2,axis=0))/(2*np.maximum(np.linalg.norm(g,axis=0),1e-12));role='object_major'
 elif family=='torus':
  normal=y.copy();normal[1:]*=1-R/np.maximum(rho,1e-30);distance=p['minor']-np.sqrt((rho-R)**2+u*u);normal=-normal/np.maximum(np.linalg.norm(normal,axis=0),1e-30);axis=Q[:,0];role='symmetry_axis'
 elif family in ['ring','shell']:
  width=p['width_ratio']*R if family=='ring' else p['thickness']/2;distance=width-abs(rad-R);normal=-np.sign(rad-R)[None]*y/np.maximum(rad,1e-30)
 elif family in ['near_touch','pair']:
  sep=2*R+p['gap'];a=y.copy();b=y.copy();a[0]-=sep/2;b[0]+=sep/2;ra=np.linalg.norm(a,axis=0);rb=np.linalg.norm(b,axis=0);normal=np.where((ra<rb)[None],-a/np.maximum(ra,1e-30),-b/np.maximum(rb,1e-30));distance=R-np.minimum(ra,rb);axis=Q[:,0];role='pair_separation'
 if normal is not None:normal=np.einsum('ij,j...->...i',Q,normal)
 return f,truth,axis,dict(**aux,inward_normal=normal,distance_for_band=distance,axis_role=role)

def diagnostic_row(dim,family,R,params=None,size=None,seed=0):return dict(case=f'diag_{dim}_{family}_{R}_{seed}',unit=f'diag_{dim}_{family}_{R}',dim=dim,family=family,R=R,params=json.dumps(params or {}),size=size or(64 if dim==2 else 40),rep=0,noise=0.,spacing=1.,seed=SEED+900000+seed,noise_seed=SEED+800000+seed,noise_kind='salt_pepper')
