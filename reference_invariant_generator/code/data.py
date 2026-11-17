from pathlib import Path
import numpy as np,pandas as pd,json,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output';OUT.mkdir(exist_ok=True);SEED=17112647

def manifest():
 p=OUT/'manifest_2d.csv'
 if p.exists():return pd.read_csv(p)
 units=[]
 def add(family,R,split,**params):units.append(dict(dim=2,family=family,R=R,split=split,role='reference' if split=='reference' else 'geometry',params=json.dumps(params),size=80,spacing=1.))
 for sp,offset in [('train',.347),('validation',.647),('test',.847)]:
  for r in ([3,5,7,10,14] if sp=='train' else [4,8,12,16,19] if sp=='validation' else [2,4,8,12,18]):add('circle',r+offset,sp)
  for i in range(2):
   R=5+3*i+offset
   add('ellipse',R,sp,axes=[1.87+.13*i+(.12 if sp=='test' else 0),.57+.11*i])
   add('stadium',3+2*i+offset,sp,Lratio=2.07+.51*i+(.33 if sp=='test' else 0))
   add('ring',8+4*i+offset,sp,width_ratio=.15+.12*i+(.04 if sp=='test' else 0))
   add('peanut',4+3*i+offset,sp,separation_ratio=1.13+.43*i+(.05 if sp=='test' else 0))
   add('rounded_box',7+4*i+offset,sp,a=1.57,b=.79,corner=.31+.09*i)
   add('pair',3+2*i+offset,sp,ratio=.67+.11*i,gap=.83+.37*i)
 for R in [3.947,6.947,9.947,13.947,17.947]:add('circle',R,'reference')
 add('ellipse',6.947,'reference',axes=[2.13,.71]);add('stadium',4.947,'reference',Lratio=2.37);add('rounded_box',8.947,'reference',a=1.63,b=.83,corner=.37)
 rows=[]
 for i,u in enumerate(units):
  role=u['role'];sp='train' if role=='reference' else u['split']
  for rot in range(3 if role!='reference' else 1):
   for shift in range(2 if role!='reference' else 1):
    for ni,noise in enumerate([0,.002,.005,.01] if role!='reference' else [0]):
     rows.append(dict(u,split=sp,unit=f'ri{i:03d}',case=f'ri{i:03d}_a{rot}_s{shift}_n{ni}',rep=rot*2+shift,rotation_id=rot,shift_id=shift,angle=(rot+.173+((i*7)%11)/53)*np.pi/3,seed=SEED+i*401+shift*43,noise_seed=SEED+900000+i*443+rot*67+shift*83,noise=noise,noise_kind='salt_pepper',formula='analytic continuous boundary; ellipse complete elliptic integral'))
 df=pd.DataFrame(rows);df.to_csv(p,index=False);(OUT/'manifest_2d.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest());return df

def fixture(row):
 from scipy.special import ellipe
 from scipy import ndimage as ndi
 d=int(row.get('dim',2));S=int(row['size']);a=float(row.get('spacing',1));R=float(row['R']);p=json.loads(row['params']);x=np.indices((S,)*d,dtype=float)-(S-1)/2;shift=np.random.default_rng(int(row['seed'])).uniform(-.47,.47,d)
 for j in range(d):x[j]=(x[j]-shift[j])*a
 angle=row['angle'];c,s=np.cos(angle),np.sin(angle);u=c*x[0]+s*x[1];v=-s*x[0]+c*x[1];axis=None;fam=row['family']
 if fam=='circle':mask=u*u+v*v<=R*R;truth=2*np.pi*R
 elif fam=='ellipse':
  aa,bb=np.array(p['axes'])*R;mask=(u/aa)**2+(v/bb)**2<=1;truth=4*aa*ellipe(1-(bb/aa)**2);axis=np.array([c,s])
 elif fam=='stadium':
  L=p['Lratio']*R;mask=np.maximum(abs(u)-L,0)**2+v*v<=R*R;truth=4*L+2*np.pi*R;axis=np.array([c,s])
 elif fam=='ring':
  rad=np.hypot(u,v);mask=abs(rad-R)<=p['width_ratio']*R;truth=4*np.pi*R
 elif fam=='peanut':
  sep=p['separation_ratio']*R;mask=((u-sep/2)**2+v*v<=R*R)|((u+sep/2)**2+v*v<=R*R);truth=4*R*(np.pi-np.arccos(sep/(2*R)));axis=np.array([c,s])
 elif fam=='rounded_box':
  aa,bb,rr=R*p['a'],R*p['b'],R*p['corner'];qx=abs(u)-(aa-rr);qy=abs(v)-(bb-rr);mask=np.hypot(np.maximum(qx,0),np.maximum(qy,0))+np.minimum(np.maximum(qx,qy),0)<=rr;truth=4*(aa+bb-2*rr)+2*np.pi*rr;axis=np.array([c,s])
 elif fam=='pair':
  r2=R*p['ratio'];sep=R+r2+p['gap']*R;mask=((u-sep/2)**2+v*v<=R*R)|((u+sep/2)**2+v*v<=r2*r2);truth=2*np.pi*(R+r2)
 else:raise ValueError(fam)
 mask=mask.copy();allowed=np.zeros_like(mask);allowed[(slice(7,-7),)*d]=True;mask^=(np.random.default_rng(int(row['noise_seed'])).random(mask.shape)<float(row['noise']))&allowed
 f=np.where(mask,1.,-1.);return f,float(truth),axis

def init(f,width=2.3,spacing=1.,physical=False):
 from scipy import ndimage as ndi
 signed=ndi.distance_transform_edt(f>0)-ndi.distance_transform_edt(f<0);s=np.tanh(signed* (spacing if physical else 1)/width);g=np.stack([(np.roll(s,-1,j)-np.roll(s,1,j))*.5 for j in range(f.ndim)],-1);v=np.sqrt(np.maximum(0,1-s*s))[...,None]*g/np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-7);n=np.concatenate([v,s[...,None]],-1);return n/np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-12)

def load(rows):
 samples=[fixture(r) for r in rows.to_dict('records')];return dict(rows=rows.reset_index(drop=True),f=np.stack([s[0] for s in samples]),n=np.stack([init(s[0]) for s in samples]),truth=np.array([s[1] for s in samples]),axes=[s[2] for s in samples])
