from data import *
from degree_legacy import degree
from topology_certificate import certificate
from skimage.measure import euler_number
TARGETS=[(.37139,-.52177,.64829,.40981),(.13,.71,-.42,.54),(-.55,.31,.42,-.65)]
def cap(n,collar=3):
 n=n.astype(float).copy();index=np.indices(n.shape[:3]);edge=np.minimum.reduce([index[j] for j in range(3)]+[n.shape[j]-1-index[j] for j in range(3)]);active=edge<collar;negative=float(n[active,-1].max());t=np.clip((collar-edge)/collar,0,1);pole=np.zeros(4);pole[-1]=-1;n=(1-t[...,None])*n+t[...,None]*pole;n/=np.linalg.norm(n,axis=-1,keepdims=True);return n,negative

def audit_degree(n,label,f=None):
 c,neg=cap(n);cert=certificate(c);values=[degree(c,target=t) for t in TARGETS];agree=len(set(x['integer_degree'] for x in values))==1;valid=neg<0 and cert['interpolation_norm_lower_bound']>1e-7 and agree
 out=dict(case=label,degree=values[0]['integer_degree'] if valid else np.nan,status='NUMERICALLY_SUPPORTED' if valid else 'INCONCLUSIVE',target_counts=json.dumps([x['integer_degree'] for x in values]),raw_negative_collar_max=neg,**cert,boundary_spread_max=max(x['boundary_spread'] for x in values),minimum_target_margin=min(x['min_barycentric_positive_coefficient'] for x in values))
 if f is not None:out.update(input_euler=int(euler_number(f>0,connectivity=1)),input_components=int(ndi.label(f>0)[1]),scalar_sign_euler=int(euler_number(n[...,-1]>0,connectivity=1)))
 return out

def synthetic_degree_field(size=33,radius=9.31,mirror=False):
 x=np.indices((size,)*3,dtype=float)-(size-1)/2;r=np.linalg.norm(x,axis=0);t=np.clip(r/radius,0,1);F=np.pi*(3*t*t-2*t*t*t);v=-np.moveaxis(x,0,-1)/np.maximum(r[...,None],1e-30)*np.sin(F)[...,None];s=np.cos(F);n=np.concatenate([v,s[...,None]],-1);n[r>=radius]=[0,0,0,-1]
 if mirror:n[...,0]*=-1
 return n

def run():
 rows=[]
 for mirror in [False,True]:rows.append(audit_degree(synthetic_degree_field(mirror=mirror),'synthetic_hedgehog_mirror' if mirror else 'synthetic_hedgehog'))
 constant=np.zeros((17,17,17,4));constant[...,-1]=-1;rows.append(audit_degree(constant,'synthetic_constant'))
 for fam,R,p in [('sphere',6.431,{}),('capsule',4.431,dict(Lratio=1.8)),('torus',10.431,dict(minor=2.731)),('pair',3.931,dict(gap=7.3))]:
  row=diagnostic_row(3,fam,R,p,size=40);f,truth,axis,aux=fixture(row);n=initialize(f)
  for _ in range(30):n=old_step(n,f)
  rows.append(audit_degree(n,f'generated_{fam}_N30',f));print('TOPOLOGY',fam,rows[-1],flush=True)
 pd.DataFrame(rows).to_csv(OUT/'spin_topology.csv',index=False);dump('topology_convention.json',dict(domain='finite cube with negative_s exterior3voxel collar compactified to southpole; periodic evolution and degreeboundary aredistinct',interpolation='normalized barycentric Freudenthal tetrahedra',sign='inward radial hedgehog +1 with thiscoordinate/targetorientation',certificate='openhemisphere or convex-hull numerical active-face lowerbound, notinterval arithmetic',no_geometry_selection=True,distinct_from=['inputEuler','inputcomponents','RPloopclass']))
 print('TOPOLOGY COMPLETE',flush=True)
if __name__=='__main__':run()
