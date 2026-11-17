from data import *
from engine import generate
from readouts import measure,READOUT_METADATA
from scipy.integrate import quad
from scipy.special import ndtr
import time,resource,sys,platform

def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=True))
def angle_error(T,axis,which):
 if axis is None:return np.nan,np.nan
 T=np.asarray(T)
 if not np.isfinite(T).all():return np.nan,np.nan
 val,vec=np.linalg.eigh(T);i=0 if which=='min' else -1;gap=(val[1]-val[0] if i==0 else val[-1]-val[-2])/max(np.linalg.norm(val),1e-30);angle=float(np.degrees(np.arccos(np.clip(abs(vec[:,i]@axis),0,1))))
 return angle,float(gap)

def planar_expected(row,meta):
 d=int(row['dim']);w=float(row['width']);L=meta['roi_side'];nu=meta['normal'];off=meta['offset'];limit=L/2*np.sum(abs(nu))+abs(off);corners=np.array(list(itertools.product([-L/2,L/2],repeat=d)));points=np.unique(corners@nu+off);points=points[(points>-limit)&(points<limit)];out={}
 for method,m in READOUT_METADATA.items():
  kind=m['kind'];eps=m.get('epsilon')
  if kind=='marching':out[method]=plane_area(d,L-float(row['spacing']),nu,off);continue
  def density(u):
   s=np.tanh(u/w);sech2=max(0.,1-s*s)
   if kind=='energy':v=sech2/(w*w)
   elif kind=='scalar_energy':v=sech2*sech2/(w*w)
   elif kind=='tv':v=.5*sech2/w
   elif kind=='weighted_tv':v=.75*sech2*sech2/w
   elif kind in ['point_coarea','integrated_coarea']:
    z=ndtr(1/eps)-ndtr(-1/eps);v=np.exp(-.5*(s/eps)**2)/(np.sqrt(2*np.pi)*eps*z)*sech2/w
   else:raise ValueError(kind)
   return v*plane_area(d,L,nu,off-u)
  out[method]=quad(density,-limit,limit,points=points,epsabs=1e-7,epsrel=1e-7,limit=200)[0]
 return out

def continuum_radial():
 rows=[]
 for d in [2,3]:
  for R in [4.,8.,16.,32.]:
   for w in [.5,1.,2.]:
    A=2*np.pi*R if d==2 else 4*np.pi*R*R
    for method,m in READOUT_METADATA.items():
     kind=m['kind'];eps=m.get('epsilon')
     if kind=='marching':v=A
     else:
      def integrand(u):
       theta=np.pi/2*(1-np.sin(np.pi*u/(2*w)));tp=-np.pi**2/(4*w)*np.cos(np.pi*u/(2*w));ss=np.cos(theta);ds=abs(np.sin(theta)*tp);r=R-u;area=2*np.pi*r if d==2 else 4*np.pi*r*r
       if kind=='energy':rho=tp*tp+(d-1)*np.sin(theta)**2/(r*r)
       elif kind=='scalar_energy':rho=ds*ds
       elif kind=='tv':rho=.5*ds
       elif kind=='weighted_tv':rho=.75*(1-ss*ss)*ds
       elif kind in ['point_coarea','integrated_coarea']:
        z=ndtr(1/eps)-ndtr(-1/eps);rho=np.exp(-.5*(ss/eps)**2)/(np.sqrt(2*np.pi)*eps*z)*ds
       else:raise ValueError(kind)
       return area*rho
      v=quad(integrand,-w,w,epsabs=1e-9,epsrel=1e-10,limit=100)[0]
     rows.append(dict(dim=d,R=R,width=w,method=method,value=v,truth=A,q=v/A,profile='compact smooth theta, poles exactly outsideband; independent1D continuum quadrature'))
 pd.DataFrame(rows).to_csv(OUT/'continuum_radial.csv',index=False)

def prepare():
 assert not (OUT/'raw.csv').exists(),'Existing benchmark cannot be re-prepared in place.'
 df=make_manifest();dump('protocol.json',dict(seed=SEED,created_utc=pd.Timestamp.now(tz='UTC').isoformat(),generator=dict(gradient='exact G4 current s',exchange='fixed axialNN sum2d',J=1,D=1,h=2,N=10,normalization='H/norm,retain previous ifnorm<=1e-12',div_v=False,initialization='binaryEDT tanh width lattice'),suites=['analytic_planar','digital_planar','curvature','interaction','shape','spacing','reference'],methods=READOUT_METADATA,primary='readout-only contrast on exactly the same frozen field; no kernels/head coefficients/eps fitted; allreadouts reported',calibration='only closedcircle/sphere reference rows atwidth2.3/a1; sharedprototype-balancedglobalLS coefficient andallindividualrulers; no testbestrulerchoice',planar='centralROI excludesperiodicboundarywall; densitytruthcellbox vs marchingtruthsamplecenterbox, independentfinitewindowcontinuumquadrature',spacing='samephysicalgeometryre-rasterized, physicalwidthfixed inspacing suite; generatorgradient/a, observationgradient/a*dV; J/h fixed, not a continuumPDEconvergenceclaim',prohibition='No reconstruction/topology/wallwidth/residualloss,gates orfallback',test_policy='new diagnostic shapes andseeds, fixed methods preregistered before evaluation; not an optimizer/modelselectionstudy; no claims of tunedmodelindependenttest superiority',manifest_sha256=hashlib.sha256((OUT/'manifest.csv').read_bytes()).hexdigest(),size=len(df)))
 dump('readout_metadata.json',READOUT_METADATA);dump('environment.json',dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),threads=1,gpu_used=False,dtype='float64',peak_rss_convention='Linux KiB'))
 print('PREPARED',len(df),'cases',df.groupby(['dim','suite']).size().to_dict())

def run():
 assert (OUT/'smoke_pass.json').exists();assert not (OUT/'raw.csv').exists(),'Do not silently repeat benchmark.'
 protocol=json.load(open(OUT/'protocol.json'));assert protocol['manifest_sha256']==hashlib.sha256((OUT/'manifest.csv').read_bytes()).hexdigest();df=make_manifest();rows=[];checks=[];tic=time.perf_counter();cpu=time.process_time();last=time.perf_counter()
 for i,row in enumerate(df.to_dict('records')):
  f,n,truth,axis,roi,meta=fixture(row);p=json.loads(row['params']);a=float(row['spacing']);tt=dict(field_seconds=0.,initialization_seconds=0.);source='analytic_oracle' if n is not None else 'fixedG4_generated'
  if n is None:
   lattice_width=float(row['width'])/a if p.get('width_policy')=='physical' else float(row['width']);n,tt=generate(f,lattice_width,a,10)
  before=n.copy();expected=planar_expected(row,meta) if source=='analytic_oracle' else {};values=measure(n,a,roi,include_orientation=True);assert np.array_equal(n,before);unit_error=float(np.max(abs(np.linalg.norm(n,axis=-1)-1)))
  for r in values:
   method=r['method'];T=r.pop('energy_tensor');M=r.pop('normal_tensor');C=r.pop('spatial_covariance');r.pop('centroid',None);role=meta['axis_role'];which='max' if role in ['plane_normal','oblate_normal','symmetry_axis'] else 'min';spatial='min' if which=='max' else 'max';ae,eg=angle_error(T,axis,which);an,ng=angle_error(M,axis,which);ac,cg=angle_error(C,axis,spatial);thistruth=truth;truthdomain='continuous_closed_boundary'
   if row['family']=='plane':
    truthdomain='sample_center_box' if method=='generated_s0_marching' else 'cell_volume_box'
    if method=='generated_s0_marching':thistruth=plane_area(int(row['dim']),meta['roi_side']-a,meta['normal'],meta['offset'])
   rec=dict(row,**r,truth=thistruth,truth_domain=truthdomain,source=source,axis_role=role,orientation_defined=axis is not None,normal_target_defined=meta['normal_target_defined'],energy_target_defined=meta['normal_target_defined'],normal_axis_error=an,spatial_axis_error=ac,energy_axis_error=ae,normal_eigengap=ng,spatial_eigengap=cg,energy_eigengap=eg,continuum_expected=expected.get(method,np.nan),unit_error=unit_error,N=0 if source=='analytic_oracle' else 10,**tt,thickness=p.get('thickness',np.nan),gap=p.get('gap',np.nan),actual_grid_size=n.shape[0],actual_domain_span=n.shape[0]*a,roi_side=meta.get('roi_side',np.nan),profile_width_kind='physical_oracle_tanh' if source=='analytic_oracle' else 'EDT_physical_init' if p.get('width_policy')=='physical' else 'EDT_lattice_init')
   rows.append(rec)
  if i%20==0 or i==len(df)-1:
   pd.DataFrame(rows).to_csv(OUT/'raw.partial.csv',index=False);print('CASE',i+1,'/',len(df),row['dim'],row['suite'],'wall',round(time.perf_counter()-tic,1),flush=True)
 pd.DataFrame(rows).to_csv(OUT/'raw.csv',index=False);(OUT/'raw.partial.csv').unlink(missing_ok=True);continuum_radial();dump('run_cost.json',dict(wall_seconds=time.perf_counter()-tic,CPU_seconds=time.process_time()-cpu,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cases=len(df),rows=len(rows),manifest_sha256=protocol['manifest_sha256']));print('RUN COMPLETE',flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);args=p.parse_args();prepare() if args.stage=='prepare' else run()
