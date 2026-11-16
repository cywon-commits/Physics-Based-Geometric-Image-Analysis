"""Post-freeze measurements; no feedback into training/selection."""
from evaluate import *

def infer(f,c,N,head,coef,spacing=1.):
 t=time.perf_counter();n=initialize(f) if c.get('init','edt')=='edt' else np.concatenate([np.zeros(f.shape+(f.ndim,),dtype='float32'),f[...,None]],-1);initial=time.perf_counter()-t;start=time.perf_counter()
 for _ in range(N):n=update(n,f,dict(c,spacing=spacing))
 field=time.perf_counter()-start;r,axis=readout(n,c,head,coef,None,spacing);return r,axis,dict(initialization_seconds=initial,field_seconds=field,observation_seconds=r['measurement_seconds'],wall_seconds=time.perf_counter()-t)

def spectral():
 models=json.load(open(OUT/'models.json'));rows=[];kernels=[]
 for d,mm in models.items():
  d=int(d)
  for name,m in mm.items():
   c=m['generator'];b,t,u=c['gp'];k=exchange_kernel(d,c);G=derivative_filters(d,b,t,u)
   for radius in [.05,.1,.25,.5,1.,2.]:
    rays=np.random.default_rng(SEED).normal(size=(256,d));rays/=np.linalg.norm(rays,axis=1,keepdims=True);wave=rays*radius;der=(1-4*b)*np.sin(wave)+2*b*np.sin(2*wave);sm=1-2*t-2*u+2*t*np.cos(wave)+2*u*np.cos(2*wave);symbol=np.stack([der[:,i]*np.prod(sm[:,[j for j in range(d) if j!=i]],axis=1) for i in range(d)],1);ratio=np.linalg.norm(symbol,axis=1)/radius;align=np.linalg.norm(symbol-wave,axis=1)/radius
    rows.append(dict(dim=d,model=name,k=radius,magnitude_bias=float(ratio.mean()-1),angular_CV=float(ratio.std()/ratio.mean()),vector_relative_error=float(align.mean()),longitudinal_cubic=float(1/6+2*b),transverse_quadratic=float(t+4*u),exchange_sum=float(k.sum()),exchange_min=float(k.min()),self_weight=float(k[(2,)*d])))
   for coord in np.ndindex(*k.shape):
    kernels.append(dict(dim=d,model=name,coordinates=json.dumps(tuple(a-2 for a in coord)),exchange=float(k[coord]),**{f'G{i}':float(G[(i,)+coord]) for i in range(d)}))
 pd.DataFrame(rows).to_csv(OUT/'fourier_symbols.csv',index=False);pd.DataFrame(kernels).to_csv(OUT/'kernel_coefficients.csv',index=False)

def run():
 assert (OUT/'test_audit.json').exists();frozen=json.load(open(OUT/'pretest_selection.json'));times=[];stress=[];refin=[];ab=[]
 models=json.load(open(OUT/'models.json'))
 for d in [2,3]:
  configs=frozen['configurations'][str(d)];unique={}
  for name in ['B','D','E','F','A','default','geometry_priority','orientation_priority']:
   if name in configs:
    c=configs[name];key=json.dumps(c,sort_keys=True);unique.setdefault(key,(name,c))
  representatives=rows_for(d,'test').query("noise==0 and rep==0 and role!='reference'").groupby('family').head(1)
  for name,c in unique.values():
   for row in representatives.to_dict('records'):
    f,truth,axis,_=fixture(row)
    for rep in range(4):
     r,ax,tt=infer(f,c['generator'],c['N'],c['heads'],c['coefficient']);times.append(dict(dim=d,model=name,N=c['N'],family=row['family'],size=row['size'],repeat=rep,kind='first_call' if rep==0 else 'warm',**tt))
   # Explicit small objects and grid refinement, post-freeze descriptive probes.
   for R in ([1.73,2.53,3.73] if d==2 else [1.63,2.43,3.63]):
    row=dict(rows_for(d,'test').iloc[0]);row.update(family='small',R=R,params='{}',noise=0.,size=32 if d==3 else 48,seed=SEED+int(R*100),case=f'small{d}_{R}',unit=f'small{d}_{R}');f,t,a,_=fixture(row);r,ax,_=infer(f,c['generator'],c['N'],c['heads'],c['coefficient']);stress.append(dict(dim=d,model=name,N=c['N'],R=R,prediction=r['prediction'],truth=t,error_pct=100*(r['prediction']/t-1),note='known_small_scale_regime_not_new_independent_test'))
  for name in ['B','D','F']:
   c=configs[name]
   for spacing in [1.5,1.,.75]:
    row=dict(rows_for(d,'test').iloc[0]);row.update(family='ellipse' if d==2 else 'sphere',R=7.83 if d==2 else 5.43,params=json.dumps(dict(axes=[2.03,.67])) if d==2 else '{}',spacing=spacing,size=int(np.ceil((56 if d==2 else 40)/spacing)),noise=0.,seed=SEED+773,case=f'refine{d}',unit=f'refine{d}')
    f,t,a,_=fixture(row);r,ax,tt=infer(f,c['generator'],c['N'],c['heads'],c['coefficient'],spacing)
    # Same train reference, rerendered at a, never use probe truth for calibration.
    rr=rows_for(d,'train').query("role=='reference'").iloc[0].to_dict();rr.update(spacing=spacing,size=int(np.ceil(48/spacing)) if d==3 else int(np.ceil(64/spacing)));fr,tr,_,_=fixture(rr);rref,_,_=infer(fr,c['generator'],c['N'],c['heads'],1.,spacing);newcoef=tr/rref['prediction']
    for mode,co in [('frozen_c',c['coefficient']),('train_rerendered_c',newcoef)]:
     val=r['prediction']*co/c['coefficient'];refin.append(dict(dim=d,model=name,method='PBGIA',mode=mode,spacing=spacing,size=row['size'],prediction=val,truth=t,error_pct=100*(val/t-1)))
    for cr in conventional(f,a):
     if np.isfinite(cr['raw_measurement']):refin.append(dict(dim=d,model=name,method=cr['method'],mode='raw',spacing=spacing,size=row['size'],prediction=cr['raw_measurement']*spacing**(d-1),truth=t,error_pct=100*(cr['raw_measurement']*spacing**(d-1)/t-1)))
  # Domain padding at fixed spacing is a boundary probe, not refinement.
  for name in ['B','D','F']:
   c=configs[name];base_row=rows_for(d,'test').query("family=='torus'" if d==3 else "family=='stadium'").iloc[0].to_dict();base_row.update(noise=0.,rep=0)
   for padding in [0,16]:
    row=dict(base_row,size=int(base_row['size'])+padding);f,t,axis,_=fixture(row);r,ax,tt=infer(f,c['generator'],c['N'],c['heads'],c['coefficient']);refin.append(dict(dim=d,model=name,method='padding_probe',mode='fixed_spacing',spacing=1.,size=row['size'],prediction=r['prediction'],truth=t,error_pct=100*(r['prediction']/t-1)))
  # Shared observer simplification at all validation units; no selection change.
  for name in ['B','D','E','F']:
   m=models[str(d)][name];cc=fields(rows_for(d,'validation').query("rep==0"),m['generator']);h=json.loads(json.dumps(m['heads']))
   for v in h.values():v['orientation']['p']=v['geometry']['p']
   ab.append(measure(cc,h,name,'shared_Kgeom_Korient'))
 pd.DataFrame(times).to_csv(OUT/'timings.csv',index=False);pd.DataFrame(stress).to_csv(OUT/'small_object_stress.csv',index=False);pd.DataFrame(refin).to_csv(OUT/'resolution_results.csv',index=False);pd.concat(ab).to_csv(OUT/'shared_observer_validation.csv',index=False);spectral()
 # Cost sensitivity uses common wall-time prefixes of actual search ledger, all three families.
 tr=pd.read_csv(OUT/'generator_trials.csv');rows=[]
 for d,g in tr[(tr.scope=='shared')&~tr.fixed_gp].groupby('dim'):
  T=g.groupby('family').seconds.sum().min()
  for frac in [.25,.5,1.]:
   for family,h in g.groupby('family'):
    eligible=h[h.seconds.cumsum()<=T*frac]
    if not len(eligible):continue
    rows.append(dict(dim=d,family=family,common_wall_budget=T*frac,evaluations=len(eligible),best_train_loss=float(eligible.loss.min()),best_geometry_only=float(eligible.loss_geometry_only.min()),best_angle_weight_02=float(eligible.loss_angle_weight_02.min())))
 pd.DataFrame(rows).to_csv(OUT/'budget_sensitivity.csv',index=False);print('DIAGNOSTICS COMPLETE',flush=True)
if __name__=='__main__':run()
