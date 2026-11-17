from evaluation import *

def run():
 assert (OUT/'test_audit_2d.json').exists();f=check_lock();models=json.load(open(OUT/'models.json'));kernelrows=[];freq=[];transfers=[];times=[]
 for name,m in models.items():
  K=kernels(m['p']);coord=np.indices((5,5))-2;m1=float((K[1]*coord[1]).sum());hypothetical=float(np.sum(K*K))
  for pos in np.ndindex(5,5):kernelrows.append(dict(model=name,row=pos[0]-2,col=pos[1]-2,Ky=K[(0,)+pos],Kx=K[(1,)+pos],first_moment=m1,actual_NN_sum=4,hypothetical_raw_tied_sum=hypothetical))
  for k in [.05,.2,.5,1.,2.,np.pi]:
   vals=[]
   for angle in np.linspace(0,2*np.pi,128,endpoint=False):
    wave=k*np.array([np.cos(angle),np.sin(angle)]);symbol=np.sum(K*np.exp(1j*np.sum(coord*wave[:,None,None],axis=0))[None],axis=(1,2));vals.append(np.linalg.norm(symbol))
   freq.append(dict(model=name,k=k,m1=m1,mean_gain=np.mean(vals),angular_CV=np.std(vals)/max(np.mean(vals),1e-30),max_gain=np.max(vals)))
 for preset in ['G4','selected_w1']:
  c=f['configurations'][preset];p=c['p'];N=c['N'];row=manifest().query("split=='train' and role=='reference' and family=='circle'").iloc[2].to_dict();fr,tr,_=fixture(row);reference=forward(fr[None],init(fr)[None],p,N)['I'][0];c0=tr/reference
  target=manifest().query("split=='test' and family=='ellipse' and noise==0").iloc[0].to_dict()
  for spacing in [.7,1.,1.4]:
   for protocol in ['lattice_width_fixed','physical_width_fixed']:
    physical=protocol=='physical_width_fixed';r=dict(target,spacing=spacing,size=int(np.ceil(80/spacing)));ff,truth,axis=fixture(r);n=init(ff,spacing=spacing,physical=physical);a=forward(ff[None],n[None],p,N,spacing=spacing)
    rr=dict(row,spacing=spacing,size=int(np.ceil(80/spacing)));rf,rt,_=fixture(rr);b=forward(rf[None],init(rf,spacing=spacing,physical=physical)[None],p,N,spacing=spacing);ca=rt/b['I'][0]
    for mode,coef in [('frozen_c',c0),('train_c_of_a',ca)]:transfers.append(dict(model=preset,spacing=spacing,protocol=protocol,calibration=mode,c=coef,I=a['I'][0],truth=truth,error_pct=100*(coef*a['I'][0]/truth-1)))
  for S in [80,112]:
   r=dict(target,size=S);ff,t,_=fixture(r);a=forward(ff[None],init(ff)[None],p,N);transfers.append(dict(model=preset,spacing=1.,protocol='padding_NOT_refinement',calibration='frozen_c',c=c0,I=a['I'][0],truth=t,error_pct=100*(c0*a['I'][0]/t-1),size=S))
  for R in [1.947,2.947,3.947]:
   r=dict(target,family='circle',params='{}',R=R,size=64);ff,t,_=fixture(r);a=forward(ff[None],init(ff)[None],p,N);transfers.append(dict(model=preset,spacing=1.,protocol='small_circle_postfreeze_probe',calibration='frozen_c',c=c0,I=a['I'][0],truth=t,error_pct=100*(c0*a['I'][0]/t-1),R=R))
  # In-process first-call and warm timings, target mask generation excluded.
  for fam in ['circle','ellipse','stadium','ring','peanut']:
   r=manifest().query('split=="test" and family==@fam and noise==0').iloc[0].to_dict();ff,t,_=fixture(r)
   for rep in range(4):
    tic=time.perf_counter();n=init(ff);initsec=time.perf_counter()-tic;a=forward(ff[None],n[None],p,N);times.append(dict(model=preset,family=fam,N=N,repeat=rep,kind='first_call_in_process' if rep==0 else 'warm',init_seconds=initsec,field_seconds=a['field_seconds'],observer_seconds=a['observer_seconds'],wall_seconds=time.perf_counter()-tic))
 pd.DataFrame(kernelrows).to_csv(OUT/'kernel_coefficients.csv',index=False);pd.DataFrame(freq).to_csv(OUT/'fourier_symbols.csv',index=False);pd.DataFrame(transfers).to_csv(OUT/'spacing_padding_small.csv',index=False);pd.DataFrame(times).to_csv(OUT/'timings.csv',index=False)
 sweep=pd.read_csv(OUT/'reference_ruler_sweep.csv');agg=sweep.groupby(['model','noise']).agg(c_CV=('c',lambda x:x.std(ddof=0)/x.mean()),c_min=('c','min'),c_max=('c','max'),MARE_median_over_refs=('MARE','median'),MARE_worst_ref=('MARE','max'),MARE_min_diagnostic_only=('MARE','min')).reset_index();agg.to_csv(OUT/'ruler_sensitivity_summary.csv',index=False);identities={name:json.dumps(dict(p=c['p'],N=c['N']),sort_keys=True) for name,c in f['configurations'].items()};sweep['rank_identity']=sweep.model.map(identities);unique=sweep.drop_duplicates(['noise','reference_unit','rank_identity']).copy();unique['MARE_rank']=unique.groupby(['noise','reference_unit']).MARE.rank(method='min');sweep=sweep.merge(unique[['noise','reference_unit','rank_identity','MARE_rank']],on=['noise','reference_unit','rank_identity'],how='left');sweep.to_csv(OUT/'reference_ruler_sweep.csv',index=False)
 # Prototype bootstrap: raw q-spread differences vs matched G4, no reference ruler involved.
 raw=pd.read_csv(OUT/'test_raw_2d.csv');bootrows=[];rng=np.random.default_rng(SEED)
 for noise in [0,.002,.005,.01]:
  base=raw[(raw.model=='G4')&(raw.noise==noise)].groupby('unit').q.mean()
  for preset in ['selected_w0','selected_w1','selected_w4']:
   q=raw[(raw.model==preset)&(raw.noise==noise)].groupby('unit').q.mean().reindex(base.index);idx=rng.integers(0,len(q),(3000,len(q)));x=q.to_numpy()[idx];b=base.to_numpy()[idx];delta=100*(np.std(x,axis=1)/np.mean(x,axis=1)-np.std(b,axis=1)/np.mean(b,axis=1));bootrows.append(dict(model=preset,noise=noise,CV_difference_pp=100*(q.std(ddof=0)/q.mean()-base.std(ddof=0)/base.mean()),CI_low=np.percentile(delta,2.5),CI_high=np.percentile(delta,97.5),units=len(q)))
 pd.DataFrame(bootrows).to_csv(OUT/'paired_bootstrap.csv',index=False);print('DIAGNOSTICS COMPLETE',flush=True)
if __name__=='__main__':run()
