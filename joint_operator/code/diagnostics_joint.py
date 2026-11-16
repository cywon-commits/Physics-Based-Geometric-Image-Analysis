from evaluate_joint import *
from infer_joint import predict
import resource

def run():
 frozen=json.load(open(OUT/'pretest_selection.json'));lock=hashlib.sha256((OUT/'pretest_selection.json').read_bytes()).hexdigest();assert lock==(OUT/'pretest_selection.sha256').read_text();df=make_manifest();models=json.load(open(OUT/'trained_models.json'));timing=[];transfer=[];profiles=[];condition=[]
 for dim in [2,3]:
  sel=frozen['selection'][str(dim)];sub=df[(df.dim==dim)&(df.split=='test')&(df.rep==0)&(df.noise==0)].groupby('family').head(1)
  for row in sub.to_dict('records'):
   f,_,_,_,_,_=fixture(row)
   for experimental in [True,False]:
    # False means the actually permitted production path (may be PBGIA-30 fallback).
    predict(f>0,frozen,experimental)
    for rep in range(3):
     _,r=predict(f>0,frozen,experimental);timing.append(dict(dim=dim,case=row['case'],family=row['family'],repeat=rep,experimental=experimental,**{k:v for k,v in r.items() if k.endswith('seconds') or k in ['mode','N']}))
   # Explicit pinned baseline, regardless of production choice.
   basef=json.loads(json.dumps(frozen));basef['selection'][str(dim)]['recommended']=False
   for rep in range(3):
    _,r=predict(f>0,basef,False);timing.append(dict(dim=dim,case=row['case'],family=row['family'],repeat=rep,experimental=False,mode='explicit_PBGIA30',N=30,**{k:v for k,v in r.items() if k.endswith('seconds')}))
  cfg=sel['generator'];o=sel['observer'];N=sel['N'];refrow=df[(df.dim==dim)&(df.role=='calibration')].iloc[0].to_dict()
  for h,D in [(4.,1.),(2.,.5)]:
   c=dict(cfg,h=h,D=D);ref=rollout(refrow,c);coef=ref['truth']/np.trace(tensor(ref['states'][N],o['p']))
   for row in sub.to_dict('records'):
    case=rollout(row,c);r,_=observe(case['states'][N],c,o['p'],case['axis'],o['observer'],o['mix'],o['which']);value=coef*r['raw_measurement'];transfer.append(dict(row,N=N,h=h,D=D,model='frozen_joint_transfer',truth=case['truth'],measured=value,error_pct=100*(value/case['truth']-1),**r,**{k:v for k,v in case['metrics'][N].items() if k!='N'}))
  for family in (['stadium','ellipse'] if dim==2 else ['capsule','sphere','torus']):
   row=sub[sub.family==family].iloc[0].to_dict();case=rollout(row,cfg)
   for it,n in case['states'].items():
    H=cfg['J']*exchange(n,cfg);H[...,:dim]+=cfg['D']*generated_gradient(n[...,-1],cfg['gp']);H[...,-1]+=cfg['h']*case['f'];hn=np.linalg.norm(H,axis=-1);condition.append(dict(dim=dim,case=row['case'],family=family,N=it,min_H_norm=float(hn.min()),near_zero_H_count=int((hn<=1e-12).sum()),**{k:v for k,v in case['metrics'][it].items() if k!='N'}))
    for lo in np.linspace(-4,3.8,40):
     w=(case['dist']>=lo)&(case['dist']<lo+.2)
     if w.any():profiles.append(dict(dim=dim,family=family,N=it,distance=lo+.1,mean_s=float(n[...,-1][w].mean()),count=int(w.sum())))
 pd.DataFrame(timing).to_csv(OUT/'timing.csv',index=False);pd.DataFrame(transfer).to_csv(OUT/'transfer_results.csv',index=False);pd.DataFrame(profiles).to_csv(OUT/'wall_profiles.csv',index=False);pd.DataFrame(condition).to_csv(OUT/'field_conditioning.csv',index=False)
 # Held-out prototype bootstrap: preserve rotations/shifts/noise together.
 test=pd.read_csv(OUT/'test_results.csv');ref=test[test.model=='PBGIA30_frozen'].set_index('case');rows=[];paired=[]
 for _,r in test[test.model=='joint_selected'].iterrows():
  b=ref.loc[r['case']];paired.append(dict(case=r['case'],unit=r.unit,dim=r.dim,family=r.family,heldout_family=r.heldout_family,noise=r.noise,geometry_delta_pp=abs(r.error_pct)-abs(b.error_pct),orientation_delta_deg=r.angle_error-b.angle_error,dice_delta=r.clean_dice-b.clean_dice,error_pct=r.error_pct,angle_error=r.angle_error))
 pairs=pd.DataFrame(paired);pairs.to_csv(OUT/'paired_comparison.csv',index=False)
 for keys,g in pairs.groupby(['dim','noise','heldout_family']):
  unit=g.groupby('unit')[['geometry_delta_pp','orientation_delta_deg','dice_delta']].mean()
  for metric in unit:
   v=unit[metric].dropna().to_numpy()
   if len(v):
    rng=np.random.default_rng(261116);boots=rng.choice(v,(3000,len(v)),replace=True).mean(1);rows.append(dict(zip(['dim','noise','heldout_family'],keys),metric=metric,mean=v.mean(),CI95_low=np.quantile(boots,.025),CI95_high=np.quantile(boots,.975),independent_units=len(v)))
 pd.DataFrame(rows).to_csv(OUT/'paired_bootstrap_ci.csv',index=False)
 # Family hold-outs, no averaging them silently into seen-family transfer.
 ss=[]
 for keys,g in test.groupby(['dim','model','family','heldout_family','noise']):
  u=g.groupby('unit').agg(error=('error_pct',lambda a:np.mean(abs(a))),angle=('angle_error','mean'));ss.append(dict(zip(['dim','model','family','heldout_family','noise'],keys),n_units=len(u),MARE=u.error.mean(),angle_MAE=u.angle.mean(),min_Dice=g.clean_dice.min(),failure_count=int(((g.clean_dice<.995)|(g.chi!=g.clean_chi)|(g.components!=g.clean_components)).sum())))
 pd.DataFrame(ss).to_csv(OUT/'test_family_summary.csv',index=False)
 # Exchange kernels, raw and controlled scale.
 kr=[]
 for dim in [2,3]:
  for family,cfg in models[str(dim)].items():
   c=cfg['generator'];K=exchange_kernel(dim,c);G=derivative_filters(dim,*c['gp']);raw=(G*G).sum(0);idx=np.indices(K.shape)-2;kr.append(dict(dim=dim,model=family,exchange=c['family'],KJ_sum=K.sum(),raw_tied_sum=raw.sum(),normalization_factor=2*dim/raw.sum(),min_coefficient=K.min(),center=K[(2,)*dim],second_moment=float((K*np.sum(idx*idx,axis=0)).sum()/K.sum()),nonzero_count=int((K>0).sum()),gp=json.dumps(c['gp']),ep=json.dumps(c['ep'])))
   np.save(OUT/f'KJ_{dim}d_{family}.npy',K)
 pd.DataFrame(kr).to_csv(OUT/'kernel_J_tied_vs_free.csv',index=False)
 fr=[]
 for dim in [2,3]:
  c=frozen['selection'][str(dim)]
  for kind,p in [('generation',c['generator']['gp']),('measurement',c['observer']['p'])]:
   b,t,u=p
   for mag in [.2,.5,1.,1.5,2.,2.8]:
    for theta in np.linspace(0,np.pi/2,31):
     direction=np.array([np.cos(theta),np.sin(theta)]+([.37] if dim==3 else []));direction/=np.linalg.norm(direction);w=mag*direction;r=[]
     for k in range(dim):
      a=2*(.5-2*b)*np.sin(w[k])+2*b*np.sin(2*w[k])
      for j in range(dim):
       if j!=k:a*=1-2*t-2*u+2*t*np.cos(w[j])+2*u*np.cos(2*w[j])
      r.append(a)
     r=np.array(r);fr.append(dict(dim=dim,kind=kind,frequency=mag,theta=theta,gain=np.linalg.norm(r)/mag,angle_error=np.degrees(np.arccos(np.clip(r@direction/max(np.linalg.norm(r),1e-14),-1,1)))))
 pd.DataFrame(fr).to_csv(OUT/'kernel_frequency_response.csv',index=False)
 # Separate domain margin check, no claim of grid refinement.
 margin=[]
 dim=3;c=frozen['selection']['3'];row=df[(df.dim==3)&(df.family=='torus')&(df.split=='test')&(df.noise==0)&(df.rep==0)].iloc[0].to_dict()
 for size in [48,64,80]:
  row=dict(row,size=size);case=rollout(row,c['generator']);r,_=observe(case['states'][c['N']],c['generator'],c['observer']['p'],case['axis'],c['observer']['observer'],c['observer']['mix'],c['observer']['which']);margin.append(dict(size=size,N=c['N'],raw_measurement=r['raw_measurement'],geometry=c['calibration']*r['raw_measurement'],**{k:v for k,v in case['metrics'][c['N']].items() if k!='N'}))
 pd.DataFrame(margin).to_csv(OUT/'margin_check.csv',index=False)
 assert lock==hashlib.sha256((OUT/'pretest_selection.json').read_bytes()).hexdigest();print('DIAGNOSTICS DONE',flush=True)

if __name__=='__main__':run()
