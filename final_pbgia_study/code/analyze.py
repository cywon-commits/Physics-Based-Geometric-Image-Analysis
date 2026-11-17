"""Derive public tables/figures from frozen results; never change a model."""
from common import *
from projective import eigendirector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)

def savefig(fig,name):
 fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=170);plt.close(fig)

def geometry_tables():
 frozen=json.load(open(OUT/'model_freeze.json'));cal=json.load(open(OUT/'calibration.json'));rows=[];long=[];orientation=[];bootstrap=[]
 for file in ['train_validation_raw.csv','test_raw.csv']:
  raw=pd.read_csv(OUT/file)
  for d,g in raw[raw.role=='geometry'].groupby('dim'):
   z=frozen['selection'][str(d)];predictions={'energy_original_c':g.energy_frozen,'energy_train_c':z['c_E']*g.I_E,'TV_raw':g.I_TV,'TV_train_c':z['c_TV']*g.I_TV,'selected':z['alpha']*z['c_E']*g.I_E+(1-z['alpha'])*z['c_TV']*g.I_TV,'generated_s0_raw':g.I_s0}
   for method,pred in predictions.items():
    t=g[['case','unit','dim','split','family','noise','truth']].copy();t['method']=method;t['estimate']=pred;t['error_percent']=100*(pred/t.truth-1);long.append(t)
  conventional=pd.read_csv(OUT/('test_conventional.csv' if file.startswith('test') else 'train_validation_conventional.csv'))
  for (d,method),g in conventional[(conventional.role=='geometry')&(conventional.method!='binary_PCA')].groupby(['dim','method']):
   c=next(x['coefficient'] for x in cal if x['dim']==d and x['source']==method)
   for suffix,factor in [('raw',1.),('train_c',c)]:
    t=g[['case','unit','dim','split','family','noise','truth']].copy();t['method']=method+'_'+suffix;t['estimate']=g.value*factor;t['error_percent']=100*(t.estimate/t.truth-1);long.append(t)
  if file.startswith('test'):
   for (d,family,noise),g in raw.groupby(['dim','family','noise']):
    for method,col in [('energy_tensor','energy_axis_error'),('scalar_normal_tensor','normal_axis_error'),('field_spatial_covariance','spatial_axis_error')]:orientation.append(dict(dim=d,family=family,noise=noise,readout=method,mean_angle_degrees=g[col].mean(),target=g.axis_role.iloc[0],comparison_is_accuracy=g.axis_role.iloc[0] not in ['undefined','pair_separation'] or method=='field_spatial_covariance'))
   for (d,family,noise),g in conventional[conventional.method=='binary_PCA'].groupby(['dim','family','noise']):orientation.append(dict(dim=d,family=family,noise=noise,readout='binary_PCA',mean_angle_degrees=g.angle_error.mean(),target=raw[raw.family==family].axis_role.iloc[0],comparison_is_accuracy=raw[raw.family==family].axis_role.iloc[0]!='undefined'))
 # Fixed descriptive smoothed conventional benchmark: no calibration/candidate fitting to test.
 sm=pd.read_csv(OUT/'test_conventional_gaussian.csv')
 for method,g in sm[sm.method!='binary_PCA'].groupby('method'):
  t=g[['case','unit','dim','split','family','noise','truth']].copy();t['method']='Gaussian1_'+method+'_raw';t['estimate']=g.value;t['error_percent']=100*(t.estimate/t.truth-1);long.append(t)
 L=pd.concat(long,ignore_index=True);L.to_csv(OUT/'geometry_errors_long.csv',index=False)
 for (d,split,method,noise),g in L.groupby(['dim','split','method','noise']):rows.append(dict(dim=d,split=split,method=method,noise=noise,MARE=float(g.groupby('unit').error_percent.apply(lambda x:abs(x).mean()).mean()),bias=float(g.error_percent.mean()),cases=len(g),prototypes=g.unit.nunique()))
 summary=pd.DataFrame(rows);summary.to_csv(OUT/'geometry_summary.csv',index=False)
 family=L[L.split=='test'].groupby(['dim','family','method','noise']).error_percent.agg(MARE=lambda x:abs(x).mean(),bias='mean').reset_index();family.to_csv(OUT/'geometry_by_family.csv',index=False)
 pd.DataFrame(orientation).to_csv(OUT/'orientation_summary.csv',index=False)
 for d in [2,3]:
  t=L[(L.split=='test')&(L.dim==d)&(L.method.isin(['selected','energy_train_c']))];t=t.assign(absolute=abs(t.error_percent)).groupby(['unit','method']).absolute.mean().unstack();delta=(t.selected-t.energy_train_c).to_numpy();rng=np.random.default_rng(SEED+d);res=rng.choice(delta,(20000,len(delta)),replace=True).mean(1);bootstrap.append(dict(dim=d,prototypes=len(delta),delta_MARE_selected_minus_energy_pp=delta.mean(),cluster_bootstrap_low=np.quantile(res,.025),cluster_bootstrap_high=np.quantile(res,.975),resamples=20000,interpretation='descriptive n=6 prototype clusters; not a broad generalization guarantee'))
 pd.DataFrame(bootstrap).to_csv(OUT/'paired_bootstrap.csv',index=False)
 return summary,L

def tensor_stability():
 tensors=np.load(OUT/'projection_tensors.npz');rows=[];scalar=[]
 for name in tensors.files:
  Q=tensors[name];a=eigendirector(Q);family,axis,repr,weight=name.split('_');tr=a['trace'];
  if repr=='full':scalar.append(dict(case=family,axis=int(axis[4:]),weight=weight,mean_leading_scalar_fraction=float(np.nanmean(a['director'][...,-1]**2)),mean_purity=float(np.nanmean(a['purity']))))
  for eps in [.001,.01]:
   for seed in range(3):
    rng=np.random.default_rng(SEED+seed);u=rng.normal(size=Q.shape[:-1]);u/=np.linalg.norm(u,axis=-1,keepdims=True);P=Q+eps*tr[...,None,None]*u[..., :,None]*u[...,None,:];b=eigendirector(P);valid=a['defined']&b['defined'];well=valid&(a['eigengap']>10*eps);angle=np.degrees(np.arccos(np.clip(abs(np.sum(a['director']*b['director'],axis=-1)),0,1)))
    rows.append(dict(tensor=name,epsilon=eps,seed=seed,common_defined_fraction=valid.mean(),well_separated_fraction=well.mean(),mean_angle_degrees=float(np.nanmean(angle[valid])) if valid.any() else np.nan,well_separated_mean_degrees=float(np.nanmean(angle[well])) if well.any() else np.nan,well_separated_max_degrees=float(np.nanmax(angle[well])) if well.any() else np.nan,policy='PSD rank1 mixture of size epsilon*trace; tensor perturbation, not input noise'))
 pd.DataFrame(rows).to_csv(OUT/'projection_tensor_perturbation.csv',index=False);pd.DataFrame(scalar).to_csv(OUT/'projection_scalar_dominance.csv',index=False)

def make_plots(summary,L):
 plt.rcParams.update({'font.size':9})
 t=summary[summary.split=='test'];methods=['energy_original_c','energy_train_c','TV_train_c','selected','generated_s0_raw'];fig,axs=plt.subplots(1,2,figsize=(11,4))
 for d,ax in zip([2,3],axs):
  for method in methods:
   g=t[(t.dim==d)&(t.method==method)].sort_values('noise');ax.plot(100*g.noise,g.MARE,'o-',label=method)
  ax.set(xlabel='Salt-pepper input probability (%)',ylabel='MARE (%)',title=f'{d}D frozen independent test (6 prototypes)');ax.legend(fontsize=7)
 savefig(fig,'geometry_test')
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for d,ax in zip([2,3],axs):
  g=t[(t.dim==d)&(t.noise==0)&(t.method.str.endswith('train_c')|t.method.isin(['selected','Crofton_raw','marching_cubes_raw']))];ax.barh(g.method,g.MARE);ax.set(xlabel='Clean MARE (%)',title=f'{d}D same test, explicit calibration labels')
 savefig(fig,'conventional_comparison')
 p=pd.read_csv(OUT/'physics_trajectory.csv');fig,axs=plt.subplots(2,2,figsize=(10,7))
 for d,j in [(2,0),(3,1)]:
  for w in [1.15,2.3,4.6]:
   g=p[(p.dim==d)&(p.model=='base_central')&(p.normal_case==1)&(p.w_init==w)];axs[0,j].semilogy(g.N,g.one_step_rms,label=f'w_init={w}');axs[1,j].plot(g.N,g.scalar_fraction,label=f'w_init={w}')
  axs[0,j].set(title=f'{d}D oblique digital plane',xlabel='Iterations',ylabel='RMS one-step residual');axs[0,j].legend();axs[1,j].set(xlabel='Iterations',ylabel='I_s/(I_s+I_v)')
 savefig(fig,'stability_energy_split')
 profiles=pd.read_csv(OUT/'wall_profiles.csv');fig,axs=plt.subplots(1,2,figsize=(10,4))
 for d,ax in zip([2,3],axs):
  for w in [1.15,2.3,4.6]:
   for N in [0,30]:
    g=profiles[(profiles.dim==d)&(profiles.model=='base_central')&(profiles.normal_case==0)&(profiles.w_init==w)&(profiles.N==N)];ax.plot(g.distance,g.scalar,linestyle='--' if N==0 else '-',label=f'w_init={w}, N={N}')
  ax.set(xlabel='Distance in pixels (linear interpolation)',ylabel='s',title=f'{d}D axial wall; initialization vs final width',xlim=(-4,4));ax.legend(fontsize=7)
 savefig(fig,'wall_profiles')
 v=pd.read_csv(OUT/'variational_comparison.csv');fig,axs=plt.subplots(1,2,figsize=(10,4))
 for d,ax in zip([2,3],axs):
  for model,g in v[v.dim==d].groupby('model'):ax.plot(g.N,g.total-g.total.iloc[0],label=model)
  ax.set(title=f'{d}D physical Hamiltonian, not readout energy',xlabel='Iterations / descent steps',ylabel='H(n_t) - H(n_0)');ax.legend(fontsize=7)
 savefig(fig,'physical_energy')
 fig,axs=plt.subplots(1,2,figsize=(10,4));b=pd.read_csv(OUT/'interaction_profiles.csv')
 for d,ax in zip([2,3],axs):
  for width in [.75,1.5,3.]:
   g=b[(b.dim==d)&(b.family=='shell')&(b.thickness==width)&(b.N==30)];ax.plot(g.distance,g.s,label=f'thickness={width}')
  ax.axhline(0,color='k',lw=.5);ax.set(xlabel='Distance along evaluation ray (pixels)',ylabel='s, N=30',title=f'{d}D thin shell failure cases');ax.legend(fontsize=8)
 savefig(fig,'thin_structure_failures')
 pixel=pd.read_csv(OUT/'pixel_pairs.csv');fig,axs=plt.subplots(1,2,figsize=(11,4))
 for d,ax in zip([2,3],axs):
  q=pixel[(pixel.dim==d)&(pixel.pair_kind=='radius_interval')]
  for method in ['selected','Crofton' if d==2 else 'marching_cubes']:
   g=q[q.method==method].sort_values('R_pixels');ax.loglog(g.R_pixels,100*g.worst_absolute_error/((g.truth_low+g.truth_high)/2),'o',label=method+' worst pair error')
  g=q[q.method=='selected'].sort_values('R_pixels');ax.loglog(g.R_pixels,g.unavoidable_relative_percent,'ks',label='proven two-shape lower bound');ax.set(xlabel='Radius / pixel spacing',ylabel='Relative error scale (%)',title=f'{d}D identical-mask smooth sphere/circle pairs');ax.legend(fontsize=7)
 savefig(fig,'pixel_information_limit')
 fig,ax=plt.subplots(figsize=(6,4));g=pd.read_csv(OUT/'pixel_pathological.csv');ax.semilogx(g.frequency,g.truth_perimeter,'o-',label='Continuous perimeter');ax.axhline(g.selected_estimate.iloc[0],ls='--',color='k',label='Same PBGIA estimate for every mask');ax.set(xlabel='Subpixel corrugation frequency',ylabel='Perimeter',title='Identical raster; unbounded curvature family');ax.legend();savefig(fig,'pathological_information_limit')
 tensors=np.load(OUT/'projection_tensors.npz');fig,axs=plt.subplots(1,3,figsize=(12,4));cand=pd.read_csv(OUT/'defect_candidates.csv')
 for axis,ax in enumerate(axs):
  Q=tensors[f'torus_axis{axis}_vector_wall'];remaining=[j for j in range(3) if j!=axis];Q=Q[...,remaining,:][...,remaining];a=eigendirector(Q);v=a['director'];coords=np.indices(Q.shape[:2]);im=ax.imshow(a['eigengap'].T,origin='lower',cmap='magma',vmin=0,vmax=1);step=3;ax.quiver(coords[0,::step,::step].T,coords[1,::step,::step].T,v[::step,::step,0].T,v[::step,::step,1].T,pivot='mid',headlength=0,headaxislength=0,headwidth=0,color='cyan',scale=20)
  c=cand[(cand.case=='torus')&(cand.axis==axis)&(cand.weight=='wall')&(cand.representation=='vector_inplane')&(cand.refined_stable)];ax.scatter(c.row+.5,c.col+.5,facecolors='none',edgecolors='lime',s=30,label='refined loop candidates');ax.set(title=f'Integrate x{axis}; in-plane RP1',xlabel=f'x{remaining[0]}',ylabel=f'x{remaining[1]}');ax.legend(fontsize=6)
 savefig(fig,'projection_axis_dependence')
 # A single tensor cell with nontrivial loop in the bilinear projected map.
 from projective import sample_plaquette,loop_invariant
 c=cand[(cand.case=='torus')&(cand.axis==0)&(cand.weight=='wall')&(cand.representation=='vector')&(cand.refined_stable)].sort_values('refined_min_gap',ascending=False).iloc[0];Q=tensors['torus_axis0_vector_wall'];i,j=int(c.row),int(c.col);t=np.linspace(0,1,101);xx,yy=np.meshgrid(t+i,t+j,indexing='ij');m=Q.shape[-1];sample=np.stack([ndi.map_coordinates(Q[...,a,b],np.stack([xx,yy]),order=1) for a in range(m) for b in range(m)],-1).reshape(101,101,m,m);a=eigendirector(sample);fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(a['eigengap'].T,extent=(i,i+1,j,j+1),origin='lower',cmap='magma');fig.colorbar(im,ax=ax,label='Leading eigenvalue gap');ax.set(title='Projected RP2 loop: Z2 = -1\nBilinear tensor candidate, not a magnetic singularity',xlabel='map coordinate 0',ylabel='map coordinate 1');savefig(fig,'projected_defect_candidate')
 # Synthetic positive controls are visually separate from generated data.
 fig,axs=plt.subplots(1,3,figsize=(10,3));phi=np.arange(40)*2*np.pi/40
 for charge,ax in zip([0.,.5,1.],axs):
  x=np.cos(phi);y=np.sin(phi);ax.quiver(x,y,np.cos(charge*phi),np.sin(charge*phi),pivot='mid',headlength=0,headaxislength=0,headwidth=0,scale=12);ax.set(aspect='equal',title=f'Synthetic RP1 charge {charge}',xlim=(-1.4,1.4),ylim=(-1.4,1.4));ax.axis('off')
 savefig(fig,'synthetic_director_controls')

def run():
 summary,L=geometry_tables();tensor_stability();make_plots(summary,L)
 print(summary.query("split=='test'").to_string(index=False));print('ANALYSIS COMPLETE',flush=True)
if __name__=='__main__':run()
