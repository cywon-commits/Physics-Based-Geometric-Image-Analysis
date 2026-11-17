from data import *
from projective import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)

def draw_map(Q,label,physical_components=None):
 a=eigendirector(Q);d=a['director'];shape=Q.shape[:2];step=max(1,min(shape)//18);yy,xx=np.indices(shape);v=d if physical_components is None else d[...,physical_components];x=v[...,0];y=v[...,1];fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].imshow(a['eigengap'].T,origin='lower',cmap='magma');axs[0].quiver(yy[::step,::step].T,xx[::step,::step].T,x[::step,::step].T,y[::step,::step].T,pivot='mid',headwidth=0,headlength=0,headaxislength=0,scale=25,color='cyan');axs[0].set_title('Unoriented lines + eigenvalue gap');im=axs[1].imshow(a['purity'].T,origin='lower',vmin=.25,vmax=1);axs[1].set_title('trace-normalized tensor purity');fig.colorbar(im,ax=axs[1]);fig.tight_layout();fig.savefig(FIG/(label+'.png'),dpi=160);plt.close(fig)

def run():
 import resource
 started=time.perf_counter();controls().to_csv(OUT/'projective_positive_controls.csv',index=False);rows=[];candidates=[];loops=[];maps=[];tensors={};case_manifest=[];costs=[]
 cases=[('sphere',7.431,{}),('capsule',4.431,dict(Lratio=1.7)),('torus',10.431,dict(minor=2.731)),('pair',3.931,dict(gap=7.3)),('ellipsoid',6.431,dict(axes=[1.83,.83,.57])),('shell',8.431,dict(thickness=2.3))]
 for family,R,params in cases:
  row=diagnostic_row(3,family,R,params,size=40,seed=len(rows));case_manifest.append(row);f,truth,axis,aux=fixture(row);n=initialize(f);field_start=time.perf_counter()
  for _ in range(30):n=old_step(n,f)
  field_seconds=time.perf_counter()-field_start;projection_start=time.perf_counter()
  for fiber_axis in range(3):
   remaining=[j for j in range(3) if j!=fiber_axis]
   for weight in ['uniform','wall']:
    for representation in ['full','vector']:
     Q,mass=fiber_tensor(n,fiber_axis,representation,weight);a=eigendirector(Q);scanned,summary=plaquette_loops(Q);label=f'{family}_axis{fiber_axis}_{representation}_{weight}';tensors[label]=Q;rows.append(dict(case=family,axis=fiber_axis,weight=weight,representation=representation,projective_dimension=Q.shape[-1]-1,defined_fraction=float(a['defined'].mean()),mean_purity=float(np.nanmean(a['purity'])),minimum_gap=float(a['eigengap'].min()),median_gap=float(np.median(a['eigengap'])),mean_effective_rank=float(np.nanmean(1/np.maximum(a['purity'],1e-30))),**summary))
     for cand in scanned:candidates.append(dict(case=family,axis=fiber_axis,weight=weight,representation=representation,**cand))
     for radius in [3.,6.,10.]:
      r=loop_invariant(sample_tensor_loop(Q,(np.array(Q.shape[:2])-1)/2,radius));loops.append(dict(case=family,axis=fiber_axis,weight=weight,representation=representation,radius=radius,**r))
     if representation=='vector':
      Q2=Q[...,remaining,:][...,remaining];b=eigendirector(Q2);cs,ss=plaquette_loops(Q2);rows.append(dict(case=family,axis=fiber_axis,weight=weight,representation='vector_inplane',projective_dimension=1,defined_fraction=float(b['defined'].mean()),mean_purity=float(np.nanmean(b['purity'])),minimum_gap=float(b['eigengap'].min()),median_gap=float(np.median(b['eigengap'])),mean_effective_rank=float(np.nanmean(1/np.maximum(b['purity'],1e-30))),**ss))
      for cand in cs:candidates.append(dict(case=family,axis=fiber_axis,weight=weight,representation='vector_inplane',**cand))
      for radius in [3.,6.,10.]:loops.append(dict(case=family,axis=fiber_axis,weight=weight,representation='vector_inplane',radius=radius,**loop_invariant(sample_tensor_loop(Q2,(np.array(Q2.shape[:2])-1)/2,radius))))
      if fiber_axis==2 and weight=='wall':
       draw_map(Q2,label+'_inplane');
       for i,j in np.ndindex(*Q2.shape[:2]):maps.append(dict(case=family,row=i,col=j,Q00=Q2[i,j,0,0],Q01=Q2[i,j,0,1],Q11=Q2[i,j,1,1],gap=b['eigengap'][i,j],defined=b['defined'][i,j]))
  costs.append(dict(case=family,field_seconds=field_seconds,projection_scan_plot_seconds=time.perf_counter()-projection_start));print('PROJECTION',family,flush=True)
 # 2D native vector director and a full S3 field slice: directorlines, never materialflow.
 for d in [2,3]:
  row=diagnostic_row(d,'stadium' if d==2 else 'capsule',4.431,dict(Lratio=2.0),size=64 if d==2 else 40);f,truth,axis,aux=fixture(row);n=initialize(f)
  for _ in range(30):n=old_step(n,f)
  if d==3:n=n[:,:,n.shape[2]//2,:];f=f[:,:,f.shape[2]//2]
  v=n[...,:2];Q=v[..., :,None]*v[...,None,:];draw_map(Q,f'native_vector_lines_{d}d');yy,xx=np.indices(f.shape);u=v[...,0];w=v[...,1];norm=np.sqrt(u*u+w*w);u=u/np.maximum(norm,1e-12);w=w/np.maximum(norm,1e-12);u[norm<.03]=np.nan;w[norm<.03]=np.nan;fig,ax=plt.subplots(figsize=(5,5));ax.imshow(f.T,origin='lower',cmap='Greys',alpha=.3);ax.streamplot(np.arange(f.shape[0]),np.arange(f.shape[1]),u.T,w.T,density=.9,arrowsize=0,color='tab:blue');ax.set_title('Integral curves: orientation convention, not transport');fig.tight_layout();fig.savefig(FIG/f'normal_integral_lines_{d}d.png',dpi=160);plt.close(fig)
 pd.DataFrame(rows).to_csv(OUT/'projection_summary.csv',index=False);pd.DataFrame(candidates).to_csv(OUT/'defect_candidates.csv',index=False);pd.DataFrame(loops).to_csv(OUT/'projection_loops.csv',index=False);pd.DataFrame(maps).to_csv(OUT/'projection_inplane_maps.csv',index=False);pd.DataFrame(case_manifest).to_csv(OUT/'projection_case_manifest.csv',index=False);pd.DataFrame(costs).to_csv(OUT/'projection_costs.csv',index=False);np.savez_compressed(OUT/'projection_tensors.npz',**tensors);dump('projection_audit.json',dict(wall_seconds=time.perf_counter()-started,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,tensors=len(tensors),positive_controls=int(controls().correct.sum()),coordinate_order='array axes x0,x1,x2; maps ordered by remaining ascending axes'));print('PROJECTION COMPLETE',flush=True)
if __name__=='__main__':run()
