from run_research import *
from benchmark_dynamic import tensor,angle_error
from topology_audit import certificate,cap,q3

def labels(series):
 return series.str.replace('Gaussian_sigma1_threshold+marching_squares_PCA','Gaussian→MS/PCA',regex=False).str.replace('Gaussian_sigma1_threshold+marching_cubes_PCA','Gaussian→MC/PCA',regex=False).str.replace('Gaussian_sigma1+structure_tensor','Gaussian/ST',regex=False).str.replace('+reference_calibrated',' (cal)',regex=False).str.replace('marching_cubes+surface_tensor','MC/surface tensor',regex=False).str.replace('marching_squares','MS/PCA',regex=False).str.replace('voxel_faces+PCA','voxel faces/PCA',regex=False)

def run():
 # Re-evaluate all P0 normal metrics using continuous theoretical normals.
 df=pd.read_csv(OUT/'p0_metrics.csv')
 for idx,c in df.iterrows():
  d,mask,nu,_,_=fixture(c['shape'],N=int(c.N),R=c.R,r=c.r,angle=c.angle)
  n=np.load(OUT/(c.run+f'_t{int(c.iteration):04d}.npz'))['n']
  for key,val in metrics(n,mask,nu,d).items():df.loc[idx,key]=val
 df.to_csv(OUT/'p0_metrics.csv',index=False)
 # Field quality and degree with each optimized generator (different experiment from measuring fixed fields).
 kernels=json.load(open(OUT/'kernels.json'))['parameters'];rows=[]
 for dim,kind,R,N in [(2,'stadium',11,64),(3,'sphere',9,48),(3,'capsule',7,48),(3,'torus',12,48)]:
  for h in [2.,4.]:
   for method,p in kernels.items():
    d,mask,nu,truth,ax=fixture(kind,N=N,R=R,angle=.61,dim=dim);n,info,_=solve(mask,dict(h=h,D=1,J=1,p=p))
    row=dict(dim=dim,shape=kind,h=h,D=1,J=1,generator_kernel=method,measurement_kernel='central',**info,**metrics(n,mask,nu,d),central_trace=float(np.trace(tensor(n,(0,0,0)))))
    if dim==3:row.update(**certificate(n),**degree(cap(n)),Q3_fd=q3(n))
    rows.append(row)
 pd.DataFrame(rows).to_csv(OUT/'generator_kernel_sweep.csv',index=False)
 # Frequency response and isotropy; all tensors use array coordinate convention axis0 first.
 freq=[]
 for dim in [2,3]:
  for method,(b,t,u) in kernels.items():
   for freqmag in [.2,.5,1.,1.5,2.,2.8]:
    for theta in np.linspace(0,np.pi/2,31):
     direction=np.array([np.cos(theta),np.sin(theta)]+([.43] if dim==3 else []));direction/=np.linalg.norm(direction);k=freqmag*direction
     response=[]
     for i in range(dim):
      val=2*(.5-2*b)*np.sin(k[i])+2*b*np.sin(2*k[i])
      for j in range(dim):
       if j!=i:val*=1-2*t-2*u+2*t*np.cos(k[j])+2*u*np.cos(2*k[j])
      response.append(val)
     response=np.array(response);gain=np.linalg.norm(response)/freqmag
     ae=np.degrees(np.arccos(np.clip(response@direction/max(np.linalg.norm(response),1e-15),-1,1)))
     freq.append(dict(dim=dim,method=method,frequency=freqmag,theta=theta,gain=gain,direction_error_deg=ae))
 pd.DataFrame(freq).to_csv(OUT/'kernel_frequency_response.csv',index=False)
 # Figures.
 fig,axs=plt.subplots(1,3,figsize=(13,3.8))
 hist=pd.read_csv(OUT/'p0_history.csv')
 for name in ['p0_01_torus','p0_07_sphere','p0_13_capsule']:
  h=hist[hist.run==name];axs[0].semilogy(h.iteration,h.residual,label=name.split('_')[-1]);axs[1].semilogy(h.iteration,h.max_residual);axs[2].semilogy(h.iteration,h.two_step_residual)
 for a,title in zip(axs,['RMS fixed-point residual','Maximum local update','Two-step RMS (odd/even)']):a.set_title(title);a.set_xlabel('iteration');a.grid(alpha=.2)
 axs[0].legend();fig.tight_layout();fig.savefig(OUT/'convergence.png',dpi=170);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(13,3.7))
 finals=df.groupby('run').tail(1)
 for name in ['p0_01_torus','p0_07_sphere','p0_13_capsule']:
  c=finals[finals.run==name].iloc[0];d,mask,nu,_,_=fixture(c['shape'],N=48,R=c.R,r=c.r,angle=c.angle);n=np.load(OUT/(name+f'_t{int(c.iteration):04d}.npz'))['n'];s=n[...,-1];v=n[...,:3];vn=np.linalg.norm(v,axis=-1);band=abs(d)<3
  bins=np.linspace(-3,3,26);xc=(bins[1:]+bins[:-1])/2;prof=[np.mean(s[(d>=lo)&(d<hi)]) if ((d>=lo)&(d<hi)).any() else np.nan for lo,hi in zip(bins[:-1],bins[1:])];axs[0].plot(xc,prof,label=c['shape'])
  inward=-(v*nu).sum(-1)/np.maximum(vn,1e-12);axs[1].hist(inward[(abs(s)<.7)&(abs(d)<1.5)],bins=np.linspace(.85,1,31),histtype='step',label=c['shape'])
  ratio=v/np.where(abs(nu)>.15,-nu,np.nan)/np.maximum(vn[...,None],1e-12)
  vals=ratio[(abs(s)<.7)&(abs(d)<1.5)];axs[2].hist(vals[np.isfinite(vals)],bins=np.linspace(0,2,51),histtype='step',label=c['shape'])
 axs[0].set(xlabel='continuous signed distance (evaluation only)',ylabel='completed s',title='Measured wall profiles');axs[1].set(xlabel='cos(v, inward normal)',title='Signed normal alignment');axs[2].set(xlabel='v_i / (|v| inward_normal_i)',title='Component ratios; |normal_i| > .15')
 for a in axs:a.legend(fontsize=8);a.grid(alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'profiles_normals.png',dpi=170);plt.close(fig)
 fig=plt.figure(figsize=(10,9))
 for row,name in enumerate(['p0_01_torus','p0_07_sphere','p0_13_capsule']):
  c=finals[finals.run==name].iloc[0]
  for col,it in enumerate([0,25,int(c.iteration)]):
   path=OUT/(name+f'_t{it:04d}.npz');n=np.load(path)['n'];verts,faces,_,_=marching_cubes(n[...,-1],0,step_size=2)
   ax=fig.add_subplot(3,3,row*3+col+1,projection='3d');ax.plot_trisurf(verts[:,0],verts[:,1],verts[:,2],triangles=faces,color=['#52a5b0','#dfb267','#a29bd4'][row],linewidth=0,alpha=.95)
   ax.set(xlim=(4,44),ylim=(4,44),zlim=(4,44),title=f"{c['shape']} t={it}");ax.set_box_aspect((1,1,1));ax.set_axis_off()
 fig.tight_layout();fig.savefig(OUT/'snapshots_3d.png',dpi=170);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(15,7));summ=pd.read_csv(OUT/'benchmark_summary.csv')
 for ax,dim in zip(axs,[2,3]):
  g=summ[(summ.dim==dim)&(summ.h==2)&(summ.noise==0)].sort_values('MARE_pct');y=np.arange(len(g));ax.errorbar(g.MARE_pct,y,xerr=np.stack([g.MARE_pct-g.CI95_low,g.CI95_high-g.MARE_pct]),fmt='o',capsize=3);ax.set_yticks(y,labels(g.method));ax.set(xlabel='Mean absolute relative geometry error (%)',title=f'{dim}D clean; case bootstrap 95% interval');ax.grid(axis='x',alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'benchmark_geometry.png',dpi=170);plt.close(fig)
 raw=pd.read_csv(OUT/'benchmark_raw.csv');g=raw[(raw.split=='test')&(raw.noise==0)&(raw.h==2)];fig,axs=plt.subplots(1,2,figsize=(15,6))
 for ax,dim in zip(axs,[2,3]):
  tab=g[g.dim==dim].pivot_table(index='method',columns='kind',values='relative_error_pct',aggfunc=lambda a:np.mean(abs(a)));im=ax.imshow(np.log10(tab.values+.05),cmap='magma',aspect='auto');ax.set_xticks(range(len(tab.columns)),tab.columns,rotation=30);ax.set_yticks(range(len(tab.index)),labels(tab.index));ax.set_title(f'{dim}D absolute error % (log color)')
  for i in range(len(tab.index)):
   for j in range(len(tab.columns)):ax.text(j,i,f'{tab.iloc[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if tab.iloc[i,j]<5 else 'black')
 fig.tight_layout();fig.savefig(OUT/'failure_heatmap.png',dpi=170);plt.close(fig)
 freq=pd.DataFrame(freq);fig,axs=plt.subplots(1,2,figsize=(10,4))
 for method in kernels:
  g=freq[(freq.dim==2)&(freq.method==method)].groupby('frequency').agg(gain=('gain','mean'),err=('direction_error_deg','max'));axs[0].plot(g.index,g.gain,'o-',label=method);axs[1].plot(g.index,g.err,'o-',label=method)
 axs[0].set(xlabel='frequency (radians / grid spacing)',ylabel='mean derivative gain / ideal');axs[1].set(xlabel='frequency',ylabel='maximum direction error (degrees)')
 for a in axs:a.legend();a.grid(alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'kernel_response.png',dpi=170);plt.close(fig)
 print('FINALIZED',flush=True)
if __name__=='__main__':run()
