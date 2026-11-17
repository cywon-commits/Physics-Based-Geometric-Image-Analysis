from data import *

def components(n,p,spacing=1.):
 grads=[derivative(n,j,p)/spacing for j in range(n.ndim-1)];Is=sum(np.sum(g[...,-1]**2,dtype=float) for g in grads)*spacing**(n.ndim-1);Iv=sum(np.sum(g[...,:-1]**2,dtype=float) for g in grads)*spacing**(n.ndim-1);return float(Is),float(Iv)

def normal_alignment(n,normal,band):
 v=n[...,:-1];vn=np.linalg.norm(v,axis=-1);g=gradient(n[...,-1]);gn=np.linalg.norm(g,axis=-1);valid=band&(vn>1e-7);cross=band&(vn>1e-7)&(gn>1e-8)
 return dict(normal_cos=float(np.mean(np.sum(v*normal,axis=-1)[valid]/vn[valid])) if valid.any() else np.nan,normal_abs_cos=float(np.mean(abs(np.sum(v*normal,axis=-1)[valid]/vn[valid]))) if valid.any() else np.nan,normal_coverage=float(valid.sum()/max(band.sum(),1)),v_grad_s_cos=float(np.mean(np.sum(v*g,axis=-1)[cross]/(vn[cross]*gn[cross]))) if cross.any() else np.nan)

def profile_width(n,point,normal,spacing=1.,radius=6.):
 # Evaluation rays only; no analytic distances enter initialization.
 u=np.linspace(-radius,radius,601);coords=((point[:,None]+normal[:,None]*u)/spacing+(np.array(n.shape[:-1])[:,None]-1)/2);s=ndi.map_coordinates(n[...,-1],coords,order=1,mode='nearest');values={}
 for level in [-.7,.7]:
  hits=np.flatnonzero((s[:-1]-level)*(s[1:]-level)<=0);cross=[]
  for j in hits:
   if s[j+1]!=s[j]:cross.append(float(u[j]+(level-s[j])*(u[j+1]-u[j])/(s[j+1]-s[j])))
  values[level]=min(cross,key=abs) if cross else np.nan
 width=abs(values[.7]-values[-.7]) if all(np.isfinite(list(values.values()))) else np.nan
 return width,u,s

def update_g4(n,f):
 H=sum(np.roll(n,k,j) for j in range(f.ndim) for k in [-1,1]);H[...,:f.ndim]+=np.stack([g4(n[...,-1],j) for j in range(f.ndim)],-1);H[...,-1]+=2*f;hn=np.linalg.norm(H,axis=-1,keepdims=True);return np.where(hn>1e-12,H/np.maximum(hn,1e-12),n).astype('float32')

def synchronous_counterexample():
 # Analytic stability counterexample, not a binary-EDT geometric fixture.
 rows=[]
 for d in [2,3]:
  sigma=np.where(np.indices((8,)*d).sum(0)%2,1.,-1.);n=np.zeros(sigma.shape+(d+1,));n[...,0]=np.sqrt(1-.7**2)*sigma;n[...,-1]=.7*sigma
  for model,update in [('base_normalized_alignment',old_step),('reciprocal_Riemannian_descent',lambda a,b:variational_step(a,b)[0])]:
   initial=physical_energies(n,sigma)['total'];final=physical_energies(update(n,sigma),sigma)['total'];rows.append(dict(dim=d,model=model,initial_energy=initial,one_step_energy=final,energy_change_per_site=(final-initial)/sigma.size,initialization='analytic checkerboard spin, NOT binary EDT'))
 pd.DataFrame(rows).to_csv(OUT/'synchronous_counterexample.csv',index=False)

def run():
 records=[];profiles=[];init_comparison=[]
 for d in [2,3]:
  S=80 if d==2 else 64;coords=np.indices((S,)*d,dtype=float)-(S-1)/2
  for angle in [0,1]:
   normal=np.zeros(d);normal[0]=1
   if angle:normal=np.array([1.,.6] if d==2 else [1.,.5,.3]);normal/=np.linalg.norm(normal)
   level=np.einsum('i,i...->...',normal,coords)+.173;f=np.where(level>=0,1.,-1.).astype('float32');band=abs(level)<1.5;normalfield=np.broadcast_to(normal,f.shape+(d,));ROI=np.zeros(f.shape,bool);m=20 if d==2 else 18;ROI[(slice(m,-m),)*d]=True
   for w in [1.15,2.3,4.6]:
    for kind in ['base_central','g4_generation_control']:
     update=old_step if kind=='base_central' else update_g4;n=initialize(f,width=w);prev=n.copy();states={}
     for N in range(41):
      energy=physical_energies(n,f);Is,Iv=components(n,BASE['kernel_by_dimension'][str(d)]);n1=update(n,f);diff=np.linalg.norm(n1-n,axis=-1);two=np.linalg.norm(n1-prev,axis=-1);row=dict(dim=d,normal_case=angle,w_init=w,model=kind,N=N,**{f'E_{k}':v for k,v in energy.items()},I_s=Is,I_v=Iv,scalar_fraction=Is/(Is+Iv),one_step_rms=float(np.sqrt(np.mean(diff**2))),one_step_max=float(diff.max()),two_step_rms=float(np.sqrt(np.mean(two**2))),band_residual=float(np.sqrt(np.mean(diff[band&ROI]**2))),unit_error=float(abs(np.linalg.norm(n,axis=-1)-1).max()))
      if N in [0,10,20,30,40]:
       point=-.173*normal;ww,u,s=profile_width(n,point,normal);row.update(w_eff_threshold_distance=ww,w_eff_tanh_scale=ww/(2*np.arctanh(.7)),**normal_alignment(n,normalfield,band&ROI))
       for a,b in zip(u,s):profiles.append(dict(dim=d,model=kind,N=N,w_init=w,normal_case=angle,distance=a,scalar=b))
       states[N]=n.copy()
      records.append(row);prev,n=n,n1
     init_comparison.append(dict(dim=d,model=kind,normal_case=angle,w_init=w,scalar_N30=states[30][...,-1].copy()))
   print('PHYSICS PLANAR',d,angle,flush=True)
 # Raw state distance is a diagnostic; no acceptance threshold/gate.
 stable=[]
 for d in [2,3]:
  for kind in ['base_central','g4_generation_control']:
   for ang in [0,1]:
    entries=[x for x in init_comparison if x['dim']==d and x['model']==kind and x['normal_case']==ang];base=next(x for x in entries if x['w_init']==2.3)['scalar_N30']
    for x in entries:stable.append(dict(dim=d,model=kind,normal_case=ang,w_init=x['w_init'],N=30,scalar_RMS_difference_to_w2p3=float(np.sqrt(np.mean((x['scalar_N30']-base)**2)))))
 pd.DataFrame(records).to_csv(OUT/'physics_trajectory.csv',index=False);pd.DataFrame(profiles).to_csv(OUT/'wall_profiles.csv',index=False);pd.DataFrame(stable).to_csv(OUT/'initialization_stability.csv',index=False)
 # Optional, explicitly separate variational gradient descent control.
 comparison=[]
 for d in [2,3]:
  row=diagnostic_row(d,'circle' if d==2 else 'sphere',6.431,size=40);f,truth,axis,aux=fixture(row);initial=initialize(f).astype(float)
  for kind in ['base_alignment','reciprocal_Riemannian_descent']:
   n=initial.copy()
   for N in range(41):
    e=physical_energies(n,f);r,_=measure(n);comparison.append(dict(dim=d,model=kind,N=N,truth=truth,**e,**r))
    n=old_step(n,f) if kind=='base_alignment' else variational_step(n,f)[0]
 pd.DataFrame(comparison).to_csv(OUT/'variational_comparison.csv',index=False)
 # Bounded physical-spacing diagnostic, same object not just padding.
 spacing_rows=[]
 for d in [2,3]:
  for a in [.75,1.,1.5]:
   S=int(np.ceil(48/a));S+=S%2;x=(np.indices((S,)*d,dtype=float)-(S-1)/2)*a
   for radius,role in [(6.431,'evaluation'),(8.431,'reference')]:
    f=np.where(np.linalg.norm(x-np.array([.173,-.291,.113][:d]).reshape((d,)+(1,)*d),axis=0)<=radius,1.,-1.).astype('float32');n=initialize(f,width=2.3/a)
    for _ in range(30):n=old_step(n,f,D=1/a)
    r,_=measure(n,spacing=a);truth=2*np.pi*radius if d==2 else 4*np.pi*radius*radius;spacing_rows.append(dict(dim=d,spacing=a,R=radius,role=role,truth=truth,**r,width_policy='physical init2.3; J/h/N fixed, not PDElimit'))
 pd.DataFrame(spacing_rows).to_csv(OUT/'spacing_diagnostics.csv',index=False);synchronous_counterexample();print('PHYSICS COMPLETE',flush=True)
if __name__=='__main__':run()
