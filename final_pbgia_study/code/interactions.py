from physics import *
def run():
 rows=[];profiles=[]
 for d in [2,3]:
  specs=[('shell',8.431,dict(thickness=t)) for t in [.75,1.5,3.,6.]]+[('near_touch',3.431,dict(gap=g)) for g in [.2,.6,1.2,3.,6.]]
  for fam,R,p in specs:
   row=diagnostic_row(d,fam,R,p,size=64 if d==2 else 40);f,truth,axis,aux=fixture(row);n=initialize(f);normal=aux['inward_normal'];band=abs(aux['distance_for_band'])<1.5;input_components=int(ndi.label(f>0)[1]);input_positive_voxels=int((f>0).sum())
   for N in range(41):
    if N in [0,10,20,30,40]:
     r,_=measure(n,axis,aux['axis_role']);Is,Iv=components(n,BASE['kernel_by_dimension'][str(d)]);center=(np.array(f.shape)-1)/2;gap_s=float(ndi.map_coordinates(n[...,-1],(center+aux['shift'])[:,None],order=1)[0]);rows.append(dict(dim=d,family=fam,R=R,thickness=p.get('thickness',np.nan),gap=p.get('gap',np.nan),N=N,truth=truth,**r,I_s=Is,I_v=Iv,scalar_fraction=Is/(Is+Iv),s_max=float(n[...,-1].max()),zero_level_exists=bool(n[...,-1].min()<0<n[...,-1].max()),input_components=input_components,input_positive_voxels=input_positive_voxels,scalar_positive_components=int(ndi.label(n[...,-1]>0)[1]),scalar_positive_voxels=int((n[...,-1]>0).sum()),gap_midpoint_s=gap_s,**normal_alignment(n,normal,band)))
     if N in [0,30,40]:
      u=np.linspace(-15,15,301);a=aux['rotation'][:,0];coords=(center[:,None]+a[:,None]*u+aux['shift'][:,None]);s=ndi.map_coordinates(n[...,-1],coords,order=1,mode='nearest')
      for x,y in zip(u,s):profiles.append(dict(dim=d,family=fam,thickness=p.get('thickness',np.nan),gap=p.get('gap',np.nan),N=N,distance=x,s=y))
    n=old_step(n,f)
 pd.DataFrame(rows).to_csv(OUT/'boundary_interactions.csv',index=False);pd.DataFrame(profiles).to_csv(OUT/'interaction_profiles.csv',index=False);print('INTERACTIONS COMPLETE',flush=True)
if __name__=='__main__':run()
