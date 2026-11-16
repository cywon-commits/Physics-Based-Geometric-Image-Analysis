from joint_study_core import *
def run():
 df=make_manifest();rows=[]
 for dim in [2,3]:
  row=df[(df.dim==dim)&(df.split=='train')&(df.noise==0)&(df.rep==0)&(df.family==('stadium' if dim==2 else 'capsule'))].iloc[0].to_dict();f,*_=fixture(row);a=initialize(f);b=a.copy()
  for t in range(30):a=step(a,f,parameter_config());b=old_step(b,f)
  rows.append(dict(check='pinned_PBGIA30_map',dim=dim,max_delta=float(abs(a-b).max()),passed=bool(np.array_equal(a,b))))
 for row in df[(df.dim==2)&(df.split=='train')&(df.noise==0)&df.family.isin(['stadium','ellipse'])].head(6).to_dict('records'):
  f,_,_,_,_,axis=fixture(row)
  for name in ['texture','length']:
   n,Q,info=historical(f,name);raw=source_legacy_function(name)(f.astype(float),kernel_dmx=old_kernel(),parameters=(info['D'],1,info['N']));delta=float(abs(n-raw[3][...,[1,0,2]]).max());rows.append(dict(check='AST_original_'+name,dim=2,case=row['case'],max_delta=delta,passed=delta<1e-10,**info,orientation_min_error=read_axis(Q,axis,'min')['angle_error'],orientation_max_error=read_axis(Q,axis,'max')['angle_error']))
 pd.DataFrame(rows).to_csv(OUT/'stage_zero_crosschecks.csv',index=False);assert all(r['passed'] for r in rows);print('STAGE ZERO PASS',len(rows))
if __name__=='__main__':run()
