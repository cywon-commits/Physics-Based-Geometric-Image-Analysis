from run_research import *
from benchmark_dynamic import tensor,conventional

def run():
 df=pd.read_csv(OUT/'benchmark_raw.csv');kern=json.load(open(OUT/'kernels.json'))['parameters'];rows=[]
 for ref,g in df[df.split=='calibration'].groupby('id'):
  c=g.iloc[0];n=np.load(OUT/(ref+'_field.npz'))['n'];mask=np.load(OUT/(ref+'_field.npz'))['mask'];base={method:val for method,val,_ in conventional(mask,None)[0]}
  for method,p in kern.items():base['PBGIA_'+method]=float(np.trace(tensor(n,p)))
  for method,val in base.items():
   coeff=c.truth/val;rows.append(dict(reference_id=ref,dim=c.dim,h=c.h,method=method,reference_R=c.R,reference_truth=c.truth,reference_raw_measurement=val,coefficient=coeff))
   calmethod=method if method.startswith('PBGIA') or method=='Gaussian_sigma1+structure_tensor' else method+'+reference_calibrated'
   idx=(df.dim==c.dim)&(df.h==c.h)&(df.method==calmethod);df.loc[idx,'calibration_coefficient']=coeff;df.loc[idx,'raw_measurement']=df.loc[idx,'measured']/coeff
   if calmethod!=method:
    idx=(df.dim==c.dim)&(df.h==c.h)&(df.method==method);df.loc[idx,'calibration_coefficient']=1.;df.loc[idx,'raw_measurement']=df.loc[idx,'measured']
 df.to_csv(OUT/'benchmark_raw.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'calibration_coefficients.csv',index=False)
 print('CALIBRATION AUDIT DONE')
if __name__=='__main__':run()
