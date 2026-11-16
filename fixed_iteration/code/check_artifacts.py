from study_core import *
def run():
 df=pd.read_csv(OUT/'split_manifest.csv');assert len(df)==296;assert df.groupby('unit').split.nunique().max()==1
 for _,g in df.groupby(['dim','family']):assert g.groupby('R').split.nunique().max()==1
 cfg=json.load(open(OUT/'selected_configuration.json'));frozen_hash=hashlib.sha256((OUT/'selected_configuration.json').read_bytes()).hexdigest();assert frozen_hash==(OUT/'selection_lock.sha256').read_text();assert frozen_hash==json.load(open(OUT/'test_audit.json'))['selection_sha256']
 assert hashlib.sha256((OUT/'split_manifest.csv').read_bytes()).hexdigest()==cfg['split_manifest_sha256']
 assert hashlib.sha256((OUT/'kernel_by_iteration.json').read_bytes()).hexdigest()==cfg['kernel_file_sha256']
 kernels=json.load(open(OUT/'kernel_by_iteration.json'))
 for dim in ['2','3']:
  for N in CHECKPOINTS:assert kernels['kernels'][dim][str(N)]['evaluations']==270
 for f in ['train_results.csv','validation_results.csv','test_results.csv']:
  a=pd.read_csv(OUT/f);assert np.isfinite(a.measured).all();assert np.allclose(a.measured,a.raw_measurement*a.calibration_coefficient);assert (a.unit_error<1e-5).all()
 test=pd.read_csv(OUT/'test_results.csv');assert set(test.N)=={cfg['selected_N'],40};assert len(test.case.unique())==132
 assert set(pd.read_csv(OUT/'train_results.csv').unit).isdisjoint(set(test.unit));assert set(pd.read_csv(OUT/'validation_results.csv').unit).isdisjoint(set(test.unit))
 for r in kernels['calibration']:assert df.set_index('case').loc[r['reference_case'],'split']=='train'
 selection=pd.read_csv(OUT/'selection_sensitivity.csv');eligible=selection[(selection.tolerance==.1)&selection.passed];assert cfg['selected_N']==int(eligible.N.min())
 degree=pd.read_csv(OUT/'degree_results.csv');assert degree.degree_target_agreement.all()
 print('PASS: split/calibration independence; locked pre-test choice; equal optimizer budgets; normalization; raw/calibrated identities; smallest eligible N; representative target agreement.')
if __name__=='__main__':run()
