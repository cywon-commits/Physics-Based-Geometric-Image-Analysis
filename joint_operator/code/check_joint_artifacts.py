from joint_study_core import *
def run():
 df=make_manifest();assert len(df)==296;assert df.groupby('unit').split.nunique().max()==1
 prior=pd.read_csv(ROOT/'vendor/pbgia30/split_manifest.csv');assert set(df[['dim','family','R']].itertuples(index=False,name=None)).isdisjoint(set(prior[['dim','family','R']].itertuples(index=False,name=None)))
 lock=hashlib.sha256((OUT/'pretest_selection.json').read_bytes()).hexdigest();frozen=json.load(open(OUT/'pretest_selection.json'));assert lock==(OUT/'pretest_selection.sha256').read_text();assert lock==json.load(open(OUT/'test_audit.json'))['selection_sha256'];assert lock==json.load(open(OUT/'deployment_gate.json'))['selection_sha256'];assert frozen['protocol_hash']==hashlib.sha256((OUT/'experiment_config.json').read_bytes()).hexdigest();assert frozen['split_hash']==hashlib.sha256((OUT/'split_manifest.csv').read_bytes()).hexdigest();assert frozen['trained_models_hash']==hashlib.sha256((OUT/'trained_models.json').read_bytes()).hexdigest()
 search=pd.read_csv(OUT/'generator_search.csv');assert (search.groupby(['dim','family']).size()==24).all();assert len(search)==144
 models=json.load(open(OUT/'trained_models.json'))
 for dim,families in models.items():
  for family,m in families.items():
   for N,obs in m['observers'].items():assert obs['evaluations']==180
   K=exchange_kernel(int(dim),m['generator']);assert K.min()>=0;assert abs(K.sum()-2*int(dim))<1e-9
 for name in ['train_results.csv','validation_results.csv','test_results.csv']:
  a=pd.read_csv(OUT/name);assert np.isfinite(a.measured).all();assert np.allclose(a.measured,a.raw_measurement*a.calibration_coefficient);assert (a.unit_error<1e-5).all();assert a.finite.all()
 test=pd.read_csv(OUT/'test_results.csv');assert test[test.model=='joint_selected']['case'].nunique()==132
 assert set(test.unit).isdisjoint(set(pd.read_csv(OUT/'train_results.csv').unit));assert set(test.unit).isdisjoint(set(pd.read_csv(OUT/'validation_results.csv').unit))
 for d in [2,3]:
  c=frozen['selection'][str(d)];assert set(test[(test.model=='joint_selected')&(test.dim==d)].N)=={c['N']}
 # PBGIA30 calibration is intentionally frozen or separately re-estimated from train; never fitted on test.
 assert (df[df.role=='calibration'].split=='train').all()
 print('PASS: 144 equal-budget generator evaluations; 180 observer evaluations per N; independent prototypes; locked pretest choice; calibration identity; normalization; production acceptance gate.')
if __name__=='__main__':run()
