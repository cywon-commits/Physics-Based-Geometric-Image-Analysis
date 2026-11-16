"""Read-only integrity/statistical audit of already frozen experiment artifacts."""
from study import *
def run():
 f=json.load(open(OUT/'pretest_selection.json'));lock=hashlib.sha256((OUT/'pretest_selection.json').read_bytes()).hexdigest();assert lock==(OUT/'pretest_selection.sha256').read_text()
 for fn,sha in f['hashes'].items():assert hashlib.sha256((OUT/fn).read_bytes()).hexdigest()==sha
 df=pd.read_csv(OUT/'test_results.csv');manifest=make_manifest();groups=0
 for (d,name),g in df.groupby(['dim','model']):
  assert len(g)==96;assert not g.case.duplicated().any();assert set(g.case)==set(manifest[(manifest.dim==d)&(manifest.split=='test')]['case']);c=f['configurations'][str(d)][name];assert set(g.N)=={c['N']};assert np.allclose(g.coefficient,c['coefficient'],rtol=1e-13);assert np.allclose(g.error_pct,100*(g.prediction/g.truth-1),atol=1e-10);assert g.finite.all();assert g.unit_error.max()<2e-5;groups+=1
 tr=pd.read_csv(OUT/'generator_trials.csv');assert len(tr)==120
 for (d,fam),g in tr[(tr.scope=='shared')&~tr.fixed_gp].groupby(['dim','family']):assert len(g)==12
 observer=pd.read_csv(OUT/'observer_trials.csv');assert len(observer)==200;assert (observer.evaluations==120).all()
 # No physical/reference mislabel: test only role geometry, calibration refers to train.
 assert (df.role=='geometry').all();assert (df.split=='test').all()
 cal=pd.read_csv(OUT/'calibration.csv');assert set(cal.reference_case)<=set(manifest[(manifest.split=='train')&(manifest.role=='reference')]['case'])
 dump('artifact_audit.json',dict(passed=True,selection_sha256=lock,test_images_independent=192,test_geometry_units=16,scored_preset_groups=groups,generator_evaluations=len(tr),observer_DE_evaluations=int(observer.evaluations.sum()),no_test_recalibration=True,no_case_filtering=True,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'code').glob('*.py')},note='stage_timings child_peak_rss_kb is cumulative maximum among children as of that stage; first_call timings are in-process, not OS-cold startup'))
 print('ARTIFACT AUDIT PASSED')
if __name__=='__main__':run()
