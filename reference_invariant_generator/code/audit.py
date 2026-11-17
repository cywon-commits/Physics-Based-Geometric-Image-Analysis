from evaluation import *
def run():
 frozen=check_lock();r=pd.read_csv(OUT/'test_raw_2d.csv');expected=set(manifest().query("split=='test'")['case'])
 for name,g in r.groupby('model'):
  assert len(g)==408;assert set(g.case)==expected;assert not g.case.duplicated().any();assert np.allclose(g.q,g.I/g.truth);assert np.allclose(g.I,g.I_scalar+g.I_vector,rtol=1e-12);assert g.unit_error.max()<1e-10
  assert g.N.nunique()==1;assert g.N.iloc[0]==frozen['configurations'][name]['N']
 refs=pd.read_csv(OUT/'reference_raw.csv');assert (refs.role=='reference').all();assert (refs.split=='train').all()
 trials=pd.read_csv(OUT/'train_trials.csv');assert len(trials)==14;trace=pd.read_csv(OUT/'train_optimizer_trace.csv')
 for _,t in trials.iterrows():assert len(trace[trace.model==t.model])==t.nfev
 # Verify read-only uploaded material, including original notebook bytes.
 provenance=json.load(open(ROOT/'reference/PROVENANCE.json'))
 for path,h in provenance['files_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
 hashes={str(p.relative_to(ROOT)):sha(p) for folder in ['code','tests','reference','design'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)};dump('source_sha256.json',hashes)
 dump('artifact_audit.json',dict(passed=True,freeze_sha256=sha(OUT/'model_freeze_2d.json'),independent_test_prototypes=17,test_images=408,optimizer_runs=14,evaluations=int(trials.nfev.sum()),all_cases_retained=True,I_equals_scalar_plus_vector=True,reference_only_train=True,original_uploaded_material_unchanged=True,threeD_status='verified_partial_only'))
 print('AUDIT PASSED')
if __name__=='__main__':run()
