from data import *
from readouts import READOUT_METADATA
import hashlib

def run():
 df=pd.read_csv(OUT/'raw.csv');manifest=make_manifest();protocol=json.load(open(OUT/'protocol.json'));assert len(manifest)==562;assert len(df)==6182;assert set(df.case)==set(manifest.case);assert protocol['manifest_sha256']==hashlib.sha256((OUT/'manifest.csv').read_bytes()).hexdigest();assert (OUT/'manifest.sha256').read_text()==protocol['manifest_sha256']
 for case,g in df.groupby('case'):
  assert len(g)==11 and not g.method.duplicated().any();assert set(g.method)==set(READOUT_METADATA);assert g.energy_scalar.nunique()==1 and g.energy_vector.nunique()==1;assert g.field_seconds.nunique()==1;assert g.unit_error.max()<1e-12
  energy=g[g.method=='energy'].iloc[0];assert np.isclose(energy.value,energy.energy_scalar+energy.energy_vector,rtol=1e-12)
 assert np.isfinite(df.value).all();assert (df.value>=0).all();assert set(df[df.source=='fixedG4_generated'].N)=={10};assert set(df[df.source=='analytic_oracle'].N)=={0}
 coeff=pd.read_csv(OUT/'calibration_coefficients.csv');refs=df[df.split=='reference'];assert refs.family.isin(['circle','sphere']).all();assert np.allclose(refs.width,2.3) and np.allclose(refs.spacing,1);assert refs.case.nunique()==12
 analyzed=pd.read_csv(OUT/'analysis_raw.csv');assert np.allclose(analyzed.prediction_global_c,analyzed.value*analyzed.coefficient);assert np.allclose(analyzed.error_global_c_pct,100*(analyzed.prediction_global_c/analyzed.truth-1))
 theory=pd.read_csv(OUT/'continuum_radial.csv');two=theory[(theory.dim==2)&theory.method.isin(['tv','weighted_tv'])];assert np.allclose(two.q,1,atol=1e-10)
 for eps in [.2,.35,.5]:
  a=theory[theory.method==f'point_coarea_eps_{eps:.2f}'].sort_values(['dim','R','width']);b=theory[theory.method==f'integrated_coarea_eps_{eps:.2f}'].sort_values(['dim','R','width']);assert np.allclose(a.value,b.value,rtol=1e-12)
 payload=dict(passed=True,cases=562,method_rows=6182,methods=11,dimensions=[2,3],actual_fixed_generator=True,all_cases_retained=True,no_model_selection=True,reference_cases=12,raw_sha256=hashlib.sha256((OUT/'raw.csv').read_bytes()).hexdigest(),manifest_sha256=protocol['manifest_sha256'],source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for base in ['code','tests','reference'] for p in (ROOT/base).rglob('*') if p.is_file() and '__pycache__' not in str(p)})
 (OUT/'artifact_audit.json').write_text(json.dumps(payload,indent=2));print('AUDIT PASSED')
if __name__=='__main__':run()
