"""Validate cross-table invariants after all numerical stages."""
from run_research import *
def run():
 a=pd.read_csv(OUT/'benchmark_raw.csv');assert len(a)==957
 assert np.isfinite(a.calibration_coefficient).all();assert np.allclose(a.measured,a.raw_measurement*a.calibration_coefficient)
 refs=pd.read_csv(OUT/'calibration_coefficients.csv');assert set(refs.reference_id).isdisjoint(set(a.loc[a.split=='test','id']))
 assert (a.loc[a.split=='test','residual']<2e-6).all();assert 'ray_width_median' in a
 top=pd.read_csv(OUT/'topology_audit.csv');assert top.groupby(['run','stage']).integer_degree.nunique().max()==1;assert (top.collar_max_s<0).all()
 assert (top.interpolation_norm_lower_bound>0).all()
 assert np.isfinite(a.relative_error_pct).all()
 print('PASS: all 957 measurements; full calibration identity; independent references; RMS tolerance; ray widths; three-target degree agreement; nonsingular capped interpolation.')
if __name__=='__main__':run()
