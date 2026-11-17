"""Audit public results/links and write or verify SHA256 inventory."""
from common import *
import re,csv,argparse

def public_files():
 return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache','mpl_cache','.git']) and p.name not in ['SHA256SUMS.txt','dependencies.log'])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--verify-hashes',action='store_true');args=parser.parse_args()
 if args.verify_hashes:
  checked=0
  for line in (ROOT/'SHA256SUMS.txt').read_text().splitlines():
   h,filename=line.split('  ',1);assert sha(ROOT/filename)==h,filename;checked+=1
  print('Verified SHA256:',checked,'files');return
 from geometry_study import check_lock
 check_lock();r=pd.read_csv(OUT/'test_raw.csv');assert len(r)==144 and r.unit.nunique()==12 and r.unit_error.max()<2e-7;assert np.isfinite(r[['I_E','I_TV','I_s0']]).all().all()
 p=pd.read_csv(OUT/'pixel_pairs.csv');assert p.identical_mask.all() and p.identical_prediction.all();assert (p.worst_absolute_error>=p.unavoidable_absolute_error-1e-9).all()
 c=pd.read_csv(OUT/'projective_positive_controls.csv');assert len(c)==10 and c.correct.all();v=json.load(open(OUT/'clean_reproduction_verification.json'));assert v['selection_exact_match'];assert all(x['max_absolute_difference']==0 for x in v['comparisons'])
 counter=pd.read_csv(OUT/'synchronous_counterexample.csv');assert (counter[counter.model=='base_normalized_alignment'].energy_change_per_site>1).all()
 required=['FINAL_RESEARCH_REPORT_KO.md','MODEL_AND_PROJECTION_THEORY.md','PIXEL_INFORMATION_LIMIT.md','RESULTS_MATRIX.csv','REPRODUCE.md','FINAL_CONCLUSIONS.md','README.md','DATA_DICTIONARY.md']
 for name in required:assert (ROOT/name).exists(),name
 links=[]
 for doc in ROOT.glob('*.md'):
  for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text()):
   if target.startswith(('https://','http://','#')):continue
   local=doc.parent/target.split('#')[0];assert local.exists(),(doc.name,target);links.append(target)
 assert max(p.stat().st_size for p in public_files())<10*1024*1024
 dump('package_audit.json',dict(status='PASS',test_images=len(r),test_prototypes=r.unit.nunique(),positive_control_cases=len(c),exact_reproduction_columns=len(v['comparisons']),local_links_checked=len(links),large_3d_checkpoints_included=False))
 (ROOT/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in public_files()));print('Package audit PASS; files',len(public_files()))

if __name__=='__main__':main()
