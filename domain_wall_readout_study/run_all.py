"""Explicit reproducibility stages; completed raw results cannot be overwritten."""
from pathlib import Path
import argparse,subprocess,sys,time,json,os,resource
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output'
def run(stage):
 commands={'prepare':['code/run_study.py','prepare'],'tests':['-m','pytest','tests','-q'],'run':['code/run_study.py','run'],'diagnostics':['code/diagnostics_extra.py'],'paired':['code/paired_comparisons.py'],'report':['code/analysis_report.py'],'audit':['code/audit.py']}
 if stage=='prepare' and (OUT/'raw.csv').exists():raise SystemExit('Existing benchmark is immutable. Archive output before a fresh run.')
 (OUT/'run_logs').mkdir(exist_ok=True,parents=True);tic=time.perf_counter();r0=resource.getrusage(resource.RUSAGE_CHILDREN)
 with (OUT/'run_logs'/f'{stage}.log').open('w') as log:p=subprocess.run([sys.executable]+commands[stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/pbgia-mpl'))
 r1=resource.getrusage(resource.RUSAGE_CHILDREN);entry=dict(stage=stage,wall_seconds=time.perf_counter()-tic,CPU_seconds=r1.ru_utime+r1.ru_stime-r0.ru_utime-r0.ru_stime,cumulative_child_peak_rss_kb=r1.ru_maxrss,returncode=p.returncode);path=OUT/'stage_timings.json';a=json.load(open(path)) if path.exists() else [];a.append(entry);path.write_text(json.dumps(a,indent=2));print(entry,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
 if stage=='tests':(OUT/'smoke_pass.json').write_text(json.dumps(dict(passed=True,log='run_logs/tests.log')))
if __name__=='__main__':
 p=argparse.ArgumentParser();names=['prepare','tests','run','diagnostics','report','paired','audit']
 for n in names:p.add_argument('--'+n,action='store_true')
 p.add_argument('--all',action='store_true');a=p.parse_args()
 for n in names:
  if a.all or getattr(a,n):run(n)
