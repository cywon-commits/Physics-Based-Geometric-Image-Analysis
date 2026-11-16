"""Explicit stage runner. Never trains or scores test outside requested budget."""
import argparse,subprocess,time,json,os,sys,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output'
def run(stage):
 commands={'smoke':[sys.executable,'code/study.py','--smoke'],'train':[sys.executable,'code/study.py','--train'],'validate':[sys.executable,'code/evaluate.py','--validate'],'test':[sys.executable,'code/evaluate.py','--test'],'diagnostics':[sys.executable,'code/diagnostics.py'],'report':[sys.executable,'code/report.py'],'tests':[sys.executable,'-m','pytest','tests','-q']}
 OUT.mkdir(exist_ok=True);(OUT/'run_logs').mkdir(exist_ok=True);t=time.perf_counter();r0=resource.getrusage(resource.RUSAGE_CHILDREN)
 with (OUT/'run_logs'/f'{stage}.log').open('w') as log:p=subprocess.run(commands[stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/pbgia-mpl'))
 r1=resource.getrusage(resource.RUSAGE_CHILDREN);entry=dict(stage=stage,wall_seconds=time.perf_counter()-t,child_CPU_seconds=r1.ru_utime+r1.ru_stime-r0.ru_utime-r0.ru_stime,child_peak_rss_kb=r1.ru_maxrss,returncode=p.returncode);path=OUT/'stage_timings.json';ledger=json.load(open(path)) if path.exists() else [];ledger.append(entry);path.write_text(json.dumps(ledger,indent=2));print(entry,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ['smoke','train','validate','test','diagnostics','report','tests']:p.add_argument('--'+name,action='store_true')
 p.add_argument('--all',action='store_true');a=p.parse_args()
 for name in ['smoke','tests','train','validate','test','diagnostics','report']:
  if a.all or getattr(a,name):run(name)
