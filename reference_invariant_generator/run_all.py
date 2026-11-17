from pathlib import Path
import argparse,subprocess,sys,time,json,os,resource
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output'
def run(stage):
 commands={'smoke':['code/preflight.py'],'tests':['-m','pytest','tests','-q'],'train2d':['code/study.py'],'validate2d':['code/evaluation.py','validate2d'],'freeze2d':['code/evaluation.py','freeze2d'],'test2d':['code/evaluation.py','test2d'],'report2d':['code/report.py'],'diagnostics':['code/diagnostics.py'],'smoke3d':['code/three_d_smoke.py']}
 if stage in ['train3d','validate3d','freeze3d','test3d','report3d']:raise SystemExit('3D is verified partial only. See output/status_3d.json. No3D trained/test numbers exist; allocate a new explicit3D budget and fresh manifest first.')
 OUT.mkdir(exist_ok=True);(OUT/'run_logs').mkdir(exist_ok=True);tic=time.perf_counter();r0=resource.getrusage(resource.RUSAGE_CHILDREN)
 with (OUT/'run_logs'/f'{stage}.log').open('w') as log:result=subprocess.run([sys.executable]+commands[stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/pbgia-mpl'))
 r1=resource.getrusage(resource.RUSAGE_CHILDREN);entry=dict(stage=stage,wall_seconds=time.perf_counter()-tic,CPU_seconds=r1.ru_utime+r1.ru_stime-r0.ru_utime-r0.ru_stime,cumulative_child_peak_rss_kb=r1.ru_maxrss,returncode=result.returncode);p=OUT/'stage_timings.json';ledger=json.load(open(p)) if p.exists() else [];ledger.append(entry);p.write_text(json.dumps(ledger,indent=2));print(entry,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':
 stages=['smoke','tests','train2d','validate2d','freeze2d','test2d','diagnostics','report2d','smoke3d','train3d','validate3d','freeze3d','test3d','report3d'];p=argparse.ArgumentParser()
 for s in stages:p.add_argument('--'+s,action='store_true')
 a=p.parse_args()
 for s in stages:
  if getattr(a,s):run(s)
