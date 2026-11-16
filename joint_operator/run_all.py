"""One-command reproduction: constrained joint study, freeze, then independent test."""
import argparse,os,sys,subprocess,time,json,platform,hashlib,shutil,resource
from pathlib import Path
import importlib.metadata as md
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--fresh',action='store_true');args=p.parse_args();out=root/'output'
if args.fresh and out.exists() and any(out.iterdir()):
 archive=root/'archived_runs';archive.mkdir(exist_ok=True);shutil.move(str(out),str(archive/time.strftime('%Y%m%dT%H%M%S')))
out.mkdir(exist_ok=True)
if (out/'test_results.csv').exists():raise SystemExit('Test already scored; use --fresh to archive this run and reproduce.')
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/pbgia-mpl',XDG_CACHE_HOME='/tmp/pbgia-cache')
info=dict(python=sys.version,platform=platform.platform(),cpu_count=os.cpu_count(),GPU_used=False,field_dtype='float32',feature_dtype='float64',OPENBLAS_NUM_THREADS=1,packages={name:md.version(name) for name in ['numpy','scipy','pandas','matplotlib','scikit-image','pytest','tabulate']})
try:info['CPU_model']=next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name'))
except Exception:pass
(out/'environment.json').write_text(json.dumps(info,indent=2))
try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
except Exception:commit=None
(out/'code_version.json').write_text(json.dumps(dict(git_commit=commit,source_sha256={str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted((root/'code').glob('*.py'))}),indent=2))
if not (out/'supplied_smoke_raw.csv').exists():shutil.copy2(root/'reference/supplied_smoke_raw.csv',out/'supplied_smoke_raw.csv')
steps=[('tests',['-m','pytest','-q','tests']),('smoke',['code/run_smoke.py']),('design',['code/design.py']),('stage_zero',['code/stage_zero.py']),('training',['code/train_joint.py']),('validation',['code/evaluate_joint.py','validate']),('test',['code/evaluate_joint.py','test']),('diagnostics',['code/diagnostics_joint.py']),('refined_validation_diagnostic',['code/refined_topology_diagnostic.py']),('artifact_checks',['code/check_joint_artifacts.py']),('report',['code/report_joint.py'])]
timings=[]
for name,argv in steps:
 print('START',name,flush=True);tic=time.perf_counter()
 with (out/(name+'.log')).open('w') as log:subprocess.run([sys.executable]+argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 timings.append(dict(stage=name,seconds=time.perf_counter()-tic));(out/'stage_timings.json').write_text(json.dumps(timings,indent=2));print('DONE',name,round(timings[-1]['seconds'],2),flush=True)
 info['peak_child_RSS_MiB']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024;(out/'environment.json').write_text(json.dumps(info,indent=2))
print('COMPLETE',out/'RESEARCH_REPORT_KO.md',flush=True)
