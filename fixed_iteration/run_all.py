"""One-command staged reproduction. Test scoring refuses to overwrite existing results."""
import os,sys,subprocess,json,platform,hashlib,importlib.metadata as md,time,argparse,shutil
from pathlib import Path
root=Path(__file__).resolve().parent;out=root/'output'
p=argparse.ArgumentParser();p.add_argument('--fresh',action='store_true');a=p.parse_args()
if a.fresh and out.exists():
 archive=root/'archived_outputs';archive.mkdir(exist_ok=True);shutil.move(str(out),str(archive/time.strftime('%Y%m%dT%H%M%S')))
if (out/'test_results.csv').exists():raise SystemExit('Test already scored. Use --fresh to archive this run and reproduce from scratch.')
out.mkdir(exist_ok=True)
environment=dict(python=sys.version,platform=platform.platform(),cpu_count=os.cpu_count(),GPU_used=False,dtype='float32',OPENBLAS_NUM_THREADS='1',packages={n:md.version(n) for n in ['numpy','scipy','pandas','matplotlib','scikit-image','pytest','tabulate']})
(out/'environment.json').write_text(json.dumps(environment,indent=2))
try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
except Exception:commit=None
(out/'code_version.json').write_text(json.dumps(dict(git_commit=commit,source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(list((root/'code').glob('*.py'))+[root/'reference/pbgia_s3_integer_degree.py'])}),indent=2))
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/pbgia-mpl',XDG_CACHE_HOME='/tmp/pbgia-cache')
steps=[('tests',['-m','pytest','-q','tests']),('smoke2',['code/fixed_iteration_checkpoints.py','--dim','2','--size','40','--out','output/smoke_2d']),('smoke3',['code/fixed_iteration_checkpoints.py','--dim','3','--size','24','--out','output/smoke_3d']),('train',['code/run_fixed_study.py','train']),('test',['code/run_fixed_study.py','test']),('transfer',['code/run_fixed_study.py','transfer']),('additional',['code/additional_checks.py']),('artifact_checks',['code/check_artifacts.py']),('report',['code/report_fixed.py'])]
timings=[]
for name,args in steps:
 print('Running',name,flush=True);t=time.perf_counter()
 with (out/(name+'.log')).open('w') as log:subprocess.run([sys.executable]+args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 timings.append(dict(stage=name,seconds=time.perf_counter()-t));(out/'stage_timings.json').write_text(json.dumps(timings,indent=2))
print('Done:',out/'REPORT_KO.md')
