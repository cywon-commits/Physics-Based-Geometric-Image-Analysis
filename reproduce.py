"""Run stages sequentially, saving stdout/stderr. Use an isolated venv."""
import argparse,os,sys,subprocess,json,hashlib,platform,importlib.metadata as metadata
from pathlib import Path
root=Path(__file__).resolve().parent;out=root/'results';out.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--stage',default='all',choices=['all','test','supplied','p0','supplemental','tilted','benchmark','finalize','width','topology','calibration','check','report']);a=p.parse_args()
json.dump({'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count(),'packages':{name:metadata.version(name) for name in ['numpy','scipy','pandas','matplotlib','scikit-image','pytest','tabulate']},'seed':61116,'spacing':1,'primary_tolerance':2e-6,'primary_max_steps':1200,'long_steps':2000,'boundary':'periodic; negative collar cap for degree only','initialization':'mask EDT only','zero_H':'retain previous unit vector'},open(out/'environment.json','w'),indent=2)
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/pbgia-mpl',XDG_CACHE_HOME='/tmp/pbgia-cache')
stages={
 'test':['-m','pytest','-q',str(root/'tests')],
 'supplied':[str(root/'code/pbgia_dynamic_normal_test.py'),'--out',str(out/'supplied_replication.csv'),'--steps','600','--N','56'],
 'p0':[str(root/'code/run_research.py'),'p0'],
 'supplemental':[str(root/'code/supplemental_validation.py')],
 'tilted':[str(root/'code/tilted_validation.py')],
 'benchmark':[str(root/'code/benchmark_dynamic.py')],
 'finalize':[str(root/'code/finalize_results.py')],
 'width':[str(root/'code/width_audit.py')],
 'topology':[str(root/'code/topology_audit.py')],
 'calibration':[str(root/'code/calibration_audit.py')],
 'check':[str(root/'code/artifact_checks.py')],
 'report':[str(root/'code/write_report.py')],
}
for stage,args in stages.items():
 if a.stage not in ['all',stage]:continue
 print('Running',stage,flush=True)
 with open(out/(stage+'_reproduce.log'),'w') as log:subprocess.run([sys.executable]+args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Complete:',out)
