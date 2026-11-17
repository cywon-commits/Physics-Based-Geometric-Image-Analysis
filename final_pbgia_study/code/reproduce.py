"""Run once in a NEW directory. Existing published outputs are never overwritten."""
from pathlib import Path
import argparse,os,shutil,subprocess,sys,time,json,resource

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workdir',required=True);args=parser.parse_args();source=Path(__file__).resolve().parents[1];target=Path(args.workdir).resolve()
 if target.exists():raise SystemExit('Choose a new workdir; refusing to overwrite an existing run.')
 target.mkdir(parents=True)
 for folder in ['code','vendor','tests','reference']:shutil.copytree(source/folder,target/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for p in source.iterdir():
  if p.is_file() and p.suffix in ['.md','.csv','.txt'] and p.name!='SHA256SUMS.txt':shutil.copy2(p,target/p.name)
 logs=target/'output/run_logs';logs.mkdir(parents=True);env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(logs/'mpl_cache'),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1');records=[]
 commands=[('preflight',['code/preflight.py']),('smoke',['-m','pytest','-q','tests','-k','not freeze_manifest_and_heldout_once_audit']),('train_validate',['code/geometry_study.py','validate']),('frozen_test',['code/geometry_study.py','test']),('physics',['code/physics.py']),('interactions',['code/interactions.py']),('spin_topology',['code/topology_study.py']),('projections',['code/projections_study.py']),('pixel_information',['code/pixel_information.py']),('controlled_benchmarks',['code/stability_benchmark.py']),('analysis',['code/analyze.py']),('final_tests',['-m','pytest','-q','tests'])]
 for stage,cmd in commands:
  print(stage,flush=True);tic=time.perf_counter()
  with open(logs/(stage+'.log'),'w') as log:subprocess.run([sys.executable]+cmd,cwd=target,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  records.append(dict(stage=stage,wall_seconds=time.perf_counter()-tic,cumulative_child_peak_rss_MiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024));(target/'output/reproduction_stage_times.json').write_text(json.dumps(records,indent=2))
 # This rerun is numerical verification, never another opportunity to tune a model.
 import numpy as np,pandas as pd
 comparisons=[]
 for file,columns in [('test_raw.csv',['I_E','I_TV','I_s0','normal_cos','energy_axis_error','spatial_axis_error']),('pixel_pairs.csv',['estimate','truth_low','truth_high']),('spin_topology.csv',['degree','interpolation_norm_lower_bound']),('projection_summary.csv',['defined_fraction','mean_purity','stable_candidates'])]:
  a=pd.read_csv(source/'output'/file);b=pd.read_csv(target/'output'/file);assert len(a)==len(b)
  for col in columns:
   error=float(np.nanmax(abs(a[col].to_numpy()-b[col].to_numpy())));assert np.allclose(a[col],b[col],rtol=1e-7,atol=1e-8,equal_nan=True),(file,col,error);comparisons.append(dict(file=file,column=col,max_absolute_difference=error))
 za=json.load(open(source/'output/model_freeze.json'))['selection'];zb=json.load(open(target/'output/model_freeze.json'))['selection'];assert za==zb
 (source/'output/clean_reproduction_verification.json').write_text(json.dumps(dict(verification_only=True,no_new_candidate=True,workdir=str(target),stage_times=records,comparisons=comparisons,selection_exact_match=True),indent=2));print('VERIFIED numerical reproduction; no additional model selection',flush=True)

if __name__=='__main__':main()
