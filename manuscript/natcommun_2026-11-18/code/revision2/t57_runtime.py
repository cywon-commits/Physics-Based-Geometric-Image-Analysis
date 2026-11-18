"""Wall-clock time per image (single CPU thread, NumPy) for PBGIA vs conventional estimators."""
import os; os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,time,json,numpy as np,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code/revision2','/home/claude/work/manuscript_v2/code']
import t23_aniso as A
from t22_select import gen
from bench_shapes import make2d,make3d
from readout import boundary
from bench_run import ms_len,mc_area
from skimage.measure import perimeter_crofton
L2=json.load(open('data/t25_law_h1.5.json')); A.NH=6
def tm(f,rep):
    f(); t=time.perf_counter()
    for _ in range(rep): f()
    return 1e3*(time.perf_counter()-t)/rep
rng=np.random.default_rng(3); m2,_,_=make2d('flower',rng); m3,_,_=make3d('torus',rng)
out={'shape2':list(m2.shape),'shape3':list(m3.shape)}
for N in (0,5,10,30):
    out[f'2d_gen_N{N}']=tm(lambda:gen(2,m2,1.0,1.5,N),20)
    out[f'3d_gen_N{N}']=tm(lambda:gen(3,m3,1.0,2.0,N),3)
s2=gen(2,m2,1.0,1.5,30); s3=gen(3,m3,1.0,2.0,30)
out['2d_readout']=tm(lambda:A.estimate(2,s2,0,L2['c'],L2['b']),20); out['3d_readout']=tm(lambda:boundary(s3),3)
out['2d_crofton']=tm(lambda:perimeter_crofton(m2,4),50); out['2d_ms']=tm(lambda:ms_len(m2.astype(float)),50)
out['2d_gauss_ms']=tm(lambda:ms_len(ndi.gaussian_filter(m2.astype(float),0.75)),50)
out['2d_median_crofton']=tm(lambda:perimeter_crofton(ndi.median_filter(m2.astype(np.uint8),3).astype(bool),4),50)
out['3d_mc']=tm(lambda:mc_area(m3.astype(float)),5); out['3d_gauss_mc']=tm(lambda:mc_area(ndi.gaussian_filter(m3.astype(float),0.5)),5)
out['3d_median_mc']=tm(lambda:mc_area(ndi.median_filter(m3.astype(np.uint8),3).astype(float)),5)
json.dump(out,open('data/t57_runtime.json','w'),indent=1); print(json.dumps(out,indent=1))
