import sys,json,numpy as np
sys.path[:0]=['/home/claude/work']
import t23_aniso as A
from t22_select import gen,split_shapes,sweep
from bench_shapes import make2d
for h in (2.0,1.5):
    A.NH=6; A.MEAS=False; c,_=A.planar_data(2,1.0,h,0); b=A.fit_delta(2,sweep(2,1.0,h,30),0,c)
    for p in (0.0,0.002,0.01):
        errs={}
        for fam,i,seed in split_shapes(2,'train'):
            rng=np.random.default_rng(seed); m,T,size=make2d(fam,rng); 
            if p>0: m=m^(np.random.default_rng(seed+7).random(m.shape)<p)
            errs.setdefault(fam,[]).append(100*(A.estimate(2,gen(2,m,1.0,h),0,c,b)/T-1))
        print(h,p,round(np.mean([np.mean(np.abs(v)) for f,v in errs.items() if f!='near_touch']),3),flush=True)
    json.dump(dict(c=c,b=b,h=h),open(f'data/t25_law_h{h}.json','w'))
