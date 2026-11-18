import sys,json,numpy as np
sys.path[:0]=['/home/claude/work']
from t22_select import gen,split_shapes,sweep,planar,fitlaw,law
from kern import tv
from readout import curvature
from bench_shapes import make3d
for h in (2.0,1.5):
    mb=planar(3,1.0,h,[0],30)[0]; b=fitlaw(3,sweep(3,1.0,h,30),0,mb)
    for p in (0.0,0.002,0.01):
        errs={}
        for fam,i,seed in split_shapes(3,'train'):
            rng=np.random.default_rng(seed); m,T,size=make3d(fam,rng)
            if p>0: m=m^(np.random.default_rng(seed+7).random(m.shape)<p)
            s=gen(3,m,1.0,h); errs.setdefault(fam,[]).append(100*(np.sum(tv(s,0)/law(np.abs(curvature(s)),3,mb,b))/T-1))
        print(h,p,round(np.mean([np.mean(np.abs(v)) for f,v in errs.items() if f!='near_touch']),3),flush=True)
    json.dump(dict(mbar=mb,coef=b,h=h),open(f'data/t26_law3_h{h}.json','w'))
