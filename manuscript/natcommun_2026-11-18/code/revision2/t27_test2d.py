"""Single test evaluation of the selected 2D readout (h=1.5, direction+curvature metrication) on the identical noisy masks of bench_run."""
import sys,json,numpy as np,pandas as pd
sys.path[:0]=['/home/claude/work']
import t23_aniso as A
from t22_select import gen,split_shapes
from bench_shapes import make2d
L=json.load(open('data/t25_law_h1.5.json')); A.NH=6
rows=[]
for fam,i,seed in split_shapes(2,'test'):
    rng=np.random.default_rng(seed); m0,T,size=make2d(fam,rng)
    for p in (0.0,0.002,0.01):
        m=m0^(rng.random(m0.shape)<p) if p>0 else m0
        rows.append(dict(family=fam,inst=i,noise=p,truth=T,PBGIA_opt=A.estimate(2,gen(2,m,1.0,1.5),0,L['c'],L['b'])))
df=pd.DataFrame(rows); df.to_csv('data/bench_test2d_opt.csv',index=False)
df['e']=100*(df.PBGIA_opt/df.truth-1).abs()
print(df[df.family!='near_touch'].groupby(['noise','family']).e.mean().groupby('noise').mean().round(3))
print(df[df.noise==0].groupby('family').e.mean().round(2))
