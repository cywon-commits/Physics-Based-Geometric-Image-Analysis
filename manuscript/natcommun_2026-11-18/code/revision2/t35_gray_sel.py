"""TRAIN selection of the gray configuration across clean and noisy conditions (flip 1%, additive gray noise 0.05/0.1)."""
import sys,json,numpy as np,pandas as pd
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from t34_gray_kappa import est
from t32_gray import phi,S
from t33_gray_train import noisy
from gray import params2d,render,gen_gray
from t22_select import split_shapes
cf=[c for c in json.load(open('data/t34_gray_kappa.json')) if c['km']==0.8 and not c['meas']]
rows=[]
for fam,i,seed in split_shapes(2,'train'):
    rng=np.random.default_rng(seed); p=params2d(fam,rng); g0=render(p,S=S)
    for kind,p_ in (('flip',0.0),('flip',0.01),('gauss',0.05),('gauss',0.1)):
        g=np.clip(noisy(g0,rng,p_,kind),0,1); ss={h:gen_gray(phi(g,0),h=h) for h in (1.5,2.0)}
        for c in cf: rows.append(dict(family=fam,kind=kind,noise=p_,cfg=f"h{c['h']}_sk{c['sk']}",e=100*abs(est(ss[c['h']],c['c'],c['b'],c['sk'],c['km'])/p['P']-1)))
d=pd.DataFrame(rows); d=d[d.family!='near_touch']
t=d.groupby(['cfg','kind','noise','family']).e.mean().groupby(['cfg','kind','noise']).mean().unstack([1,2]); t['avg']=t.mean(1); print(t.round(3))
