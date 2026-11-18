import sys,json,numpy as np,pandas as pd
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
import t23_aniso as A
from t22_select import gen
from skimage.measure import perimeter_crofton
L=json.load(open('data/t25_law_h1.5.json')); A.NH=6
# paired bootstrap vs comparators (2D)
a=pd.read_csv('data/bench_raw.csv'); a=a[(a.dim==2)&(a.split=='test')]; b=pd.read_csv('data/bench_test2d_opt.csv')
x=a.merge(b[['family','inst','noise','PBGIA_opt']],on=['family','inst','noise']); C=json.load(open('data/bench_constants.json'))['2']
rng=np.random.default_rng(1); out=[]
for nz in (0,0.002,0.01):
    t=x[(x.noise==nz)&(x.family!='near_touch')]; ep=(t.PBGIA_opt/t.truth-1).abs()*100
    for name,col,k in (('crofton_cal','crofton',C['constants']['crofton_cal']),('gauss_cal',C['gauss_sigma'],C['constants']['gauss_cal']),('median_cal','median_crofton',C['constants']['median_crofton_cal']),('PBGIA_h2','PBGIA',1.0)):
        ec=(t[col]*k/t.truth-1).abs()*100; diff=(ep-ec).values; fam=t.family.values; fams=np.unique(fam)
        bs=[np.mean([np.mean(rng.choice(diff[fam==f],size=(fam==f).sum())) for f in fams]) for _ in range(4000)]
        est=np.mean([diff[fam==f].mean() for f in fams]); out.append(dict(noise=nz,comparator=name,diff=round(est,3),lo=round(np.percentile(bs,2.5),3),hi=round(np.percentile(bs,97.5),3)))
r=pd.DataFrame(out); r.to_csv('data/bench_paired_opt2d.csv',index=False); print(r.to_string())
# gap sweep with the selected readout
g=pd.read_csv('data/bench_gap.csv'); rng=np.random.default_rng(3); rows=[]; n=96; R=12
for gap in np.arange(0.25,8.01,0.25):
    for rep in range(6):
        t=rng.uniform(0,np.pi); sh=rng.uniform(-.5,.5,2); yy,xx=np.mgrid[:n,:n]+0.0; yy-=(n-1)/2+sh[0]; xx-=(n-1)/2+sh[1]
        u=np.cos(t)*xx+np.sin(t)*yy; v=-np.sin(t)*xx+np.cos(t)*yy; d=R+gap/2
        m=(np.hypot(u-d,v)<=R)|(np.hypot(u+d,v)<=R); P=4*np.pi*R
        rows.append(dict(gap=gap,rep=rep,e_opt=100*(A.estimate(2,gen(2,m,1.0,1.5),0,L['c'],L['b'])/P-1)))
gg=pd.DataFrame(rows); g=g.merge(gg,on=['gap','rep']); g.to_csv('data/bench_gap.csv',index=False)
print(g.groupby('gap')[['e_pbgia','e_opt','e_crofton']].mean().round(2).head(12).to_string())
