import sys,numpy as np,pandas as pd
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code/revision2','/home/claude/work/manuscript_v2/code']
from readout import boundary
from real_image_examples import generate_field
from skimage.measure import perimeter_crofton
rng=np.random.default_rng(3); rows=[]; n=96; R=12
for g in np.arange(0.25,8.01,0.25):
    for rep in range(6):
        t=rng.uniform(0,np.pi); sh=rng.uniform(-.5,.5,2); y,x=np.mgrid[:n,:n]+0.0; y-=(n-1)/2+sh[0]; x-=(n-1)/2+sh[1]
        u=np.cos(t)*x+np.sin(t)*y; v=-np.sin(t)*x+np.cos(t)*y; d=R+g/2
        m=(np.hypot(u-d,v)<=R)|(np.hypot(u+d,v)<=R); P=4*np.pi*R
        s=generate_field(m,30)[2].astype(float)
        mid=np.abs(u)<=max(g/2,0.5); band=mid&(np.abs(v)<2)
        rows.append(dict(gap=g,rep=rep,e_pbgia=100*(boundary(s)/P-1),e_crofton=100*(perimeter_crofton(m,4)/P-1),
                         s_gap=float(s[band].max()) if band.any() else np.nan, connected=int(__import__('scipy.ndimage',fromlist=['label']).label(m)[1]==1)))
df=pd.DataFrame(rows); df.to_csv('data/bench_gap.csv',index=False)
print(df.groupby('gap')[['e_pbgia','e_crofton','s_gap','connected']].mean().round(2).to_string())
