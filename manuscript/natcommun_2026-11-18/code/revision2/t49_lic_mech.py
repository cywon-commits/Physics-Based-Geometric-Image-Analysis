"""Why does the five-cut stack isolate the LIC disclinations at smaller sigma than the continuous source?
Tests: (a) more cuts (K=9,16,32 quantiles, tensors pooled); (b) filtering the input levels (t48 filtstack) already failed;
(c) continuous source with unit-length transverse vector (weight removed)."""
import sys,numpy as np,pandas as pd,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code','/home/claude/work/manuscript_v2/code/revision2']
from dirscale import qfield,defect_list,raw_tensor
from gray import gen_gray
from real_image_examples import generate_field
img=np.load('/home/claude/work/manuscript_v2/data/revision2/t15_lic_img.npy')
def cuts(K): return raw_tensor(img,thresholds=tuple((np.arange(K)+.5)/K*0.7+0.15))
def cont_unit():
    sm=ndi.gaussian_filter(img,.8); lo,hi=np.quantile(sm,(.02,.98)); g=np.clip((sm-lo)/(hi-lo),0,1)
    vr,vc,s=gen_gray(2*g-1,h=2.0,full=True); n=np.hypot(vr,vc)+1e-12; return np.stack([(vr/n)**2,(vc/n)**2,vr*vc/n**2])
def cont_h(h):
    sm=ndi.gaussian_filter(img,.8); lo,hi=np.quantile(sm,(.02,.98)); g=np.clip((sm-lo)/(hi-lo),0,1)
    vr,vc,s=gen_gray(2*g-1,h=h,full=True); return np.stack([vr*vr,vc*vc,vr*vc])
R={'cuts5':cuts(5),'cuts9':cuts(9),'cuts16':cuts(16),'cuts32':cuts(32),'cont_unit':cont_unit(),'cont_h1':cont_h(1.0),'cont_h4':cont_h(4.0),'cont_h2':cont_h(2.0)}
rows=[]
for s in [2,3,4,6,8,10]:
    for m,Rm in R.items(): rows.append(dict(method=m,sigma=s,n=len(defect_list(qfield(Rm,s)[0],margin=24))))
print(pd.DataFrame(rows).pivot(index='sigma',columns='method',values='n').to_string())
