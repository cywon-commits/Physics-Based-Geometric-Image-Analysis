"""Stadium size study re-scored with the final readout (same masks as new_quantitative_studies.plot3_stadium)."""
import sys,csv,numpy as np
sys.path.insert(0,'..'); sys.path.insert(0,'.')
import new_quantitative_studies as nq
from readout import boundary
from skimage.measure import perimeter_crofton
rows=[]
for r in [9,11,13,15,17,19,21,23,25]:
    for ang in (0,27,54):
        mask,_=nq.stadium(n=160,R=r,L=2*r,theta=ang); vr,vc,s=nq.generate_field(mask,nsteps=30)
        P=2*np.pi*r+4*r
        rows.append(dict(R=r,rotation_deg=ang,P_true=P,B_curv=boundary(s),B_planar=boundary(s,False),crofton=float(perimeter_crofton(mask,4))))
with open('../../data/revision2/t13_stadium.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
import pandas as pd; d=pd.DataFrame(rows)
for m in ('B_curv','B_planar','crofton'): d['e_'+m]=100*(d[m]/d.P_true-1)
print(d[['e_B_curv','e_B_planar','e_crofton']].abs().mean().round(3).to_dict(), 'R>9:',d[d.R>9][['e_B_curv','e_B_planar','e_crofton']].abs().mean().round(3).to_dict())
