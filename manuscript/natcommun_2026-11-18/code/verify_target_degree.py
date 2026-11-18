"""Independent regular-target stability check on freshly generated 3D fields."""
from pathlib import Path
import csv, numpy as np
from new_quantitative_studies import shape3,generate3
from pbgia_s3_integer_degree import degree
out=Path(__file__).resolve().parents[1]
targets=[(.37139,-.52177,.64829,.40981),(.1729,.7312,-.4401,.4917),(-.6333,.2184,.3842,-.6361)]
rows=[]
for shape in ['sphere','capsule','torus','pair','fibers']:
    mask,sd,area=shape3(68,shape)
    n=generate3(mask)
    for i,target in enumerate(targets):
        res=degree(n,target=target)
        rows.append(dict(shape=shape,target_id=i,degree=res['integer_degree'],hits=res['number_of_preimages'],border_spread=res['boundary_spread'],min_positive_coeff=res['min_barycentric_positive_coefficient']))
    print(shape,[r['degree'] for r in rows if r['shape']==shape],flush=True)
with open(out/'data'/'3d_target_stability.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
