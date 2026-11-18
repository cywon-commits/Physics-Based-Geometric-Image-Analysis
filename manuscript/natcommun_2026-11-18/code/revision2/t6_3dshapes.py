import sys,numpy as np,csv,scipy.ndimage as ndi
sys.path.insert(0,'/home/claude/repo/Physics-Based-Geometric-Image-Analysis/manuscript/integrated_2026-11-17/code')
import new_quantitative_studies as nq
from skimage.measure import euler_number, marching_cubes, mesh_surface_area, label
from common import tv_per
M3=1.0160; rows=[]
for kind in ['sphere','capsule','torus','pair','fibers','porous']:
    mask,sd,area=nq.shape3(68,kind)
    n=nq.generate3(mask,30); s=n[...,3]
    gen=s>0
    deg=nq.degree(n) if kind=='porous' else None
    vv,ff,_,_=marching_cubes(mask.astype(np.float32),0.5); amc=mesh_surface_area(vv,ff)
    row=dict(kind=kind,area_true=area if area else np.nan,TV_area=tv_per(s)/M3,MC_raw=amc,
             chi_input=euler_number(mask,3),chi_generated=euler_number(gen,3),comp_input=label(mask,connectivity=3).max(),
             comp_generated=label(gen,connectivity=3).max(),changed_voxels=int(np.sum(gen!=mask)),
             degree=(deg['integer_degree'] if deg else np.nan))
    if area: row['TV_err_pct']=100*(row['TV_area']/area-1); row['MC_raw_err_pct']=100*(amc/area-1)
    rows.append(row); print(row,flush=True)
with open('data/t6_3dshapes.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
