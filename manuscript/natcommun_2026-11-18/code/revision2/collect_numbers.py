import csv,numpy as np,json
rd=lambda f:list(csv.DictReader(open(f))); M2=1.0189; out={}
st=[x for x in rd('/home/claude/repo/Physics-Based-Geometric-Image-Analysis/manuscript/integrated_2026-11-17/data/stadium_scaling.csv') if x['radius_pix']!='9']
e=np.array([float(x['TV_raw'])/M2/float(x['P_true'])-1 for x in st])*100
out['stadium_n']=len(st); out['stadium_refree_MARE']=np.mean(abs(e)); out['stadium_refree_max']=abs(e).max()
out['stadium_crofton_MARE']=np.mean([abs(float(x['crofton'])/float(x['P_true'])-1)*100 for x in st])
out['stadium_refcal_MARE']=np.mean([abs(float(x['relative_TV_pct'])) for x in st])
d=[x for x in rd('data/t3_disk.csv') if x['N']=='30']
for lo,hi in ((8,99),(4,7)):
    q=[x for x in d if lo<=float(x['R'])<=hi]
    out[f'disk_{lo}_{hi}_TV']=np.mean([abs(float(x['TV'])/M2/float(x['P_true'])-1)*100 for x in q])
    out[f'disk_{lo}_{hi}_crof']=np.mean([abs(float(x['crofton'])/float(x['P_true'])-1)*100 for x in q])
out['disk_max_abs_meanu_R>=8']=max(abs(np.mean([float(x['mean_u']) for x in d if float(x['R'])==R])) for R in {float(x['R']) for x in d} if R>=8)
out['disk_min_dice']=min(float(x['dice']) for x in d)
d0=[x for x in rd('data/t3_disk.csv') if x['N']=='0' and float(x['R'])>=8]
out['disk_N0_TV']=np.mean([abs(float(x['TV'])/M2/float(x['P_true'])-1)*100 for x in d0])
n=rd('data/t4_noise.csv')
for p in ('0.0','0.002','0.01','0.05'):
    for k,m in (('TV_N30',M2),('TV_N10',M2),('TV_N0',M2),('crofton',1),('crofton_median3',1)):
        out[f'noise_{p}_{k}']=np.mean([abs(float(x[k])/m/float(x['P_true'])-1)*100 for x in n if x['noise']==p])
    out[f'noise_{p}_dice30']=np.mean([float(x['dice_N30']) for x in n if x['noise']==p])
s=rd('data/t6_3dshapes.csv')
for x in s: out['3d_'+x['kind']]={k:x[k] for k in ('TV_err_pct','MC_raw_err_pct','chi_input','chi_generated','degree','changed_voxels')}
out['3d_TV_MARE_4']=np.mean([abs(float(x['TV_err_pct'])) for x in s if x['kind'] in ('sphere','capsule','torus','pair')])
sp=rd('data/t5_spheres.csv'); out['sphere_u2']=np.mean([float(x['mean_u2']) for x in sp if float(x['R'])>=8])
for R in ('6','10','20'): out['sphere_ratio_'+R]=np.mean([float(x['TV'])/float(x['A_true'])/1.0160 for x in sp if x['R']==R])
json.dump(out,open('data/numbers.json','w'),indent=1,default=float); print(json.dumps(out,indent=0,default=float))
