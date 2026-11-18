import sys,numpy as np
sys.path.insert(0,'/home/claude/repo/Physics-Based-Geometric-Image-Analysis/manuscript/integrated_2026-11-17/code')
import new_quantitative_studies as nq
from skimage.measure import euler_number,label
mask,_,_=nq.shape3(68,'porous'); n=nq.generate3(mask,30); g=n[...,3]>0
for name,m in (('input',mask),('generated',g)):
    print(name,'chi6',euler_number(m,1),'chi26',euler_number(m,3),'comp6',label(m,connectivity=1).max(),'comp26',label(m,connectivity=3).max(),'vox',m.sum())
lab,k=label(mask,connectivity=1,return_num=True); sizes=np.bincount(lab.ravel())[1:]; print('input 6-comp sizes',sorted(sizes)[:8])
print('removed',np.sum(mask&~g),'added',np.sum(g&~mask))
