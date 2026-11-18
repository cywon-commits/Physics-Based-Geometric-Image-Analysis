"""Decompose error into geometry bias and pose (rasterization) spread; quantization-law fit var(P)=c*int kappa^2 ds; disk sweep vs info floor."""
import numpy as np,pandas as pd,json
d=pd.read_csv('data/t39_pose_all.csv'); M=['PBGIA_bin','gauss_bin','crofton_bin','PBGIA_gray','ms_gray']; out={}
for m in M: d['e_'+m]=100*(d[m]/d.truth-1)
g=d.groupby(['family','inst']).agg(truth=('truth','first'),k2=('k2','first'),**{f'{s}_{m}':(f'e_{m}',s) for m in M for s in ('mean','std')},
     **{f'mae_{m}':(f'e_{m}',lambda x:np.mean(np.abs(x))) for m in M})
fam=g.groupby('family').agg(**{f'{k}_{m}':(f'{k}_{m}',(lambda x:np.mean(np.abs(x))) if k=='mean' else 'mean') for m in M for k in ('mean','std','mae')})
print(fam.round(3).T.to_string())
for sub,gg in (('all',g),('no_flower',g.reset_index().query("family!='flower'"))):
    for m in M:
        sd=gg['std_'+m]/100*gg.truth; c=float(np.sum(sd**2*gg.k2)/np.sum(gg.k2**2)); r=float(np.corrcoef(sd**2,gg.k2)[0,1])
        mae=float(gg.groupby('family')['mae_'+m].mean().mean()) if 'family' in gg else float(gg['mae_'+m].mean())
        noise_only=float((0.798*gg['std_'+m]).groupby(gg['family'] if 'family' in gg else gg.index.get_level_values(0)).mean().mean())
        bias=float(gg['mean_'+m].abs().groupby(gg['family'] if 'family' in gg else gg.index.get_level_values(0)).mean().mean())
        out[f'{sub}:{m}']=dict(MARE=mae,pose_only_MARE=noise_only,mean_abs_bias=bias,c_k2=c,corr_k2=r)
r=pd.DataFrame(out).T; print(r.round(4).to_string()); r.to_csv('data/t41_floor_decomp.csv')
D=pd.read_csv('data/t39_disk_sweep.csv'); I=pd.read_csv('data/t31_infofloor.csv')
for m in M: D['e_'+m]=100*(D[m]/D.truth-1)
s=D.groupby('R').agg(**{f'std_{m}':(f'e_{m}','std') for m in M},**{f'bias_{m}':(f'e_{m}','mean') for m in M},**{f'rms_{m}':(f'e_{m}',lambda x:np.sqrt(np.mean(x**2))) for m in M})
s=s.join(I.groupby('R').rel_std.mean().rename('info_floor_std'),how='left'); print(s.round(3).to_string()); s.to_csv('data/t41_disk_sweep_summary.csv')
# power-law fits of pose std for R>=5
for m in ['PBGIA_bin','PBGIA_gray','ms_gray','gauss_bin','crofton_bin']:
    q=s[s.index>=5]; a,b=np.polyfit(np.log(q.index),np.log(q['std_'+m]),1); print(m,'std ~ R^%.2f, prefactor %.3f'%(a,np.exp(b)))
q=I.groupby('R').rel_std.mean(); a,b=np.polyfit(np.log(q.index),np.log(q.values),1); print('info floor ~R^%.2f pref %.3f ; theory 100/(2pi)=15.9'%(a,np.exp(b)))
