import sys,json,numpy as np,pandas as pd
sys.path[:0]=['.']
from style import *
B='../data/revision2/benchmark'; OUT='../figures'
d=pd.read_csv(f'{B}/bench_raw.csv'); o=pd.read_csv(f'{B}/bench_test2d_opt.csv'); o['dim']=2; o['split']='test'
d=d.merge(o[['dim','split','family','inst','noise','PBGIA_opt']],on=['dim','split','family','inst','noise'],how='left'); CST=json.load(open(f'{B}/bench_constants.json')); gap=pd.read_csv(f'{B}/bench_gap.csv')
def methods(dim):
    c=CST[str(dim)]['constants']; g=CST[str(dim)]['gauss_sigma']
    L=['PBGIA, no calibration','Crofton (2D) / marching cubes (3D), calibrated','Gaussian + marching squares / cubes, calibrated','median filter + Crofton / marching cubes, calibrated']
    if dim==2: return [(L[0],'PBGIA_opt',1,'pbgia','o',True),(L[1],'crofton',c['crofton_cal'],'conv','^',True),(L[2],g,c['gauss_cal'],'smooth','s',True),(L[3],'median_crofton',c['median_crofton_cal'],'smooth','D',False)]
    return [(L[0],'PBGIA',1,'pbgia','o',True),(L[1],'mc',c['mc_cal'],'conv','^',True),(L[2],g,c['gauss_cal'],'smooth','s',True),(L[3],'median_mc',c['median_mc_cal'],'smooth','D',False)]
fams={2:['disk','ellipse','stadium','ring','flower','pair','near_touch'],3:['sphere','ellipsoid','capsule','torus','bumpy','pair','near_touch']}
fig=plt.figure(figsize=(FULL,6.7),layout='constrained'); gs=fig.add_gridspec(3,2,width_ratios=[1.7,1],height_ratios=[1,1,0.9])
for row,dim in enumerate((2,3)):
    a=fig.add_subplot(gs[row,0]); b=fig.add_subplot(gs[row,1]); te=d[(d.dim==dim)&(d.split=='test')]
    M=methods(dim)
    for i,(lb,col,k,c,mk,fill) in enumerate(M):
        t0=te[te.noise==0]; off=(i-1.5)*0.18
        for j,f in enumerate(fams[dim]):
            e=(t0[t0.family==f][col]*k/t0[t0.family==f].truth-1).abs()*100
            a.plot(np.full(len(e),j+off),e.clip(lower=1e-3),mk,color=C[c],ms=1.8,alpha=0.35,mew=0)
            a.plot(j+off,e.mean(),mk,color=C[c] if fill else 'white',mec=C[c],mew=0.9,ms=4.5,label=lb if (j==0 and dim==2) else None)
        mu=[]
        for nz in (0,0.002,0.01):
            q=te[(te.noise==nz)&(te.family!='near_touch')]; mu.append(np.mean([((q[q.family==f][col]*k/q[q.family==f].truth-1).abs()*100).mean() for f in fams[dim][:-1]]))
        b.plot([5e-4,0.002,0.01],mu,mk+'-',color=C[c],mfc=C[c] if fill else 'white',ms=4)
    a.set_xticks(range(7),[f.replace('near_touch','near touch') for f in fams[dim]],rotation=20,ha='right'); a.set_yscale('log'); a.set_ylim(3e-3,30)
    a.set_ylabel(('2D perimeter' if dim==2 else '3D surface area')+'\n|error| (%)'); a.grid(axis='x',visible=False)
    b.set(xscale='log',yscale='log',xlabel='flip rate',ylabel='MARE (%)'); b.set_xticks([5e-4,0.002,0.01],['0','0.002','0.01']); b.minorticks_off()
    label(a,'ac'[row]); label(b,'bd'[row])
h_,l_=fig.axes[0].get_legend_handles_labels(); fig.legend(h_,l_,loc='outside upper center',ncol=2,columnspacing=1.0,handletextpad=0.2)
e=fig.add_subplot(gs[2,:]); g=gap.groupby('gap').agg(p=('e_opt','mean'),ps=('e_opt','std'),c=('e_crofton','mean'),cs=('e_crofton','std'),conn=('connected','mean')).reset_index()
e.axvspan(0,g[g.conn>0].gap.max()+0.125,color=C['grid'],alpha=0.9,lw=0); e.text(0.05,-0.6,'raster\nmerges',fontsize=6.5,va='top')
e.errorbar(g.gap,g.p,yerr=g.ps,fmt='o-',color=C['pbgia'],ms=3,capsize=1.2,elinewidth=0.6,label='PBGIA')
e.errorbar(g.gap,g.c,yerr=g.cs,fmt='^-',color=C['conv'],ms=3,capsize=1.2,elinewidth=0.6,label='Crofton, raw')
e.axhline(0,color=C['theory'],lw=0.7); e.axvline(2.0,color=C['theory'],ls='--',lw=0.8); e.text(2.1,-6.5,'gap $\\approx 2a$:\nlevel sets merge below',fontsize=6.5)
e.set(xlabel='gap between two disks of radius 12 (pixels)',ylabel='perimeter error (%)',xlim=(0,8.2)); e.legend(loc='lower right'); label(e,'e')
save(fig,'Fig5_benchmark',OUT); print('ok')
