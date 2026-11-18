"""Fig. 8: cell nuclei (BBBC038/DSB2018 StarDist subset, CC0): counts against expert annotation."""
import sys,glob,json,zlib,numpy as np,pandas as pd,scipy.ndimage as ndi
sys.path[:0]=['.','/home/claude/work','/home/claude/work/manuscript_v2/code/revision2','/home/claude/work/manuscript_v2/code']
from style import *
import matplotlib as mpl
import t60_cells as T
from gray import gen_gray
from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries
D='/home/claude/work/data'; OUT='../figures'
R=pd.read_csv(f'{D}/t60_cells.csv'); cfg=json.load(open(f'{D}/t60_cells_cfg.json'))
M=[('otsu','Otsu','#b9b8b0','v'),('tuned',f'Otsu, tuned ($\\sigma={cfg["sig"]}$, $\\geq{cfg["amin"]}$ px)','#eb6834','^'),('pbb','PBGIA, binary source','#8fb8e8','s'),('pbc','PBGIA, continuous source','#0b3d78','o')]
f=glob.glob(T.ROOT+'/test/images/9520aff4*.tif')[0]; I,L,ngt,merged=T.base(f); lo,hi,t=T.levels(I)
r0,r1,c0,c1=150,410,180,500; Iv=np.clip((I-lo)/(hi-lo),-0.1,1.3)
m0=I>threshold_otsu(I); P=8; g=np.clip((np.pad(I,P,constant_values=lo)-lo)/(hi-lo),0,1); s=gen_gray(2*g-1,h=1.5)[P:-P,P:-P]
fig=plt.figure(figsize=(FULL,4.5),layout='constrained'); gs=fig.add_gridspec(2,3,height_ratios=[1,1])
def show(ax,img,title,overlay=None,cmap='gray',**kw):
    ax.imshow(img[r0:r1,c0:c1],cmap=cmap,**kw); noaxis(ax); ax.set_title(title,fontsize=7.2)
a=fig.add_subplot(gs[0,0]); show(a,Iv,f'image and annotation ({ngt} nuclei)',vmin=-0.1,vmax=1.3)
bd=find_boundaries(L,mode='inner')[r0:r1,c0:c1]; a.contour(bd,[0.5],colors=['#ffd166'],linewidths=0.5); imlabel(a,'a')
b=fig.add_subplot(gs[0,1]); lab,n=ndi.label(m0); show(b,m0.astype(float),f'Otsu: {n} objects',vmin=0,vmax=1); imlabel(b,'b')
c=fig.add_subplot(gs[0,2]); nP=ndi.label(s>0)[1]; show(c,Iv,f'PBGIA, $s=0$: {nP} objects',vmin=-0.1,vmax=1.3); c.contour(s[r0:r1,c0:c1],[0],colors=['#2a78d6'],linewidths=0.7); imlabel(c,'c')
d=fig.add_subplot(gs[1,0]); R0=R[R.noise==0]
for k,lb,col,mk in M:
    d.plot(R0.ngt,R0[k+'_n'],mk,color=col,ms=3,mec='white',mew=0.3,label=lb)
d.plot([5,400],[5,400],color=C['theory'],lw=0.8); d.set(xscale='log',yscale='log',xlabel='annotated nuclei',ylabel='objects counted',xlim=(6,400),ylim=(4,2000)); d.minorticks_off(); label(d,'d')
e=fig.add_subplot(gs[1,1])
for k,lb,col,mk in M:
    q=R.groupby('noise').apply(lambda x: 100*np.mean(np.abs(x[k+'_n']/x.ngt-1)),include_groups=False); e.plot(q.index,q.values,mk+'-',color=col,ms=3.5,label=lb)
e.set(yscale='log',xlabel='added noise (s.d. / contrast)',ylabel='count error vs annotation (%)',xticks=[0,0.1,0.2],ylim=(8,3e4)); e.minorticks_off(); label(e,'e')
h=fig.add_subplot(gs[1,2])
h.plot(R0.merged,R0.pbc_n,'o',color='#0b3d78',ms=3,mec='white',mew=0.3,label='PBGIA, continuous')
h.plot(R0.merged,R0.tuned_n,'^',color='#eb6834',ms=3,mec='white',mew=0.3,label='Otsu, tuned')
h.plot([5,300],[5,300],color=C['theory'],lw=0.8); h.set(xscale='log',yscale='log',xlabel='annotated nuclei, gaps $<2$ px merged',ylabel='objects counted',xlim=(6,300),ylim=(6,300)); h.minorticks_off(); label(h,'f')
fig.legend(*d.get_legend_handles_labels(),loc='lower center',ncol=4,bbox_to_anchor=(0.5,-0.06),fontsize=6.5)
save(fig,'Fig10_cells',OUT); print('ok', n, nP)
