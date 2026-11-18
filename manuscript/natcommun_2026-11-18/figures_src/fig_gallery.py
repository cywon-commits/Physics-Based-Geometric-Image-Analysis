import sys,numpy as np
sys.path[:0]=['.','../code']
from style import *
from matplotlib.collections import LineCollection
import multithreshold_directors as mt
sys.path.insert(0,'/home/claude/work'); sys.path.insert(0,'../code/revision2')
from t47_cont_textures import raw_cont
import scipy.ndimage as ndi
def cont_field(img,sigma=8):
    rr,cc,rc=[ndi.gaussian_filter(x,sigma) for x in raw_cont(img)]; gap=np.hypot(cc-rr,2*rc)
    return {'tangent':((.5*np.degrees(np.arctan2(-2*rc,cc-rr)))%180+90)%180,'gap':gap,'coherence':gap/(rr+cc+1e-12)}
OUT='../figures'
cases=mt.construct_cases()
def stretch(img):
    lo,hi=np.percentile(img,[1,99]); return np.clip((img-lo)/max(hi-lo,1e-9),0,1)
def directors(ax,img,res,color,step=11,max_length=11,gate=.25,crop=None):
    H,W=img.shape; yy,xx=np.mgrid[step//2:H:step,step//2:W:step]
    a=res['tangent'][yy,xx]; coh=res['coherence'][yy,xx]; amp=res['gap'][yy,xx]
    mag=np.clip(amp/max(float(np.percentile(res['gap'],95)),1e-10),0,1); ok=(coh>=gate)&(mag>=.07)
    if crop is not None: ok&=(yy>=crop[0])&(yy<crop[1])&(xx>=crop[0])&(xx<crop[1])
    L=1.5+max_length*mag[ok]; u=np.cos(np.deg2rad(a[ok])); v=-np.sin(np.deg2rad(a[ok])); xc=xx[ok]; yc=yy[ok]
    seg=np.stack([np.stack([xc-.5*L*u,yc-.5*L*v],-1),np.stack([xc+.5*L*u,yc+.5*L*v],-1)],1)
    ax.imshow(0.25+0.5*stretch(img),cmap='gray',vmin=0,vmax=1); ax.add_collection(LineCollection(seg,colors=color,linewidths=0.9))
    if crop is not None: ax.set_xlim(crop[0],crop[1]); ax.set_ylim(crop[1],crop[0])
# ---- merged gallery: 4 representative textures
pick=[(0,'curved strata\n(synthetic)'),(1,'collagen-like\nfibers (synthetic)'),(4,'textile weave\n(synthetic)'),(7,'retinal vessels\n(CC0)')]
fig,ax=plt.subplots(4,4,figsize=(FULL,FULL*0.99),layout='constrained')
for i,(idx,nm) in enumerate(pick):
    im=cases[idx][1]; p=mt.multi_field(im); st=mt.structure_tensor(im)
    ax[i,0].imshow(stretch(im),cmap='gray',vmin=0,vmax=1); directors(ax[i,1],im,p,C['pbgia']); directors(ax[i,2],im,cont_field(im),C['pbgia']); directors(ax[i,3],im,st,C['conv'])
    ax[i,0].set_ylabel(nm,fontsize=7)
ax[0,0].set_title('input'); ax[0,1].set_title('PBGIA, five cuts'); ax[0,2].set_title('PBGIA, continuous'); ax[0,3].set_title('structure tensor')
for a,l in zip(ax.ravel(),'abcdefghijklmnop'):
    yl=a.get_ylabel(); noaxis(a); a.set_ylabel(yl,fontsize=7); imlabel(a,l)
save(fig,'FigA1_gallery',OUT)
# ---- deskew, central crop only (rotation borders excluded)
im=cases[6][1]; p=mt.multi_field(im); rot=mt.img_rotate(im,10); pro=mt.multi_field(rot)
roi=mt.roi_center(im.shape,.67); a0,_=mt.axis_mean(p['tangent'],p['gap'],roi); a1,_=mt.axis_mean(pro['tangent'],pro['gap'],roi)
inf=float(mt.angle_delta(a1,a0)); cor=mt.img_rotate(rot,-inf)
c0,c1=70,250   # central 180x180 window, free of rotation borders
fig,ax=plt.subplots(1,4,figsize=(FULL,1.75),layout='constrained')
for a,img,t in ((ax[0],im,'original'),(ax[1],rot,'rotated by $+10^\\circ$'),(ax[3],cor,f'corrected by ${-inf:+.2f}^\\circ$')):
    a.imshow(stretch(img)[c0:c1,c0:c1],cmap='gray',vmin=0,vmax=1); a.set_title(t)
directors(ax[2],rot,pro,C['pbgia'],step=9,max_length=9,crop=(c0,c1)); ax[2].set_title('PBGIA director')
for a,l in zip(ax,'abcd'): noaxis(a); imlabel(a,l)
save(fig,'FigA2_deskew',OUT); print('inferred',inf)
