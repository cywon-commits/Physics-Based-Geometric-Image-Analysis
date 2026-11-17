"""Correct Fig1 wall normal width; crop to common valid ROI in brick deskew."""
from pathlib import Path
import sys
import numpy as np, scipy.ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from skimage import data
OUT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(OUT/'code'))
from multithreshold_directors import multi_field, strength_segments, resize_gray
from real_image_examples import img_rotate,axis_mean,angle_delta,roi_center
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'savefig.bbox':'tight'})
# Fig 1 is an illustration of exact circular offset, not simulated 2D
n=260;R=60;yy,xx=np.mgrid[:n,:n];x=xx-(n-1)/2;y=yy-(n-1)/2
u=R-np.hypot(x,y);f=np.where(u>=0,1.,-1.)
s=np.tanh(u/3)
spin=np.stack([np.cos(np.arctan2(y,x))*np.sqrt(np.maximum(0,1-s*s)),
               np.sin(np.arctan2(y,x))*np.sqrt(np.maximum(0,1-s*s)),s],axis=-1)
gy,gx=np.gradient(s);density=np.hypot(gx,gy)
fig,axs=plt.subplots(1,4,figsize=(13.1,3.4),layout='constrained')
axs[0].imshow(f,cmap='gray',vmin=-1,vmax=1);axs[0].set_title('(a) Binary $Z_2$ field')
axs[1].imshow(s,cmap='coolwarm',vmin=-1,vmax=1);axs[1].set_title('(b) Smooth unit-spin wall')
axs[2].imshow(density,cmap='magma');axs[2].set_title('(c) Boundary-localized gradient')
axs[3].contour(x,y,u,levels=[-4,0,4],colors=['#d18a4d','#262f3a','#3d91b0'],linewidths=[2,2.2,2]);axs[3].set_aspect('equal')
axs[3].set_xlim(-84,84);axs[3].set_ylim(-84,84)
axs[3].annotate('',xy=(R+4,0),xytext=(R-4,0),arrowprops={'arrowstyle':'<->','linewidth':1.4,'color':'black'})
axs[3].text(-75,72,'$u=\pm4$ offsets',fontsize=9);axs[3].set_title('(d) Constant normal-offset wall')
for ax in axs:ax.axis('off')
fig.savefig(OUT/'figures'/'Fig01_conceptual_pipeline.pdf');fig.savefig(OUT/'figures'/'Fig01_conceptual_pipeline.png',dpi=160);plt.close(fig)

# Fig6: original + rotated + directors + corrected, all cropped to same valid core
im=resize_gray(data.brick());p0=multi_field(im)
tilt=10;rot=img_rotate(im,tilt);p=multi_field(rot)
a0,_=axis_mean(p0['tangent'],p0['gap'],roi_center(im.shape,.67))
a1,_=axis_mean(p['tangent'],p['gap'],roi_center(im.shape,.67))
inferred=float(angle_delta(a1,a0));corrected=img_rotate(rot,-inferred)
valid1=ndi.rotate(np.ones_like(im,dtype=np.float32),tilt,reshape=False,order=0,mode='constant',cval=0)>0.5
valid2=ndi.rotate(valid1.astype(np.float32),-inferred,reshape=False,order=0,mode='constant',cval=0)>0.5
# common valid square, plus 4px margin
valid=valid1&valid2
center=np.array(valid.shape)//2
r=0
for k in range(20,145):
    y0,y1=center[0]-k,center[0]+k; x0,x1=center[1]-k,center[1]+k
    if valid[y0:y1,x0:x1].all(): r=k
    else:break
r=max(40,r-4)
y0,y1=center[0]-r,center[0]+r;x0,x1=center[1]-r,center[1]+r
crop=lambda a:a[y0:y1,x0:x1]
fig,axes=plt.subplots(1,4,figsize=(12.3,3.35),layout='constrained')
axes[0].imshow(crop(im),cmap='gray',vmin=0,vmax=1);axes[0].set_title('(a) CC0 brick texture')
axes[1].imshow(crop(rot),cmap='gray',vmin=0,vmax=1);axes[1].set_title('(b) imposed $+10^\circ$')
sub={key:crop(val) if isinstance(val,np.ndarray) and val.shape==im.shape else val for key,val in p.items()}
strength_segments(axes[2],crop(rot),sub,step=6,max_length=9,gate=.25);axes[2].set_title('(c) Gap-scaled directors')
axes[3].imshow(crop(corrected),cmap='gray',vmin=0,vmax=1);axes[3].set_title(f'(d) corrected ${-inferred:+.2f}^\circ$')
for ax in axes:ax.axis('off')
fig.savefig(OUT/'figures'/'Fig09_multithreshold_deskew.pdf');fig.savefig(OUT/'figures'/'Fig09_multithreshold_deskew.png',dpi=160);plt.close(fig)
(OUT/'data'/'brick_valid_crop.csv').write_text('inserted_rotation_deg,inferred_relative_rotation_deg,crop_side_px\n'+f'{tilt},{inferred},{2*r}\n')
print('FIG6 valid crop',2*r,'shift',inferred,flush=True)
