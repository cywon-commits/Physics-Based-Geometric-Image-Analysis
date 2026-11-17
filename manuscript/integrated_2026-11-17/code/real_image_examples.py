"""Reproducible, newly evaluated REAL CC0 image PBGIA examples.

Inputs: scikit-image 0.26.0 bundled data: brick, grass, retina.
Each is documented as CC0 in skimage.data function docstrings.
No images supplied by the user or third-party copyright-uncertain photos are used.

Output: manuscript PDF figures and CSV of rotation-equivariance measures.
This is a NEW exploratory pilot, not part of the frozen final PBGIA benchmark.
"""
import csv
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
from skimage import data, color, filters, transform
from skimage.filters import frangi, threshold_otsu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
(OUT/'figures').mkdir(exist_ok=True)
(OUT/'data').mkdir(exist_ok=True)
(OUT/'source_images').mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})


def normalize_0_1(a):
    a=np.asarray(a,dtype=np.float32)
    return np.clip(a/255.0,0,1)


def rasterize(gray,blur=0.8):
    # one fixed, non-trained, global grayscale threshold
    sm=ndi.gaussian_filter(gray,blur)
    return sm>threshold_otsu(sm)


def generate_field(mask,nsteps=30,J=1.0,D=1.0,h=2.0,w_init=2.3):
    """2D PBGIA central-gradient synchronized normalized alignment.
       Grid: axis 0 is image row, axis 1 image column; periodic dynamics.
    """
    b=np.asarray(mask,bool)
    f=np.where(b,1.0,-1.0).astype(np.float32)
    xi=ndi.distance_transform_edt(b)-ndi.distance_transform_edt(~b)
    s=np.tanh(xi/w_init).astype(np.float32)
    dr=(np.roll(s,-1,0)-np.roll(s,1,0))*0.5
    dc=(np.roll(s,-1,1)-np.roll(s,1,1))*0.5
    norm=np.sqrt(dr*dr+dc*dc)
    scale=np.sqrt(np.maximum(0,1-s*s))/np.maximum(norm,1e-12)
    vr=dr*scale; vc=dc*scale
    for t in range(nsteps):
        Hr=J*(np.roll(vr,-1,0)+np.roll(vr,1,0)+np.roll(vr,-1,1)+np.roll(vr,1,1))
        Hc=J*(np.roll(vc,-1,0)+np.roll(vc,1,0)+np.roll(vc,-1,1)+np.roll(vc,1,1))
        Hs=J*(np.roll(s,-1,0)+np.roll(s,1,0)+np.roll(s,-1,1)+np.roll(s,1,1))+h*f
        Hr+=D*(np.roll(s,-1,0)-np.roll(s,1,0))*0.5
        Hc+=D*(np.roll(s,-1,1)-np.roll(s,1,1))*0.5
        H=np.sqrt(Hr*Hr+Hc*Hc+Hs*Hs)
        valid=H>1e-12
        vr=np.where(valid,Hr/np.maximum(H,1e-12),vr)
        vc=np.where(valid,Hc/np.maximum(H,1e-12),vc)
        s=np.where(valid,Hs/np.maximum(H,1e-12),s)
    return vr,vc,s


def tensor_from_vector(vr,vc,sigma=11.0):
    rr=ndi.gaussian_filter(vr*vr,sigma)
    cc=ndi.gaussian_filter(vc*vc,sigma)
    rc=ndi.gaussian_filter(vr*vc,sigma)
    tr=rr+cc
    # Angle of tangential (perpendicular to normal) director measured CCW
    # from horizontal image axis, modulo 180 deg.
    normal=np.degrees(.5*np.arctan2(-2*rc,cc-rr))
    tang=(normal+90)%180
    gap=np.hypot(cc-rr,2*rc)
    coherence=gap/(tr+1e-12)
    return tang, coherence,tr,rr,cc,rc


def image_st(gray,sigma=11.,pre=1.2):
    gray=ndi.gaussian_filter(gray,pre)
    gr,gc=np.gradient(gray)
    return tensor_from_vector(gr,gc,sigma=sigma)


def angle_delta(a,b):
    return (a-b+90.)%180.-90.


def axis_mean(theta,weights,roi=None):
    # axial circular mean in 0-180 deg
    w=np.asarray(weights,dtype=np.float64)
    if roi is not None: w=w*np.asarray(roi)
    x=np.sum(w*np.cos(2*np.deg2rad(theta)))
    y=np.sum(w*np.sin(2*np.deg2rad(theta)))
    angle=(0.5*np.degrees(np.arctan2(y,x)))%180
    order=np.hypot(x,y)/max(np.sum(w),1e-15)
    return float(angle),float(order)


def img_rotate(img,theta):
    return ndi.rotate(img,float(theta),reshape=False,order=1,mode='reflect',prefilter=False)


def roi_center(shape,fraction=.65):
    R,C=shape; cy,cx=np.ogrid[:R,:C]
    return ((cy-R/2)/(R*fraction/2))**2+((cx-C/2)/(C*fraction/2))**2 < 1


def process_real(gray,crop=None,sz=320,sigma=10):
    if crop is not None:
        gray=gray[crop]
    gray=transform.resize(gray,(sz,sz),anti_aliasing=True,preserve_range=True).astype('float32')
    mask=rasterize(gray)
    vr,vc,s=generate_field(mask)
    p=tensor_from_vector(vr,vc,sigma=sigma)
    st=image_st(gray,sigma=sigma)
    return gray,mask,(vr,vc,s),p,st


def field_overlay(ax,img,theta,coh,tr,step=16,threshold=.4,black=False):
    ax.imshow(img,cmap='gray' if img.ndim==2 else None,vmin=0,vmax=1)
    yy,xx=np.mgrid[step//2:img.shape[0]:step,step//2:img.shape[1]:step]
    tt=theta[yy,xx]; ww=coh[yy,xx]; signal=tr[yy,xx]
    allowed=(ww>threshold)&(signal>np.percentile(tr,35))
    xx=xx[allowed];yy=yy[allowed];tt=tt[allowed]
    L=step*.65;dx=np.cos(np.deg2rad(tt))*L; dy=-np.sin(np.deg2rad(tt))*L
    ax.quiver(xx,yy,dx,dy,pivot='mid',angles='xy',scale_units='xy',scale=1,
              width=.005,color='yellow' if not black else 'navy',headlength=0,headwidth=0,headaxislength=0)
    ax.axis('off')


def crop_mid(a,frac=.78):
    H,W=a.shape[:2]; h=int(H*frac); w=int(W*frac)
    return a[(H-h)//2:(H+h)//2,(W-w)//2:(W+w)//2]


def retina_vessel_input():
    a=data.retina()
    rgb=transform.resize(a,(520,520),anti_aliasing=True,preserve_range=True)
    green=rgb[:,:,1].astype('float32')/255
    # Local dark-tube response; suppress the artificial circular field boundary.
    smooth=ndi.gaussian_filter(green,17)
    residual=np.clip(smooth-green,0,None)
    Y,X=np.ogrid[:520,:520]
    inner=((X-260)**2+(Y-260)**2)<(185**2)
    residual[~inner]=0
    return residual


def synthetic_strata(size=320,amplitude=22.0,period=190.,stripe=17.,seed=19):
    y,x=np.mgrid[:size,:size]
    k=2*np.pi/period
    phase=y-amplitude*np.sin(k*x)
    rng=np.random.default_rng(seed)
    gray=np.clip(.5+.35*np.sin(2*np.pi*phase/stripe)+.025*rng.standard_normal((size,size)),0,1)
    true_tang=(-np.degrees(np.arctan(amplitude*k*np.cos(k*x))))%180
    return gray.astype('float32'),true_tang


def make_figures():
    brick=normalize_0_1(data.brick())
    retinal=retina_vessel_input()
    synthetic,truth=synthetic_strata()
    for key,a in [('brick',brick),('retina_vessels',retinal)]:
        np.save(OUT/'source_images'/f'{key}.npy',a)
    np.save(OUT/'data'/'synthetic_strata_ground_truth.npy',truth)

    fig,axs=plt.subplots(2,4,figsize=(12.6,6.7),layout='constrained')
    lines=[]
    # Row 1: explicitly synthetic ground truth and controlled comparison.
    gray,mask,sp,pbg,st=process_real(synthetic,sz=320,sigma=10)
    tang,coh,tr,*_=pbg; tangst,cst,tst,*_=st
    domainwall=np.abs(sp[2])<.7
    valid=(coh>.2)&(cst>.2)&(tr>np.percentile(tr,30))&domainwall
    errp=np.abs(angle_delta(tang[valid],truth[valid]))
    errs=np.abs(angle_delta(tangst[valid],truth[valid]))
    lines.append({'image':'synthetic curved stratification', 'valid_fraction':float(np.mean(valid)),
                  'median_axial_difference_deg':float(np.median(np.abs(angle_delta(tang[valid],tangst[valid])))),
                  'pbgia_mae_gt_deg':float(np.mean(errp)),'st_mae_gt_deg':float(np.mean(errs)),
                  'note':'analytic tangent ground truth'})
    axs[0,0].imshow(synthetic,cmap='gray',vmin=0,vmax=1);axs[0,0].set_title('(a) Synthetic curved layers')
    axs[0,1].imshow(sp[2],cmap='coolwarm',vmin=-1,vmax=1);axs[0,1].set_title('(b) Generated $s$ (N=30)')
    field_overlay(axs[0,2],synthetic,tang,coh,tr,step=19,threshold=.20)
    axs[0,2].set_title('(c) PBGIA tangent director')
    field_overlay(axs[0,3],synthetic,truth,np.ones_like(truth),np.ones_like(truth),step=19,threshold=.2)
    axs[0,3].set_title('(d) Analytic tangent orientation')

    # Row 2: genuine CC0 fundus photo; retinal vessels yield a binary field.
    gray,mask,sp,pbg,st=process_real(retinal,sz=320,sigma=8)
    tang,coh,tr,*_=pbg; tangst,cst,tst,*_=st
    axs[1,0].imshow(gray,cmap='gray');axs[1,0].set_title('(e) CC0 retinal vessel contrast')
    axs[1,1].imshow(mask,cmap='gray');axs[1,1].set_title('(f) Binary $Z_2$ input')
    field_overlay(axs[1,2],gray,tang,coh,tr,step=17,threshold=.50)
    axs[1,2].set_title('(g) PBGIA director, selected pixels')
    field_overlay(axs[1,3],gray,tangst,cst,tst,step=17,threshold=.50)
    axs[1,3].set_title('(h) Image structure tensor')
    valid=(coh>.5)&(cst>.5)&(tr>np.percentile(tr,65))&(tst>np.percentile(tst,65))
    error=np.abs(angle_delta(tang[valid],tangst[valid]))
    lines.append({'image':'CC0 retinal vessel photograph','valid_fraction':float(np.mean(valid)),
                  'median_axial_difference_deg':float(np.median(error)) if len(error) else np.nan,
                  'pbgia_mae_gt_deg':np.nan,'st_mae_gt_deg':np.nan,
                  'note':'agreement only; molecular director not measured; vessel mask via grayscale threshold'})
    for ax in axs.flat:ax.axis('off')
    fig.savefig(OUT/'figures'/'Fig06_real_director_CC0.pdf',bbox_inches='tight')
    fig.savefig(OUT/'figures'/'Fig06_real_director_CC0.png',dpi=160,bbox_inches='tight')
    plt.close(fig)

    # Global wall direction / deskew relative to known inserted rotations (not true architectural vertical)
    # Brick image contains strong elongated mortar lines.
    tilts=[-15,-10,-5,0,5,10,15]
    brick_small=transform.resize(brick,(320,320),anti_aliasing=True).astype(np.float32)
    values=[]
    for deg in tilts:
        rot=img_rotate(brick_small,deg)
        gray,mask,sp,pbg,st=process_real(rot,sz=320,sigma=9)
        tang,coh,tr,*_=pbg;tst,cst,trs,*_=st
        roi=roi_center(gray.shape,.67)
        # weighted double-angle orientation (normal angle rotated to tangential axis)
        angle1,order1=axis_mean(tang,coh*tr,roi)
        angle2,order2=axis_mean(tst,cst*trs,roi)
        values.append({'inserted_rotation_deg':deg,'pbgia_axis_deg':angle1,'pbgia_axial_order':order1,
                       'structure_tensor_axis_deg':angle2,'structure_tensor_axial_order':order2})
    # Relative angle to reference at 0 so the test is valid without unmeasured perspective ground truth.
    ref=values[tilts.index(0)]
    for r in values:
        r['pbgia_relative_deg']=angle_delta(r['pbgia_axis_deg'],ref['pbgia_axis_deg'])
        r['st_relative_deg']=angle_delta(r['structure_tensor_axis_deg'],ref['structure_tensor_axis_deg'])
        r['pbgia_error_deg']=angle_delta(r['pbgia_relative_deg'],r['inserted_rotation_deg'])
        r['st_error_deg']=angle_delta(r['st_relative_deg'],r['inserted_rotation_deg'])
    with open(OUT/'data'/'deskew_angle_sweep.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=values[0].keys());w.writeheader();w.writerows(values)
    with open(OUT/'data'/'director_agreement.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=lines[0].keys());w.writeheader();w.writerows(lines)
    tilt=10; vals=values[tilts.index(tilt)]
    raw=img_rotate(brick_small,tilt)
    corrected=img_rotate(raw,-vals['pbgia_relative_deg'])
    # The right panel should be close to reference (apart from rotation crop/interpolation)
    fig,axs=plt.subplots(1,4,figsize=(12.5,3.6),layout='constrained')
    axs[0].imshow(brick_small,cmap='gray');axs[0].set_title('(a) CC0 reference texture')
    axs[1].imshow(raw,cmap='gray');axs[1].set_title(f'(b) Rotated +{tilt:.0f} deg')
    gray,mask,sp,pbg,st=process_real(raw,sz=320,sigma=9)
    field_overlay(axs[2],gray,pbg[0],pbg[1],pbg[2],step=22,threshold=.42)
    axs[2].set_title('(c) PBGIA orientation readout')
    axs[3].imshow(corrected,cmap='gray');axs[3].set_title(f'(d) Corrected by {-vals["pbgia_relative_deg"]:+.1f} deg')
    for ax in axs:ax.axis('off')
    fig.savefig(OUT/'figures'/'Fig09_CC0_brick_rotation.pdf',bbox_inches='tight')
    fig.savefig(OUT/'figures'/'Fig09_CC0_brick_rotation.png',bbox_inches='tight',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.4,3.9))
    x=np.asarray(tilts); y1=np.array([r['pbgia_relative_deg'] for r in values]);y2=np.array([r['st_relative_deg'] for r in values])
    ax.plot(x,x,'k--',label='Ideal equivariance')
    ax.plot(x,y1,'o-',label='PBGIA binary field + tensor')
    ax.plot(x,y2,'s-',label='Intensity structure tensor')
    ax.set(xlabel='Inserted rotation (degrees)',ylabel='Estimated relative axis shift (degrees)',xlim=(-17,17),ylim=(-18,18))
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'figures'/'FigS01_rotation_equivariance.pdf',bbox_inches='tight')
    plt.close(fig)

    # Fig03: actual generated PBGIA texture for synthetic ellipse, NOT analytic field
    R=C=280;yr,xr=np.mgrid[:R,:C];a=np.deg2rad(27);x=(xr-C/2)*np.cos(a)+(yr-R/2)*np.sin(a);y=-(xr-C/2)*np.sin(a)+(yr-R/2)*np.cos(a)
    b=(x/73.)**2+(y/41.)**2<=1
    vr,vc,s=generate_field(b)
    # geometric energy proxy central gradient
    grads=[np.gradient(z) for z in [s,vr,vc]]
    energy=sum(q*q for gr in grads for q in gr)
    fig,axs=plt.subplots(1,4,figsize=(12.2,3.25),layout='constrained')
    axs[0].imshow(b,cmap='gray');axs[0].set_title('(a) Binary ellipse')
    axs[1].imshow(s,cmap='coolwarm',vmin=-1,vmax=1);axs[1].set_title('(b) Generated $s$ after $N=30$')
    field_overlay(axs[2],s,*tensor_from_vector(vr,vc,sigma=6)[:3],step=14,threshold=.18)
    axs[2].set_title('(c) Boundary tangent director')
    axs[3].imshow(energy,cmap='magma');axs[3].set_title('(d) Geometric gradient energy')
    for ax in axs:ax.axis('off')
    fig.savefig(OUT/'figures'/'Fig03_actual_N30_ellipse.pdf',bbox_inches='tight')
    fig.savefig(OUT/'figures'/'Fig03_actual_N30_ellipse.png',bbox_inches='tight',dpi=150)
    plt.close(fig)
    print('DE-SKEW',[(q['inserted_rotation_deg'],round(q['pbgia_relative_deg'],2),round(q['st_relative_deg'],2)) for q in values])
    print('DIRECTOR',lines)
    print('MAX ERR PBGIA',max(abs(x['pbgia_error_deg']) for x in values))
    return values,lines

if __name__=='__main__': make_figures()
