"""Reproducible multi-cutoff Z2->S2->tensor director study; no model fitting.
The positive-semidefinite tensors are summed before diagonalization, never directors.
Segments are dense and their LENGTH encodes eigenvalue gap (anisotropy strength).
Original image copyright-uncertain photos are excluded.
"""
import sys,csv
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from skimage import data,transform,filters

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'code'))
from real_image_examples import generate_field,normalize_0_1,img_rotate,axis_mean,angle_delta,roi_center,synthetic_strata,retina_vessel_input
from additional_pilots import synthetic_collagen, synthetic_weave, synthetic_wood, synthetic_leaf, synthetic_buildings

plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})


def resize_gray(a,shape=(320,320)):
    arr=np.asarray(a)
    if arr.ndim==3:
        arr=(arr.astype(float)[...,:3]*np.array([.2989,.587,.114])).sum(axis=-1)
    if arr.max()>1.5: arr=arr/255.0
    return transform.resize(arr,shape,anti_aliasing=True,preserve_range=True).astype('float32').clip(0,1)


def multi_field(img, sigma=8, thresholds=(.15,.325,.5,.675,.85), smooth=.8):
    # Quantile-based multi cutoff: retains tonal domain structure, no Otsu single threshold.
    sm=ndi.gaussian_filter(img,smooth)
    cuts=np.unique(np.quantile(sm,thresholds))
    Mrr=Mcc=Mrc=None
    activations=[]
    for q in cuts:
        mask=sm>q
        frac=mask.mean()
        if frac<.005 or frac>.995: continue
        vr,vc,s=generate_field(mask,nsteps=30)
        # M carries magnitude and orientation; Gaussian convolution of outer products.
        rr=ndi.gaussian_filter(vr*vr,sigma)
        cc=ndi.gaussian_filter(vc*vc,sigma)
        rc=ndi.gaussian_filter(vr*vc,sigma)
        if Mrr is None: Mrr=np.zeros_like(rr);Mcc=np.zeros_like(cc);Mrc=np.zeros_like(rc)
        Mrr+=rr;Mcc+=cc;Mrc+=rc
        activations.append(float(frac))
    if not activations: raise RuntimeError('no useful threshold')
    Mrr/=len(activations);Mcc/=len(activations);Mrc/=len(activations)
    trace=Mrr+Mcc
    gap=np.hypot(Mcc-Mrr,2*Mrc)
    coherence=gap/(trace+1e-12)
    normal=(.5*np.degrees(np.arctan2(-2*Mrc,Mcc-Mrr)))%180
    tang=(normal+90)%180
    return {'tangent':tang,'gap':gap,'trace':trace,'coherence':coherence,'cuts':cuts,'n':len(activations),'Mrr':Mrr,'Mcc':Mcc,'Mrc':Mrc}


def structure_tensor(gray, sigma=8,pre=1.2):
    a=ndi.gaussian_filter(gray,pre)
    gr,gc=np.gradient(a)
    Mrr=ndi.gaussian_filter(gr*gr,sigma)
    Mcc=ndi.gaussian_filter(gc*gc,sigma)
    Mrc=ndi.gaussian_filter(gr*gc,sigma)
    trace=Mrr+Mcc
    gap=np.hypot(Mcc-Mrr,2*Mrc)
    coherence=gap/(trace+1e-12)
    normal=(.5*np.degrees(np.arctan2(-2*Mrc,Mcc-Mrr)))%180
    return {'tangent':(normal+90)%180,'gap':gap,'trace':trace,'coherence':coherence}


def strength_segments(ax,img,res,step=8,gate=.28,max_length=13,color='#ffe54b',scale95=None):
    H,W=img.shape
    yy,xx=np.mgrid[step//2:H:step,step//2:W:step]
    a=res['tangent'][yy,xx]; coherence=res['coherence'][yy,xx]
    amp=res['gap'][yy,xx]
    # Use a global normalization set by the entire field, not each selected point.
    s95=float(np.percentile(res['gap'],95)) if scale95 is None else float(scale95)
    mag=np.clip(amp/max(s95,1e-10),0,1)
    valid=(coherence>=gate)&(mag>=.07)
    L=(1.4+max_length*mag[valid])
    u=np.cos(np.deg2rad(a[valid]));v=-np.sin(np.deg2rad(a[valid]))
    xc=xx[valid];yc=yy[valid]
    seg=np.stack([np.stack([xc-.5*L*u,yc-.5*L*v],axis=-1),np.stack([xc+.5*L*u,yc+.5*L*v],axis=-1)],axis=1)
    ax.imshow(img,cmap='gray',vmin=0,vmax=1)
    ax.add_collection(LineCollection(seg,colors=color,linewidths=.85,alpha=.92))
    ax.axis('off')
    return float(np.mean(valid)),s95


def equivariance(gray):
    rots=(-15,-10,-5,0,5,10,15)
    result=[]
    for r in rots:
        a=img_rotate(gray,r)
        p=multi_field(a)
        b=structure_tensor(a)
        roi=roi_center(a.shape,.67)
        ap,op=axis_mean(p['tangent'],p['gap'],roi)
        ast,ost=axis_mean(b['tangent'],b['gap'],roi)
        result.append((r,ap,ast,op,ost))
    p0=result[3][1];s0=result[3][2]
    e1=[abs(angle_delta(angle_delta(x[1],p0),x[0])) for x in result]
    e2=[abs(angle_delta(angle_delta(x[2],s0),x[0])) for x in result]
    return float(np.mean(e1)),float(np.mean(e2)),result


def construct_cases():
    st,truth=synthetic_strata(size=320,amplitude=22,period=190.,stripe=17,seed=19)
    cases=[('Curved strata (synthetic)',st,'synthetic',truth),
           ('Collagen-like fibers (synthetic)',synthetic_collagen(),'synthetic',None),
           ('Leaf veins (synthetic)',synthetic_leaf(),'synthetic',None),
           ('Wood grain (synthetic)',synthetic_wood(),'synthetic',None),
           ('Textile weave (synthetic)',synthetic_weave(),'synthetic',None),
           ('Building array (synthetic)',synthetic_buildings(),'synthetic',None),
           ('CC0 brick wall',data.brick(),'CC0 scikit-image',None),
           ('CC0 retina vessel contrast',retina_vessel_input(),'CC0 scikit-image',None),
           ('CC0 grass texture',data.grass(),'CC0 scikit-image',None)]
    return [(nm,resize_gray(img),typ,truth) for nm,img,typ,truth in cases]


def run():
    cases=construct_cases()
    records=[]
    fig,axes=plt.subplots(len(cases),4,figsize=(12.2,2.45*len(cases)),layout='constrained')
    for i,(name,im,typ,truth) in enumerate(cases):
        p=multi_field(im); s=structure_tensor(im)
        # Example-specific QA: discard global thresholds that miss actual veins in strongly structured inputs
        cov,norm=strength_segments(axes[i,2],im,p,step=8,max_length=12,gate=.25)
        cst,_=strength_segments(axes[i,3],im,s,step=8,max_length=12,gate=.25,color='#48dce9')
        axes[i,0].imshow(im,cmap='gray',vmin=0,vmax=1)
        axes[i,0].axis('off')
        axes[i,1].imshow(p['gap'],cmap='magma',vmin=0,vmax=np.percentile(p['gap'],99))
        axes[i,1].axis('off')
        axes[i,0].set_title(name+'\ninput')
        axes[i,1].set_title('PBGIA tensor gap')
        axes[i,2].set_title('PBGIA: length $\propto$ anisotropy')
        axes[i,3].set_title('Image structure tensor')
        roi=roi_center(im.shape,.67)
        aa,oo=axis_mean(p['tangent'],p['gap'],roi)
        bb,ob=axis_mean(s['tangent'],s['gap'],roi)
        gt_p=gt_s=float('nan')
        if truth is not None:
            valid=(p['coherence']>.25)&(p['gap']>np.percentile(p['gap'],30))
            gt_p=float(np.mean(np.abs(angle_delta(p['tangent'][valid],truth[valid]))))
            gt_s=float(np.mean(np.abs(angle_delta(s['tangent'][valid],truth[valid]))))
        # Run full equivariance on each image; 7 rotations and 5 thresholds, 35 fields
        rp,rs,_=equivariance(im)
        records.append({'example':name,'origin':typ,'thresholds':p['n'],
                        'selected_fraction_pbgia':cov,'selected_fraction_ST':cst,
                        'global_axis_pbgia_deg':aa,'global_axis_ST_deg':bb,
                        'global_order_pbgia':oo,'global_order_ST':ob,
                        'rotation_MAE_PBGIA_deg':rp,'rotation_MAE_ST_deg':rs,
                        'local_GT_MAE_PBGIA_deg':gt_p,'local_GT_MAE_ST_deg':gt_s})
        print(i+1,name,'MAE rot',round(rp,3),round(rs,3),'gt',round(gt_p,3),round(gt_s,3),flush=True)
    fig.savefig(HERE/'figures'/'FigS_multithreshold_application_gallery.pdf')
    fig.savefig(HERE/'figures'/'FigS_multithreshold_application_gallery.png',dpi=130)
    plt.close(fig)
    with open(HERE/'data'/'multithreshold_applications.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    # Main Figure 6: deliberate example selection (a validated GT; two plausible application motifs)
    pick=[0,1,2]
    fig,axes=plt.subplots(len(pick),3,figsize=(10.7,2.6*len(pick)),layout='constrained')
    for j,idx in enumerate(pick):
        name,im,typ,_=cases[idx]
        p=multi_field(im)
        axes[j,0].imshow(im,cmap='gray',vmin=0,vmax=1);axes[j,0].axis('off')
        axes[j,1].imshow(p['gap'],cmap='magma',vmin=0,vmax=np.percentile(p['gap'],99));axes[j,1].axis('off')
        strength_segments(axes[j,2],im,p,step=6,max_length=10,gate=.25)
        axes[j,0].set_title(f'({chr(97+3*j)}) {name}')
        axes[j,1].set_title(f'({chr(98+3*j)}) $\\lambda_1-\\lambda_2$')
        axes[j,2].set_title(f'({chr(99+3*j)}) multi-cutoff director, length $\\propto$ gap')
    fig.savefig(HERE/'figures'/'Fig06_multithreshold_directors.pdf')
    fig.savefig(HERE/'figures'/'Fig06_multithreshold_directors.png',dpi=175)
    plt.close(fig)
    # Main deskew example uses CC0 brick photograph, transform +10 degrees
    im=cases[6][1];p=multi_field(im)
    tilt=10; rot=img_rotate(im,tilt)
    pro=multi_field(rot)
    a0,_=axis_mean(p['tangent'],p['gap'],roi_center(im.shape,.67))
    a1,_=axis_mean(pro['tangent'],pro['gap'],roi_center(im.shape,.67))
    inferred=float(angle_delta(a1,a0))
    corrected=img_rotate(rot,-inferred)
    fig,axes=plt.subplots(1,4,figsize=(12.1,3.1),layout='constrained')
    axes[0].imshow(im,cmap='gray',vmin=0,vmax=1);axes[0].set_title('(a) CC0 brick texture')
    axes[1].imshow(rot,cmap='gray',vmin=0,vmax=1);axes[1].set_title('(b) imposed $+10^\\circ$')
    strength_segments(axes[2],rot,pro,step=6,max_length=10,gate=.25)
    axes[2].set_title('(c) tensor-gap-weighted director')
    axes[3].imshow(corrected,cmap='gray',vmin=0,vmax=1);axes[3].set_title(f'(d) correction ${-inferred:+.1f}^\\circ$')
    for ax in axes:ax.axis('off')
    fig.savefig(HERE/'figures'/'Fig09_multithreshold_deskew.pdf')
    fig.savefig(HERE/'figures'/'Fig09_multithreshold_deskew.png',dpi=175)
    plt.close(fig)
    return records,inferred

if __name__=='__main__':
    run()
