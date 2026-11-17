from pathlib import Path
import csv
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('/mnt/data/PBGIA_additional_examples_20261117')
FIG=OUT/'figures'; DAT=OUT/'data'
FIG.mkdir(parents=True, exist_ok=True); DAT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})

# ---------- Core PBGIA helpers (adapted from manuscript code) ----------
def normalize_0_1(a):
    a=np.asarray(a,dtype=np.float32)
    if a.ndim==3:
        if a.shape[2]==4: a=a[...,:3]
        a=(0.2989*a[...,0]+0.5870*a[...,1]+0.1140*a[...,2])
    if a.max()>1.5: a=a/255.0
    return np.clip(a,0,1)


def to_gray(a):
    return normalize_0_1(a)


def generate_field(mask,nsteps=30,J=1.0,D=1.0,h=2.0,w_init=2.3):
    b=np.asarray(mask,bool)
    f=np.where(b,1.0,-1.0).astype(np.float32)
    xi=ndi.distance_transform_edt(b)-ndi.distance_transform_edt(~b)
    s=np.tanh(xi/w_init).astype(np.float32)
    dr=(np.roll(s,-1,0)-np.roll(s,1,0))*0.5
    dc=(np.roll(s,-1,1)-np.roll(s,1,1))*0.5
    norm=np.sqrt(dr*dr+dc*dc)
    scale=np.sqrt(np.maximum(0,1-s*s))/np.maximum(norm,1e-12)
    vr=dr*scale; vc=dc*scale
    for _ in range(nsteps):
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


def choose_mask(gray, sigma=0.8, prefer='auto'):
    sm=ndi.gaussian_filter(gray,sigma)
    th=float(sm.mean()) if np.std(sm)<1e-6 else float(np.quantile(sm,0.5))
    # Otsu fallback without skimage
    hist,bin_edges=np.histogram(sm.ravel(),bins=256,range=(0,1))
    total=sm.size
    sum_total=np.dot(hist,(bin_edges[:-1]+bin_edges[1:])*0.5)
    sum_b=0.0; w_b=0.0; var_max=-1.0; otsu=th
    mids=(bin_edges[:-1]+bin_edges[1:])*0.5
    for i in range(256):
        w_b += hist[i]
        if w_b==0: continue
        w_f = total-w_b
        if w_f==0: break
        sum_b += mids[i]*hist[i]
        m_b=sum_b/w_b; m_f=(sum_total-sum_b)/w_f
        var_between=w_b*w_f*(m_b-m_f)*(m_b-m_f)
        if var_between>var_max:
            var_max=var_between; otsu=mids[i]
    if prefer=='bright':
        return sm>otsu
    if prefer=='dark':
        return sm<otsu
    # try both and keep more coherent one after a short run
    best=None; best_score=-1
    for name,mask in [('bright',sm>otsu),('dark',sm<otsu)]:
        frac=mask.mean()
        if frac<0.03 or frac>0.97: continue
        vr,vc,s=generate_field(mask,nsteps=18)
        tang,coh,tr,*_=tensor_from_vector(vr,vc,sigma=9)
        roi=roi_center(gray.shape,.72)
        sel=roi&(coh>0.35)&(tr>np.percentile(tr,50))&(np.abs(s)<0.85)
        score=float(np.mean(coh[sel])) if np.any(sel) else 0.0
        score*=1.0-abs(frac-0.5)
        if score>best_score:
            best_score=score; best=mask
    return best if best is not None else (sm>otsu)


def process(gray, prefer='auto', sigma=10, nsteps=30):
    gray=to_gray(gray)
    if gray.shape[0]!=320 or gray.shape[1]!=320:
        zoom=(320/gray.shape[0],320/gray.shape[1])
        gray=ndi.zoom(gray, zoom, order=1)
    mask=choose_mask(gray, prefer=prefer)
    vr,vc,s=generate_field(mask,nsteps=nsteps)
    pbg=tensor_from_vector(vr,vc,sigma=sigma)
    st=image_st(gray,sigma=sigma)
    return gray,mask,(vr,vc,s),pbg,st


def local_metrics(sp,pbg,st,truth=None):
    tang,coh,tr,*_=pbg; tangst,cst,tst,*_=st
    domainwall=np.abs(sp[2])<0.8
    valid=(coh>0.35)&(cst>0.35)&(tr>np.percentile(tr,45))&(tst>np.percentile(tst,45))&domainwall
    out={
        'selected_fraction':float(np.mean(valid)),
        'local_pbgia_vs_st_median_deg': float(np.median(np.abs(angle_delta(tang[valid],tangst[valid])))) if np.any(valid) else np.nan,
    }
    if truth is not None:
        good=(coh>0.2)&domainwall
        out['pbgia_mae_gt_deg']=float(np.mean(np.abs(angle_delta(tang[good],truth[good])))) if np.any(good) else np.nan
        out['st_mae_gt_deg']=float(np.mean(np.abs(angle_delta(tangst[good],truth[good])))) if np.any(good) else np.nan
    else:
        out['pbgia_mae_gt_deg']=np.nan; out['st_mae_gt_deg']=np.nan
    return out, valid


def rotation_metrics(gray, prefer='auto', sigma=9):
    tilts=[-15,-10,-5,0,5,10,15]
    vals=[]
    for deg in tilts:
        rot=img_rotate(gray,deg)
        g,m,sp,pbg,st=process(rot,prefer=prefer,sigma=sigma)
        tang,coh,tr,*_=pbg; tst,cst,trs,*_=st
        roi=roi_center(g.shape,.67)
        angle1,order1=axis_mean(tang,coh*tr*(np.abs(sp[2])<0.9),roi)
        angle2,order2=axis_mean(tst,cst*trs,roi)
        vals.append({'tilt':deg,'pbgia':angle1,'st':angle2,'pord':order1,'stord':order2})
    ref=vals[tilts.index(0)]
    pe=[]; se=[]
    for v in vals:
        rp=angle_delta(v['pbgia'],ref['pbgia']); rs=angle_delta(v['st'],ref['st'])
        pe.append(abs(angle_delta(rp,v['tilt'])))
        se.append(abs(angle_delta(rs,v['tilt'])))
    return float(np.mean(pe)), float(np.mean(se)), vals

# ---------- Synthetic image generators ----------

def synthetic_strata(size=320, amplitude=20.0, period=180., stripe=16., seed=19):
    y,x=np.mgrid[:size,:size]
    k=2*np.pi/period
    phase=y-amplitude*np.sin(k*x)
    rng=np.random.default_rng(seed)
    gray=np.clip(.55+.35*np.sin(2*np.pi*phase/stripe)+.03*rng.standard_normal((size,size)),0,1)
    true_tang=(-np.degrees(np.arctan(amplitude*k*np.cos(k*x))))%180
    return gray.astype('float32'), true_tang


def synthetic_collagen(size=320, seed=0):
    rng=np.random.default_rng(seed)
    y,x=np.mgrid[:size,:size]
    img=np.ones((size,size),np.float32)*0.75
    for _ in range(36):
        x0=rng.uniform(-0.1,1.1)*size
        y0=rng.uniform(0,size)
        theta=np.deg2rad(rng.normal(18,12))
        curv=rng.normal(0,0.15)
        for t in np.linspace(-1.2,1.2,700):
            xs=x0 + 115*t*np.cos(theta)
            ys=y0 + 115*t*np.sin(theta) + 22*curv*np.sin(2*np.pi*t)
            if 0<=xs<size and 0<=ys<size:
                xi=int(xs); yi=int(ys)
                img[max(0,yi-1):min(size,yi+2),max(0,xi-1):min(size,xi+2)] -= rng.uniform(0.02,0.06)
    img=ndi.gaussian_filter(img,1.4)
    img += 0.06*rng.standard_normal((size,size))
    return np.clip(img,0,1)


def synthetic_weave(size=320, seed=1):
    rng=np.random.default_rng(seed)
    y,x=np.mgrid[:size,:size]
    x2=(x*np.cos(np.deg2rad(12))+y*np.sin(np.deg2rad(12)))
    y2=(-x*np.sin(np.deg2rad(12))+y*np.cos(np.deg2rad(12)))
    p=22.0; w=6.5
    warp=np.exp(-((np.mod(x2,p)-p/2)/w)**2)
    weft=np.exp(-((np.mod(y2,p)-p/2)/w)**2)
    over=((np.floor(x2/p)+np.floor(y2/p))%2)
    img=0.25+0.45*np.maximum(warp*(0.7+0.3*over), weft*(0.7+0.3*(1-over)))
    img=ndi.gaussian_filter(img,1.0)+0.05*rng.standard_normal((size,size))
    return np.clip(img,0,1)


def synthetic_wood(size=320, seed=2):
    rng=np.random.default_rng(seed)
    y,x=np.mgrid[:size,:size]
    x0=x + 18*np.sin(y/35.) + 7*np.sin(y/9.)
    rings=np.sin((x0 + 8*np.sin(x/70.0))/8.0)
    img=0.5+0.23*rings+0.08*np.sin(y/4.5)+0.03*rng.standard_normal((size,size))
    img=ndi.gaussian_filter(img,0.8)
    return np.clip(img,0,1)


def synthetic_leaf(size=320, seed=3):
    rng=np.random.default_rng(seed)
    y,x=np.mgrid[:size,:size]
    cx=size/2; cy=size/2
    xx=(x-cx)/(0.42*size); yy=(y-cy)/(0.28*size)
    leaf=((xx**2 + yy**2 + 0.2*xx**4) < 1.0)
    img=np.ones((size,size),np.float32)
    img[leaf]=0.82
    veins=np.zeros_like(img)
    # midrib
    veins[:,size//2-1:size//2+1]=1
    # branches
    for side in (-1,1):
        for t in np.linspace(0.14,0.86,10):
            y0=int(t*size)
            length=int((0.12+0.20*(1-abs(t-0.5)*2))*size)
            ang=np.deg2rad(side*(25+rng.uniform(-7,7)))
            for s in np.linspace(0,1,length):
                xs=int(cx + side*s*length*np.cos(ang)*0.85)
                ys=int(y0 + s*length*np.sin(ang)*0.22)
                if 0<=xs<size and 0<=ys<size and leaf[ys,xs]:
                    veins[max(0,ys-1):min(size,ys+2),max(0,xs-1):min(size,xs+2)] = 1
    veins=ndi.gaussian_filter(veins,1.0)
    img -= 0.55*veins
    img += 0.03*rng.standard_normal((size,size))
    img[~leaf]=1.0
    return np.clip(img,0,1)


def synthetic_buildings(size=320, seed=4):
    rng=np.random.default_rng(seed)
    img=np.ones((size,size),np.float32)*0.92
    img[:90]=0.75  # sky gradient to be added
    horizon=95
    img[:horizon]=np.linspace(0.8,0.95,horizon)[:,None]
    x=8
    while x<size-20:
        w=int(rng.integers(14,32)); h=int(rng.integers(95,200))
        y0=size-h
        shade=float(rng.uniform(0.35,0.75))
        img[y0:size, x:x+w]=shade
        # windows
        for yy in range(y0+6,size-6,12):
            for xx in range(x+3,x+w-3,8):
                img[yy:yy+4, xx:xx+3]=min(0.95, shade+0.18)
        x += w + int(rng.integers(3,8))
    img=ndi.gaussian_filter(img,0.5)
    return np.clip(img,0,1)


def load_user(path):
    return np.array(Image.open(path))


def crop_rod_panel(arr, which='middle'):
    arr=arr[55:arr.shape[0]-2,:,:3]  # remove title bar
    h,w,_=arr.shape
    thirds=[(0,w//3),(w//3,2*w//3),(2*w//3,w)]
    idx={'left':0,'middle':1,'right':2}[which]
    a,b=thirds[idx]
    crop=arr[:,a+4:b-4]
    return crop


def field_overlay(ax,img,theta,coh,tr,step=18,threshold=.42,color='yellow'):
    ax.imshow(img,cmap='gray',vmin=0,vmax=1)
    yy,xx=np.mgrid[step//2:img.shape[0]:step, step//2:img.shape[1]:step]
    tt=theta[yy,xx]; ww=coh[yy,xx]; signal=tr[yy,xx]
    allowed=(ww>threshold)&(signal>np.percentile(tr,40))
    xx=xx[allowed]; yy=yy[allowed]; tt=tt[allowed]
    L=step*.65; dx=np.cos(np.deg2rad(tt))*L; dy=-np.sin(np.deg2rad(tt))*L
    ax.quiver(xx,yy,dx,dy,pivot='mid',angles='xy',scale_units='xy',scale=1,
              width=.0045,color=color,headlength=0,headwidth=0,headaxislength=0)
    ax.axis('off')


def run():
    rod_path='/mnt/data/ghostwriter_images/context/76e49ab2-17d3-5f3b-8447-f154c6ddbf52.png'
    bld_path='/mnt/data/ghostwriter_images/context/477580f7-d441-54c5-a572-6bb0cb1c804e.png'
    examples=[]
    strata, truth = synthetic_strata()
    examples.append(dict(name='collagen_like_tissue', group='candidate', source='synthetic fibrous collagen-like texture', gray=synthetic_collagen(), prefer='dark', truth=None, rights='synthetic'))
    examples.append(dict(name='geological_strata', group='candidate', source='synthetic curved strata', gray=strata, prefer='bright', truth=truth, rights='synthetic'))
    examples.append(dict(name='textile_weave', group='candidate', source='synthetic woven fabric', gray=synthetic_weave(), prefer='bright', truth=None, rights='synthetic'))
    examples.append(dict(name='wood_grain', group='candidate', source='synthetic wood-grain texture', gray=synthetic_wood(), prefer='bright', truth=None, rights='synthetic'))
    examples.append(dict(name='leaf_venation', group='candidate', source='synthetic leaf venation', gray=synthetic_leaf(), prefer='dark', truth=None, rights='synthetic'))
    examples.append(dict(name='rod_bacteria', group='exploratory', source='user-provided internet image; not manuscript-safe', gray=load_user(rod_path), prefer='bright', truth=None, rights='user_provided_not_for_publication'))
    examples.append(dict(name='building_array', group='exploratory', source='user-provided internet image; not manuscript-safe', gray=load_user(bld_path), prefer='bright', truth=None, rights='user_provided_not_for_publication'))
    examples.append(dict(name='brick_wall_open_proxy', group='candidate', source='local CC0/open brick preview used as architectural proxy', gray=load_user('/mnt/data/_preview_brick.png'), prefer='bright', truth=None, rights='local_open_preview'))
    examples.append(dict(name='retina_venation_proxy', group='candidate', source='local CC0/open retinal preview used as venation proxy', gray=load_user('/mnt/data/_preview_retina.png'), prefer='dark', truth=None, rights='local_open_preview'))
    examples.append(dict(name='grass_fiber_proxy', group='candidate', source='local CC0/open grass preview used as fibrous texture proxy', gray=load_user('/mnt/data/_preview_grass.png'), prefer='bright', truth=None, rights='local_open_preview'))

    rows=[]
    n=len(examples)
    fig,axs=plt.subplots(n,4,figsize=(10.2,2.2*n),layout='constrained')
    if n==1: axs=np.array([axs])
    for i,ex in enumerate(examples):
        gray=to_gray(ex['gray'])
        if ex['name']=='rod_bacteria':
            gray=to_gray(crop_rod_panel(ex['gray'],'middle'))
        elif ex['name']=='building_array':
            gray=to_gray(ex['gray'])[40:360,:620]
        g,mask,sp,pbg,st=process(gray,prefer=ex['prefer'])
        tang,coh,tr,*_=pbg; tangst,cst,tst,*_=st
        metrics,valid=local_metrics(sp,pbg,st,ex['truth'])
        pmae,smae,_=rotation_metrics(g,prefer=ex['prefer'])
        a1,o1=axis_mean(tang,coh*tr*(np.abs(sp[2])<0.9),roi_center(g.shape,.67))
        a2,o2=axis_mean(tangst,cst*tst,roi_center(g.shape,.67))
        row=dict(name=ex['name'], group=ex['group'], source=ex['source'], rights=ex['rights'],
                 global_axis_pbgia_deg=a1, global_axis_st_deg=a2,
                 global_order_pbgia=o1, global_order_st=o2,
                 rotation_mae_pbgia_deg=pmae, rotation_mae_st_deg=smae,
                 **metrics)
        rows.append(row)
        axs[i,0].imshow(g,cmap='gray',vmin=0,vmax=1); axs[i,0].set_title(f"{ex['name']}\ninput")
        axs[i,1].imshow(mask,cmap='gray'); axs[i,1].set_title('binary $Z_2$')
        field_overlay(axs[i,2],g,tang,coh,tr,step=18,threshold=.42,color='yellow'); axs[i,2].set_title('PBGIA')
        field_overlay(axs[i,3],g,tangst,cst,tst,step=18,threshold=.42,color='cyan'); axs[i,3].set_title('structure tensor')
        for j in range(4): axs[i,j].axis('off')
    fig.savefig(FIG/'all_examples_montage.png',dpi=160)
    fig.savefig(FIG/'all_examples_montage.pdf')
    plt.close(fig)

    # Small bar plot for deskew/orientation accuracy
    cand=[r for r in rows]
    names=[r['name'] for r in cand]
    x=np.arange(len(names)); width=0.38
    fig,ax=plt.subplots(figsize=(12,3.8))
    ax.bar(x-width/2,[r['rotation_mae_pbgia_deg'] for r in cand],width,label='PBGIA')
    ax.bar(x+width/2,[r['rotation_mae_st_deg'] for r in cand],width,label='Structure tensor')
    ax.set_xticks(x); ax.set_xticklabels(names,rotation=40,ha='right')
    ax.set_ylabel('Mean abs. relative rotation error (deg)')
    ax.set_title('Global-axis equivariance on candidate and exploratory examples')
    ax.legend(fontsize=8); ax.grid(alpha=.2,axis='y')
    fig.savefig(FIG/'rotation_error_summary.png',dpi=160)
    fig.savefig(FIG/'rotation_error_summary.pdf')
    plt.close(fig)

    with open(DAT/'example_metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # concise markdown report
    def fmt(x):
        return 'NA' if (isinstance(x,float) and np.isnan(x)) else (f'{x:.2f}' if isinstance(x,(int,float,np.floating)) else str(x))
    lines=[]
    lines.append('# Additional PBGIA application pilots\n')
    lines.append('This pilot expands beyond fingerprint-like textures to multiple application motifs requested by the user. It mixes manuscript-safe synthetic/open examples with exploratory user-provided internet images that should **not** be reused in publication.\n')
    lines.append('## Example summary\n')
    lines.append('| name | group | rotation MAE PBGIA (deg) | rotation MAE ST (deg) | local median PBGIA-ST (deg) | PBGIA GT MAE (deg) | ST GT MAE (deg) | comment |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---|')
    for r in rows:
        comment=[]
        if r['pbgia_mae_gt_deg']==r['pbgia_mae_gt_deg']:
            if r['pbgia_mae_gt_deg']<r['st_mae_gt_deg']: comment.append('PBGIA better to local GT')
            else: comment.append('ST better to local GT')
        if r['rotation_mae_pbgia_deg']<r['rotation_mae_st_deg']:
            comment.append('PBGIA better global deskew')
        else:
            comment.append('ST better global deskew')
        if r['group']=='exploratory':
            comment.append('not manuscript-safe source')
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            r['name'], r['group'], fmt(r['rotation_mae_pbgia_deg']), fmt(r['rotation_mae_st_deg']),
            fmt(r['local_pbgia_vs_st_median_deg']), fmt(r['pbgia_mae_gt_deg']), fmt(r['st_mae_gt_deg']), '; '.join(comment)))
    lines.append('\n## Preliminary selection advice\n')
    lines.append('- **Local director extraction figure**: geological_strata, collagen_like_tissue, leaf_venation, rod_bacteria (exploratory only), and retina_venation_proxy are the most visually informative.\n')
    lines.append('- **Global orientation / deskew figure**: building_array (exploratory only), brick_wall_open_proxy, textile_weave, and wood_grain provide plausible global-axis tests.\n')
    lines.append('- **Publication-safe candidates right now**: collagen_like_tissue, geological_strata, textile_weave, wood_grain, leaf_venation, brick_wall_open_proxy, retina_venation_proxy, grass_fiber_proxy.\n')
    lines.append('- **Need later source substitution if used in the paper**: rod_bacteria and building_array.\n')
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('saved to', OUT)

if __name__=='__main__':
    run()
