"""Director field from colour textures: which scalars feed the threshold stack?
Synthetic: curved strata (analytic tangent) whose stripes are carried by luminance in the left half and by an
isoluminant chroma modulation in the right half; iid channel noise. Error = |q|-weighted mean axial error per half.
Real CC0 colour images: rotation consistency of the global axis (as Table multithresh)."""
import sys,json,numpy as np,pandas as pd,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code','/home/claude/work/manuscript_v2/code/revision2']
from real_image_examples import generate_field,axis_mean,angle_delta,img_rotate,roi_center
from gray import gen_gray
from skimage import data,transform
from multiprocessing import Pool
W=np.array([.299,.587,.114]); QS=(.15,.325,.5,.675,.85); SIG=8
def tens_cuts(a,smooth=.8):
    sm=ndi.gaussian_filter(a,smooth); acc=np.zeros((3,)+a.shape); n=0
    for q in np.unique(np.quantile(sm,QS)):
        m=sm>q
        if m.mean()<.005 or m.mean()>.995: continue
        vr,vc,s=generate_field(m,nsteps=30); acc+=np.stack([vr*vr,vc*vc,vr*vc]); n+=1
    return acc/max(n,1),n
def tens_cont(a,smooth=.8):
    sm=ndi.gaussian_filter(a,smooth); lo,hi=np.quantile(sm,(.02,.98)); g=np.clip((sm-lo)/(hi-lo),0,1)
    vr,vc,s=gen_gray(2*g-1,h=2.0,full=True); return np.stack([vr*vr,vc*vc,vr*vc]),1
def st(chs,pre=1.2):
    acc=0
    for a in chs:
        gr,gc=np.gradient(ndi.gaussian_filter(a,pre)); acc=acc+np.stack([gr*gr,gc*gc,gr*gc])
    return acc
def director(T,sig=SIG):
    rr,cc,rc=[ndi.gaussian_filter(x,sig,mode='reflect') for x in T]; gap=np.hypot(cc-rr,2*rc)
    return ((.5*np.degrees(np.arctan2(-2*rc,cc-rr)))%180+90)%180, gap, gap/(rr+cc+1e-12)
def scalars(I):
    X=I.reshape(3,-1); Xc=X-X.mean(1,keepdims=True); ev,U=np.linalg.eigh(Xc@Xc.T)
    R,G_,B=I; return dict(Y=[np.tensordot(W,I,1)],RGB=[R,G_,B],OPP=[np.tensordot(W,I,1),R-G_,(R+G_)/2-B],PCA1=[np.tensordot(U[:,-1],I,1)])
def methods(I):
    S=scalars(I); out={}
    for name in ('Y','RGB','OPP','PCA1'):
        Ts=[tens_cuts(a)[0] for a in S[name]]; out[f'{name}_cuts']=sum(Ts)/len(Ts)
    out['Y_cont']=tens_cont(S['Y'][0])[0]
    out['ST_Y']=st(S['Y']); out['ST_color']=st(S['RGB'])        # Di Zenzo colour structure tensor = sum over channels
    return out
def synth(sd,seed=19,size=320,amp=22.,period=190.,stripe=17.):
    y,x=np.mgrid[:size,:size]; k=2*np.pi/period; ph=y-amp*np.sin(k*x); m=.35*np.sin(2*np.pi*ph/stripe)
    truth=(-np.degrees(np.arctan(amp*k*np.cos(k*x))))%180
    rng=np.random.default_rng(seed); dl=np.array([1.,1.,1.]); di=rng.normal(size=3); di-=W*(W@di)/(W@W); di/=np.abs(di).max()
    left=(x<size/2)[None]; I=0.5+np.where(left,m[None]*dl[:,None,None],m[None]*di[:,None,None])
    return I+rng.normal(0,sd,I.shape),truth
def syn_job(sd):
    I,truth=synth(sd); M=methods(I); rows=[]; x=np.arange(320)[None,:]*np.ones((320,1))
    band=(np.abs(x-160)>20)&(x>20)&(x<300)&(np.arange(320)[:,None]>20)&(np.arange(320)[:,None]<300)
    for name,T in M.items():
        tang,gap,coh=director(T)
        for half,msk in (('lum (left)',band&(x<160)),('isolum (right)',band&(x>160))):
            w=gap*msk; e=np.abs(angle_delta(tang,truth)); rows.append(dict(noise=sd,method=name,half=half,err=float(np.sum(w*e)/np.sum(w)),coh=float(np.sum(w*coh)/np.sum(w))))
    return rows
def rot_job(item):
    name,img=item; rows=[]; res={}
    for r in (-15,-10,-5,0,5,10,15):
        a=np.stack([img_rotate(img[c],r) for c in range(3)]); M=methods(a); roi=roi_center(a.shape[1:],.67)
        for k,T in M.items():
            tang,gap,_=director(T); res.setdefault(k,[]).append((r,axis_mean(tang,gap,roi)[0]))
    for k,v in res.items():
        a0=[x for x in v if x[0]==0][0][1]; rows.append(dict(image=name,method=k,rot_err=float(np.mean([abs(angle_delta(angle_delta(a_,a0),r)) for r,a_ in v]))))
    return rows
def load(a):
    a=transform.resize(np.asarray(a,float)[...,:3]/255.,(320,320),anti_aliasing=True); return np.moveaxis(a,-1,0)
if __name__=='__main__':
    with Pool(2) as pool:
        syn=[r for rr in pool.map(syn_job,[0.0,0.05,0.1]) for r in rr]
    d=pd.DataFrame(syn); d.to_csv('data/t45_color_director_synth.csv',index=False)
    print(d.pivot_table(index=['method'],columns=['half','noise'],values='err').round(2).to_string())
    imgs=[('immunohistochemistry',load(data.immunohistochemistry())),('coffee',load(data.coffee())),('chelsea',load(data.chelsea())),
          ('retina',load(data.retina())),('rocket',load(data.rocket()))]
    with Pool(2) as pool: rr=[r for x in pool.map(rot_job,imgs) for r in x]
    e=pd.DataFrame(rr); e.to_csv('data/t45_color_director_rot.csv',index=False); print(e.pivot(index='method',columns='image',values='rot_err').round(2).to_string())
