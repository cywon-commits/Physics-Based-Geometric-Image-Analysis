"""Real Kerr image of a labyrinth domain pattern (TU Wien extended-depth-of-field MOKE data set, CC-BY 4.0; Figure 6,
10 degree tilt: raw crop and the authors' binarization).
Independent morphological ground truth for director defects: in a stripe pattern, a stripe end is a +1/2 disclination of the
stripe director and a three-fold branch point is a -1/2 disclination, in either phase. We count both on pruned skeletons of
the authors' binary image and compare with the plaquette charges of the PBGIA director field."""
import sys,json,numpy as np,scipy.ndimage as ndi,pandas as pd
from PIL import Image
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code','/home/claude/work/manuscript_v2/code/revision2']
from gray import gen_gray
from t38_corek import feats
from kern import tv
from dirscale import qfield,defect_list
from skimage.morphology import skeletonize
from skimage.measure import perimeter_crofton,find_contours
G=json.load(open('/home/claude/work/data/t38_gray_selected.json'))
raw=np.asarray(Image.open('/home/claude/work/data/labyrinth/raw10.png')).astype(float); raw=raw[...,:3].mean(-1) if raw.ndim==3 else raw
B=np.asarray(Image.open('/home/claude/work/data/labyrinth/bin10.png')); B=(B[...,0] if B.ndim==3 else B)>127
n=raw.shape[0]; M=12; inner=np.zeros((n,n),bool); inner[M:-M,M:-M]=True
# period from the radial power spectrum
F=np.abs(np.fft.fft2(raw-raw.mean()))**2; ky,kx=np.meshgrid(np.fft.fftfreq(n),np.fft.fftfreq(n),indexing='ij'); kr=np.hypot(ky,kx)
rb=np.arange(0.01,0.4,0.0025); pw=[F[(kr>=a)&(kr<a+0.0025)].mean() for a in rb]; LAM=float(1/(rb[int(np.argmax(pw))]+0.00125))
# PBGIA continuous source (raw image, light smoothing, 2-98% normalization)
sm=ndi.gaussian_filter(raw,0.8); lo,hi=np.quantile(sm,(.02,.98)); g=np.clip((sm-lo)/(hi-lo),0,1)
c=np.median(g); vr,vc,s=gen_gray(np.clip(2*(g-c)/max(c,1-c),-1,1),h=G['h'],full=True)
P=s>0
X,k=feats(s); cc,bb,km=G['c'],G['b'],G['km']; mp=cc[0]+X@np.array(cc[1:]); kcl=np.clip(k,0,km)
L_pbgia=float(np.sum((tv(s,0)/(mp+bb[0]*kcl+bb[1]*kcl**2+bb[2]*kcl**3))[inner]))
A_in=float(inner.sum())
L_crof=perimeter_crofton(B[M:-M,M:-M],4)        # includes the window edge contribution; remove it below
edge=0.0
def wall_len_ms(img,level):
    return sum(np.hypot(*np.diff(c_,axis=0).T).sum() for c_ in find_contours(img[M:-M,M:-M].astype(float),level))
L_ms_bin=wall_len_ms(ndi.gaussian_filter(B.astype(float),0.75),0.5)
agree=float(np.mean((P==B)[inner])); agree_flip=float(np.mean((P!=B)[inner])); polarity=1 if agree>=agree_flip else -1
# ---- skeleton features (both phases), pruned of spurs shorter than LAM/2
K8=np.ones((3,3)); K8[1,1]=0
def nbcount(sk): return ndi.convolve(sk.astype(int),K8,mode='constant')*sk
def prune(sk,lmin):
    sk=sk.copy()
    for _ in range(3):
        nb=nbcount(sk); ends=list(zip(*np.nonzero((nb==1)&sk))); changed=False
        for e in ends:
            path=[e]; cur=e; prev=None
            while True:
                r,c_=cur; nbrs=[(r+dr,c_+dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr or dc) and 0<=r+dr<n and 0<=c_+dc<n and sk[r+dr,c_+dc] and (r+dr,c_+dc)!=prev and (r+dr,c_+dc) not in path]
                if len(nbrs)!=1: break
                prev=cur; cur=nbrs[0]
                if nbcount(sk[max(cur[0]-1,0):cur[0]+2,max(cur[1]-1,0):cur[1]+2]).max()>=0 and ndi.convolve(sk[max(cur[0]-1,0):cur[0]+2,max(cur[1]-1,0):cur[1]+2].astype(int),np.ones((3,3)),mode='constant').max()>0:
                    pass
                if sum(sk[cur[0]+dr,cur[1]+dc] for dr in (-1,0,1) for dc in (-1,0,1) if (dr or dc) and 0<=cur[0]+dr<n and 0<=cur[1]+dc<n)>=3: break
                path.append(cur)
                if len(path)>lmin: break
            if len(path)<=lmin and len(nbrs)!=0:
                for p_ in path: sk[p_]=False
                changed=True
        if not changed: break
    return skeletonize(sk)
def features(phase):
    sk=prune(skeletonize(phase),int(LAM/2)); nb=nbcount(sk)
    ends=np.array(list(zip(*np.nonzero((nb==1)&sk)))).reshape(-1,2).astype(float)
    jl,nj=ndi.label((nb>=3)&sk,structure=np.ones((3,3))); junc=np.array(ndi.center_of_mass(np.ones_like(nb),jl,range(1,nj+1))).reshape(-1,2)
    return sk,ends,junc
truth=[]
for ph in (B,~B):
    sk,e,j=features(ph); truth+= [(r,c_,+.5) for r,c_ in e]+[(r,c_,-.5) for r,c_ in j]
truth=[t for t in truth if M+4<=t[0]<n-M-4 and M+4<=t[1]<n-M-4]
def match(defs,rad):
    used=set(); tp={+.5:0,-.5:0}
    for r0,c0,k0 in truth:
        best=None
        for i,(r,c_,k_) in enumerate(defs):
            if i in used or k_!=k0: continue
            d=np.hypot(r-r0,c_-c0)
            if d<=rad and (best is None or d<best[0]): best=(d,i)
        if best: used.add(best[1]); tp[k0]+=1
    return tp
gs=np.hypot((np.roll(s,-1,0)-np.roll(s,1,0))/2,(np.roll(s,-1,1)-np.roll(s,1,1))/2); nrm=vr*vr+vc*vc+1e-12
RAWS={'PBGIA_cont_w':np.stack([gs*vr*vr/nrm,gs*vc*vc/nrm,gs*vr*vc/nrm]),'PBGIA_cont':np.stack([vr*vr,vc*vc,vr*vc])}
gr,gc=np.gradient(ndi.gaussian_filter(raw,1.0)); RAWS['ST']=np.stack([gr*gr,gc*gc,gr*gc])
rows=[]
Np=sum(1 for t in truth if t[2]>0); Nm=len(truth)-Np
for m,R in RAWS.items():
    for sig in (1.0,1.5,2.0,2.5,3.0,4.0):
        q,_=qfield(R,sig); d=defect_list(q,margin=M+4); tp=match(d,LAM/2)
        npl=sum(1 for z in d if z[2]>0); nmi=len(d)-npl
        rows.append(dict(method=m,sigma=sig,n_plus=npl,n_minus=nmi,truth_plus=Np,truth_minus=Nm,tp_plus=tp[+.5],tp_minus=tp[-.5],
                         recall=(tp[+.5]+tp[-.5])/max(len(truth),1),precision=(tp[+.5]+tp[-.5])/max(len(d),1)))
R=pd.DataFrame(rows); print(R.round(3).to_string())
res=dict(period_fft_px=LAM,pixel_agreement_with_authors_binary=max(agree,agree_flip),
         wall_density_pbgia=L_pbgia/A_in,period_from_wall_density_pbgia=2*A_in/L_pbgia,
         period_from_wall_density_crofton_authors=2*A_in/L_crof,period_from_wall_density_ms_authors=2*A_in/L_ms_bin,
         n_truth_plus=Np,n_truth_minus=Nm)
print(json.dumps(res,indent=1)); json.dump(res,open('/home/claude/work/data/t53_labyrinth.json','w'),indent=1); R.to_csv('/home/claude/work/data/t53_labyrinth_defects.csv',index=False)
np.savez('/home/claude/work/data/t53_labyrinth.npz',raw=raw,B=B,s=s,vr=vr,vc=vc,truth=np.array(truth),gs=gs)
