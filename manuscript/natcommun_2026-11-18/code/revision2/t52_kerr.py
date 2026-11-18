"""Real Kerr-microscopy bubble-domain growth sequences (KerrPy repository, S. Guddeti, GPL-3.0; two pulse protocols,
nine frames each). Difference images against the saturated reference; per frame: area and perimeter of the growing domain by
(a) median 5x5 + Otsu + open/close (KerrPy-style), (b) Gaussian sigma=2 + Otsu + marching squares, (c) PBGIA binary (single
cut, selected 2D readout), (d) PBGIA continuous source (gray readout). Growth under identical pulses is smooth, so the residual
from a smooth fit measures each method's frame-to-frame noise."""
import sys,glob,json,numpy as np,pandas as pd,scipy.ndimage as ndi
from PIL import Image
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from gray import gen_gray
from t38_corek import est as est_gray, feats
from t22_select import gen
from kern import tv
from skimage.filters import threshold_otsu
from skimage.measure import perimeter_crofton,find_contours
import t23_aniso as A
A.NH=6; L=json.load(open('/home/claude/work/data/t25_law_h1.5.json')); G=json.load(open('/home/claude/work/data/t38_gray_selected.json'))
UM=100/310.   # micrometres per pixel (100 um scale bar = 310 px)
def load(f): return np.asarray(Image.open(f)).astype(float)[:1020]
def largest(b):
    lab,n=ndi.label(b)
    if n==0: return b
    k=np.argmax(ndi.sum(b,lab,range(1,n+1)))+1; return lab==k
def ms_len(a,level):
    return sum(np.hypot(*np.diff(c,axis=0).T).sum() for c in find_contours(a,level))
def masked_readout(s,region,law):
    if law=='gray':
        X,k=feats(s); c,b,km=G['c'],G['b'],G['km']
    else:
        n_,k=A.normals(s); X=A.feat(2,n_); c,b,km=L['c'],L['b'],A.KM[2]
    mp=c[0]+X@np.array(c[1:]); kc=np.clip(k,0,km); return float(np.sum((tv(s,0)/(mp+b[0]*kc+b[1]*kc**2+b[2]*kc**3))[region]))
from scipy.stats import norm
def snr_rule(D):
    """Per-pixel contrast-to-noise ratio and the smallest Gaussian pre-smoothing sigma for which the mid-level cut
    flips at most p*=0.3% of plateau pixels (isolated flips, removable by the sign rule): Phi(-SNR_sigma/2)<=p*,
    with SNR_sigma = SNR_1 * 2 sqrt(pi) sigma for white noise."""
    s2=ndi.gaussian_filter(D,3.0); t=threshold_otsu(s2); lo=np.median(D[s2<t]); hi=np.median(D[s2>=t])
    hp=D-ndi.gaussian_filter(D,2.0); noise=1.4826*np.median(np.abs(hp-np.median(hp)))/np.sqrt(1-1/(4*np.pi*4)*0-0.0)  # high-pass noise estimate
    noise=np.std(D[s2>=t][:200000]) if True else noise
    snr1=abs(hi-lo)/noise; need=2*norm.isf(0.003); sig=max(0.8,np.ceil(2*need/(snr1*2*np.sqrt(np.pi)))/2)
    return lo,hi,snr1,sig
def ring(comp,w=4):
    comp=ndi.binary_fill_holes(comp); return ndi.binary_dilation(comp,iterations=w)&~ndi.binary_erosion(comp,iterations=w)
def analyze(D,lev):
    out={}; lo,hi,snr1,sig=lev; out['_snr']=(snr1,sig)
    md=ndi.median_filter(D,5); b=largest(md<threshold_otsu(md)); b=ndi.binary_closing(ndi.binary_opening(b,iterations=2),iterations=2); b=largest(b)
    out['median_otsu']=(b.sum(),perimeter_crofton(b,4))
    gs=ndi.gaussian_filter(D,sig); t=(lo+hi)/2; comp=ndi.binary_fill_holes(largest(gs<t))
    cs=find_contours(np.where(ndi.binary_dilation(comp,iterations=3),gs,hi),t); c=max(cs,key=len)
    out['gauss_ms']=(comp.sum(),np.hypot(*np.diff(c,axis=0).T).sum())
    g=np.clip((gs-hi)/(lo-hi),0,1)
    sb=gen(2,g>0.5,1.0,1.5); comp=largest(sb>0); out['PBGIA_bin']=(float(np.sum(((1+sb)/2)[ndi.binary_dilation(comp,iterations=4)])),masked_readout(sb,ring(comp),'bin'))
    sc=gen_gray(2*g-1,h=G['h']); comp=largest(sc>0); out['PBGIA_cont']=(float(np.sum(((1+sc)/2)[ndi.binary_dilation(comp,iterations=4)])),masked_readout(sc,ring(comp),'gray'))
    return out,sc,g
rows=[]; keep={}
for si,d in enumerate(sorted(glob.glob('/home/claude/baalkikhaal/kerrpy/DataSimplified/*/*/'))):
    ref=load(d+'INITIALUP SATURATION.png'); fl=sorted(glob.glob(d+'1*_*.png'))
    Dl=load(fl[-1])-ref; lev=snr_rule(Dl-np.median(Dl))   # contrast and noise are properties of the optics: estimated once per sequence (largest domain)
    for f in fl:
        D=load(f)-ref; D-=np.median(D); o,sc,g=analyze(D,lev); fr=int(f.split('/')[-1].split('_')[0])
        snr1,sig=o.pop('_snr')
        for m,(a,p) in o.items(): rows.append(dict(seq=si,frame=fr,method=m,area_um2=a*UM*UM,perim_um=p*UM,snr1=snr1,sigma=sig))
        if fr in (12,18): keep[f'{si}_{fr}']=(D,sc,g)
        print(si,fr,{m:(round(a*UM*UM),round(p*UM,1)) for m,(a,p) in o.items()},flush=True)
df=pd.DataFrame(rows); df.to_csv('/home/claude/work/data/t52_kerr.csv',index=False)
np.savez_compressed('/home/claude/work/data/t52_kerr_fields.npz',**{k+'_'+n:v for k,(D,sc,g) in keep.items() for n,v in (('D',D),('s',sc),('g',g))})
# smoothness: residual of a cubic fit in frame index, relative to the mean
res=[]
for (si,m),q in df.groupby(['seq','method']):
    for col in ('area_um2','perim_um'):
        x=q.frame.values; y=q[col].values; c=np.polyfit(x,y,3); r=y-np.polyval(c,x)
        res.append(dict(seq=si,method=m,quantity=col,rel_resid_pct=100*np.sqrt(np.mean(r**2))/np.mean(y)))
R=pd.DataFrame(res); R.to_csv('/home/claude/work/data/t52_kerr_smooth.csv',index=False)
print(R.pivot_table(index=['quantity','method'],columns='seq',values='rel_resid_pct').round(2).to_string())
