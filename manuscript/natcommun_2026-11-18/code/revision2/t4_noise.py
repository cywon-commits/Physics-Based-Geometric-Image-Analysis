import numpy as np, csv, scipy.ndimage as ndi
from skimage.measure import perimeter_crofton
from common import generate_field, tv_per
# (a) local stability rules: isolated features in uniform background, h=2,J=1,D=1, N=30
n=40; feats={}
def put(cells):
    m=np.zeros((n,n),bool)
    for r,c in cells: m[20+r,20+c]=True
    return m
feats['single']=put([(0,0)]); feats['domino']=put([(0,0),(0,1)])
feats['L-tromino']=put([(0,0),(0,1),(1,0)]); feats['2x2']=put([(0,0),(0,1),(1,0),(1,1)])
feats['line1x9']=put([(0,k) for k in range(9)]); feats['plus']=put([(0,0),(1,0),(-1,0),(0,1),(0,-1)])
feats['3x3']=put([(a,b) for a in (-1,0,1) for b in (-1,0,1)])
rule=[]
for h in (1.0,2.0,3.0,4.5):
    for k,m in feats.items():
        for inv in (False,True):  # object (+) in (-) background, or hole (-) in (+)
            mm=~m if inv else m
            vr,vc,s=generate_field(mm,nsteps=30,h=h)
            region=m; kept=np.mean((s[region]>0)==(mm[region]))
            rule.append(dict(h=h,feature=k,hole=inv,npx=int(m.sum()),fraction_kept=kept))
    print(h,[(r['feature'],round(r['fraction_kept'],2)) for r in rule[-14:] if not r['hole']],flush=True)
with open('data/t4_local_rules.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rule[0]));w.writeheader();w.writerows(rule)
# (b) salt-and-pepper perimeter test on stadiums
def stadium(n,R,th):
    yy,xx=np.mgrid[:n,:n]; xx=xx-(n-1)/2; yy=-(yy-(n-1)/2); a=np.deg2rad(th); L=2*R
    lo=np.cos(a)*xx+np.sin(a)*yy; tr=-np.sin(a)*xx+np.cos(a)*yy
    return R-np.hypot(np.maximum(np.abs(lo)-L/2,0),tr)>=0
rng=np.random.default_rng(7); rows=[]
for R in (11,15,19,23):
    for th in (0,27,54):
        m0=stadium(160,R,th); P=2*np.pi*R+4*R
        for p in (0.0,0.002,0.01,0.05):
            for rep in range(3 if p>0 else 1):
                flip=rng.random(m0.shape)<p; m=m0^flip
                out=dict(R=R,theta=th,noise=p,rep=rep,P_true=P)
                for N in (0,10,30):
                    vr,vc,s=generate_field(m,nsteps=N); out[f'TV_N{N}']=tv_per(s)
                    out[f'dice_N{N}']=2*np.sum((s>0)&m0)/(np.sum(s>0)+m0.sum())
                out['crofton']=perimeter_crofton(m,4)
                out['crofton_median3']=perimeter_crofton(ndi.median_filter(m.astype(np.uint8),3).astype(bool),4)
                rows.append(out)
    print(R,flush=True)
with open('data/t4_noise.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
