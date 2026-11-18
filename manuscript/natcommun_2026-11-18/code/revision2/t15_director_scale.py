import sys,csv,numpy as np,scipy.ndimage as ndi
sys.path[:0]=['/home/claude/work','/home/claude/work/manuscript_v2/code']
from dirscale import *
import multithreshold_directors as mt
from real_image_examples import img_rotate
rng=np.random.default_rng(5); n=320; y,x=np.mgrid[:n,:n]+0.0
# ---- synthetic texture with known disclinations (image coords: x=col, y=row downward; angle from +x toward -y (up))
truth=[(110,100,+.5),(110,140,-.5),(220,110,+.5),(220,220,-.5)]   # (row,col,k): close pair d=40, far pair d=110
psi=np.full((n,n),0.0)
for r0,c0,k in truth: psi+=k*np.arctan2(-(y-r0),x-c0)
def lic(psi,L=14,noise_sigma=0.9):
    nz=ndi.gaussian_filter(rng.standard_normal((n,n)),noise_sigma); out=np.zeros((n,n)); cnt=0
    for sgn in (1,-1):
        py,px=y.copy(),x.copy(); dy=-np.sin(psi)*sgn; dx=np.cos(psi)*sgn
        for t in range(L):
            out+=ndi.map_coordinates(nz,[py,px],order=1,mode='reflect'); cnt+=1
            a=ndi.map_coordinates(psi,[py,px],order=0,mode='nearest'); ty,tx=-np.sin(a),np.cos(a)
            flip=np.sign(ty*dy+tx*dx); flip[flip==0]=1; dy,dx=ty*flip,tx*flip; py=py+dy; px=px+dx
    o=out/cnt; o=(o-o.mean())/o.std(); return np.clip(0.5+0.18*o,0,1)
img=lic(psi); np.save('data/t15_lic_img.npy',img); np.save('data/t15_lic_psi.npy',psi)
raw=raw_tensor(img); rawst=st_raw(img)
sigmas=[1.5,2,3,4,6,8,10,12,14,16,18,20,24,28,32,40,50]
rows=[]
def match(defs,truth,rad):
    used=set(); hit=0
    for r0,c0,k0 in truth:
        for i,(r,c,k) in enumerate(defs):
            if i not in used and k==k0 and np.hypot(r-r0,c-c0)<=rad: used.add(i); hit+=1; break
    return hit
excl=np.ones((n,n),bool)
for r0,c0,_ in truth: excl&=np.hypot(y-r0,x-c0)>20
inner=(y>24)&(y<n-24)&(x>24)&(x<n-24)&excl
for s in sigmas:
    for name,R in (('PBGIA',raw),('ST',rawst)):
        q,tr=qfield(R,s); d=defect_list(q,margin=24)
        tan=tangent_deg(q); gt=(np.degrees(psi))%180; err=np.abs((tan-gt+90)%180-90)
        rows.append(dict(method=name,sigma=s,n_plus=sum(1 for z in d if z[2]>0),n_minus=sum(1 for z in d if z[2]<0),net=sum(z[2] for z in d),
                         matched=match(d,truth,max(6,s)),local_err=float(np.mean(err[inner]))))
    print(s,rows[-2],rows[-1]['matched'],flush=True)
with open('data/t15_lic_sigma.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
np.save('data/t15_lic_raw.npy',raw)
