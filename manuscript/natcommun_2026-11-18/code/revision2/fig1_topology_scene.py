import sys,numpy as np,scipy.ndimage as ndi
sys.path[:0]=['.','../code/revision2','../code','/home/claude/work']
import new_quantitative_studies as nq
n=46; y,x=np.mgrid[:n,:n].astype(float)
disk=np.hypot(x-11,y-12)<6.5
ann=(np.hypot(x-33,y-13)<8.5)&(np.hypot(x-33,y-13)>3.8)
two=((np.abs(x-23)<15)&(np.abs(y-33.5)<7.5))&~(np.hypot(x-16,y-33.5)<3.4)&~(np.hypot(x-30,y-33.5)<3.4)
m=disk|ann|two
vr,vc,s=nq.generate_field(m,nsteps=30); vr,vc,s=[a.astype(float) for a in (vr,vc,s)]
print('Q total',nq.q2_solid_angle(vr,vc,s))
nn=np.stack([vc,-vr,s],-1); a=nn[:-1,:-1];b=nn[:-1,1:];c=nn[1:,1:];d=nn[1:,:-1]
def om(p,q,r):
    num=np.einsum('...i,...i->...',p,np.cross(q,r)); den=1+(p*q).sum(-1)+(q*r).sum(-1)+(r*p).sum(-1); return 2*np.arctan2(num,den)
dens=-(om(a,b,c)+om(a,c,d))/(4*np.pi)
lab,nl=ndi.label(m); _,ind=ndi.distance_transform_edt(lab==0,return_indices=True); vor=lab[ind[0],ind[1]][:-1,:-1]
for k in range(1,nl+1): print(k,dens[vor==k].sum())
np.savez('/tmp/p/topo.npz',m=m,vr=vr,vc=vc,s=s)
