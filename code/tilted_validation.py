from run_research import *
from topology_audit import certificate,cap,q3

def run():
 rows=[];axis=np.array([.65,.60,.47]);axis/=np.linalg.norm(axis);N=56
 xyz=np.indices((N,)*3,dtype=np.float32)-(N-1)/2
 for k,sh in enumerate((.15,-.3,.2)):xyz[k]-=sh
 u=np.einsum('i,i...->...',axis,xyz);radial=xyz-axis[:,None,None,None]*u;rho=np.linalg.norm(radial,axis=0)
 for i,(kind,R,r) in enumerate([('torus',12,4.5),('torus',14,3),('capsule',7,0)]):
  if kind=='torus':d=r-np.sqrt((rho-R)**2+u*u);normal=radial*(1-R/np.maximum(rho,1e-9))+axis[:,None,None,None]*u
  else:d=R-np.sqrt(rho*rho+np.maximum(abs(u)-R,0)**2);normal=xyz-axis[:,None,None,None]*np.clip(u,-R,R)
  normal=np.moveaxis(normal,0,-1);normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-9);mask=d>=0
  n,info,hist=solve(mask,dict(h=2,D=1,J=1),steps=1200)
  for t in [(.37139,-.52177,.64829,.40981),(-.6,.2,.3,.7),(.2,.8,-.5,-.2)]:rows.append(dict(shape=kind,R=R,r=r,N=N,axis=str(axis),target=str(t),h=2,D=1,J=1,**info,**metrics(n,mask,normal,d),**certificate(n),**degree(cap(n),t),Q3_fd=q3(n)))
  np.savez_compressed(OUT/f'tilted_{i}_field.npz',n=n,mask=mask);pd.DataFrame(rows).to_csv(OUT/'tilted_validation.csv',index=False)
 print('TILTED DONE',flush=True)
if __name__=='__main__':run()
