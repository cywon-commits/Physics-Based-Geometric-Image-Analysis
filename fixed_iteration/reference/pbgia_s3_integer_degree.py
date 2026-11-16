"""Integer degree of a 3D lattice n(x,y,z) in S^3 via regular-value preimage counts.
Each cubic cell is split into six positively oriented Freudenthal tetrahedra.
Mapping of each tetra is n=(sum barycentric*n_vertex)/norm; count positive-ray intersections with generic target p.
Only valid when interpolated tetrahedra have nonzero norm and boundary is fixed, with no singular tetrahedra.
"""
import numpy as np
from itertools import permutations


def degree(n, target=(.37139,-.52177,.64829,.40981), boundary_report=True, verbose=False):
    n=np.asarray(n,dtype=np.float64)
    target=np.array(target,dtype=np.float64);target/=np.linalg.norm(target)
    shape=np.array(n.shape[:3]);assert n.shape[-1]==4
    # choose cube origin and edges; split 6 tetra using 6 permutations
    origin=np.indices(tuple(shape-1),dtype=np.int32).reshape(3,-1).T
    deg=0;total_hits=0;near_singular=0; minimum_margin=np.inf;details=[]
    for perm in permutations(range(3)):
        offsets=np.zeros((4,3),dtype=int)
        for j,axis in enumerate(perm):offsets[j+1]=offsets[j];offsets[j+1,axis]+=1
        # sorted permutation sign of columns gives orientation
        P=np.eye(3,dtype=int)[:,list(perm)]
        orientation=int(round(np.linalg.det(P)))
        vertices=np.stack([n[tuple((origin+off).T)] for off in offsets],axis=1)
        # fast coarse screening: target in positive cone implies each coordinate sign compatible
        possible=np.ones(vertices.shape[0],dtype=bool)
        for a in range(4):
            if target[a]>1e-8: possible &= (vertices[:,:,a].max(axis=1)>0)
            elif target[a]<-1e-8: possible &= (vertices[:,:,a].min(axis=1)<0)
        selected=vertices[possible]
        if not len(selected):continue
        M=selected.transpose(0,2,1)
        determinants=np.linalg.det(M)
        nonsingular=np.abs(determinants)>1e-10
        near_singular+=int(np.sum(~nonsingular))
        M=M[nonsingular];determinants=determinants[nonsingular]
        sol=np.linalg.solve(M,np.broadcast_to(target,(len(M),4))[...,None])[...,0]
        ishit=(sol.min(axis=1)>1e-8)
        hits=int(np.sum(ishit)); total_hits+=hits
        if hits:
            signs=np.sign(determinants[ishit])*orientation
            contribution=int(np.sum(signs));deg+=contribution
            minimum_margin=min(minimum_margin,float(sol[ishit].min()))
            details.append((perm,hits,contribution))
    if boundary_report:
        border=np.concatenate([n[0].reshape(-1,4),n[-1].reshape(-1,4),n[:,0].reshape(-1,4),n[:,-1].reshape(-1,4),n[:,:,0].reshape(-1,4),n[:,:,-1].reshape(-1,4)])
        bd_spread=float(np.max(np.linalg.norm(border-border.mean(axis=0),axis=1)))
    else:bd_spread=None
    if verbose:print('degree',deg,'hits',total_hits,'margin',minimum_margin,'near singular tets',near_singular,'boundary spread',bd_spread,'details',details,flush=True)
    return {'integer_degree':deg,'number_of_preimages':total_hits,'near_degenerate_tetrahedra':near_singular,'min_barycentric_positive_coefficient':minimum_margin,'boundary_spread':bd_spread}

if __name__=='__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0,'/mnt/data')
    import pbgia_3d_validation as model
    for shape,p,shift in [('sphere',{'R':9},(0,0,0)),('capsule',{'R':7,'L':16,'axis':[.65,.6,.47]},(-.25,.19,-.32)),('torus',{'R':12,'r':4.5},(.15,-.3,.2))]:
        d,f,_=model.field(shape,p,shift)
        init=model.init(d)
        print(shape,'initial analytic',degree(init,verbose=True),flush=True)
        n,parts,rms,vals=model.evolve(d,f,steps=250)
        print(shape,'relaxed',degree(n,verbose=True),'finite-diff',model.q3(n),'rms',rms,flush=True)
