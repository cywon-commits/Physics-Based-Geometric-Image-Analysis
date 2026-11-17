"""Independent GT, coordinate, frozen-generator and orientation-identifiability checks."""
import sys
import json
from pathlib import Path
import itertools
import numpy as np
from numpy.polynomial.legendre import leggauss
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from data import plane_area, fixture, rotation
from engine import initialize, gradient4, step, generate
from readouts import measure
import operators as previous


def test_exact_plane_square_axis_and_diagonal_sections():
    side=12.3
    assert np.isclose(plane_area(2,side,np.array([1.,0]),0),side)
    diagonal=np.array([1.,1.])/np.sqrt(2)
    assert np.isclose(plane_area(2,side,diagonal,0),np.sqrt(2)*side)
    for off in [0.,.371,side/4,-side/4]:
        expected=np.sqrt(2)*side-2*abs(off)
        assert np.isclose(plane_area(2,side,diagonal,off),expected,atol=2e-11)
        assert np.isclose(plane_area(2,side,-diagonal,-off),expected,atol=2e-11)
    assert plane_area(2,side,np.array([1.,0]),side)==0


def test_exact_plane_cube_axis_and_two_diagonal_sections():
    side=9.7
    assert np.isclose(plane_area(3,side,np.array([1.,0,0]),0),side**2)
    diagonal=np.array([1.,1.,0])/np.sqrt(2)
    for off in [0.,.371,-1.2]:
        expected=side*(np.sqrt(2)*side-2*abs(off))
        assert np.isclose(plane_area(3,side,diagonal,off),expected,atol=1e-10)
    tri=np.ones(3)/np.sqrt(3)
    assert np.isclose(plane_area(3,side,tri,0),3*np.sqrt(3)*side**2/4,atol=1e-10)
    # Independent Irwin-Hall convolution formula for a shifted (1,1,1) slice.
    for off in [.371,-.371,2.]:
        u=1.5*side-np.sqrt(3)*off
        expected=np.sqrt(3)/2*sum((-1)**k*c*max(u-k*side,0)**2
                                  for k,c in enumerate([1,3,3,1]))
        assert np.isclose(plane_area(3,side,tri,off),expected,atol=1e-10)
    assert plane_area(3,side,np.array([1.,0,0]),side)==0


def _plane_row(d,spacing,rotation_id=0,shift_id=1):
    return dict(dim=d,family='plane',R=np.nan,spacing=spacing,size=20,
                params=json.dumps(dict(oracle=True,roi_side=8)),width=1.6,
                rotation_id=rotation_id,shift_id=shift_id,rotation_seed=1234,
                seed=4321,noise_seed=999,noise=0.)


def test_plane_oracle_uses_physical_width_roi_and_shift():
    for d,rot in [(2,2),(3,2)]:
        for spacing in [1.,.5]:
            row=_plane_row(d,spacing,rot)
            f,n,truth,axis,roi,meta=fixture(row)
            assert f is None
            assert np.allclose(np.linalg.norm(n,axis=-1),1,atol=2e-15)
            S=n.shape[0];x=(np.indices((S,)*d)-(S-1)/2)*spacing
            level=np.einsum('i,i...->...',axis,x)+.371
            assert np.allclose(n[...,-1],np.tanh(level/1.6),atol=1e-14)
            assert all((s.stop-s.start)*spacing==8 for s in roi)
            assert np.isclose(truth,plane_area(d,8,axis,.371))
            assert meta['axis_role']=='plane_normal'
            assert meta['normal_target_defined']
    # These normals and ROI represent the same physical patch at both spacings.
    a=fixture(_plane_row(3,1.,2))[2]
    b=fixture(_plane_row(3,.5,2))[2]
    assert a==b


def test_frozen_g4_generation_matches_previous_physics_and_preserves_f():
    for d in [2,3]:
        x=np.indices((20,)*d)-(20-1)/2
        f=np.where(np.linalg.norm(x,axis=0)<4.7,1.,-1.)
        before=f.copy();n=initialize(f);old=previous.initialize(f.astype('float32'))
        config=previous.parameter_config('axial',[-1/12,0,0])
        for _ in range(10):
            n=step(n,f)
            old=previous.step(old,f.astype('float32'),config)
        assert np.allclose(n,old,atol=3e-6)
        generated,meta=generate(f,N=10)
        assert np.array_equal(generated,n)
        assert np.array_equal(f,before)
        assert np.max(abs(np.linalg.norm(generated,axis=-1)-1))<1e-12
        assert meta['field_seconds']>=0 and meta['initialization_seconds']>=0
        # The physical derivative response changes only by 1/a in the field map.
        state=initialize(f);spacing=.7
        H=sum(np.roll(state,k,j) for j in range(d) for k in [-1,1])
        H[...,:d]+=previous.generated_gradient(state[...,-1],[-1/12,0,0])/spacing
        H[...,-1]+=2*f
        expected=H/np.linalg.norm(H,axis=-1,keepdims=True)
        assert np.allclose(step(state,f,spacing),expected,atol=1e-14)


def test_zero_effective_field_keeps_previous_state_and_unit_norm():
    for d in [2,3]:
        shape=(9,)*d;n=np.zeros(shape+(d+1,));n[...,-1]=1;center=(4,)*d
        n[center]=np.r_[1.,np.zeros(d)]
        amplitude=np.sqrt(1-1/d**2)
        for axis in range(d):
            for sign in [-1,1]:
                pos=list(center);pos[axis]+=sign;pos=tuple(pos)
                n[pos]=np.r_[sign*amplitude,np.zeros(d-1),-1/d]
        f=np.ones(shape);before=f.copy();result=step(n,f)
        assert np.array_equal(result[center],n[center])
        assert np.array_equal(f,before)
        assert np.allclose(np.linalg.norm(result,axis=-1),1,atol=1e-14)


def test_unit_field_energy_contains_phase_and_orientation_bending_terms():
    S=96;a=.02;x=(np.arange(S)-(S-1)/2)*a
    theta=.7+.12*x[:,None];phi=.3*x[None,:]
    theta=np.broadcast_to(theta,(S,S));phi=np.broadcast_to(phi,(S,S))
    n=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],-1)
    roi=(slice(24,72),slice(24,72))
    rows={r['method']:r for r in measure(n,a,roi,include_orientation=False,include_marching=False)}
    phase=np.full((S,S),.12**2)
    bending=np.sin(theta)**2*.3**2
    expected=float((phase[roi]+bending[roi]).sum()*a*a)
    assert np.isclose(rows['energy']['value'],expected,rtol=2e-9)
    # Scalar gradient energy does not equal the full phase energy either.
    expected_scalar=float((np.sin(theta[roi])**2*.12**2).sum()*a*a)
    assert np.isclose(rows['scalar_energy']['value'],expected_scalar,rtol=2e-9)
    assert rows['energy']['value']>float(phase[roi].sum()*a*a)


def test_equal_two_spheres_normal_tensor_cannot_identify_separation_axis():
    radius=2.7;separation=13.1
    z,w=leggauss(20);phi=np.arange(48)*2*np.pi/48
    rr=np.sqrt(1-z[:,None]**2)
    normals=np.stack([np.broadcast_to(rr*np.cos(phi),(len(z),len(phi))),
                      np.broadcast_to(rr*np.sin(phi),(len(z),len(phi))),
                      np.broadcast_to(z[:,None],(len(z),len(phi)))],-1).reshape(-1,3)
    weights=np.broadcast_to((w[:,None]*(2*np.pi/48))*radius**2,(len(z),len(phi))).reshape(-1)
    area=8*np.pi*radius**2
    normal_tensor=2*(normals.T@(weights[:,None]*normals))
    assert np.allclose(normal_tensor,(area/3)*np.eye(3),atol=1e-12)
    all_covariances=[]
    for axis in [np.array([1.,0,0]),np.array([0.,1,0]),np.array([1.,2.,3.])/np.sqrt(14)]:
        points=np.concatenate([radius*normals-separation/2*axis,
                               radius*normals+separation/2*axis])
        ww=np.tile(weights,2);center=np.sum(ww[:,None]*points,axis=0)/area
        cov=(points-center).T@(ww[:,None]*(points-center))/area
        expected=(radius**2/3)*np.eye(3)+(separation**2/4)*np.outer(axis,axis)
        assert np.allclose(cov,expected,atol=1e-12)
        vec=np.linalg.eigh(cov)[1][:,-1]
        assert abs(vec@axis)>1-1e-13
        all_covariances.append(cov)
    # The normal tensor is identical for differently oriented pairs, whereas
    # positional covariance distinguishes them: this is non-identifiability,
    # not a failure that more precise normal-eigenvector computation can fix.
    assert not np.allclose(all_covariances[0],all_covariances[1])
    assert np.ptp(np.linalg.eigvalsh(normal_tensor))<1e-12


def test_plane_equation_rescaling_preserves_area():
    for d in [2,3]:
        normal=np.arange(1,d+1,dtype=float);normal/=np.linalg.norm(normal)
        reference=plane_area(d,11.2,normal,.371)
        assert np.isclose(plane_area(d,11.2,4*normal,4*.371),reference,rtol=1e-12)
        assert np.isclose(plane_area(d,11.2,-3*normal,-3*.371),reference,rtol=1e-12)


def test_closed_geometry_gt_matches_exact_sphere_shell_capsule_torus_spheroids():
    R=3.4
    def shape(d,family,params):
        return fixture(dict(dim=d,family=family,R=R,spacing=1.,size=24,
                            params=json.dumps(params),width=2.3,rotation_id=0,
                            shift_id=0,rotation_seed=12,seed=34,noise_seed=56,noise=0.))
    for d in [2,3]:
        expected=2*np.pi*R if d==2 else 4*np.pi*R**2
        assert np.isclose(shape(d,'circle' if d==2 else 'sphere',{})[2],expected)
        thickness=1.2
        expected=4*np.pi*R if d==2 else 4*np.pi*((R+thickness/2)**2+(R-thickness/2)**2)
        assert np.isclose(shape(d,'shell',dict(thickness=thickness))[2],expected)
        Lratio=1.7;L=Lratio*R
        expected=2*np.pi*R+4*L if d==2 else 4*np.pi*R**2+4*np.pi*R*L
        assert np.isclose(shape(d,'stadium' if d==2 else 'capsule',dict(Lratio=Lratio))[2],expected)
        expected=4*np.pi*R if d==2 else 8*np.pi*R**2
        assert np.isclose(shape(d,'pair_far',dict(gap=2.1))[2],expected)
    minor=.7
    assert np.isclose(shape(3,'torus',dict(minor=minor))[2],4*np.pi**2*R*minor)
    eccentricity=np.sqrt(3)/2
    prolate=2*np.pi*R**2*(1+2/eccentricity*np.arcsin(eccentricity))
    assert np.isclose(shape(3,'ellipsoid',dict(axes=[2,1,1]))[2],prolate,rtol=2e-12)
    oblate=2*np.pi*(2*R)**2*(1+(1-eccentricity**2)/eccentricity*np.arctanh(eccentricity))
    assert np.isclose(shape(3,'oblate',dict(axes=[2,2,1]))[2],oblate,rtol=2e-12)
