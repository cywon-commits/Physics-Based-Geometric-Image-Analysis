"""Readout mathematics tests, not reconstruction/selection acceptance gates."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from readouts import measure, derivative_g4, orientation_tensors


def periodic_planar(A, size=512, transverse=16, spacing=.125, phase=.37):
    x=(np.arange(size)-phase)*spacing
    s=np.tanh(A*np.sin(2*np.pi*x/(size*spacing)))
    s=np.broadcast_to(s[:, None], (size, transverse))
    n=np.zeros(s.shape+(3,))
    n[..., 0]=np.sqrt(np.maximum(0, 1-s*s));n[..., 2]=s
    return n


def test_resolved_planar_profile_and_energy_width_dependence():
    profiles=[]
    target=2*16*.125
    for A in [4, 8, 12]:
        rows={r['method']:r for r in measure(periodic_planar(A),spacing=.125,include_marching=False)}
        profiles.append(rows)
        for name in ['tv','weighted_tv','point_coarea_eps_0.20','point_coarea_eps_0.35',
                     'point_coarea_eps_0.50','integrated_coarea_eps_0.20',
                     'integrated_coarea_eps_0.35','integrated_coarea_eps_0.50']:
            assert abs(rows[name]['value']/target-1)<.003, (A,name,rows[name]['value'])
            assert np.isclose(np.trace(rows[name]['normal_tensor']), rows[name]['value'],rtol=1e-13)
        energy=rows['energy']
        assert np.isclose(energy['value'],energy['energy_scalar']+energy['energy_vector'],rtol=1e-14)
        assert np.isclose(np.trace(energy['energy_tensor']),energy['value'],rtol=1e-14)
    assert profiles[-1]['energy']['value']/profiles[0]['energy']['value']>2


def test_underresolved_point_delta_can_miss_wall_and_mapped_tv_is_not_exact():
    point=[];integrated=[];target=2*8
    for phase in [0,.25,.5]:
        n=periodic_planar(80,size=64,transverse=8,spacing=1.,phase=phase)
        r={x['method']:x for x in measure(n,include_orientation=False,include_marching=False)}
        point.append(r['point_coarea_eps_0.20']['value'])
        integrated.append(r['integrated_coarea_eps_0.20']['value'])
    assert np.ptp(point)>target
    assert min(point)<.01*target
    assert min(integrated)>.8*target
    # Higher-order G4 can ring and overcount even after integrated CDF mapping.
    assert max(abs(np.array(integrated)/target-1))>.05


def test_roi_is_applied_after_full_periodic_derivative():
    size=96;a=.5;width=2.;x=(np.arange(size)-(size-1)/2)*a
    s=np.broadcast_to(np.tanh((x-.13)/width)[:,None],(size,size))
    n=np.zeros(s.shape+(3,));n[...,0]=np.sqrt(1-s*s);n[...,2]=s
    before=n.copy();roi=(slice(24,72),slice(24,72))
    r={x['method']:x for x in measure(n,a,roi,include_marching=False)}
    direct=.5*np.sqrt(sum(derivative_g4(n[...,-1],j,a)[roi]**2 for j in [0,1])).sum()*a*a
    assert np.isclose(r['tv']['value'],direct,rtol=1e-14)
    # One flat wall, finite ROI crossing from approximately -1 to +1.
    assert abs(r['tv']['value']/(48*a)-1)<2e-4
    assert np.array_equal(n,before)
    wrong=.5*np.sqrt(sum(derivative_g4(n[roi][...,-1],j,a)**2 for j in [0,1])).sum()*a*a
    assert abs(wrong-r['tv']['value'])>10
    t=orientation_tensors(n,'tv',a,roi)
    assert np.allclose(t['normal_tensor'],r['tv']['normal_tensor'])
    assert t['normal_tensor'][0,0]>100*t['normal_tensor'][1,1]


def test_three_d_units_covariance_and_no_mutation():
    size=(96,12,10);a=.5;x=(np.arange(size[0])-(size[0]-1)/2)*a
    s=np.broadcast_to(np.tanh(x[:,None,None]/1.5),size)
    n=np.zeros(size+(4,));n[...,0]=np.sqrt(1-s*s);n[...,3]=s
    roi=(slice(24,72),slice(None),slice(None));before=n.copy()
    rows={r['method']:r for r in measure(n,a,roi,include_marching=False)}
    expected=size[1]*size[2]*a*a
    assert abs(rows['tv']['value']/expected-1)<1e-6
    assert np.allclose(rows['tv']['centroid'],np.zeros(3),atol=1e-12)
    assert rows['tv']['spatial_covariance'][1,1]>rows['tv']['spatial_covariance'][0,0]
    assert np.array_equal(n,before)
    assert np.isclose(rows['energy']['value'],np.trace(rows['energy']['energy_tensor']))
    assert all(r['measurement_seconds']>=r['common_derivative_seconds']>=0 for r in rows.values())


def test_constant_field_zero_density_has_undefined_position_moment():
    n=np.zeros((12,14,3));n[...,2]=-1
    for r in measure(n,include_marching=False):
        assert r['value']==0
        assert r['energy_scalar']==r['energy_vector']==0
        assert not r['covariance_defined']
        assert np.isnan(r['spatial_covariance']).all()
        assert np.array_equal(r['normal_tensor'],np.zeros((2,2)))


def test_marching_measures_generated_zero_surface_and_roi_open_patch():
    pytest.importorskip('skimage')
    size=80;a=.5;coords=np.indices((size,size),dtype=float)-(size-1)/2
    radius=6.;s=np.tanh((radius-np.linalg.norm(coords*a,axis=0))/1.3)
    n=np.zeros(s.shape+(3,));n[...,0]=np.sqrt(1-s*s);n[...,2]=s
    rows={r['method']:r for r in measure(n,a)};r=rows['generated_s0_marching']
    assert abs(r['value']/(2*np.pi*radius)-1)<.003
    assert np.allclose(r['centroid'],0,atol=1e-10)
    assert np.isclose(np.trace(r['normal_tensor']),r['value'])
    x=(np.arange(size)-(size-1)/2)*a;s=np.broadcast_to(np.tanh(x[:,None]),(size,size))
    n[...,0]=np.sqrt(1-s*s);n[...,2]=s
    r={r['method']:r for r in measure(n,a,(slice(20,60),slice(20,60)))}['generated_s0_marching']
    assert np.isclose(r['value'],39*a)
