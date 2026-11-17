"""Fixed geometric readouts of a domain-wall field, without fitted calibration.

Input is spatial+(d+1), d=2 or 3; vector components follow array axes. Spatial
G4 derivatives are periodic and physical (1/spacing). A supplied contiguous ROI
is applied *after* differentiation, including for mapped-CDF derivatives.

The coarea readouts measure weighted level sets of the generated scalar s. They
are not guaranteed to measure the original input object's boundary. Profile
invariance holds for resolved, monotone flat transitions spanning [-1,1]; finite
windows, curvature, shifted walls, overlapping walls and discretization can bias
these estimates. In particular G4 is not a monotone/TV-diminishing derivative.

Point coarea uses a Gaussian delta normalized on [-1,1]. Integrated coarea first
maps s with that delta's normalized CDF, then integrates |G4 H(s)|. Its normals
use the common G4 s direction so only the readout density changes. Marching uses
its own segment/triangle normals, explicitly measuring the generated s=0 set.
No shape/topology/reconstruction gates or learned conversion factors occur here.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
import numpy as np
from scipy.special import ndtr

COAREA_EPSILONS = (0.2, 0.35, 0.5)

# Declarative metadata is used by the runner/report, never to learn or calibrate.
READOUT_METADATA = {
    'energy': dict(kind='energy', epsilon=None,
                   discrete_density='sum_i,a (G4_i n_a)^2',
                   continuum_density='sum_i,a (partial_i n_a)^2',
                   flat_wall_integrand='sum_a (dn_a/du)^2',
                   physical_units='length^(d-2)', normal_rule='common G4(s) normal'),
    'scalar_energy': dict(kind='scalar_energy', epsilon=None,
                          discrete_density='sum_i (G4_i s)^2',
                          continuum_density='|grad s|^2',
                          flat_wall_integrand='(ds/du)^2',
                          physical_units='length^(d-2)', normal_rule='common G4(s) normal'),
    'tv': dict(kind='tv', epsilon=None,
               discrete_density='0.5 |G4 s|', continuum_density='0.5 |grad s|',
               flat_wall_integrand='0.5 |ds/du|', physical_units='length^(d-1)',
               normal_rule='common G4(s) normal'),
    'weighted_tv': dict(kind='weighted_tv', epsilon=None,
                        discrete_density='0.75 max(0,1-s^2) |G4 s|',
                        continuum_density='0.75 (1-s^2) |grad s| for |s|<=1',
                        flat_wall_integrand='0.75 (1-s^2) |ds/du|',
                        physical_units='length^(d-1)', normal_rule='common G4(s) normal'),
    'generated_s0_marching': dict(kind='marching', epsilon=None,
                                discrete_density='native s=0 contour length / triangle area',
                                continuum_density='delta(s) |grad s|',
                                flat_wall_integrand='zero-level plane patch area',
                                physical_units='length^(d-1)',
                                normal_rule='native segment/triangle normal',
                                roi_footprint='ROI sample-center span, open patches, no capping'),
}
for _epsilon in COAREA_EPSILONS:
    for _prefix in ['point_coarea', 'integrated_coarea']:
        READOUT_METADATA[f'{_prefix}_eps_{_epsilon:.2f}'] = dict(
            kind=_prefix, epsilon=_epsilon,
            delta='exp(-s^2/(2 epsilon^2)) / (sqrt(2 pi) epsilon Z)',
            delta_normalization='Z=Phi(1/epsilon)-Phi(-1/epsilon)',
            discrete_density=('delta_epsilon(s) |G4 s|' if _prefix == 'point_coarea'
                              else '|G4 H_epsilon(s)|'),
            CDF='H=(Phi(s/epsilon)-Phi(-1/epsilon))/Z',
            continuum_density='delta_epsilon(s) |grad s|',
            flat_wall_integrand='delta_epsilon(s) |ds/du|',
            physical_units='length^(d-1)', normal_rule='common G4(s) normal')


def derivative_g4(a: np.ndarray, axis: int, spacing: float = 1.0) -> np.ndarray:
    """Fourth-order centered physical derivative; correlation sign is positive."""
    return ((2.0 / 3.0) * (np.roll(a, -1, axis) - np.roll(a, 1, axis))
            - (1.0 / 12.0) * (np.roll(a, -2, axis) - np.roll(a, 2, axis))) / spacing


def _prepare(n, spacing, roi):
    a = np.asarray(n, dtype=np.float64)
    d = a.ndim - 1
    if d not in (2, 3) or a.shape[-1] != d + 1:
        raise ValueError("n must have spatial+(d+1) shape for d=2 or 3")
    if not np.isfinite(a).all():
        raise ValueError("n must be finite")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be a finite positive isotropic spacing")
    if roi is None:
        roi = (slice(None),) * d
    if len(roi) != d or not all(isinstance(s, slice) for s in roi):
        raise ValueError("roi must be a tuple of contiguous spatial slices")
    bounds = [s.indices(size) for s, size in zip(roi, a.shape[:-1])]
    if any(step != 1 or stop <= start for start, stop, step in bounds):
        raise ValueError("roi must be nonempty with unit slice steps")
    region = tuple(slice(start, stop) for start, stop, _ in bounds)
    coordinates = [(np.arange(start, stop) - (size - 1) / 2) * spacing
                   for (start, stop, _), size in zip(bounds, a.shape[:-1])]
    return a, d, region, coordinates


def _moments(density, normals, coordinates, dV):
    """All geometry mass is retained; undefined normal mass is only diagnosed."""
    shape = density.shape
    d = len(shape)
    weights = density.reshape(-1) * dV
    mass = float(weights.sum())
    nu = normals.reshape(-1, d)
    normal_tensor = nu.T @ (weights[:, None] * nu)
    defined = np.einsum("ni,ni->n", nu, nu) > 0
    defined_mass = float(weights[defined].sum())
    if mass <= 0:
        return dict(normal_tensor=normal_tensor,
                    spatial_covariance=np.full((d, d), np.nan),
                    centroid=np.full(d, np.nan), mass=mass,
                    normal_defined_mass=defined_mass,
                    normal_undefined_mass=mass-defined_mass,
                    covariance_defined=False)
    coords = [np.broadcast_to(x.reshape((1,)*j + (len(x),) + (1,)*(d-j-1)), shape)
              for j, x in enumerate(coordinates)]
    centroid = np.array([np.sum(density*x)*dV/mass for x in coords])
    centered = [x-centroid[j] for j, x in enumerate(coords)]
    covariance = np.array([[np.sum(density*x*y)*dV/mass for y in centered]
                           for x in centered])
    return dict(normal_tensor=normal_tensor, spatial_covariance=covariance,
                centroid=centroid, mass=mass, normal_defined_mass=defined_mass,
                normal_undefined_mass=max(0.0, mass-defined_mass),
                covariance_defined=True)


def _marching(s, spacing, region, full_shape):
    """Native zero-level patch moments; open ROI boundaries are not capped.

    The marching domain is the span of ROI sample centers, not m full voxel cells.
    For disconnected/wrapped sets the Euclidean covariance has its usual linear-
    coordinate limitation. skimage is imported only when this method is requested.
    """
    d = s.ndim
    normal = np.zeros((d, d))
    centroid_sum = np.zeros(d)
    second_sum = np.zeros((d, d))
    mass = 0.0
    offset = np.array([r.start - (size-1)/2 for r, size in zip(region, full_shape)]) * spacing
    sample = s[region]
    if min(sample.shape) < 2 or not (float(sample.min()) < 0 < float(sample.max())):
        return dict(value=0.0, normal_tensor=normal,
                    spatial_covariance=np.full((d, d), np.nan), centroid=np.full(d, np.nan),
                    mass=0.0, normal_defined_mass=0.0, normal_undefined_mass=0.0,
                    covariance_defined=False)
    if d == 2:
        from skimage.measure import find_contours
        for contour in find_contours(sample, level=0.0):
            points = contour*spacing + offset
            delta = np.diff(points, axis=0)
            lengths = np.linalg.norm(delta, axis=1)
            good = lengths > 0
            delta, lengths = delta[good], lengths[good]
            midpoint = ((points[:-1]+points[1:])/2)[good]
            nu = np.stack([-delta[:, 1], delta[:, 0]], axis=1)/lengths[:, None]
            mass += float(lengths.sum())
            normal += nu.T @ (lengths[:, None]*nu)
            centroid_sum += np.sum(lengths[:, None]*midpoint, axis=0)
            second_sum += (midpoint.T @ (lengths[:, None]*midpoint)
                           + delta.T @ (lengths[:, None]*delta)/12)
    else:
        from skimage.measure import marching_cubes
        vertices, faces, _, _ = marching_cubes(sample.astype(np.float32), level=0.0,
                                               spacing=(spacing,)*3, allow_degenerate=False)
        triangles = vertices[faces].astype(np.float64)+offset
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        cross = np.cross(b-a, c-a)
        areas = np.linalg.norm(cross, axis=1)/2
        good = areas > 0
        triangles, cross, areas = triangles[good], cross[good], areas[good]
        middle = triangles.mean(axis=1)
        nu = cross/(2*areas[:, None])
        mass = float(areas.sum())
        normal = nu.T @ (areas[:, None]*nu)
        centroid_sum = np.sum(areas[:, None]*middle, axis=0)
        second_sum = middle.T @ (areas[:, None]*middle)
        centered = triangles-middle[:, None, :]
        second_sum += np.einsum('n,nki,nkj->ij', areas, centered, centered)/12
    if mass > 0:
        center = centroid_sum/mass
        covariance = second_sum/mass-np.outer(center, center)
    else:
        center, covariance = np.full(d, np.nan), np.full((d, d), np.nan)
    return dict(value=mass, normal_tensor=normal, spatial_covariance=covariance,
                centroid=center, mass=mass, normal_defined_mass=mass,
                normal_undefined_mass=0.0, covariance_defined=mass > 0)


def measure(n, spacing: float = 1.0, roi: Sequence[slice] | None = None,
            include_orientation: bool = True, include_marching: bool = True):
    """Return per-method scalar records, optionally with small orientation arrays.

    Methods: energy, scalar_energy, tv, weighted_tv, point_coarea_eps_0.20/0.35/0.50,
    integrated_coarea_eps_0.20/0.35/0.50, and optional generated_s0_marching.
    energy_scalar/energy_vector are the same component-energy diagnostics in every
    record, not per-method values. measurement_seconds includes a common derivative
    pass plus that method's density/moment cost. These shared-pass numbers are not
    isolated optimized single-head latency benchmarks.
    """
    start = time.perf_counter()
    a, d, region, coordinates = _prepare(n, float(spacing), roi)
    dV = spacing**d
    s_full = a[..., -1]
    derivatives = [derivative_g4(a, j, spacing)[region] for j in range(d)]
    scalar_energy_density = sum(g[..., -1]**2 for g in derivatives)
    vector_energy_density = sum(np.sum(g[..., :-1]**2, axis=-1) for g in derivatives)
    energy_density = scalar_energy_density+vector_energy_density
    energy_tensor = np.array([[np.sum(x*y)*dV for y in derivatives] for x in derivatives])
    energy_scalar = float(scalar_energy_density.sum()*dV)
    energy_vector = float(vector_energy_density.sum()*dV)
    g_s = np.stack([g[..., -1] for g in derivatives], axis=-1)
    norm_s = np.linalg.norm(g_s, axis=-1)
    normals = np.divide(g_s, norm_s[..., None], out=np.zeros_like(g_s), where=norm_s[..., None] > 0)
    s = s_full[region]
    common_seconds = time.perf_counter()-start
    result = []

    def record(method, density, own_start, kind, extra=None):
        row = dict(method=method, value=float(density.sum()*dV),
                   energy_scalar=energy_scalar, energy_vector=energy_vector,
                   scalar_fraction=energy_scalar/(energy_scalar+energy_vector)
                   if energy_scalar+energy_vector > 0 else np.nan,
                   readout_kind=kind, common_derivative_seconds=common_seconds)
        if include_orientation:
            row.update(_moments(density, normals, coordinates, dV))
            row['energy_tensor'] = energy_tensor.copy()
        if extra:
            row.update(extra)
        row['method_seconds'] = time.perf_counter()-own_start
        row['measurement_seconds'] = common_seconds+row['method_seconds']
        result.append(row)

    own = time.perf_counter()
    record('energy', energy_density, own, 'Dirichlet energy; units length^(d-2)')
    own = time.perf_counter()
    record('scalar_energy', scalar_energy_density, own, 'Scalar Dirichlet energy; units length^(d-2)')
    own = time.perf_counter()
    record('tv', .5*norm_s, own, 'Mean generated level-set measure over s in [-1,1]')
    own = time.perf_counter()
    record('weighted_tv', .75*np.maximum(0.0, 1-s*s)*norm_s, own,
           'Profile-weighted generated level-set measure')
    for epsilon in COAREA_EPSILONS:
        own = time.perf_counter()
        z = float(ndtr(1/epsilon)-ndtr(-1/epsilon))
        delta = np.exp(-.5*(s/epsilon)**2)/(np.sqrt(2*np.pi)*epsilon*z)
        record(f'point_coarea_eps_{epsilon:.2f}', delta*norm_s, own,
               'Point-sampled Gaussian-delta coarea of generated s',
               dict(epsilon=epsilon, delta_normalization=z))
        own = time.perf_counter()
        mapped = (ndtr(s_full/epsilon)-ndtr(-1/epsilon))/z
        mapped_gradients = [derivative_g4(mapped, j, spacing)[region] for j in range(d)]
        density = np.sqrt(sum(g*g for g in mapped_gradients))
        record(f'integrated_coarea_eps_{epsilon:.2f}', density, own,
               'G4 TV of normalized Gaussian-CDF(s); generated level-set measure',
               dict(epsilon=epsilon, delta_normalization=z))
    if include_marching:
        own = time.perf_counter()
        m = _marching(s_full, spacing, region, a.shape[:-1])
        value = m.pop('value')
        row = dict(method='generated_s0_marching', value=value,
                   energy_scalar=energy_scalar, energy_vector=energy_vector,
                   scalar_fraction=energy_scalar/(energy_scalar+energy_vector)
                   if energy_scalar+energy_vector > 0 else np.nan,
                   readout_kind='Generated s=0 contours/triangles; ROI sample-center span, no capping',
                   common_derivative_seconds=common_seconds)
        if include_orientation:
            row.update(m)
            row['energy_tensor'] = energy_tensor.copy()
        row['method_seconds'] = time.perf_counter()-own
        row['measurement_seconds'] = common_seconds+row['method_seconds']
        result.append(row)
    return result


def orientation_tensors(n, method: str, spacing: float = 1.0, roi=None):
    """Slow reference API; use measure(include_orientation=True) for shared work."""
    rows = measure(n, spacing, roi, include_orientation=True,
                   include_marching=method == 'generated_s0_marching')
    match = next((row for row in rows if row['method'] == method), None)
    if match is None:
        raise ValueError(f'Unknown readout method: {method}')
    keys = ['energy_tensor', 'normal_tensor', 'spatial_covariance', 'centroid',
            'mass', 'normal_defined_mass', 'normal_undefined_mass', 'covariance_defined']
    return {key: match[key] for key in keys}
