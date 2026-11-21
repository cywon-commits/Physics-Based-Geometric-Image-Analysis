"""Boundary extent from the total variation of s, divided by the metrication factor.

    A_{d-1} = sum_x (1/2)|G s|_x / [ m(phi_x) + Delta(kappa_x) ]

m(phi) is the planar metrication (lattice-symmetric harmonics, unity on the mirror directions) and
Delta(kappa) the curvature correction. Both are properties of the generator at fixed settings: the
constants below hold only for the settings they were fitted at (paper, Methods "Readout constants").
For other settings, refit them with pbgia.calibrate.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from .generator import Field, relax

__all__ = ["CONSTANTS", "tv_density", "curvature", "smoothed_normals", "boundary_measure",
           "perimeter", "surface_area"]

#: Fitted readout constants (paper settings, D/J = 1, N = 30, synchronous updates).
CONSTANTS = {
    # 2D readout: h/J = 1.5, direction-resolved planar law (cos 4j phi, j <= 6) + cubic curvature law
    "2d_direction": dict(
        h=1.5, kmax=0.45,
        c=[1.0129386834703689, 0.002141680565239055, -0.00489617503219151, -0.0017921457701057337,
           -0.0031126180624257613, -0.0005470763847376691, -0.0017217197777804445],
        b=[0.05714424276442897, -1.1520158680456973, 0.9241351646791964]),
    # orientation-averaged laws at h/J = 2 (3D readout; 2D variant for reference)
    "2d_average": dict(h=2.0, kmax=0.45, mbar=1.018862336220135,
                       b=[0.03832279585463617, -0.7483716972129485, 0.4327409658981928]),
    # 3D: planar walls at least 9 voxels apart (41 normals, Voronoi weights), spheres for the curvature law
    "3d_average": dict(h=2.0, kmax=0.85, mbar=1.0191629063916545,
                       b=[0.05310278949390283, -0.6601592190153166, 0.37014894797983766]),
}


def _grad(z):
    return [(np.roll(z, -1, a) - np.roll(z, 1, a)) * 0.5 for a in range(z.ndim)]


def tv_density(s) -> np.ndarray:
    """Per-site boundary density (1/2)|G s| with centered differences (periodic)."""
    s = np.asarray(s, np.float64)
    return 0.5 * np.sqrt(sum(g * g for g in _grad(s)))


def smoothed_normals(s, sigma: float = 1.0):
    """Unit normals of s smoothed with a Gaussian (periodic) and |kappa| = |div n|."""
    ss = ndi.gaussian_filter(np.asarray(s, np.float64), sigma, mode="wrap")
    g = _grad(ss)
    nr = np.sqrt(sum(x * x for x in g)) + 1e-12
    n = [x / nr for x in g]
    k = -sum((np.roll(n[a], -1, a) - np.roll(n[a], 1, a)) * 0.5 for a in range(ss.ndim))
    return n, np.abs(k)


def curvature(s, sigma: float = 1.0) -> np.ndarray:
    """Signed curvature -div(grad s~/|grad s~|) of the smoothed field (2D: kappa, 3D: 2H)."""
    ss = ndi.gaussian_filter(np.asarray(s, np.float64), sigma, mode="wrap")
    g = _grad(ss)
    nr = np.sqrt(sum(x * x for x in g)) + 1e-12
    return -sum((np.roll(g[a] / nr, -1, a) - np.roll(g[a] / nr, 1, a)) * 0.5 for a in range(ss.ndim))


def _s(x):
    return x.s if isinstance(x, Field) else np.asarray(x)


def boundary_measure(field_or_s, law: str | None = None) -> float:
    """Perimeter (2D) or surface area (3D) in pixel units from a relaxed field.

    law: '2d_direction' (default in 2D; field must come from h/J = 1.5),
         '2d_average' or '3d_average' (default in 3D; field from h/J = 2).
    """
    s = np.asarray(_s(field_or_s), np.float64)
    d = s.ndim
    law = law or ("2d_direction" if d == 2 else "3d_average")
    C = CONSTANTS[law]
    if isinstance(field_or_s, Field) and field_or_s.params.get("chiral_stencil", "standard") != "standard":
        raise ValueError("extent constants were fitted with the standard chiral stencil; relax with "
                         "chiral_stencil='standard' for extent (the isotropic stencil is for directions)")
    if isinstance(field_or_s, Field) and abs(field_or_s.params.get("h", C["h"]) - C["h"]) > 1e-9:
        raise ValueError(f"law {law!r} was fitted at h/J={C['h']}, field has h/J={field_or_s.params['h']}; "
                         "use matching settings or refit with pbgia.calibrate")
    w = tv_density(s)
    n, k = smoothed_normals(s)
    k = np.clip(k, 0, C["kmax"])
    b = C["b"]
    delta = b[0] * k + b[1] * k * k + b[2] * k ** 3
    if law == "2d_direction":
        if d != 2:
            raise ValueError("2d_direction law needs a 2D field")
        phi = np.arctan2(n[0], n[1])
        c = C["c"]
        m = c[0] + sum(c[j] * np.cos(4 * j * phi) for j in range(1, len(c)))
    else:
        m = C["mbar"]
    return float(np.sum(w / (m + delta)))


def perimeter(mask, **kw) -> float:
    """Perimeter of a 2D binary mask (pixel units) with the paper's 2D readout settings."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 2:
        raise ValueError("perimeter needs a 2D mask")
    return boundary_measure(relax(mask, h=CONSTANTS["2d_direction"]["h"], **kw), "2d_direction")


def surface_area(mask, **kw) -> float:
    """Surface area of a 3D binary mask (voxel units) with the paper's 3D readout settings."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 3:
        raise ValueError("surface_area needs a 3D mask")
    return boundary_measure(relax(mask, h=CONSTANTS["3d_average"]["h"], **kw), "3d_average")
