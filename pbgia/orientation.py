"""Projective director, orientation tensor and shape axes.

Neighboring walls of a striped texture have opposite normals, so the vector v averages out; its outer
product survives. With x along columns and y up (y = -row):

    T_sigma = G_sigma * (v v^T),     q_sigma = (T_xx - T_yy) + 2i T_xy = |q| e^{2 i phi},

phi being the director (normal) angle measured counterclockwise from +x. Lattice symmetry adds a small,
fixed bias to the normal angle, dphi = sum_j a_j sin(4 j phi), which correct_bias removes.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from .generator import Field
from .readout import tv_density

__all__ = ["BIAS", "q0", "q_sigma", "director_angle", "correct_bias", "shape_axis", "covariance_axis",
           "inward_normals"]

#: Normal-angle bias coefficients a_j (radians), j = 1..3, fitted on planar walls (2D, h/J = 1.5, N = 30).
BIAS = {1.5: [0.04167200660227819, 0.005595058964697164, 0.0019031204836057495]}


def _v(x):
    if isinstance(x, Field):
        return [np.asarray(c, np.float64) for c in x.v]
    return [np.asarray(c, np.float64) for c in x]


def q0(field_or_v, normalize: bool = False) -> np.ndarray:
    """Pixel-level complex order parameter (unsmoothed) of a 2D field."""
    vr, vc = _v(field_or_v)
    if normalize:
        nn = vr * vr + vc * vc + 1e-12
        vr, vc = vr / np.sqrt(nn), vc / np.sqrt(nn)
    return (vc * vc - vr * vr) - 2j * vr * vc


def q_sigma(field_or_q0, sigma: float, mode: str = "reflect"):
    """Scale-sigma order parameter q_sigma and tr T_sigma (Gaussian window, reflecting boundaries)."""
    if isinstance(field_or_q0, np.ndarray) and np.iscomplexobj(field_or_q0):
        q = field_or_q0
        tr = None
    else:
        vr, vc = _v(field_or_q0)
        q = (vc * vc - vr * vr) - 2j * vr * vc
        tr = ndi.gaussian_filter(vr * vr + vc * vc, sigma, mode=mode)
    qs = ndi.gaussian_filter(q.real, sigma, mode=mode) + 1j * ndi.gaussian_filter(q.imag, sigma, mode=mode)
    return qs, tr


def director_angle(q) -> np.ndarray:
    """Normal-director angle phi in [0, pi), counterclockwise from +x (columns), y up."""
    return (0.5 * np.angle(q)) % np.pi


def correct_bias(phi, a=None):
    """Remove the lattice bias dphi = sum_j a_j sin(4 j phi) from a measured normal angle (radians).

    The constants are for the standard chiral stencil. With relax(..., chiral_stencil="isotropic") the
    bias is already removed at its source (a_1 about 0.5 deg) and this correction should not be applied."""
    a = BIAS[1.5] if a is None else a
    return phi - sum(a[j] * np.sin(4 * (j + 1) * phi) for j in range(len(a)))


def inward_normals(field: Field):
    """Unit inward normals (row, col components) on the walls; zero where |v| vanishes."""
    vr, vc = _v(field)
    nr = np.sqrt(vr * vr + vc * vc)
    ok = nr > 1e-6
    return np.where(ok, vr / np.maximum(nr, 1e-12), 0.0), np.where(ok, vc / np.maximum(nr, 1e-12), 0.0)


def shape_axis(field: Field, region=None, bias_correct: bool = True) -> float:
    """Long-axis angle (radians in [0, pi), counterclockwise from +x, y up) of an elongated shape from
    the tensor v v^T weighted by the wall density (1/2)|G s|, pooled over the shape (or a boolean region),
    with the lattice bias removed. Bias coefficients hold for fields relaxed at h/J = 1.5.

    The bias correction is exact for straight walls, so this is most accurate for shapes with long
    straight sides (stadium: < 0.1 deg); for ellipses typical errors are ~0.4 deg. For the global axis of
    a single connected shape, covariance_axis is simpler and slightly more accurate; the tensor form is
    useful because it can be pooled over any region or window (see q_sigma)."""
    vr, vc = _v(field)
    w = tv_density(field.s)
    if region is not None:
        w = w * np.asarray(region, float)
    Z = np.sum(w * ((vc * vc - vr * vr) - 2j * vr * vc))
    phi_n = director_angle(Z)
    if bias_correct:
        if field.params.get("h", 1.5) != 1.5:
            raise ValueError("bias coefficients were fitted at h/J = 1.5; relax with h=1.5 or use bias_correct=False")
        phi_n = correct_bias(phi_n)
    # normals of an elongated shape cluster perpendicular to its long axis
    return float((phi_n + np.pi / 2) % np.pi)


def covariance_axis(field: Field) -> float:
    """Long-axis angle (radians in [0, pi), counterclockwise from +x, y up) from the spatial covariance
    of the wall positions, weighted by |grad s| (the 'wall-weighted covariance axis' of the paper)."""
    s = np.asarray(field.s, np.float64)
    gy, gx = np.gradient(s)
    w = np.hypot(gx, gy)
    w = w / (w.sum() + 1e-12)
    yr, xr = np.indices(s.shape)
    x = xr.astype(float)
    y = -yr.astype(float)
    mx, my = np.sum(w * x), np.sum(w * y)
    cxx = np.sum(w * (x - mx) ** 2)
    cxy = np.sum(w * (x - mx) * (y - my))
    cyy = np.sum(w * (y - my) ** 2)
    _, V = np.linalg.eigh([[cxx, cxy], [cxy, cyy]])
    v = V[:, -1]
    return float(np.arctan2(v[1], v[0]) % np.pi)
