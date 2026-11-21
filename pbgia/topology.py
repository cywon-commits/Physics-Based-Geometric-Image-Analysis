"""Integer degree of the oriented spin field and the Euler characteristic of the generated domain.

2D: each plaquette is split into two triangles; Q_2 = sum Omega / 4 pi with the signed solid angle
    Omega = 2 atan2[n1.(n2 x n3), 1 + n1.n2 + n2.n3 + n3.n1].
3D: the field is interpolated linearly on six Freudenthal tetrahedra per cell and normalized; the signed
    preimages of a generic regular value are counted.
Wherever the walls saturate, the degree summed over the region nearest to a domain equals its Euler
characteristic.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
import scipy.ndimage as ndi

from .generator import Field

__all__ = ["solid_angle_density", "degree2d", "degree_per_domain", "degree3d", "euler_characteristic"]


def _n2(field_or_n):
    if isinstance(field_or_n, Field):
        vr, vc = field_or_n.v
        s = field_or_n.s
    else:
        vr, vc, s = field_or_n
    # physical frame: x = columns, y = up (-rows), z = s
    return np.stack([vc, -vr, s], axis=-1).astype(np.float64)


def _omega(p, q, r):
    num = np.einsum("...i,...i->...", p, np.cross(q, r))
    den = 1 + np.sum(p * q, axis=-1) + np.sum(q * r, axis=-1) + np.sum(r * p, axis=-1)
    return 2 * np.arctan2(num, den)


def solid_angle_density(field_or_n) -> np.ndarray:
    """Degree density per plaquette (shape rows-1, cols-1); sums to the total degree Q_2."""
    n = _n2(field_or_n)
    a, b, c, d = n[:-1, :-1], n[:-1, 1:], n[1:, 1:], n[1:, :-1]
    # (a, b, c) is clockwise for x right, y up: hence the minus sign
    return -(_omega(a, b, c) + _omega(a, c, d)) / (4 * np.pi)


def degree2d(field_or_n) -> float:
    """Total degree Q_2 of a 2D field (equals the Euler characteristic of {s > 0} when walls saturate)."""
    return float(np.sum(solid_angle_density(field_or_n)))


def degree_per_domain(field: Field):
    """Degree summed over the region nearest to each connected domain of {s > 0} (8-connected).
    Returns (labels, list of degrees). Region assignment uses the distance transform of the domains."""
    dom = field.s > 0
    lab, n = ndi.label(dom, structure=np.ones((3, 3)))
    if n == 0:
        return lab, []
    _, (ir, ic) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    nearest = lab[ir, ic]
    rho = solid_angle_density(field)
    pl = nearest[:-1, :-1]
    return lab, [float(rho[pl == k].sum()) for k in range(1, n + 1)]


def euler_characteristic(mask) -> int:
    """Euler characteristic of a 2D binary mask, 8-connected foreground / 4-connected background."""
    m = np.asarray(mask, bool)
    _, nl = ndi.label(m, structure=np.ones((3, 3)))
    _, nh = ndi.label(np.pad(~m, 1, constant_values=True))
    return int(nl - (nh - 1))


def degree3d(field_or_n, target=(0.37139, -0.52177, 0.64829, 0.40981)) -> dict:
    """Integer degree of a 3D field n = (v_0, v_1, v_2, s) in S^3 by counting signed preimages of a generic
    regular value on six Freudenthal tetrahedra per cell. Returns the degree and diagnostics."""
    if isinstance(field_or_n, Field):
        n = field_or_n.n.astype(np.float64)
    else:
        n = np.asarray(field_or_n, np.float64)
    t = np.array(target, np.float64)
    t /= np.linalg.norm(t)
    shape = np.array(n.shape[:3])
    origin = np.indices(tuple(shape - 1), dtype=np.int32).reshape(3, -1).T
    deg = 0
    hits_total = 0
    near_singular = 0
    for perm in permutations(range(3)):
        offs = np.zeros((4, 3), dtype=int)
        for j, ax in enumerate(perm):
            offs[j + 1] = offs[j]
            offs[j + 1, ax] += 1
        orientation = int(round(np.linalg.det(np.eye(3, dtype=int)[:, list(perm)])))
        verts = np.stack([n[tuple((origin + o).T)] for o in offs], axis=1)
        possible = np.ones(verts.shape[0], bool)
        for a in range(4):
            if t[a] > 1e-8:
                possible &= verts[:, :, a].max(axis=1) > 0
            elif t[a] < -1e-8:
                possible &= verts[:, :, a].min(axis=1) < 0
        sel = verts[possible]
        if not len(sel):
            continue
        M = sel.transpose(0, 2, 1)
        det = np.linalg.det(M)
        ok = np.abs(det) > 1e-10
        near_singular += int(np.sum(~ok))
        M, det = M[ok], det[ok]
        sol = np.linalg.solve(M, np.broadcast_to(t, (len(M), 4))[..., None])[..., 0]
        hit = sol.min(axis=1) > 1e-8
        hits_total += int(hit.sum())
        deg += int(np.sum(np.sign(det[hit]))) * orientation
    return dict(degree=deg, preimages=hits_total, near_singular_tetrahedra=near_singular)
