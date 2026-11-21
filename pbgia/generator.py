"""The PBGIA generator: unit vectors n = (v, s) on the pixel (2D) or voxel (3D) lattice, aligned with
the local field

    H_v = J sum_nb v + D G s,        H_s = J sum_nb s + h f,        n <- H / |H|,

where f is the source (binary image mapped to +-1, or a continuous source in [-1, 1]), the sums run over
the 2d axial neighbors, and G is the centered difference.

Chiral stencil. With chiral_stencil="standard" (the paper's extent benchmark), G is the plain centered
difference. With chiral_stencil="isotropic", each component of G is averaged over the transverse
neighbors with weights [1/6, 2/3, 1/6] along every transverse axis, which cancels the leading lattice
anisotropy of the difference (cubic term proportional to q_x |q|^2). This removes the orientation bias
of the wall normal at its source (a_1: 2.4 deg -> 0.5 deg) and lowers the per-pixel normal scatter; the
paper recommends it for directions (normals, directors) and keeps the standard stencil for extent. Boundaries are periodic (pad the input if the
image touches its border). Array axis 0 is the image row, axis 1 the column, axis 2 the slice.

The fixed points do not depend on the update schedule; the schedules differ only in how fast they get
there (sync: all sites at once, as in the paper's benchmark; redblack: the two sublattices in turn,
about twice as fast; damped: n <- normalize((1-gamma) n + gamma H/|H|)).
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
import scipy.ndimage as ndi

__all__ = ["Field", "initial_state", "relax", "binary_source", "DEFAULTS", "ALPHA_ISO"]

#: Transverse weight of the isotropic chiral stencil.
ALPHA_ISO = 1.0 / 6.0

#: Settings used in the paper. 2D readout: h/J = 1.5; 3D readout and reference figures: h/J = 2.
DEFAULTS = {
    2: dict(J=1.0, D=1.0, h=1.5, n_steps=30),
    3: dict(J=1.0, D=1.0, h=2.0, n_steps=30),
}


@dataclass
class Field:
    """Relaxed spin texture.

    v : list of arrays, one per lattice axis (transverse spin, points along the inward boundary normal)
    s : array, longitudinal spin (positive inside the object, zero level = boundary)
    source : the source f used
    params : generator settings
    """

    v: list
    s: np.ndarray
    source: np.ndarray
    params: dict = dc_field(default_factory=dict)

    @property
    def ndim(self) -> int:
        return self.s.ndim

    @property
    def n(self) -> np.ndarray:
        """Spin as one array with the components last: (v_0, ..., v_{d-1}, s)."""
        return np.stack(list(self.v) + [self.s], axis=-1)

    def mask(self) -> np.ndarray:
        """Generated domain {s > 0}."""
        return self.s > 0


def binary_source(mask, dtype=np.float32) -> np.ndarray:
    """Map a binary image to the source f = +1 (object) / -1 (background)."""
    return np.where(np.asarray(mask, bool), 1.0, -1.0).astype(dtype)


def initial_state(f, w_init: float = 2.3, dtype=np.float32):
    """Distance-transform start: s = tanh(xi / w_init) with xi the signed distance of {f > 0},
    v along the centered gradient of s with length sqrt(1 - s^2)."""
    f = np.asarray(f)
    b = f > 0
    xi = ndi.distance_transform_edt(b) - ndi.distance_transform_edt(~b)
    s = np.tanh(xi / w_init).astype(dtype)
    g = [(np.roll(s, -1, a) - np.roll(s, 1, a)) * dtype(0.5) for a in range(f.ndim)]
    nr = np.sqrt(sum(x * x for x in g))
    sc = np.sqrt(np.maximum(0, 1 - s * s)) / np.maximum(nr, 1e-12)
    return [x * sc for x in g], s


def _chiral_gradient(s, stencil="standard"):
    """Centered difference of s along each axis; "isotropic" adds the transverse average 1/6."""
    d = s.ndim
    out = []
    for c in range(d):
        g = (np.roll(s, -1, c) - np.roll(s, 1, c)) * 0.5
        if stencil == "isotropic":
            a = ALPHA_ISO
            for b in range(d):
                if b != c:
                    g = (1 - 2 * a) * g + a * (np.roll(g, 1, b) + np.roll(g, -1, b))
        elif stencil != "standard":
            raise ValueError(f"unknown chiral_stencil {stencil!r}")
        out.append(g)
    return out


def _local_field(v, s, f, J, D, h, stencil="standard"):
    d = s.ndim

    def nb(z):
        return sum(np.roll(z, k, a) for a in range(d) for k in (1, -1))

    G = _chiral_gradient(s, stencil)
    Hv = [J * nb(v[c]) + D * G[c] for c in range(d)]
    Hs = J * nb(s) + h * f
    return Hv, Hs


def relax(f, n_steps: int | None = None, J: float = 1.0, D: float = 1.0, h: float | None = None,
          schedule: str = "sync", gamma: float = 0.5, w_init: float = 2.3, dtype=np.float32,
          state=None, chiral_stencil: str = "standard") -> Field:
    """Relax the spin texture for a source f (2D or 3D array of values in [-1, 1]).

    Defaults (n_steps, h) are those of the paper for the array dimension (see DEFAULTS).
    A binary mask may be passed directly; it is mapped to +-1.
    chiral_stencil: "standard" (extent; the paper's benchmark) or "isotropic" (directions).
    """
    f = np.asarray(f)
    if f.dtype == bool:
        f = binary_source(f, dtype)
    f = f.astype(dtype)
    d = f.ndim
    if d not in (2, 3):
        raise ValueError("source must be a 2D or 3D array")
    h = DEFAULTS[d]["h"] if h is None else h
    n_steps = DEFAULTS[d]["n_steps"] if n_steps is None else n_steps
    if state is None:
        v, s = initial_state(f, w_init, dtype)
    else:
        v, s = [np.asarray(x, dtype) for x in state[0]], np.asarray(state[1], dtype)
    par = (np.indices(f.shape).sum(0) % 2).astype(bool)

    def apply(v, s, sel_mask, g=1.0):
        Hv, Hs = _local_field(v, s, f, J, D, h, chiral_stencil)
        H = np.sqrt(sum(x * x for x in Hv) + Hs * Hs)
        ok = H > 1e-12
        Hm = np.maximum(H, 1e-12)
        nv = [x / Hm for x in Hv]
        ns = Hs / Hm
        if g != 1.0:
            nv = [(1 - g) * a + g * b for a, b in zip(v, nv)]
            ns = (1 - g) * s + g * ns
            nn = np.maximum(np.sqrt(sum(x * x for x in nv) + ns * ns), 1e-12)
            nv = [x / nn for x in nv]
            ns = ns / nn
        sel = ok if sel_mask is None else (ok & sel_mask)
        return [np.where(sel, a, b) for a, b in zip(nv, v)], np.where(sel, ns, s)

    for _ in range(n_steps):
        if schedule == "sync":
            v, s = apply(v, s, None)
        elif schedule == "redblack":
            v, s = apply(v, s, ~par)
            v, s = apply(v, s, par)
        elif schedule == "damped":
            v, s = apply(v, s, None, gamma)
        else:
            raise ValueError(f"unknown schedule {schedule!r}")
    params = dict(J=J, D=D, h=h, n_steps=n_steps, schedule=schedule, w_init=w_init,
                  chiral_stencil=chiral_stencil)
    return Field(v=list(v), s=s, source=f, params=params)
