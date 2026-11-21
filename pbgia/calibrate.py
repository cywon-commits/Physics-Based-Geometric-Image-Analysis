"""Refit the 2D readout constants for other generator settings.

The constants of the boundary readout are properties of the generator at fixed (J, D, h, N), not of the
shapes being measured. calibrate_2d recomputes them as in the paper:
  1. planar walls at all rational orientations p/q (p <= q <= 12) on periodic 240^2 arrays give
     m(phi) = c0 + sum_j c_j cos(4 j phi), j <= 6 (least squares);
  2. disks of radius 2.5-40 (six subpixel centers each) give the curvature law
     Delta(k) = b1 k + b2 k^2 + b3 k^3 from the residual TV / true perimeter - <m(phi)>.
With the paper settings (h/J = 1.5, D/J = 1, N = 30) it reproduces CONSTANTS['2d_direction'].
"""
from __future__ import annotations

import math

import numpy as np

from .generator import relax
from .readout import smoothed_normals, tv_density

__all__ = ["calibrate_2d"]

N_HARMONICS = 6
DISK_RADII = (2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 16, 20, 28, 40)


def _s(mask, D, h, n_steps):
    return relax(mask, n_steps=n_steps, D=D, h=h).s.astype(np.float64)


def calibrate_2d(D: float = 1.0, h: float = 1.5, n_steps: int = 30, L: int = 240, seed: int = 11,
                 kmax: float = 0.45) -> dict:
    """Return a readout law dict usable as pbgia.readout.CONSTANTS entry (keys h, kmax, c, b)."""
    y, x = np.mgrid[:L, :L] + 0.5
    X, Y = [], []
    for p in range(0, 13):
        for q in range(1, 13):
            if p > q or math.gcd(p, q) != 1:
                continue
            s = _s(((-p * x + q * y) % L) < L / 2, D, h, n_steps)
            phi = math.atan2(p, q)
            X.append([math.cos(4 * j * phi) for j in range(1, N_HARMONICS + 1)])
            Y.append(tv_density(s).sum() / (2 * L * math.hypot(p, q)))
    A = np.c_[np.ones(len(X)), np.array(X)]
    c = np.linalg.lstsq(A, np.array(Y), rcond=None)[0]

    rng = np.random.default_rng(seed)
    kk, res = [], []
    for R in DISK_RADII:
        for _ in range(6):
            n = int(2 * R + 24)
            ctr = n / 2 + rng.uniform(-0.5, 0.5, 2)
            r = np.sqrt(sum((np.indices((n, n))[a] - ctr[a]) ** 2 for a in range(2)))
            s = _s(r <= R, D, h, n_steps)
            w = tv_density(s)
            nrm, _ = smoothed_normals(s)
            phi = np.arctan2(nrm[0], nrm[1])
            mp = c[0] + sum(c[j] * np.cos(4 * j * phi) for j in range(1, N_HARMONICS + 1))
            res.append(w.sum() / (2 * np.pi * R) - np.sum(w * mp) / w.sum())
            kk.append(1.0 / R)
    kk = np.array(kk)
    b = np.linalg.lstsq(np.c_[kk, kk ** 2, kk ** 3], np.array(res), rcond=None)[0]
    return dict(h=h, D=D, n_steps=n_steps, kmax=kmax, c=c.tolist(), b=b.tolist())
