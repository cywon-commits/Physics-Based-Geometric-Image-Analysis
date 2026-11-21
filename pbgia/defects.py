"""Half-integer defects of the projective director.

Defects are zeros of q_sigma. On the lattice each plaquette carries an exact multiple of 1/2,

    k = (1/4 pi) * (sum of wrapped phase differences of arg q_sigma around the plaquette),

and plaquette charges sum exactly to the winding along any closed boundary.
"""
from __future__ import annotations

import numpy as np

from .orientation import q_sigma

__all__ = ["plaquette_charges", "defects", "defects_at_scale"]


def plaquette_charges(q) -> np.ndarray:
    """Charge (multiples of 1/2, counterclockwise with y up) on each plaquette of a 2D complex field.
    Output shape is (rows-1, cols-1); plaquette (r, c) has its center at (r+1/2, c+1/2)."""
    a = np.angle(q)

    def d(u, v):
        return (v - u + np.pi) % (2 * np.pi) - np.pi

    w = (d(a[:-1, :-1], a[:-1, 1:]) + d(a[:-1, 1:], a[1:, 1:]) + d(a[1:, 1:], a[1:, :-1])
         + d(a[1:, :-1], a[:-1, :-1]))
    # loop (r,c)->(r,c+1)->(r+1,c+1)->(r+1,c) is clockwise for y up; the sign gives the standard charge
    return -np.rint(w / (2 * np.pi)) / 2.0


def defects(q, margin: int = 0, trace=None, trace_floor: float = 1e-6):
    """List of (row, col, charge) for nonzero plaquette charges of q.

    margin   : ignore plaquettes within this many pixels of the border
    trace    : optional tr T_sigma; plaquettes where it is below trace_floor * max are dropped
               (zeros far from every wall, where the phase is undefined)
    """
    k = plaquette_charges(q)
    r, c = np.nonzero(k)
    keep = (r >= margin) & (c >= margin) & (r < k.shape[0] - margin) & (c < k.shape[1] - margin)
    if trace is not None:
        t = np.asarray(trace)
        tp = 0.25 * (t[:-1, :-1] + t[:-1, 1:] + t[1:, :-1] + t[1:, 1:])
        keep &= tp[r, c] >= trace_floor * t.max()
    return [(float(rr) + 0.5, float(cc) + 0.5, float(k[rr, cc])) for rr, cc in zip(r[keep], c[keep])]


def defects_at_scale(field, sigma: float, margin: int = 0, trace_floor: float = 1e-6):
    """Defects of the director of a 2D field at window sigma."""
    qs, tr = q_sigma(field, sigma)
    return defects(qs, margin=margin, trace=tr, trace_floor=trace_floor)
