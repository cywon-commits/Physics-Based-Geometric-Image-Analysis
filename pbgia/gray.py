"""Grayscale input.

Extent: the continuous source h (2g - 1), with g the intensity normalized between background and object,
is the average of the binary sources over all cut levels (layer-cake identity), so it transmits subpixel
boundary position to linear order.
Orientation: the director pools the transverse fields generated at several intensity quantiles.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from .generator import Field, relax

__all__ = ["otsu_threshold", "intensity_levels", "continuous_source", "relax_gray", "pooled_q0"]


def otsu_threshold(img, bins: int = 256) -> float:
    """Otsu threshold (between-class variance maximum) of an image, without extra dependencies."""
    x = np.asarray(img, np.float64).ravel()
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    mids = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * mids)
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    var = w0 * w1 * (m0 - m1) ** 2
    return float(mids[int(np.argmax(var))])


def intensity_levels(img, smooth: float = 1.0):
    """Background and object levels: medians of the two Otsu classes of a copy smoothed with `smooth`."""
    img = np.asarray(img, np.float64)
    sm = ndi.gaussian_filter(img, smooth)
    t = otsu_threshold(sm)
    return float(np.median(img[sm <= t])), float(np.median(img[sm > t]))


def continuous_source(img, levels=None) -> np.ndarray:
    """Source 2g - 1 in [-1, 1], with g the intensity clipped and normalized between the two levels."""
    img = np.asarray(img, np.float64)
    lo, hi = intensity_levels(img) if levels is None else levels
    g = np.clip((img - lo) / (hi - lo), 0, 1)
    return 2 * g - 1


def relax_gray(img, levels=None, **kw) -> Field:
    """Relax the generator with the continuous source of a grayscale image (float64 arrays)."""
    kw.setdefault("dtype", np.float64)
    return relax(continuous_source(img, levels), **kw)


def pooled_q0(img, quantiles=(0.15, 0.325, 0.5, 0.675, 0.85), smooth: float = 0.8, **kw) -> np.ndarray:
    """Pixel-level director field q0 of a grayscale image, pooled over binary fields generated at the
    given intensity quantiles after smoothing with `smooth` pixels (cuts covering <0.5% or >99.5% of
    the image are skipped)."""
    sm = ndi.gaussian_filter(np.asarray(img, np.float64), smooth)
    acc = None
    n = 0
    for q in np.unique(np.quantile(sm, quantiles)):
        m = sm > q
        if m.mean() < 0.005 or m.mean() > 0.995:
            continue
        f = relax(m, **kw)
        vr, vc = [np.asarray(c, np.float64) for c in f.v]
        z = (vc * vc - vr * vr) - 2j * vr * vc
        acc = z if acc is None else acc + z
        n += 1
    if n == 0:
        raise ValueError("no usable intensity cut")
    return acc / n
