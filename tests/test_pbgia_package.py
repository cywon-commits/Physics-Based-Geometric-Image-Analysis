"""Regression tests of the pbgia package against results reported in the paper.

Run with:  python -m pytest tests/test_pbgia_package.py -q
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import pbgia  # noqa: E402
from pbgia_fixtures.benchmark_shapes import make2d  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "pbgia_fixtures")
FAM2 = ["disk", "ellipse", "stadium", "ring", "flower", "pair", "near_touch"]


def benchmark_jobs_2d(split="test"):
    """Shape seeds of the paper's 2D benchmark (8 training and 30 test shapes per family)."""
    seed = 20261118
    jobs = []
    for fam in FAM2:
        for sp, nn in (("train", 8), ("test", 30)):
            for i in range(nn):
                seed += 1
                if sp == split:
                    jobs.append((fam, i, seed))
    return jobs


def stadium(n=160, R=17, L=34, theta=0.0):
    yy, xx = np.mgrid[:n, :n]
    xx = xx - (n - 1) / 2
    yy = -(yy - (n - 1) / 2)
    a = np.deg2rad(theta)
    lo = np.cos(a) * xx + np.sin(a) * yy
    tr = -np.sin(a) * xx + np.cos(a) * yy
    return R - np.hypot(np.maximum(np.abs(lo) - L / 2, 0), tr) >= 0


def disk(n, cy, cx, R):
    y, x = np.mgrid[:n, :n]
    return (y - cy) ** 2 + (x - cx) ** 2 <= R * R


# ---------------------------------------------------------------- generator
def test_unit_spins_and_schedules_share_fixed_points():
    m = disk(64, 31.4, 32.2, 14)
    a = pbgia.relax(m, h=1.5, n_steps=200)
    b = pbgia.relax(m, h=1.5, n_steps=200, schedule="redblack")
    assert np.allclose(np.linalg.norm(a.n, axis=-1), 1, atol=1e-5)
    assert np.max(np.abs(a.s - b.s)) < 1e-4


def test_sign_rule_isolated_pixel():
    """An isolated pixel survives only if h > 2dJ (= 4 in 2D)."""
    m = np.zeros((21, 21), bool)
    m[10, 10] = True
    assert pbgia.relax(m, h=1.5).s[10, 10] < 0
    assert pbgia.relax(m, h=5.0).s[10, 10] > 0


def test_planar_metrication_is_one_on_mirror_directions():
    L = 120
    y, x = np.mgrid[:L, :L] + 0.5
    for p, q in ((0, 1), (1, 1)):
        s = pbgia.relax(((-p * x + q * y) % L) < L / 2, h=2.0).s
        m = pbgia.tv_density(s).sum() / (2 * L * math.hypot(p, q))
        assert abs(m - 1) < 1e-6


def test_clean_masks_reproduced_at_reference_setting():
    for fam, i, seed in benchmark_jobs_2d("test"):
        m, _, _ = make2d(fam, np.random.default_rng(seed))
        assert np.array_equal(pbgia.relax(m, h=2.0).s > 0, m), (fam, i)


# ---------------------------------------------------------------- readout
def test_benchmark_2d_reproduces_paper():
    """Clean and noisy 2D benchmark: per-shape values equal the paper's to 1e-6; MARE 0.54%, 0.58%, 0.81%."""
    ref = pd.read_csv(os.path.join(FIX, "bench_test2d_paper.csv"))
    rows = []
    for fam, i, seed in benchmark_jobs_2d("test"):
        rng = np.random.default_rng(seed)
        m0, T, _ = make2d(fam, rng)
        for p in (0.0, 0.002, 0.01):
            m = m0 ^ (rng.random(m0.shape) < p) if p > 0 else m0
            rows.append(dict(family=fam, inst=i, noise=p, truth=T, P=pbgia.perimeter(m)))
    d = pd.DataFrame(rows).merge(ref, on=["family", "inst", "noise"])
    assert np.max(np.abs(d.P / d.PBGIA_opt - 1)) < 1e-6
    d["e"] = 100 * np.abs(d.P / d.truth_x - 1)
    mare = d[d.family != "near_touch"].groupby(["noise", "family"]).e.mean().groupby("noise").mean()
    assert mare[0.0] == pytest.approx(0.544, abs=0.001)
    assert mare[0.002] == pytest.approx(0.580, abs=0.001)
    assert mare[0.01] == pytest.approx(0.814, abs=0.001)


def test_calibration_reproduces_constants():
    C = pbgia.calibrate_2d(D=1.0, h=1.5, n_steps=30)
    K = pbgia.CONSTANTS["2d_direction"]
    assert np.allclose(C["c"], K["c"], atol=1e-9)
    assert np.allclose(C["b"], K["b"], atol=1e-5)


def test_readout_refuses_mismatched_settings():
    f = pbgia.relax(disk(64, 31.5, 32.0, 12), h=2.0)
    with pytest.raises(ValueError):
        pbgia.boundary_measure(f, "2d_direction")


def test_disk_perimeter():
    P = pbgia.perimeter(disk(96, 47.3, 48.1, 20))
    assert abs(P / (2 * np.pi * 20) - 1) < 0.005


# ---------------------------------------------------------------- orientation
def test_stadium_covariance_axis_matches_paper():
    """Twelve rotations of an R = 17 stadium: mean axis error 0.214 degrees (Supplementary Table 3)."""
    errs = []
    for th in range(0, 180, 15):
        a = np.degrees(pbgia.covariance_axis(pbgia.relax(stadium(theta=th), h=2.0)))
        errs.append(abs((a - th + 90) % 180 - 90))
    assert np.mean(errs) == pytest.approx(0.2141, abs=1e-3)


def test_stadium_tensor_axis_with_bias_correction():
    errs = []
    for th in range(0, 180, 15):
        a = np.degrees(pbgia.shape_axis(pbgia.relax(stadium(theta=th), h=1.5)))
        errs.append(abs((a - th + 90) % 180 - 90))
    assert max(errs) < 0.2


# ---------------------------------------------------------------- defects and topology
def three_arm(n=120):
    y, x = np.mgrid[:n, :n] + 0.0
    cx, cy = 60.3, 59.6
    m = np.hypot(x - cx, y - cy) <= 9
    for a in (np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3):
        ux, uy = np.cos(a), -np.sin(a)
        t = (x - cx) * ux + (y - cy) * uy
        d = np.abs(-(x - cx) * uy + (y - cy) * ux)
        m |= (t >= 0) & (t <= 38) & (d <= 7)
    return m


def test_three_arm_defects():
    """One +1/2 defect per end cap and one -1/2 at the branch point; total = winding of the outline."""
    D = pbgia.defects_at_scale(pbgia.relax(three_arm()), sigma=3)
    assert sorted(k for *_, k in D) == [-0.5, 0.5, 0.5, 0.5]


def test_degree_equals_euler_characteristic_2d():
    n = 160
    y, x = np.mgrid[:n, :n]
    d1 = disk(n, 40.3, 40.6, 22)
    ann = disk(n, 41.2, 118.7, 26) & ~disk(n, 41.2, 118.7, 12)
    two_holes = disk(n, 116.4, 80.1, 36) & ~disk(n, 116.4, 62.0, 9) & ~disk(n, 116.4, 98.3, 9)
    m = d1 | ann | two_holes
    f = pbgia.relax(m)
    assert pbgia.degree2d(f) == pytest.approx(pbgia.euler_characteristic(m), abs=1e-4)
    _, per = pbgia.degree_per_domain(f)
    assert sorted(round(q, 3) for q in per) == [-1.0, 0.0, 1.0]


def test_degree_3d_ball_and_torus():
    z, y, x = np.indices((40, 40, 40)) + 0.0
    ball = (z - 19.6) ** 2 + (y - 20.2) ** 2 + (x - 19.9) ** 2 <= 11 ** 2
    torus = (np.hypot(y - 20.1, x - 19.8) - 12) ** 2 + (z - 19.7) ** 2 <= 4.5 ** 2
    assert pbgia.degree3d(pbgia.relax(ball))["degree"] == 1
    assert pbgia.degree3d(pbgia.relax(torus))["degree"] == 0


# ---------------------------------------------------------------- grayscale
def test_continuous_source_of_binary_image_equals_binary_field():
    m = disk(64, 30.6, 33.1, 15)
    img = np.where(m, 0.9, 0.1)
    a = pbgia.relax_gray(img, levels=(0.1, 0.9), h=1.5)
    b = pbgia.relax(m, h=1.5, dtype=np.float64)
    assert np.max(np.abs(a.s - b.s)) < 1e-12


def test_otsu_separates_two_levels():
    rng = np.random.default_rng(1)
    img = np.where(disk(80, 40, 40, 20), 0.8, 0.2) + rng.normal(0, 0.02, (80, 80))
    lo, hi = pbgia.intensity_levels(img)
    assert abs(lo - 0.2) < 0.02 and abs(hi - 0.8) < 0.02


def test_isotropic_chiral_stencil_removes_normal_bias():
    """Straight walls at 22.5 deg - phi: the isotropic chiral stencil removes most of the lattice bias,
    and the standard stencil is unchanged by the new option."""
    import numpy as np
    from pbgia.generator import relax
    L = 240
    y, x = np.mgrid[:L, :L] + 0.5
    p, q = 1, 3                                   # primitive normal, phi = atan(1/3) = 18.4 deg
    m = ((-p * x + q * y) % L) < L / 2
    true = np.arctan2(q, -p)
    out = {}
    for st in ("standard", "isotropic"):
        F = relax(m, h=1.5, chiral_stencil=st)
        vr, vc = [np.asarray(c, np.float64) for c in F.v]
        w = np.hypot(vr, vc)
        Q = [(z * w).sum() for z in (vr * vr, vc * vc, vr * vc)]
        ph = 0.5 * np.arctan2(2 * Q[2], Q[1] - Q[0])
        out[st] = abs(np.degrees(np.angle(np.exp(2j * (ph - true))) / 2))
    assert out["standard"] > 1.0
    assert out["isotropic"] < 0.5 * out["standard"]
    F0 = relax(m, h=1.5)
    F1 = relax(m, h=1.5, chiral_stencil="standard")
    assert np.array_equal(F0.s, F1.s)


def test_isotropic_chiral_stencil_3d_runs():
    import numpy as np
    from pbgia.generator import relax
    z, y, x = np.mgrid[:32, :32, :32] - 15.5
    F = relax(x * x + y * y + z * z < 100, chiral_stencil="isotropic")
    assert F.s.shape == (32, 32, 32) and np.isfinite(F.s).all()
