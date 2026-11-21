"""pbgia quick start: perimeter, axis, defects and degree of synthetic shapes.

    python examples/pbgia_quickstart.py
"""
import numpy as np

import pbgia

n = 160
y, x = np.mgrid[:n, :n] + 0.0


def disk(cy, cx, R):
    return (y - cy) ** 2 + (x - cx) ** 2 <= R * R


# 1. Perimeter of a disk (exact 2 pi R)
R = 20
P = pbgia.perimeter(disk(47.3, 48.1, R))
print(f"disk R={R}: perimeter {P:.2f} (exact {2 * np.pi * R:.2f}, error {100 * (P / (2 * np.pi * R) - 1):+.2f}%)")

# 2. Axis of a tilted ellipse (y up, degrees counterclockwise from +x)
theta = np.deg2rad(32.0)
u = (x - 80.2) * np.cos(theta) - (y - 79.6) * np.sin(theta)
v = (x - 80.2) * np.sin(theta) + (y - 79.6) * np.cos(theta)
ellipse = (u / 40) ** 2 + (v / 16) ** 2 <= 1
f = pbgia.relax(ellipse)                       # 2D default: h/J = 1.5, D/J = 1, 30 updates
print(f"ellipse axis: covariance {np.degrees(pbgia.covariance_axis(f)):.2f} deg, "
      f"normal tensor {np.degrees(pbgia.shape_axis(f)):.2f} deg (true 32.00)")

# 3. Director defects of a three-armed object at window sigma = 3
cx, cy = 80.3, 79.6
arms = np.hypot(x - cx, y - cy) <= 9
for a in np.pi / 2 + np.array([0, 2, 4]) * np.pi / 3:
    t = (x - cx) * np.cos(a) - (y - cy) * np.sin(a)
    d = np.abs((x - cx) * np.sin(a) + (y - cy) * np.cos(a))
    arms |= (t >= 0) & (t <= 50) & (d <= 7)
D = pbgia.defects_at_scale(pbgia.relax(arms), sigma=3)
print("three-armed object defects (row, col, charge):", [(round(r), round(c), k) for r, c, k in D])

# 4. Degree = Euler characteristic, per domain
m = disk(40, 40, 22) | (disk(40, 118, 26) & ~disk(40, 118, 12))
m |= disk(116, 80, 36) & ~disk(116, 62, 9) & ~disk(116, 98, 9)
f = pbgia.relax(m)
_, per = pbgia.degree_per_domain(f)
print(f"total degree {pbgia.degree2d(f):.4f}, Euler characteristic {pbgia.euler_characteristic(m)}, "
      f"per domain {[round(q, 3) for q in per]}")
