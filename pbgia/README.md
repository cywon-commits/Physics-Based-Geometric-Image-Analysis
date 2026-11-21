# pbgia — physics-based geometric image analysis

A binary image acts as a magnetic field on unit-vector spins `n = (v, s)`. The relaxed spin texture
carries the geometry of the image: the longitudinal spin `s` changes sign on the boundary, the transverse
spin `v` points along the inward normal, its tensor defines a projective director with half-integer
defects, and the degree of `n` equals the Euler characteristic of the generated domain. This package
collects the generator and readouts used in the paper into one tested library.

## Install

```bash
pip install git+https://github.com/cywon-commits/Physics-Based-Geometric-Image-Analysis
# or, from a clone
pip install .
```

Dependencies: NumPy and SciPy.

## Quick start

```python
import numpy as np
import pbgia

y, x = np.mgrid[:96, :96]
disk = (x - 47.3) ** 2 + (y - 48.1) ** 2 <= 20 ** 2

pbgia.perimeter(disk)                 # 125.90 (exact 2*pi*20 = 125.66)

f = pbgia.relax(disk)                 # Field: f.v (transverse), f.s (longitudinal), f.n
pbgia.degree2d(f)                     # 1.0 = Euler characteristic of a disk
pbgia.covariance_axis(f)              # long axis (radians, counterclockwise from +x, y up)
pbgia.defects_at_scale(f, sigma=3)    # [(row, col, charge), ...] of the director at window 3
```

`examples/pbgia_quickstart.py` runs these readouts on a disk, an ellipse, a three-armed object and a
set of domains with holes.

## What is in the package

| Module | Contents |
|---|---|
| `generator` | `relax` (2D/3D, schedules `sync`, `redblack`, `damped`; `chiral_stencil` `standard`/`isotropic`), `initial_state`, `Field`, `DEFAULTS` |
| `readout` | `perimeter`, `surface_area`, `boundary_measure`, `tv_density`, `curvature`, fitted `CONSTANTS` |
| `calibrate` | `calibrate_2d`: refit the 2D readout constants for other generator settings |
| `orientation` | `q0`, `q_sigma` (director at window sigma), `shape_axis`, `covariance_axis`, `correct_bias` |
| `defects` | `plaquette_charges`, `defects`, `defects_at_scale` |
| `topology` | `degree2d`, `degree_per_domain`, `degree3d`, `euler_characteristic` |
| `gray` | `continuous_source`, `relax_gray`, `intensity_levels`, `otsu_threshold`, `pooled_q0` |

## Settings and constants

The readout constants are properties of the generator at fixed settings, not of the shapes measured.
The defaults are those of the paper: `D/J = 1`, `N = 30` updates, `h/J = 1.5` for the 2D readout and
`h/J = 2` for the 3D readout. `boundary_measure` refuses a field relaxed at other settings; use
`calibrate_2d(D=..., h=..., n_steps=...)` to obtain constants for them (about two seconds).

For directions (normals, directors, tilt), relax with `chiral_stencil="isotropic"`: the chiral term then
uses a transversely averaged difference (weights 1/6, 2/3, 1/6), which removes the lattice bias of the
normal at its source and lowers the per-pixel scatter. Do not apply `correct_bias` to such a field.
Extent readouts keep the standard stencil, for which their constants were fitted, and refuse the other.

Conventions: array axis 0 is the image row, axis 1 the column (axis 2 the slice in 3D); angles and
charges use x along columns and y up. Boundaries are periodic: pad images whose objects touch the border.

## Validation

`tests/test_pbgia_package.py` checks the package against the paper:

- the 2D benchmark (210 test shapes, clean and with 0.2% and 1% flips) is reproduced shape by shape to
  1e-6, with mean absolute relative errors 0.544%, 0.580% and 0.814%;
- `calibrate_2d` at the paper settings reproduces the published constants;
- the planar metrication is one on the mirror directions; isolated pixels vanish below `h = 2dJ`;
- clean benchmark masks are reproduced exactly at `h/J = 2`;
- the stadium covariance axis error is 0.214 deg; a three-armed object has three +1/2 and one -1/2
  defects; the degree equals the Euler characteristic in 2D (per domain) and in 3D (ball, solid torus).

```bash
pip install .[test]
python -m pytest tests/test_pbgia_package.py -q
```

## Limitations of version 0.1

- Pure NumPy on one CPU core: a 512 x 640 image takes about 0.2 s to relax (30 updates) and 0.04 s to read out;
  a 64^3 volume about 0.3 s. Large 3D volumes are slow.
- `calibrate_2d` covers the 2D readout; 3D constants are provided only for the paper settings.
- The bias correction in `shape_axis` is exact for straight walls; for the global axis of a single
  shape, `covariance_axis` is simpler and slightly more accurate.
