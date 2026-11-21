"""PBGIA: physics-based geometric image analysis.

A binary (or grayscale) image acts as a field on unit-vector spins; the relaxed spin texture gives
boundary extent, normals, a projective director with half-integer defects, and an integer degree equal
to the Euler characteristic of the generated domain.

Quick start
-----------
>>> import numpy as np, pbgia
>>> y, x = np.mgrid[:96, :96]
>>> disk = (x - 47.3) ** 2 + (y - 48.1) ** 2 <= 20 ** 2
>>> pbgia.perimeter(disk)                    # ~ 2 pi 20 = 125.66
>>> f = pbgia.relax(disk)                    # Field: f.v (transverse), f.s (longitudinal)
>>> pbgia.degree2d(f)                        # 1.0 (= Euler characteristic of a disk)
>>> pbgia.defects_at_scale(f, sigma=3)       # [] for a convex shape

Settings follow the paper (Methods): 2D readout h/J = 1.5, 3D readout h/J = 2, D/J = 1, N = 30.
"""
from .generator import DEFAULTS, Field, binary_source, initial_state, relax
from .readout import CONSTANTS, boundary_measure, curvature, perimeter, surface_area, tv_density
from .orientation import (BIAS, correct_bias, covariance_axis, director_angle, inward_normals, q0,
                          q_sigma, shape_axis)
from .defects import defects, defects_at_scale, plaquette_charges
from .topology import (degree2d, degree3d, degree_per_domain, euler_characteristic,
                       solid_angle_density)
from .gray import continuous_source, intensity_levels, otsu_threshold, pooled_q0, relax_gray
from .calibrate import calibrate_2d

__version__ = "0.1.0"

__all__ = [
    "DEFAULTS", "Field", "binary_source", "initial_state", "relax",
    "CONSTANTS", "boundary_measure", "curvature", "perimeter", "surface_area", "tv_density",
    "BIAS", "correct_bias", "covariance_axis", "director_angle", "inward_normals", "q0", "q_sigma",
    "shape_axis",
    "defects", "defects_at_scale", "plaquette_charges",
    "degree2d", "degree3d", "degree_per_domain", "euler_characteristic", "solid_angle_density",
    "continuous_source", "intensity_levels", "otsu_threshold", "pooled_q0", "relax_gray",
    "calibrate_2d",
]
