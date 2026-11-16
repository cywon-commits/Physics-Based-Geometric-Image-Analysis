# PBGIA 3D RP² defect reconstruction: pilot numerical validation

Date: 2026-11-16. Code: `pbgia_rp2_3d_test.py` plus execution wrappers.

## Synthetic data and numerical procedure

83³ voxels, 125 nonoverlapping spherocylinders on 5×5×5 array, grid spacing 13, radius 2, cylinder half-length 3, with optional independent center offset ±0.38 voxels. Input directions: uniform; director angle ±φ/2 (nontrivial Z₂ line); φ (trivial Z₂ class); and radial director (degree-one hedgehog). The input voxel mask is f=±1. Gradient-flow dynamics for 4-component unit vector n=(v_x,v_y,v_z,s) use A=2, h=1.5 or 8, ε=0.16, a finite-difference 6-neighbor Laplacian, Gaussian-filtered ∇f, dt=0.028, 55 projected gradient updates. The initial order parameter is inferred from the voxel mask's Euclidean distance transform, **not** the analytic shape SDF. This is a pilot, not a convergence study.

Spatial tensor Q_ij=Σ_a ∂i n_a ∂j n_a (or directly from f), Gaussian smoothing with σ=7,9,11 voxels. The director is the smallest-eigenvalue eigenvector because the capsule long axis has weakest normal-gradient response. We interpolate the **tensor**, not the signed eigenvectors, before eigen-decomposition at a query point.

Line classification: director sign holonomy on a circle of radius 20 voxels, z=center, 144 samples. +1 is trivial, -1 is nontrivial Z₂. For point hedgehog: sample a sphere radius 20, triangulate it, lift the sign of the director continuously over edges and sum oriented spherical triangle areas /4π. A valid integer degree requires **zero sign-lift inconsistencies**; otherwise the reported real number is INVALID as a topological degree.

## Observed results

Original voxel-gradient tensors (no magnetization relaxation) reproduce expected classes on all ideal patterns at σ=7 and 11. Hedgehog degree approximately ±1 with zero sign-lift conflicts.

At h=1.5 after 55 updates, the small capsules essentially disappear: positive-voxel recall for the +1/2 line = 0 and hedgehog ≈0.000075, despite overall voxel label agreement ~97.7% (dominated by the background). The line becomes trivial; hedgehog sign-lift becomes inconsistent, so no integer degree can be assigned.

At h=8, the positive-voxel recall for the +1/2 line ≈0.925 and hedgehog ≈0.881, and the desired Z₂ line classification and hedgehog degree=1 are recovered for σ=7,9,11. The jittered ±0.38 voxel cases also recover their class at σ=9; uniform and full-vortex controls remain trivial. h=3 and h=5 show intermediate preservation, indicating an operating regime depends critically on minimum feature size.

**Caveats:** h=8 implies λ=sqrt(2A/h)≈0.707 voxels, too narrow for accurate subvoxel interfacial geometry; this does not demonstrate accurate area measurement. Only one grid size, feature size, short relaxation, few deterministic geometries, one seeded jitter, and chosen test loops/spheres were studied. Further multi-radius, noise, full relaxation, and baseline comparisons are needed. Voxel-gradient tensor performs very well, so no advantage over ordinary image processing is claimed here.
