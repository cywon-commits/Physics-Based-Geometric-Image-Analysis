# PBGIA wall-matched derivative kernel: reproducible benchmark

2D/3D relaxed fields from binary masks only; A=2, h=2, epsilon=0.1, lambda=sqrt(2) pixels/voxels. 2D 140 updates, 3D 125 updates, projected gradient dynamics. Periodic roll used in exchange; domain far from boundaries. Derivative tensor excludes 6-pixel edge buffer.

Derivative measurement kernels are tuned with fixed exchange relaxation: this does not modify the spin dynamics or the complete wall energy.

Learned p=(d2,t1,t2)=(np.float64(-0.09480746830344577), np.float64(3.0205220446634856e-16), np.float64(0.0)); 5x5 derivative Dx factorizes into x difference and transverse smoothing. In 3D apply same 1D smoothing on both transverse axes.

Objective combines coefficient of variation of trace(T) at same shape and radius and axis angular error, plus mild regularization. Reference R14 circle and R8 sphere calibrate geometrical scale (not fit on test). Training radius 14; held-out 23; 3D transfer R8 sphere, R10 sphere and R7 capsule. Random orientation and subpixel translation.

CAUTION: finite steps, not rigorously converged; report residuals. Only 5 3D test items and small 2D corpus, not a claim of general superiority. Angle eigenvectors for elongated shapes correspond to smallest gradient eigenvalue. Shape-specific boundary curvature changes trace(T), and trace(T) is not the full wall energy.

| method       | split      |   n |   length_MARE_pct |   length_RMSE_pct |   angle_MAE_deg |   angle_RMSE_deg |   residual_max |
|:-------------|:-----------|----:|------------------:|------------------:|----------------:|-----------------:|---------------:|
| Scharr-like  | test       |   8 |          0.673756 |          1.08078  |        0.197128 |         0.221672 |     0.0017603  |
| Scharr-like  | train      |   6 |          0.706869 |          0.758923 |        0.189871 |         0.233329 |     0.00246209 |
| Scharr-like  | transfer3d |   3 |          0.518417 |          0.810166 |        0.219316 |         0.219316 |     0.00912355 |
| central      | test       |   8 |          0.92716  |          1.82532  |        1.08589  |         1.22116  |     0.0017603  |
| central      | train      |   6 |          0.676014 |          0.882051 |        0.678593 |         0.850365 |     0.00246209 |
| central      | transfer3d |   3 |          0.644877 |          0.869958 |        0.423769 |         0.423769 |     0.00912355 |
| fourth-order | test       |   8 |          0.658776 |          1.26679  |        0.381828 |         0.456223 |     0.0017603  |
| fourth-order | train      |   6 |          0.321866 |          0.349378 |        0.340417 |         0.416617 |     0.00246209 |
| fourth-order | transfer3d |   3 |          0.505846 |          0.802089 |        0.223426 |         0.223426 |     0.00912355 |
| learned      | test       |   8 |          0.638992 |          1.19746  |        0.283512 |         0.355876 |     0.0017603  |
| learned      | train      |   6 |          0.37802  |          0.399575 |        0.292634 |         0.366504 |     0.00246209 |
| learned      | transfer3d |   3 |          0.487746 |          0.798628 |        0.206679 |         0.206679 |     0.00912355 |
| smoothed-4th | test       |   8 |          0.636242 |          0.807813 |        0.945277 |         1.02651  |     0.0017603  |
| smoothed-4th | train      |   6 |          1.75749  |          1.85596  |        0.51865  |         0.599055 |     0.00246209 |
| smoothed-4th | transfer3d |   3 |          0.672236 |          0.861301 |        0.499316 |         0.499316 |     0.00912355 |