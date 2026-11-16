# PBGIA hybrid Gaussian preprocessing experiment

## Methods
Input: 112×112 binary mask containing a disk (r=27), protrusions, two holes and an isolated small disk. Four independent 1.2% salt-and-pepper corruptions, seeds 800–803.

Pipelines: direct PBGIA; Gaussian smoothing of the ±1 image with sigma=0.8/1.2/1.8 followed by threshold at 0 and PBGIA; Gaussian plus threshold and direct Crofton perimeter. Fixed exchange A=2, symmetry-breaking EPS=0.1, h=1,2,4,8, 450 projected-gradient iterations. Same model and data as the preceding scale-space experiment. The core relaxation uses a finite-iteration stationary estimate, not a proof of global minimum.

For each h, energy is normalized with one clean disk r=27:
 L_energy=2*pi*27 * E_sample/E_reference_disk.
For comparison, the target of the irregular shape is its **clean digital Crofton perimeter** = 241.503985 pixels, not an analytically known continuum perimeter. We also compute E_sample/E_clean_same_shape - 1 to isolate energy changes from noise/preprocessing.

## Main results
At h=4 and Gaussian sigma=0.8, the mean energy-derived perimeter absolute error:
- Direct PBGIA: 54.859%.
- Hybrid: 5.901%.
- Gaussian baseline using Crofton: 4.706%.
The corresponding mean energy change from the clean same-shape model:
- Direct: +58.564%.
- Hybrid: -3.650%.

The best hybrid for energy MAPE over tested values was h=4, sigma=0.8, energy MAPE=5.901%.

**Interpretation:** Gaussian prefiltering removes isolated pixel fluctuations that otherwise contribute excess field/exchange energy. It also removes genuine small structures. Thus the hybrid greatly improves energy perimeter relative to noisy direct PBGIA but does not beat tuned Gaussian+Crofton on this synthetic dataset.

**Caveats:** The Gaussian width is not independently fitted; four noise replicas, a single mask family, fixed 450 iterations. The clean reference boundary's Crofton estimate itself has discretization error, and h=8 has lambda=0.707 pixel (underresolved). Do not infer broad superiority.

## Included files
- pbgia_hybrid_benchmark.py (regenerates CSV)
- pbgia_scale_space_experiment.py (physical model functions)
- pbgia_hybrid_results.csv: case-by-case
- pbgia_hybrid_summary.csv
- pbgia_gaussian_baseline.csv
- pbgia_hybrid_comparison.png, pbgia_hybrid_montage.png.
