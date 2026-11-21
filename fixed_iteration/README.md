# PBGIA fixed-iteration/kernel study

This is an executed fixed-iteration study following [CODEX_TASK_KO.md](../docs/ai_provenance/codex/CODEX_TASK_fixed_iteration_KO.md). Read [the Korean report](output/REPORT_KO.md) and [the frozen configuration](output/selected_configuration.json) for the selected N, kernels, accuracy, timing and limitations.

## Reproduce in one command

Python 3.12, CPU; no GPU required. Install once:

```bash
python -m pip install -r requirements.txt
```

Then reproduce all stages with:

```bash
python run_all.py --fresh
```

`--fresh` archives an existing output directory before a new run; it does not delete earlier results. Full runs take several minutes and generate roughly 1–2 GB of reproducible field caches. Every input is advanced to 40 once and its five requested checkpoints are reused. Runtime diagnostics, kernel training, disk I/O, and dedicated deployment repeats are recorded separately from the production single-N pipeline.

Without `--fresh`, the runner refuses to overwrite an already scored test set. Do not retune using the delivered test results. To independently evaluate a new hypothesis, predeclare a new split/protocol and preserve this run's locked configuration and test audit.

The scientific stages are `python code/run_fixed_study.py design|train|test|transfer`, followed by `python code/additional_checks.py` and `python code/report_fixed.py`. Tests: `python -m pytest -q tests`.

## Single-image inference

```bash
python code/predict_fixed.py mask.npy --out prediction.json
```

`mask.npy` is a 2D/3D boolean, 0/1 or ±1 array (`>0` means inside). Inference reads the frozen fixed N and dimension-specific measurement kernel from `output/selected_configuration.json`. It uses no adaptive stopping. Output includes a geometry estimate in unit pixels/voxel², tensor/eigengap, warning about orientation interpretation, and separately measured initialization/field/measurement times. The accompanying NPZ stores the field and binary f.

The supplied calibration is for unit isotropic spacing, J=1,h=2,D=1, float32, binary EDT initialization and sufficiently padded periodic arrays. It is not a physical-unit or natural-image guarantee.

## Model

`Hv=J*sum_axial(v_current)+D*central_gradient(s_current)`, `Hs=J*sum_axial(s_current)+h*f`, then synchronous direct normalization. No div(v), no grad(f) alignment, no Laplacian substitution. If |H|<=1e-12 retain the previous unit spin. Measurement tensor trace is not called an energy, and short-run geometry accuracy is not called field convergence.

## Artifacts

- `output/split_manifest.csv`, `design.json`: geometry-level split, seeds, and preregistered selection rule.
- `output/kernel_by_iteration.json`: equal-budget 3-parameter fits per N/dimension, train-only reference calibration, validation scores.
- `output/selected_configuration.json`, `selection_lock.sha256`, `test_audit.json`: frozen pre-test selection and audit.
- `checkpoint_metrics.csv`, `train_results.csv`, `validation_results.csv`, `test_results.csv`, `baseline_results.csv`, `timing.csv`: raw experiments.
- `paired_differences.csv`, `paired_bootstrap_ci.csv`, family summaries and sensitivity/Pareto tables: uncertainty, held-out family results and trade-offs.
- `output/figures/`, `wall_profiles.csv`, `refinement_margin_checks.csv`, `transfer_results.csv`: figures and diagnostic checks.
- `output/cache/`: content-keyed NPZ fields and exact measurement-basis Gram cache. Large caches are excluded from the lightweight deliverable; regenerate them from the included generators/manifests.
- `reference/`: the user-supplied prior package and historical material, kept separate from current numerical evidence. Previous workspace results were accessible but never pooled into this study.

Geometry prototypes are split before training. Rotation, shift and noise variants remain in their prototype's split. Rings (2D) and ellipsoids (3D) are family-held-out. Three noise rates (0,0.2%,1%) are reported separately. Bootstrap resamples independent geometry prototypes, not derivative images. Only connected elongated shapes have a defined full-gradient-tensor orientation target; separated circular/spherical objects do not.

Original handoff smoke files remain under `smoke_2d/` and `smoke_3d/`. `output/environment.json` and `code_version.json` record this run; `output/deliverable_manifest.sha256` records delivered sources/data excluding caches and transient logs. No license is asserted for the supplied historical research material.
