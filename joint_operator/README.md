# PBGIA joint operator / nematic study

This directory executes the user specification in [CODEX_TASK_KO.md](CODEX_TASK_KO.md). It learns generation derivatives, positive exchange operators, measurement kernels, orientation tensors/ensembles and fixed iteration counts in 2D and 3D. Previously learned coefficients are used **only as pinned controls**, not as initial learned solutions.

Read [the detailed Korean report](output/RESEARCH_REPORT_KO.md), [the pre-test selection](output/pretest_selection.json), and [the independent deployment acceptance gate](output/deployment_gate.json). A candidate that misses the declared shape/topology guard is experimental; the inference CLI retains PBGIA-30 as its default fallback rather than silently weakening the target.

## Current execution outcome

The frozen selection kept the axial+central generator rather than promoting a learned generator. In 2D, the new gradient observer at N=10 passed the shape guards: clean geometry MARE 1.184% versus PBGIA-30 1.073%, direction MAE 0.332° versus 0.370°, and measured median pipeline 5.4 ms versus 13.9 ms. This is a speed/direction–geometry trade-off, not universal improvement. No 3D production upgrade passed the locked criteria; the default retains PBGIA-30.

A promising independent 3D exchange kernel preserved validation shape (minimum clean Dice about .99845), but the original conservative hemisphere certificate blocked it. A **post-freeze, validation-only** active-face convex-hull diagnostic found positive interpolation-norm bounds on those representative fields. The locked model/test decisions were not revised and that alternative was not newly scored on the already used test set. This distinction is part of the negative/qualified findings.

The current run's total training time is preserved. Some fine-grained training expense rows were overwritten by a logging initializer; the append bug was fixed, missing timings were not invented, and the report documents the limitation. Fresh reproductions with the fixed source preserve that ledger.

## One-command reproduction

Install Python 3.12 dependencies once:

```bash
python -m pip install -r requirements.txt
```

Run:

```bash
python run_all.py --fresh
```

`--fresh` archives an existing output directory without deleting it. Without it, an already scored test set cannot be overwritten. CPU execution takes roughly 15–25 minutes on the recorded machine; hardware/load affect this. No GPU is required. The pipeline performs tests and supplied smoke, source crosschecks, controlled ablations, two-round equal-budget learning, full train/validation rechecks, frozen selection, independent test, diagnostics and report generation. Do not use delivered final-test data to retune and then call it an independent test again.

## Frozen inference

```bash
python code/infer_joint.py mask.npy --out prediction.json
python code/infer_joint.py mask.npy --experimental --out experimental.json
```

Inputs are 2D/3D boolean or 0/1/±1 masks (`>0` is inside), isotropic unit spacing. Output includes geometry, director axis/eigengap, input-mask fidelity, warnings and timing; NPZ stores the field. The default requires both validation recommendation and the predeclared final-test acceptance gate. `--experimental` reproduces the frozen learned candidate even when it is not recommended for production. It is explicitly labeled in output. Fixed N only; no adaptive stop.

The geometry scale is calibrated on independent training circles/spheres. The selected dimension-specific configuration, correlation sign, bounds, observer choice and normalization are explicit in JSON. Raw observed trace is not an energy.

## Model

`Hv = J*(KJ*v_current) + D*(Ggen*s_current)`, `Hs = J*(KJ*s_current) + h*f`, followed by synchronous normalization. `f` is frozen. No div(v), no fixed grad(f), no Laplacian substitution. Small H retains the old unit spin. The main initialization exactly matches the pinned binary-EDT PBGIA-30 source; historical binary initialization is a separate control.

E0 is axial positive exchange. E1 is the Hadamard coefficient square sum of derivative filters, with raw and scale-normalized controls. E2 is a learned independent nonnegative, inversion/axis-symmetric orbit mixture of self, nearest, second-axial and face-diagonal weights with sum 2d. New Ggen and Kobs are separately learned separable antisymmetric 5x5 / 5x5x5 families. Bond/direct/gradient/normalized-ensemble tensors and min/max conventions are learned without test-angle leakage.

## Data and audit trail

- `output/experiment_config.json`, `split_manifest.csv`: preregistered split/budgets/constraints. 51 independent units, with all rotations/shifts/noise derivatives grouped together. No radius-family tuple overlaps the prior GitHub split. Rings and ellipsoids are family-held-out.
- `trained_generator_kernels.json`, `trained_measurement_kernels.json`, `trained_models.json`: learned operators and N-specific observers, seeds/budgets/status.
- `pretest_selection.json`, its SHA, `test_audit.json`, `deployment_gate.json`: immutable parameter selection and separate independent-test acceptance/rejection.
- `factorial_ablation_raw.csv`, `orientation_observer_comparison.csv`, `generator_search.csv`, train/validation/test CSVs: actual numerical results.
- `kernel_J_tied_vs_free.csv`, `joint_pareto.csv`, `baseline_comparison.csv`, `topology_audit.csv`, `timing.csv`, paired geometry-unit bootstrap tables and family summaries.
- `output/figures/`: angular error/Dice, confidence, kernel maps/response, Pareto, profiles and failures.
- `output/checkpoints/`: generated representative state bundles; the full study can be regenerated from configs/manifests. NPZ checkpoints and archived runs are omitted from the GitHub submission and stored separately in the workspace.
- `vendor/provenance.json`: GitHub source commit and blob hashes. `vendor/pbgia30/` is a reference/control snapshot, not this study's training data.
- `reference/`: original user Python code and historical notes. Source functions are isolated by AST; importing the original Colab's top-level optimization loops is intentionally avoided.

Geometry, mask fidelity, finite-time residuals and compactified S3 degree are distinct observables. A small geometry error does not establish convergence. An open-hemisphere tetrahedron certificate is sufficient; failure is marked uncertified, not falsely reported as a valid integer degree. Conventional raw and train-calibrated geometry plus PCA/structure/surface tensors are all retained. Negative results and uncertainty from the small prototype count are part of the report.

`output/environment.json`, `code_version.json`, `stage_timings.json`, test logs and `deliverable_manifest.sha256` provide reproducibility metadata. No universal optimality, SO(3) invariance, BPS or exact physical DMI claim is made.
