# Reproduction

This directory contains the original handoff unchanged except for new research scripts, tests, requirements, this document, and results. The authority is `CODEX_TASK_KO.md` (now in `docs/ai_provenance/codex/`).

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

No GPU or connected service is needed. Expect several minutes on a modern CPU and approximately 250 MB of field checkpoints. Exact wall-clock times and float roundoff depend on the machine. The full runner serializes stages; do not run finalize/width concurrently with benchmark, because they augment its CSVs. For a single stage use `python reproduce.py --stage benchmark` (then finalize, width and report to refresh derived results).

All generated files go to `results/`. Every new simulation uses float32 fields, float64 summation for measurement tensors, seed 61116, and unit isotropic grid spacing. Seeds, geometry parameters, shifts, dimensions, model parameters, stopping residual and timing are in per-run CSVs. The new benchmark never imports the legacy benchmark module because that module executes projected-gradient experiments on import.

The provided script is separately reproduced at 56³ for 600 iterations, with all eight parameter/shape pairs and static/dynamic controls. Its own output has no convergence residual and is not the convergence certificate. `run_research.py` supplies that certificate at 48³; long/domain checks use 48³ and 64³; tilted validation uses 56³.

`results/kernels.json` fixes the selected measurement kernel and training/calibration identities. `results/environment.json` records the observed environment. `results/config.json` records the experiment scope. `results/manifest.sha256` hashes the delivered sources and results (excluding logs and caches).

Fields are compressed NPZ files with `n` and `mask`; `n[..., -1]` is the scalar and preceding channels follow array-axis order. For example:

```bash
python code/resume_checkpoint.py results/p0_01_torus_t0036.npz --steps 1000 --h 2 --D 1 --J 1 --out results/continued.npz
```

This continuation CLI uses the default central dynamic generator and alpha=1. Supply parameters explicitly: NPZ snapshots contain fields and mask, not the run configuration. CSV run IDs map checkpoints to their complete settings. The supplemental validation deliberately continues through 2,000 updates even after the stopping tolerance is reached.

Figures are PNG. Raw tables and kernels are CSV/JSON. `RESEARCH_REPORT.md` explains estimators, calibration, topology qualifications, failures, and limitations. Historical pilot data remain in `pilot/`; they are never combined with current model evidence.
