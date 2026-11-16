# PBGIA Codex handoff — dynamic normal field (2026-11-16 revision)

**Required model:** dynamic `grad(s)` each iteration, fixed binary bias field `f`, neighbor exchange, no `div(v)`, and direct normalization `n_next=H/|H|`. See `CODEX_TASK_KO.md`.

Start: `python code/pbgia_dynamic_normal_test.py --out results.csv --steps 600`; dependencies: numpy scipy pandas scikit-image. For larger 3D volumes use checkpointing or sufficient runtime. The current pilot ran only selected torus cases and does not demonstrate optimization superiority or long-term stability.

`reference/` includes the two user-provided Colab source `.py` files and LaTeX research notes. `code/` includes past work for context, but their physical update rules are not authoritative. `pilot/pbgia_dynamic_normal_results.csv` is a partial early benchmark; see revision task for full validation requirements.
