# PBGIA manuscript and reproducibility release (2026-11-17)

This folder collects the integrated interdisciplinary manuscript and its figures, numerical results, and scripts. It is a working draft, not a journal-submitted version.

## Entry points
- `main.tex`, `refs.bib`, and `Makefile`: English manuscript sources.
- `main.pdf`: preview of the integrated manuscript.
- `figures/`: original vector PDF figures used by the LaTeX source.
- `code/`: pilot analysis, 2D/3D topology and stadium validation scripts.
- `data/`: output CSV/JSON from the numerical pilots and independent benchmarks.
- `source_images/`: CC0 `scikit-image` sample arrays where redistributable.

## Reproduction and scope
Install Python packages: `numpy`, `scipy`, `scikit-image`, `matplotlib`, `pandas`, and `pillow`. Run the scripts from this folder after adjusting source paths in older scripts referencing the original `/mnt/data` workspace. The scripts originated in an exploratory workspace and are supplied with their original data outputs; **not all commands are currently fully portable**. The canonical benchmark dataset is preserved in `final_pbgia_study/` of the associated repository (see PR #6).

Compiling the manuscript requires `pdflatex`, `bibtex`, and the bundled `figures/` PDFs.

The numerical experiments in this folder consist of synthetic 2D and 3D validation and CC0 real-image pilot analyses. Actual rock micro-CT data have not been analyzed in this release. Some `code/` scripts contain optional exploratory paths to third-party image samples not included here; see `SOURCE_IMAGE_RIGHTS.md` and `SOURCE_RIGHTS_NEW.md` before redistributing any media.

## Important context
- Frozen PBGIA geometry benchmark source revision: `b2107084d19fcdf54c9350cf3f0e013fed8afa4c`.
- The GitHub branch hosts a proposed release; main is not changed until the PR is merged.
- This directory avoids author information or funding claims that have not been confirmed.
