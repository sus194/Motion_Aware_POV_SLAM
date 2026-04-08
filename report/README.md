# Report

This folder contains the LaTeX report for the simulation-only motion-aware POV-SLAM prototype.

Build from this directory if a LaTeX distribution is installed:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The report expects the generated figures to exist in `../outputs/figures/`.
