# Not-Just-Speedup

This repository contains the companion analyses, figures, living model updates, and sanitized datasets for our study of how coding agents develop and optimize GPU high-performance computing (HPC) kernels.

Existing evaluations of coding agents mostly report end-to-end speedups under fixed budgets. Speedup alone does not explain why an agent reaches high performance or fails to do so. In this site, we document the post-verification bugs, roofline breakdowns, algorithm changes, profiler and hardware ablations, trace-analyzer audits, and post-freeze model evaluations behind our study of 28 models across four harnesses on NVIDIA B300 GPUs.

## Analysis

1. [Breaking the Roofline Bound](docs/posts/01-breaking-the-roofline.md): per-model tables for reduced precision, 16-bit DPX, NVRTC, and temporal blocking.
2. [Algorithm Changes and Reward Hacking](docs/posts/02-algorithms-vs-hacking.md): all 19 verified algorithm changes, 13 false positives, and two reward-hacking incidents.
3. [Ranking and Wrap-Up](docs/posts/03-ranking-and-wrap-up.md): model rankings for kernel optimization for hardware features alone, and with algorithm changes.
4. [Session Behavior](docs/posts/04-session-behavior.md): how each model's sessions divide their steps and time.
5. [Profiler and No-GPU Ablations](docs/posts/05-profilers-and-no-gpu.md): 185-pair Nsight Compute/Systems ablation and 30 sessions from a faulty-GPU incident.
6. [Bug Analysis](docs/posts/06-silent-gpu-bugs.md): all failures by where they were caught, and how agents handle bugs (G5).
7. [Bugs Beyond the Verifier](docs/posts/07-bugs-beyond-the-verifier.md): bugs the Analyzer finds in solutions the Verifier passed, confirmed by the Bug Verifier, with verbatim CUDA cases.
8. [Analyzer Cross-Check: Claude against Codex](docs/posts/08-analyzer-reliability.md): Claude against Codex agreement on the Level-2 labels of 180 traces, over 17 rubric fields.

## Cost

[Cost, Tokens, and Time per Session](docs/cost.md): input and output tokens, active time, and list-price cost of all 840 sessions, per kernel and model.

## Stencil Comparison

[Comparison Against Hand-Built Stencil Implementations](docs/stencil.md): agent `heat-3d` solutions against AN5D, Bricks, EBISU, DRStencil, and Devito on B300 GPUs.

## Inspection Framework

[Inspection Framework](docs/framework.md): design principles, components, and the correctness, attainment, and cost metrics.

## Repository Layout

- `docs/`: MkDocs Material source pages and vector SVG figures.
- `data/`: sanitized CSV and JSON datasets (`attainment.csv`, `bugs.csv`, `bugs_summary.json`, `behavior.csv`, `cost.csv`, `profiler-pairs.csv`, `no-gpu.csv`); columns are described in `data/README.md`.
- `scripts/`: unified palette (`palette.py`) and TikZ/Libertine figure generators (`build_bug_analysis.py`, `build_cost.py`). The pre-push anonymity check (`check_anonymity.py`) is kept locally and not tracked.
- `.github/workflows/pages.yml`: GitHub Pages build and deployment workflow.

## Local Preview (Anaconda)

```bash
conda env create -f environment.yml
conda activate njs-docs
python scripts/check_anonymity.py   # local only
mkdocs serve
```

Note that GitHub Actions and Cloudflare Pages read `requirements.txt` during cloud builds. The scripts under `scripts/` use only the Python standard library plus TeX Live (`pdflatex` and `pdftocairo`) to rebuild the figures.
