# Working notes for agents on this site (AGENTS.md is canonical; CLAUDE.md imports it)

This repository is the companion website of a paper under double-blind review. It holds the analyses,
figures and sanitized data that do not fit in the paper. Two agents maintain it (Claude Code and
Antigravity); the owner decides what is published.

## Anonymity (hard rule)

- Nothing in a tracked file may identify the authors or their institution: no personal names, user names,
  e-mail addresses, GitHub handles, local paths, institution names, or private share links.
  RiVault may be named; the institution behind it may not.
- Internal round names, case IDs and agent IDs are allowed.
- Run `python scripts/check_anonymity.py` before every commit. The script is local only and gitignored,
  because its pattern list names the authors: never commit it, never copy its patterns into a tracked file.
- Commits use the site's anonymous identity (`not-just-speedup <not-just-speedup@users.noreply.github.com>`).

## Facts

- The paper is the source of truth. Every number on the site must match the paper or be recomputable from
  `data/`. When you change a number, recompute it from `data/` and say how.
- The study has 28 models (Kimi K3.1 Pro was added on 2026-11-23); 840 sessions.
- Report faster and slower counts, not only a geomean near 1. Do not restate a table in the prose next to it.
- Define every label before using it (G1-G11, Level 1/2, Bug Verifier); prefer plain words.

## Build and data

- `data/attainment.csv`, `data/behavior.csv`, the bug tables and the bug figures come from
  `scripts/build_bug_analysis.py`; `data/cost.csv` and the cost figures from `scripts/build_cost.py`. Both read the
  paper repository and the trial records beside this checkout. Run `build_bug_analysis.py` first.
- Colors: mint and blue follow the paper; on the Cost page, red is list-price cost, blue-violet input tokens, plum output tokens, blue time (`palette.py`).
- `docs/assets/attainment.svg`, `behavior.svg`, `heat3d-sota.svg` and `overview.svg` are the paper's figures,
  converted with `pdftocairo -svg`.
- `docs/code/files/<case>/` holds published solutions exactly as frozen (`solution.cu`, `NOTES.txt` = the agent's NOTES.md)
  and the PolyBench reference kernel (`kernel.c`, plus a license header); never edit them. The pages in `docs/code/`
  include them with snippets.
- Check `mkdocs build --strict` before every commit.

## Git

- Commit only what you changed, with an explicit file list. Push only with the owner's OK.
- Pushing to `main` deploys the site (GitHub Pages workflow); hosting is not decided yet.
