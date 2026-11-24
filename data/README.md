# Datasets

All CSV and JSON files behind this site live in the `data/` directory. Personal identifiers, local filesystem paths, and internal review annotations are removed before publication and checked automatically by `scripts/check_anonymity.py`.

## Roofline Attainment (`data/attainment.csv`)

Per-(model, kernel) scoring results for all 28 models, 840 rows.

| Column | Description |
| :--- | :--- |
| `case`, `kernel`, `category` | PolyBench/C 4.2.1 case ID, kernel name, and suite category |
| `agent`, `model`, `framework`, `weight_class` | Agent ID, display name, harness, and `closed` or `open` |
| `state`, `time_ms`, `attainment` | Final outcome after Bug Verification, scored runtime (ms), and roofline attainment |
| `binding`, `compute_peak` | Binding resource (`compute` or `hbm`) and nominal working precision (`FP32`, `FP64`, `INT32`) |
| `tensor_cores`, `dpx`, `emulation`, `precision_kinds` | Detected reduced-precision or 16-bit DPX features |
| `temporal_blocking`, `algorithm_changed` | Verified temporal blocking and algorithm changes |
| `bug_marked`, `bug_verification` | Borderline bug flag and sanitized Bug Verifier verdict |

## Bugs Dataset (`data/bugs.csv`, `data/bugs_summary.json`) {#bugs-dataset}

Structured ledger of all 77 Bug Verifier claims across 67 Level-1 passing solutions (`data/bugs.csv`), together with per-model, per-root-cause, and per-kernel aggregates (`data/bugs_summary.json`).

| Column | Description |
| :--- | :--- |
| `agent`, `model`, `framework`, `weight_class` | Model and harness identifiers |
| `case`, `kernel`, `kernel_category`, `binding` | Affected kernel metadata |
| `claim_id`, `verdict` | Claim ID and verdict (`reproduced`, `not-reproduced`, `not-a-bug`, `cannot-test`) |
| `bug_class`, `scoring_outcome` | `computation`, `engineering`, or `waived`, and final scoring bucket |
| `root_cause` | One of eight GPU/CUDA root-cause categories |
| `summary`, `borderline_note` | Concise defect summary and note for kept borderline bugs |

## Cost Dataset (`data/cost.csv`) {#cost-dataset}

Per-session tokens, list-price cost, and time for all 840 sessions, the values behind Table 1 of the paper.

| Column | Description |
| :--- | :--- |
| `agent`, `model`, `framework`, `weight_class`, `case`, `kernel`, `category` | Session identifiers |
| `state`, `attainment`, `bug_marked` | Final outcome, roofline attainment, and the bug-marked flag of the paper's Figure 2 |
| `input_tokens`, `cache_read_tokens`, `cache_write_tokens`, `input_tokens_total` | Fresh input, cache reads, cache writes, and their sum |
| `output_tokens`, `reasoning_tokens` | Billed output (reasoning included) and the reasoning part of it |
| `usd` | List-price cost (empty for unpriced models) |
| `active_min`, `llm_min`, `tool_min` | Active time $T_{e2e}$, time waiting on the model, and time in tools (minutes) |

## Behavior and Ablation Tables

- `data/behavior.csv`: per-session (840 rows) step counts from deterministic `steps.v3` preprocessing, active times ($T_{e2e}$, $T_{llm}$, $T_{tool}$), parameter-search command shares, context compactions, and Analyzer labels.
- `data/profiler-pairs.csv`: 185 paired sessions across seven models with Nsight Compute and Nsight Systems enabled against forbidden.
- `data/no-gpu.csv`: 30 sessions executed during the CUDA error 46 faulty-GPU incident, paired with fresh working-GPU re-runs.

## Solution Code {#solution-code}

[Solution Code](../docs/code/index.md) publishes, for every kernel where a solution changes the algorithm, the fastest such solution, plus the fastest time-stepping `heat-3d` solution: the PolyBench reference kernel, the agent's frozen `solution.cu`, and its `NOTES.md`, unmodified.

## Figures

The attainment, behavior, `heat-3d`, and framework overview figures are the paper's own, converted to SVG (the overview without its section numbers). The bug and agreement figures are drawn by `scripts/build_bug_analysis.py` and the cost figures by `scripts/build_cost.py`, with one palette module (`scripts/palette.py`) in Linux Libertine through TikZ. Both scripts read our internal trial records, so they document how the published tables were derived but cannot be run from this repository alone.
