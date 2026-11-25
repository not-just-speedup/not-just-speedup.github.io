# Analysis

Each page below expands on one part of our evaluation of coding agents on GPU HPC kernels, covering results and methodological checks that do not fit within a conference page limit.

- [1. Breaking the Roofline Bound](01-breaking-the-roofline.md): complete per-model precision and temporal-blocking tables, analysis of the 32 below-bound reduced-precision solutions, and how each model's sessions spend their steps and time.
- [2. Algorithm Changes and Reward Hacking](02-algorithms-vs-hacking.md): complete list of all 19 verified algorithm changes, the spectral DST-I derivation for `adi`, 13 Analyzer false positives, and the `floyd-warshall` and `ludcmp` reward-hacking cases.
- [3. Ranking and Wrap-Up](03-ranking-and-wrap-up.md): two model rankings, kernel optimization for hardware features alone on the 24 kernels without algorithm changes, and all 30 kernels with them.
- [4. Session Behavior](04-session-behavior.md): how each model's 30 sessions divide their steps and time, with the Analyzer's performance-engineering labels.
- [5. Profiler and No-GPU Ablations](05-profilers-and-no-gpu.md): 185 paired Nsight Compute/Systems sessions across seven models and 30 sessions from a CUDA error 46 GPU failure.
- [6. Bug Analysis](06-silent-gpu-bugs.md): all 102 failed sessions by where they were caught, and how agents handle bugs during a session (G5).
- [7. Bugs Beyond the Verifier](07-bugs-beyond-the-verifier.md): the bugs the Analyzer finds in solutions the Verifier passed, confirmed by the Bug Verifier: 77 claims, 56 reproduced defects, failure rates by model, root causes, and six CUDA case studies.
- [8. Analyzer Cross-Check: Claude against Codex](08-analyzer-reliability.md): Claude against Codex agreement on the Level-2 labels of 180 traces, over 17 rubric fields.
