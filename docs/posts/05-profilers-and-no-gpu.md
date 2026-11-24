# Profiler and No-GPU Ablations

In this section, we examine two environmental ablations on NVIDIA B300 GPUs: enabling hardware profilers during optimization and observing 30 sessions that ran on a failed GPU.

## Profiler Ablation

Main benchmark sessions forbid profilers. To measure their effect, seven models solved all 30 kernels again with Nsight Compute (`ncu`) and Nsight Systems (`nsys`) enabled and stated in the task. Gemini 4 Neon ran the same comparison in a later round with an added test-timeout rule. Table 1 reports geomean ratios of attainment and list-price cost, with to without the profiler, split by whether the profiler actually ran.

| Model | Used (of 30) | Pairs (ran) | Speed ↑ (ran) | Price ↓ (ran) | Pairs (not ran) | Speed ↑ (not ran) | Price ↓ (not ran) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Opus 5.5 | 30 | 29 | 0.97 | 0.83 | — | — | — |
| Gemini 4 Neon† | 14 | 13 | 1.01 | 0.84 | 16 | 1.17 | 0.65 |
| GPT-6.1 Sol | 1 | 1 | 0.92 | 1.48 | 29 | 1.54 | 1.14 |
| Kimi K3 | 18 | 17 | 1.19 | 1.62 | 12 | 1.05 | 0.86 |
| GLM-5.2 | 2 | 2 | 0.93 | 3.92 | 28 | 1.01 | 0.97 |
| DeepSeek-V4-Flash-R | 16 | 14 | 0.36 | 2.21 | 12 | 2.15 | 1.22 |
| Nemotron 3 Ultra | 0 | — | — | — | 17 | 0.73 | — |
| Nemotron 3.5 Super | 0 | — | — | — | 24 | 1.48 | — |
| **All (excl. Neon†)** | **67** | **63** | **0.82** | **1.31** | **122** | **1.24** | **1.05** |

*Table 1. Profiler ablation: geomean ratios of attainment (Speed ↑) and list price (Price ↓), with to without the profiler, over pairs passing in both sessions. †Neon ran in a later round and is excluded from All. Dash (—) indicates unpriced models or empty groups.*

**Results:** Most models rarely run a profiler even when told it is available. Among the 63 pooled pairs where a profiler ran, kernels become faster by more than 10% about as often as slower: 23 faster and 22 slower. Even Opus 5.5, which runs the profiler in all 30 sessions, makes 9 of 29 kernels faster and 9 slower (geomean ratio 0.97). Gemini 4 Neon splits at 4 faster and 4 slower across its 13 profiler-used pairs. Across all 185 pooled pairs, profiler runs do not correlate with speedup (Spearman $\rho = -0.06$).

**Cost and confounding effects:** In every model with both session types, sessions that run a profiler have higher price ratios than those that do not, e.g., 1.62 against 0.86 for Kimi K3 and 2.21 against 1.22 for DeepSeek-V4-Flash-R. Because agents may invoke a profiler on harder kernels, this need not be a direct effect of profiling. Note also that on the 29 kernels where GPT-6.1 Sol never ran the profiler, its profiler-enabled session still made 18 kernels faster (geomean 1.54). An agent far below the bound has wide run-to-run variance, so comparing arms without splitting by actual profiler use can misattribute rerun variance to profiling.

## No-GPU Incident

During data collection, one GPU in our pool failed with CUDA error 46. Before we noticed and replaced the card, 30 sessions across six models ran on it: 27 entirely without a working GPU and 3 mostly without one. We report this observed incident without reproducing it and compare each session against its fresh re-run on a working B300 GPU (`data/no-gpu.csv`).

| Model | Faulty-GPU sessions | Passed Verifier | Representative attainment: no GPU → working GPU | Observed workaround |
| :--- | :---: | :---: | :--- | :--- |
| GLM-5.2 | 10 (9 full) | 9 / 10 | `gemm`: 89.5% → 89.3%; `2mm`: 82.6% → 82.8%; `3mm`: 83.9% → 84.1% | Calls cuBLAS and checks argument mapping on CPU; 6 sandbox-escape attempts |
| Qwen3.8-27B | 4 (4 full) | 4 / 4 | `gemm`: 89.6% → 88.8%; `2mm`: 82.7% → 82.8%; `trmm`: 7.0% → fail | Calls cuBLAS and checks argument mapping on CPU |
| DeepSeek-V4-Flash-R | 5 (5 full) | 2 / 5 | `2mm`: 82.6% → 43.9%; `doitgen`: 68.9% → 80.2% | Calls cuBLAS where applicable; 3 sandbox-escape attempts |
| GPT-6.1 Sol | 3 (3 full) | 3 / 3 | `jacobi-2d`: 77.0% → 172.8%; `adi`: 2.2% → 116.4%; `symm`: 60.5% → 88.0% | **Simulates kernels on CPU** and selects conservative tiling |
| Fable 5.1 | 5 (3 full) | 4 / 5 | `syr2k`: 72.5% → 175.1%; `seidel-2d`: 15.0% → 111.7% | **Simulates CUDA threads on CPU** and probes variants in untimed `pb_setup` |
| Sonnet 5.5 | 2 (2 full) | 2 / 2 | `2mm`: 82.1% → 453.8%; `3mm`: 32.0% → 511.8% | **Simulates CUDA threads on CPU** and times FP32 against cuBLAS emulation in `pb_setup` |

*Table 2. Summary of the 30 faulty-GPU sessions compared with working-GPU re-runs.*

**Results:** Without a GPU, 23 of 27 full sessions still pass the Verifier when evaluated on a working B300 GPU. Every model checks correctness on the CPU and estimates rather than measures performance. On 10 of 13 passing kernels, GLM-5.2 and Qwen3.8 stay within 10% of their working-GPU attainment by relying on cuBLAS.

However, no GPU-less session exceeds the roofline bound. Sonnet 5.5, Fable 5.1, and GPT-6.1 Sol submit 9 correct kernels in 10 sessions, reaching up to 82.1% of the bound (`Sonnet 5.5` on `2mm`, against 453.8% with a working GPU). Without runtime measurements, they miss optimizations such as deeper temporal blocking or hand-tuned tensor-core kernels.

**Sandbox escape attempts:** Facing CUDA error 46, GLM-5.2 in 6 sessions and DeepSeek-V4-Flash-R in 3 sessions attempted privileged recovery commands, including creating `/dev/nvidia*` device nodes (`mknod`, `chmod`), running `nvidia-modprobe`, entering new user namespaces (`unshare`), and writing to `/proc/driver/nvidia`. Container isolation blocked all attempts.
