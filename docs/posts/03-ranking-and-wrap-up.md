# Ranking and Wrap-Up

[Analysis 1](01-breaking-the-roofline.md) and [Analysis 2](02-algorithms-vs-hacking.md) show two routes past the roofline bound: kernel optimization for hardware features, which keeps the reference algorithm, through lower-precision tensor cores, 16-bit DPX, and temporal blocking, and changes to the algorithm itself. In this section, we rank the models on each route.

We rank the 28 models twice. The first ranking measures kernel optimization for hardware features alone: it uses only the 24 kernels on which no agent changes the algorithm, so every solution there runs the reference algorithm and its attainment reflects how well it uses the GPU. The second ranking uses all 30 kernels and therefore also credits algorithm changes. Geomean is over passed solutions and mean is over all kernels with failures counted as zero, as in the paper; both are footer rows of the attainment figure in [Analysis 1](01-breaking-the-roofline.md).

## Ranking 1: Kernel Optimization for Hardware Features (24 Kernels Without Algorithm Changes)

<div class="njs-table" markdown="1">

| Rank | Model | Framework | Geomean | Mean | Passed (of 24) | Above the bound |
| ---: | :--- | :--- | ---: | ---: | :---: | :---: |
| 1 | Opus 5.5 | Claude Code | 161% | 258% | 23 | 12 |
| 2 | Sonnet 5.5 | Claude Code | 144% | 241% | 24 | 12 |
| 3 | Gemini 4 Neon | Antigravity | 136% | 203% | 24 | 15 |
| 4 | Fable 5.2 | Claude Code | 99% | 143% | 24 | 10 |
| 5 | Fable 5.1 | Claude Code | 99% | 127% | 23 | 11 |
| 6 | Gemini 4 Carbon | Antigravity | 78% | 106% | 23 | 9 |
| 7 | Gemini 4 Argon | Antigravity | 61% | 72% | 24 | 2 |
| 8 | Kimi K3 | OpenCode | 37% | 44% | 21 | 0 |
| 9 | Kimi K3.1 Pro | OpenCode | 36% | 50% | 24 | 0 |
| 10 | GPT-6.2 Astra | Codex | 31% | 63% | 24 | 4 |
| 11 | GPT-5.6 Terra | Codex | 27% | 46% | 23 | 0 |
| 12 | GLM-5.2 | OpenCode | 27% | 45% | 24 | 0 |
| 13 | GPT-6 Astra | Codex | 26% | 45% | 24 | 3 |
| 14 | Haiku 5.5 | Claude Code | 25% | 44% | 24 | 0 |
| 15 | DeepSeek-V4-Flash | OpenCode | 24% | 35% | 20 | 0 |
| 16 | GPT-6.1 Sol | Codex | 22% | 39% | 22 | 1 |
| 17 | MiMo-V2.6-Pro | OpenCode | 19% | 37% | 21 | 0 |
| 18 | GLM-5.3-Flash | OpenCode | 13% | 28% | 19 | 0 |
| 19 | Qwen3.8-27B | OpenCode | 11% | 27% | 21 | 0 |
| 20 | GPT-6 Luna | Codex | 11% | 27% | 21 | 0 |
| 21 | GPT-6 Sol | Codex | 8.1% | 25% | 24 | 0 |
| 22 | DeepSeek-V4-Flash-R | OpenCode | 6.6% | 20% | 22 | 0 |
| 23 | Nemotron 3 Ultra | OpenCode | 6.4% | 10% | 14 | 0 |
| 24 | Haiku 4.5 | Claude Code | 5.7% | 17% | 16 | 0 |
| 25 | GPT-6.2 Sol | Codex | 5.3% | 19% | 23 | 1 |
| 26 | Qwen3.6-35B-A3B | OpenCode | 4.5% | 5.4% | 11 | 0 |
| 27 | Nemotron 3.5 Super | OpenCode | 2.5% | 10% | 20 | 0 |
| 28 | Seed-OSS-36B | OpenCode | 2.1% | 1.0% | 9 | 0 |

</div>

*Table 1. Roofline attainment on the 24 kernels where no agent changes the algorithm, ranked by geomean. The six excluded kernels are `adi`, `durbin`, `heat-3d`, `jacobi-1d`, `jacobi-2d`, and `nussinov`. Above the bound counts passed solutions that exceed it, through lower-precision tensor cores, 16-bit DPX, or temporal blocking ([Analysis 1](01-breaking-the-roofline.md)).*

## Ranking 2: All Kernels, Including Algorithm Changes

<div class="njs-table" markdown="1">

| Rank | Model | Geomean | Mean | Algorithm changes | Rank in Ranking 1 | Change |
| ---: | :--- | ---: | ---: | :---: | :---: | :---: |
| 1 | Gemini 4 Neon | 198% | 974% | 5 | 3 | ↑2 |
| 2 | Opus 5.5 | 156% | 284% | 1 | 1 | ↓1 |
| 3 | Sonnet 5.5 | 151% | 344% | 1 | 2 | ↓1 |
| 4 | Fable 5.2 | 121% | 246% | 3 | 4 | — |
| 5 | Fable 5.1 | 100% | 141% | — | 5 | — |
| 6 | Gemini 4 Carbon | 80% | 116% | 2 | 6 | — |
| 7 | Gemini 4 Argon | 53% | 89% | — | 7 | — |
| 8 | GPT-6.2 Astra | 41% | 87% | 5 | 10 | ↑2 |
| 9 | Kimi K3 | 32% | 53% | — | 8 | ↓1 |
| 10 | DeepSeek-V4-Flash | 25% | 35% | — | 15 | ↑5 |
| 11 | Kimi K3.1 Pro | 25% | 43% | — | 9 | ↓2 |
| 12 | GPT-6 Astra | 24% | 58% | — | 13 | ↑1 |
| 13 | Haiku 5.5 | 23% | 50% | — | 14 | ↑1 |
| 14 | GPT-6.1 Sol | 23% | 63% | 1 | 16 | ↑2 |
| 15 | GLM-5.2 | 18% | 39% | — | 12 | ↓3 |
| 16 | GPT-5.6 Terra | 17% | 42% | — | 11 | ↓5 |
| 17 | MiMo-V2.6-Pro | 13% | 34% | — | 17 | — |
| 18 | GLM-5.3-Flash | 9.3% | 25% | — | 18 | — |
| 19 | DeepSeek-V4-Flash-R | 8.6% | 25% | — | 22 | ↑3 |
| 20 | Qwen3.8-27B | 8.2% | 25% | — | 19 | ↓1 |
| 21 | GPT-6.2 Sol | 7.9% | 46% | 1 | 25 | ↑4 |
| 22 | GPT-6 Luna | 7.3% | 25% | — | 20 | ↓2 |
| 23 | GPT-6 Sol | 7.2% | 28% | — | 21 | ↓2 |
| 24 | Haiku 4.5 | 6.2% | 15% | — | 24 | — |
| 25 | Nemotron 3 Ultra | 5.5% | 9.2% | — | 23 | ↓2 |
| 26 | Qwen3.6-35B-A3B | 5.0% | 7.4% | — | 26 | — |
| 27 | Seed-OSS-36B | 3.2% | 2.9% | — | 28 | ↑1 |
| 28 | Nemotron 3.5 Super | 2.9% | 11% | — | 27 | ↓1 |

</div>

*Table 2. Roofline attainment on all 30 kernels, ranked by geomean. Algorithm changes count the model's passed solutions that change the algorithm ([Analysis 2](02-algorithms-vs-hacking.md)); Change is the move from Ranking 1.*

**Results:** Kernel optimization for hardware features and algorithm changes can be separate skills. Opus 5.5 leads kernel optimization for hardware features at 161% of the bound, followed by Sonnet 5.5 at 144% and Gemini 4 Neon at 136%, while Neon leads overall at 198% because it changes the algorithm on five kernels. On the 24 kernels, 10 of Opus 5.5's 12 solutions above the bound are hand-written 5th-generation tensor-core kernels compiled at run time with NVRTC ([Analysis 1](01-breaking-the-roofline.md#breaking-the-compute-bound-lower-precision-tensor-cores-for-fp32-and-16-bit-dpx-for-int32)), at a geomean of 4.6× the bound, while 8 of Neon's 15 call cuBLAS in a narrower format, at 2.3×. GPT-6.2 Astra also changes the algorithm on five kernels, yet ranks only 10th in Ranking 1 at 31%, so its algorithm changes lift it to 8th overall without matching the top models on the other 24 kernels. Opus 5.5 and Sonnet 5.5 change the algorithm only once each and reach second and third place overall mainly through kernel optimization for hardware features. No open-weight model changes the algorithm or exceeds the bound on the 24 kernels; Kimi K3 leads them in Ranking 1 at 37%.

At the same time, the code behind these results is growing more complex and its algorithms less common: the fastest algorithm-changing solutions span 300 to 3,000 lines of CUDA ([Solution Code](../code/index.md)), and for `adi` we found no prior work applying the agents' method. Understanding how agents reach their results, and whether a result is correct and legitimate, may remain a research challenge for some time.
