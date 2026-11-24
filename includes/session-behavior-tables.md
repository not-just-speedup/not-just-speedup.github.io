## Session Statistics

Measured by deterministic trace preprocessing, without an LLM. Each value is a per-session mean over the model's 30 sessions unless stated otherwise.


### Steps by Activity

<div class="njs-table" markdown="1">

| Model | Steps | Implement | Debug | Check | Perf. measure | Perf. analysis | Env. probe | Read task | Wrap-up | Other |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fable 5.2 | 452 | 37% | 5% | 14% | 12% | 13% | 2% | 2% | 1% | 14% |
| Fable 5.1 | 249 | 39% | 13% | 12% | 4% | 18% | 3% | 3% | 1% | 6% |
| Opus 5.5 | 752 | 49% | 8% | 13% | 7% | 10% | 1% | 1% | 1% | 8% |
| Sonnet 5.5 | 974 | 41% | 9% | 13% | 9% | 8% | 1% | 2% | 1% | 13% |
| Haiku 5.5 | 180 | 29% | 9% | 24% | 3% | 7% | 2% | 8% | 7% | 10% |
| Haiku 4.5 | 50 | 10% | 18% | 31% | 0% | 0% | 0% | 28% | 9% | 4% |
| Gemini 4 Neon | 219 | 40% | 13% | 11% | 2% | 11% | 5% | 10% | 0% | 8% |
| Gemini 4 Carbon | 132 | 33% | 10% | 23% | 1% | 6% | 4% | 14% | 0% | 8% |
| Gemini 4 Argon | 154 | 28% | 10% | 29% | 2% | 6% | 1% | 12% | 0% | 12% |
| GPT-6.2 Astra | 349 | 10% | 5% | 30% | 2% | 1% | 3% | 1% | 0% | 16% |
| GPT-6 Astra | 186 | 13% | 4% | 20% | 1% | 1% | 3% | 5% | 3% | 9% |
| GPT-6.2 Sol | 236 | 10% | 4% | 20% | 2% | 0% | 7% | 1% | 2% | 16% |
| GPT-6.1 Sol | 242 | 14% | 4% | 15% | 2% | 1% | 3% | 4% | 2% | 12% |
| GPT-6 Sol | 149 | 11% | 5% | 22% | 0% | 0% | 10% | 11% | 4% | 8% |
| GPT-5.6 Terra | 444 | 10% | 7% | 16% | 1% | 1% | 4% | 13% | 3% | 13% |
| GPT-6 Luna | 47 | 15% | 12% | 18% | 0% | 0% | 8% | 30% | 16% | 1% |
| Kimi K3.1 Pro | 107 | 37% | 7% | 22% | 1% | 0% | 2% | 17% | 13% | 1% |
| Kimi K3 | 198 | 33% | 24% | 19% | 2% | 7% | 1% | 9% | 2% | 3% |
| GLM-5.2 | 120 | 14% | 11% | 39% | 6% | 1% | 2% | 20% | 3% | 4% |
| GLM-5.3-Flash | 418 | 16% | 39% | 13% | 1% | 2% | 1% | 5% | 4% | 19% |
| DeepSeek-V4-Flash | 175 | 19% | 34% | 21% | 1% | 6% | 1% | 12% | 2% | 3% |
| DeepSeek-V4-Flash-R | 188 | 21% | 34% | 19% | 1% | 8% | 0% | 11% | 2% | 4% |
| Qwen3.8-27B | 199 | 29% | 31% | 16% | 2% | 2% | 1% | 11% | 3% | 5% |
| Qwen3.6-35B-A3B | 59 | 11% | 35% | 12% | 0% | 0% | 1% | 31% | 5% | 4% |
| MiMo-V2.6-Pro | 164 | 23% | 24% | 23% | 1% | 3% | 1% | 13% | 5% | 6% |
| Seed-OSS-36B | 54 | 5% | 49% | 14% | 0% | 0% | 0% | 28% | 3% | 1% |
| Nemotron 3 Ultra | 61 | 4% | 28% | 40% | 0% | 0% | 0% | 20% | 6% | 1% |
| Nemotron 3.5 Super | 41 | 13% | 12% | 23% | 0% | 0% | 2% | 41% | 8% | 1% |
| **All** | **236** | **28%** | **14%** | **18%** | **4%** | **6%** | **2%** | **7%** | **2%** | **10%** |

</div>

*Table 1. Mean steps per session and the share of steps by activity.*


### Time, Tools, and Context

<div class="njs-table" markdown="1">

| Model | Active (min) | Waiting on LLM | Running tools | Shell steps | Enumeration share | Sessions with compaction | Sessions naming the kernel |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fable 5.2 | 232 | 50% | 50% | 138 | 36% | 4 | 28 |
| Fable 5.1 | 128 | 71% | 28% | 64 | 43% | · | 29 |
| Opus 5.5 | 214 | 61% | 38% | 208 | 37% | 13 | 30 |
| Sonnet 5.5 | 345 | 40% | 61% | 272 | 38% | 21 | 30 |
| Haiku 5.5 | 32 | 55% | 44% | 35 | 43% | · | 28 |
| Haiku 4.5 | 3 | 71% | 26% | 6 | 3% | · | 22 |
| Gemini 4 Neon | 75 | 83% | 29% | 30 | 3% | 16 | 30 |
| Gemini 4 Carbon | 63 | 73% | 30% | 24 | 3% | 1 | 28 |
| Gemini 4 Argon | 46 | 81% | 20% | 30 | 4% | 11 | 29 |
| GPT-6.2 Astra | 23 | 109% | 9% | 98 | 3% | · | 22 |
| GPT-6 Astra | 13 | 97% | 14% | 44 | 1% | · | 22 |
| GPT-6.2 Sol | 13 | 97% | 13% | 63 | 4% | · | 22 |
| GPT-6.1 Sol | 23 | 104% | 9% | 57 | 0% | · | 25 |
| GPT-6 Sol | 11 | 93% | 11% | 30 | 2% | · | 24 |
| GPT-5.6 Terra | 17 | 110% | 29% | 88 | 2% | · | 24 |
| GPT-6 Luna | 6 | 96% | 1% | 8 | 0% | · | 23 |
| Kimi K3.1 Pro | 23 | 99% | 3% | 21 | 7% | · | 10 |
| Kimi K3 | 99 | 88% | 9% | 56 | 23% | · | 26 |
| GLM-5.2 | 75 | 86% | 3% | 30 | 21% | · | 27 |
| GLM-5.3-Flash | 176 | 79% | 15% | 278 | 8% | · | 21 |
| DeepSeek-V4-Flash | 27 | 79% | 20% | 61 | 8% | · | 23 |
| DeepSeek-V4-Flash-R | 33 | 80% | 19% | 69 | 14% | · | 23 |
| Qwen3.8-27B | 64 | 93% | 6% | 41 | 24% | · | 19 |
| Qwen3.6-35B-A3B | 18 | 26% | 0% | 11 | 1% | · | 13 |
| MiMo-V2.6-Pro | 49 | 93% | 7% | 40 | 16% | · | 20 |
| Seed-OSS-36B | 7 | 99% | 0% | 10 | 0% | · | 12 |
| Nemotron 3 Ultra | 2 | 96% | 1% | 16 | 0% | · | 22 |
| Nemotron 3.5 Super | 2 | 92% | 4% | 7 | 4% | · | 21 |
| **All** | **65** | **68%** | **32%** | **65** | **20%** | **66** | **653** |

</div>

*Table 2. Mean active time, its split between waiting on the model and running tools (the two can overlap), mean shell steps, the share of shell steps that enumerate parameters, and the number of sessions (of 30) with at least one context compaction and that name the PolyBench kernel.*


## Analyzer Labels by Goal

Counts of sessions (of 30 per model) with each label, from the Level-2 Analyzer's reports; the goals are defined on the [Evaluation Framework](../framework.md#analyzer-goals) page. A dot means zero.


### G2: Correctness Judgement

<div class="njs-table" markdown="1">

| Model | yes | partly | no | unknown |
| :--- | ---: | ---: | ---: | ---: |
| Fable 5.2 | 29 | 1 | · | · |
| Fable 5.1 | 28 | · | 2 | · |
| Opus 5.5 | 30 | · | · | · |
| Sonnet 5.5 | 30 | · | · | · |
| Haiku 5.5 | 29 | 1 | · | · |
| Haiku 4.5 | 28 | · | 2 | · |
| Gemini 4 Neon | 29 | · | 1 | · |
| Gemini 4 Carbon | 29 | 1 | · | · |
| Gemini 4 Argon | 30 | · | · | · |
| GPT-6.2 Astra | 29 | 1 | · | · |
| GPT-6 Astra | 30 | · | · | · |
| GPT-6.2 Sol | 30 | · | · | · |
| GPT-6.1 Sol | 30 | · | · | · |
| GPT-6 Sol | 29 | 1 | · | · |
| GPT-5.6 Terra | 30 | · | · | · |
| GPT-6 Luna | 28 | 1 | 1 | · |
| Kimi K3.1 Pro | 29 | · | · | · |
| Kimi K3 | 25 | 5 | · | · |
| GLM-5.2 | 29 | 1 | · | · |
| GLM-5.3-Flash | 23 | 6 | · | 1 |
| DeepSeek-V4-Flash | 26 | 4 | · | · |
| DeepSeek-V4-Flash-R | 29 | 1 | · | · |
| Qwen3.8-27B | 26 | 3 | · | 1 |
| Qwen3.6-35B-A3B | 18 | 5 | 5 | 2 |
| MiMo-V2.6-Pro | 25 | 4 | 1 | · |
| Seed-OSS-36B | 17 | 3 | 6 | 4 |
| Nemotron 3 Ultra | 28 | 2 | · | · |
| Nemotron 3.5 Super | 28 | 2 | · | · |
| **All** | **771** | **42** | **18** | **8** |

</div>

*Table 3. Whether the agent's own correctness judgement was right.*


### G3: Performance Model

<div class="njs-table" markdown="1">

| Model | quantitative | qualitative | none |
| :--- | ---: | ---: | ---: |
| Fable 5.2 | 30 | · | · |
| Fable 5.1 | 29 | 1 | · |
| Opus 5.5 | 30 | · | · |
| Sonnet 5.5 | 30 | · | · |
| Haiku 5.5 | 16 | 13 | 1 |
| Haiku 4.5 | · | 20 | 10 |
| Gemini 4 Neon | 29 | 1 | · |
| Gemini 4 Carbon | 24 | 5 | 1 |
| Gemini 4 Argon | 8 | 20 | 2 |
| GPT-6.2 Astra | · | 23 | 7 |
| GPT-6 Astra | · | 25 | 5 |
| GPT-6.2 Sol | · | 28 | 2 |
| GPT-6.1 Sol | · | 27 | 3 |
| GPT-6 Sol | · | 23 | 7 |
| GPT-5.6 Terra | · | 26 | 4 |
| GPT-6 Luna | · | 26 | 4 |
| Kimi K3.1 Pro | · | 24 | 6 |
| Kimi K3 | 23 | 6 | 1 |
| GLM-5.2 | 9 | 20 | 1 |
| GLM-5.3-Flash | 12 | 14 | 4 |
| DeepSeek-V4-Flash | 12 | 17 | 1 |
| DeepSeek-V4-Flash-R | 17 | 13 | · |
| Qwen3.8-27B | 8 | 20 | 2 |
| Qwen3.6-35B-A3B | 1 | 17 | 12 |
| MiMo-V2.6-Pro | 2 | 25 | 3 |
| Seed-OSS-36B | · | 3 | 27 |
| Nemotron 3 Ultra | · | 21 | 9 |
| Nemotron 3.5 Super | · | 29 | 1 |
| **All** | **280** | **447** | **113** |

</div>

*Table 4. Kind of performance model: quantitative only when the agent compares its numbers with an upper bound.*

<div class="njs-table" markdown="1">

| Model | compute | hbm-bandwidth | cache-bandwidth | on-chip | latency-occupancy | launch-overhead | other | none |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fable 5.2 | 17 | 7 | 1 | 1 | 3 | 1 | · | · |
| Fable 5.1 | 16 | 7 | · | 1 | 6 | · | · | · |
| Opus 5.5 | 16 | 8 | 2 | 1 | 3 | · | · | · |
| Sonnet 5.5 | 10 | 9 | 1 | 4 | 6 | · | · | · |
| Haiku 5.5 | 14 | 3 | · | 5 | 6 | 1 | · | 1 |
| Haiku 4.5 | 5 | 6 | · | · | 6 | 1 | 2 | 10 |
| Gemini 4 Neon | 17 | 11 | 1 | · | 1 | · | · | · |
| Gemini 4 Carbon | 20 | 7 | 1 | · | 1 | · | · | 1 |
| Gemini 4 Argon | 13 | 10 | · | 1 | 4 | · | · | 2 |
| GPT-6.2 Astra | 13 | 4 | · | · | 1 | · | 5 | 7 |
| GPT-6 Astra | 16 | 4 | · | · | 4 | · | 1 | 5 |
| GPT-6.2 Sol | 19 | 6 | · | 1 | 1 | · | · | 3 |
| GPT-6.1 Sol | 15 | 7 | · | · | 4 | · | 1 | 3 |
| GPT-6 Sol | 11 | 6 | · | · | 5 | 1 | · | 7 |
| GPT-5.6 Terra | 11 | 10 | · | · | 2 | 2 | 1 | 4 |
| GPT-6 Luna | 11 | 9 | · | · | 5 | 1 | · | 4 |
| Kimi K3.1 Pro | 8 | 9 | · | 1 | 3 | · | 3 | 6 |
| Kimi K3 | 16 | 7 | · | · | 6 | · | · | 1 |
| GLM-5.2 | 11 | 10 | 1 | 1 | 4 | 1 | 1 | 1 |
| GLM-5.3-Flash | 10 | 11 | 1 | · | 4 | · | · | 4 |
| DeepSeek-V4-Flash | 14 | 11 | · | · | 2 | 1 | 1 | 1 |
| DeepSeek-V4-Flash-R | 13 | 10 | 1 | 1 | 4 | · | 1 | · |
| Qwen3.8-27B | 10 | 10 | · | 1 | 5 | 2 | · | 2 |
| Qwen3.6-35B-A3B | 3 | 10 | · | · | 4 | · | 1 | 12 |
| MiMo-V2.6-Pro | 13 | 9 | · | · | 3 | 1 | 1 | 3 |
| Seed-OSS-36B | · | 2 | · | · | · | · | 1 | 27 |
| Nemotron 3 Ultra | 3 | 11 | · | · | 6 | · | 1 | 9 |
| Nemotron 3.5 Super | 5 | 18 | · | · | 5 | 1 | · | 1 |
| **All** | **330** | **232** | **9** | **18** | **104** | **13** | **20** | **114** |

</div>

*Table 5. The final bottleneck the agent names for its solution at the large sizes.*

<div class="njs-table" markdown="1">

| Model | runtime | static | both | none |
| :--- | ---: | ---: | ---: | ---: |
| Fable 5.2 | 3 | · | 27 | · |
| Fable 5.1 | 5 | · | 24 | 1 |
| Opus 5.5 | 1 | · | 29 | · |
| Sonnet 5.5 | · | · | 30 | · |
| Haiku 5.5 | 2 | · | 14 | 14 |
| Haiku 4.5 | · | · | · | 30 |
| Gemini 4 Neon | 3 | · | 26 | 1 |
| Gemini 4 Carbon | 3 | · | 21 | 6 |
| Gemini 4 Argon | 3 | · | 5 | 22 |
| GPT-6.2 Astra | · | · | · | 30 |
| GPT-6 Astra | · | · | · | 30 |
| GPT-6.2 Sol | · | · | · | 30 |
| GPT-6.1 Sol | · | · | · | 30 |
| GPT-6 Sol | · | · | · | 30 |
| GPT-5.6 Terra | · | · | · | 30 |
| GPT-6 Luna | · | · | · | 30 |
| Kimi K3.1 Pro | · | · | · | 30 |
| Kimi K3 | 2 | · | 21 | 7 |
| GLM-5.2 | 3 | 1 | 5 | 21 |
| GLM-5.3-Flash | 1 | 2 | 9 | 18 |
| DeepSeek-V4-Flash | 1 | 1 | 10 | 18 |
| DeepSeek-V4-Flash-R | 5 | 1 | 11 | 13 |
| Qwen3.8-27B | 6 | 1 | 1 | 22 |
| Qwen3.6-35B-A3B | · | 1 | · | 29 |
| MiMo-V2.6-Pro | · | · | 2 | 28 |
| Seed-OSS-36B | · | · | · | 30 |
| Nemotron 3 Ultra | · | · | · | 30 |
| Nemotron 3.5 Super | · | · | · | 30 |
| **All** | **38** | **7** | **235** | **560** |

</div>

*Table 6. Where a quantitative model's numbers come from: runtime measurements, static analysis, or both.*

<div class="njs-table" markdown="1">

| Model | Helped | Final check agrees | Final check disagrees |
| :--- | ---: | ---: | ---: |
| Fable 5.2 | 30 | 19 | 8 |
| Fable 5.1 | 30 | 21 | 8 |
| Opus 5.5 | 30 | 19 | 11 |
| Sonnet 5.5 | 30 | 16 | 14 |
| Haiku 5.5 | 23 | 13 | 14 |
| Haiku 4.5 | · | 2 | 20 |
| Gemini 4 Neon | 30 | 15 | 10 |
| Gemini 4 Carbon | 26 | 16 | 11 |
| Gemini 4 Argon | 24 | 16 | 11 |
| GPT-6.2 Astra | 19 | 7 | 14 |
| GPT-6 Astra | 13 | 7 | 19 |
| GPT-6.2 Sol | 11 | 6 | 22 |
| GPT-6.1 Sol | 17 | 7 | 18 |
| GPT-6 Sol | 3 | 7 | 18 |
| GPT-5.6 Terra | 7 | 14 | 13 |
| GPT-6 Luna | · | 10 | 13 |
| Kimi K3.1 Pro | 11 | 11 | 14 |
| Kimi K3 | 19 | 14 | 15 |
| GLM-5.2 | 10 | 15 | 12 |
| GLM-5.3-Flash | 4 | 12 | 14 |
| DeepSeek-V4-Flash | 13 | 16 | 13 |
| DeepSeek-V4-Flash-R | 17 | 15 | 15 |
| Qwen3.8-27B | 11 | 9 | 20 |
| Qwen3.6-35B-A3B | · | 1 | 19 |
| MiMo-V2.6-Pro | 8 | 7 | 19 |
| Seed-OSS-36B | · | · | 19 |
| Nemotron 3 Ultra | · | 2 | 23 |
| Nemotron 3.5 Super | · | 2 | 27 |
| **All** | **386** | **299** | **434** |

</div>

*Table 7. Sessions where the bottleneck analysis drove a kept, measured gain, and where the final result reaches 50% of the named resource's peak (or 30% of the agent's latency model).*


### G3: Stopping

<div class="njs-table" markdown="1">

| Model | Said it was done | Continued without gain | Room left (Analyzer) | justified | reason-wrong | no-reason | continued-without-gain |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fable 5.2 | 5 | 22 | 27 | · | 1 | 7 | 22 |
| Fable 5.1 | 7 | 14 | 29 | 1 | 3 | 12 | 14 |
| Opus 5.5 | 9 | 18 | 24 | 5 | · | 7 | 18 |
| Sonnet 5.5 | 8 | 27 | 29 | · | 1 | 2 | 27 |
| Haiku 5.5 | 3 | 12 | 30 | · | 2 | 16 | 12 |
| Haiku 4.5 | · | · | 30 | · | · | 30 | · |
| Gemini 4 Neon | 4 | 7 | 29 | · | 2 | 21 | 7 |
| Gemini 4 Carbon | 2 | 10 | 30 | · | 1 | 19 | 10 |
| Gemini 4 Argon | 2 | 16 | 30 | · | · | 14 | 16 |
| GPT-6.2 Astra | · | 6 | 29 | · | · | 24 | 6 |
| GPT-6 Astra | · | 8 | 30 | · | · | 22 | 8 |
| GPT-6.2 Sol | · | 8 | 30 | · | · | 22 | 8 |
| GPT-6.1 Sol | · | 6 | 30 | · | · | 24 | 6 |
| GPT-6 Sol | · | 5 | 30 | · | · | 25 | 5 |
| GPT-5.6 Terra | 1 | 8 | 30 | · | 1 | 21 | 8 |
| GPT-6 Luna | · | · | 30 | · | · | 30 | · |
| Kimi K3.1 Pro | 1 | 11 | 30 | · | 1 | 18 | 11 |
| Kimi K3 | 10 | 12 | 30 | · | 8 | 10 | 12 |
| GLM-5.2 | 11 | 6 | 30 | · | 9 | 15 | 6 |
| GLM-5.3-Flash | 7 | 17 | 30 | · | 2 | 11 | 17 |
| DeepSeek-V4-Flash | 13 | 6 | 30 | · | 11 | 13 | 6 |
| DeepSeek-V4-Flash-R | 14 | 8 | 30 | · | 8 | 14 | 8 |
| Qwen3.8-27B | 13 | 5 | 30 | · | 11 | 14 | 5 |
| Qwen3.6-35B-A3B | 1 | 1 | 30 | · | 1 | 28 | 1 |
| MiMo-V2.6-Pro | 10 | 13 | 30 | · | 4 | 13 | 13 |
| Seed-OSS-36B | 1 | 2 | 30 | · | · | 28 | 2 |
| Nemotron 3 Ultra | 1 | · | 29 | · | 1 | 29 | · |
| Nemotron 3.5 Super | 2 | · | 30 | · | 2 | 28 | · |
| **All** | **125** | **248** | **826** | **6** | **69** | **517** | **248** |

</div>

*Table 8. The agent's own stopping judgement, the tail without gain, the Analyzer's judgement of room left, and the stopping label: justified only when the agent says it is done with a reason that holds and did not keep trying without gain.*


### G4: Strategy

<div class="njs-table" markdown="1">

| Model | strategy | mixed | search | none |
| :--- | ---: | ---: | ---: | ---: |
| Fable 5.2 | 29 | 1 | · | · |
| Fable 5.1 | 29 | 1 | · | · |
| Opus 5.5 | 30 | · | · | · |
| Sonnet 5.5 | 28 | 2 | · | · |
| Haiku 5.5 | 29 | 1 | · | · |
| Haiku 4.5 | 1 | · | · | 29 |
| Gemini 4 Neon | 26 | 4 | · | · |
| Gemini 4 Carbon | 27 | 3 | · | · |
| Gemini 4 Argon | 21 | 9 | · | · |
| GPT-6.2 Astra | 14 | 16 | · | · |
| GPT-6 Astra | 28 | 2 | · | · |
| GPT-6.2 Sol | 27 | 3 | · | · |
| GPT-6.1 Sol | 27 | 3 | · | · |
| GPT-6 Sol | 23 | 1 | · | 6 |
| GPT-5.6 Terra | 22 | 2 | · | 6 |
| GPT-6 Luna | 9 | · | · | 21 |
| Kimi K3.1 Pro | 22 | 3 | · | 5 |
| Kimi K3 | 24 | 1 | · | 5 |
| GLM-5.2 | 25 | 3 | · | 2 |
| GLM-5.3-Flash | 2 | 16 | 1 | 11 |
| DeepSeek-V4-Flash | 18 | 5 | · | 7 |
| DeepSeek-V4-Flash-R | 18 | 9 | · | 3 |
| Qwen3.8-27B | 21 | 2 | · | 7 |
| Qwen3.6-35B-A3B | · | 3 | · | 27 |
| MiMo-V2.6-Pro | 16 | 8 | · | 5 |
| Seed-OSS-36B | · | 1 | 2 | 27 |
| Nemotron 3 Ultra | · | · | 1 | 29 |
| Nemotron 3.5 Super | 1 | · | · | 28 |
| **All** | **517** | **99** | **4** | **218** |

</div>

*Table 9. Strategy (hypothesis, change, measurement, decision), undirected search, a mix, or no optimization attempted.*


### G1: Optimization Techniques Kept

<div class="njs-table" markdown="1">

| Model | Techniques tried | library call | shared-memory tiling | register reuse or streaming | wide or coalesced loads | buffers and storage layout | fusion of passes or steps | thread and block mapping | synchronisation and contention | reduction scheme | algebraic or algorithmic rewrite | precision and numerics | hardware-specific instruction or feature | launch overhead | size-dependent parameter or path | compiler-level code tuning |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fable 5.2 | 26 | 12 | 15 | 22 | 20 | 21 | 21 | 25 | 12 | 8 | 17 | 28 | 10 | 26 | 30 | 24 |
| Fable 5.1 | 24 | 10 | 11 | 17 | 21 | 23 | 14 | 24 | 11 | 14 | 15 | 26 | 17 | 26 | 26 | 19 |
| Opus 5.5 | 35 | 9 | 13 | 18 | 19 | 24 | 16 | 26 | 27 | 10 | 14 | 29 | 26 | 28 | 30 | 26 |
| Sonnet 5.5 | 34 | 5 | 15 | 22 | 24 | 24 | 19 | 28 | 17 | 12 | 14 | 29 | 20 | 26 | 30 | 27 |
| Haiku 5.5 | 15 | 12 | 11 | 12 | 15 | 20 | 9 | 19 | 3 | 10 | 10 | 27 | 3 | 16 | 16 | 12 |
| Haiku 4.5 | 4 | 9 | · | · | · | 5 | 1 | 9 | · | 3 | 2 | 1 | · | 1 | · | · |
| Gemini 4 Neon | 24 | 15 | 12 | 13 | 19 | 22 | 24 | 19 | 11 | 12 | 26 | 26 | 17 | 28 | 30 | 19 |
| Gemini 4 Carbon | 19 | 12 | 9 | 15 | 19 | 20 | 18 | 18 | 8 | 11 | 14 | 26 | 13 | 29 | 30 | 16 |
| Gemini 4 Argon | 18 | 14 | 13 | 18 | 15 | 17 | 20 | 18 | 13 | 8 | 11 | 18 | 4 | 29 | 30 | 16 |
| GPT-6.2 Astra | 18 | 9 | 10 | 8 | 9 | 22 | 11 | 14 | 3 | 14 | 18 | 26 | 15 | 28 | 28 | 7 |
| GPT-6 Astra | 12 | 12 | 11 | 5 | · | 15 | 10 | 16 | 3 | 8 | 14 | 25 | 2 | 29 | 29 | 8 |
| GPT-6.2 Sol | 14 | 11 | 12 | 11 | 11 | 15 | 9 | 13 | 4 | 11 | 13 | 28 | 5 | 28 | 30 | 10 |
| GPT-6.1 Sol | 14 | 11 | 11 | 5 | 7 | 15 | 16 | 15 | 2 | 10 | 13 | 26 | 4 | 28 | 29 | 9 |
| GPT-6 Sol | 9 | 15 | 3 | · | 1 | 12 | 7 | 16 | 2 | 10 | 5 | 24 | · | 22 | 27 | 3 |
| GPT-5.6 Terra | 10 | 19 | 6 | 1 | · | 23 | 12 | 10 | 2 | 4 | 8 | 22 | 1 | 19 | 10 | 4 |
| GPT-6 Luna | 6 | 17 | 4 | 1 | 1 | 20 | 12 | 11 | 1 | 3 | 7 | 18 | · | 10 | 6 | 2 |
| Kimi K3.1 Pro | 10 | 17 | 6 | · | 3 | 20 | 11 | 13 | 5 | 6 | 6 | 11 | · | 29 | 18 | 8 |
| Kimi K3 | 16 | 14 | 9 | 10 | 11 | 18 | 13 | 15 | 5 | 7 | 13 | 15 | 5 | 25 | 18 | 10 |
| GLM-5.2 | 10 | 19 | 6 | 1 | 2 | 14 | 4 | 8 | 3 | 4 | 5 | 17 | · | 24 | 10 | 5 |
| GLM-5.3-Flash | 11 | 15 | 3 | 3 | 1 | 20 | 10 | 15 | 6 | 5 | 7 | 16 | 1 | 15 | 9 | 3 |
| DeepSeek-V4-Flash | 11 | 18 | 4 | 2 | 3 | 20 | 8 | 15 | 2 | 7 | 11 | 18 | · | 15 | 10 | 4 |
| DeepSeek-V4-Flash-R | 12 | 15 | 5 | 2 | 5 | 15 | 7 | 12 | 3 | 7 | 10 | 24 | 1 | 19 | 11 | 3 |
| Qwen3.8-27B | 11 | 11 | 3 | · | 5 | 6 | 11 | 12 | 4 | 7 | 6 | 21 | 1 | 17 | 7 | 7 |
| Qwen3.6-35B-A3B | 5 | 2 | · | 1 | · | 8 | · | 9 | 4 | 2 | · | 2 | · | 1 | · | 5 |
| MiMo-V2.6-Pro | 11 | 15 | 7 | 2 | 5 | 20 | 7 | 13 | 1 | 6 | 13 | 21 | · | 18 | 8 | 3 |
| Seed-OSS-36B | 5 | · | · | · | 1 | 2 | 2 | 16 | 4 | · | · | 4 | · | · | · | 2 |
| Nemotron 3 Ultra | 4 | 3 | 1 | · | · | 4 | 3 | 7 | 2 | 1 | · | 4 | · | · | 1 | 1 |
| Nemotron 3.5 Super | 4 | 2 | 2 | · | · | 7 | 5 | 9 | 1 | 6 | 1 | 13 | · | 1 | · | 5 |
| **All** | **14** | **323** | **202** | **189** | **217** | **452** | **300** | **425** | **159** | **206** | **273** | **545** | **145** | **537** | **473** | **258** |

</div>

*Table 10. Mean techniques tried per session, and the number of sessions whose final solution keeps at least one technique of each family.*


### G5: Errors and Dead Ends

<div class="njs-table" markdown="1">

| Model | Errors | Dead ends | compile or link | tool or environment | wrong result | latent defect or edge case | crash or runtime error | timeout or hang | kernel launch or resource limit | slower or no gain | measurement problem | wrong belief or prediction |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fable 5.2 | 18 | 1.7 | 61 | 120 | 20 | 33 | 15 | 23 | 21 | 139 | 70 | 33 |
| Fable 5.1 | 18 | 1.8 | 82 | 81 | 34 | 30 | 16 | 17 | 27 | 153 | 52 | 41 |
| Opus 5.5 | 27 | 3.3 | 126 | 102 | 55 | 54 | 23 | 28 | 29 | 223 | 78 | 83 |
| Sonnet 5.5 | 30 | 3.1 | 127 | 160 | 40 | 70 | 23 | 29 | 24 | 237 | 103 | 83 |
| Haiku 5.5 | 9 | 0.7 | 37 | 33 | 5 | 18 | 3 | 5 | 3 | 72 | 34 | 47 |
| Haiku 4.5 | 3 | 0.1 | 5 | 3 | 23 | 22 | 2 | 1 | · | 7 | 7 | 16 |
| Gemini 4 Neon | 15 | 1.1 | 87 | 43 | 40 | 14 | 10 | 15 | 36 | 129 | 44 | 25 |
| Gemini 4 Carbon | 9 | 0.8 | 43 | 29 | 14 | 9 | 4 | 5 | 23 | 113 | 21 | 15 |
| Gemini 4 Argon | 9 | 0.9 | 23 | 21 | 18 | 9 | · | 6 | 24 | 146 | 10 | 8 |
| GPT-6.2 Astra | 9 | 1.4 | 31 | 58 | 7 | 8 | 1 | · | 6 | 138 | 7 | 8 |
| GPT-6 Astra | 6 | 0.4 | 28 | 65 | 8 | 8 | · | · | 2 | 49 | 8 | 6 |
| GPT-6.2 Sol | 6 | 0.8 | 6 | 80 | 3 | 12 | 1 | · | 2 | 58 | 13 | 6 |
| GPT-6.1 Sol | 7 | 0.7 | 32 | 57 | 5 | 22 | · | 1 | 3 | 65 | 14 | 11 |
| GPT-6 Sol | 5 | 0.3 | 7 | 75 | 2 | 9 | · | · | · | 37 | 13 | 7 |
| GPT-5.6 Terra | 8 | 0.5 | 22 | 105 | 2 | 16 | 1 | 5 | 1 | 37 | 40 | 14 |
| GPT-6 Luna | 2 | 0.1 | 10 | 37 | 4 | 9 | · | · | · | · | 4 | 2 |
| Kimi K3.1 Pro | 5 | 0.5 | 9 | 30 | 7 | 16 | 1 | 2 | · | 56 | 9 | 16 |
| Kimi K3 | 14 | 1.4 | 82 | 66 | 28 | 24 | 11 | 19 | 10 | 80 | 59 | 41 |
| GLM-5.2 | 8 | 0.7 | 17 | 59 | 5 | 11 | 4 | 23 | 4 | 34 | 29 | 34 |
| GLM-5.3-Flash | 17 | 2.5 | 113 | 104 | 46 | 21 | 19 | 28 | 9 | 24 | 61 | 57 |
| DeepSeek-V4-Flash | 15 | 1.4 | 77 | 68 | 57 | 20 | 19 | 35 | 4 | 47 | 70 | 54 |
| DeepSeek-V4-Flash-R | 17 | 1.5 | 100 | 78 | 38 | 17 | 25 | 32 | 15 | 49 | 78 | 63 |
| Qwen3.8-27B | 14 | 1.2 | 114 | 75 | 23 | 24 | 10 | 24 | 2 | 28 | 61 | 52 |
| Qwen3.6-35B-A3B | 5 | 0.5 | 38 | 42 | 18 | 22 | 1 | 2 | 1 | 4 | 6 | 20 |
| MiMo-V2.6-Pro | 12 | 1.2 | 64 | 77 | 18 | 23 | 4 | 19 | 9 | 40 | 47 | 43 |
| Seed-OSS-36B | 5 | 0.7 | 34 | 23 | 40 | 31 | 4 | · | · | 8 | 3 | 16 |
| Nemotron 3 Ultra | 4 | 0.2 | 37 | 6 | 21 | 10 | 3 | 2 | · | 4 | 3 | 22 |
| Nemotron 3.5 Super | 3 | 0.2 | 5 | 27 | 4 | 10 | 1 | 2 | · | 3 | 6 | 27 |
| **All** | **11** | **1.1** | **1417** | **1724** | **585** | **572** | **201** | **323** | **255** | **1980** | **950** | **850** |

</div>

*Table 11. Mean errors and dead ends per session, and the total number of errors of each kind.*

<div class="njs-table" markdown="1">

| Model | diagnosed, then fixed | changed without a stated diagnosis | reverted or dropped | worked around or accepted | ignored or not noticed | nothing needed |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fable 5.2 | 292 | 23 | 85 | 102 | 16 | 13 |
| Fable 5.1 | 333 | 10 | 99 | 58 | 25 | 7 |
| Opus 5.5 | 481 | 28 | 182 | 91 | 5 | 15 |
| Sonnet 5.5 | 546 | 10 | 168 | 144 | 12 | 22 |
| Haiku 5.5 | 152 | 4 | 42 | 50 | 27 | 5 |
| Haiku 4.5 | 29 | 5 | 3 | 2 | 45 | · |
| Gemini 4 Neon | 251 | 34 | 80 | 58 | 17 | 5 |
| Gemini 4 Carbon | 114 | 29 | 54 | 30 | 34 | 13 |
| Gemini 4 Argon | 98 | 37 | 75 | 36 | 20 | 5 |
| GPT-6.2 Astra | 64 | 20 | 101 | 53 | 10 | 18 |
| GPT-6 Astra | 90 | 7 | 27 | 32 | 8 | 12 |
| GPT-6.2 Sol | 56 | 21 | 25 | 50 | 22 | 12 |
| GPT-6.1 Sol | 102 | 8 | 35 | 39 | 13 | 13 |
| GPT-6 Sol | 56 | 15 | 9 | 29 | 13 | 30 |
| GPT-5.6 Terra | 87 | 25 | 26 | 74 | 28 | 6 |
| GPT-6 Luna | 31 | 4 | · | 15 | 10 | 10 |
| Kimi K3.1 Pro | 74 | 18 | 33 | 20 | 8 | 2 |
| Kimi K3 | 243 | 19 | 56 | 53 | 44 | 4 |
| GLM-5.2 | 85 | 21 | 32 | 38 | 47 | 2 |
| GLM-5.3-Flash | 273 | 36 | 30 | 68 | 75 | 8 |
| DeepSeek-V4-Flash | 255 | 28 | 51 | 51 | 65 | 4 |
| DeepSeek-V4-Flash-R | 266 | 39 | 54 | 64 | 69 | 5 |
| Qwen3.8-27B | 227 | 29 | 42 | 46 | 66 | 5 |
| Qwen3.6-35B-A3B | 45 | 16 | 9 | 16 | 47 | 6 |
| MiMo-V2.6-Pro | 167 | 17 | 47 | 40 | 69 | 12 |
| Seed-OSS-36B | 57 | 10 | 16 | 8 | 43 | 4 |
| Nemotron 3 Ultra | 39 | 12 | 10 | 8 | 41 | · |
| Nemotron 3.5 Super | 15 | 11 | 3 | 6 | 47 | 1 |
| **All** | **4528** | **536** | **1394** | **1281** | **926** | **239** |

</div>

*Table 12. The agent's response to each error, totals.*


### G6: Strengths and Weaknesses

<div class="njs-table" markdown="1">

| Model | performance model or bound | measurement at the scored scale | measurement discipline | correctness testing | library use | kernel design | acceptance and precision | process and time | device knowledge | no optimisation attempted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fable 5.2 | 29 | 16 | 22 | 30 | 12 | 28 | 18 | · | 30 | · |
| Fable 5.1 | 24 | 13 | 20 | 29 | 17 | 31 | 19 | · | 22 | · |
| Opus 5.5 | 32 | 19 | 27 | 30 | 6 | 35 | 28 | 1 | 27 | · |
| Sonnet 5.5 | 30 | 17 | 25 | 29 | 1 | 34 | 26 | · | 28 | · |
| Haiku 5.5 | 9 | 15 | 20 | 30 | 12 | 26 | 23 | · | 13 | · |
| Haiku 4.5 | · | · | · | 22 | 10 | 24 | 11 | · | · | · |
| Gemini 4 Neon | 25 | 25 | 10 | 28 | 6 | 36 | 23 | 1 | 27 | · |
| Gemini 4 Carbon | 13 | 16 | 5 | 30 | 18 | 31 | 21 | · | 24 | · |
| Gemini 4 Argon | 6 | 13 | 5 | 30 | 17 | 33 | 18 | · | 20 | · |
| GPT-6.2 Astra | · | 2 | 6 | 31 | 10 | 30 | 20 | 8 | 17 | · |
| GPT-6 Astra | · | 3 | 11 | 31 | 16 | 25 | 26 | 1 | 10 | · |
| GPT-6.2 Sol | · | 3 | 13 | 30 | 14 | 25 | 21 | 2 | 9 | · |
| GPT-6.1 Sol | · | 4 | 13 | 30 | 16 | 27 | 23 | 2 | 11 | · |
| GPT-6 Sol | · | · | 3 | 32 | 17 | 19 | 25 | · | · | · |
| GPT-5.6 Terra | · | 2 | 8 | 31 | 26 | 23 | 24 | · | 4 | · |
| GPT-6 Luna | · | · | · | · | 21 | 31 | 24 | · | 2 | · |
| Kimi K3.1 Pro | · | 3 | 9 | 33 | 19 | 31 | 19 | · | 3 | · |
| Kimi K3 | 16 | 17 | 12 | 28 | 18 | 23 | 19 | · | 13 | · |
| GLM-5.2 | 2 | 7 | 10 | 31 | 25 | 17 | 22 | · | 1 | · |
| GLM-5.3-Flash | 5 | 8 | 4 | 30 | 17 | 17 | 20 | · | 1 | · |
| DeepSeek-V4-Flash | 7 | 10 | 10 | 33 | 17 | 20 | 17 | · | 9 | · |
| DeepSeek-V4-Flash-R | 7 | 14 | 12 | 30 | 20 | 20 | 20 | · | 5 | · |
| Qwen3.8-27B | 4 | 4 | 11 | 35 | 15 | 17 | 24 | · | 3 | · |
| Qwen3.6-35B-A3B | · | · | 1 | 13 | 3 | 25 | 14 | 1 | · | · |
| MiMo-V2.6-Pro | 1 | 10 | 10 | 36 | 17 | 20 | 18 | · | 6 | · |
| Seed-OSS-36B | · | · | · | 21 | · | 21 | 9 | · | · | · |
| Nemotron 3 Ultra | · | · | · | 31 | 4 | 20 | 9 | · | · | · |
| Nemotron 3.5 Super | · | · | · | 25 | 3 | 23 | 23 | · | · | · |
| **All** | **210** | **221** | **267** | **789** | **377** | **712** | **564** | **16** | **285** | **·** |

</div>

*Table 13. Strengths by theme, totals.*

<div class="njs-table" markdown="1">

| Model | performance model or bound | measurement at the scored scale | measurement discipline | correctness testing | library use | kernel design | acceptance and precision | process and time | device knowledge | no optimisation attempted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fable 5.2 | 22 | 18 | 8 | 1 | 1 | 17 | 15 | 34 | 1 | · |
| Fable 5.1 | 20 | 22 | 15 | 6 | · | 20 | 11 | 35 | · | · |
| Opus 5.5 | 30 | 25 | 16 | 5 | · | 15 | 10 | 38 | · | · |
| Sonnet 5.5 | 19 | 23 | 8 | 3 | 2 | 14 | 9 | 37 | · | · |
| Haiku 5.5 | 30 | 14 | 16 | · | 1 | 22 | 22 | 31 | 5 | · |
| Haiku 4.5 | 30 | 30 | 8 | 8 | 10 | 23 | 6 | 24 | 12 | 22 |
| Gemini 4 Neon | 23 | 12 | 19 | 8 | · | 22 | 11 | 29 | · | · |
| Gemini 4 Carbon | 31 | 26 | 22 | 7 | 1 | 13 | 14 | 30 | 1 | · |
| Gemini 4 Argon | 30 | 23 | 23 | 9 | 1 | 13 | 12 | 29 | 2 | · |
| GPT-6.2 Astra | 31 | 29 | 28 | · | 5 | 12 | 16 | 29 | 4 | · |
| GPT-6 Astra | 30 | 28 | 21 | · | 3 | 21 | 11 | 30 | 2 | · |
| GPT-6.2 Sol | 30 | 26 | 16 | 1 | 2 | 18 | 20 | 31 | 12 | · |
| GPT-6.1 Sol | 30 | 24 | 13 | 2 | 3 | 20 | 13 | 33 | 4 | · |
| GPT-6 Sol | 31 | 28 | 19 | · | 1 | 15 | 19 | 29 | 15 | 2 |
| GPT-5.6 Terra | 32 | 27 | 22 | · | 2 | 15 | 15 | 31 | 5 | 1 |
| GPT-6 Luna | 30 | 30 | 2 | 28 | 1 | 21 | 13 | 23 | 8 | 11 |
| Kimi K3.1 Pro | 30 | 26 | 13 | 3 | 1 | 15 | 12 | 29 | 13 | 4 |
| Kimi K3 | 30 | 14 | 16 | 8 | 1 | 18 | 14 | 34 | 4 | 1 |
| GLM-5.2 | 31 | 19 | 23 | 2 | 3 | 21 | 15 | 30 | 8 | 2 |
| GLM-5.3-Flash | 30 | 12 | 21 | 8 | 7 | 18 | 17 | 34 | 11 | 5 |
| DeepSeek-V4-Flash | 34 | 15 | 20 | 5 | 3 | 19 | 15 | 31 | 17 | 4 |
| DeepSeek-V4-Flash-R | 31 | 17 | 18 | 3 | 3 | 23 | 18 | 29 | 15 | 1 |
| Qwen3.8-27B | 30 | 22 | 21 | 2 | 7 | 22 | 13 | 29 | 12 | 5 |
| Qwen3.6-35B-A3B | 23 | 23 | 5 | 19 | 13 | 25 | 3 | 26 | 13 | 22 |
| MiMo-V2.6-Pro | 31 | 19 | 26 | 5 | 2 | 19 | 15 | 31 | 9 | 1 |
| Seed-OSS-36B | 24 | 26 | 1 | 17 | 12 | 27 | 5 | 27 | 13 | 22 |
| Nemotron 3 Ultra | 30 | 30 | 4 | · | 15 | 26 | 5 | 26 | 17 | 26 |
| Nemotron 3.5 Super | 30 | 30 | 4 | 7 | 13 | 25 | 14 | 18 | 10 | 25 |
| **All** | **803** | **638** | **428** | **157** | **113** | **539** | **363** | **837** | **213** | **154** |

</div>

*Table 14. Weaknesses by theme, totals.*


### G7: Verification Result

<div class="njs-table" markdown="1">

| Model | why-good | why-weak | both | why-failed | Failure catchable by the agent |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Fable 5.2 | 10 | 2 | 18 | · | 14 |
| Fable 5.1 | 6 | 2 | 20 | 2 | 18 |
| Opus 5.5 | 22 | 1 | 7 | · | 8 |
| Sonnet 5.5 | 16 | · | 13 | 1 | 14 |
| Haiku 5.5 | · | 17 | 13 | · | 30 |
| Haiku 4.5 | · | 18 | 3 | 9 | 30 |
| Gemini 4 Neon | 16 | 1 | 12 | 1 | 11 |
| Gemini 4 Carbon | 2 | 5 | 23 | · | 21 |
| Gemini 4 Argon | 5 | 12 | 13 | · | 25 |
| GPT-6.2 Astra | 1 | 15 | 14 | · | 22 |
| GPT-6 Astra | · | 16 | 13 | 1 | 27 |
| GPT-6.2 Sol | · | 23 | 6 | 1 | 28 |
| GPT-6.1 Sol | · | 22 | 8 | · | 26 |
| GPT-6 Sol | · | 27 | 2 | 1 | 29 |
| GPT-5.6 Terra | · | 22 | 8 | · | 29 |
| GPT-6 Luna | · | 22 | 5 | 3 | 30 |
| Kimi K3.1 Pro | 1 | 21 | 8 | · | 28 |
| Kimi K3 | · | 18 | 11 | 1 | 29 |
| GLM-5.2 | · | 23 | 7 | · | 30 |
| GLM-5.3-Flash | · | 20 | 5 | 5 | 29 |
| DeepSeek-V4-Flash | 1 | 22 | 6 | 1 | 29 |
| DeepSeek-V4-Flash-R | · | 22 | 7 | 1 | 30 |
| Qwen3.8-27B | · | 25 | 4 | 1 | 30 |
| Qwen3.6-35B-A3B | · | 17 | · | 13 | 30 |
| MiMo-V2.6-Pro | · | 23 | 3 | 4 | 29 |
| Seed-OSS-36B | · | 12 | · | 18 | 26 |
| Nemotron 3 Ultra | · | 18 | · | 12 | 27 |
| Nemotron 3.5 Super | · | 24 | 1 | 5 | 30 |
| **All** | **80** | **450** | **230** | **80** | **709** |

</div>

*Table 15. Focus of the explanation of the level-1 result (why the solution failed, is weak, is good, or both), and sessions where a check the agent ran could have caught the failure or weakness the Analyzer identifies.*


### G8 to G10: Legitimacy, Precision, and Generality

<div class="njs-table" markdown="1">

| Model | no-finding | suspect | confirmed-hack | same-precision | lower-accepted-within-tolerance | loss-caught | loss-not-caught | higher-precision | unknown |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fable 5.2 | 30 | · | · | 21 | 9 | · | · | · | · |
| Fable 5.1 | 29 | 1 | · | 21 | 9 | · | · | · | · |
| Opus 5.5 | 28 | 2 | · | 15 | 15 | · | · | · | · |
| Sonnet 5.5 | 30 | · | · | 18 | 12 | · | · | · | · |
| Haiku 5.5 | 30 | · | · | 27 | · | · | · | 3 | · |
| Haiku 4.5 | 30 | · | · | 30 | · | · | · | · | · |
| Gemini 4 Neon | 30 | · | · | 17 | 13 | · | · | · | · |
| Gemini 4 Carbon | 30 | · | · | 24 | 6 | · | · | · | · |
| Gemini 4 Argon | 30 | · | · | 28 | 2 | · | · | · | · |
| GPT-6.2 Astra | 30 | · | · | 24 | 5 | · | · | 1 | · |
| GPT-6 Astra | 30 | · | · | 28 | 2 | · | · | · | · |
| GPT-6.2 Sol | 30 | · | · | 25 | · | · | · | 5 | · |
| GPT-6.1 Sol | 30 | · | · | 24 | 3 | · | · | 3 | · |
| GPT-6 Sol | 30 | · | · | 27 | · | · | · | 3 | · |
| GPT-5.6 Terra | 30 | · | · | 30 | · | · | · | · | · |
| GPT-6 Luna | 30 | · | · | 27 | · | · | · | 3 | · |
| Kimi K3.1 Pro | 30 | · | · | 29 | · | · | · | 1 | · |
| Kimi K3 | 30 | · | · | 29 | 1 | · | · | · | · |
| GLM-5.2 | 30 | · | · | 30 | · | · | · | · | · |
| GLM-5.3-Flash | 30 | · | · | 24 | · | · | · | 5 | 1 |
| DeepSeek-V4-Flash | 30 | · | · | 28 | · | · | · | 2 | · |
| DeepSeek-V4-Flash-R | 30 | · | · | 23 | 1 | · | · | 6 | · |
| Qwen3.8-27B | 30 | · | · | 26 | · | · | · | 3 | 1 |
| Qwen3.6-35B-A3B | 30 | · | · | 28 | · | · | · | · | 2 |
| MiMo-V2.6-Pro | 30 | · | · | 28 | · | · | · | 1 | 1 |
| Seed-OSS-36B | 30 | · | · | 28 | · | · | · | · | 2 |
| Nemotron 3 Ultra | 30 | · | · | 30 | · | · | · | · | · |
| Nemotron 3.5 Super | 30 | · | · | 29 | · | · | · | 1 | · |
| **All** | **837** | **3** | **·** | **718** | **78** | **·** | **·** | **37** | **7** |

</div>

*Table 16. G8 reward-hacking verdict and G9 precision acceptance.*

<div class="njs-table" markdown="1">

| Model | Algorithmic changes | Not general | Bugs on allowed inputs | input values are bounded | only part of the results is required as output | another stated condition | code path our tests never ran |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fable 5.2 | 160 | 2 | 1 | 12 | 6 | 7 | 45 |
| Fable 5.1 | 157 | 3 | 5 | 4 | 3 | 8 | 45 |
| Opus 5.5 | 154 | 11 | 2 | 17 | 8 | 11 | 41 |
| Sonnet 5.5 | 186 | 10 | 5 | 15 | 4 | 13 | 47 |
| Haiku 5.5 | 127 | · | · | 1 | 2 | 8 | 20 |
| Haiku 4.5 | 46 | · | 19 | · | · | · | 2 |
| Gemini 4 Neon | 180 | 4 | 3 | 9 | 8 | 9 | 33 |
| Gemini 4 Carbon | 160 | · | 4 | 1 | 7 | 5 | 53 |
| Gemini 4 Argon | 146 | 2 | 12 | 3 | 7 | 5 | 34 |
| GPT-6.2 Astra | 149 | 3 | · | 8 | 2 | 12 | 42 |
| GPT-6 Astra | 131 | 3 | · | 6 | 3 | 4 | 37 |
| GPT-6.2 Sol | 142 | 3 | 1 | 10 | 1 | 10 | 39 |
| GPT-6.1 Sol | 138 | 1 | 3 | 3 | 2 | 7 | 37 |
| GPT-6 Sol | 101 | · | · | 1 | 3 | 4 | 24 |
| GPT-5.6 Terra | 110 | · | 2 | 1 | · | 5 | 21 |
| GPT-6 Luna | 76 | · | 3 | 1 | 3 | 7 | 14 |
| Kimi K3.1 Pro | 96 | · | 1 | 1 | 4 | 3 | 20 |
| Kimi K3 | 116 | · | 9 | 1 | 2 | 4 | 23 |
| GLM-5.2 | 81 | 1 | 4 | 1 | 2 | 5 | 11 |
| GLM-5.3-Flash | 86 | 3 | 15 | 3 | 1 | 5 | 15 |
| DeepSeek-V4-Flash | 92 | · | 13 | 1 | 3 | 6 | 16 |
| DeepSeek-V4-Flash-R | 94 | 1 | 8 | 1 | 3 | 4 | 14 |
| Qwen3.8-27B | 93 | · | 10 | 1 | 2 | 4 | 12 |
| Qwen3.6-35B-A3B | 56 | 9 | 38 | · | · | 2 | 4 |
| MiMo-V2.6-Pro | 91 | 2 | 11 | 1 | 3 | 3 | 16 |
| Seed-OSS-36B | 41 | 8 | 25 | 1 | · | · | · |
| Nemotron 3 Ultra | 44 | 1 | 18 | 1 | · | 1 | 1 |
| Nemotron 3.5 Super | 55 | · | 18 | 1 | 1 | 2 | 9 |
| **All** | **3108** | **67** | **230** | **105** | **80** | **154** | **675** |

</div>

*Table 17. G10 totals: algorithmic changes the Analyzer marks (any change of loop structure or algorithm, a broader notion than Analysis 2's verified algorithm changes), those judged not general, bugs claimed on allowed inputs, and task conditions relied on by kind.*


### G11: Enumeration

<div class="njs-table" markdown="1">

| Model | guided | mixed | brute-force | none | Necessary | Narrowable | Avoidable |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fable 5.2 | 30 | · | · | · | 921 | 97 | 22 |
| Fable 5.1 | 29 | 1 | · | · | 654 | 86 | 18 |
| Opus 5.5 | 30 | · | · | · | 876 | 93 | 48 |
| Sonnet 5.5 | 30 | · | · | · | 887 | 101 | 30 |
| Haiku 5.5 | 30 | · | · | · | 470 | 37 | 24 |
| Haiku 4.5 | 3 | 3 | 2 | 22 | 5 | 6 | 2 |
| Gemini 4 Neon | 27 | 3 | · | · | 328 | 93 | 5 |
| Gemini 4 Carbon | 30 | · | · | · | 272 | 31 | 2 |
| Gemini 4 Argon | 27 | 3 | · | · | 225 | 32 | 6 |
| GPT-6.2 Astra | 29 | 1 | · | · | 199 | 11 | 3 |
| GPT-6 Astra | 29 | · | · | 1 | 120 | 6 | 3 |
| GPT-6.2 Sol | 30 | · | · | · | 169 | 13 | 5 |
| GPT-6.1 Sol | 29 | 1 | · | · | 168 | 14 | 8 |
| GPT-6 Sol | 26 | · | · | 4 | 86 | 8 | 1 |
| GPT-5.6 Terra | 29 | 1 | · | · | 210 | 20 | 15 |
| GPT-6 Luna | · | · | · | 30 | · | · | · |
| Kimi K3.1 Pro | 30 | · | · | · | 191 | 16 | 7 |
| Kimi K3 | 30 | · | · | · | 361 | 43 | 19 |
| GLM-5.2 | 27 | 2 | · | 1 | 211 | 50 | 18 |
| GLM-5.3-Flash | 14 | 12 | 3 | 1 | 191 | 61 | 71 |
| DeepSeek-V4-Flash | 25 | 4 | · | 1 | 217 | 41 | 37 |
| DeepSeek-V4-Flash-R | 19 | 8 | 1 | 2 | 241 | 66 | 50 |
| Qwen3.8-27B | 24 | 4 | 1 | 1 | 235 | 59 | 40 |
| Qwen3.6-35B-A3B | 8 | 1 | · | 21 | 13 | 2 | · |
| MiMo-V2.6-Pro | 20 | 6 | · | 4 | 227 | 53 | 45 |
| Seed-OSS-36B | · | · | 1 | 29 | · | · | 1 |
| Nemotron 3 Ultra | · | · | 1 | 29 | · | · | 1 |
| Nemotron 3.5 Super | 28 | · | · | 2 | 51 | · | · |
| **All** | **633** | **50** | **9** | **148** | **7528** | **1039** | **481** |

</div>

*Table 18. G11 verdict per session, and the total enumerations judged necessary, narrowable by an analysis, or avoidable.*

