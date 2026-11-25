# Bug Analysis

In this section, we account for every failed session, whether the Verifier or the Bug Verifier caught it, and summarize how agents handle the bugs they meet during a session (G5). The bugs that pass the Verifier and are found only afterwards are analyzed in [Analysis 7](07-bugs-beyond-the-verifier.md).

## All Failures, by Where They Were Caught

Of the 840 sessions, 102 do not count as passed. The Verifier catches 74 of them, and the Bug Verifier catches the other 28 after they had passed the Verifier. Table 1 splits every failure by its stage and kind.

<div class="njs-table" markdown="1">

| Model | No solution | Build failure | Crash | Wrong output (Verifier) | Too slow | Wrong output (Bug Verifier) | Engineering error (Bug Verifier) | Total failed |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fable 5.2 | · | · | · | · | · | · | · | · |
| Fable 5.1 | · | · | · | 2 | · | · | · | 2 |
| Opus 5.5 | · | · | · | · | · | · | 1 | 1 |
| Sonnet 5.5 | · | · | · | · | · | · | · | · |
| Haiku 5.5 | · | · | · | · | · | · | · | · |
| Haiku 4.5 | · | · | · | 2 | 7 | 1 | 3 | 13 |
| Gemini 4 Neon | · | · | · | 1 | · | · | · | 1 |
| Gemini 4 Carbon | · | · | · | · | · | 1 | · | 1 |
| Gemini 4 Argon | · | · | · | · | · | · | · | · |
| GPT-6.2 Astra | · | · | · | · | · | · | · | · |
| GPT-6 Astra | · | · | · | · | · | · | · | · |
| GPT-6.2 Sol | · | · | · | · | 1 | · | · | 1 |
| GPT-6.1 Sol | · | · | · | · | · | · | 2 | 2 |
| GPT-6 Sol | · | · | · | · | · | · | · | · |
| GPT-5.6 Terra | · | · | · | · | · | 1 | · | 1 |
| GPT-6 Luna | · | · | 1 | 1 | 1 | · | · | 3 |
| Kimi K3.1 Pro | · | · | · | · | · | 1 | · | 1 |
| Kimi K3 | · | · | · | 1 | · | 3 | 1 | 5 |
| GLM-5.2 | · | · | · | · | · | · | · | · |
| GLM-5.3-Flash | 2 | · | · | 2 | · | 2 | 1 | 7 |
| DeepSeek-V4-Flash | · | · | · | 1 | · | 4 | 1 | 6 |
| DeepSeek-V4-Flash-R | · | · | · | · | 1 | · | 2 | 3 |
| Qwen3.8-27B | 1 | · | · | · | · | 1 | 1 | 3 |
| Qwen3.6-35B-A3B | 1 | 4 | 1 | 4 | 3 | · | 1 | 14 |
| MiMo-V2.6-Pro | 1 | · | · | 2 | · | · | · | 3 |
| Seed-OSS-36B | 2 | 4 | 1 | 5 | 6 | · | · | 18 |
| Nemotron 3 Ultra | · | 2 | · | · | 9 | · | 1 | 12 |
| Nemotron 3.5 Super | · | · | · | · | 5 | · | · | 5 |
| **All** | **7** | **10** | **3** | **21** | **33** | **14** | **14** | **102** |

</div>

*Table 1. Failed sessions by model, stage, and kind. Verifier: no solution submitted, a solution that does not build under our toolchain, a crash, a wrong output, or a run more than 1.5× slower than the multi-threaded CPU reference. Bug Verifier: wrong outputs and engineering errors on allowed inputs beyond the Verifier's, described below. Too slow is a performance gate, not a bug.*

Most Verifier failures were within the agent's reach: for 68 of the 74, the Analyzer (G7) judges that a check the agent could have run would have caught the failure. Beyond the 28 failures, the Bug Verifier also reproduces 27 borderline bugs in 25 passed solutions that are kept and marked, and it does not confirm 21 of the Analyzer's 77 claims ([Analysis 7](07-bugs-beyond-the-verifier.md)).

## How Agents Handle Bugs During a Session

The failures above are what remains at the end. During a session, agents meet many more bugs in their own code and fix most of them. For every significant error in a trace, the Analyzer (G5) records its kind and what the agent did next. Five kinds are bugs in the agent's code: a wrong result its own test reports, a hidden defect or edge case found by reading the code or on an unusual size, a crash, a hang or timeout, and a launch or resource limit such as too much shared memory. Across the 840 sessions the Analyzer lists 1,936 such bugs, 2.3 per session; the other errors it lists, such as compile errors and slower versions, are development setbacks and are summarized in [Analysis 4](04-session-behavior.md#development-setbacks).

Each bar below is one model: its length is the bugs per session, and its segments show how the agent responded. The column on the right counts the bugs left at the end, from Table 1: wrong outputs and crashes caught by the Verifier and failures found by the Bug Verifier. Choose a kind to see it alone; hover a segment for the count.

<div data-njs-errors="wrong result,latent defect or edge case,crash or runtime error,timeout or hang,kernel launch or resource limit" data-final="1"></div>
<script src="../../javascripts/errors-matrix-data.js"></script>
<script src="../../javascripts/errors-bars.js"></script>

**Results:** Agents fix the bugs their tests report and miss those that stay silent. They fix 66% of wrong results and 63% of crashes after a diagnosis, and ignore only 2% to 4% of them. Hidden defects and edge cases are different: agents fix 60% but ignore or do not notice 24%, and these silent defects are the kind the Bug Verifier finds after the Verifier has passed them ([Analysis 7](07-bugs-beyond-the-verifier.md)). Hangs and timeouts are more often worked around, for example with a time limit or a fallback, than diagnosed (42% against 35%).
