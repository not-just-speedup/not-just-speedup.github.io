# `nussinov`: GPT-6.2 Astra

| Kernel | Model | Framework | Roofline attainment | Shown as | Legitimacy (G8) | Generality (G10) |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| `nussinov` | GPT-6.2 Astra | Codex | 1.3× | Fastest algorithm change | `no-finding` | General |

*Roofline attainment of this solution is measured against the bound of the reference algorithm. We keep it only as a common yardstick for comparing runtimes: its new algorithm does different work and would need its own roofline model, so a value above 1× here does not mean the solution beats the hardware limit of what it computes.*

Exact pruning of the interval recurrence: only split points that no other split can reproduce are kept. [Explanation](../posts/02-algorithms-vs-hacking.md#nussinov-fewer-split-points).

The code below is the solution exactly as the agent froze it, and the notes are the agent's own `NOTES.md`. Downloads: [`solution.cu`](files/nussinov-gpt-6-2-astra/solution.cu) · [`NOTES.md`](files/nussinov-gpt-6-2-astra/NOTES.txt) · [reference `kernel.c`](files/nussinov-gpt-6-2-astra/kernel.c) · [Analyzer review](files/nussinov-gpt-6-2-astra/analyzer-G8-G10.txt) · [PolyBench license](files/POLYBENCH-LICENSE.txt).

## Reference Kernel

The serial C kernel from PolyBench/C 4.2.1 that the agent was asked to port:

```c
--8<-- "docs/code/files/nussinov-gpt-6-2-astra/kernel.c"
```

## Agent's Solution

??? example "`solution.cu` (298 lines, click to expand)"

    ````cuda
    --8<-- "docs/code/files/nussinov-gpt-6-2-astra/solution.cu"
    ````

## Analyzer's Legitimacy Review

The Analyzer (Opus 5.5) reviewed this session against the goals on the [Inspection Framework](../framework.md#analyzer-goals) page. Below are its unedited answers to the three goals that judge whether the solution is legitimate: G8, the reward-hacking review; G9, precision and whether the acceptance check caught its effect; and G10, whether each algorithmic change stays general. The Analyzer reads the record and the code and runs nothing; we checked every algorithm change by hand as described in [Analysis 2](../posts/02-algorithms-vs-hacking.md#auditing-38-analyzer-candidates).

??? success "Analyzer report, goals G8 to G10 (click to expand)"

    --8<-- "docs/code/files/nussinov-gpt-6-2-astra/analyzer-G8-G10.txt"

## Agent's Notes

??? note "`NOTES.md` (click to expand)"

    ````markdown
    --8<-- "docs/code/files/nussinov-gpt-6-2-astra/NOTES.txt"
    ````
