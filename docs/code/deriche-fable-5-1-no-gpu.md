# `deriche`: Fable 5.1 without a GPU

| Kernel | Model | Framework | Roofline attainment: no GPU → working GPU | Shown as |
| :--- | :--- | :--- | :---: | :--- |
| `deriche` | Fable 5.1 | Claude Code | 26.1% → 24.5% | CUDA kernel simulated on the CPU |

This session ran entirely on the faulty card of the [no-GPU incident](../posts/05-profilers-and-no-gpu.md#no-gpu-incident): every CUDA call failed with error 46. Fable 5.1 still submitted a solution that passes the Verifier on a working B300 GPU. To test kernels it could not run, it wrote a host-side SIMT simulator, `sim_shim.h`, that compiles the actual `solution.cu` with g++ and runs one `std::thread` per CUDA thread. Real barriers stand in for `__syncthreads` and `__syncwarp`, and per-warp exchange buffers stand in for the warp shuffles. Shared memory is filled with NaN and surrounded by NaN guard zones, so a stray read shows up as a NaN mismatch and a stray write corrupts a guard. The harness `sim_test.cpp` then compares the simulated kernels bit for bit with a CPU emulation of the same arithmetic, `emu.h`, on 14 shapes chosen to cover every code path. As a guard for what the simulator cannot check, such as launch limits, `pb_setup` runs a self-test on the evaluation GPU and falls back to simple one-thread-per-line kernels if the fast ones disagree with a host baseline.

Without timings, the agent estimated performance from the SASS instruction mix instead of measuring it. Its fresh re-run on a working GPU reaches a similar attainment, 24.5%; on `deriche`, no model in the study exceeds 28% of the bound (Opus 5.5 at 27.8%).

The files below are unmodified: the frozen solution, the agent's own `NOTES.md`, and the three simulator files from its scratch directory. Downloads: [`solution.cu`](files/deriche-fable-5-1-no-gpu/solution.cu) · [`NOTES.md`](files/deriche-fable-5-1-no-gpu/NOTES.txt) · [`sim_shim.h`](files/deriche-fable-5-1-no-gpu/sim_shim.h) · [`sim_test.cpp`](files/deriche-fable-5-1-no-gpu/sim_test.cpp) · [`emu.h`](files/deriche-fable-5-1-no-gpu/emu.h) · [reference `kernel.c`](files/deriche-fable-5-1-no-gpu/kernel.c) · [PolyBench license](files/POLYBENCH-LICENSE.txt).

## Reference Kernel

The serial C kernel from PolyBench/C 4.2.1 that the agent was asked to port:

```c
--8<-- "docs/code/files/deriche-fable-5-1-no-gpu/kernel.c"
```

## The Agent's SIMT Simulator

??? example "`sim_shim.h` (113 lines): CUDA built-ins mapped onto CPU threads (click to expand)"

    ```cpp
    --8<-- "docs/code/files/deriche-fable-5-1-no-gpu/sim_shim.h"
    ```

??? example "`sim_test.cpp` (98 lines): runs `solution.cu` in the simulator and compares it with the emulation (click to expand)"

    ```cpp
    --8<-- "docs/code/files/deriche-fable-5-1-no-gpu/sim_test.cpp"
    ```

??? example "`emu.h` (238 lines): CPU emulation of the kernel arithmetic (click to expand)"

    ```cpp
    --8<-- "docs/code/files/deriche-fable-5-1-no-gpu/emu.h"
    ```

## Agent's Solution

??? example "`solution.cu` (748 lines, click to expand)"

    ````cuda
    --8<-- "docs/code/files/deriche-fable-5-1-no-gpu/solution.cu"
    ````

## Agent's Notes

The section "Important caveat: hardware state during this session" describes the simulator in the agent's own words.

??? note "`NOTES.md` (click to expand)"

    ````markdown
    --8<-- "docs/code/files/deriche-fable-5-1-no-gpu/NOTES.txt"
    ````
