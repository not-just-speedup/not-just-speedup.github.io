// Host-side SIMT simulator shim: lets the actual CUDA kernel source be compiled with
// g++ and executed with one std::thread per CUDA thread, with real barriers for
// __syncwarp/__syncthreads and shuffle exchange through per-warp buffers.
#pragma once
#include <cuda_runtime.h>  // vector types (host pass)
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>

#undef __global__
#define __global__
#undef __launch_bounds__
#define __launch_bounds__(...)
#undef __device__
#define __device__
#undef __host__
#define __host__
#undef __forceinline__
#define __forceinline__ inline

class SimBarrier {
  int n_, count_ = 0;
  unsigned gen_ = 0;
  std::mutex m_;
  std::condition_variable cv_;
 public:
  explicit SimBarrier(int n) : n_(n) {}
  void wait() {
    std::unique_lock<std::mutex> lk(m_);
    const unsigned g = gen_;
    if (++count_ == n_) { count_ = 0; ++gen_; cv_.notify_all(); }
    else cv_.wait(lk, [&] { return g != gen_; });
  }
};
struct SimWarp {
  SimBarrier bar{32};
  float xchg[2][32];
};
struct SimCtx {
  int tid = 0, lane = 0, warp = 0, block = 0, bdim = 0;
  float* smem = nullptr;
  SimWarp* ws = nullptr;
  SimBarrier* bs = nullptr;
  int parity = 0;
};
inline SimCtx& sim_ctx() { static thread_local SimCtx c; return c; }
struct SimDim { unsigned x; };
#define threadIdx (SimDim{(unsigned)sim_ctx().tid})
#define blockIdx (SimDim{(unsigned)sim_ctx().block})
#define blockDim (SimDim{(unsigned)sim_ctx().bdim})
#define PB_SMEM_DECL(name) float* name = sim_ctx().smem

inline float sim_shfl_generic(float v, int src, bool valid) {
  SimCtx& c = sim_ctx();
  SimWarp& w = *c.ws;
  const int p = c.parity;
  c.parity ^= 1;
  w.xchg[p][c.lane] = v;
  w.bar.wait();
  return valid ? w.xchg[p][src] : v;
}
inline float __shfl_down_sync(unsigned, float v, int d) {
  const int src = sim_ctx().lane + d;
  return sim_shfl_generic(v, src, src < 32);
}
inline float __shfl_up_sync(unsigned, float v, int d) {
  const int src = sim_ctx().lane - d;
  return sim_shfl_generic(v, src, src >= 0);
}
inline float __shfl_sync(unsigned, float v, int srcLane) {
  return sim_shfl_generic(v, srcLane & 31, true);
}
inline void __syncwarp(unsigned = 0xffffffffu) { sim_ctx().ws->bar.wait(); }
inline void __syncthreads() { sim_ctx().bs->wait(); }

// Run `body` for grid x block threads, blocks sequentially, threads of a block concurrently.
template <class F>
void sim_launch(unsigned grid, unsigned block, size_t smem_bytes, F body) {
  const int nwarps = (int)((block + 31) / 32);
  const size_t GUARD = 256;
  const float NAN_FILL = std::numeric_limits<float>::quiet_NaN();
  for (unsigned b = 0; b < grid; ++b) {
    // shared memory: NaN-filled (uninitialised on hardware) with NaN guard zones on both sides
    std::vector<float> smem(smem_bytes / sizeof(float) + 2 * GUARD, NAN_FILL);
    SimBarrier bs((int)block);
    std::vector<std::unique_ptr<SimWarp>> ws;
    for (int w = 0; w < nwarps; ++w) ws.emplace_back(new SimWarp());
    std::vector<std::thread> th;
    for (unsigned t = 0; t < block; ++t) {
      th.emplace_back([&, t, b] {
        SimCtx& c = sim_ctx();
        c.tid = (int)t; c.lane = (int)(t & 31); c.warp = (int)(t >> 5); c.block = (int)b; c.bdim = (int)block;
        c.smem = smem.data() + GUARD; c.ws = ws[c.warp].get(); c.bs = &bs; c.parity = 0;
        body();
      });
    }
    for (auto& x : th) x.join();
    for (size_t g = 0; g < GUARD; ++g) {
      if (!std::isnan(smem[g]) || !std::isnan(smem[smem.size() - 1 - g])) {
        std::fprintf(stderr, "SMEM GUARD CORRUPTED in block %u (offset %zu)\n", b, g);
        std::abort();
      }
    }
  }
}
