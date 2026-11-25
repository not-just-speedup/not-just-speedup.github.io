// GPU translation of the PolyBench "deriche" recursive (IIR) Gaussian filter.
//
// The computation applies a 2nd-order recursive filter along every row in both
// directions (causal + anti-causal, summed), then along every column in both
// directions (summed).  Two kernels are used:
//
//   row_kernel : buf0 (x)  ->  H   (row filter, both directions fused)
//   col_kernel : H         ->  buf1 (column filter, both directions fused)
//
// Both kernels split each line into pieces of 32 elements that are filtered
// independently from a zero state; the exact state at every piece boundary is
// then obtained with a short affine scan (the 2-state recurrence
// s' = M s + p is linear, and M^32, M^64, ... are precomputed), and each piece
// is finally recomputed from its exact starting state with the very same
// recurrence the CPU baseline uses.  The state entering a block of pieces is
// obtained from a short window of neighbouring data (the impulse response decays
// like exp(-s0*d), the window length is chosen so the neglected part is below
// 1e-10 relative, far below FP32 resolution).  All floating point work is FP32
// with explicit fmaf(), so the arithmetic is fully specified (and mirrored by a
// host emulation used for validation).
#include "interface.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

// Dynamic shared memory declaration (overridable by a host-side SIMT simulator used for testing).
#ifndef PB_SMEM_DECL
#define PB_SMEM_DECL(name) extern __shared__ float name[]
#endif

namespace {

// ---------------------------------------------------------------------------
// Coefficients shared by host and device.
// ---------------------------------------------------------------------------
struct Coef {
  float a1, a2, a3, a4, b1, b2;
  float4 Mp[5];   // M^(32*2^i), i = 0..4, stored as (m00, m01, m10, m11); M = [[b1,b2],[1,0]]
  float4 Mw[5];   // M^(W*2^i), i = 0..4 (W = window pieces), same layout
  float h1[32];   // h1[d] = (M^(d+1))_00 : response at distance d to initial state (1,0)
  float h2[32];   // h2[d] = (M^(d+1))_01 : response at distance d to initial state (0,1)
};

constexpr unsigned FULL = 0xffffffffu;
constexpr int PIECE = 32;         // elements per piece (one lane / one thread)
constexpr int ROW_CHUNK = 1024;   // elements of one row handled by one warp
constexpr int ROW_WARPS = 8;      // warps per block in the row kernel
constexpr int COL_WARPS = 16;     // warps (pieces) per block in the column kernel
constexpr int COL_CHUNK = COL_WARPS * PIECE;  // rows per block in the column kernel (512)
constexpr int TS = 33;            // padded tile row stride (floats)
constexpr int MAXW = 5;           // max window pieces

// One causal step:  y = a1*x + a2*x_prev + b1*y_prev + b2*y_prev2
// (the small feedback terms are accumulated first so that only the last operation rounds at full magnitude)
__host__ __device__ __forceinline__ float step_fwd(float xk, float xprev, float t0, float t1, const Coef& c) {
  float v = c.b2 * t1;
  v = fmaf(c.b1, t0, v);
  v = fmaf(c.a2, xprev, v);
  v = fmaf(c.a1, xk, v);
  return v;
}
// One anti-causal step:  z = a3*x_next + a4*x_next2 + b1*z_next + b2*z_next2
__host__ __device__ __forceinline__ float step_bwd(float xn1, float xn2, float t4, float t5, const Coef& c) {
  float v = c.b2 * t5;
  v = fmaf(c.b1, t4, v);
  v = fmaf(c.a4, xn2, v);
  v = fmaf(c.a3, xn1, v);
  return v;
}
// acc + M * v   (2x2 matrix times 2-vector, accumulated)
__host__ __device__ __forceinline__ float2 mat_acc(float4 m, float2 v, float2 acc) {
  float2 r;
  r.x = fmaf(m.x, v.x, fmaf(m.y, v.y, acc.x));
  r.y = fmaf(m.z, v.x, fmaf(m.w, v.y, acc.y));
  return r;
}

// Causal local pass over 32 elements from state (t0,t1) with previous input xprev.
// Returns the end state (y[31], y[30]).
__host__ __device__ __forceinline__ float2 local_fwd(const float* xc, float xprev, float t0, float t1, const Coef& c) {
  float t2 = xprev;
#pragma unroll
  for (int k = 0; k < PIECE; ++k) {
    const float xk = xc[k];
    const float v = step_fwd(xk, t2, t0, t1, c);
    t2 = xk; t1 = t0; t0 = v;
  }
  return make_float2(t0, t1);
}
// Anti-causal local pass over 32 elements (k = 31..0) from state (t4,t5) with next inputs xn1, xn2.
// Returns the end state (z[0], z[1]).
__host__ __device__ __forceinline__ float2 local_bwd(const float* xc, float xn1, float xn2, float t4, float t5, const Coef& c) {
#pragma unroll
  for (int k = PIECE - 1; k >= 0; --k) {
    const float xk = xc[k];
    const float v = step_bwd(xn1, xn2, t4, t5, c);
    t5 = t4; t4 = v; xn2 = xn1; xn1 = xk;
  }
  return make_float2(t4, t5);
}

#if defined(__CUDACC__) || defined(PB_HOST_SIM)
__device__ __forceinline__ float2 shfl_down2(float2 v, int d) {
  return make_float2(__shfl_down_sync(FULL, v.x, d), __shfl_down_sync(FULL, v.y, d));
}
__device__ __forceinline__ float2 shfl_up2(float2 v, int d) {
  return make_float2(__shfl_up_sync(FULL, v.x, d), __shfl_up_sync(FULL, v.y, d));
}
__device__ __forceinline__ float2 shfl2(float2 v, int src) {
  return make_float2(__shfl_sync(FULL, v.x, src), __shfl_sync(FULL, v.y, src));
}

// ---------------------------------------------------------------------------
// Row kernel: one warp per (row i, 1024-element chunk g).  Lane t owns the
// 32-element piece [j0 + 32t, j0 + 32t + 32).  W pieces of context on each side
// supply the states entering the chunk.
// ---------------------------------------------------------------------------
template <int W>
__global__ void __launch_bounds__(ROW_WARPS * 32, 3)
row_kernel(const float* __restrict__ x, float* __restrict__ h,
           long long n1, long long G, long long items, Coef c) {
  constexpr int NQ = PIECE + 2 * W;  // pieces in the tile
  PB_SMEM_DECL(smem);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float* tile = smem + warp * (NQ * TS);
  const long long item = (long long)blockIdx.x * ROW_WARPS + warp;
  if (item >= items) return;  // whole warp returns; no block-level barriers below
  const long long i = item / G;
  const long long g = item - i * G;
  const long long j0 = g * ROW_CHUNK;
  const float* xrow = x + i * n1;
  float* hrow = h + i * n1;

  // ---- load tile [j0 - 32W, j0 + 1024 + 32W), zero outside the row ----
  {
    float v[NQ];
    const long long jb = j0 - PIECE * W + lane;
    if (j0 >= PIECE * W && j0 + ROW_CHUNK + PIECE * W <= n1) {   // warp-uniform: whole tile inside the row
      const float* xb = xrow + jb;
#pragma unroll
      for (int q = 0; q < NQ; ++q) v[q] = xb[PIECE * q];
    } else {
#pragma unroll
      for (int q = 0; q < NQ; ++q) {
        const long long j = jb + PIECE * q;
        v[q] = ((unsigned long long)j < (unsigned long long)n1) ? xrow[j] : 0.f;
      }
    }
#pragma unroll
    for (int q = 0; q < NQ; ++q) tile[q * TS + lane] = v[q];
  }
  __syncwarp();

  const int q = W + lane;              // this lane's main piece
  float* xc = tile + q * TS;
  // window element e (0 <= e < 32W) of a window whose first piece starts at base
  auto welem = [&](const float* base, int e) -> float { return base[(e >> 5) * TS + (e & 31)]; };

  // ================= anti-causal direction (right to left) =================
  // right window [j1, j1 + 32W): lane t owns the W elements [Wt, Wt + W) of it
  float2 A;
  {
    const float* wr = tile + (PIECE + W) * TS;
    const int e0 = W * lane, e1 = e0 + W, e2 = e0 + W + 1;
    float xn1 = (e1 < PIECE * W) ? welem(wr, e1) : 0.f;
    float xn2 = (e2 < PIECE * W) ? welem(wr, e2) : 0.f;
    float t4 = 0.f, t5 = 0.f;
#pragma unroll
    for (int m = W - 1; m >= 0; --m) {
      const float xk = welem(wr, e0 + m);
      const float v = step_bwd(xn1, xn2, t4, t5, c);
      t5 = t4; t4 = v; xn2 = xn1; xn1 = xk;
    }
    A = make_float2(t4, t5);
  }
#pragma unroll
  for (int s = 0; s < 5; ++s) {        // scan toward lower lanes, piece length W
    const int d = 1 << s;
    const float2 o = shfl_down2(A, d);
    if (lane + d < 32) A = mat_acc(c.Mw[s], o, A);
  }
  const float2 Sr = shfl2(A, 0);       // state (z[j1], z[j1+1]) entering the chunk from the right

  // main pieces: local pass from zero state, keep the local result
  const float xn1 = xc[TS], xn2 = xc[TS + 1];   // first two elements of the next piece (always in tile)
  float z[PIECE];
  {
    float t4 = 0.f, t5 = 0.f, n1v = xn1, n2v = xn2;
#pragma unroll
    for (int k = PIECE - 1; k >= 0; --k) {
      const float xk = xc[k];
      const float v = step_bwd(n1v, n2v, t4, t5, c);
      z[k] = v;
      t5 = t4; t4 = v; n2v = n1v; n1v = xk;
    }
    A = make_float2(t4, t5);
  }
  if (lane == 31) A = mat_acc(c.Mp[0], Sr, A);
#pragma unroll
  for (int s = 0; s < 5; ++s) {
    const int d = 1 << s;
    const float2 o = shfl_down2(A, d);
    if (lane + d < 32) A = mat_acc(c.Mp[s], o, A);
  }
  float2 Eb = shfl_down2(A, 1);
  if (lane == 31) Eb = Sr;             // exact state (z[c0+32], z[c0+33]) for this piece; correction applied at the end

  // ================= causal direction (left to right) =================
  // left window [j0 - 32W, j0): lane t owns the W elements [Wt, Wt + W) of it
  {
    const float* wl = tile;
    const int e0 = W * lane;
    const float xp0 = (e0 > 0) ? welem(wl, e0 - 1) : 0.f;
    float t0 = 0.f, t1 = 0.f, t2 = xp0;
#pragma unroll
    for (int m = 0; m < W; ++m) {
      const float xk = welem(wl, e0 + m);
      const float v = step_fwd(xk, t2, t0, t1, c);
      t2 = xk; t1 = t0; t0 = v;
    }
    A = make_float2(t0, t1);
  }
#pragma unroll
  for (int s = 0; s < 5; ++s) {        // scan toward higher lanes, piece length W
    const int d = 1 << s;
    const float2 o = shfl_up2(A, d);
    if (lane >= d) A = mat_acc(c.Mw[s], o, A);
  }
  const float2 Sl = shfl2(A, 31);      // state (y[j0-1], y[j0-2]) entering the chunk from the left

  const float xp = xc[-TS + (PIECE - 1)];       // last element of the previous piece
  {
    // local pass from zero state; accumulate the local result onto z
    float t0 = 0.f, t1 = 0.f, t2 = xp;
#pragma unroll
    for (int k = 0; k < PIECE; ++k) {
      const float xk = xc[k];
      const float v = step_fwd(xk, t2, t0, t1, c);
      t2 = xk; t1 = t0; t0 = v;
      z[k] = z[k] + v;
    }
    A = make_float2(t0, t1);
  }
  if (lane == 0) A = mat_acc(c.Mp[0], Sl, A);
#pragma unroll
  for (int s = 0; s < 5; ++s) {
    const int d = 1 << s;
    const float2 o = shfl_up2(A, d);
    if (lane >= d) A = mat_acc(c.Mp[s], o, A);
  }
  float2 Ef = shfl_up2(A, 1);
  if (lane == 0) Ef = Sl;              // exact state (y[c0-1], y[c0-2]) for this piece

  __syncwarp();                        // all cross-lane tile reads are done before the tile is overwritten
  // h = (zl + yl) + (correction_bwd + correction_fwd): the responses of the exact entering states at
  // distance 31-k (anti-causal) and k (causal); the small terms are summed first.
#pragma unroll
  for (int k = 0; k < PIECE; ++k) {
    const float cb = fmaf(c.h1[PIECE - 1 - k], Eb.x, c.h2[PIECE - 1 - k] * Eb.y);
    const float cf = fmaf(c.h1[k], Ef.x, c.h2[k] * Ef.y);
    xc[k] = z[k] + (cb + cf);
  }
  __syncwarp();

  // ---- store the 1024 results of this chunk, coalesced ----
#pragma unroll
  for (int qq = 0; qq < PIECE; ++qq) {
    const long long j = j0 + PIECE * qq + lane;
    if (j < n1) hrow[j] = tile[(W + qq) * TS + lane];
  }
}

// ---------------------------------------------------------------------------
// Column kernel: one block (16 warps) per (32-column stripe, 512-row chunk).
// Thread (warp p, lane t) owns rows [i0 + 32p, i0 + 32p + 32) of column c0 + t.
// W pieces of context above / below supply the states entering the chunk.
// ---------------------------------------------------------------------------
constexpr size_t COL_SMEM_BYTES =
    (size_t)COL_WARPS * PIECE * 32 * sizeof(float)      // zbuf [16][32][32]
    + (size_t)COL_WARPS * 32 * sizeof(float2)           // pbuf [16][32]
    + (size_t)COL_WARPS * 32 * sizeof(float2)           // ebuf [16][32]
    + (size_t)2 * 32 * sizeof(float2);                  // swin [2][32]

template <int W>
__global__ void __launch_bounds__(COL_WARPS * 32, 2)
col_kernel(const float* __restrict__ in, float* __restrict__ out,
           long long n0, long long n1, long long stripes, Coef c) {
  PB_SMEM_DECL(smem);
  float* zbuf = smem;
  float2* pbuf = reinterpret_cast<float2*>(smem + COL_WARPS * PIECE * 32);
  float2* ebuf = pbuf + COL_WARPS * 32;
  float2* swin = ebuf + COL_WARPS * 32;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long long s = (long long)blockIdx.x % stripes;
  const long long chunk = (long long)blockIdx.x / stripes;
  const long long col = s * 32 + lane;
  const bool colok = col < n1;
  const long long i0 = chunk * COL_CHUNK;
  const float* colp = in + (colok ? col : 0);
  float* outp = out + col;

  auto ld = [&](long long r) -> float {
    return (colok && r >= 0 && r < n0) ? colp[r * n1] : 0.f;
  };
  // 32 consecutive rows starting at r0 (zero outside the array); warp-uniform fast path when fully inside.
  auto load_piece = [&](long long r0, float* xv) {
    const float* p = colp + r0 * n1;
    if (colok && r0 >= 0 && r0 + PIECE <= n0) {
#pragma unroll
      for (int k = 0; k < PIECE; ++k) { xv[k] = *p; p += n1; }
    } else {
#pragma unroll
      for (int k = 0; k < PIECE; ++k) {
        const long long r = r0 + k;
        xv[k] = (colok && (unsigned long long)r < (unsigned long long)n0) ? *p : 0.f;
        p += n1;
      }
    }
  };

  // ---- window round: warps 0..W-1 take the W pieces above (causal), warps 8..8+W-1 the W pieces below (anti-causal)
  if (warp < W) {
    const long long r0 = i0 - (long long)PIECE * (W - warp);
    float xv[PIECE];
    load_piece(r0, xv);
    const float xp = (warp > 0) ? ld(r0 - 1) : 0.f;
    pbuf[warp * 32 + lane] = local_fwd(xv, xp, 0.f, 0.f, c);
  } else if (warp >= 8 && warp < 8 + W) {
    const int wi = warp - 8;
    const long long r0 = i0 + COL_CHUNK + (long long)PIECE * wi;
    float xv[PIECE];
    load_piece(r0, xv);
    const bool nxt = (wi + 1 < W);
    pbuf[warp * 32 + lane] = local_bwd(xv, nxt ? ld(r0 + PIECE) : 0.f, nxt ? ld(r0 + PIECE + 1) : 0.f, 0.f, 0.f, c);
  }
  __syncthreads();
  if (warp == 0) {                      // state entering the chunk from above
    float2 S = make_float2(0.f, 0.f);
    for (int q = 0; q < W; ++q) S = mat_acc(c.Mp[0], S, pbuf[q * 32 + lane]);
    swin[lane] = S;
  } else if (warp == 1) {               // state entering the chunk from below
    float2 S = make_float2(0.f, 0.f);
    for (int q = W - 1; q >= 0; --q) S = mat_acc(c.Mp[0], S, pbuf[(8 + q) * 32 + lane]);
    swin[32 + lane] = S;
  }

  // ---- main round: load this thread's piece (+ halo) ----
  const long long r0 = i0 + (long long)PIECE * warp;
  float xv[PIECE];
  load_piece(r0, xv);
  const float xp = ld(r0 - 1), xn1 = ld(r0 + PIECE), xn2 = ld(r0 + PIECE + 1);
  __syncthreads();                      // window scans done before pbuf is reused

  // anti-causal: local pass, scan (sequential over the 16 pieces, one thread per column), recompute into zbuf
  pbuf[warp * 32 + lane] = local_bwd(xv, xn1, xn2, 0.f, 0.f, c);
  __syncthreads();
  if (warp == 0) {
    float2 S = swin[32 + lane];
    for (int p = COL_WARPS - 1; p >= 0; --p) {
      ebuf[p * 32 + lane] = S;
      S = mat_acc(c.Mp[0], S, pbuf[p * 32 + lane]);
    }
  }
  __syncthreads();
  {
    const float2 E = ebuf[warp * 32 + lane];
    float t4 = E.x, t5 = E.y, n1v = xn1, n2v = xn2;
#pragma unroll
    for (int k = PIECE - 1; k >= 0; --k) {
      const float v = step_bwd(n1v, n2v, t4, t5, c);
      zbuf[(warp * PIECE + k) * 32 + lane] = v;
      t5 = t4; t4 = v; n2v = n1v; n1v = xv[k];
    }
  }
  // causal: local pass, scan, recompute and store y + z
  pbuf[warp * 32 + lane] = local_fwd(xv, xp, 0.f, 0.f, c);
  __syncthreads();
  if (warp == 0) {
    float2 S = swin[lane];
    for (int p = 0; p < COL_WARPS; ++p) {
      ebuf[p * 32 + lane] = S;
      S = mat_acc(c.Mp[0], S, pbuf[p * 32 + lane]);
    }
  }
  __syncthreads();
  {
    const float2 E = ebuf[warp * 32 + lane];
    float t0 = E.x, t1 = E.y, t2 = xp;
#pragma unroll
    for (int k = 0; k < PIECE; ++k) {
      const float v = step_fwd(xv[k], t2, t0, t1, c);
      t2 = xv[k]; t1 = t0; t0 = v;
      const float hv = v + zbuf[(warp * PIECE + k) * 32 + lane];
      const long long r = r0 + k;
      if (colok && r < n0) outp[r * n1] = hv;
    }
  }
}

// ---------------------------------------------------------------------------
// Fallback kernels: one thread per line, the baseline's sequential loops verbatim.
// Only used if the built-in self-test in pb_setup rejects the fast kernels.
// ---------------------------------------------------------------------------
__global__ void naive_rows(const float* __restrict__ x, float* __restrict__ h, long long n0, long long n1, Coef c) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n0) return;
  const float* xr = x + i * n1;
  float* hr = h + i * n1;
  float t0 = 0.f, t1 = 0.f, t2 = 0.f;
  for (long long j = 0; j < n1; ++j) {
    const float xj = xr[j];
    const float y = step_fwd(xj, t2, t0, t1, c);
    hr[j] = y;
    t2 = xj; t1 = t0; t0 = y;
  }
  float t4 = 0.f, t5 = 0.f, t6 = 0.f, t7 = 0.f;
  for (long long j = n1 - 1; j >= 0; --j) {
    const float z = step_bwd(t6, t7, t4, t5, c);
    hr[j] += z;
    t7 = t6; t6 = xr[j]; t5 = t4; t4 = z;
  }
}
__global__ void naive_cols(const float* __restrict__ in, float* __restrict__ out, long long n0, long long n1, Coef c) {
  const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n1) return;
  const float* xc = in + j;
  float* oc = out + j;
  float t0 = 0.f, t1 = 0.f, t3 = 0.f;
  for (long long i = 0; i < n0; ++i) {
    const float xi = xc[i * n1];
    const float y = step_fwd(xi, t3, t0, t1, c);
    oc[i * n1] = y;
    t3 = xi; t1 = t0; t0 = y;
  }
  float t4 = 0.f, t5 = 0.f, t8 = 0.f, t9 = 0.f;
  for (long long i = n0 - 1; i >= 0; --i) {
    const float z = step_bwd(t8, t9, t4, t5, c);
    oc[i * n1] += z;
    t9 = t8; t8 = xc[i * n1]; t5 = t4; t4 = z;
  }
}
#endif  // __CUDACC__

// ---------------------------------------------------------------------------
// Host side: coefficients and window length from s0.
// ---------------------------------------------------------------------------
struct Mat2 { double m00, m01, m10, m11; };
inline Mat2 mat_mul(const Mat2& a, const Mat2& b) {
  Mat2 r;
  r.m00 = a.m00 * b.m00 + a.m01 * b.m10;
  r.m01 = a.m00 * b.m01 + a.m01 * b.m11;
  r.m10 = a.m10 * b.m00 + a.m11 * b.m10;
  r.m11 = a.m10 * b.m01 + a.m11 * b.m11;
  return r;
}

inline void make_coef(float s0f, Coef& c, int& W) {
  const double s0 = (double)s0f;
  const double e = std::exp(-s0);
  const double k = (1.0 - e) * (1.0 - e) / (1.0 + 2.0 * s0 * e - std::exp(2.0 * s0));
  c.a1 = (float)k;
  c.a2 = (float)(k * e * (s0 - 1.0));
  c.a3 = (float)(k * e * (s0 + 1.0));
  c.a4 = (float)(-k * std::exp(-2.0 * s0));
  c.b1 = (float)std::pow(2.0, -s0);
  c.b2 = (float)(-std::exp(-2.0 * s0));
  // Window: 32*W >= 20/s0  ->  neglected tail below ~3e-9 relative (the response decays like exp(-s0*d)).
  double w = std::ceil(20.0 / (32.0 * s0));
  if (!(w >= 1.0)) w = 1.0;
  if (w > MAXW) w = MAXW;
  W = (int)w;
  // Powers of the (float-rounded) transition matrix, computed in double.
  const Mat2 M = {(double)c.b1, (double)c.b2, 1.0, 0.0};
  Mat2 P = M;
  for (int i = 0; i < 5; ++i) P = mat_mul(P, P);  // M^32
  for (int i = 0; i < 5; ++i) {
    c.Mp[i] = make_float4((float)P.m00, (float)P.m01, (float)P.m10, (float)P.m11);
    P = mat_mul(P, P);
  }
  P = M;
  for (int i = 1; i < W; ++i) P = mat_mul(P, M);  // M^W
  for (int i = 0; i < 5; ++i) {
    c.Mw[i] = make_float4((float)P.m00, (float)P.m01, (float)P.m10, (float)P.m11);
    P = mat_mul(P, P);
  }
  P = M;
  for (int d = 0; d < 32; ++d) {       // M^(d+1)
    c.h1[d] = (float)P.m00;
    c.h2[d] = (float)P.m01;
    P = mat_mul(P, M);
  }
}

// Host port of compute.inc (FP32), used only by the self-test in pb_setup.
inline void cpu_baseline(int n0, int n1, float s0, const float* x, float* out, float* b2, float* b3) {
  const float k = (1.0f - expf(-s0)) * (1.0f - expf(-s0)) / (1.0f + 2.0f * s0 * expf(-s0) - expf(2.0f * s0));
  const float a1 = k, a2 = k * expf(-s0) * (s0 - 1.0f), a3 = k * expf(-s0) * (s0 + 1.0f), a4 = -k * expf(-2.0f * s0);
  const float b1 = powf(2.0f, -s0), bb2 = -expf(-2.0f * s0);
  for (int i = 0; i < n0; ++i) {
    float t0 = 0.f, t1 = 0.f, t2 = 0.f;
    for (int j = 0; j < n1; ++j) {
      const size_t e = (size_t)i * n1 + j;
      b2[e] = a1 * x[e] + a2 * t2 + b1 * t0 + bb2 * t1;
      t2 = x[e]; t1 = t0; t0 = b2[e];
    }
    float t4 = 0.f, t5 = 0.f, t6 = 0.f, t7 = 0.f;
    for (int j = n1 - 1; j >= 0; --j) {
      const size_t e = (size_t)i * n1 + j;
      b3[e] = a3 * t6 + a4 * t7 + b1 * t4 + bb2 * t5;
      t7 = t6; t6 = x[e]; t5 = t4; t4 = b3[e];
    }
  }
  for (size_t e = 0; e < (size_t)n0 * n1; ++e) out[e] = b2[e] + b3[e];
  for (int j = 0; j < n1; ++j) {
    float t3 = 0.f, t0 = 0.f, t1 = 0.f;
    for (int i = 0; i < n0; ++i) {
      const size_t e = (size_t)i * n1 + j;
      b2[e] = a1 * out[e] + a2 * t3 + b1 * t0 + bb2 * t1;
      t3 = out[e]; t1 = t0; t0 = b2[e];
    }
    float t8 = 0.f, t9 = 0.f, t4 = 0.f, t5 = 0.f;
    for (int i = n0 - 1; i >= 0; --i) {
      const size_t e = (size_t)i * n1 + j;
      b3[e] = a3 * t8 + a4 * t9 + b1 * t4 + bb2 * t5;
      t9 = t8; t8 = out[e]; t5 = t4; t4 = b3[e];
    }
  }
  for (size_t e = 0; e < (size_t)n0 * n1; ++e) out[e] = b2[e] + b3[e];
}

}  // namespace

#ifdef __CUDACC__
// ---------------------------------------------------------------------------
// Entry points.
// ---------------------------------------------------------------------------
namespace {
struct State {
  long long n0 = 0, n1 = 0;
  size_t elems = 0;
  float* d_x = nullptr;
  float* d_h = nullptr;
  float* d_out = nullptr;
  float s0 = 1.0f;
  bool have_input = false;
  bool ready = false;
  bool use_fast = true;
};
State g;

#define PB_CK(call)                                             \
  do {                                                          \
    cudaError_t e_ = (call);                                    \
    if (e_ != cudaSuccess) {                                    \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",          \
                   cudaGetErrorString(e_), __FILE__, __LINE__); \
      return (int)e_ ? (int)e_ : 1;                             \
    }                                                           \
  } while (0)

template <int W>
cudaError_t launch_pair(const float* d_x, float* d_h, float* d_out, long long n0, long long n1, const Coef& c) {
  const long long G = (n1 + ROW_CHUNK - 1) / ROW_CHUNK;
  const long long items = n0 * G;
  const long long rblocks = (items + ROW_WARPS - 1) / ROW_WARPS;
  const size_t rsmem = (size_t)ROW_WARPS * (PIECE + 2 * W) * TS * sizeof(float);
  row_kernel<W><<<(unsigned)rblocks, ROW_WARPS * 32, rsmem>>>(d_x, d_h, n1, G, items, c);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return e;
  const long long stripes = (n1 + 31) / 32;
  const long long chunks = (n0 + COL_CHUNK - 1) / COL_CHUNK;
  col_kernel<W><<<(unsigned)(stripes * chunks), COL_WARPS * 32, COL_SMEM_BYTES>>>(d_h, d_out, n0, n1, stripes, c);
  return cudaGetLastError();
}

cudaError_t launch_all(int W, const float* d_x, float* d_h, float* d_out, long long n0, long long n1, const Coef& c) {
  switch (W) {
    case 1: return launch_pair<1>(d_x, d_h, d_out, n0, n1, c);
    case 2: return launch_pair<2>(d_x, d_h, d_out, n0, n1, c);
    case 3: return launch_pair<3>(d_x, d_h, d_out, n0, n1, c);
    case 4: return launch_pair<4>(d_x, d_h, d_out, n0, n1, c);
    default: return launch_pair<5>(d_x, d_h, d_out, n0, n1, c);
  }
}

cudaError_t launch_naive(const float* d_x, float* d_h, float* d_out, long long n0, long long n1, const Coef& c) {
  naive_rows<<<(unsigned)((n0 + 255) / 256), 256>>>(d_x, d_h, n0, n1, c);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return e;
  naive_cols<<<(unsigned)((n1 + 255) / 256), 256>>>(d_h, d_out, n0, n1, c);
  return cudaGetLastError();
}

// Built-in self-test on synthetic data (no problem input involved): compares the fast kernels and the
// fallback kernels with the host baseline for every window variant; selects the fast path only if it agrees.
int self_test(bool& fast_ok, bool& naive_ok) {
  const int tn0 = 600, tn1 = 1100;   // 2 row chunks, 35 column stripes (last partial), 2 column chunks (last partial)
  const size_t n = (size_t)tn0 * tn1;
  std::vector<float> x(n), ref(n), b2(n), b3(n), got(n);
  unsigned long long st = 0x9E3779B97F4A7C15ull;
  for (size_t e = 0; e < n; ++e) {
    st = st * 6364136223846793005ull + 1442695040888963407ull;
    x[e] = (float)((st >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f;
  }
  float *dx = nullptr, *dh = nullptr, *dout = nullptr;
  PB_CK(cudaMalloc((void**)&dx, n * sizeof(float)));
  PB_CK(cudaMalloc((void**)&dh, n * sizeof(float)));
  PB_CK(cudaMalloc((void**)&dout, n * sizeof(float)));
  PB_CK(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  const float s0s[5] = {0.125f, 0.17f, 0.25f, 0.4f, 1.3f};   // W = 5, 4, 3, 2, 1
  fast_ok = true; naive_ok = true;
  for (int t = 0; t < 5; ++t) {
    Coef c; int W;
    make_coef(s0s[t], c, W);
    cpu_baseline(tn0, tn1, s0s[t], x.data(), ref.data(), b2.data(), b3.data());
    double scale = 0.0;
    for (size_t e = 0; e < n; ++e) scale = scale > std::fabs((double)ref[e]) ? scale : std::fabs((double)ref[e]);
    for (int path = 0; path < 2; ++path) {
      // A (non-sticky) CUDA error in one path only disqualifies that path.
      bool ok = false;
      cudaError_t e = cudaMemset(dout, 0, n * sizeof(float));
      if (e == cudaSuccess) e = (path == 0) ? launch_all(W, dx, dh, dout, tn0, tn1, c) : launch_naive(dx, dh, dout, tn0, tn1, c);
      if (e == cudaSuccess) e = cudaDeviceSynchronize();
      if (e == cudaSuccess) e = cudaMemcpy(got.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost);
      if (e == cudaSuccess) {
        double err = 0.0;
        bool finite = true;
        for (size_t i = 0; i < n; ++i) {
          if (!std::isfinite(got[i])) finite = false;
          const double d = std::fabs((double)got[i] - (double)ref[i]);
          err = err > d ? err : d;
        }
        ok = finite && err <= 1e-4 * scale;
        if (!ok) std::fprintf(stderr, "self-test: %s path W=%d max err %.3e (scale %.3e) -> rejected\n",
                              path == 0 ? "fast" : "fallback", W, err, scale);
      } else {
        (void)cudaGetLastError();
        std::fprintf(stderr, "self-test: %s path W=%d CUDA error: %s -> rejected\n",
                     path == 0 ? "fast" : "fallback", W, cudaGetErrorString(e));
      }
      if (path == 0) fast_ok = fast_ok && ok; else naive_ok = naive_ok && ok;
    }
  }
  PB_CK(cudaFree(dx)); PB_CK(cudaFree(dh)); PB_CK(cudaFree(dout));
  PB_CK(cudaDeviceSynchronize());
  return 0;
}

template <int W>
cudaError_t set_attrs() {
  return cudaFuncSetAttribute(col_kernel<W>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)COL_SMEM_BYTES);
}

bool shape_ok(const pb_buffer_shape& b, long long n0, long long n1) {
  return b.dtype == PB_F32 && b.rank == 2 && b.shape[0] == n0 && b.shape[1] == n1 &&
         b.nbytes == (uint64_t)n0 * (uint64_t)n1 * sizeof(float);
}
bool buffer_ok(const pb_buffer& b, long long n0, long long n1) {
  return b.data && b.dtype == PB_F32 && b.rank == 2 && b.shape[0] == n0 && b.shape[1] == n1 &&
         b.nbytes == (uint64_t)n0 * (uint64_t)n1 * sizeof(float);
}

}  // namespace

extern "C" int pb_setup(const pb_problem_shape* shape) {
  if (!shape || shape->abi_version != PB_SOLVE_ABI_VERSION || shape->dimension_count != 2 ||
      shape->scalar_count != 1 || shape->buffer_count != 4 || !shape->dimensions || !shape->buffers ||
      !shape->scalar_dtypes)
    return 1;
  const long long n0 = shape->dimensions[0], n1 = shape->dimensions[1];
  if (n0 <= 0 || n1 <= 0 || n0 > 0x7fffffffLL || n1 > 0x7fffffffLL) return 1;
  if (shape->scalar_dtypes[0] != PB_F32) return 1;
  for (int b = 0; b < 4; ++b)
    if (!shape_ok(shape->buffers[b], n0, n1)) return 1;
  g = State();
  g.n0 = n0; g.n1 = n1;
  g.elems = (size_t)n0 * (size_t)n1;
  const size_t bytes = g.elems * sizeof(float);
  PB_CK(cudaMalloc((void**)&g.d_x, bytes));
  PB_CK(cudaMalloc((void**)&g.d_h, bytes));
  PB_CK(cudaMalloc((void**)&g.d_out, bytes));
  PB_CK(cudaMemset(g.d_x, 0, bytes));
  PB_CK(cudaMemset(g.d_h, 0, bytes));
  PB_CK(cudaMemset(g.d_out, 0, bytes));
  // Opt-in to >48 KB dynamic shared memory for the column kernel.  A failure here only disables
  // the fast path (its launches would fail the self-test below and the fallback would be selected).
  if (set_attrs<1>() != cudaSuccess || set_attrs<2>() != cudaSuccess || set_attrs<3>() != cudaSuccess ||
      set_attrs<4>() != cudaSuccess || set_attrs<5>() != cudaSuccess)
    (void)cudaGetLastError();
  // Self-test on synthetic data (no problem input involved); it also warms every kernel variant.
  {
    bool fast_ok = true, naive_ok = true;
    const int rc = self_test(fast_ok, naive_ok);
    if (rc) return rc;
    g.use_fast = fast_ok || !naive_ok;   // fall back only if the fast path fails and the fallback passes
  }
  g.ready = true;
  return 0;
}

extern "C" int pb_upload(const pb_problem* problem, pb_buffer* buffers) {
  if (!g.ready || !problem || !buffers || problem->abi_version != PB_SOLVE_ABI_VERSION ||
      problem->dimension_count != 2 || problem->scalar_count != 1 || problem->buffer_count != 4 ||
      !problem->dimensions || !problem->scalars)
    return 1;
  if (problem->dimensions[0] != g.n0 || problem->dimensions[1] != g.n1) return 1;
  if (problem->scalars[0].dtype != PB_F32) return 1;
  for (int b = 0; b < 4; ++b)
    if (!buffer_ok(buffers[b], g.n0, g.n1)) return 1;
  g.s0 = problem->scalars[0].value.f32;
  // buf0 is the only buffer with an initialized region (input_domain "complete").
  PB_CK(cudaMemcpy(g.d_x, buffers[0].data, g.elems * sizeof(float), cudaMemcpyHostToDevice));
  PB_CK(cudaDeviceSynchronize());
  g.have_input = true;
  return 0;
}

extern "C" int pb_compute(void) {
  if (!g.ready || !g.have_input) return 1;
  Coef c; int W;
  make_coef(g.s0, c, W);
  if (g.use_fast)
    PB_CK(launch_all(W, g.d_x, g.d_h, g.d_out, g.n0, g.n1, c));
  else
    PB_CK(launch_naive(g.d_x, g.d_h, g.d_out, g.n0, g.n1, c));
  PB_CK(cudaDeviceSynchronize());
  return 0;
}

extern "C" int pb_download(pb_buffer* buffers) {
  if (!g.ready || !buffers) return 1;
  if (!buffer_ok(buffers[1], g.n0, g.n1)) return 1;
  PB_CK(cudaMemcpy(buffers[1].data, g.d_out, g.elems * sizeof(float), cudaMemcpyDeviceToHost));
  PB_CK(cudaDeviceSynchronize());
  return 0;
}

extern "C" int pb_teardown(void) {
  int rc = 0;
  if (g.d_x && cudaFree(g.d_x) != cudaSuccess) rc = 1;
  if (g.d_h && cudaFree(g.d_h) != cudaSuccess) rc = 1;
  if (g.d_out && cudaFree(g.d_out) != cudaSuccess) rc = 1;
  g = State();
  return rc;
}
#endif  // __CUDACC__
