// GPU solve of A x = b for the ludcmp case (buf0 = A, buf1 = b, buf2 = x).
//
// The CPU baseline factors A = LU without pivoting and solves by substitution.
// Only x (buf2) is a designated output, and it is accepted by how well it
// satisfies A x = b.  A is strictly diagonally dominant with a unit diagonal,
// so this candidate solves A x = b by preconditioned iterative refinement:
//
//   M  = diagonal + "big" off-diagonal entries (|a_ij| > 1/7; strict row
//        dominance allows at most six of them per row), extracted from A in
//        the first pass into a small ELL structure;
//   r  = b - A x, computed accurately: products of big entries are summed
//        exactly as float-float pairs, the small entries in float;
//   x += P r, where P ~ M^-1 is applied by Jacobi sweeps on the sparse M.
//
// Each outer iteration is one streaming pass over A plus a short sparse solve;
// x0 = b.  The loop stops once the predicted remaining error is below the
// float resolution of max|x| (2^-26 .. 2^-24 of it, by size).  Execution paths:
//
//   n <= TINY_MAX_N: no iteration; one small kernel replays the baseline's LU
//     and substitutions with the same operation order and rounding, so tiny
//     systems (where any method sits at the rounding floor) get exactly the
//     baseline's result.
//   n <= PERSIST_MAX_N: one persistent cooperative kernel (one block per SM)
//     in which every block holds x in shared memory and applies the
//     preconditioner redundantly; blocks exchange only residuals, as tagged
//     64-bit words, and keep their rows of A in shared memory when they fit.
//   larger n (or no cooperative launch): a CUDA graph of separate kernels
//     (extraction pass, dense residual passes, single-block sparse solve)
//     chained with programmatic dependent launch.
//
// If the iteration does not converge (or the structure is unusable) a direct
// LU (cuSOLVER, no pivoting, as in the baseline) with refinement is used.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "interface.h"

namespace {

constexpr int KMAX = 8;               // ELL capacity per row (off-diagonal big entries)
constexpr int DENSE_THREADS = 256;    // threads per block, dense passes (warp per row)
constexpr int DENSE_WARPS = DENSE_THREADS / 32;
constexpr int SPARSE_THREADS = 1024;  // single block for the sparse preconditioner
constexpr int SPARSE_RMAX = 8;        // rows per thread kept in registers
constexpr int MAX_ITERS = 48;         // outer steps before giving up on iteration
constexpr int MORE_ITERS = 2;         // outer iterations per continuation launch
constexpr int GUARD = 4;              // zero floats in front of b and x copies

__host__ __device__ __forceinline__ float big_threshold() { return 1.0f / 7.0f; }

struct State {
  int W;           // ELL width in use (max big off-diagonals in a row)
  int bad;         // structure unusable (overflow, weak diagonal)
  int iter;        // completed outer steps
  int done;        // 0 running, 1 converged, 2 unusable, 3 no convergence
  float zeta_prev; // max |correction| of the previous step
  int pad[3];
};

struct Ctx {
  int n = 0;
  size_t nn = 0;
  float* A = nullptr;     // buf0, n*n floats + zero tail
  float* bpad = nullptr;  // buf1 at bpad + GUARD, zero guards
  float* b = nullptr;
  float* x = nullptr;     // buf2 (output)
  float* y = nullptr;     // buf3 (scratch, uploaded, unused)
  float* xpad = nullptr;  // current iterate at xpad + GUARD, zero guards
  float* r = nullptr;
  float* invd = nullptr;
  int* ell_col = nullptr;
  float* ell_g = nullptr;   // off-diagonal big entries scaled by 1/diag
  float* zbuf = nullptr;    // z buffers in global memory for very large n
  State* st = nullptr;
  int* h_status = nullptr;  // pinned, mapped
  int* d_status = nullptr;
  cudaStream_t stream = nullptr;
  cudaGraphExec_t g_main = nullptr, g_more = nullptr;
  int main_iters = 0;
  int dense_blocks = 0;
  size_t sparse_smem = 0;
  bool sparse_global = false;
  int sparse_r = 0;         // rows per thread, 0 = generic path only
  int nsweep = 0;           // Jacobi sweeps on M per outer iteration
  // persistent cooperative path
  bool persist = false;
  const void* pfn = nullptr;
  int pgrid = 0;
  int npad = 0;
  size_t psmem = 0;
  unsigned long long* rtag = nullptr;  // two tagged residual buffers
  int* part = nullptr;      // per-block partial flags (persistent path)
  int* gpart = nullptr;     // per-block partial flags (graph path)
  int* d_iters = nullptr;
  unsigned* d_epoch = nullptr;  // launch counter, part of the residual tags
  cudaGraphExec_t g_persist = nullptr;
  cudaGraphExec_t g_tiny = nullptr;   // n <= TINY_MAX_N
  int tol_exp = -26;
  int cache_rows = 0;
  int nqmax = 0;
  int acache_off = 0;
  // fallback
  cublasHandle_t blas = nullptr;
  cusolverDnHandle_t solver = nullptr;
  float* F = nullptr;
  float* work = nullptr;
  int lwork = 0;
  int* d_info = nullptr;
  bool ready = false;
};
Ctx g;

// ---------------------------------------------------------------------------
// float-float helpers
__device__ __forceinline__ void two_sum(float a, float b, float& s, float& e) {
  s = a + b;
  const float bb = s - a;
  e = (a - (s - bb)) + (b - bb);
}
// (hi, lo) += a * x exactly (up to second-order rounding of lo)
__device__ __forceinline__ void ff_add_prod(float& hi, float& lo, float a, float x) {
  const float p = a * x;
  const float pe = fmaf(a, x, -p);
  float s, e;
  two_sum(hi, p, s, e);
  hi = s;
  lo += e + pe;
}
__device__ __forceinline__ void ff_warp_sum(float& hi, float& lo) {
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    const float oh = __shfl_xor_sync(0xffffffffu, hi, o);
    const float ol = __shfl_xor_sync(0xffffffffu, lo, o);
    float s, e;
    two_sum(hi, oh, s, e);
    hi = s;
    lo += ol + e;
  }
}
__device__ __forceinline__ float warp_sum_f(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
// Programmatic dependent launch: let the next kernel in the stream start its
// prologue early / wait until the previous kernel's results are visible.
__device__ __forceinline__ void pdl_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// b - (hi + lo) - light, evaluated in float-float
__device__ __forceinline__ float residual_combine(float bi, float hi, float lo, float light) {
  float s, e, s2, e2;
  two_sum(bi, -hi, s, e);
  two_sum(s, -light, s2, e2);
  return s2 + ((e + e2) - lo);
}

// x values for the float4 k of a row whose first element sits M floats past
// an aligned boundary: xp is the guarded vector (value of column c at
// xp[GUARD + c]), so float4 k needs xp[4 + 4k - M + t], t = 0..3.
template <int M>
__device__ __forceinline__ float4 load_x4(const float4* __restrict__ xp4, int k) {
  if (M == 0) return __ldg(xp4 + k + 1);
  const float4 lo = __ldg(xp4 + k), hi = __ldg(xp4 + k + 1);
  if (M == 1) return make_float4(lo.w, hi.x, hi.y, hi.z);
  if (M == 2) return make_float4(lo.z, lo.w, hi.x, hi.y);
  return make_float4(lo.y, lo.z, lo.w, hi.x);
}

// Per-warp scratch used while extracting a row's big entries.
struct RowScratch {
  int cnt;
  int col[KMAX];
  float val[KMAX];
};

// Big elements of one float4: exact products into (hi, lo); when extracting,
// also record them (valid off-diagonal columns) and the diagonal.
template <int M, bool EXTRACT>
__device__ __forceinline__ void big_elements(const float4& a, const float4& xv, int k, int row,
                                             int n, float tau, float& hi, float& lo, float& dg,
                                             RowScratch* rs) {
  const float av[4] = {a.x, a.y, a.z, a.w};
  const float xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    if (fabsf(av[t]) > tau) {
      ff_add_prod(hi, lo, av[t], xw[t]);
      if (EXTRACT) {
        const int c = 4 * k - M + t;
        if (c == row) {
          dg = av[t];
        } else if (c >= 0 && c < n) {
          const int pos = atomicAdd(&rs->cnt, 1);
          if (pos < KMAX) { rs->col[pos] = c; rs->val[pos] = av[t]; }
        }
      }
    }
  }
}

// Accumulates one float4 of a row: small entries into s, big ones exactly.
template <int M, bool EXTRACT>
__device__ __forceinline__ void accum4(const float4& a, const float4& xv, int k, int row, int n,
                                       float tau, float4& s, float& hi, float& lo, float& dg,
                                       RowScratch* rs) {
  const bool bx = fabsf(a.x) > tau, by = fabsf(a.y) > tau;
  const bool bz = fabsf(a.z) > tau, bw = fabsf(a.w) > tau;
  s.x = fmaf(bx ? 0.f : a.x, xv.x, s.x);
  s.y = fmaf(by ? 0.f : a.y, xv.y, s.y);
  s.z = fmaf(bz ? 0.f : a.z, xv.z, s.z);
  s.w = fmaf(bw ? 0.f : a.w, xv.w, s.w);
  if (bx || by || bz || bw) big_elements<M, EXTRACT>(a, xv, k, row, n, tau, hi, lo, dg, rs);
}

// Streams one row (lane-strided float4): light = sum of small entries * x
// (float), (hi, lo) = sum of big entries * x (float-float).  Entries outside
// the row meet zero x.
template <int M, bool EXTRACT>
__device__ __forceinline__ void stream_row(const float4* __restrict__ a4, int nq,
                                           const float4* __restrict__ xp4, int lane, int row,
                                           int n, float tau, float& light, float& hi, float& lo,
                                           float& dg, RowScratch* rs) {
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  int k = lane;
  for (; k + 96 < nq; k += 128) {
    float4 a[4], xv[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) a[u] = __ldg(a4 + k + 32 * u);
#pragma unroll
    for (int u = 0; u < 4; ++u) xv[u] = load_x4<M>(xp4, k + 32 * u);
#pragma unroll
    for (int u = 0; u < 4; ++u)
      accum4<M, EXTRACT>(a[u], xv[u], k + 32 * u, row, n, tau, s, hi, lo, dg, rs);
  }
  for (; k < nq; k += 32)
    accum4<M, EXTRACT>(__ldg(a4 + k), load_x4<M>(xp4, k), k, row, n, tau, s, hi, lo, dg, rs);
  light = (s.x + s.y) + (s.z + s.w);
}

struct RowGeom {
  const float4* a4;
  int nq;
  int m;
};
__device__ __forceinline__ RowGeom row_geom(const float* __restrict__ A, int n, int row) {
  const size_t s = (size_t)row * (size_t)n;
  const size_t q0 = s >> 2;
  RowGeom g;
  g.a4 = reinterpret_cast<const float4*>(A) + q0;
  g.m = (int)(s & 3);
  g.nq = (int)(((s + (size_t)n - 1) >> 2) - q0 + 1);
  return g;
}

template <bool EXTRACT>
__device__ __forceinline__ void process_row(const RowGeom& rg, int n, int row,
                                            const float* __restrict__ xp, int lane, float tau,
                                            float& light, float& hi, float& lo, float& dg,
                                            RowScratch* rs) {
  const float4* xp4 = reinterpret_cast<const float4*>(xp);
  switch (rg.m) {
    case 0: stream_row<0, EXTRACT>(rg.a4, rg.nq, xp4, lane, row, n, tau, light, hi, lo, dg, rs); break;
    case 1: stream_row<1, EXTRACT>(rg.a4, rg.nq, xp4, lane, row, n, tau, light, hi, lo, dg, rs); break;
    case 2: stream_row<2, EXTRACT>(rg.a4, rg.nq, xp4, lane, row, n, tau, light, hi, lo, dg, rs); break;
    default: stream_row<3, EXTRACT>(rg.a4, rg.nq, xp4, lane, row, n, tau, light, hi, lo, dg, rs); break;
  }
}

// ---------------------------------------------------------------------------
// Pass 0: residual of x0 = b, plus the big-entry structure of every row
// (columns sorted), 1/diag and the ELL width.
__global__ void __launch_bounds__(DENSE_THREADS)
k_extract(const float* __restrict__ A, const float* __restrict__ bpad, int n,
          float* __restrict__ r, float* __restrict__ invd, int* __restrict__ ell_col,
          float* __restrict__ ell_g, float* __restrict__ xpad, int* __restrict__ part) {
  const float tau = big_threshold();
  __shared__ RowScratch s_rs[DENSE_WARPS];
  __shared__ int s_wmax, s_bad;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  pdl_launch_dependents();
  if (threadIdx.x == 0) { s_wmax = 0; s_bad = 0; }
  if (blockIdx.x == 0 && threadIdx.x < GUARD) xpad[threadIdx.x] = 0.f;
  if (blockIdx.x == 0 && threadIdx.x < 12) xpad[n + GUARD + threadIdx.x] = 0.f;
  __syncthreads();
  int wmax = 0, bad = 0;
  RowScratch* rs = &s_rs[w];
  for (int row = blockIdx.x * DENSE_WARPS + w; row < n; row += gridDim.x * DENSE_WARPS) {
    if (lane == 0) rs->cnt = 0;
    __syncwarp();
    const float bi = lane == 0 ? __ldg(bpad + GUARD + row) : 0.f;
    float light, hi = 0.f, lo = 0.f, dg = 0.f;
    process_row<true>(row_geom(A, n, row), n, row, bpad, lane, tau, light, hi, lo, dg, rs);
    __syncwarp();
    light = warp_sum_f(light);
    ff_warp_sum(hi, lo);
    dg = warp_sum_f(dg);  // exactly one lane holds the diagonal
    const int cnt = rs->cnt;
    const int cu = cnt < KMAX ? cnt : KMAX;
    const float id = 1.0f / dg;
    if (lane < KMAX) {
      if (lane < cu) {
        const int mc = rs->col[lane];
        const float mv = rs->val[lane];
        int rank = 0;
        for (int j = 0; j < cu; ++j) rank += (rs->col[j] < mc);
        ell_col[(size_t)rank * n + row] = mc;
        ell_g[(size_t)rank * n + row] = mv * id;
      } else {
        ell_col[(size_t)lane * n + row] = 0;
        ell_g[(size_t)lane * n + row] = 0.f;
      }
    }
    if (lane == 0) {
      if (!(fabsf(dg) > tau) || cnt > KMAX) bad = 1;
      wmax = max(wmax, cu);
      invd[row] = id;
      r[row] = residual_combine(bi, hi, lo, light);
    }
    __syncwarp();
  }
  if (lane == 0) {
    if (wmax) atomicMax(&s_wmax, wmax);
    if (bad) atomicOr(&s_bad, 1);
  }
  __syncthreads();
  if (threadIdx.x == 0) part[blockIdx.x] = s_wmax | (s_bad << 16);
}

// Residual pass: r = b - A x with x = the guarded iterate.
__global__ void __launch_bounds__(DENSE_THREADS)
k_dense(const float* __restrict__ A, const float* __restrict__ bpad, int n,
        const float* __restrict__ xpad, float* __restrict__ r, const State* __restrict__ st) {
  const float tau = big_threshold();
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  pdl_launch_dependents();
  pdl_wait();
  if (*((volatile const int*)&st->done)) return;
  for (int row = blockIdx.x * DENSE_WARPS + w; row < n; row += gridDim.x * DENSE_WARPS) {
    const float bi = lane == 0 ? __ldg(bpad + GUARD + row) : 0.f;
    float light, hi = 0.f, lo = 0.f, dg = 0.f;
    process_row<false>(row_geom(A, n, row), n, row, xpad, lane, tau, light, hi, lo, dg, nullptr);
    light = warp_sum_f(light);
    ff_warp_sum(hi, lo);
    if (lane == 0) r[row] = residual_combine(bi, hi, lo, light);
  }
}

// Preconditioner + update (one block): z = P r with P ~ M^-1 by nsweep Jacobi
// sweeps on M, x += z, convergence test.  R = rows per thread held in
// registers (0: generic path reading the structure from memory each sweep).
template <int R>
__global__ void __launch_bounds__(SPARSE_THREADS, 1)
k_sparse(int n, int nsweep, const float* __restrict__ r, const float* __restrict__ invd,
         const int* __restrict__ ell_col, const float* __restrict__ ell_g,
         const float* __restrict__ bpad, float* __restrict__ xpad, float* __restrict__ xout,
         float* zglob, State* st, int* status, int max_iters, int tol_exp,
         const int* __restrict__ part, int nparts) {
  extern __shared__ float smem[];
  __shared__ float s_red[2][SPARSE_THREADS / 32];
  __shared__ int s_nf;
  __shared__ int s_part[SPARSE_THREADS / 32];
  constexpr int RR = R > 0 ? R : 1;
  const int tid = threadIdx.x;
  pdl_launch_dependents();
  pdl_wait();
  int W, iter, bad;
  float zeta_prev;
  if (nparts > 0) {
    // First step of a solve: combine the extraction pass's per-block flags
    // and start a fresh state.
    int v = 0;
    for (int i = tid; i < nparts; i += SPARSE_THREADS) {
      const int p = part[i];
      v = max(v & 0xffff, p & 0xffff) | ((v | p) & 0xffff0000);
    }
    v = (int)__reduce_max_sync(0xffffffffu, (unsigned)(v & 0xffff)) |
        (int)(__reduce_or_sync(0xffffffffu, (unsigned)v) & 0xffff0000u);
    if ((tid & 31) == 0) s_part[tid >> 5] = v;
    __syncthreads();
    W = 0;
    bad = 0;
    for (int i = 0; i < SPARSE_THREADS / 32; ++i) {
      W = max(W, s_part[i] & 0xffff);
      bad |= s_part[i] >> 16;
    }
    iter = 0;
    zeta_prev = 1.f;
  } else {
    if (*((volatile int*)&st->done)) return;
    W = st->W;
    bad = 0;
    iter = st->iter;
    zeta_prev = st->zeta_prev;
  }
  if (bad) {
    if (tid == 0) {
      st->done = 2;
      *((volatile int*)status) = 2;
    }
    return;
  }
  if (tid == 0) s_nf = 0;
  const float* xold = (iter == 0 ? bpad : xpad) + GUARD;
  float* zp = zglob ? zglob : smem;
  float* zn = zp + n;
  float zmax = 0.f, xmax = 0.f;
  int nf = 0;
  if (R > 0 && W <= 2) {
    float sv[RR], g0[RR], g1[RR], xo[RR];
    int c0[RR], c1[RR];
#pragma unroll
    for (int j = 0; j < RR; ++j) {
      const int i = tid + j * SPARSE_THREADS;
      const bool ok = j < R && i < n;
      const int ii = ok ? i : 0;
      c0[j] = ell_col[ii];
      c1[j] = ell_col[n + ii];
      g0[j] = ok ? ell_g[ii] : 0.f;
      g1[j] = ok ? ell_g[n + ii] : 0.f;
      sv[j] = ok ? invd[ii] * r[ii] : 0.f;
      xo[j] = xold[ii];
    }
#pragma unroll
    for (int j = 0; j < RR; ++j) {
      const int i = tid + j * SPARSE_THREADS;
      if (j < R && i < n) zp[i] = sv[j];
    }
    __syncthreads();
    for (int sweep = 0; sweep < nsweep; ++sweep) {
#pragma unroll
      for (int j = 0; j < RR; ++j) {
        const int i = tid + j * SPARSE_THREADS;
        if (j < R && i < n) zn[i] = fmaf(-g1[j], zp[c1[j]], fmaf(-g0[j], zp[c0[j]], sv[j]));
      }
      __syncthreads();
      float* t = zp; zp = zn; zn = t;
    }
#pragma unroll
    for (int j = 0; j < RR; ++j) {
      const int i = tid + j * SPARSE_THREADS;
      if (j < R && i < n) {
        const float z = zp[i];
        const float xn = xo[j] + z;
        xpad[GUARD + i] = xn;
        xout[i] = xn;
        if (!(fabsf(xn) <= 3.0e38f) || !(fabsf(z) <= 3.0e38f)) nf = 1;
        zmax = fmaxf(zmax, fabsf(z));
        xmax = fmaxf(xmax, fabsf(xn));
      }
    }
  } else {
    for (int i = tid; i < n; i += SPARSE_THREADS) zp[i] = invd[i] * r[i];
    __syncthreads();
    for (int sweep = 0; sweep < nsweep; ++sweep) {
      for (int i = tid; i < n; i += SPARSE_THREADS) {
        float acc = invd[i] * r[i];
        for (int q = 0; q < W; ++q)
          acc = fmaf(-ell_g[(size_t)q * n + i], zp[ell_col[(size_t)q * n + i]], acc);
        zn[i] = acc;
      }
      __syncthreads();
      float* t = zp; zp = zn; zn = t;
    }
    for (int i = tid; i < n; i += SPARSE_THREADS) {
      const float z = zp[i];
      const float xn = xold[i] + z;
      xpad[GUARD + i] = xn;
      xout[i] = xn;
      if (!(fabsf(xn) <= 3.0e38f) || !(fabsf(z) <= 3.0e38f)) nf = 1;
      zmax = fmaxf(zmax, fabsf(z));
      xmax = fmaxf(xmax, fabsf(xn));
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    zmax = fmaxf(zmax, __shfl_xor_sync(0xffffffffu, zmax, o));
    xmax = fmaxf(xmax, __shfl_xor_sync(0xffffffffu, xmax, o));
  }
  if (nf) s_nf = 1;
  if ((tid & 31) == 0) { s_red[0][tid >> 5] = zmax; s_red[1][tid >> 5] = xmax; }
  __syncthreads();
  if (tid == 0) {
    float zeta = 0.f, xi = 0.f;
    for (int i = 0; i < SPARSE_THREADS / 32; ++i) {
      zeta = fmaxf(zeta, s_red[0][i]);
      xi = fmaxf(xi, s_red[1][i]);
    }
    int done = 0;
    if (s_nf) {
      done = 2;
    } else if (zeta == 0.f) {
      done = 1;
    } else if (iter >= 1) {
      // rho: observed contraction; stop when the predicted remaining error is
      // below 2^tol_exp * max|x|, or when the corrections have stalled at the
      // rounding level.
      const float rho = zeta / zeta_prev;
      if (rho <= 0.5f && zeta * fmaxf(rho, 1e-3f) <= ldexpf(xi, tol_exp)) done = 1;
      else if (rho > 0.5f && zeta <= ldexpf(xi, -22)) done = 1;
    }
    if (!done && iter + 1 >= max_iters) done = 3;
    st->W = W;
    st->bad = 0;
    st->zeta_prev = zeta;
    st->iter = iter + 1;
    st->done = done;
    if (done) *((volatile int*)status) = done;
  }
}

// ---------------------------------------------------------------------------
// Persistent cooperative variant (one block per SM, at most one per row).
// Every block keeps its own copy of x in shared memory and applies the
// preconditioner redundantly, so the only communication per outer iteration
// is the residual: each block writes the residual of its rows as 64-bit
// words {value, tag}, and readers poll until every word carries the tag of
// the current launch and iteration.  Residuals are double-buffered; a block
// writes iteration k+1 only after reading all of iteration k, which every
// other block wrote only after reading all of iteration k-1, so a buffer is
// never overwritten while it can still be read.  Rows are assigned round-
// robin (block B owns rows B, B + G, ...), and the first rows of each block
// are kept in shared memory after the first pass.
constexpr int PT = 1024;
constexpr int PWARPS = PT / 32;
// Above this size the separate-kernel path streams A faster than the
// persistent kernel (measured crossover on B300).
constexpr int PERSIST_MAX_N = 5500;

__device__ __forceinline__ unsigned long long ld_relaxed_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long pack_r(float r, unsigned tag) {
  return ((unsigned long long)tag << 32) | (unsigned long long)__float_as_uint(r);
}
// Residual word i of the given buffer once it carries tag.
__device__ __forceinline__ float poll_r(const unsigned long long* p, unsigned tag) {
  unsigned long long v = ld_relaxed_u64(p);
  while ((unsigned)(v >> 32) != tag) {
    __nanosleep(32);
    v = ld_relaxed_u64(p);
  }
  return __uint_as_float((unsigned)v);
}

// Big elements of one float4 (runtime misalignment m): exact products into
// (hi, lo); when extracting, also record them and the diagonal.
template <bool EXTRACT>
__device__ __forceinline__ void big_elements_m(const float4& a, const float4& xv, int k, int m,
                                               int row, int n, float tau, float& hi, float& lo,
                                               float& dg, RowScratch* rs) {
  const float av[4] = {a.x, a.y, a.z, a.w};
  const float xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    if (fabsf(av[t]) > tau) {
      ff_add_prod(hi, lo, av[t], xw[t]);
      if (EXTRACT) {
        const int c = 4 * k - m + t;
        if (c == row) {
          dg = av[t];
        } else if (c >= 0 && c < n) {
          const int pos = atomicAdd(&rs->cnt, 1);
          if (pos < KMAX) { rs->col[pos] = c; rs->val[pos] = av[t]; }
        }
      }
    }
  }
}

template <bool EXTRACT>
__device__ __forceinline__ void accum4_m(const float4& a, const float4& xv, int k, int m, int row,
                                         int n, float tau, float4& s, float& hi, float& lo,
                                         float& dg, RowScratch* rs) {
  const bool bx = fabsf(a.x) > tau, by = fabsf(a.y) > tau;
  const bool bz = fabsf(a.z) > tau, bw = fabsf(a.w) > tau;
  s.x = fmaf(bx ? 0.f : a.x, xv.x, s.x);
  s.y = fmaf(by ? 0.f : a.y, xv.y, s.y);
  s.z = fmaf(bz ? 0.f : a.z, xv.z, s.z);
  s.w = fmaf(bw ? 0.f : a.w, xv.w, s.w);
  if (bx || by || bz || bw) big_elements_m<EXTRACT>(a, xv, k, m, row, n, tau, hi, lo, dg, rs);
}

// Streams one row of the persistent kernel.  x4 is the shifted copy of x in
// shared memory that matches this row's alignment (x4[k] holds the x values
// of the row's float4 k).  ASRC: 0 = A from global memory, 1 = from the
// shared-memory row cache a4s, 2 = from global memory, also filling a4s.
template <bool EXTRACT, int ASRC>
__device__ __forceinline__ void stream_row_p(const float4* __restrict__ a4, float4* a4s, int nq,
                                             const float4* x4, int m, int lane, int row, int n,
                                             float tau, float& light, float& hi, float& lo,
                                             float& dg, RowScratch* rs) {
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  int k = lane;
  for (; k + 96 < nq; k += 128) {
    float4 a[4], xv[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) a[u] = ASRC == 1 ? a4s[k + 32 * u] : __ldg(a4 + k + 32 * u);
#pragma unroll
    for (int u = 0; u < 4; ++u) xv[u] = x4[k + 32 * u];
    if (ASRC == 2) {
#pragma unroll
      for (int u = 0; u < 4; ++u) a4s[k + 32 * u] = a[u];
    }
#pragma unroll
    for (int u = 0; u < 4; ++u)
      accum4_m<EXTRACT>(a[u], xv[u], k + 32 * u, m, row, n, tau, s, hi, lo, dg, rs);
  }
  for (; k < nq; k += 32) {
    const float4 a = ASRC == 1 ? a4s[k] : __ldg(a4 + k);
    if (ASRC == 2) a4s[k] = a;
    accum4_m<EXTRACT>(a, x4[k], k, m, row, n, tau, s, hi, lo, dg, rs);
  }
  light = (s.x + s.y) + (s.z + s.w);
}

// Shared memory: nx shifted copies of x (npad floats each; copy m holds x_i
// at i + m, zeros elsewhere) | za [n] | zb [n] | sv [n if R == 0] | A cache.
template <int R>
__global__ void __launch_bounds__(PT, 1)
k_persist(const float* __restrict__ A, const float* __restrict__ bpad, int n, int npad,
          unsigned long long* __restrict__ rtag, float* __restrict__ invd,
          int* __restrict__ ell_col, float* __restrict__ ell_g, int* __restrict__ part,
          float* __restrict__ xout, int* status, int* iters_out, int nsweep, int max_iters,
          unsigned* epoch_ctr, int cache_rows, int nqmax, int acache_off, int tol_exp) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  extern __shared__ float4 sm4[];
  const int nx = (n & 3) ? 4 : 1;
  float* xc = reinterpret_cast<float*>(sm4);
  float* za = xc + (size_t)nx * npad;
  float* zb = za + n;
  float* svs = zb + n;  // R == 0 only
  float4* acache = sm4 + acache_off;
  __shared__ RowScratch s_rs[PWARPS];
  __shared__ float s_red[2][PWARPS];
  __shared__ int s_i[PWARPS];
  constexpr int RR = R > 0 ? R : 1;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int G = gridDim.x, B = blockIdx.x;
  const float tau = big_threshold();
  // Launch epoch (part of every residual tag): read here by all blocks and
  // advanced by block 0 once every block has passed the first grid barrier.
  const unsigned epoch = *((volatile unsigned*)epoch_ctr);
  // x0 = b in every shifted copy (copy m holds x_i at i + m).
  for (int m = 0; m < nx; ++m)
    for (int j = tid; j < npad; j += PT) {
      const int i = j - m;
      xc[(size_t)m * npad + j] = (i >= 0 && i < n) ? __ldg(bpad + GUARD + i) : 0.f;
    }
  __syncthreads();

  // Phase 0: residual of x0 = b and big-entry structure of every row.
  int wmax = 0, bad = 0;
  RowScratch* rs = &s_rs[w];
  for (int t = w;; t += PWARPS) {
    const long long rowl = (long long)B + (long long)t * G;
    if (rowl >= n) break;
    const int row = (int)rowl;
    if (lane == 0) rs->cnt = 0;
    __syncwarp();
    const float bi = lane == 0 ? __ldg(bpad + GUARD + row) : 0.f;
    float light, hi = 0.f, lo = 0.f, dg = 0.f;
    const RowGeom rg = row_geom(A, n, row);
    const float4* x4 = reinterpret_cast<const float4*>(xc + (size_t)(nx == 1 ? 0 : rg.m) * npad);
    if (t < cache_rows)
      stream_row_p<true, 2>(rg.a4, acache + (size_t)t * nqmax, rg.nq, x4, rg.m, lane, row, n, tau,
                            light, hi, lo, dg, rs);
    else
      stream_row_p<true, 0>(rg.a4, nullptr, rg.nq, x4, rg.m, lane, row, n, tau, light, hi, lo, dg,
                            rs);
    __syncwarp();
    light = warp_sum_f(light);
    ff_warp_sum(hi, lo);
    dg = warp_sum_f(dg);
    const int cnt_r = rs->cnt;
    const int cu = cnt_r < KMAX ? cnt_r : KMAX;
    const float id = 1.0f / dg;
    if (lane < KMAX) {
      if (lane < cu) {
        const int mc = rs->col[lane];
        const float mv = rs->val[lane];
        int rank = 0;
        for (int j = 0; j < cu; ++j) rank += (rs->col[j] < mc);
        ell_col[(size_t)rank * n + row] = mc;
        ell_g[(size_t)rank * n + row] = mv * id;
      } else {
        ell_col[(size_t)lane * n + row] = 0;
        ell_g[(size_t)lane * n + row] = 0.f;
      }
    }
    if (lane == 0) {
      if (!(fabsf(dg) > tau) || cnt_r > KMAX) bad = 1;
      wmax = max(wmax, cu);
      invd[row] = id;
      st_relaxed_u64(rtag + row, pack_r(residual_combine(bi, hi, lo, light), epoch * 64u));
    }
    __syncwarp();
  }
  wmax = __reduce_max_sync(0xffffffffu, wmax);
  bad = (int)__reduce_or_sync(0xffffffffu, (unsigned)bad);
  if (lane == 0) s_i[w] = wmax | (bad << 16);
  __syncthreads();
  if (tid == 0) {
    int v = 0;
    for (int i = 0; i < PWARPS; ++i) v = max(v & 0xffff, s_i[i] & 0xffff) | ((v | s_i[i]) & 0xffff0000);
    part[B] = v;
  }
  grid.sync();
  if (B == 0 && tid == 0) *((volatile unsigned*)epoch_ctr) = epoch % 0x3ffffffu + 1u;
  int pv = 0;
  for (int i = tid; i < G; i += PT) {
    const int v = __ldcg(part + i);
    pv = max(pv & 0xffff, v & 0xffff) | ((pv | v) & 0xffff0000);
  }
  pv = (int)__reduce_max_sync(0xffffffffu, (unsigned)(pv & 0xffff)) |
       (int)(__reduce_or_sync(0xffffffffu, (unsigned)pv) & 0xffff0000u);
  __syncthreads();
  if (lane == 0) s_i[w] = pv;
  __syncthreads();
  int W = 0, badall = 0;
  for (int i = 0; i < PWARPS; ++i) { W = max(W, s_i[i] & 0xffff); badall |= s_i[i] >> 16; }
  if (badall) {
    if (B == 0 && tid == 0) {
      *iters_out = 0;
      *((volatile int*)status) = 2;
    }
    return;
  }

  int iter = 0, done = 0;
  float zeta_prev = 1.f;
  for (;;) {
    const unsigned tag = epoch * 64u + (unsigned)iter;
    const unsigned long long* rt = rtag + (size_t)(iter & 1) * n;
    float* zp = za;
    float* zn = zb;
    float zmax = 0.f, xmax = 0.f;
    int nf = 0;
    if (R > 0 && W <= 2) {
      float sv[RR], g0[RR], g1[RR];
      int c0[RR], c1[RR];
#pragma unroll
      for (int j = 0; j < RR; ++j) {
        const int i = tid + j * PT;
        const bool ok = j < R && i < n;
        const int ii = ok ? i : 0;
        c0[j] = __ldcg(ell_col + ii);
        c1[j] = __ldcg(ell_col + n + ii);
        g0[j] = ok ? __ldcg(ell_g + ii) : 0.f;
        g1[j] = ok ? __ldcg(ell_g + n + ii) : 0.f;
        sv[j] = ok ? __ldcg(invd + ii) : 0.f;
      }
      // Residual words of this thread's rows: all loads in flight at once,
      // then wait for any that do not carry the current tag yet.
      unsigned long long rv[RR];
#pragma unroll
      for (int j = 0; j < RR; ++j) {
        const int i = tid + j * PT;
        rv[j] = (j < R && i < n) ? ld_relaxed_u64(rt + i) : ((unsigned long long)tag << 32);
      }
#pragma unroll
      for (int j = 0; j < RR; ++j) {
        const int i = tid + j * PT;
        while ((unsigned)(rv[j] >> 32) != tag) {
          __nanosleep(32);
          rv[j] = ld_relaxed_u64(rt + i);
        }
        if (j < R && i < n) {
          sv[j] *= __uint_as_float((unsigned)rv[j]);
          zp[i] = sv[j];
        }
      }
      __syncthreads();
      for (int sweep = 0; sweep < nsweep; ++sweep) {
#pragma unroll
        for (int j = 0; j < RR; ++j) {
          const int i = tid + j * PT;
          if (j < R && i < n) zn[i] = fmaf(-g1[j], zp[c1[j]], fmaf(-g0[j], zp[c0[j]], sv[j]));
        }
        __syncthreads();
        float* t = zp; zp = zn; zn = t;
      }
    } else {
      for (int i = tid; i < n; i += PT) {
        const float v = __ldcg(invd + i) * poll_r(rt + i, tag);
        svs[i] = v;
        zp[i] = v;
      }
      __syncthreads();
      for (int sweep = 0; sweep < nsweep; ++sweep) {
        for (int i = tid; i < n; i += PT) {
          float acc = svs[i];
          for (int q = 0; q < W; ++q)
            acc = fmaf(-__ldcg(ell_g + (size_t)q * n + i), zp[__ldcg(ell_col + (size_t)q * n + i)],
                       acc);
          zn[i] = acc;
        }
        __syncthreads();
        float* t = zp; zp = zn; zn = t;
      }
    }
    for (int i = tid; i < n; i += PT) {
      const float z = zp[i];
      const float xn = xc[i] + z;
      for (int m = 0; m < nx; ++m) xc[(size_t)m * npad + i + m] = xn;
      if (!(fabsf(xn) <= 3.0e38f) || !(fabsf(z) <= 3.0e38f)) nf = 1;
      zmax = fmaxf(zmax, fabsf(z));
      xmax = fmaxf(xmax, fabsf(xn));
    }
    // Block-wide max of |z|, |x| and the non-finite flag; every warp reduces
    // the per-warp partials itself, so all threads reach the same decision.
    zmax = __int_as_float(__reduce_max_sync(0xffffffffu, __float_as_int(zmax)));
    xmax = __int_as_float(__reduce_max_sync(0xffffffffu, __float_as_int(xmax)));
    nf = (int)__reduce_or_sync(0xffffffffu, (unsigned)nf);
    if (lane == 0) { s_red[0][w] = zmax; s_red[1][w] = xmax; s_i[w] = nf; }
    __syncthreads();
    const float zeta = __int_as_float(__reduce_max_sync(0xffffffffu, __float_as_int(s_red[0][lane])));
    const float xi = __int_as_float(__reduce_max_sync(0xffffffffu, __float_as_int(s_red[1][lane])));
    const int anynf = (int)__reduce_or_sync(0xffffffffu, (unsigned)s_i[lane]);
    int d = 0;
    if (anynf) {
      d = 2;
    } else if (zeta == 0.f) {
      d = 1;
    } else if (iter >= 1) {
      // rho: observed contraction; stop when the predicted remaining error is
      // below 2^tol_exp * max|x|, or when the corrections have stalled at the
      // rounding level.
      const float rho = zeta / zeta_prev;
      if (rho <= 0.5f && zeta * fmaxf(rho, 1e-3f) <= ldexpf(xi, tol_exp)) d = 1;
      else if (rho > 0.5f && zeta <= ldexpf(xi, -22)) d = 1;
    }
    if (!d && iter + 1 >= max_iters) d = 3;
    zeta_prev = zeta;
    done = d;
    ++iter;
    if (done) break;
    // Residual of the new iterate for this block's rows.
    unsigned long long* rn = rtag + (size_t)(iter & 1) * n;
    const unsigned ntag = epoch * 64u + (unsigned)iter;
    for (int t = w;; t += PWARPS) {
      const long long rowl = (long long)B + (long long)t * G;
      if (rowl >= n) break;
      const int row = (int)rowl;
      const float bi = lane == 0 ? __ldg(bpad + GUARD + row) : 0.f;
      float light, hi = 0.f, lo = 0.f, dg = 0.f;
      const RowGeom rg = row_geom(A, n, row);
      const float4* x4 = reinterpret_cast<const float4*>(xc + (size_t)(nx == 1 ? 0 : rg.m) * npad);
      if (t < cache_rows)
        stream_row_p<false, 1>(rg.a4, acache + (size_t)t * nqmax, rg.nq, x4, rg.m, lane, row, n,
                               tau, light, hi, lo, dg, nullptr);
      else
        stream_row_p<false, 0>(rg.a4, nullptr, rg.nq, x4, rg.m, lane, row, n, tau, light, hi, lo,
                               dg, nullptr);
      light = warp_sum_f(light);
      ff_warp_sum(hi, lo);
      if (lane == 0) st_relaxed_u64(rn + row, pack_r(residual_combine(bi, hi, lo, light), ntag));
    }
    // z and x of this iteration must not be overwritten before every warp of
    // the block has finished using them.
    __syncthreads();
  }
  for (int i = B * PT + tid; i < n; i += G * PT) xout[i] = xc[i];
  if (B == 0 && tid == 0) {
    *iters_out = iter;
    *((volatile int*)status) = done;
  }
}

// ---------------------------------------------------------------------------
// Tiny systems (n <= TINY_MAX_N): the baseline's own algorithm, with the
// same operation order and separately rounded IEEE products, differences and
// quotients (the baseline is plain C without contraction), so the result
// matches the baseline's float arithmetic.  Warp i holds row i of L\U, one
// lane per column.  At step s every warp i > s applies row s (complete after
// step s-1) to its row, so the n rows advance together, one block barrier per
// step.
constexpr int TINY_MAX_N = 24;

__global__ void __launch_bounds__(32 * TINY_MAX_N)
k_tiny(const float* __restrict__ A, const float* __restrict__ b, float* __restrict__ x, int n,
       int* status) {
  __shared__ float a[TINY_MAX_N][TINY_MAX_N + 1];
  __shared__ float y[TINY_MAX_N];
  const int lane = threadIdx.x & 31, i = threadIdx.x >> 5;
  // Row i: lane j accumulates w_j = a[i][j] - sum_k l_ik u_kj in increasing
  // k; L entries (j < i) are divided by u_jj once complete.
  float w = lane < n ? A[(size_t)i * n + lane] : 0.f;
  if (i == 0 && lane < n) a[0][lane] = w;
  for (int s = 0; s < n - 1; ++s) {
    if (i == s && s > 0 && lane < n) a[s][lane] = w;
    __syncthreads();
    if (i > s) {
      if (lane == s) w = __fdiv_rn(w, a[s][s]);
      const float lis = __shfl_sync(0xffffffffu, w, s);
      if (lane > s && lane < n) w = __fsub_rn(w, __fmul_rn(lis, a[s][lane]));
    }
  }
  if (i == n - 1 && n > 1 && lane < n) a[n - 1][lane] = w;
  __syncthreads();
  if (i != 0) return;
  // L y = b, sums in increasing j.
  w = lane < n ? b[lane] : 0.f;
  for (int j = 0; j < n; ++j) {
    const float yj = __shfl_sync(0xffffffffu, w, j);
    if (lane > j && lane < n) w = __fsub_rn(w, __fmul_rn(a[lane][j], yj));
  }
  if (lane < n) y[lane] = w;
  __syncwarp();
  // U x = y, sums in increasing j as in the baseline; y[] is overwritten by x.
  if (lane == 0) {
    for (int r = n - 1; r >= 0; --r) {
      float v = y[r];
      int j = r + 1;
      for (; j + 7 < n; j += 8) {
        float p[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) p[u] = __fmul_rn(a[r][j + u], y[j + u]);
#pragma unroll
        for (int u = 0; u < 8; ++u) v = __fsub_rn(v, p[u]);
      }
      for (; j < n; ++j) v = __fsub_rn(v, __fmul_rn(a[r][j], y[j]));
      y[r] = __fdiv_rn(v, a[r][r]);
    }
  }
  __syncwarp();
  if (lane < n) x[lane] = y[lane];
  if (lane == 0) *((volatile int*)status) = 1;
}

// Fallback helpers -----------------------------------------------------------
// r = b - A x (double accumulation; generic, no structure assumptions).
__global__ void __launch_bounds__(DENSE_THREADS)
k_resid_generic(const float* __restrict__ A, const float* __restrict__ b,
                const float* __restrict__ x, int n, float* __restrict__ r) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for (int row = blockIdx.x * DENSE_WARPS + w; row < n; row += gridDim.x * DENSE_WARPS) {
    const float* a = A + (size_t)row * n;
    double acc = 0.0;
    for (int j = lane; j < n; j += 32) acc += (double)a[j] * (double)x[j];
#pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) r[row] = (float)((double)b[row] - acc);
  }
}

// Setup-time warm-up data: A = I.
__global__ void k_identity(float* __restrict__ A, int n) {
  const size_t nn = (size_t)n * (size_t)n;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nn;
       i += (size_t)gridDim.x * blockDim.x)
    A[i] = (i % ((size_t)n + 1) == 0) ? 1.0f : 0.0f;
}

__global__ void k_add(int n, const float* __restrict__ d, float* __restrict__ x) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    x[i] += d[i];
}

// ---------------------------------------------------------------------------
template <typename... Args>
cudaError_t launch_pdl(void (*f)(Args...), int grid, int block, size_t smem, cudaStream_t s,
                       Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, f, args...);
}

cudaError_t launch_sparse(cudaStream_t s, bool first = false) {
  float* zg = g.sparse_global ? g.zbuf : nullptr;
  void (*f)(int, int, const float*, const float*, const int*, const float*, const float*, float*,
            float*, float*, State*, int*, int, int, const int*, int);
  switch (g.sparse_r) {
    case 1: f = k_sparse<1>; break;
    case 2: f = k_sparse<2>; break;
    case 4: f = k_sparse<4>; break;
    case 8: f = k_sparse<8>; break;
    default: f = k_sparse<0>; break;
  }
  return launch_pdl(f, 1, SPARSE_THREADS, g.sparse_smem, s, g.n, g.nsweep, (const float*)g.r,
                    (const float*)g.invd, (const int*)g.ell_col, (const float*)g.ell_g,
                    (const float*)g.bpad, g.xpad, g.x, zg, g.st, g.d_status, (int)MAX_ITERS,
                    g.tol_exp, (const int*)g.gpart, first ? g.dense_blocks : 0);
}

cudaError_t launch_iteration(cudaStream_t s) {
  cudaError_t e = launch_pdl(k_dense, g.dense_blocks, DENSE_THREADS, 0, s, (const float*)g.A,
                             (const float*)g.bpad, g.n, (const float*)g.xpad, g.r,
                             (const State*)g.st);
  if (e) return e;
  return launch_sparse(s);
}

cudaError_t capture(cudaGraphExec_t* out, bool head, int iters) {
  cudaGraph_t graph = nullptr;
  cudaError_t e = cudaStreamBeginCapture(g.stream, cudaStreamCaptureModeThreadLocal);
  if (e) return e;
  cudaError_t le = cudaSuccess;
  if (head) {
    k_extract<<<g.dense_blocks, DENSE_THREADS, 0, g.stream>>>(g.A, g.bpad, g.n, g.r, g.invd,
                                                             g.ell_col, g.ell_g, g.xpad, g.gpart);
    le = cudaGetLastError();
    if (!le) le = launch_sparse(g.stream, true);
  }
  for (int i = 0; i < iters && !le; ++i) le = launch_iteration(g.stream);
  e = cudaStreamEndCapture(g.stream, &graph);
  if (le) { if (graph) cudaGraphDestroy(graph); return le; }
  if (e) return e;
  e = cudaGraphInstantiate(out, graph, 0);
  cudaGraphDestroy(graph);
  return e;
}

// Outer steps expected for generator-like inputs: the contraction per step
// is about 0.2 / sqrt(n); the stopping test needs rho^(k+1) / 2 <= 2^tol_exp.
int expected_iters(int n, int tol_exp) {
  const double rho = std::fmin(0.2 / std::sqrt((double)(n > 1 ? n : 1)), 0.05);
  const double need = std::log(2.0 * std::ldexp(1.0, tol_exp)) / std::log(rho);  // k + 1
  int k = (int)std::ceil(need - 1e-9) - 1;
  if (k < 1) k = 1;
  return k;
}

cudaError_t launch_persist(cudaStream_t s) {
  const float* A = g.A;
  const float* bp = g.bpad;
  int n = g.n, npad = g.npad, nsweep = g.nsweep, maxit = MAX_ITERS;
  void* args[] = {(void*)&A, (void*)&bp, (void*)&n, (void*)&npad, (void*)&g.rtag,
                  (void*)&g.invd, (void*)&g.ell_col, (void*)&g.ell_g, (void*)&g.part,
                  (void*)&g.x, (void*)&g.d_status, (void*)&g.d_iters, (void*)&nsweep,
                  (void*)&maxit, (void*)&g.d_epoch, (void*)&g.cache_rows, (void*)&g.nqmax,
                  (void*)&g.acache_off, (void*)&g.tol_exp};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(g.pgrid);
  cfg.blockDim = dim3(PT);
  cfg.dynamicSmemBytes = g.psmem;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeCooperative;
  attr[0].val.cooperative = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelExC(&cfg, g.pfn, args);
}

// The persistent kernel as a one-node graph (launches faster than a direct
// cooperative launch).
cudaError_t capture_persist() {
  cudaGraph_t graph = nullptr;
  cudaError_t e = cudaStreamBeginCapture(g.stream, cudaStreamCaptureModeThreadLocal);
  if (e) return e;
  const cudaError_t le = launch_persist(g.stream);
  e = cudaStreamEndCapture(g.stream, &graph);
  if (le) { if (graph) cudaGraphDestroy(graph); return le; }
  if (e) return e;
  e = cudaGraphInstantiate(&g.g_persist, graph, 0);
  cudaGraphDestroy(graph);
  return e;
}

// Direct solve: LU without pivoting of the column-major view (= A^T), then
// A x = U^T L^T x = b, followed by refinement steps.
int fallback_solve() {
  const int n = g.n;
  cudaError_t ce;
  if (!g.blas) {
    if (cublasCreate(&g.blas) != CUBLAS_STATUS_SUCCESS) return 1;
    if (cublasSetStream(g.blas, g.stream) != CUBLAS_STATUS_SUCCESS) return 1;
  }
  if (!g.solver) {
    if (cusolverDnCreate(&g.solver) != CUSOLVER_STATUS_SUCCESS) return 1;
    if (cusolverDnSetStream(g.solver, g.stream) != CUSOLVER_STATUS_SUCCESS) return 1;
  }
  if (!g.F) {
    if ((ce = cudaMalloc(&g.F, g.nn * sizeof(float)))) return (int)ce;
  }
  if (!g.work) {
    if (cusolverDnSgetrf_bufferSize(g.solver, n, n, g.F, n, &g.lwork) != CUSOLVER_STATUS_SUCCESS)
      return 1;
    if ((ce = cudaMalloc(&g.work, (size_t)(g.lwork > 0 ? g.lwork : 1) * sizeof(float))))
      return (int)ce;
  }
  if (!g.d_info) {
    if ((ce = cudaMalloc(&g.d_info, sizeof(int)))) return (int)ce;
  }
  if ((ce = cudaMemcpyAsync(g.F, g.A, g.nn * sizeof(float), cudaMemcpyDeviceToDevice, g.stream)))
    return (int)ce;
  if (cusolverDnSgetrf(g.solver, n, n, g.F, n, g.work, nullptr, g.d_info) !=
      CUSOLVER_STATUS_SUCCESS)
    return 1;
  if ((ce = cudaMemcpyAsync(g.x, g.b, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice,
                            g.stream)))
    return (int)ce;
  auto lu_solve = [&](float* v) -> int {
    if (cublasStrsv(g.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, g.F, n,
                    v, 1) != CUBLAS_STATUS_SUCCESS)
      return 1;
    if (cublasStrsv(g.blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT, n, g.F, n, v,
                    1) != CUBLAS_STATUS_SUCCESS)
      return 1;
    return 0;
  };
  if (lu_solve(g.x)) return 1;
  const int rb = (n + DENSE_WARPS - 1) / DENSE_WARPS;
  const int rblocks = rb < 4096 ? rb : 4096;
  const int ab = (n + 255) / 256 < 1024 ? (n + 255) / 256 : 1024;
  for (int it = 0; it < 3; ++it) {
    k_resid_generic<<<rblocks, DENSE_THREADS, 0, g.stream>>>(g.A, g.b, g.x, n, g.r);
    if ((ce = cudaGetLastError())) return (int)ce;
    if (lu_solve(g.r)) return 1;
    k_add<<<ab, 256, 0, g.stream>>>(n, g.r, g.x);
    if ((ce = cudaGetLastError())) return (int)ce;
  }
  if ((ce = cudaStreamSynchronize(g.stream))) return (int)ce;
  return 0;
}

void release_all() {
  if (g.g_main) cudaGraphExecDestroy(g.g_main);
  if (g.g_more) cudaGraphExecDestroy(g.g_more);
  if (g.blas) cublasDestroy(g.blas);
  if (g.solver) cusolverDnDestroy(g.solver);
  float* fbufs[] = {g.A, g.bpad, g.x, g.y, g.xpad, g.r, g.invd, g.ell_g, g.zbuf, g.F, g.work};
  for (float* p : fbufs) if (p) cudaFree(p);
  if (g.rtag) cudaFree(g.rtag);
  if (g.ell_col) cudaFree(g.ell_col);
  if (g.part) cudaFree(g.part);
  if (g.gpart) cudaFree(g.gpart);
  if (g.d_iters) cudaFree(g.d_iters);
  if (g.d_epoch) cudaFree(g.d_epoch);
  if (g.g_persist) cudaGraphExecDestroy(g.g_persist);
  if (g.g_tiny) cudaGraphExecDestroy(g.g_tiny);
  if (g.st) cudaFree(g.st);
  if (g.d_info) cudaFree(g.d_info);
  if (g.h_status) cudaFreeHost(g.h_status);
  if (g.stream) cudaStreamDestroy(g.stream);
  g = Ctx();
}

bool shape_ok(uint32_t dtype, uint32_t rank, const int64_t* shape, uint64_t nbytes, int n,
              int want_rank) {
  if (dtype != PB_F32 || (int)rank != want_rank) return false;
  uint64_t count = 1;
  for (int i = 0; i < want_rank; ++i) {
    if (shape[i] != n) return false;
    count *= (uint64_t)n;
  }
  return nbytes == count * sizeof(float);
}

int setup_impl(int n) {
  g.n = n;
  g.nn = (size_t)n * (size_t)n;
  cudaError_t e;
  const size_t nv = (size_t)n;
  const size_t npad = ((nv + 2 * GUARD + 12 + 3) / 4) * 4;
  if ((e = cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking))) return (int)e;
  if ((e = cudaMalloc(&g.A, (g.nn + 4) * sizeof(float)))) return (int)e;
  if ((e = cudaMemset(g.A + g.nn, 0, 4 * sizeof(float)))) return (int)e;
  if ((e = cudaMalloc(&g.bpad, npad * sizeof(float)))) return (int)e;
  if ((e = cudaMemset(g.bpad, 0, npad * sizeof(float)))) return (int)e;
  g.b = g.bpad + GUARD;
  if ((e = cudaMalloc(&g.x, nv * sizeof(float)))) return (int)e;
  if ((e = cudaMalloc(&g.y, nv * sizeof(float)))) return (int)e;
  if ((e = cudaMalloc(&g.xpad, npad * sizeof(float)))) return (int)e;
  if ((e = cudaMemset(g.xpad, 0, npad * sizeof(float)))) return (int)e;
  if ((e = cudaMalloc(&g.r, nv * sizeof(float)))) return (int)e;
  if ((e = cudaMalloc(&g.invd, nv * sizeof(float)))) return (int)e;
  if ((e = cudaMalloc(&g.ell_col, KMAX * nv * sizeof(int)))) return (int)e;
  if ((e = cudaMalloc(&g.ell_g, KMAX * nv * sizeof(float)))) return (int)e;
  if ((e = cudaMalloc(&g.st, sizeof(State)))) return (int)e;
  if ((e = cudaMemset(g.st, 0, sizeof(State)))) return (int)e;
  if ((e = cudaHostAlloc(&g.h_status, sizeof(int), cudaHostAllocMapped))) return (int)e;
  if ((e = cudaHostGetDevicePointer(&g.d_status, g.h_status, 0))) return (int)e;
  *((volatile int*)g.h_status) = 0;
  int dev = 0, nsm = 0, optin = 0;
  if ((e = cudaGetDevice(&dev))) return (int)e;
  if ((e = cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev))) return (int)e;
  if ((e = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev)))
    return (int)e;
  const long long want = (n + DENSE_WARPS - 1) / DENSE_WARPS;
  const long long cap = (long long)nsm * (2048 / DENSE_THREADS);
  g.dense_blocks = (int)(want < cap ? want : cap);
  if ((e = cudaMalloc(&g.gpart, (size_t)g.dense_blocks * sizeof(int)))) return (int)e;
  const int rows = (n + SPARSE_THREADS - 1) / SPARSE_THREADS;
  g.sparse_r = rows <= 1 ? 1 : rows <= 2 ? 2 : rows <= 4 ? 4 : rows <= SPARSE_RMAX ? 8 : 0;
  const size_t zbytes = 2 * nv * sizeof(float);
  const size_t static_smem = 2 * SPARSE_THREADS / 32 * sizeof(float) + 64;
  if (zbytes + static_smem <= (size_t)optin) {
    g.sparse_smem = zbytes;
    g.sparse_global = false;
  } else {
    g.sparse_smem = 0;
    g.sparse_global = true;
    if ((e = cudaMalloc(&g.zbuf, zbytes))) return (int)e;
  }
  const void* sk[] = {(const void*)k_sparse<0>, (const void*)k_sparse<1>, (const void*)k_sparse<2>,
                      (const void*)k_sparse<4>, (const void*)k_sparse<8>};
  for (const void* f : sk)
    if ((e = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)g.sparse_smem)))
      return (int)e;
  // Jacobi sweeps per outer iteration, chosen so that the inner solve does
  // not add an outer iteration (measured on generator-like inputs).
  g.nsweep = n < 700 ? 4 : n < 1300 ? 6 : n <= PERSIST_MAX_N ? 5 : 6;
  // Target for the remaining error, relative to max|x|: a quarter ulp for
  // tiny systems, up to one ulp for large ones (where the LU baseline's own
  // error is tens of ulps).
  g.tol_exp = n < 64 ? -26 : n < 1000 ? -25 : -24;
  // Persistent cooperative path: x, two z buffers (and r for the generic
  // preconditioner path) per block in shared memory, then as many of the
  // block's rows of A as fit.
  {
    int coop = 0;
    if ((e = cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev))) return (int)e;
    const int pr = (n + PT - 1) / PT;
    const int R = pr <= 1 ? 1 : pr <= 2 ? 2 : pr <= 4 ? 4 : pr <= 6 ? 6 : 0;
    const void* fn = R == 1 ? (const void*)k_persist<1> : R == 2 ? (const void*)k_persist<2>
                   : R == 4 ? (const void*)k_persist<4> : R == 6 ? (const void*)k_persist<6>
                   : (const void*)k_persist<0>;
    cudaFuncAttributes fa;
    if ((e = cudaFuncGetAttributes(&fa, fn))) return (int)e;
    const size_t nx = (n & 3) ? 4 : 1;
    const size_t base_floats = nx * npad + 2 * nv + (R == 0 ? nv : 0);
    const size_t base = ((base_floats + 3) / 4) * 16;
    const size_t limit = (size_t)optin > fa.sharedSizeBytes ? (size_t)optin - fa.sharedSizeBytes : 0;
    const int grid = n < nsm ? n : nsm;
    if (coop && n <= PERSIST_MAX_N && base <= limit) {
      g.npad = (int)npad;
      g.nqmax = (n + 2) / 4 + 1;
      g.acache_off = (int)(base / 16);
      const long long rows_per_block = (n + grid - 1) / grid;
      const long long fit = (long long)((limit - base) / ((size_t)g.nqmax * 16));
      g.cache_rows = (int)(fit < rows_per_block ? fit : rows_per_block);
      g.psmem = base + (size_t)g.cache_rows * g.nqmax * 16;
      if ((e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)g.psmem)))
        return (int)e;
      int per_sm = 0;
      if ((e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, PT, g.psmem)))
        return (int)e;
      if (per_sm >= 1) {
        g.persist = true;
        g.pfn = fn;
        g.pgrid = grid;
        if ((e = cudaMalloc(&g.rtag, 2 * nv * sizeof(unsigned long long)))) return (int)e;
        if ((e = cudaMemset(g.rtag, 0, 2 * nv * sizeof(unsigned long long)))) return (int)e;
        if ((e = cudaMalloc(&g.part, (size_t)grid * sizeof(int)))) return (int)e;
        if ((e = cudaMalloc(&g.d_iters, sizeof(int)))) return (int)e;
        if ((e = cudaMalloc(&g.d_epoch, sizeof(unsigned)))) return (int)e;
        const unsigned one = 1u;
        if ((e = cudaMemcpy(g.d_epoch, &one, sizeof(unsigned), cudaMemcpyHostToDevice)))
          return (int)e;
        if (capture_persist() != cudaSuccess) {
          // Not launchable as a cooperative graph: use the separate-kernel path.
          cudaGetLastError();
          g.persist = false;
        }
      }
    }
  }
  if (n <= TINY_MAX_N) {
    cudaGraph_t graph = nullptr;
    if ((e = cudaStreamBeginCapture(g.stream, cudaStreamCaptureModeThreadLocal))) return (int)e;
    k_tiny<<<1, 32 * n, 0, g.stream>>>(g.A, g.b, g.x, n, g.d_status);
    const cudaError_t le = cudaGetLastError();
    e = cudaStreamEndCapture(g.stream, &graph);
    if (le) { if (graph) cudaGraphDestroy(graph); return (int)le; }
    if (e) return (int)e;
    e = cudaGraphInstantiate(&g.g_tiny, graph, 0);
    cudaGraphDestroy(graph);
    if (e) return (int)e;
  }
  g.main_iters = expected_iters(n, g.tol_exp);
  if ((e = capture(&g.g_main, true, g.main_iters))) return (int)e;
  if ((e = capture(&g.g_more, false, MORE_ITERS))) return (int)e;
  // Library handles for the direct fallback solve.
  if (cublasCreate(&g.blas) != CUBLAS_STATUS_SUCCESS) return 1;
  if (cublasSetStream(g.blas, g.stream) != CUBLAS_STATUS_SUCCESS) return 1;
  if (cusolverDnCreate(&g.solver) != CUSOLVER_STATUS_SUCCESS) return 1;
  if (cusolverDnSetStream(g.solver, g.stream) != CUSOLVER_STATUS_SUCCESS) return 1;
  // Warm-up on placeholder data (A = I, b = 0) so that first-launch costs of
  // the graph and of every kernel are paid here; results are discarded.
  k_identity<<<1024, 256, 0, g.stream>>>(g.A, n);
  if ((e = cudaGetLastError())) return (int)e;
  if ((e = cudaGraphLaunch(g.g_main, g.stream))) return (int)e;
  {
    const State s_run = {0, 0, 1, 0, 1.0f, {0, 0, 0}};
    if ((e = cudaMemcpyAsync(g.st, &s_run, sizeof(State), cudaMemcpyHostToDevice, g.stream)))
      return (int)e;
    if ((e = launch_iteration(g.stream))) return (int)e;
  }
  if ((e = cudaGraphLaunch(g.g_more, g.stream))) return (int)e;
  if (g.persist && (e = cudaGraphLaunch(g.g_persist, g.stream))) return (int)e;
  if (g.g_tiny && (e = cudaGraphLaunch(g.g_tiny, g.stream))) return (int)e;
  if ((e = cudaStreamSynchronize(g.stream))) return (int)e;
  *((volatile int*)g.h_status) = 0;
  return 0;
}

}  // namespace


extern "C" int pb_setup(const pb_problem_shape* shape) {
  if (!shape || shape->abi_version != PB_SOLVE_ABI_VERSION || shape->dimension_count != 1 ||
      shape->buffer_count != 4 || !shape->dimensions || !shape->buffers)
    return 1;
  const int64_t n64 = shape->dimensions[0];
  if (n64 <= 0 || n64 > 0x7fffffff) return 1;
  const int n = (int)n64;
  const int ranks[4] = {2, 1, 1, 1};
  for (int i = 0; i < 4; ++i) {
    const pb_buffer_shape& b = shape->buffers[i];
    if (!shape_ok(b.dtype, b.rank, b.shape, b.nbytes, n, ranks[i])) return 1;
  }
  release_all();
  const int rc = setup_impl(n);
  if (rc) {
    release_all();
    return rc;
  }
  g.ready = true;
  return 0;
}

extern "C" int pb_upload(const pb_problem* problem, pb_buffer* buffers) {
  if (!g.ready || !problem || !buffers || problem->abi_version != PB_SOLVE_ABI_VERSION ||
      problem->dimension_count != 1 || problem->buffer_count != 4 || !problem->dimensions ||
      problem->dimensions[0] != g.n)
    return 1;
  const int ranks[4] = {2, 1, 1, 1};
  for (int i = 0; i < 4; ++i) {
    if (!buffers[i].data ||
        !shape_ok(buffers[i].dtype, buffers[i].rank, buffers[i].shape, buffers[i].nbytes, g.n,
                  ranks[i]))
      return 1;
  }
  float* dst[4] = {g.A, g.b, g.x, g.y};
  for (int i = 0; i < 4; ++i) {
    const cudaError_t e = cudaMemcpyAsync(dst[i], buffers[i].data, buffers[i].nbytes,
                                          cudaMemcpyHostToDevice, g.stream);
    if (e) return (int)e;
  }
  return (int)cudaStreamSynchronize(g.stream);
}

extern "C" int pb_compute(void) {
  if (!g.ready) return 1;
  cudaError_t e;
  *((volatile int*)g.h_status) = 0;
  if (g.g_tiny) {
    if ((e = cudaGraphLaunch(g.g_tiny, g.stream))) return (int)e;
    return (int)cudaStreamSynchronize(g.stream);
  }
  if (g.persist) {
    e = cudaGraphLaunch(g.g_persist, g.stream);
    if (e == cudaSuccess) {
      if ((e = cudaStreamSynchronize(g.stream))) return (int)e;
      const int status = *((volatile int*)g.h_status);
      if (status != 1) return fallback_solve();
      return 0;
    }
    // Co-residency for the cooperative launch is not available (for example
    // because the device is shared); use the graph path from now on.
    cudaGetLastError();
    g.persist = false;
    *((volatile int*)g.h_status) = 0;
  }
  if ((e = cudaGraphLaunch(g.g_main, g.stream))) return (int)e;
  if ((e = cudaStreamSynchronize(g.stream))) return (int)e;
  int status = *((volatile int*)g.h_status);
  // Continue while the device reports neither convergence nor failure; the
  // device gives up (status 3) after MAX_ITERS steps.
  for (int guard = 0; status == 0 && guard < MAX_ITERS; guard += MORE_ITERS) {
    if ((e = cudaGraphLaunch(g.g_more, g.stream))) return (int)e;
    if ((e = cudaStreamSynchronize(g.stream))) return (int)e;
    status = *((volatile int*)g.h_status);
  }
  if (status != 1) return fallback_solve();
  return 0;
}

extern "C" int pb_download(pb_buffer* buffers) {
  if (!g.ready || !buffers || !buffers[2].data ||
      buffers[2].nbytes != (uint64_t)g.n * sizeof(float))
    return 1;
  const cudaError_t e = cudaMemcpyAsync(buffers[2].data, g.x, buffers[2].nbytes,
                                        cudaMemcpyDeviceToHost, g.stream);
  if (e) return (int)e;
  return (int)cudaStreamSynchronize(g.stream);
}

extern "C" int pb_teardown(void) {
  cudaError_t e = cudaSuccess;
  if (g.stream) e = cudaStreamSynchronize(g.stream);
  release_all();
  return (int)e;
}
