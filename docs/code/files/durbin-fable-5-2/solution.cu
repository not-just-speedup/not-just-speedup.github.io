// GPU solver for the Yule-Walker system  T y = -r  (the Durbin recursion of the
// CPU baseline), T = symmetric positive definite Toeplitz matrix with unit
// diagonal and r on its off-diagonals.
//
// See NOTES.md for the full rationale.  Outline of pb_compute (one CUDA graph,
// all control flow on the device):
//   1. "direct phase": the first K steps of the recursion (Schur form, no inner
//      products) in one thread block -> predictor polynomial a_m of order m
//      (K = m = 253; for n <= 512 all n steps, m = n-1).
//   2. preconditioned conjugate gradients on  T y = -r.  The preconditioner is
//      the exact inverse (Gohberg-Semencul form) of the Toeplitz matrix of the
//      AR(m) model given by a_m.  The iteration runs in single precision; the
//      true residual is evaluated at "anchors" in float-float arithmetic (~48
//      bits), where the correction is folded into the float-float iterate.
//      Products with T: direct O(n^2) sums for n <= 5120, otherwise an own
//      FFT (cuFFT is not available).  The result is accepted only when the
//      residual meets the tolerance (see k_decide / k_decide_anchor).
//   3. fallback (IF node): the full recursion in double precision, one small
//      kernel per step, if (2) does not reach the tolerance.
//   n <= kExactN is solved by a bit-exact replica of the CPU recursion instead.
#include "interface.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace pbsol {

// ----------------------------------------------------------------------------
// tunables
// ----------------------------------------------------------------------------
static int kExactN = 128;          // n <= kExactN: bit-exact replica of the CPU recursion
static int kMCap = 253;            // order of the AR model used as preconditioner (125 or 253)
static int kFullCap = 511;         // n - 1 <= this: the direct phase runs all n steps (<= 511)
static int kDirectMatvecN = 5120;  // n <= this: O(n^2) products instead of FFT (measured crossover ~5300)
static int kLocalLg = 9;           // log2 of the chunk transformed in shared memory
static int kRadixF = 4;            // max stages per register pass, single precision
static int kRadixX = 4;            // max stages per register pass, float-float
static int kSched1 = 3, kSched2 = 2;            // fixed prefix (see Flow), n > kSchedSmallN
static int kSched1Small = 2, kSched2Small = 2;  // fixed prefix, smaller n
static int kSchedSmallN = 700;
static int kSelfDotN = 4096;        // n <= this: k_upd computes p.q itself (no k_dot kernel) ...
static int kSelfDotBlock = 1024;    // ... in blocks of this many threads
static double kTolRel = 7.450580596923828125e-09;  // 2^-27: accept when residual norms <= tol * norms of y
static float kGap = 8.0f;          // safety factor of the bound on (true - recursive) residual
static float kFloor = 1.0e-4f;     // (re-)anchor when the recursive residual is below this * anchor
static const int kMaxAnchors = 40;
static const int kT2Unroll = 64;   // fallback steps per WHILE iteration
static const int kPartMax = 256;   // max number of block partials for reductions
static const int kRedBlock = 256;  // threads per block in reduction kernels

// device scalar slots (double)
enum { SC_BETA = 0, SC_ALPHA0 = 1, SC_ALPHA1 = 2, SC_COUNT = 8 };
// device scalar slots (float)
enum { FS_RZ0 = 0, FS_RZ1 = 1, FS_ANCHOR = 2, FS_YMAX = 3, FS_RR = 4, FS_TR = 5, FS_ANCHOR2 = 6, FS_Y2 = 7,
       FS_SCALE = 8, FS_RR2 = 9, FS_TR2 = 10, FS_RRREQ = 11, FS_RR2REQ = 12, FS_ALPHA = 13,
       FS_PATH = 14, FS_TNORM = 15, FS_PSCALE = 16, FS_COUNT = 24 };
// device int slots
enum { IC_STATUS = 0, IC_ITER = 1, IC_T2BASE = 2, IC_DIRBAD = 3, IC_T2LIMIT = 4, IC_T2RUNS = 5,
       IC_CGIT = 6, IC_NANCHOR = 7, IC_DISCARD = 8, IC_NDISCARD = 9, IC_FORCEFAIL = 10,
       IC_SLOW = 11, IC_IN2 = 12, IC_OUT = 13, IC_FALL = 14, IC_COUNT = 16 };
enum { ST_RUNNING = 0, ST_OK = 1, ST_FAIL = 2 };

// Control flow.  The graph starts with a fixed prefix - a few single precision
// iterations, one anchor, a few more iterations - that is executed
// unconditionally, because that is what a typical input needs and device-side
// conditional nodes are comparatively expensive.  The decisions inside the
// prefix only keep the books ("soft"); the last one decides what remains to be
// done, through three flags:
//   in2 : another single precision iteration
//   out : an anchor (true residual in float-float, fold the correction)
//   fall: the iteration failed, run the fallback
// mirrored in ic[] and in the condition variables of the conditional nodes
// (in2 drives two WHILE nodes; a condition variable can belong to one node
// only).  Everything behind the prefix sits in one IF node ("slow": set as
// soon as any flag is raised), so the common path evaluates one condition.
// (Alternatives measured: three or four top-level conditional nodes cost
// 25-30 us more; replacing the IF by a status word that pb_compute reads -
// through a copy node, or written by a kernel into mapped host memory - was
// not faster inside the real graph.)
struct Flow { cudaGraphConditionalHandle in2a, in2b, out, fall, slow; };
static __device__ __forceinline__ void set_flow(int* __restrict__ ic, const Flow& f, int in2,
                                                int out, int fall) {
  // only touch what changes (both ic[] and the condition variables start at 0)
  if (ic[IC_IN2] != in2) {
    ic[IC_IN2] = in2;
    cudaGraphSetConditional(f.in2a, (unsigned)in2);
    cudaGraphSetConditional(f.in2b, (unsigned)in2);
  }
  if (ic[IC_OUT] != out) { ic[IC_OUT] = out; cudaGraphSetConditional(f.out, (unsigned)out); }
  if (ic[IC_FALL] != fall) { ic[IC_FALL] = fall; cudaGraphSetConditional(f.fall, (unsigned)fall); }
  if ((in2 | out | fall) != 0 && ic[IC_SLOW] == 0) { ic[IC_SLOW] = 1; cudaGraphSetConditional(f.slow, 1u); }
}

// power of two that brings a quantity of magnitude ~v to ~1 (exact scaling)
static __device__ __forceinline__ float pow2_scale(float v) {
  float scale = 1.0f;
  if (v > 0.0f && v < 1.0e38f) {
    int e;
    frexpf(v, &e);
    e = -e;
    if (e > 100) e = 100;
    if (e < -100) e = -100;
    scale = ldexpf(1.0f, e);
  }
  return scale;
}

// ----------------------------------------------------------------------------
// direct phase: in-place Schur-Levinson recursion inside one thread block.
//   Arrays Mv[0..L-1], S[0..L-1], L = K+2.  At step k pair s couples S[s] with
//   Mv[(s+k+1) mod L]:
//     s in [0, K-k-1]   : generator window  (S[s] = v_k(s+k), Mv = u_k(s+k+1))
//     s in [K-k, K+1]   : predictor polynomial a_k and its reverse
//   Every pair is updated in place by (x,y) <- (x + al*y, y + al*x), al = alpha
//   of the step; pair 0 is the pivot pair (its x becomes 0, its y is beta).
//   After K steps a_K(i) = Mv[i-1], i = 1..K.
//   One thread per pair: blockDim.x == L (<= 1024).  The step loops have no
//   divergent branch (divergence is expensive on this device).  alpha for the
//   coming steps is computed by lane 0 from the freshly updated first pairs
//   (the other lanes of warp 0 execute the same instructions on harmless
//   constants) and published in shared memory.
// ----------------------------------------------------------------------------
// -x/y in double via a float reciprocal and one Newton step (relative error
// ~4e-15); a double precision division is very slow on this device.
static __device__ __forceinline__ double neg_ratio_d(double x, double y) {
  if (!(y > 1.0e-30 && y < 1.0e30)) return -x / y;
  double rd = (double)(1.0f / (float)y);
  rd = fma(rd, fma(-y, rd, 1.0), rd);
  return -x * rd;
}
static __device__ __forceinline__ float neg_ratio(float x, float y) { return -x / y; }

// Two steps per barrier (a barrier per step was measured to cost about 40%
// more).  A thread advances its pair by two steps at once: the second step
// needs the neighbouring pair's Mv value after the first step, which the
// thread computes itself from the neighbour's old values.  Since threads then
// read each other's elements, the arrays are double buffered (read one copy,
// write the other).  Lane 0 derives the next two alphas from the two-step
// results of pairs 0, 1 and 2.  Every value is produced by the same operations
// as in a one-step-per-barrier form (verified to give identical bits); that
// form finishes the job (odd counts, snapshot step).
//   msnap: step before which a_k, its reverse and beta_k are stored as the
//   preconditioner model (K-1 when K == n; otherwise the final a_K is used)
__global__ void k_direct(const float* __restrict__ r, int K, int msnap,
                          float* __restrict__ a_snap, float* __restrict__ b_snap,
                          float* __restrict__ poly_out, double* __restrict__ sc,
                          float* __restrict__ fs, int* __restrict__ ic) {
  extern __shared__ float smem_d2[];
  const int L = K + 2;
  const int s = threadIdx.x;
  float* Mc = smem_d2;        // L
  float* Sc = Mc + L;         // L + 1
  float* Mn = Sc + L + 1;     // L
  float* Sn = Mn + L;         // L + 1
  float* dum = Sn + L + 1;    // 64 scratch slots
  float* sal = dum + 64;      // alpha of step k in slot k & 3
  Mc[s] = (s == 0) ? 0.0f : ((s <= K) ? r[s - 1] : 1.0f);
  Sc[s] = (s == 0) ? 1.0f : ((s < K) ? r[s - 1] : ((s == K) ? 0.0f : 1.0f));
  int bad = 0;
  if (s == 0) {
    // alpha of steps 0 and 1 (the latter by the look-ahead formula applied to
    // the initial values of pairs 0 and 1; K >= 3)
    const float r0 = r[0], r1 = r[1];
    const float al0 = -r0;
    const float ny = fmaf(al0, r0, 1.0f);
    const float nx2 = fmaf(al0, r0, r1);
    const bool ok = ny > 0.0f;
    sal[0] = al0;
    sal[1] = ok ? neg_ratio(nx2, ny) : 0.0f;
    sal[2] = 0.0f;
    sal[3] = 0.0f;
    bad = ok ? 0 : 1;
    Sc[L] = 0.0f;
    Sn[L] = 0.0f;
  }
  __syncthreads();
  if (s < 32) {
    // problem scale: a power of two ~ 1 / max |r[0..K)| ; the single precision
    // iteration works on residuals and corrections multiplied by it
    float mx = 0.0f;
    for (int q = s + 1; q <= K; q += 32) mx = fmaxf(mx, fabsf(r[q - 1]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (s == 0) fs[FS_PSCALE] = pow2_scale(mx);
  }
  int idx = (s + 1 == L) ? 0 : (s + 1);
  const int rem = (s == 0) ? -1 : (K - 1 - s);  // step at which the pair leaves the window
  const bool w0 = (s < 32);
  const bool l0 = (s == 0);
  const bool lastp = (s == L - 1);
  const int kd = ((msnap < K) ? msnap : K) & ~1;  // steps [0, kd) are done in pairs
  int k = 0;
  for (; k < kd; k += 2) {
    const float a0 = sal[k & 3], a1 = sal[(k & 3) + 1];
    int idxp = idx + 1;
    idxp = (idxp == L) ? 0 : idxp;
    const float x = Mc[idx], y = Sc[s], xp = Mc[idxp], yp = Sc[s + 1];
    // step k: own S, and the Mv value of the neighbouring pair (zero if that
    // pair is the pivot pair)
    float y1 = fmaf(a0, x, y);
    y1 = (k == rem) ? 0.0f : y1;
    float x1p = fmaf(a0, yp, xp);
    x1p = lastp ? 0.0f : x1p;
    // step k + 1
    float x2 = fmaf(a1, y1, x1p);
    float y2 = fmaf(a1, x1p, y1);
    y2 = (k + 1 == rem) ? 0.0f : y2;
    x2 = l0 ? 0.0f : x2;
    if (w0) {
      // lane 0: y2 = beta after step k+1; with pair 1 (x2_1, y2_1) and pair 2
      // (x2_2) it yields alpha of steps k+2 and k+3.  Other lanes divide constants.
      const float x2_1 = __shfl_sync(0xffffffffu, x2, 1);
      const float y2_1 = __shfl_sync(0xffffffffu, y2, 1);
      const float x2_2 = __shfl_sync(0xffffffffu, x2, 2);
      const bool ok0 = l0 && (y2 > 0.0f);
      float an0 = neg_ratio(ok0 ? x2_1 : 0.5f, ok0 ? y2 : 1.0f);
      an0 = (l0 && !ok0) ? 0.0f : an0;
      const float bp = fmaf(an0, x2_1, y2);   // beta after step k+2
      const float up = fmaf(an0, y2_1, x2_2);  // pivot of step k+3
      const bool ok1 = l0 && (bp > 0.0f);
      float an1 = neg_ratio(ok1 ? up : 0.5f, ok1 ? bp : 1.0f);
      an1 = (l0 && !ok1) ? 0.0f : an1;
      bad |= (l0 && !(ok0 && ok1)) ? 1 : 0;
      float* pa = l0 ? (sal + ((k + 2) & 3)) : (dum + 2 * s);
      pa[0] = an0;
      pa[1] = an1;
    }
    Mn[idxp] = x2;
    Sn[s] = y2;
    idx += 2;
    idx = (idx >= L) ? (idx - L) : idx;
    float* t = Mc; Mc = Mn; Mn = t;
    t = Sc; Sc = Sn; Sn = t;
    __syncthreads();
  }
  for (; k < K; ++k) {
    if (k == msnap) {
      // snapshot of a_k and beta_k (each thread reads its own element only);
      // b is a reversed: b[u] = a[k-u]
      if (idx < k) {
        const float v = Mc[idx];
        a_snap[idx + 1] = v;
        b_snap[k - idx - 1] = v;
      }
      if (l0) { a_snap[0] = 1.0f; b_snap[k] = 1.0f; sc[SC_BETA] = (double)Sc[0]; }
    }
    const float al = sal[k & 3];
    const float x = Mc[idx], y = Sc[s];
    float nx = fmaf(al, y, x);
    float ny = fmaf(al, x, y);
    if (w0) {
      int idx2 = idx + 1;
      idx2 = (idx2 == L) ? 0 : idx2;
      const float x2 = Mc[idx2], y2 = Sc[s + 1];
      const float nx2 = fmaf(al, y2, x2);
      // lane 0: ny = new beta, nx2 = new pivot; other lanes divide constants
      const bool ok = l0 && (ny > 0.0f);
      const float an = neg_ratio(ok ? nx2 : 0.5f, ok ? ny : 1.0f);
      bad |= (l0 && !ok) ? 1 : 0;
      float* pa = l0 ? (sal + ((k + 1) & 3)) : (dum + 2 * s);
      *pa = (l0 && !ok) ? 0.0f : an;
      nx = l0 ? 0.0f : nx;
    }
    ny = (k == rem) ? 0.0f : ny;
    Mc[idx] = nx;
    Sc[s] = ny;
    idx = (idx + 1 == L) ? 0 : (idx + 1);
    __syncthreads();
  }
  // results: a_K(i) = Mv[i-1], i = 1..K ; beta_K = S[0]
  if (s < K) {
    const float v = Mc[s];
    poly_out[s] = v;
    if (msnap >= K) { a_snap[s + 1] = v; b_snap[K - s - 1] = v; }
  }
  if (l0) {
    const float bK = Sc[0];
    if (msnap >= K) {
      sc[SC_BETA] = (double)bK;
      a_snap[0] = 1.0f;
      b_snap[K] = 1.0f;
    }
    ic[IC_DIRBAD] = (bad || !(bK > 0.0f)) ? 1 : 0;
  }
}


// ----------------------------------------------------------------------------
// "float-float" numbers: an unevaluated sum hi + lo of two floats (|lo| <= half
// an ulp of hi), ~48 significant bits.  Built from error-free transformations
// (two-sum, fma based two-product).  Native double precision is ~58x slower
// than single precision on this device, so the few places that need more than
// 24 bits (the true residual at an anchor) use this arithmetic instead.
// ----------------------------------------------------------------------------
struct ff { float hi, lo; };
struct cff { ff x, y; };  // complex float-float, 16 bytes

static __device__ __forceinline__ ff ff_fast_two_sum(float a, float b) {  // |a| >= |b|
  ff r;
  r.hi = a + b;
  r.lo = b - (r.hi - a);
  return r;
}
static __device__ __forceinline__ ff ff_two_sum(float a, float b) {
  ff r;
  r.hi = a + b;
  const float bb = r.hi - a;
  r.lo = (a - (r.hi - bb)) + (b - bb);
  return r;
}
static __device__ __forceinline__ ff operator+(ff x, ff y) {
  const ff s = ff_two_sum(x.hi, y.hi);
  return ff_fast_two_sum(s.hi, s.lo + (x.lo + y.lo));
}
static __device__ __forceinline__ ff operator-(ff x) { ff r; r.hi = -x.hi; r.lo = -x.lo; return r; }
static __device__ __forceinline__ ff operator-(ff x, ff y) { return x + (-y); }
static __device__ __forceinline__ ff ff_add_f(ff x, float f) {
  const ff s = ff_two_sum(x.hi, f);
  return ff_fast_two_sum(s.hi, s.lo + x.lo);
}
static __device__ __forceinline__ ff operator*(ff x, ff y) {
  const float p = x.hi * y.hi;
  float e = fmaf(x.hi, y.hi, -p);
  e = fmaf(x.hi, y.lo, e);
  e = fmaf(x.lo, y.hi, e);
  return ff_fast_two_sum(p, e);
}
static __device__ __forceinline__ ff ff_half(ff x) { ff r; r.hi = 0.5f * x.hi; r.lo = 0.5f * x.lo; return r; }
// a*b - c*d and a*b + c*d
static __device__ __forceinline__ ff ff_mulsub(ff a, ff b, ff c, ff d) {
  const float p1 = a.hi * b.hi, p2 = c.hi * d.hi;
  const float e1 = fmaf(a.hi, b.hi, -p1), e2 = fmaf(c.hi, d.hi, -p2);
  const ff s = ff_two_sum(p1, -p2);
  float e = e1 - e2;
  e = fmaf(a.hi, b.lo, e);
  e = fmaf(a.lo, b.hi, e);
  e = fmaf(-c.hi, d.lo, e);
  e = fmaf(-c.lo, d.hi, e);
  return ff_fast_two_sum(s.hi, s.lo + e);
}
static __device__ __forceinline__ ff ff_muladd(ff a, ff b, ff c, ff d) {
  const float p1 = a.hi * b.hi, p2 = c.hi * d.hi;
  const float e1 = fmaf(a.hi, b.hi, -p1), e2 = fmaf(c.hi, d.hi, -p2);
  const ff s = ff_two_sum(p1, p2);
  float e = e1 + e2;
  e = fmaf(a.hi, b.lo, e);
  e = fmaf(a.lo, b.hi, e);
  e = fmaf(c.hi, d.lo, e);
  e = fmaf(c.lo, d.hi, e);
  return ff_fast_two_sum(s.hi, s.lo + e);
}

// ----------------------------------------------------------------------------
// FFT (complex, size M = 2^lgM; single precision and float-float).
//   Twiddle table W[j] = exp(-2 pi i j / N), N = 2M, j < M.
//   Forward: DIF (natural order in, bit reversed out).
//   Inverse: DIT (bit reversed in, natural out), unscaled.
//   "Global" pass: R consecutive radix-2 stages done in registers on groups of
//   2^R elements spaced `stride` apart.  "Local" pass: the lgC stages with the
//   smallest strides for one contiguous chunk of 2^lgC elements in shared memory.
// ----------------------------------------------------------------------------
template <class T> struct CplxOf;
template <> struct CplxOf<float> { typedef float2 type; };
template <> struct CplxOf<ff> { typedef cff type; };

static __device__ __forceinline__ float2 cmul(float2 a, float2 w) {
  float2 o;
  o.x = fmaf(a.x, w.x, -(a.y * w.y));
  o.y = fmaf(a.x, w.y, a.y * w.x);
  return o;
}
static __device__ __forceinline__ float2 cmulc(float2 a, float2 w) {  // a * conj(w)
  float2 o;
  o.x = fmaf(a.x, w.x, a.y * w.y);
  o.y = fmaf(a.y, w.x, -(a.x * w.y));
  return o;
}
static __device__ __forceinline__ cff cmul(cff a, cff w) {
  cff o;
  o.x = ff_mulsub(a.x, w.x, a.y, w.y);
  o.y = ff_muladd(a.x, w.y, a.y, w.x);
  return o;
}
static __device__ __forceinline__ cff cmulc(cff a, cff w) {  // a * conj(w)
  cff o;
  o.x = ff_muladd(a.x, w.x, a.y, w.y);
  o.y = ff_mulsub(a.y, w.x, a.x, w.y);
  return o;
}

template <class T, int R, int INV>
__global__ void k_fft(const typename CplxOf<T>::type* in, typename CplxOf<T>::type* out,
                      const typename CplxOf<T>::type* __restrict__ W, int lgStride, int lgM,
                      long long ngroups) {
  typedef typename CplxOf<T>::type C;
  const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= ngroups) return;
  const int G = 1 << R;
  const long long stride = 1LL << lgStride;
  const long long j = g & (stride - 1);
  const long long blk = g >> lgStride;
  const long long base = (blk << (lgStride + R)) + j;
  const int lgL = lgStride + R;
  C v[G];
#pragma unroll
  for (int i = 0; i < G; ++i) v[i] = in[base + ((long long)i << lgStride)];
  if (!INV) {
#pragma unroll
    for (int t = 0; t < R; ++t) {
      const int half = 1 << (R - t - 1);
      const int sh = (lgM + 1) - (lgL - t);
      C tw[G / 2];
#pragma unroll
      for (int u = 0; u < half; ++u) tw[u] = W[(j + ((long long)u << lgStride)) << sh];
#pragma unroll
      for (int c = 0; c < (1 << t); ++c) {
#pragma unroll
        for (int u = 0; u < half; ++u) {
          const int i0 = c * (2 * half) + u, i1 = i0 + half;
          const C a = v[i0], b = v[i1];
          C s, d;
          s.x = a.x + b.x; s.y = a.y + b.y;
          d.x = a.x - b.x; d.y = a.y - b.y;
          v[i0] = s;
          v[i1] = cmul(d, tw[u]);
        }
      }
    }
  } else {
#pragma unroll
    for (int t = R - 1; t >= 0; --t) {
      const int half = 1 << (R - t - 1);
      const int sh = (lgM + 1) - (lgL - t);
      C tw[G / 2];
#pragma unroll
      for (int u = 0; u < half; ++u) tw[u] = W[(j + ((long long)u << lgStride)) << sh];
#pragma unroll
      for (int c = 0; c < (1 << t); ++c) {
#pragma unroll
        for (int u = 0; u < half; ++u) {
          const int i0 = c * (2 * half) + u, i1 = i0 + half;
          const C a = v[i0], b = cmulc(v[i1], tw[u]);
          C s, d;
          s.x = a.x + b.x; s.y = a.y + b.y;
          d.x = a.x - b.x; d.y = a.y - b.y;
          v[i0] = s;
          v[i1] = d;
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < G; ++i) out[base + ((long long)i << lgStride)] = v[i];
}

// local pass: blockDim.x == C/2 threads (C = 2^lgC >= 2), one chunk per block
// WL[j] = exp(-2 pi i j / Cn), j < Cn/2 (compact table for the local stages)
template <class T, int INV>
__global__ void k_fft_local(const typename CplxOf<T>::type* in, typename CplxOf<T>::type* out,
                            const typename CplxOf<T>::type* __restrict__ WL, int lgC) {
  typedef typename CplxOf<T>::type C;
  extern __shared__ double smem_fft[];
  const int Cn = 1 << lgC;
  const int hC = Cn >> 1;
  C* sm = reinterpret_cast<C*>(smem_fft);  // Cn elements
  C* tw = sm + Cn;                         // hC twiddles: exp(-2 pi i j / Cn)
  const int t = threadIdx.x;
  const long long off = (long long)blockIdx.x << lgC;
  sm[t] = in[off + t];
  sm[t + hC] = in[off + t + hC];
  tw[t] = WL[t];
  __syncthreads();
  if (!INV) {
    for (int s = lgC; s >= 1; --s) {
      const int half = 1 << (s - 1);
      const int j = t & (half - 1);
      const int i0 = ((t >> (s - 1)) << s) + j;
      const int i1 = i0 + half;
      const C w = tw[j << (lgC - s)];
      const C a = sm[i0], b = sm[i1];
      C su, d;
      su.x = a.x + b.x; su.y = a.y + b.y;
      d.x = a.x - b.x; d.y = a.y - b.y;
      sm[i0] = su;
      sm[i1] = cmul(d, w);
      __syncthreads();
    }
  } else {
    for (int s = 1; s <= lgC; ++s) {
      const int half = 1 << (s - 1);
      const int j = t & (half - 1);
      const int i0 = ((t >> (s - 1)) << s) + j;
      const int i1 = i0 + half;
      const C w = tw[j << (lgC - s)];
      const C a = sm[i0], b = cmulc(sm[i1], w);
      C su, d;
      su.x = a.x + b.x; su.y = a.y + b.y;
      d.x = a.x - b.x; d.y = a.y - b.y;
      sm[i0] = su;
      sm[i1] = d;
      __syncthreads();
    }
  }
  out[off + t] = sm[t];
  out[off + t + hC] = sm[t + hC];
}

static __device__ __forceinline__ long long brev(long long k, int lg) {
  // bit reversal of the low lg bits (1 <= lg <= 32)
  return (long long)(__brev((unsigned int)k) >> (32 - lg));
}

// circulant first row of the (2M x 2M) embedding, packed as M complex numbers
__global__ void k_packc(const float* __restrict__ r, long long n, long long M,
                        cff* __restrict__ Z) {
  const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= M) return;
  const long long N = 2 * M;
  float v[2];
  for (int q = 0; q < 2; ++q) {
    const long long d = 2 * j + q;
    float c = 0.0f;
    if (d == 0) c = 1.0f;
    else if (d <= n - 1) c = r[d - 1];
    else if (d >= N - (n - 1)) c = r[N - d - 1];
    v[q] = c;
  }
  cff o;
  o.x.hi = v[0]; o.x.lo = 0.0f;
  o.y.hi = v[1]; o.y.lo = 0.0f;
  Z[j] = o;
}

// symbol: c_k = (DFT_N c)_k / M, k = 0..M   (real because c is even), in
// float-float (CHX) and rounded to single precision (CHF).  rM = 1/M (exact).
// Stored in bit reversed order, CH[brev(k)] = c_k for k < M and CH[M] = c_M, so
// that it lines up with the bit reversed spectrum the forward transform leaves.
__global__ void k_symbol(const cff* __restrict__ Z, const cff* __restrict__ W,
                         ff* __restrict__ CHX, float* __restrict__ CHF, long long M,
                         int lgM, float rM) {
  const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (k > M) return;
  const long long ka = (k == M) ? 0 : k;
  const long long kb = (k == 0 || k == M) ? 0 : (M - k);
  const long long pa = brev(ka, lgM);
  const cff A = Z[pa], B = Z[brev(kb, lgM)];
  cff w;
  if (k == M) { w.x.hi = -1.0f; w.x.lo = 0.0f; w.y.hi = 0.0f; w.y.lo = 0.0f; }
  else w = W[k];
  const ff Dx = A.x - B.x, Dy = A.y + B.y;
  ff c = ff_half((A.x + B.x) + ff_muladd(w.x, Dy, w.y, Dx));
  c.hi *= rM;
  c.lo *= rM;
  const long long dst = (k == M) ? M : pa;
  CHX[dst] = c;
  CHF[dst] = c.hi + c.lo;
}

// One pair (k, M-k) of the product with the symbol: A = Z_k, B = Z_{M-k} are
// entries of the half-size transform of the packed real sequence; they are
// unpacked to the spectrum of the real sequence, multiplied by the symbol
// (ck = c_k, cm = c_{M-k}) and packed again.  w = exp(-2 pi i k / N).
static __device__ __forceinline__ void pw_pair(float2& A, float2& B, float2 w, float ck, float cm) {
  const float h = 0.5f;
  const float Sx = A.x + B.x, Sy = A.y - B.y, Dx = A.x - B.x, Dy = A.y + B.y;
  const float Px = fmaf(w.x, Dx, -(w.y * Dy)), Py = fmaf(w.x, Dy, w.y * Dx);
  const float Hkx = ck * (h * (Sx + Py)), Hky = ck * (h * (Sy - Px));
  const float Hmx = cm * (h * (Sx - Py)), Hmy = cm * (h * (-Sy - Px));
  const float S2x = Hkx + Hmx, S2y = Hky - Hmy, D2x = Hkx - Hmx, D2y = Hky + Hmy;
  const float P2x = fmaf(w.x, D2x, w.y * D2y), P2y = fmaf(w.x, D2y, -(w.y * D2x));
  A.x = h * (S2x - P2y); A.y = h * (S2y + P2x);
  B.x = h * (S2x + P2y); B.y = h * (P2x - S2y);
}
static __device__ __forceinline__ void pw_pair(cff& A, cff& B, cff w, ff ck, ff cm) {
  cff S, D;
  S.x = A.x + B.x; S.y = A.y - B.y;
  D.x = A.x - B.x; D.y = A.y + B.y;
  const cff P = cmul(D, w);
  const ff Hkx = ck * ff_half(S.x + P.y), Hky = ck * ff_half(S.y - P.x);
  const ff Hmx = cm * ff_half(S.x - P.y), Hmy = cm * ff_half(-(S.y + P.x));
  cff D2;
  const ff S2x = Hkx + Hmx, S2y = Hky - Hmy;
  D2.x = Hkx - Hmx; D2.y = Hky + Hmy;
  const cff P2 = cmulc(D2, w);
  A.x = ff_half(S2x - P2.y); A.y = ff_half(S2y + P2.x);
  B.x = ff_half(S2x + P2.y); B.y = ff_half(P2.x - S2y);
}

// Middle of the product with T, fused into one kernel: the last lgC forward
// stages, the multiplication by the symbol, and the first lgC inverse stages.
// After the global forward passes the array consists of M/Cn chunks; chunk c
// holds the indices k with k mod 2^b = brev_b(c) (b = lgM - lgC), in bit
// reversed order of k div 2^b.  The partner M-k of an index k lives in the
// chunk of -k mod 2^b, at the mirrored position Cn-1-t.  One block therefore
// handles a pair of chunks (or one of the two self-paired chunks, klow = 0 and
// klow = 2^(b-1)): blockDim.x == Cn, threads [0,Cn/2) do the butterflies of
// chunk A and threads [Cn/2,Cn) those of chunk B.
//   WL: exp(-2 pi i j / Cn), j < Cn/2 ; WB[p] = W[brev(p)] ; CH: see k_symbol
template <class T>
__global__ void k_fft_mid(const typename CplxOf<T>::type* in, typename CplxOf<T>::type* out,
                          const typename CplxOf<T>::type* __restrict__ WL,
                          const typename CplxOf<T>::type* __restrict__ WB,
                          const T* __restrict__ CH, int lgC, int lgM) {
  typedef typename CplxOf<T>::type C;
  extern __shared__ double smem_mid[];
  const int Cn = 1 << lgC, hC = Cn >> 1;
  C* A = reinterpret_cast<C*>(smem_mid);
  C* B = A + Cn;
  C* tw = B + Cn;
  const int b = lgM - lgC;
  const int q = blockIdx.x;
  const int halfb = (b >= 1) ? (1 << (b - 1)) : -1;
  const bool self0 = (q == 0);            // klow = 0
  const bool selfh = (q == halfb);        // klow = 2^(b-1)
  const bool two = !(self0 || selfh);
  const long long cA = (b >= 1) ? brev(q, b) : 0;
  const long long cB = two ? brev((1LL << b) - q, b) : cA;
  const long long offA = cA << lgC, offB = cB << lgC;
  const long long M = 1LL << lgM;
  const int t = threadIdx.x;
  A[t] = in[offA + t];
  if (two) B[t] = in[offB + t];
  if (t < hC) tw[t] = WL[t];
  __syncthreads();
  const int u = t & (hC - 1);
  C* X = (t < hC) ? A : B;
  const bool act = (t < hC) || two;
  // forward stages lgC .. 1
  for (int s = lgC; s >= 1; --s) {
    if (act) {
      const int half = 1 << (s - 1);
      const int j = u & (half - 1);
      const int i0 = ((u >> (s - 1)) << s) + j;
      const int i1 = i0 + half;
      const C w = tw[j << (lgC - s)];
      const C a = X[i0], bb = X[i1];
      C su, d;
      su.x = a.x + bb.x; su.y = a.y + bb.y;
      d.x = a.x - bb.x; d.y = a.y - bb.y;
      X[i0] = su;
      X[i1] = cmul(d, w);
    }
    __syncthreads();
  }
  // product with the symbol
  if (two) {
    const int pa = t, pb = Cn - 1 - t;
    C a = A[pa], bb = B[pb];
    pw_pair(a, bb, WB[offA + pa], CH[offA + pa], CH[offB + pb]);
    A[pa] = a;
    B[pb] = bb;
  } else if (selfh) {
    if (t < hC) {
      const int pa = t, pb = Cn - 1 - t;
      C a = A[pa], bb = A[pb];
      pw_pair(a, bb, WB[offA + pa], CH[offA + pa], CH[offA + pb]);
      A[pa] = a;
      A[pb] = bb;
    }
  } else if (t <= hC) {
    // klow = 0: k = j * 2^b pairs with (Cn - j) * 2^b ; j = 0 and j = Cn/2 pair with themselves
    const int j = t;
    const int pa = (int)brev(j, lgC);
    const int pb = (int)brev((Cn - j) & (Cn - 1), lgC);
    C a = A[pa], bb = A[pb];
    const T cm = (j == 0) ? CH[M] : CH[pb];
    pw_pair(a, bb, WB[pa], CH[pa], cm);
    A[pa] = a;
    if (pb != pa) A[pb] = bb;
  }
  __syncthreads();
  // inverse stages 1 .. lgC
  for (int s = 1; s <= lgC; ++s) {
    if (act) {
      const int half = 1 << (s - 1);
      const int j = u & (half - 1);
      const int i0 = ((u >> (s - 1)) << s) + j;
      const int i1 = i0 + half;
      const C w = tw[j << (lgC - s)];
      const C a = X[i0], bb = cmulc(X[i1], w);
      C su, d;
      su.x = a.x + bb.x; su.y = a.y + bb.y;
      d.x = a.x - bb.x; d.y = a.y - bb.y;
      X[i0] = su;
      X[i1] = d;
    }
    __syncthreads();
  }
  out[offA + t] = A[t];
  if (two) out[offB + t] = B[t];
}

// ----------------------------------------------------------------------------
// direct O(n^2) product q = T p for small n (single precision): one warp per row
//   RR[d] = R(d), d = 0..n-1: the input buffer itself shifted by one element,
//   with R(0) = 1 stored in front of it (see setup)
// ----------------------------------------------------------------------------
__global__ void k_matvec_direct(const float* __restrict__ RR, const float* __restrict__ p,
                                float* __restrict__ q, int n) {
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= n) return;
  float acc = 0.0f;
  for (int j = lane; j < n; j += 32) {
    const int d = (j >= row) ? (j - row) : (row - j);
    acc = fmaf(RR[d], p[j], acc);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) q[row] = acc;
}

// The same product in float-float for the anchor (y in float-float): every
// lane accumulates its share of the row with error-free products and sums (the
// low order parts are added up in single precision), then the lanes are added
// in float-float.
__global__ void k_matvec_direct_x(const float* __restrict__ RR, const ff* __restrict__ Y,
                                  ff* __restrict__ Q, int n) {
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= n) return;
  float sh = 0.0f, sl = 0.0f;
  for (int j = lane; j < n; j += 32) {
    const int d = (j >= row) ? (j - row) : (row - j);
    const float c = RR[d];
    const ff y = Y[j];
    const float p = c * y.hi;
    const float e = fmaf(c, y.hi, -p);  // p + e == c * y.hi exactly
    const ff s = ff_two_sum(sh, p);     // s.hi + s.lo == sh + p exactly
    sh = s.hi;
    sl = fmaf(c, y.lo, sl + (s.lo + e));
  }
  ff v = ff_two_sum(sh, sl);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    ff u;
    u.hi = __shfl_xor_sync(0xffffffffu, v.hi, o);
    u.lo = __shfl_xor_sync(0xffffffffu, v.lo, o);
    v = v + u;
  }
  if (lane == 0) Q[row] = v;
}

// ----------------------------------------------------------------------------
// block reductions (single precision; shuffles and double precision operations
// are both expensive on this device, so they are used sparingly)
// ----------------------------------------------------------------------------
static __device__ __forceinline__ float block_sum_f(float v) {  // result in thread 0
  __shared__ float redf_s[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if (lane == 0) redf_s[wid] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  float t = 0.0f;
  if (wid == 0) {
    t = (lane < nw) ? redf_s[lane] : 0.0f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
  }
  return t;
}
// the block total, returned to every thread
static __device__ __forceinline__ float block_sum_all_f(float v) {
  __shared__ float bsa_tot;
  const float t = block_sum_f(v);
  if (threadIdx.x == 0) bsa_tot = t;
  __syncthreads();
  return bsa_tot;
}
static __device__ __forceinline__ float block_max_f(float v) {  // result in thread 0
  __shared__ float redm_s[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  if (lane == 0) redm_s[wid] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  float t = 0.0f;
  if (wid == 0) {
    t = (lane < nw) ? redm_s[lane] : 0.0f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
  }
  return t;
}
// sum / max of `cnt` (<= kPartMax) block partials: computed by warp 0, returned
// to every thread of the block (identical in every block)
static __device__ __forceinline__ float partials_sum_f(const float* __restrict__ part, int cnt) {
  __shared__ float ps_tot;
  if (threadIdx.x < 32) {
    float v = 0.0f;
    for (int i = threadIdx.x; i < cnt; i += 32) v += part[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) ps_tot = v;
  }
  __syncthreads();
  return ps_tot;
}
static __device__ __forceinline__ float partials_max_f(const float* __restrict__ part, int cnt) {
  __shared__ float pm_tot;
  if (threadIdx.x < 32) {
    float v = 0.0f;
    for (int i = threadIdx.x; i < cnt; i += 32) v = fmaxf(v, part[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) pm_tot = v;
  }
  __syncthreads();
  return pm_tot;
}
static __device__ __forceinline__ float clamp_abs_f(float v) {
  const float a = fabsf(v);
  return (a <= 3.0e38f) ? a : 3.0e38f;  // NaN/Inf -> huge
}

// ----------------------------------------------------------------------------
// iteration kernels
//   Y  (float-float): iterate at the last anchor;  DF (float): correction since then
//   RF (float) : residual (true residual at an anchor, recursive afterwards)
//   RF, DF, PF, QF, ZF and the norms kept in fs[] are in units of 1/FS_PSCALE
//   (a power of two set by the direct phase), which makes the iteration
//   independent of the magnitude of the data
//   PF, QF, ZF (float): search direction, T*PF, preconditioned residual
// ----------------------------------------------------------------------------
// per-invocation iteration state (called by one thread at the start of the
// iteration part of the graph)
static __device__ __forceinline__ void reset_state(float* __restrict__ fs, int* __restrict__ ic) {
  ic[IC_STATUS] = ST_RUNNING;
  ic[IC_ITER] = 0;
  ic[IC_CGIT] = 0;
  ic[IC_NANCHOR] = 0;
  ic[IC_DISCARD] = 0;
  ic[IC_NDISCARD] = 0;
  ic[IC_SLOW] = 0;
  ic[IC_IN2] = 0;
  ic[IC_OUT] = 0;
  ic[IC_FALL] = 0;
  fs[FS_RZ0] = 0.0f;
  fs[FS_RZ1] = 0.0f;
  fs[FS_ANCHOR] = 0.0f;
  fs[FS_YMAX] = 0.0f;
  fs[FS_RR] = 0.0f;
  fs[FS_TR] = 0.0f;
  fs[FS_ANCHOR2] = 0.0f;
  fs[FS_Y2] = 0.0f;
  fs[FS_SCALE] = 1.0f;
  fs[FS_RR2] = 0.0f;
  fs[FS_TR2] = 0.0f;
  fs[FS_RRREQ] = 0.0f;
  fs[FS_RR2REQ] = 0.0f;
  fs[FS_ALPHA] = 0.0f;
  fs[FS_PATH] = 0.0f;
}

// fold the single precision correction into the float-float iterate
// (or drop it, if the single precision iteration was found to diverge)
__global__ void k_anchor_add(ff* __restrict__ Y, float* __restrict__ DF,
                             const float* __restrict__ fs, const int* __restrict__ ic, int n) {
  const bool discard = ic[IC_DISCARD] != 0;
  const float inv = 1.0f / fs[FS_PSCALE];  // exact (power of two)
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (!discard) Y[i] = ff_add_f(Y[i], DF[i] * inv);
  DF[i] = 0.0f;
}

// true residual rho = -r - T y (Q = T y in float-float); block partials of the
// max norms and of the scaled sums of squares of rho and y
__global__ void k_res(const float* __restrict__ r, const ff* __restrict__ Q,
                      const ff* __restrict__ Y, float* __restrict__ RF,
                      float* __restrict__ pmaxr, float* __restrict__ pmaxy,
                      float* __restrict__ psqr, float* __restrict__ psqy,
                      const float* __restrict__ fs, int n) {
  const int tot = gridDim.x * blockDim.x;
  const float scale = pow2_scale(fs[FS_YMAX]);
  const float ps = fs[FS_PSCALE];
  float ar = 0.0f, ay = 0.0f, sr = 0.0f, sy = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += tot) {
    const ff q = Q[i];
    const ff t = ff_two_sum(r[i], q.hi);   // exact
    const float v = -(t.hi + (t.lo + q.lo)) * ps;
    const float y = Y[i].hi * ps;
    RF[i] = v;
    ar = fmaxf(ar, clamp_abs_f(v));
    ay = fmaxf(ay, clamp_abs_f(y));
    const float vs = v * scale, ys = y * scale;
    sr = fmaf(vs, vs, sr);
    sy = fmaf(ys, ys, sy);
  }
  ar = block_max_f(ar);
  __syncthreads();
  ay = block_max_f(ay);
  __syncthreads();
  sr = block_sum_f(sr);
  __syncthreads();
  sy = block_sum_f(sy);
  if (threadIdx.x == 0) {
    pmaxr[blockIdx.x] = ar; pmaxy[blockIdx.x] = ay;
    psqr[blockIdx.x] = sr; psqy[blockIdx.x] = sy;
  }
}

// decision after an anchor (true residual known).  The result is accepted
// when both the max norm and the 2-norm of the residual are below tolrel times
// the corresponding norm of y (a few per cent of what rounding y to single
// precision produces).  soft: inside the fixed prefix (keep the books only).
__global__ void k_decide_anchor(const float* __restrict__ pmaxr, const float* __restrict__ pmaxy,
                                const float* __restrict__ psqr, const float* __restrict__ psqy,
                                int nparts, float* __restrict__ fs, int* __restrict__ ic,
                                float tolrel, int itcap, int soft, Flow flow) {
  const float tr = partials_max_f(pmaxr, nparts);
  __syncthreads();
  const float ym = partials_max_f(pmaxy, nparts);
  __syncthreads();
  const float s2r = partials_sum_f(psqr, nparts);
  __syncthreads();
  const float s2y = partials_sum_f(psqy, nparts);
  if (threadIdx.x == 0) {
    const float sres = pow2_scale(fs[FS_YMAX]);  // the scale k_res used
    const float tr2 = sqrtf(s2r) / sres;
    const float y2 = sqrtf(s2y) / sres;
    const int na = ic[IC_NANCHOR] + 1;
    const float prev2 = fs[FS_ANCHOR2];
    const bool discarded = ic[IC_DISCARD] != 0;  // the iterate is the previous anchor's
    int nd = ic[IC_NDISCARD];
    if (discarded) { ++nd; ic[IC_NDISCARD] = nd; ic[IC_DISCARD] = 0; }
    int status = ST_RUNNING;
    if (!(tr < 1.0e38f) || !(ym < 1.0e38f) || !(tr2 < 1.0e38f) || !(y2 < 1.0e38f)) status = ST_FAIL;
    else if (tr <= tolrel * ym && tr2 <= tolrel * y2) status = ST_OK;
    else if (discarded ? (nd > 2) : (na >= 2 && tr2 > 0.5f * prev2)) status = ST_FAIL;  // no progress
    else if (na > kMaxAnchors || ic[IC_ITER] >= itcap) status = ST_FAIL;
    // Keep the conjugate direction across the anchor only if the true residual
    // confirms the recursive one; otherwise restart the iteration from it.
    const float rreq2 = fs[FS_RR2REQ];
    const bool keep = !discarded && rreq2 > 0.0f && tr2 <= 1.25f * rreq2 && tr2 >= 0.8f * rreq2;
    if (!keep) ic[IC_CGIT] = 0;
    fs[FS_ANCHOR] = tr;
    fs[FS_ANCHOR2] = tr2;
    fs[FS_TR] = tr;
    fs[FS_TR2] = tr2;
    fs[FS_YMAX] = ym;
    fs[FS_Y2] = y2;
    fs[FS_SCALE] = pow2_scale(tr);  // for the single precision sums of squares
    fs[FS_PATH] = 0.0f;
    ic[IC_NANCHOR] = na;
    if (ic[IC_STATUS] == ST_FAIL) status = ST_FAIL;  // an earlier failure stands
    if (soft) {
      if (status == ST_FAIL) ic[IC_STATUS] = ST_FAIL;
    } else {
      ic[IC_STATUS] = status;
      set_flow(ic, flow, status == ST_RUNNING ? 1 : 0, 0, status == ST_FAIL ? 1 : 0);
    }
  }
}

// ---- FIR machinery (preconditioner and initial residual) -------------------
//   taps[0..mp), mp = NTG*32 (zero padded), NTG = 2^LGTG tap groups of 32 taps.
//   One warp produces 32*NOG consecutive outputs, NOG = 32/NTG output groups.
//   lane = NTG*og + tg: the lane accumulates, for the 32 outputs of output
//   group og, the 32 taps of tap group tg with a 32 wide sliding window held in
//   registers; the partial sums are then reduced over the tap groups (LGTG
//   shuffle rounds).  Global memory loads are the scarce resource on this
//   device, and this layout needs only about 2*mp/32 of them per output.
struct F32 { float v[32]; };

template <int LGTG>
static __device__ __forceinline__ void reduce_tg(F32& a) {
#pragma unroll
  for (int o = (1 << LGTG) >> 1; o > 0; o >>= 1) {
#pragma unroll
    for (int k = 0; k < 32; ++k) a.v[k] += __shfl_xor_sync(0xffffffffu, a.v[k], o);
  }
}
// component tg + NTG*k of a (tg in 0..NTG-1)
template <int LGTG>
static __device__ __forceinline__ float pick(const F32& a, int tg, int k) {
  const int NTG = 1 << LGTG;
  float r = a.v[NTG * k];
#pragma unroll
  for (int q = 1; q < NTG; ++q) r = (tg == q) ? a.v[NTG * k + q] : r;
  return r;
}
// acc[o] += sum_{j in [32*tg, 32*tg+32)} taps[j] * x[o + j], o = 0..31
static __device__ __forceinline__ void fir_corr32(F32& acc, const float* __restrict__ taps,
                                                   const float* __restrict__ x, int tg) {
  const float* tp = taps + 32 * tg;
  const float* xp = x + 32 * tg;
  float xw[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) xw[k] = xp[k];
#pragma unroll
  for (int u = 0; u < 32; ++u) {
    const float tap = tp[u];
    // the window element for output o is xw[(o + u) & 31] = xp[u + o]
#pragma unroll
    for (int o = 0; o < 32; ++o) acc.v[o] = fmaf(tap, xw[(o + u) & 31], acc.v[o]);
    xw[u] = xp[u + 32];
  }
}
// acc[o] += sgn * sum_{j in [32*tg, 32*tg+32)} taps[j] * w[o - j], o = 0..31
static __device__ __forceinline__ void fir_conv32(F32& acc, const float* __restrict__ taps,
                                                   const float* __restrict__ w, int tg, float sgn) {
  const float* tp = taps + 32 * tg;
  const float* wp = w - 32 * tg;
  // ww[c & 31] holds wp[c] for the 32 offsets c currently needed
  float ww[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) ww[k] = wp[k];
#pragma unroll
  for (int u = 0; u < 32; ++u) {
    const float tap = sgn * tp[u];
    // output o needs wp[o - u]: residue (o - u) mod 32
#pragma unroll
    for (int o = 0; o < 32; ++o) acc.v[o] = fmaf(tap, ww[(o - u) & 31], acc.v[o]);
    ww[(31 - u) & 31] = wp[-u - 1];
  }
}

// Start of the iteration: y0 = (a_m, 0) (Y, float-float), correction DF = 0, and
// the initial residual: rho0[i] = -sum_j a[j] r[i-j] for i >= K (K = m here)
// and 0 for i < K, where the first K equations hold up to single precision
// rounding (the anchor accounts for that later).  Also the norms of rho0 and
// y0, and the per-invocation state.  r: padded in front.
template <int LGTG>
__global__ void k_res0(const float* __restrict__ a, const float* __restrict__ r,
                       const float* __restrict__ poly, int K, float* __restrict__ RF,
                       ff* __restrict__ Y, float* __restrict__ DF,
                       float* __restrict__ pmaxr, float* __restrict__ psqr,
                       float* __restrict__ pmaxy, float* __restrict__ psqy,
                       float* __restrict__ psumr, float* __restrict__ fs,
                       int* __restrict__ ic, int n, int g1) {
  if (blockIdx.x == 0 && threadIdx.x == 0) reset_state(fs, ic);
  const float ps = fs[FS_PSCALE];
  const int NTG = 1 << LGTG, NOG = 32 >> LGTG, OPW = 32 * NOG;
  const int lane = threadIdx.x & 31, tg = lane & (NTG - 1), og32 = (lane >> LGTG) << 5;
  const int wpb = blockDim.x >> 5;
  const int totw = gridDim.x * wpb;
  float ar = 0.f, sr = 0.f, ay = 0.f, sy = 0.f, s1 = 0.f;
  for (int g = blockIdx.x * wpb + (threadIdx.x >> 5); g < g1; g += totw) {
    F32 acc;
#pragma unroll
    for (int k = 0; k < 32; ++k) acc.v[k] = 0.f;
    if (OPW * g + OPW > K) {  // warp-uniform: rows below K are zero
      fir_conv32(acc, a, r + OPW * g + og32, tg, -1.0f);
      reduce_tg<LGTG>(acc);
    }
    const int i0 = OPW * g + og32 + tg;
#pragma unroll
    for (int k = 0; k < NOG; ++k) {
      const int i = i0 + NTG * k;
      const float v = (i >= K && i < n) ? pick<LGTG>(acc, tg, k) * ps : 0.f;
      RF[i] = v;
      ar = fmaxf(ar, clamp_abs_f(v));
      sr = fmaf(v, v, sr);
      const float pv = (i < K) ? poly[i] : 0.f;
      if (i < n) {
        ff y0;
        y0.hi = pv;
        y0.lo = 0.f;
        Y[i] = y0;
        DF[i] = 0.f;
      }
      const float y = pv * ps;
      ay = fmaxf(ay, clamp_abs_f(y));
      sy = fmaf(y, y, sy);
      s1 += (i < n) ? clamp_abs_f(r[i]) : 0.f;
    }
  }
  ar = block_max_f(ar);
  __syncthreads();
  ay = block_max_f(ay);
  __syncthreads();
  sr = block_sum_f(sr);
  __syncthreads();
  sy = block_sum_f(sy);
  __syncthreads();
  s1 = block_sum_f(s1);
  if (threadIdx.x == 0) {
    pmaxr[blockIdx.x] = ar; psqr[blockIdx.x] = sr;
    pmaxy[blockIdx.x] = ay; psqy[blockIdx.x] = sy;
    psumr[blockIdx.x] = s1;
  }
}

// decision before the first single precision loop
__global__ void k_decide0(const float* __restrict__ pmaxr, const float* __restrict__ psqr,
                          const float* __restrict__ pmaxy, const float* __restrict__ psqy,
                          const float* __restrict__ psumr,
                          int nparts, const double* __restrict__ sc, float* __restrict__ fs,
                          int* __restrict__ ic, float tolrel, Flow flow) {
  const float rr = partials_max_f(pmaxr, nparts);
  __syncthreads();
  const float ssq = partials_sum_f(psqr, nparts);
  __syncthreads();
  const float ym = partials_max_f(pmaxy, nparts);
  __syncthreads();
  const float sy = partials_sum_f(psqy, nparts);
  __syncthreads();
  const float s1 = partials_sum_f(psumr, nparts);
  if (threadIdx.x == 0) {
    const float rr2 = sqrtf(ssq), y2 = sqrtf(sy);
    const float tnorm = 1.0f + 2.0f * s1;  // >= ||T||_inf >= ||T||_2
    fs[FS_TNORM] = tnorm;
    // (IC_FORCEFAIL is a test hook: it is never set outside development runs)
    const bool bad = ic[IC_DIRBAD] != 0 || !(sc[SC_BETA] > 0.0) || !(rr < 1.0e38f) ||
                     !(rr2 < 1.0e38f) || !(ym < 1.0e38f) || !(y2 < 1.0e38f) ||
                     !(tnorm < 1.0e38f) || ic[IC_FORCEFAIL] != 0;
    fs[FS_ANCHOR] = rr;
    fs[FS_ANCHOR2] = rr2;
    fs[FS_YMAX] = ym;
    fs[FS_Y2] = y2;
    fs[FS_SCALE] = 1.0f;
    fs[FS_RR] = rr;
    fs[FS_RR2] = rr2;
    fs[FS_RRREQ] = rr;
    fs[FS_RR2REQ] = rr2;
    // (the fixed prefix that follows runs in any case; only a failure is recorded)
    ic[IC_STATUS] = bad ? ST_FAIL : ST_RUNNING;
    set_flow(ic, flow, 0, 0, bad ? 1 : 0);
  }
}

// ---- preconditioner: z = (L_a L_a^T - L_z L_z^T) rho / beta  (Gohberg-Semencul
// inverse of the AR(m) model matrix), two FIR passes over zero padded arrays.
//   taps a[0..m] (a[0] = 1) and b[u] = a[m-u].
// stage 1:  w = L_a^T rho (warps [0, g1)),  w2 = L_z^T rho (warps [g1, g1+gc))
//   rf: zero padded residual ; w: points at w[0] (zeros before it)
//   w2abs: points at absolute index cbase = (n-m) rounded down to a multiple
//   of the outputs per warp (zeros before it)
template <int LGTG>
__global__ void k_prec1(const float* __restrict__ a, const float* __restrict__ b,
                        const float* __restrict__ rf, float* __restrict__ w,
                        float* __restrict__ w2abs, int n, int m, int g1, int gc) {
  const int NTG = 1 << LGTG, NOG = 32 >> LGTG, OPW = 32 * NOG;
  const int lane = threadIdx.x & 31, tg = lane & (NTG - 1), og32 = (lane >> LGTG) << 5;
  const int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  F32 acc;
#pragma unroll
  for (int k = 0; k < 32; ++k) acc.v[k] = 0.f;
  if (g < g1) {
    fir_corr32(acc, a, rf + OPW * g + og32, tg);
    reduce_tg<LGTG>(acc);
    float* dst = w + OPW * g + og32 + tg;
#pragma unroll
    for (int k = 0; k < NOG; ++k) dst[NTG * k] = pick<LGTG>(acc, tg, k);
  } else if (g < g1 + gc) {
    const int c = g - g1;
    const int cbase = (n - m) & ~(OPW - 1);
    fir_corr32(acc, b, rf + cbase + OPW * c + og32, tg);
    reduce_tg<LGTG>(acc);
    const int i0 = cbase + OPW * c + og32 + tg;
    float* dst = w2abs + OPW * c + og32 + tg;
#pragma unroll
    for (int k = 0; k < NOG; ++k) dst[NTG * k] = (i0 + NTG * k < n - m) ? 0.f : pick<LGTG>(acc, tg, k);
  }
}

// stage 2:  z = (L_a w - L_z w2) / beta ; partial sums of rho.z (one per block)
//   warps [0, g1)      : zf = L_a w / beta                 (all outputs)
//   warps [g1, g1+gc)  : zb = -L_z w2 / beta               (outputs from cbase on)
//   so z = zf + zb where zb exists.  The two parts are produced by different
//   warps (and added by the consumer, k_pupd) because a warp that had to do
//   both convolutions would take twice as long as all the others.
template <int LGTG>
__global__ void k_prec2(const float* __restrict__ a, const float* __restrict__ b,
                        const float* __restrict__ w, const float* __restrict__ w2abs,
                        const float* __restrict__ rf, float* __restrict__ zf,
                        float* __restrict__ zb, const double* __restrict__ sc,
                        float* __restrict__ prz, int n, int m, int g1, int gc) {
  const int NTG = 1 << LGTG, NOG = 32 >> LGTG, OPW = 32 * NOG;
  const int lane = threadIdx.x & 31, tg = lane & (NTG - 1), og32 = (lane >> LGTG) << 5;
  const int wpb = blockDim.x >> 5;
  const int totw = gridDim.x * wpb;
  const float rbeta = 1.0f / (float)sc[SC_BETA];
  const int cbase = (n - m) & ~(OPW - 1);
  float term = 0.0f;
  for (int g = blockIdx.x * wpb + (threadIdx.x >> 5); g < g1 + gc; g += totw) {
    F32 acc;
#pragma unroll
    for (int k = 0; k < 32; ++k) acc.v[k] = 0.f;
    // (warp-uniform selection; one instance of the convolution code)
    const bool tail = (g >= g1);
    const int rel = OPW * (tail ? (g - g1) : g) + og32;  // first output, relative to the part
    const float* taps = tail ? b : a;
    const float* src = (tail ? w2abs : w) + rel;
    fir_conv32(acc, taps, src, tg, tail ? -1.0f : 1.0f);
    reduce_tg<LGTG>(acc);
    const int i0 = (tail ? cbase : 0) + rel + tg;
    float* dst = (tail ? zb : zf) + rel + tg;
#pragma unroll
    for (int k = 0; k < NOG; ++k) {
      const int i = i0 + NTG * k;
      const float z = (i < n) ? pick<LGTG>(acc, tg, k) * rbeta : 0.f;
      dst[NTG * k] = z;
      term = fmaf(rf[i], z, term);
    }
  }
  const float t = block_sum_f(term);
  if (threadIdx.x == 0) prz[blockIdx.x] = t;
}

// rz, bt ; p = z + bt p ; block partials of the (scaled) sum of squares of p
//   z = ZF + ZB (ZB: the second part of the preconditioner, rows >= cbase)
__global__ void k_pupd(const float* __restrict__ ZF, const float* __restrict__ ZB, int cbase,
                       float* __restrict__ PF, const float* __restrict__ prz, int nparts,
                       float* __restrict__ fs, const int* __restrict__ ic,
                       float* __restrict__ psqp, int n) {
  const float rz = partials_sum_f(prz, nparts);
  const int it = ic[IC_ITER];
  const int cg = ic[IC_CGIT];
  const float rz_prev = fs[FS_RZ0 + (it & 1)];
  const float scale = fs[FS_SCALE];
  float bt = 0.0f;
  if (cg > 0 && rz_prev > 0.0f) bt = rz / rz_prev;
  if (!(fabsf(bt) < 1.0e30f)) bt = 0.0f;
  const int tot = gridDim.x * blockDim.x;
  float sq = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += tot) {
    // (on a restart the old direction is not read at all: it may hold anything,
    // e.g. the leftovers of an earlier invocation that failed)
    float zz = ZF[i];
    if (i >= cbase) zz += ZB[i - cbase];
    const float p = (cg > 0) ? fmaf(bt, PF[i], zz) : zz;
    PF[i] = p;
    const float ps = p * scale;
    sq = fmaf(ps, ps, sq);
  }
  __syncthreads();
  sq = block_sum_f(sq);
  if (threadIdx.x == 0) {
    psqp[blockIdx.x] = sq;
    if (blockIdx.x == 0) fs[FS_RZ0 + ((it + 1) & 1)] = rz;
  }
}

// partial sums of p.q
__global__ void k_dot(const float* __restrict__ PF, const float* __restrict__ QF,
                      float* __restrict__ ppq, int n) {
  const int tot = gridDim.x * blockDim.x;
  float term = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += tot) term = fmaf(PF[i], QF[i], term);
  const float t = block_sum_f(term);
  if (threadIdx.x == 0) ppq[blockIdx.x] = t;
}

// alpha = rz/pq ; correction += alpha p ; residual -= alpha q ; block partials
// of max |residual| and of the (scaled) sum of squares.
//   selfdot: no k_dot kernel ran; every block computes the complete p.q itself
//   (same operations in the same order in every block, hence the same value).
//   For small n that is cheaper than one more kernel.
__global__ void k_upd(const float* __restrict__ PF, const float* __restrict__ QF,
                      float* __restrict__ DF, float* __restrict__ RF,
                      const float* __restrict__ ppq, int nparts, int selfdot,
                      float* __restrict__ fs, const int* __restrict__ ic,
                      float* __restrict__ pmaxr, float* __restrict__ psq, int n) {
  float pq;
  if (selfdot) {
    float term = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) term = fmaf(PF[i], QF[i], term);
    pq = block_sum_all_f(term);
  } else {
    pq = partials_sum_f(ppq, nparts);
  }
  const int it = ic[IC_ITER];
  const float rz = fs[FS_RZ0 + ((it + 1) & 1)];
  const float scale = fs[FS_SCALE];
  float al = rz / pq;
  // rz == pq == 0: the residual is exactly zero, nothing to do (not a breakdown)
  const bool zero = (rz == 0.0f) && (pq == 0.0f);
  const bool bad = !zero && (!(pq > 0.0f) || !(rz > 0.0f) || !(fabsf(al) < 1.0e30f));
  if (bad || zero) al = 0.0f;
  const int tot = gridDim.x * blockDim.x;
  float ar = 0.0f, sq = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += tot) {
    DF[i] = fmaf(al, PF[i], DF[i]);
    const float v = fmaf(-al, QF[i], RF[i]);
    RF[i] = v;
    ar = fmaxf(ar, clamp_abs_f(v));
    const float vs = v * scale;
    sq = fmaf(vs, vs, sq);
  }
  if (bad) ar = 3.0e38f;
  ar = block_max_f(ar);
  __syncthreads();
  sq = block_sum_f(sq);
  if (threadIdx.x == 0) {
    pmaxr[blockIdx.x] = ar; psq[blockIdx.x] = sq;
    if (blockIdx.x == 0) fs[FS_ALPHA] = al;
  }
}

// decision after a single precision iteration.  rr: recursive residual.
// The true residual of Y + DF differs from it because the products with T and
// the accumulation of DF are done in single precision; that difference is
// bounded (2-norm) by about eps * ||T|| * sum_k |alpha_k| ||p_k||, the length
// of the path walked since the last anchor.  The result is accepted without
// another anchor only if gapk times that bound is far below the tolerance.
//   soft: inside the fixed prefix - keep the books, decide nothing.  (The
//   iterations before the first anchor are all soft: the residual they start
//   from was computed in single precision, so they can only end in an anchor.)
__global__ void k_decide(const float* __restrict__ pmaxr, const float* __restrict__ psq, int uparts,
                         const float* __restrict__ psqp, int nparts, float* __restrict__ fs,
                         int* __restrict__ ic, int soft, float tolrel, float gapk,
                         float floork, int itcap, Flow flow) {
  const float rr = partials_max_f(pmaxr, uparts);
  __syncthreads();
  const float ssq = partials_sum_f(psq, uparts);
  __syncthreads();
  const float ssp = partials_sum_f(psqp, nparts);
  if (threadIdx.x == 0) {
    const int it = ic[IC_ITER] + 1;
    ic[IC_ITER] = it;
    ic[IC_CGIT] = ic[IC_CGIT] + 1;
    const float scale = fs[FS_SCALE];
    const float rr2 = sqrtf(ssq) / scale;
    const float path = fs[FS_PATH] + fabsf(fs[FS_ALPHA]) * (sqrtf(ssp) / scale);
    fs[FS_PATH] = path;
    fs[FS_RR] = rr;
    fs[FS_RR2] = rr2;
#ifdef PB_HIST
    if (it < 40) { fs[FS_COUNT + 2 * it] = rr; fs[FS_COUNT + 2 * it + 1] = rr2; }
#endif
    const float anchor2 = fs[FS_ANCHOR2];
    const float tol = tolrel * fs[FS_YMAX], tol2 = tolrel * fs[FS_Y2];
    const bool below = (rr <= 0.9f * tol) && (rr2 <= 0.9f * tol2);
    const float gap = gapk * 5.9604645e-8f * fs[FS_TNORM] * path;
    const bool gap_ok = (gap <= 0.1f * tol) && (gap <= 0.1f * tol2);
    const bool finite = (rr < 1.0e38f) && (rr2 < 1.0e38f) && (path < 1.0e38f);
    const bool diverging = finite && !below && (rr2 > 4.0f * anchor2);
    // diverging: drop the correction accumulated since the anchor and restart from it
    if (diverging) ic[IC_DISCARD] = 1;
    if (soft) {
      // the anchor of the prefix compares the true residual with this one
      fs[FS_RRREQ] = rr;
      fs[FS_RR2REQ] = rr2;
      if (!finite) ic[IC_STATUS] = ST_FAIL;
    } else {
      int status = ST_RUNNING;
      int cont = 1, anchor = 0;
      if (!finite || ic[IC_STATUS] == ST_FAIL) status = ST_FAIL;
      else if (below && gap_ok) status = ST_OK;
      else if (below || rr2 <= floork * anchor2 || diverging) {
        cont = 0; anchor = 1;
        fs[FS_RRREQ] = rr; fs[FS_RR2REQ] = rr2;
      }
      else if (it >= itcap) status = ST_FAIL;
      ic[IC_STATUS] = status;
      if (status != ST_RUNNING) { cont = 0; anchor = 0; }
      set_flow(ic, flow, cont, anchor, status == ST_FAIL ? 1 : 0);
    }
  }
}

// result of the iteration -> float output (correctly rounded sum of the parts)
__global__ void k_final(const ff* __restrict__ Y, const float* __restrict__ DF,
                        float* __restrict__ out, const float* __restrict__ fs,
                        const int* __restrict__ ic, int n) {
  if (ic[IC_STATUS] != ST_OK) return;
  const float inv = 1.0f / fs[FS_PSCALE];  // exact (power of two)
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const ff y = Y[i];
    const ff t = ff_two_sum(y.hi, DF[i] * inv);
    out[i] = t.hi + (t.lo + y.lo);
  }
}

// ----------------------------------------------------------------------------
// fallback: full recursion in double precision, one kernel per step.
//   Same in-place scheme as k_direct with K = n, arrays in global memory.
//   alpha for step k is read from sc[SC_ALPHA0 + (k&1)] and the next one is
//   written to the other slot by the thread that owns pairs 0 and 1.
// ----------------------------------------------------------------------------
__global__ void k_t2_init(const float* __restrict__ r, double* __restrict__ Mv,
                          double* __restrict__ S, double* __restrict__ sc,
                          int* __restrict__ ic, int n) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  const int L = n + 2;
  if (s < L) {
    Mv[s] = (s == 0) ? 0.0 : ((s <= n) ? (double)r[s - 1] : 1.0);
    S[s] = (s == 0) ? 1.0 : ((s < n) ? (double)r[s - 1] : ((s == n) ? 0.0 : 1.0));
  }
  if (s == 0) {
    sc[SC_ALPHA0] = -(double)r[0];
    ic[IC_T2BASE] = 0;
  }
}

__global__ void k_t2_step(double* __restrict__ Mv, double* __restrict__ S,
                          double* __restrict__ sc, const int* __restrict__ ic, int n, int jstep) {
  const int k = ic[IC_T2BASE] + jstep;
  if (k >= ic[IC_T2LIMIT]) return;
  const int L = n + 2;
  const double al = sc[SC_ALPHA0 + (k & 1)];
  const int s = blockIdx.x * blockDim.x + threadIdx.x + 1;  // pairs 1..L-1
  if (s < L) {
    int idx = s + k + 1;
    if (idx >= L) idx -= L;
    const double x = Mv[idx], y = S[s];
    double nx = fma(al, y, x);
    double ny = fma(al, x, y);
    if (s == n - k - 1 && k < n - 1) ny = 0.0;
    if (s == 1) {
      // this thread also owns pair 0 (pivot)
      const double y0 = S[0];
      const double x0 = Mv[k + 1];
      double ny0 = fma(al, x0, y0);
      if (n - k - 1 == 0 && k < n - 1) ny0 = 0.0;
      Mv[k + 1] = 0.0;
      S[0] = ny0;
      double an = 0.0;
      if (k + 1 < n) {
        an = neg_ratio_d(nx, ny0);  // no sign restriction: same as the recursion itself
        if (!(fabs(an) < 1.0e300)) an = 0.0;
      }
      sc[SC_ALPHA0 + ((k + 1) & 1)] = an;
    }
    Mv[idx] = nx;
    S[s] = ny;
  }
}

__global__ void k_t2_cond(int* __restrict__ ic, int n, int unroll,
                          cudaGraphConditionalHandle h) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const int b = ic[IC_T2BASE] + unroll;
    ic[IC_T2BASE] = b;
    ic[IC_T2RUNS] += 1;
    cudaGraphSetConditional(h, b < ic[IC_T2LIMIT] ? 1u : 0u);
  }
}

__global__ void k_t2_final(const double* __restrict__ Mv, float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float f = (float)Mv[i];
    if (!(fabsf(f) <= 3.0e38f)) f = 0.0f;
    out[i] = f;
  }
}

// ----------------------------------------------------------------------------
// very small n: the CPU baseline's recursion itself, operation by operation in
// single precision (explicitly rounded add/mul/div, no fused multiply-add, the
// inner product accumulated in the baseline's order), so the result is the
// baseline's result bit for bit.  For such n the baseline's rounding errors
// are so few that no other answer is reliably "at least as good".
//   blockDim.x = n rounded up to a multiple of 32 ; shared: Ya, Yb, P, R
//   (blockDim.x elements each)
// ----------------------------------------------------------------------------
__global__ void k_exact(const float* __restrict__ r, float* __restrict__ out, int n) {
  extern __shared__ float4 smem_ex[];  // (16 byte alignment for the wide loads below)
  const int th = blockDim.x;           // a multiple of 32, >= n
  float* Ya = reinterpret_cast<float*>(smem_ex);
  float* Yb = Ya + th;
  float* P = Yb + th;
  float* R = P + th;
  __shared__ float s_t0[32];           // slot 0: the current coefficient
  const int i = threadIdx.x;
  // The code below is free of divergent branches (they are expensive on this
  // device): every thread works on valid memory and selects what it keeps, and
  // all lanes of warp 0 do the sequential part that only lane 0 needs.
  const float ri = (i < n) ? r[i] : 0.0f;
  const float r0 = r[0];
  R[i] = ri;
  Ya[i] = (i == 0) ? -ri : 0.0f;
  Yb[i] = 0.0f;
  P[i] = (i == 0) ? __fmul_rn(ri, -ri) : 0.0f;  // the product of step 1
  if (i < 32) s_t0[i] = -r0;
  float t1 = 1.0f;  // warp 0 only
  __syncthreads();
  float* yc = Ya;
  float* yn = Yb;
  // Step k: P[0..k) holds the products r[k-q-1]*y[q] (formed at the end of the
  // previous step); they are added up in order and the new coefficient is
  // published; then every thread updates its element and forms its product for
  // the next step.
  for (int k = 1; k < n; ++k) {
    if (i < 32) {
      const float t0p = s_t0[0];
      t1 = __fmul_rn(__fsub_rn(1.0f, __fmul_rn(t0p, t0p)), t1);
      // the baseline's sequential sum (four elements per load)
      float t2 = 0.0f;
      const float4* P4 = reinterpret_cast<const float4*>(P);
      int q = 0;
      for (; q + 4 <= k; q += 4) {
        const float4 v = P4[q >> 2];
        t2 = __fadd_rn(t2, v.x);
        t2 = __fadd_rn(t2, v.y);
        t2 = __fadd_rn(t2, v.z);
        t2 = __fadd_rn(t2, v.w);
      }
      for (; q < k; ++q) t2 = __fadd_rn(t2, P[q]);
      s_t0[i] = __fdiv_rn(-__fadd_rn(R[k], t2), t1);
    }
    __syncthreads();
    const float t0 = s_t0[0];
    const int jp = k - i - 1, jr = k - i;
    const float upd = __fadd_rn(yc[i], __fmul_rn(t0, yc[jp < 0 ? 0 : jp]));
    const float yi = (i < k) ? upd : ((i == k) ? t0 : 0.0f);
    yn[i] = yi;
    P[i] = __fmul_rn(R[jr < 0 ? 0 : jr], yi);
    __syncthreads();
    float* tmp = yc; yc = yn; yn = tmp;
  }
  if (i < n) out[i] = yc[i];
}

// n == 1
__global__ void k_n1(const float* __restrict__ r, float* __restrict__ out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = -r[0];
}

// ----------------------------------------------------------------------------
// host side
// ----------------------------------------------------------------------------
struct State {
  bool ready = false;
  long long n = 0;
  int mode = 0;  // 0: n==1, 2: iterative, 3: exact replica of the recursion
  int K = 0, m = 0, msnap = 0;
  int lgM = 0;
  long long M = 0;
  bool use_fft = false;
  int itcap = 0;
  int mp = 0, woff = 0, g1 = 0, gc = 0;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  std::vector<void*> allocs;
  float *d_rbuf = nullptr, *d_r = nullptr, *d_out = nullptr;  // d_r = d_rbuf + roff
  size_t roff = 0;
  int lgtg = 3, opw = 128;  // FIR layout: tap groups, outputs per warp
  int sched1 = 0, sched2 = 1;  // fixed prefix: iterations before / after the first anchor
  // direct phase / preconditioner
  float *d_a = nullptr, *d_b = nullptr, *d_poly = nullptr, *d_RF = nullptr, *d_w = nullptr,
        *d_w2 = nullptr, *d_zf = nullptr, *d_zb = nullptr;
  // single precision iteration
  float *d_DF = nullptr, *d_PF = nullptr, *d_QF = nullptr, *d_CHF = nullptr,
        *d_fs = nullptr, *d_pa = nullptr, *d_pb = nullptr, *d_pc = nullptr, *d_pe = nullptr,
        *d_pf = nullptr;
  float2 *d_ZF = nullptr, *d_WF = nullptr, *d_WLF = nullptr, *d_WBF = nullptr;
  // float-float anchor
  ff *d_Y = nullptr, *d_QX = nullptr, *d_CHX = nullptr;
  cff *d_ZX = nullptr, *d_WX = nullptr, *d_WLX = nullptr, *d_WBX = nullptr;
  double* d_sc = nullptr;
  // fallback
  double *d_t2Mv = nullptr, *d_t2S = nullptr;
  int* d_ic = nullptr;
};
static State g;

#define PB_CK(call)                         \
  do {                                      \
    cudaError_t pb_e_ = (call);             \
    if (pb_e_ != cudaSuccess) return 1;     \
  } while (0)

template <class T>
static int dev_alloc(T** p, size_t count) {
  void* q = nullptr;
  if (count == 0) count = 1;
  if (cudaMalloc(&q, count * sizeof(T)) != cudaSuccess) return 1;
  g.allocs.push_back(q);
  *p = (T*)q;
  return 0;
}
template <class T>
static int dev_alloc_zero(T** p, size_t count) {
  if (dev_alloc(p, count)) return 1;
  if (count == 0) count = 1;
  return cudaMemset(*p, 0, count * sizeof(T)) == cudaSuccess ? 0 : 1;
}

static void release_all() {
  if (g.stream) {
    // a failed setup may leave the stream capturing
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(g.stream, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
      cudaGraph_t tmp = nullptr;
      cudaStreamEndCapture(g.stream, &tmp);
    }
    cudaGetLastError();
  }
  if (g.exec) { cudaGraphExecDestroy(g.exec); g.exec = nullptr; }
  if (g.graph) { cudaGraphDestroy(g.graph); g.graph = nullptr; }
  for (void* p : g.allocs) cudaFree(p);
  g.allocs.clear();
  if (g.stream) { cudaStreamDestroy(g.stream); g.stream = nullptr; }
  g.ready = false;
}

static inline unsigned nblocks(long long count, int bs) {
  long long b = (count + bs - 1) / bs;
  if (b < 1) b = 1;
  return (unsigned)b;
}
static inline int launch_ok() { return cudaGetLastError() == cudaSuccess ? 0 : 1; }

// ---- FFT launch helpers ----
template <class T> struct FftTabs;
template <> struct FftTabs<float> {
  static const float2* W() { return g.d_WF; }
  static const float2* WL() { return g.d_WLF; }
  static const float2* WB() { return g.d_WBF; }
  static const float* CH() { return g.d_CHF; }
  static int rmax() { return kRadixF; }
};
template <> struct FftTabs<ff> {
  static const cff* W() { return g.d_WX; }
  static const cff* WL() { return g.d_WLX; }
  static const cff* WB() { return g.d_WBX; }
  static const ff* CH() { return g.d_CHX; }
  static int rmax() { return kRadixX; }
};

template <class T>
static int launch_fft_pass(const typename CplxOf<T>::type* in, typename CplxOf<T>::type* out,
                           int R, int inv, int lgStride) {
  const typename CplxOf<T>::type* W = FftTabs<T>::W();
  const long long ngroups = g.M >> R;
  const int bs = 64;
  const unsigned nb = nblocks(ngroups, bs);
  cudaStream_t st = g.stream;
#define PB_FFT_CASE(RR)                                                                 \
  case RR:                                                                              \
    if (inv) k_fft<T, RR, 1><<<nb, bs, 0, st>>>(in, out, W, lgStride, g.lgM, ngroups);  \
    else k_fft<T, RR, 0><<<nb, bs, 0, st>>>(in, out, W, lgStride, g.lgM, ngroups);      \
    break;
  switch (R) {
    PB_FFT_CASE(1)
    PB_FFT_CASE(2)
    PB_FFT_CASE(3)
    PB_FFT_CASE(4)
    default: return 1;
  }
#undef PB_FFT_CASE
  return launch_ok();
}
template <class T>
static int launch_fft_local(const typename CplxOf<T>::type* in, typename CplxOf<T>::type* out,
                            int inv, int lgC) {
  typedef typename CplxOf<T>::type C;
  const int Cn = 1 << lgC;
  const int threads = Cn / 2;
  const unsigned nb = (unsigned)(g.M >> lgC);
  const size_t smem = (size_t)(Cn + Cn / 2) * sizeof(C);
  if (inv) k_fft_local<T, 1><<<nb, threads, smem, g.stream>>>(in, out, FftTabs<T>::WL(), lgC);
  else k_fft_local<T, 0><<<nb, threads, smem, g.stream>>>(in, out, FftTabs<T>::WL(), lgC);
  return launch_ok();
}
static inline int fft_local_lg() { return g.lgM < kLocalLg ? g.lgM : kLocalLg; }
// radices of the global passes, top stage first: as few passes of at most rmax
// (<= 4) stages as possible.  (Radix 32 was measured to be slower: a pass costs
// a kernel launch plus a per-thread latency that grows with 2^R, because loads
// are serialised per warp.)
static std::vector<int> fft_radices(int rmax) {
  std::vector<int> v;
  if (rmax > 4) rmax = 4;
  if (rmax < 1) rmax = 1;
  int rem = g.lgM - fft_local_lg();
  const int passes = (rem + rmax - 1) / rmax;
  for (int i = 0; i < passes; ++i) {
    const int R = (rem + (passes - i) - 1) / (passes - i);
    v.push_back(R);
    rem -= R;
  }
  return v;
}
// forward: src (natural order) -> Z (bit reversed)
template <class T>
static int launch_fft_forward(const typename CplxOf<T>::type* src, typename CplxOf<T>::type* Z) {
  const std::vector<int> rad = fft_radices(FftTabs<T>::rmax());
  int lgL = g.lgM;
  const typename CplxOf<T>::type* in = src;
  for (size_t i = 0; i < rad.size(); ++i) {
    if (launch_fft_pass<T>(in, Z, rad[i], 0, lgL - rad[i])) return 1;
    in = Z;
    lgL -= rad[i];
  }
  return launch_fft_local<T>(in, Z, 0, lgL);
}

// dst = T * src for real sequences packed as M complex numbers: global forward
// passes, fused middle kernel, global inverse passes (Z is the work array)
template <class T>
static int launch_matvec(const typename CplxOf<T>::type* src, typename CplxOf<T>::type* Z,
                         typename CplxOf<T>::type* dst) {
  typedef typename CplxOf<T>::type C;
  const std::vector<int> rad = fft_radices(FftTabs<T>::rmax());
  const int lgC = fft_local_lg();
  int lgL = g.lgM;
  const C* in = src;
  for (size_t i = 0; i < rad.size(); ++i) {
    if (launch_fft_pass<T>(in, Z, rad[i], 0, lgL - rad[i])) return 1;
    in = Z;
    lgL -= rad[i];
  }
  {
    const int Cn = 1 << lgC;
    const int b = g.lgM - lgC;
    const unsigned nb = (b >= 1) ? ((1u << (b - 1)) + 1u) : 1u;
    const size_t smem = (size_t)(2 * Cn + Cn / 2) * sizeof(C);
    k_fft_mid<T><<<nb, Cn, smem, g.stream>>>(in, rad.empty() ? dst : Z, FftTabs<T>::WL(),
                                              FftTabs<T>::WB(), FftTabs<T>::CH(), lgC, g.lgM);
    if (launch_ok()) return 1;
  }
  for (size_t ii = rad.size(); ii-- > 0;) {
    lgL += rad[ii];
    if (launch_fft_pass<T>(Z, ii == 0 ? dst : Z, rad[ii], 1, lgL - rad[ii])) return 1;
  }
  return 0;
}

// QX = T * Y (float-float)
static int launch_matvec_x() {
  if (!g.use_fft) {
    const int n = (int)g.n;
    k_matvec_direct_x<<<nblocks(n, 8), 256, 0, g.stream>>>(g.d_r - 1, g.d_Y, g.d_QX, n);
    return launch_ok();
  }
  return launch_matvec<ff>(reinterpret_cast<const cff*>(g.d_Y), g.d_ZX, reinterpret_cast<cff*>(g.d_QX));
}
// QF = T * PF (single precision)
static int launch_matvec_f() {
  cudaStream_t st = g.stream;
  const int n = (int)g.n;
  if (!g.use_fft) {
    k_matvec_direct<<<nblocks(n, 8), 256, 0, st>>>(g.d_r - 1, g.d_PF, g.d_QF, n);
    return launch_ok();
  }
  return launch_matvec<float>(reinterpret_cast<const float2*>(g.d_PF), g.d_ZF, reinterpret_cast<float2*>(g.d_QF));
}

static int add_cond_node(cudaGraph_t parent, const cudaGraphNode_t* deps, size_t ndeps,
                         cudaGraphConditionalHandle h, cudaGraphConditionalNodeType type,
                         cudaGraphNode_t* node_out, cudaGraph_t* body_out) {
  cudaGraphNodeParams* p = (cudaGraphNodeParams*)calloc(1, sizeof(cudaGraphNodeParams));
  if (!p) return 1;
  p->type = cudaGraphNodeTypeConditional;
  p->conditional.handle = h;
  p->conditional.type = type;
  p->conditional.size = 1;
  const cudaError_t e = cudaGraphAddNode(node_out, parent, deps, nullptr, ndeps, p);
  if (e == cudaSuccess) *body_out = p->conditional.phGraph_out[0];
  free(p);
  return e == cudaSuccess ? 0 : 1;
}

// append a conditional node after everything captured so far on g.stream and
// make later captured launches depend on it
static int capture_cond(cudaGraphConditionalHandle h, cudaGraphConditionalNodeType type,
                        cudaGraph_t* body_out) {
  cudaStreamCaptureStatus status;
  cudaGraph_t cur = nullptr;
  const cudaGraphNode_t* d = nullptr;
  size_t nd = 0;
  PB_CK(cudaStreamGetCaptureInfo(g.stream, &status, nullptr, &cur, &d, nullptr, &nd));
  if (status != cudaStreamCaptureStatusActive) return 1;
  std::vector<cudaGraphNode_t> deps(d, d + nd);
  cudaGraphNode_t node;
  if (add_cond_node(cur, deps.data(), deps.size(), h, type, &node, body_out)) return 1;
  PB_CK(cudaStreamUpdateCaptureDependencies(g.stream, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
  return 0;
}
static int begin_capture(cudaGraph_t graph) {
  PB_CK(cudaStreamBeginCaptureToGraph(g.stream, graph, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
  return 0;
}
static int end_capture() {
  cudaGraph_t tmp;
  PB_CK(cudaStreamEndCapture(g.stream, &tmp));
  return 0;
}

struct Sizes { int nparts, fparts, pparts, uparts, ubs; };  // partial counts: vector kernels, k_res0, k_prec2, k_upd (+ its block size)

template <int LG>
static void launch_prec(const Sizes& z, unsigned nb1, int fbs, float* w0, float* w2abs, int n) {
  cudaStream_t st = g.stream;
  k_prec1<LG><<<nb1, fbs, 0, st>>>(g.d_a, g.d_b, g.d_RF, w0, w2abs, n, g.m, g.g1, g.gc);
  k_prec2<LG><<<z.pparts, fbs, 0, st>>>(g.d_a, g.d_b, w0, w2abs, g.d_RF, g.d_zf, g.d_zb, g.d_sc, g.d_pc,
                                        n, g.m, g.g1, g.gc);
}
template <int LG>
static void launch_res0(const Sizes& z, int fbs, int n) {
  k_res0<LG><<<z.fparts, fbs, 0, g.stream>>>(g.d_a, g.d_r, g.d_poly, g.K, g.d_RF, g.d_Y, g.d_DF, g.d_pa,
                                             g.d_pb, g.d_pc, g.d_pe, g.d_pf, g.d_fs, g.d_ic, n, g.g1);
}

// one single precision PCG iteration followed by its decision
static int capture_iteration(const Flow& flow, int soft, const Sizes& z) {
  cudaStream_t st = g.stream;
  const int n = (int)g.n;
  float* w0 = g.d_w + g.woff;
  float* w2abs = g.d_w2 + g.woff;
  const int fbs = 128;
  const unsigned nb1 = nblocks(g.g1 + g.gc, fbs / 32);
  switch (g.lgtg) {
    case 2: launch_prec<2>(z, nb1, fbs, w0, w2abs, n); break;
    case 3: launch_prec<3>(z, nb1, fbs, w0, w2abs, n); break;
    case 4: launch_prec<4>(z, nb1, fbs, w0, w2abs, n); break;
    default: return 1;
  }
  k_pupd<<<z.nparts, kRedBlock, 0, st>>>(g.d_zf, g.d_zb, (n - g.m) & ~(g.opw - 1), g.d_PF, g.d_pc, z.pparts,
                                         g.d_fs, g.d_ic, g.d_pe, n);
  if (launch_ok()) return 1;
  if (launch_matvec_f()) return 1;
  const int selfdot = (n <= kSelfDotN) ? 1 : 0;
  if (!selfdot) k_dot<<<z.nparts, kRedBlock, 0, st>>>(g.d_PF, g.d_QF, g.d_pc, n);
  k_upd<<<z.uparts, z.ubs, 0, st>>>(g.d_PF, g.d_QF, g.d_DF, g.d_RF, g.d_pc, z.nparts, selfdot,
                                    g.d_fs, g.d_ic, g.d_pa, g.d_pb, n);
  k_decide<<<1, 32, 0, st>>>(g.d_pa, g.d_pb, z.uparts, g.d_pe, z.nparts, g.d_fs, g.d_ic, soft,
                             (float)kTolRel, kGap, kFloor, g.itcap, flow);
  return launch_ok();
}

// anchor: fold the correction, true residual in float-float, decision
//   skip_add: there is no correction yet (anchor right after the start)
static int capture_anchor(const Flow& flow, int soft, const Sizes& z, bool skip_add = false) {
  cudaStream_t st = g.stream;
  const int n = (int)g.n;
  if (!skip_add) k_anchor_add<<<nblocks(n, 256), 256, 0, st>>>(g.d_Y, g.d_DF, g.d_fs, g.d_ic, n);
  if (launch_ok()) return 1;
  if (launch_matvec_x()) return 1;
  k_res<<<z.nparts, kRedBlock, 0, st>>>(g.d_r, g.d_QX, g.d_Y, g.d_RF, g.d_pa, g.d_pb, g.d_pc, g.d_pe, g.d_fs, n);
  k_decide_anchor<<<1, 32, 0, st>>>(g.d_pa, g.d_pb, g.d_pc, g.d_pe, z.nparts, g.d_fs, g.d_ic,
                                    (float)kTolRel, g.itcap, soft, flow);
  return launch_ok();
}

static int build_graph() {
  cudaStream_t st = g.stream;
  const int n = (int)g.n;
  PB_CK(cudaGraphCreate(&g.graph, 0));
  if (g.mode == 0) {
    if (begin_capture(g.graph)) return 1;
    k_n1<<<1, 32, 0, st>>>(g.d_r, g.d_out);
    if (launch_ok() || end_capture()) return 1;
    PB_CK(cudaGraphInstantiate(&g.exec, g.graph, 0));
    return 0;
  }
  if (g.mode == 3) {
    int th = ((n + 31) / 32) * 32;
    if (th > 1024) return 1;
    if (begin_capture(g.graph)) return 1;
    k_exact<<<1, th, (size_t)4 * th * sizeof(float), st>>>(g.d_r, g.d_out, n);
    if (launch_ok() || end_capture()) return 1;
    PB_CK(cudaGraphInstantiate(&g.exec, g.graph, 0));
    return 0;
  }
  // ---- mode 2: direct phase + mixed precision PCG (+ fallback) ----
  const int L = g.K + 2;  // one thread per pair
  if (L > 1024) return 1;
  if (L < 131) return 1;  // (mode 2 means n >= 129; the kernel needs a full warp and K >= 3)
  const size_t dir_smem = (size_t)(4 * L + 2 + 64 + 4) * sizeof(float);
  Flow flow;
  cudaGraphConditionalHandle hT2;
  PB_CK(cudaGraphConditionalHandleCreate(&flow.in2a, g.graph, 0, cudaGraphCondAssignDefault));
  PB_CK(cudaGraphConditionalHandleCreate(&flow.in2b, g.graph, 0, cudaGraphCondAssignDefault));
  PB_CK(cudaGraphConditionalHandleCreate(&flow.out, g.graph, 0, cudaGraphCondAssignDefault));
  PB_CK(cudaGraphConditionalHandleCreate(&flow.fall, g.graph, 0, cudaGraphCondAssignDefault));
  PB_CK(cudaGraphConditionalHandleCreate(&flow.slow, g.graph, 0, cudaGraphCondAssignDefault));
  PB_CK(cudaGraphConditionalHandleCreate(&hT2, g.graph, 1, cudaGraphCondAssignDefault));
  const int bs = 256;
  const int nb_n = (int)nblocks(n, bs);
  Sizes z;
  z.nparts = (int)nblocks(n, kRedBlock);
  if (z.nparts > kPartMax) z.nparts = kPartMax;  // threads then loop over several elements
  const int fbs = 128;
  z.fparts = (int)nblocks(g.g1, fbs / 32);
  if (z.fparts > kPartMax) z.fparts = kPartMax;
  z.pparts = (int)nblocks(g.g1 + g.gc, fbs / 32);
  if (z.pparts > kPartMax) z.pparts = kPartMax;
  // k_upd: big blocks when it computes p.q itself (fewer elements per thread)
  z.ubs = (n <= kSelfDotN) ? kSelfDotBlock : kRedBlock;
  z.uparts = (int)nblocks(n, z.ubs);
  if (z.uparts > kPartMax) z.uparts = kPartMax;
  cudaGraph_t slowBody = nullptr, in2Body = nullptr, outBody = nullptr, in2bBody = nullptr,
              fallBody = nullptr, t2Body = nullptr;

  if (begin_capture(g.graph)) return 1;
  k_direct<<<1, L, dir_smem, st>>>(g.d_r, g.K, g.msnap, g.d_a, g.d_b, g.d_poly, g.d_sc, g.d_fs, g.d_ic);
  if (g.use_fft) {
    // symbol of the circulant embedding (float-float, and rounded to float)
    k_packc<<<nblocks(g.M, 256), 256, 0, st>>>(g.d_r, g.n, g.M, g.d_ZX);
    if (launch_ok()) return 1;
    if (launch_fft_forward<ff>(g.d_ZX, g.d_ZX)) return 1;
    k_symbol<<<nblocks(g.M + 1, 64), 64, 0, st>>>(g.d_ZX, g.d_WX, g.d_CHX, g.d_CHF, g.M, g.lgM,
                                                 (float)(1.0 / (double)g.M));
  }
  switch (g.lgtg) {
    case 2: launch_res0<2>(z, fbs, n); break;
    case 3: launch_res0<3>(z, fbs, n); break;
    case 4: launch_res0<4>(z, fbs, n); break;
    default: return 1;
  }
  k_decide0<<<1, 32, 0, st>>>(g.d_pa, g.d_pb, g.d_pc, g.d_pe, g.d_pf, z.fparts, g.d_sc, g.d_fs, g.d_ic,
                              (float)kTolRel, flow);
  if (launch_ok()) return 1;
  // fixed prefix: sched1 iterations, anchor, sched2 (>= 1) iterations; only the
  // last decision is a real one
  for (int i = 0; i < g.sched1; ++i)
    if (capture_iteration(flow, 1, z)) return 1;
  if (capture_anchor(flow, 1, z, g.sched1 == 0)) return 1;
  for (int i = 0; i < g.sched2; ++i)
    if (capture_iteration(flow, i == g.sched2 - 1 ? 0 : 1, z)) return 1;
  // the result, if the prefix was enough (k_final does nothing otherwise) ...
  k_final<<<nb_n, bs, 0, st>>>(g.d_Y, g.d_DF, g.d_out, g.d_fs, g.d_ic, n);
  if (launch_ok()) return 1;
  // ... and whatever the prefix did not cover.  (One conditional node, placed
  // last: each one on the common path was measured to cost several microseconds.)
  if (capture_cond(flow.slow, cudaGraphCondTypeIf, &slowBody)) return 1;
  if (end_capture()) return 1;

  // slow path: more iterations; anchors, each followed by iterations; the
  // result; the fallback if the iteration failed
  if (begin_capture(slowBody)) return 1;
  if (capture_cond(flow.in2a, cudaGraphCondTypeWhile, &in2Body)) return 1;
  if (capture_cond(flow.out, cudaGraphCondTypeWhile, &outBody)) return 1;
  k_final<<<nb_n, bs, 0, st>>>(g.d_Y, g.d_DF, g.d_out, g.d_fs, g.d_ic, n);
  if (launch_ok()) return 1;
  if (capture_cond(flow.fall, cudaGraphCondTypeIf, &fallBody)) return 1;
  if (end_capture()) return 1;

  if (begin_capture(in2Body)) return 1;
  if (capture_iteration(flow, 0, z)) return 1;
  if (end_capture()) return 1;

  if (begin_capture(outBody)) return 1;
  if (capture_anchor(flow, 0, z)) return 1;
  if (capture_cond(flow.in2b, cudaGraphCondTypeWhile, &in2bBody)) return 1;
  if (end_capture()) return 1;
  if (begin_capture(in2bBody)) return 1;
  if (capture_iteration(flow, 0, z)) return 1;
  if (end_capture()) return 1;

  // ---- fallback body: init, WHILE(step x unroll), final ----
  if (begin_capture(fallBody)) return 1;
  k_t2_init<<<nblocks(n + 2, bs), bs, 0, st>>>(g.d_r, g.d_t2Mv, g.d_t2S, g.d_sc, g.d_ic, n);
  if (launch_ok()) return 1;
  if (capture_cond(hT2, cudaGraphCondTypeWhile, &t2Body)) return 1;
  k_t2_final<<<nb_n, bs, 0, st>>>(g.d_t2Mv, g.d_out, n);
  if (launch_ok() || end_capture()) return 1;

  if (begin_capture(t2Body)) return 1;
  for (int j = 0; j < kT2Unroll; ++j)
    k_t2_step<<<nblocks(n + 1, bs), bs, 0, st>>>(g.d_t2Mv, g.d_t2S, g.d_sc, g.d_ic, n, j);
  k_t2_cond<<<1, 32, 0, st>>>(g.d_ic, n, kT2Unroll, hT2);
  if (launch_ok() || end_capture()) return 1;

  PB_CK(cudaGraphInstantiate(&g.exec, g.graph, 0));
  return 0;
}

static int setup_impl(const pb_problem_shape* shape) {
  if (!shape || shape->abi_version != PB_SOLVE_ABI_VERSION) return 1;
  if (shape->dimension_count != 1 || shape->buffer_count != 2 || !shape->dimensions) return 1;
  const long long n = shape->dimensions[0];
  if (n <= 0 || n > 2147483647LL - 4096) return 1;
  g.n = n;
  PB_CK(cudaStreamCreate(&g.stream));
  const size_t sn = (size_t)n;
  if (n == 1) {
    g.mode = 0;
  } else if (n <= kExactN) {
    g.mode = 3;
  } else {
    g.mode = 2;
    if (n - 1 <= kFullCap || n - 1 <= kMCap) { g.K = (int)n; g.m = (int)n - 1; g.msnap = g.m; }
    else { g.K = kMCap; g.m = kMCap; g.msnap = g.K + 1; }
    // fixed prefix: what inputs of the published kind need at this size (lower
    // bounds; the WHILE nodes add what is missing)
    if (g.K == (int)n) { g.sched1 = 0; g.sched2 = 1; }
    else if (n <= kSchedSmallN) { g.sched1 = kSched1Small; g.sched2 = kSched2Small; }
    else { g.sched1 = kSched1; g.sched2 = kSched2; }
    if (g.sched2 < 1) g.sched2 = 1;
    // FIR geometry: mp = padded number of taps (128 or 256)
    g.lgtg = (g.m + 1 <= 128) ? 2 : ((g.m + 1 <= 256) ? 3 : 4);
    g.mp = 32 << g.lgtg;
    if (g.m + 1 > g.mp) return 1;
    g.opw = 32 * (32 >> g.lgtg);
    g.woff = g.mp + 64;
  }
  // input buffer with zero padding in front (the FIR kernels read before r[0])
  g.roff = (g.mode == 2) ? (size_t)g.woff : 0;
  if (dev_alloc_zero(&g.d_rbuf, g.roff + sn + 512)) return 1;
  g.d_r = g.d_rbuf + g.roff;
  if (g.mode == 2) {
    // R(0) = 1 in front of the input: d_r - 1 is then the sequence R(0..n-1)
    // (the kernels that rely on the zero padding never use this element with a
    // nonzero weight)
    const float one = 1.0f;
    PB_CK(cudaMemcpy(g.d_r - 1, &one, sizeof(float), cudaMemcpyHostToDevice));
  }
  if (dev_alloc_zero(&g.d_out, sn)) return 1;
  if (dev_alloc_zero(&g.d_sc, SC_COUNT)) return 1;
  if (dev_alloc_zero(&g.d_ic, IC_COUNT)) return 1;
  if (g.mode == 2) {
    g.use_fft = n > kDirectMatvecN;
    long long cap = n / 32;
    if (cap < 24) cap = 24;
    if (cap > 500) cap = 500;
    g.itcap = (int)cap;
    const size_t opw = (size_t)g.opw;
    const size_t nG = ((sn + opw - 1) / opw) * opw;  // outputs are produced in groups of opw
    g.g1 = (int)(nG / opw);
    g.gc = (int)((nG - (size_t)(((int)n - g.m) & ~(g.opw - 1))) / opw);
    // buffers whose zero padding the FIR kernels rely on (never overwritten)
    if (dev_alloc_zero(&g.d_a, (size_t)g.mp + 64)) return 1;
    if (dev_alloc_zero(&g.d_b, (size_t)g.mp + 64)) return 1;
    if (dev_alloc_zero(&g.d_RF, nG + (size_t)g.mp + 64)) return 1;
    if (dev_alloc_zero(&g.d_w, (size_t)g.woff + nG + 64)) return 1;
    if (dev_alloc_zero(&g.d_w2, (size_t)g.woff + (size_t)g.gc * opw + 64)) return 1;
    if (dev_alloc_zero(&g.d_zf, nG + 64)) return 1;
    if (dev_alloc_zero(&g.d_zb, (size_t)g.gc * opw + 64)) return 1;
    if (dev_alloc_zero(&g.d_poly, (size_t)g.K)) return 1;
    if (dev_alloc_zero(&g.d_DF, sn)) return 1;
    if (dev_alloc_zero(&g.d_fs, FS_COUNT + 96)) return 1;
    {
      const float one = 1.0f;  // problem scale until the direct phase sets it
      PB_CK(cudaMemcpy(g.d_fs + FS_PSCALE, &one, sizeof(float), cudaMemcpyHostToDevice));
    }
    if (dev_alloc_zero(&g.d_pa, kPartMax)) return 1;
    if (dev_alloc_zero(&g.d_pb, kPartMax)) return 1;
    if (dev_alloc_zero(&g.d_pc, kPartMax)) return 1;
    if (dev_alloc_zero(&g.d_pe, kPartMax)) return 1;
    if (dev_alloc_zero(&g.d_pf, kPartMax)) return 1;
    if (dev_alloc_zero(&g.d_t2Mv, sn + 2)) return 1;
    if (dev_alloc_zero(&g.d_t2S, sn + 2)) return 1;
    {
      int lg = 1;
      while ((1LL << lg) < n) ++lg;
      g.lgM = lg;
      g.M = 1LL << lg;
      const size_t M = (size_t)g.M, N = 2 * M;
      // Y (and PF) are read as M complex numbers by the first FFT pass: their
      // tails [n, N) stay zero
      if (dev_alloc_zero(&g.d_Y, N)) return 1;
      if (dev_alloc_zero(&g.d_QX, N)) return 1;
      if (dev_alloc_zero(&g.d_ZX, M)) return 1;
      if (dev_alloc_zero(&g.d_WX, M)) return 1;
      if (dev_alloc_zero(&g.d_CHX, M + 1)) return 1;
      if (dev_alloc_zero(&g.d_CHF, M + 1)) return 1;
      // twiddle tables (shape dependent only)
      std::vector<cff> wx(M);
      std::vector<float2> wf(g.use_fft ? M : 1);
      const long double step = 3.14159265358979323846264338327950288L / (long double)g.M;  // 2 pi / N
      for (size_t j = 0; j < M; ++j) {
        const long double ang = step * (long double)j;
        const long double c = cosl(ang), sn2 = -sinl(ang);
        wx[j].x.hi = (float)c;
        wx[j].x.lo = (float)(c - (long double)wx[j].x.hi);
        wx[j].y.hi = (float)sn2;
        wx[j].y.lo = (float)(sn2 - (long double)wx[j].y.hi);
        if (g.use_fft) { wf[j].x = wx[j].x.hi; wf[j].y = wx[j].y.hi; }
      }
      PB_CK(cudaMemcpy(g.d_WX, wx.data(), M * sizeof(cff), cudaMemcpyHostToDevice));
      // the same table in bit reversed order (for the fused middle kernel)
      {
        std::vector<cff> wb(M);
        for (size_t pidx = 0; pidx < M; ++pidx) {
          size_t rv = 0;
          for (int bit = 0; bit < g.lgM; ++bit) rv |= ((pidx >> bit) & 1) << (g.lgM - 1 - bit);
          wb[pidx] = wx[rv];
        }
        if (dev_alloc_zero(&g.d_WBX, M)) return 1;
        PB_CK(cudaMemcpy(g.d_WBX, wb.data(), M * sizeof(cff), cudaMemcpyHostToDevice));
        if (g.use_fft) {
          std::vector<float2> wbf(M);
          for (size_t pidx = 0; pidx < M; ++pidx) { wbf[pidx].x = wb[pidx].x.hi; wbf[pidx].y = wb[pidx].y.hi; }
          if (dev_alloc_zero(&g.d_WBF, M)) return 1;
          PB_CK(cudaMemcpy(g.d_WBF, wbf.data(), M * sizeof(float2), cudaMemcpyHostToDevice));
        }
      }
      // compact tables for the shared-memory stages: exp(-2 pi i j / C), j < C/2
      const int lgC = fft_local_lg();
      const size_t hC = (size_t)1 << (lgC > 0 ? lgC - 1 : 0);
      const size_t stepC = (size_t)1 << (g.lgM + 1 - lgC);
      std::vector<cff> wlx(hC);
      std::vector<float2> wlf(hC);
      for (size_t j = 0; j < hC; ++j) {
        wlx[j] = wx[j * stepC];
        wlf[j].x = wlx[j].x.hi;
        wlf[j].y = wlx[j].y.hi;
      }
      if (dev_alloc_zero(&g.d_WLX, hC)) return 1;
      PB_CK(cudaMemcpy(g.d_WLX, wlx.data(), hC * sizeof(cff), cudaMemcpyHostToDevice));
      if (g.use_fft) {
        if (dev_alloc_zero(&g.d_WLF, hC)) return 1;
        PB_CK(cudaMemcpy(g.d_WLF, wlf.data(), hC * sizeof(float2), cudaMemcpyHostToDevice));
        if (dev_alloc_zero(&g.d_PF, N)) return 1;
        if (dev_alloc_zero(&g.d_QF, N)) return 1;
        if (dev_alloc_zero(&g.d_ZF, M)) return 1;
        if (dev_alloc_zero(&g.d_WF, M)) return 1;
        PB_CK(cudaMemcpy(g.d_WF, wf.data(), M * sizeof(float2), cudaMemcpyHostToDevice));
      } else {
        if (dev_alloc_zero(&g.d_PF, sn)) return 1;
        if (dev_alloc_zero(&g.d_QF, sn)) return 1;
      }
    }
  }
  if (build_graph()) return 1;
  // Warm-up launches on synthetic data (no problem data exists yet): they load
  // every kernel and the graph machinery so that the first real invocation does
  // not pay one-time costs.  The results are discarded.
  {
    std::vector<float> wr(sn);
    for (long long i = 0; i < n; ++i) {
      const double d = (double)(i + 1);
      wr[(size_t)i] = (float)(((i & 1) ? -0.25 : 0.25) / (d * d));
    }
    int lim = (int)n;
    PB_CK(cudaMemcpy(g.d_ic + IC_T2LIMIT, &lim, sizeof(int), cudaMemcpyHostToDevice));
    PB_CK(cudaMemcpy(g.d_r, wr.data(), sn * sizeof(float), cudaMemcpyHostToDevice));
    PB_CK(cudaGraphLaunch(g.exec, g.stream));
    PB_CK(cudaStreamSynchronize(g.stream));
    if (g.mode == 2) {
      // exercise the fallback branch for a few steps only
      lim = (int)(n < 2 * kT2Unroll ? n : 2 * kT2Unroll);
      PB_CK(cudaMemcpy(g.d_ic + IC_T2LIMIT, &lim, sizeof(int), cudaMemcpyHostToDevice));
      wr[0] = 1.0f;
      PB_CK(cudaMemcpy(g.d_r, wr.data(), sn * sizeof(float), cudaMemcpyHostToDevice));
      PB_CK(cudaGraphLaunch(g.exec, g.stream));
      PB_CK(cudaStreamSynchronize(g.stream));
      lim = (int)n;
      PB_CK(cudaMemcpy(g.d_ic + IC_T2LIMIT, &lim, sizeof(int), cudaMemcpyHostToDevice));
    }
    PB_CK(cudaMemset(g.d_r, 0, sn * sizeof(float)));
    PB_CK(cudaMemset(g.d_out, 0, sn * sizeof(float)));
    PB_CK(cudaDeviceSynchronize());
    PB_CK(cudaGetLastError());
  }
  g.ready = true;
  return 0;
}

}  // namespace pbsol

extern "C" int pb_setup(const pb_problem_shape* shape) {
  using namespace pbsol;
  if (g.ready || !g.allocs.empty() || g.stream) release_all();
  if (setup_impl(shape)) {
    release_all();
    return 1;
  }
  return 0;
}

extern "C" int pb_upload(const pb_problem* problem, pb_buffer* buffers) {
  using namespace pbsol;
  if (!g.ready || !problem || !buffers) return 1;
  if (problem->abi_version != PB_SOLVE_ABI_VERSION || problem->dimension_count != 1 ||
      problem->buffer_count != 2 || !problem->dimensions)
    return 1;
  if (problem->dimensions[0] != g.n) return 1;
  const size_t bytes = (size_t)g.n * sizeof(float);
  // buffer 0 is the only initialized buffer (buffer 1 is the output)
  if (!buffers[0].data || buffers[0].nbytes < bytes) return 1;
  PB_CK(cudaMemcpy(g.d_r, buffers[0].data, bytes, cudaMemcpyHostToDevice));
  return 0;
}

extern "C" int pb_compute(void) {
  using namespace pbsol;
  if (!g.ready) return 1;
  PB_CK(cudaGraphLaunch(g.exec, g.stream));
  PB_CK(cudaStreamSynchronize(g.stream));
  PB_CK(cudaGetLastError());
  return 0;
}

extern "C" int pb_download(pb_buffer* buffers) {
  using namespace pbsol;
  if (!g.ready || !buffers) return 1;
  const size_t bytes = (size_t)g.n * sizeof(float);
  if (!buffers[1].data || buffers[1].nbytes < bytes) return 1;
  PB_CK(cudaMemcpy(buffers[1].data, g.d_out, bytes, cudaMemcpyDeviceToHost));
  return 0;
}

extern "C" int pb_teardown(void) {
  using namespace pbsol;
  release_all();
  return 0;
}
