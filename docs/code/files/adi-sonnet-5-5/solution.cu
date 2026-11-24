// GPU translation of the PolyBench ADI kernel (baseline/compute.inc), case-24.
//
//   buf0 = u (input/output, n1 x n1), buf1..buf3 = v, p, q (scratch, never read before written).
//
// The n0 Peaceman-Rachford steps are a linear map with a closed form.  With U = 1 + W the all-ones grid is a fixed point
// and  W_final = P (W Q) + (P W_L) e1^T + (P W_R) e2^T,  where P, Q are the n0-th powers of the one-step propagators.
// P = S diag(G) S is Toeplitz-minus-Hankel with kernel kappa(d) = (1/L) sum_k G_k cos(pi k d / L), so it is applied to
// every line by a zero-padded FFT convolution (two real lines packed in one complex FFT, any FFT size M >= 2N).
// Multipliers G, H, F are evaluated in double-double on the device (every pb_compute; nothing is cached across calls).
//
// Compute paths (chosen in pb_setup from (n0, n1) only):
//  * SPECTRAL FP64 (default): FFTs and operators in double precision.
//  * SPECTRAL df64: same algorithm, line passes in double-float (two FP32 numbers) arithmetic on the FP32 pipes; used only
//    where the CPU baseline's own error is far larger than df64's (~1e-14); see want_df().
//  * REPLICA: a literal, FMA-free (bit-exact) transcription of the baseline, used for tiny shapes and as the fallback for
//    shapes whose FFT does not fit in shared memory.
// See NOTES.md for the derivation, the accuracy evidence and the measurements.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
#include <vector>
#include <algorithm>
#include "interface.h"


#ifdef PB_DEV_TIMING
#include <string>
static std::vector<std::pair<std::string, cudaEvent_t>> g_tev;
static cudaStream_t g_dev_stream = nullptr;
static void dev_mark(const char* name) {
  cudaEvent_t e;
  cudaEventCreate(&e);
  cudaEventRecord(e, g_dev_stream);
  g_tev.push_back({name, e});
}
static void dev_report() {
  cudaDeviceSynchronize();
  for (size_t i = 1; i < g_tev.size(); i++) {
    float ms = 0;
    cudaEventElapsedTime(&ms, g_tev[i - 1].second, g_tev[i].second);
    fprintf(stderr, "  [timing] %-14s %9.1f us\n", g_tev[i].first.c_str(), ms * 1000.0f);
  }
  for (auto& p : g_tev) cudaEventDestroy(p.second);
  g_tev.clear();
}
#define PB_MARK(name) dev_mark(name)
#define PB_REPORT() dev_report()
#else
#define PB_MARK(name) ((void)0)
#define PB_REPORT() ((void)0)
#endif

// ---------------------------------------------------------------------------------------------------------------
// error handling
// ---------------------------------------------------------------------------------------------------------------
#define PB_CK(call)                                                                                             \
  do {                                                                                                          \
    cudaError_t pb_e_ = (call);                                                                                 \
    if (pb_e_ != cudaSuccess) return (int)pb_e_ ? (int)pb_e_ : 1;                                              \
  } while (0)

// ---------------------------------------------------------------------------------------------------------------
// double-double arithmetic (device)
// ---------------------------------------------------------------------------------------------------------------
struct dd { double hi, lo; };

__device__ __forceinline__ dd dd_two_sum(double a, double b) {
  double s = a + b, bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
}
__device__ __forceinline__ dd dd_quick_two_sum(double a, double b) {
  double s = a + b;
  return {s, b - (s - a)};
}
__device__ __forceinline__ dd dd_two_prod(double a, double b) {
  double p = __dmul_rn(a, b);
  return {p, fma(a, b, -p)};
}
__device__ __forceinline__ dd dd_add(dd a, dd b) {
  dd s = dd_two_sum(a.hi, b.hi), t = dd_two_sum(a.lo, b.lo);
  s.lo += t.hi;
  s = dd_quick_two_sum(s.hi, s.lo);
  s.lo += t.lo;
  return dd_quick_two_sum(s.hi, s.lo);
}
__device__ __forceinline__ dd dd_neg(dd a) { return {-a.hi, -a.lo}; }
__device__ __forceinline__ dd dd_sub(dd a, dd b) { return dd_add(a, dd_neg(b)); }
__device__ __forceinline__ dd dd_mul(dd a, dd b) {
  dd p = dd_two_prod(a.hi, b.hi);
  p.lo += a.hi * b.lo + a.lo * b.hi;
  return dd_quick_two_sum(p.hi, p.lo);
}
__device__ __forceinline__ dd dd_mul_d(dd a, double b) {
  dd p = dd_two_prod(a.hi, b);
  p.lo += a.lo * b;
  return dd_quick_two_sum(p.hi, p.lo);
}
// a / b with one IEEE division (reciprocal) and FMA-latency refinement steps: ~2e-32 relative accuracy.
__device__ __forceinline__ dd dd_div(dd a, dd b) {
  const double r0 = 1.0 / b.hi;
  const double q1 = a.hi * r0;
  const double e = fma(-q1, b.hi, a.hi);
  const double t = fma(-q1, b.lo, a.lo);
  const double q2 = (e + t) * r0;
  return dd_quick_two_sum(q1, q2);
}
// shorter-latency addition (error is 2^-104 of the larger operand, which is all we need here)
__device__ __forceinline__ dd dd_add_fast(dd a, dd b) {
  dd s = dd_two_sum(a.hi, b.hi);
  s.lo += a.lo + b.lo;
  return dd_quick_two_sum(s.hi, s.lo);
}
// a^e for e >= 0 (a >= 0)
__device__ __forceinline__ dd dd_powu(dd a, unsigned e) {
  dd r = {1.0, 0.0};
  while (e) {
    if (e & 1u) r = dd_mul(r, a);
    e >>= 1;
    if (e) a = dd_mul(a, a);
  }
  return r;
}


// Programmatic dependent launch (sm_90+): a dependent kernel may start early and waits here for its predecessor.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// ---------------------------------------------------------------------------------------------------------------
// plan data shared by host and device
// ---------------------------------------------------------------------------------------------------------------
#ifndef PB_U
#define PB_U 2
#endif
#define PB_MAX_STAGES 12
struct FftParams {
  int M, S;
  int rad[PB_MAX_STAGES], mprev[PB_MAX_STAGES], twoff[PB_MAX_STAGES];
};

// ---------------------------------------------------------------------------------------------------------------
// complex helpers and radix butterflies
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ double2 cadd(double2 a, double2 b) { return make_double2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ double2 csub(double2 a, double2 b) { return make_double2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ double2 cmul(double2 a, double2 w) {  // a * w
  return make_double2(fma(a.x, w.x, -(a.y * w.y)), fma(a.x, w.y, a.y * w.x));
}
__device__ __forceinline__ double2 cmulc(double2 a, double2 w) {  // a * conj(w)
  return make_double2(fma(a.x, w.x, a.y * w.y), fma(a.y, w.x, -(a.x * w.y)));
}
// sg * i * z   (sg = +1 or -1)
template <bool NEG>
__device__ __forceinline__ double2 mul_i(double2 z) {
  return NEG ? make_double2(z.y, -z.x) : make_double2(-z.y, z.x);
}

// forward DFT_R uses exp(-2 pi i pq/R); inverse uses exp(+...)
template <int R, bool INV>
__device__ __forceinline__ void dft_bfly(double2* x);

template <bool INV>
__device__ __forceinline__ void dft_bfly2(double2* x) {
  double2 a = x[0], b = x[1];
  x[0] = cadd(a, b);
  x[1] = csub(a, b);
}
template <bool INV>
__device__ __forceinline__ void dft_bfly3(double2* x) {
  const double s = 0.86602540378443864676;
  double2 t1 = cadd(x[1], x[2]), d = csub(x[1], x[2]);
  double2 t2 = make_double2(fma(-0.5, t1.x, x[0].x), fma(-0.5, t1.y, x[0].y));
  double2 y0 = cadd(x[0], t1);
  // forward: y1 = t2 - i s d, y2 = t2 + i s d ; inverse: signs swapped
  double2 id = INV ? make_double2(-s * d.y, s * d.x) : make_double2(s * d.y, -s * d.x);  // INV: +i s d ; FWD: -i s d
  x[0] = y0;
  x[1] = cadd(t2, id);
  x[2] = csub(t2, id);
}
template <bool INV>
__device__ __forceinline__ void dft_bfly4(double2* x) {
  double2 a0 = cadd(x[0], x[2]), a1 = csub(x[0], x[2]);
  double2 b0 = cadd(x[1], x[3]), b1 = csub(x[1], x[3]);
  double2 ib1 = mul_i<!INV>(b1);  // forward: -i*b1 ; inverse: +i*b1
  x[0] = cadd(a0, b0);
  x[2] = csub(a0, b0);
  x[1] = cadd(a1, ib1);
  x[3] = csub(a1, ib1);
}
template <bool INV>
__device__ __forceinline__ void dft_bfly5(double2* x) {
  const double c1 = 0.30901699437494742410, c2 = -0.80901699437494742410;
  const double s1 = 0.95105651629515357212, s2 = 0.58778525229247312917;
  double2 a1 = cadd(x[1], x[4]), a2 = cadd(x[2], x[3]), d1 = csub(x[1], x[4]), d2 = csub(x[2], x[3]);
  double2 y0 = cadd(cadd(x[0], a1), a2);
  double2 m1 = make_double2(fma(c2, a2.x, fma(c1, a1.x, x[0].x)), fma(c2, a2.y, fma(c1, a1.y, x[0].y)));
  double2 m2 = make_double2(fma(c1, a2.x, fma(c2, a1.x, x[0].x)), fma(c1, a2.y, fma(c2, a1.y, x[0].y)));
  double2 n1 = make_double2(fma(s2, d2.x, s1 * d1.x), fma(s2, d2.y, s1 * d1.y));
  double2 n2 = make_double2(fma(-s1, d2.x, s2 * d1.x), fma(-s1, d2.y, s2 * d1.y));
  double2 in1 = mul_i<!INV>(n1), in2 = mul_i<!INV>(n2);
  x[0] = y0;
  x[1] = cadd(m1, in1);
  x[4] = csub(m1, in1);
  x[2] = cadd(m2, in2);
  x[3] = csub(m2, in2);
}
template <> __device__ __forceinline__ void dft_bfly<2, false>(double2* x) { dft_bfly2<false>(x); }
template <> __device__ __forceinline__ void dft_bfly<2, true>(double2* x) { dft_bfly2<true>(x); }
template <> __device__ __forceinline__ void dft_bfly<3, false>(double2* x) { dft_bfly3<false>(x); }
template <> __device__ __forceinline__ void dft_bfly<3, true>(double2* x) { dft_bfly3<true>(x); }
template <> __device__ __forceinline__ void dft_bfly<4, false>(double2* x) { dft_bfly4<false>(x); }
template <> __device__ __forceinline__ void dft_bfly<4, true>(double2* x) { dft_bfly4<true>(x); }
template <> __device__ __forceinline__ void dft_bfly<5, false>(double2* x) { dft_bfly5<false>(x); }
template <> __device__ __forceinline__ void dft_bfly<5, true>(double2* x) { dft_bfly5<true>(x); }

// One in-place DIT stage over the whole shared array.
template <int R, bool INV>
__device__ __forceinline__ void fft_stage(double2* s, const double2* __restrict__ tw, int mprev, int M, int tid, int T) {
  const int nb = M / R;
  for (int idx = tid; idx < nb; idx += T) {
    const int g = idx / mprev;
    const int n = idx - g * mprev;
    const int base = g * mprev * R + n;
    double2 x[R];
#pragma unroll
    for (int q = 0; q < R; q++) x[q] = s[base + q * mprev];
    if (!INV) {
      if (mprev != 1) {  // first stage: every twiddle is exactly 1
#pragma unroll
        for (int q = 1; q < R; q++) x[q] = cmul(x[q], __ldg(&tw[(q - 1) * mprev + n]));
      }
      dft_bfly<R, false>(x);
    } else {
      dft_bfly<R, true>(x);
      if (mprev != 1) {
#pragma unroll
        for (int q = 1; q < R; q++) x[q] = cmulc(x[q], __ldg(&tw[(q - 1) * mprev + n]));
      }
    }
#pragma unroll
    for (int p = 0; p < R; p++) s[base + p * mprev] = x[p];
  }
}

template <bool INV>
__device__ __forceinline__ void fft_stage_dispatch(double2* s, const double2* __restrict__ tw, const FftParams& P, int st, int tid,
                                                    int T) {
  const double2* twst = tw + P.twoff[st];
  switch (P.rad[st]) {
    case 2: fft_stage<2, INV>(s, twst, P.mprev[st], P.M, tid, T); break;
    case 3: fft_stage<3, INV>(s, twst, P.mprev[st], P.M, tid, T); break;
    case 4: fft_stage<4, INV>(s, twst, P.mprev[st], P.M, tid, T); break;
    default: fft_stage<5, INV>(s, twst, P.mprev[st], P.M, tid, T); break;
  }
}


// ---------------------------------------------------------------------------------------------------------------
// double-float ("df64": hi + lo, two FP32 numbers, ~46 bits) complex arithmetic for the large-shape line passes.
// The FP32 pipes run ~6.6x faster than the FP64 pipe on this device.  Used only where the baseline's own error is
// orders of magnitude larger than the ~1e-13 this arithmetic produces (see NOTES.md).
// ---------------------------------------------------------------------------------------------------------------
struct df { float hi, lo; };
struct cdf { df re, im; };

__device__ __forceinline__ df df_two_sum(float a, float b) {
  float s = a + b, v = s - a;
  return {s, (a - (s - v)) + (b - v)};
}
__device__ __forceinline__ df df_qts(float a, float b) {
  float s = a + b;
  return {s, b - (s - a)};
}
__device__ __forceinline__ df df_add(df a, df b) {
  df s = df_two_sum(a.hi, b.hi);
  s.lo += a.lo + b.lo;
  return df_qts(s.hi, s.lo);
}
__device__ __forceinline__ df df_sub(df a, df b) {
  df s = df_two_sum(a.hi, -b.hi);
  s.lo += a.lo - b.lo;
  return df_qts(s.hi, s.lo);
}
__device__ __forceinline__ df df_mul(df a, df b) {
  const float p = a.hi * b.hi;
  float e = fmaf(a.hi, b.hi, -p);
  e = fmaf(a.hi, b.lo, e);
  e = fmaf(a.lo, b.hi, e);
  return df_qts(p, e);
}
__device__ __forceinline__ df df_neg(df a) { return {-a.hi, -a.lo}; }
__device__ __forceinline__ df df_scale_pow2(df a, float c) { return {a.hi * c, a.lo * c}; }
__device__ __forceinline__ cdf cdf_add(cdf a, cdf b) { return {df_add(a.re, b.re), df_add(a.im, b.im)}; }
__device__ __forceinline__ cdf cdf_sub(cdf a, cdf b) { return {df_sub(a.re, b.re), df_sub(a.im, b.im)}; }
// a * w (fused: one renormalisation per component)
__device__ __forceinline__ cdf cdf_mul(cdf a, cdf w) {
  const float t1 = a.re.hi * w.re.hi, e1 = fmaf(a.re.hi, w.re.hi, -t1);
  const float t2 = a.im.hi * w.im.hi, e2 = fmaf(a.im.hi, w.im.hi, -t2);
  const float c1 = fmaf(a.re.hi, w.re.lo, a.re.lo * w.re.hi);
  const float c2 = fmaf(a.im.hi, w.im.lo, a.im.lo * w.im.hi);
  const df s = df_two_sum(t1, -t2);
  cdf r;
  r.re = df_qts(s.hi, s.lo + ((e1 - e2) + (c1 - c2)));
  const float u1 = a.re.hi * w.im.hi, f1 = fmaf(a.re.hi, w.im.hi, -u1);
  const float u2 = a.im.hi * w.re.hi, f2 = fmaf(a.im.hi, w.re.hi, -u2);
  const float d1 = fmaf(a.re.hi, w.im.lo, a.re.lo * w.im.hi);
  const float d2 = fmaf(a.im.hi, w.re.lo, a.im.lo * w.re.hi);
  const df s2 = df_two_sum(u1, u2);
  r.im = df_qts(s2.hi, s2.lo + ((f1 + f2) + (d1 + d2)));
  return r;
}
// a * conj(w)
__device__ __forceinline__ cdf cdf_mulc(cdf a, cdf w) {
  w.im.hi = -w.im.hi;
  w.im.lo = -w.im.lo;
  return cdf_mul(a, w);
}
__device__ __forceinline__ cdf cdf_scale(cdf a, df c) { return {df_mul(a.re, c), df_mul(a.im, c)}; }
// sg * i * z with sg = -1 (NEG) or +1
template <bool NEG>
__device__ __forceinline__ cdf cdf_mul_i(cdf z) {
  return NEG ? cdf{z.im, df_neg(z.re)} : cdf{df_neg(z.im), z.re};
}
__device__ __forceinline__ cdf cdf_load(const float4 v) { return cdf{df{v.x, v.y}, df{v.z, v.w}}; }
__device__ __forceinline__ float4 cdf_store(const cdf z) { return make_float4(z.re.hi, z.re.lo, z.im.hi, z.im.lo); }

constexpr double kDfS3 = 0.86602540378443864676;
constexpr double kDfC1 = 0.30901699437494742410, kDfC2 = -0.80901699437494742410;
constexpr double kDfS1 = 0.95105651629515357212, kDfS2 = 0.58778525229247312917;
#define DF_CONST(x) df{(float)(x), (float)((x) - (double)(float)(x))}

template <int R, bool INV>
__device__ __forceinline__ void dft_bfly_df(cdf* x);

template <bool INV>
__device__ __forceinline__ void dft_bfly2_df(cdf* x) {
  const cdf a = x[0], b = x[1];
  x[0] = cdf_add(a, b);
  x[1] = cdf_sub(a, b);
}
template <bool INV>
__device__ __forceinline__ void dft_bfly3_df(cdf* x) {
  const df s = DF_CONST(kDfS3);
  const cdf t1 = cdf_add(x[1], x[2]), d = cdf_sub(x[1], x[2]);
  const cdf half = {df_scale_pow2(t1.re, 0.5f), df_scale_pow2(t1.im, 0.5f)};
  const cdf t2 = cdf_sub(x[0], half);
  const cdf y0 = cdf_add(x[0], t1);
  const cdf sd = cdf_scale(d, s);
  const cdf id = cdf_mul_i<!INV>(sd);  // forward: -i s d ; inverse: +i s d
  x[0] = y0;
  x[1] = cdf_add(t2, id);
  x[2] = cdf_sub(t2, id);
}
template <bool INV>
__device__ __forceinline__ void dft_bfly4_df(cdf* x) {
  const cdf a0 = cdf_add(x[0], x[2]), a1 = cdf_sub(x[0], x[2]);
  const cdf b0 = cdf_add(x[1], x[3]), b1 = cdf_sub(x[1], x[3]);
  const cdf ib1 = cdf_mul_i<!INV>(b1);
  x[0] = cdf_add(a0, b0);
  x[2] = cdf_sub(a0, b0);
  x[1] = cdf_add(a1, ib1);
  x[3] = cdf_sub(a1, ib1);
}
template <bool INV>
__device__ __forceinline__ void dft_bfly5_df(cdf* x) {
  const df c1 = DF_CONST(kDfC1), c2 = DF_CONST(kDfC2), s1 = DF_CONST(kDfS1), s2 = DF_CONST(kDfS2);
  const cdf a1 = cdf_add(x[1], x[4]), a2 = cdf_add(x[2], x[3]), d1 = cdf_sub(x[1], x[4]), d2 = cdf_sub(x[2], x[3]);
  const cdf y0 = cdf_add(cdf_add(x[0], a1), a2);
  const cdf m1 = cdf_add(x[0], cdf_add(cdf_scale(a1, c1), cdf_scale(a2, c2)));
  const cdf m2 = cdf_add(x[0], cdf_add(cdf_scale(a1, c2), cdf_scale(a2, c1)));
  const cdf n1 = cdf_add(cdf_scale(d1, s1), cdf_scale(d2, s2));
  const cdf n2 = cdf_sub(cdf_scale(d1, s2), cdf_scale(d2, s1));
  const cdf in1 = cdf_mul_i<!INV>(n1), in2 = cdf_mul_i<!INV>(n2);
  x[0] = y0;
  x[1] = cdf_add(m1, in1);
  x[4] = cdf_sub(m1, in1);
  x[2] = cdf_add(m2, in2);
  x[3] = cdf_sub(m2, in2);
}
template <> __device__ __forceinline__ void dft_bfly_df<2, false>(cdf* x) { dft_bfly2_df<false>(x); }
template <> __device__ __forceinline__ void dft_bfly_df<2, true>(cdf* x) { dft_bfly2_df<true>(x); }
template <> __device__ __forceinline__ void dft_bfly_df<3, false>(cdf* x) { dft_bfly3_df<false>(x); }
template <> __device__ __forceinline__ void dft_bfly_df<3, true>(cdf* x) { dft_bfly3_df<true>(x); }
template <> __device__ __forceinline__ void dft_bfly_df<4, false>(cdf* x) { dft_bfly4_df<false>(x); }
template <> __device__ __forceinline__ void dft_bfly_df<4, true>(cdf* x) { dft_bfly4_df<true>(x); }
template <> __device__ __forceinline__ void dft_bfly_df<5, false>(cdf* x) { dft_bfly5_df<false>(x); }
template <> __device__ __forceinline__ void dft_bfly_df<5, true>(cdf* x) { dft_bfly5_df<true>(x); }

template <int R, bool INV>
__device__ __forceinline__ void bfly_df_run(float4* s, const int base, const int mprev, const cdf* w) {
  cdf x[R];
#pragma unroll
  for (int q = 0; q < R; q++) x[q] = cdf_load(s[base + q * mprev]);
  if (!INV) {
#pragma unroll
    for (int q = 1; q < R; q++) x[q] = cdf_mul(x[q], w[q - 1]);
    dft_bfly_df<R, false>(x);
  } else {
    dft_bfly_df<R, true>(x);
#pragma unroll
    for (int q = 1; q < R; q++) x[q] = cdf_mulc(x[q], w[q - 1]);
  }
#pragma unroll
  for (int p = 0; p < R; p++) s[base + p * mprev] = cdf_store(x[p]);
}

template <int R, bool INV>
__device__ __forceinline__ void fft_stage_df(float4* s, const float4* __restrict__ tw, int mprev, int M, int tid, int T) {
  const int nb = M / R;
  if (((mprev & (mprev - 1)) == 0) && ((T & (T - 1)) == 0) && mprev <= T) {
    // every thread keeps the same butterfly index n for all its iterations: hoist the twiddles
    const int sh = __ffs(mprev) - 1;
    const int n = tid & (mprev - 1);
    cdf w[R > 1 ? R - 1 : 1];
#pragma unroll
    for (int q = 1; q < R; q++) w[q - 1] = cdf_load(__ldg(&tw[(q - 1) * mprev + n]));
    const int ngroups = nb >> sh;
    for (int g = tid >> sh; g < ngroups; g += (T >> sh)) bfly_df_run<R, INV>(s, ((g * R) << sh) + n, mprev, w);
  } else if ((mprev & (mprev - 1)) == 0) {
    const int sh = __ffs(mprev) - 1;
    for (int idx = tid; idx < nb; idx += T) {
      const int g = idx >> sh;
      const int n = idx & (mprev - 1);
      cdf w[R > 1 ? R - 1 : 1];
#pragma unroll
      for (int q = 1; q < R; q++) w[q - 1] = cdf_load(__ldg(&tw[(q - 1) * mprev + n]));
      bfly_df_run<R, INV>(s, ((g * R) << sh) + n, mprev, w);
    }
  } else {
    for (int idx = tid; idx < nb; idx += T) {
      const int g = idx / mprev;
      const int n = idx - g * mprev;
      cdf w[R > 1 ? R - 1 : 1];
#pragma unroll
      for (int q = 1; q < R; q++) w[q - 1] = cdf_load(__ldg(&tw[(q - 1) * mprev + n]));
      bfly_df_run<R, INV>(s, g * mprev * R + n, mprev, w);
    }
  }
}

// First stage (mprev == 1): all twiddles are exactly 1, so skip loading and multiplying them (no twiddle registers).
template <int R, bool INV>
__device__ __forceinline__ void fft_stage_df_first(float4* s, int M, int tid, int T) {
  const int nb = M / R;
  for (int g = tid; g < nb; g += T) {
    cdf x[R];
#pragma unroll
    for (int q = 0; q < R; q++) x[q] = cdf_load(s[g * R + q]);
    dft_bfly_df<R, INV>(x);
#pragma unroll
    for (int p = 0; p < R; p++) s[g * R + p] = cdf_store(x[p]);
  }
}

template <bool INV>
__device__ __forceinline__ void fft_stage_df_dispatch(float4* s, const float4* __restrict__ tw, const FftParams& P, int st, int tid,
                                                      int T) {
  if (st == 0) {  // mprev == 1
    switch (P.rad[0]) {
      case 2: fft_stage_df_first<2, INV>(s, P.M, tid, T); break;
      case 3: fft_stage_df_first<3, INV>(s, P.M, tid, T); break;
      case 4: fft_stage_df_first<4, INV>(s, P.M, tid, T); break;
      default: fft_stage_df_first<5, INV>(s, P.M, tid, T); break;
    }
    return;
  }
  const float4* twst = tw + P.twoff[st];
  switch (P.rad[st]) {
    case 2: fft_stage_df<2, INV>(s, twst, P.mprev[st], P.M, tid, T); break;
    case 3: fft_stage_df<3, INV>(s, twst, P.mprev[st], P.M, tid, T); break;
    case 4: fft_stage_df<4, INV>(s, twst, P.mprev[st], P.M, tid, T); break;
    default: fft_stage_df<5, INV>(s, twst, P.mprev[st], P.M, tid, T); break;
  }
}

// double -> df64 with 32-bit integer operations only (no FP64-pipe arithmetic): hi = x rounded to 24 bits, lo = remainder.
// Exact to 2^-49 relative for |x| in [2^-70, 2^100); outside that range (zero, tiny, huge, non-finite) hi = (float)x, lo = 0.
__device__ __forceinline__ df df_from_double_int(double x) {
  const unsigned lo32 = (unsigned)__double2loint(x), hi32 = (unsigned)__double2hiint(x);
  const int e = (int)((hi32 >> 20) & 0x7FFu);
  if ((unsigned)(e - 953) > 170u) return df{(float)x, 0.0f};
  const unsigned m_top = (((hi32 & 0xFFFFFu) | 0x100000u) << 3) | (lo32 >> 29);   // truncated 24-bit significand
  const unsigned low29 = lo32 & 0x1FFFFFFFu;
  const unsigned rb = low29 >> 28;                                           // round bit
  const int r = (int)low29 - (int)(rb << 29);                                // remainder in [-2^28, 2^28)
  const unsigned hm = m_top + rb;                                            // <= 2^24
  float hi = (float)(int)hm * __uint_as_float((unsigned)(e - 1023 - 23 + 127) << 23);
  float lo = (float)r * __uint_as_float((unsigned)(e - 1023 - 52 + 127) << 23);
  if (hi32 >> 31) {
    hi = -hi;
    lo = -lo;
  }
  return df{hi, lo};
}
// df64 (normalised: |lo| <= ulp(hi)/2) -> double, bit-identical to (double)hi + (double)lo but without any FP64-pipe
// instruction (not even predicated-off ones, which still occupy that pipe): the bit pattern of |hi| as a double plus lo
// expressed in units of the double ulp (integer add; carries and borrows across binades are exact because doubles are
// monotonic in their bit patterns).  When hi is a power of two and lo < 0 the result lies in the lower binade, whose ulp is
// half as big (hence the extra factor 2).  |hi| < 2^-75 is flushed to zero (absolute error < 1e-22).
__device__ __forceinline__ double df_to_double_int(df a) {
  const unsigned uh = __float_as_uint(a.hi);
  const unsigned e = (uh >> 23) & 0xFFu;
  const unsigned s = uh >> 31;
  const unsigned m23 = uh & 0x7FFFFFu;
  const float lo_s = s ? -a.lo : a.lo;                       // lo relative to |hi|
  const bool borrow = (m23 == 0u) && (lo_s < 0.0f);
  const unsigned se = (unsigned)((borrow ? 307 : 306) - (int)e) << 23;          // exponent field of 2^(52-E) or 2^(53-E)
  const float lq = fminf(fmaxf(lo_s * __uint_as_float(se & 0x7F800000u), -1073741824.0f), 1073741824.0f);
  unsigned long long bits = ((unsigned long long)(e + 896u) << 52) | ((unsigned long long)m23 << 29);
  bits += (unsigned long long)(long long)__float2int_rn(lq);
  bits |= (unsigned long long)s << 63;
  if (e - 52u > 202u) bits = (e == 0xFFu) ? (((unsigned long long)s << 63) | (0x7FFull << 52) | ((unsigned long long)m23 << 29))
                                          : ((unsigned long long)s << 63);   // inf/nan : zero/tiny (flushed)
  return __longlong_as_double((long long)bits);
}
__device__ __forceinline__ df df_load2(const float2 v) { return df{v.x, v.y}; }

// ---------------------------------------------------------------------------------------------------------------
// K0a: multipliers for the mode pair (k, L-k), k = 1..K0 (K0 = N/2), plus the middle mode L/2 when N is odd.
//   G_k = g_k^n0,  H_k = h_k^n0,  Fs_k = (2 rho_y / L) h_k^(n0-1) sin(pi k / L) / (1 + rho_y lambda_k)   (double-double)
//   Pair arrays:  Gs = G_k + G_{L-k}, Gd = G_k - G_{L-k}  (same for H and Fs), used for cos/sin sums with parity.
// ---------------------------------------------------------------------------------------------------------------
struct ModeVals { double G, H, F; };

// hf == false: G = g^n0 (lane type "G");  hf == true: H = h^n0 and F (lane type "H/F").  Both lane types run the same
// convergent base^exponent chain (G: base |g|, exponent n0 ; H/F: base |h|, exponent n0-1); only the short tail differs.
__device__ __forceinline__ ModeVals mode_values(int n0, int n1, int L, int m, const double* __restrict__ s_hi,
                                                const double* __restrict__ s_lo, const double* __restrict__ c_hi,
                                                const double* __restrict__ c_lo, const dd rho_x, const bool hf) {
  const dd one = {1.0, 0.0};
  const dd rho_y = {0.5 * rho_x.hi, 0.5 * rho_x.lo};
  const dd s = {s_hi[m], s_lo[m]};
  dd lam = dd_mul(s, s);
  lam.hi *= 4.0;
  lam.lo *= 4.0;
  const dd rho = hf ? rho_y : rho_x;
  const dd x = dd_mul(rho, lam);
  const dd op = dd_add_fast(one, x), om = dd_add_fast(one, dd_neg(x));
  const dd gh = dd_div(om, op);                       // g (G lanes) or h (H/F lanes)
  const bool neg = gh.hi < 0.0;
  const dd ag = neg ? dd_neg(gh) : gh;
  const unsigned e = hf ? (unsigned)(n0 - 1) : (unsigned)n0;
  const dd pw = dd_powu(ag, e);                       // |gh|^e   (0 for gh == 0, e > 0 ; 1 for e == 0)
  ModeVals r;
  r.G = 0.0;
  r.H = 0.0;
  r.F = 0.0;
  if (!hf) {
    r.G = (neg && (n0 & 1)) ? -pw.hi : pw.hi;
  } else {
    const dd p = dd_mul(pw, ag);                      // |h|^n0
    r.H = (neg && (n0 & 1)) ? -p.hi : p.hi;
    const dd pm1s = (neg && ((n0 - 1) & 1)) ? dd_neg(pw) : pw;   // h^(n0-1)
    if (pm1s.hi != 0.0) {
      const dd f = dd_div(pm1s, op);                  // h^(n0-1) / (1 + rho_y lambda)
      dd sin1 = dd_mul(s, dd{c_hi[m], c_lo[m]});      // sin(pi k / L) = 2 sin(pi k/2L) cos(pi k/2L)
      sin1.hi *= 2.0;
      sin1.lo *= 2.0;
      const dd cE = dd_div(dd{2.0 * rho_y.hi, 2.0 * rho_y.lo}, dd{(double)L, 0.0});
      r.F = dd_mul(dd_mul(f, sin1), cE).hi;
    }
  }
  return r;
}

// out layout: [0,K0) Gs | [K0,2K0) Gd | Hs | Hd | Es | Ed | mid: Gm, Hm, Fm | consts: cpq.hi, cpq.lo
// Lane quadruple (4p..4p+3) = (mode k: G branch, mode k: H/F branch, mode L-k: G branch, mode L-k: H/F branch); the
// branches are independent dependent-chains, so they run concurrently; sums/differences are formed with shuffles.
__global__ void mult_kernel(int n0, int n1, int N, int L, int M, const double* __restrict__ s_hi, const double* __restrict__ s_lo,
                            const double* __restrict__ c_hi, const double* __restrict__ c_lo, double* __restrict__ out) {
  pdl_trigger();
  const int K0 = N / 2;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int p = tid >> 2;
  const int sub = tid & 3;  // bit0: 0 = G branch, 1 = H/F branch ; bit1: 0 = mode k, 1 = mode L-k
  const dd rho_x = dd_div(dd{(double)n1 * (double)n1, 0.0}, dd{(double)n0, 0.0});
  int m;  // 0-based mode index
  if (p < K0) m = (sub & 2) ? (L - (p + 1) - 1) : p;
  else if (p == K0 && (N & 1)) m = (L / 2) - 1;  // middle mode
  else m = 0;
  const bool hf = (sub & 1) != 0;
  const ModeVals v = mode_values(n0, n1, L, m, s_hi, s_lo, c_hi, c_lo, rho_x, hf);
  // partner = same branch, other mode (lane ^ 2)
  const double oV = __shfl_xor_sync(0xffffffffu, v.G, 2);
  const double oH = __shfl_xor_sync(0xffffffffu, v.H, 2);
  const double oF = __shfl_xor_sync(0xffffffffu, v.F, 2);
  if (p < K0) {
    if (sub == 0) out[p] = v.G + oV;                       // Gs
    else if (sub == 2) out[K0 + p] = oV - v.G;             // Gd = G_k - G_{L-k}
    else if (sub == 1) { out[2 * K0 + p] = v.H + oH; out[4 * K0 + p] = v.F + oF; }
    else { out[3 * K0 + p] = oH - v.H; out[5 * K0 + p] = oF - v.F; }
  } else if (p == K0 && (N & 1)) {
    if (sub == 0) out[6 * K0 + 0] = v.G;
    if (sub == 1) { out[6 * K0 + 1] = v.H; out[6 * K0 + 2] = v.F; }
  }
  if (tid == 0) {
    dd cpq = dd_div(dd{1.0, 0.0}, dd_mul(dd{(double)L, 0.0}, dd{(double)M, 0.0}));
    out[6 * K0 + 3] = cpq.hi;
    out[6 * K0 + 4] = cpq.lo;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// K0b: one warp per output.  warps 0..L: kappa_P(d), kappa_Q(d) = c * sum_k {G_k, H_k} cos(pi k d / L)  (mirrored to 2L-d;
//      c = 1/(L M) folds the unnormalised inverse FFT).  warps L+1..L+N: e1(i) = sum_l Fs_l sin(pi l i / L)   (e2 = reverse).
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ double dd_scale(double sum, double chi, double clo) {
  double p = __dmul_rn(sum, chi);
  double e = fma(sum, chi, -p);
  return p + (e + sum * clo);
}

__global__ void __launch_bounds__(256) kappa_kernel(int N, int L, const double* __restrict__ cosT, const double* __restrict__ sinT,
                                                    const double* __restrict__ pair, double* __restrict__ kP,
                                                    double* __restrict__ kQ, double* __restrict__ e1) {
  pdl_trigger();
  pdl_wait();
  const int lane = threadIdx.x & 31;
  const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (gw > L + N) return;
  const int K0 = N / 2;
  const int L2 = 2 * L;
  const bool has_mid = (N & 1) != 0;
  if (gw <= L) {
    const int d = gw;
    const double* Gx = pair + ((d & 1) ? K0 : 0);
    const double* Hx = pair + 2 * K0 + ((d & 1) ? K0 : 0);
    int t = (int)(((long long)(lane + 1) * (long long)d) % L2);
    const int step = (int)(((long long)32 * (long long)d) % L2);
    double aP = 0.0, aQ = 0.0;
    for (int p = lane; p < K0; p += 32) {
      const double c = __ldg(&cosT[t]);
      aP = fma(__ldg(&Gx[p]), c, aP);
      aQ = fma(__ldg(&Hx[p]), c, aQ);
      t += step;
      if (t >= L2) t -= L2;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      aP += __shfl_down_sync(0xffffffffu, aP, o);
      aQ += __shfl_down_sync(0xffffffffu, aQ, o);
    }
    if (lane == 0) {
      if (has_mid) {
        const int r = d & 3;
        const double cm = r == 0 ? 1.0 : (r == 2 ? -1.0 : 0.0);  // cos(pi d / 2)
        aP = fma(pair[6 * K0 + 0], cm, aP);
        aQ = fma(pair[6 * K0 + 1], cm, aQ);
      }
      const double chi = pair[6 * K0 + 3], clo = pair[6 * K0 + 4];
      const double vP = dd_scale(aP, chi, clo), vQ = dd_scale(aQ, chi, clo);
      kP[d] = vP;
      kQ[d] = vQ;
      const int dm = L2 - d;  // kappa(2L - d) = kappa(d)
      if (dm != d && dm <= 2 * N) {
        kP[dm] = vP;
        kQ[dm] = vQ;
      }
    }
  } else {
    const int i = gw - L;  // 1..N
    const double* Ex = pair + 4 * K0 + ((i & 1) ? 0 : K0);  // i odd: Es, i even: Ed
    int t = (int)(((long long)(lane + 1) * (long long)i) % L2);
    const int step = (int)(((long long)32 * (long long)i) % L2);
    double a1 = 0.0;
    for (int p = lane; p < K0; p += 32) {
      a1 = fma(__ldg(&Ex[p]), __ldg(&sinT[t]), a1);
      t += step;
      if (t >= L2) t -= L2;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a1 += __shfl_down_sync(0xffffffffu, a1, o);
    if (lane == 0) {
      if (has_mid) {
        const int r = i & 3;
        const double sm = r == 1 ? 1.0 : (r == 3 ? -1.0 : 0.0);  // sin(pi i / 2)
        a1 = fma(pair[6 * K0 + 2], sm, a1);
      }
      e1[i - 1] = a1;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// K0c: FFT of the packed kernel arrays (kT + i kH) and extraction of the frequency-domain operator
//      Yhat[f] = A[f] Z[f] + B[f] Z[M-f]
// ---------------------------------------------------------------------------------------------------------------
__global__ void kernel_spectrum_kernel(const FftParams P, int N, const double2* __restrict__ tw, const int* __restrict__ perm,
                                       const double* __restrict__ kP, const double* __restrict__ kQ, double* __restrict__ A_all,
                                       double2* __restrict__ B_all) {
  extern __shared__ double2 sm[];
  pdl_trigger();
  pdl_wait();
  const int M = P.M;
  const int op = blockIdx.x;  // 0: P, 1: Q
  const double* kap = op == 0 ? kP : kQ;
  double* A = A_all + (size_t)op * M;
  double2* B = B_all + (size_t)op * M;
  const int tid = threadIdx.x, T = blockDim.x;
  for (int jp = tid; jp < M; jp += T) {
    double re = 0.0, im = 0.0;
    if (jp <= N - 1) re = kap[jp];
    else if (jp >= M - N + 1) re = kap[M - jp];
    if (jp >= 1 && jp <= 2 * N) im = kap[jp];
    else if (jp == 0 && M == 2 * N) im = kap[2 * N];
    sm[perm[jp]] = make_double2(re, im);
  }
  __syncthreads();
  for (int st = 0; st < P.S; st++) {
    fft_stage_dispatch<false>(sm, tw, P, st, tid, T);
    __syncthreads();
  }
  for (int f = tid; f < M; f += T) {
    const int fm = f == 0 ? 0 : M - f;
    double2 zf = sm[f], zm = sm[fm];
    const double av = 0.5 * (zf.x + zm.x);
    const double2 bv = make_double2(-0.5 * (zf.y + zm.y), 0.5 * (zf.x - zm.x));
    A[f] = av;
    B[f] = bv;
  }
}


// df64 version of the kernel-spectrum FFT (used together with the df64 line passes): writes only A_df / B_df.
__global__ void kernel_spectrum_kernel_df(const FftParams P, int N, const float4* __restrict__ tw_df, const int* __restrict__ perm,
                                          const double* __restrict__ kP, const double* __restrict__ kQ,
                                          float2* __restrict__ A_df, float4* __restrict__ B_df) {
  extern __shared__ float4 smf[];
  pdl_trigger();
  pdl_wait();
  const int M = P.M;
  const int op = blockIdx.x;
  const double* kap = op == 0 ? kP : kQ;
  const int tid = threadIdx.x, T = blockDim.x;
  for (int jp = tid; jp < M; jp += T) {
    double re = 0.0, im = 0.0;
    if (jp <= N - 1) re = kap[jp];
    else if (jp >= M - N + 1) re = kap[M - jp];
    if (jp >= 1 && jp <= 2 * N) im = kap[jp];
    else if (jp == 0 && M == 2 * N) im = kap[2 * N];
    const df dr = df_from_double_int(re), di = df_from_double_int(im);
    smf[perm[jp]] = make_float4(dr.hi, dr.lo, di.hi, di.lo);
  }
  __syncthreads();
  for (int st = 0; st < P.S; st++) {
    fft_stage_df_dispatch<false>(smf, tw_df, P, st, tid, T);
    __syncthreads();
  }
  for (int f = tid; f < M; f += T) {
    const int fm = f == 0 ? 0 : M - f;
    const cdf zf = cdf_load(smf[f]), zm = cdf_load(smf[fm]);
    const df av = df_scale_pow2(df_add(zf.re, zm.re), 0.5f);
    const df bre = df_scale_pow2(df_neg(df_add(zf.im, zm.im)), 0.5f);
    const df bim = df_scale_pow2(df_sub(zf.re, zm.re), 0.5f);
    A_df[(size_t)op * M + f] = make_float2(av.hi, av.lo);
    B_df[(size_t)op * M + f] = make_float4(bre.hi, bre.lo, bim.hi, bim.lo);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// K1 / K2: line passes
// ---------------------------------------------------------------------------------------------------------------
struct PassArgs {
  FftParams P;
  int N, n1, ldT;
  const double2* tw;
  const int* perm;
  const double* A_all;   // [2][M]  (0: P, 1: Q)
  const double2* B_all;  // [2][M]
  const double* d_in;
  double* d_out;
  double* T1;      // [N][ldT]
  double* pwl;     // [N]
  double* pwr;     // [N]
  const double* e1;  // [N]   (e2[i] = e1[N-1-i])
  const float4* tw_df;   // df64 twiddles (same layout as tw)
  const float2* A_df;    // [2][M]
  const float4* B_df;    // [2][M]
  float2* T1df;          // df64 intermediate [N][ldT] (df path)
  float2* pwldf;         // [N]
  float2* pwrdf;         // [N]
};

// MODE 0: row pass (apply Q along each row of W, transposed store) + extra block for the boundary vectors (apply P).
// MODE 1: column pass (apply P along each column of W Q) + rank-2 correction + "1 +" epilogue.
template <int MODE, int TPB>
__global__ void __launch_bounds__(TPB) pass_kernel(const __grid_constant__ PassArgs a) {
  extern __shared__ double2 sm[];
  const FftParams& P = a.P;
  const int M = P.M, N = a.N, n1 = a.n1;
  const int tid = threadIdx.x, T = blockDim.x;
  const int b = blockIdx.x;
  const int nb = (N + 1) / 2;
  bool special = false;
  if (MODE == 0) special = (b == nb);
  const int l1 = 2 * b, l2 = 2 * b + 1;  // line indices (rows for MODE 0, columns of W Q for MODE 1)
  const bool has2 = l2 < N;
  const int op = (MODE == 0 && !special) ? 1 : 0;  // row pass uses Q (op 1); column pass and boundary vectors use P (op 0)
  const double* A = a.A_all + (size_t)op * M;
  const double2* B = a.B_all + (size_t)op * M;

  // ---- load: array index jp = 1..N holds element jp-1 of the line (batched: U independent loads in flight) ----
  constexpr int U = PB_U;
  for (int jp0 = tid; jp0 < M; jp0 += T * U) {
    double x1[U], x2[U];
    int pp[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int jp = jp0 + u * T;
      pp[u] = jp < M ? __ldg(&a.perm[jp]) : 0;
      double v1 = 0.0, v2 = 0.0;
      if (jp >= 1 && jp <= N) {
        const int e = jp - 1;
        if (MODE == 0) {
          if (!special) {
            v1 = a.d_in[(size_t)(l1 + 1) * n1 + (e + 1)] - 1.0;
            if (has2) v2 = a.d_in[(size_t)(l2 + 1) * n1 + (e + 1)] - 1.0;
          } else {
            v1 = a.d_in[(size_t)(e + 1) * n1 + 0] - 1.0;
            v2 = a.d_in[(size_t)(e + 1) * n1 + (n1 - 1)] - 1.0;
          }
        } else {
          v1 = a.T1[(size_t)l1 * a.ldT + e];
          if (has2) v2 = a.T1[(size_t)l2 * a.ldT + e];
        }
      }
      x1[u] = v1;
      x2[u] = v2;
    }
#pragma unroll
    for (int u = 0; u < U; u++)
      if (jp0 + u * T < M) sm[pp[u]] = make_double2(x1[u], x2[u]);
  }
  __syncthreads();
  for (int st = 0; st < P.S; st++) {
    fft_stage_dispatch<false>(sm, a.tw, P, st, tid, T);
    __syncthreads();
  }
  // ---- frequency-domain operator (batched loads of the operator tables) ----
  pdl_wait();
  for (int f0 = tid; f0 <= M / 2; f0 += T * U) {
    double a1[U], a2[U];
    double2 b1[U], b2[U], z1[U], z2[U];
    bool ok[U], two[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int f = f0 + u * T;
      ok[u] = f <= M / 2;
      const int ff = ok[u] ? f : 0;
      const int fm = ff == 0 ? 0 : M - ff;
      two[u] = fm != ff;
      a1[u] = __ldg(&A[ff]);
      b1[u] = __ldg(&B[ff]);
      a2[u] = __ldg(&A[fm]);
      b2[u] = __ldg(&B[fm]);
      z1[u] = sm[ff];
      z2[u] = sm[fm];
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      if (!ok[u]) continue;
      const int f = f0 + u * T;
      const int fm = f == 0 ? 0 : M - f;
      double2 y1 = make_double2(fma(b1[u].x, z2[u].x, fma(-b1[u].y, z2[u].y, a1[u] * z1[u].x)),
                                fma(b1[u].x, z2[u].y, fma(b1[u].y, z2[u].x, a1[u] * z1[u].y)));
      if (two[u]) {
        double2 y2 = make_double2(fma(b2[u].x, z1[u].x, fma(-b2[u].y, z1[u].y, a2[u] * z2[u].x)),
                                  fma(b2[u].x, z1[u].y, fma(b2[u].y, z1[u].x, a2[u] * z2[u].y)));
        sm[f] = y1;
        sm[fm] = y2;
      } else {
        sm[f] = y1;
      }
    }
  }
  __syncthreads();
  for (int st = P.S - 1; st >= 0; st--) {
    fft_stage_dispatch<true>(sm, a.tw, P, st, tid, T);
    __syncthreads();
  }
  // ---- store (batched: perm / operator-vector loads for U elements are issued before any is used) ----
  if (MODE == 0) {
    if (!special) {
      for (int jp0 = tid + 1; jp0 <= N; jp0 += T * U) {
        int pp[U];
        double2 v[U];
#pragma unroll
        for (int u = 0; u < U; u++) pp[u] = (jp0 + u * T <= N) ? __ldg(&a.perm[jp0 + u * T]) : 0;
#pragma unroll
        for (int u = 0; u < U; u++) v[u] = sm[pp[u]];
#pragma unroll
        for (int u = 0; u < U; u++) {
          const int jp = jp0 + u * T;
          if (jp <= N) {
            double* dst = a.T1 + (size_t)(jp - 1) * a.ldT + l1;
            if (has2) *reinterpret_cast<double2*>(dst) = v[u];
            else dst[0] = v[u].x;
          }
        }
      }
      if (tid < 4) {  // boundary columns of the output rows
        const int row = (tid & 1) ? l2 : l1;
        if (row < N) a.d_out[(size_t)(row + 1) * n1 + ((tid & 2) ? (n1 - 1) : 0)] = 1.0;
      }
    } else {
      for (int jp = tid + 1; jp <= N; jp += T) {
        const double2 v = sm[a.perm[jp]];
        a.pwl[jp - 1] = v.x;
        a.pwr[jp - 1] = v.y;
      }
    }
  } else {
    // boundary-column correction: W[j][i] += pwl[j] e1[i] + pwr[j] e2[i]
    const double e1a = a.e1[l1], e2a = a.e1[N - 1 - l1];
    double e1b = 0.0, e2b = 0.0;
    if (has2) {
      e1b = a.e1[l2];
      e2b = a.e1[N - 1 - l2];
    }
    for (int jp0 = tid + 1; jp0 <= N; jp0 += T * U) {
      int pp[U];
      double wl[U], wr[U];
      double2 v[U];
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int jp = jp0 + u * T;
        const bool ok = jp <= N;
        pp[u] = ok ? __ldg(&a.perm[jp]) : 0;
        wl[u] = ok ? a.pwl[jp - 1] : 0.0;
        wr[u] = ok ? a.pwr[jp - 1] : 0.0;
      }
#pragma unroll
      for (int u = 0; u < U; u++) v[u] = sm[pp[u]];
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int jp = jp0 + u * T;
        if (jp > N) continue;
        const int j = jp - 1;
        double w1 = fma(wr[u], e2a, fma(wl[u], e1a, v[u].x));
        a.d_out[(size_t)(j + 1) * n1 + (l1 + 1)] = 1.0 + w1;
        if (has2) {
          double w2 = fma(wr[u], e2b, fma(wl[u], e1b, v[u].y));
          a.d_out[(size_t)(j + 1) * n1 + (l2 + 1)] = 1.0 + w2;
        }
      }
    }
  }
}


#ifdef PB_DEV_PHASE
__device__ long long g_phase[16];
#define PHASE(k) do { if (blockIdx.x == 7 && threadIdx.x == 0) g_phase[k] = clock64(); } while (0)
#else
#define PHASE(k) ((void)0)
#endif
// df64 version of pass_kernel: identical data flow, FFT and frequency-domain operator in double-float arithmetic.
// The intermediate array T1 and the boundary vectors are kept as (hi, lo) float pairs, so the only FP64-pipe work left
// is one DADD per output element (input split / output assembly use integer bit operations).
template <int MODE, int TPB, int MINB>
__global__ void __launch_bounds__(TPB, MINB) pass_kernel_df(const __grid_constant__ PassArgs a) {
  extern __shared__ float4 smf[];
  PHASE(0);
  const FftParams& P = a.P;
  const int M = P.M, N = a.N, n1 = a.n1;
  const int tid = threadIdx.x, T = blockDim.x;
  const int b = blockIdx.x;
  const int nb = (N + 1) / 2;
  bool special = false;
  if (MODE == 0) special = (b == nb);
  const int l1 = 2 * b, l2 = 2 * b + 1;
  const bool has2 = l2 < N;
  const int op = (MODE == 0 && !special) ? 1 : 0;
  const float2* A = a.A_df + (size_t)op * M;
  const float4* B = a.B_df + (size_t)op * M;
  constexpr int U = PB_U;
  const df one_df = {1.0f, 0.0f};

  for (int jp0 = tid; jp0 < M; jp0 += T * U) {
    df x1[U], x2[U];
    int pp[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int jp = jp0 + u * T;
      pp[u] = jp < M ? __ldg(&a.perm[jp]) : 0;
      df v1 = {0.0f, 0.0f}, v2 = {0.0f, 0.0f};
      if (jp >= 1 && jp <= N) {
        const int e = jp - 1;
        if (MODE == 0) {
          if (!special) {
            v1 = df_sub(df_from_double_int(a.d_in[(size_t)(l1 + 1) * n1 + (e + 1)]), one_df);
            if (has2) v2 = df_sub(df_from_double_int(a.d_in[(size_t)(l2 + 1) * n1 + (e + 1)]), one_df);
          } else {
            v1 = df_sub(df_from_double_int(a.d_in[(size_t)(e + 1) * n1 + 0]), one_df);
            v2 = df_sub(df_from_double_int(a.d_in[(size_t)(e + 1) * n1 + (n1 - 1)]), one_df);
          }
        } else {
          v1 = df_load2(a.T1df[(size_t)l1 * a.ldT + e]);
          if (has2) v2 = df_load2(a.T1df[(size_t)l2 * a.ldT + e]);
        }
      }
      x1[u] = v1;
      x2[u] = v2;
    }
#pragma unroll
    for (int u = 0; u < U; u++)
      if (jp0 + u * T < M) smf[pp[u]] = make_float4(x1[u].hi, x1[u].lo, x2[u].hi, x2[u].lo);
  }
  __syncthreads();
  PHASE(1);
#ifndef PB_ABL_NOFFT
  for (int st = 0; st < P.S; st++) {
    fft_stage_df_dispatch<false>(smf, a.tw_df, P, st, tid, T);
    __syncthreads();
  }
#endif
  PHASE(2);
  pdl_wait();
  for (int f0 = tid; f0 <= M / 2; f0 += T * U) {
    float2 a1v[U], a2v[U];
    float4 b1v[U], b2v[U], z1v[U], z2v[U];
    bool ok[U], two[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int f = f0 + u * T;
      ok[u] = f <= M / 2;
      const int ff = ok[u] ? f : 0;
      const int fm = ff == 0 ? 0 : M - ff;
      two[u] = fm != ff;
      a1v[u] = __ldg(&A[ff]);
      b1v[u] = __ldg(&B[ff]);
      a2v[u] = __ldg(&A[fm]);
      b2v[u] = __ldg(&B[fm]);
      z1v[u] = smf[ff];
      z2v[u] = smf[fm];
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      if (!ok[u]) continue;
      const int f = f0 + u * T;
      const int fm = f == 0 ? 0 : M - f;
      const cdf z1 = cdf_load(z1v[u]), z2 = cdf_load(z2v[u]);
      const df a1 = df{a1v[u].x, a1v[u].y};
      const cdf b1 = cdf_load(b1v[u]);
      const cdf y1 = cdf_add(cdf{df_mul(a1, z1.re), df_mul(a1, z1.im)}, cdf_mul(z2, b1));
      if (two[u]) {
        const df a2 = df{a2v[u].x, a2v[u].y};
        const cdf b2 = cdf_load(b2v[u]);
        const cdf y2 = cdf_add(cdf{df_mul(a2, z2.re), df_mul(a2, z2.im)}, cdf_mul(z1, b2));
        smf[f] = cdf_store(y1);
        smf[fm] = cdf_store(y2);
      } else {
        smf[f] = cdf_store(y1);
      }
    }
  }
  __syncthreads();
  PHASE(3);
#ifndef PB_ABL_NOFFT
  for (int st = P.S - 1; st >= 0; st--) {
    fft_stage_df_dispatch<true>(smf, a.tw_df, P, st, tid, T);
    __syncthreads();
  }
#endif
  PHASE(4);
  if (MODE == 0) {
    if (!special) {
      for (int jp0 = tid + 1; jp0 <= N; jp0 += T * U) {
        int pp[U];
        float4 v[U];
#pragma unroll
        for (int u = 0; u < U; u++) pp[u] = (jp0 + u * T <= N) ? __ldg(&a.perm[jp0 + u * T]) : 0;
#pragma unroll
        for (int u = 0; u < U; u++) v[u] = smf[pp[u]];
#pragma unroll
        for (int u = 0; u < U; u++) {
          const int jp = jp0 + u * T;
          if (jp <= N) {
            float2* dst = a.T1df + (size_t)(jp - 1) * a.ldT + l1;
            if (has2) *reinterpret_cast<float4*>(dst) = v[u];
            else dst[0] = make_float2(v[u].x, v[u].y);
          }
        }
      }
      if (tid < 4) {
        const int row = (tid & 1) ? l2 : l1;
        if (row < N) a.d_out[(size_t)(row + 1) * n1 + ((tid & 2) ? (n1 - 1) : 0)] = 1.0;
      }
    } else {
      for (int jp = tid + 1; jp <= N; jp += T) {
        const float4 v = smf[a.perm[jp]];
        a.pwldf[jp - 1] = make_float2(v.x, v.y);
        a.pwrdf[jp - 1] = make_float2(v.z, v.w);
      }
    }
  } else {
    const df e1a = df_from_double_int(a.e1[l1]), e2a = df_from_double_int(a.e1[N - 1 - l1]);
    df e1b = {0.0f, 0.0f}, e2b = {0.0f, 0.0f};
    if (has2) {
      e1b = df_from_double_int(a.e1[l2]);
      e2b = df_from_double_int(a.e1[N - 1 - l2]);
    }
    for (int jp0 = tid + 1; jp0 <= N; jp0 += T * U) {
      int pp[U];
      float2 wl[U], wr[U];
      float4 v[U];
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int jp = jp0 + u * T;
        const bool ok = jp <= N;
        pp[u] = ok ? __ldg(&a.perm[jp]) : 0;
        wl[u] = ok ? a.pwldf[jp - 1] : make_float2(0.f, 0.f);
        wr[u] = ok ? a.pwrdf[jp - 1] : make_float2(0.f, 0.f);
      }
#pragma unroll
      for (int u = 0; u < U; u++) v[u] = smf[pp[u]];
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int jp = jp0 + u * T;
        if (jp > N) continue;
        const cdf z = cdf_load(v[u]);
        const df wld = df_load2(wl[u]), wrd = df_load2(wr[u]);
        const int j = jp - 1;
        df w1 = df_add(df_add(z.re, df_mul(wld, e1a)), df_mul(wrd, e2a));
        a.d_out[(size_t)(j + 1) * n1 + (l1 + 1)] = df_to_double_int(df_add(w1, one_df));
        if (has2) {
          df w2 = df_add(df_add(z.im, df_mul(wld, e1b)), df_mul(wrd, e2b));
          a.d_out[(size_t)(j + 1) * n1 + (l2 + 1)] = df_to_double_int(df_add(w2, one_df));
        }
      }
    }
  }
  PHASE(5);
}

// ---------------------------------------------------------------------------------------------------------------
// REPLICA kernels: bit-exact transcription of baseline/compute.inc (IEEE intrinsics, no FMA contraction)
// ---------------------------------------------------------------------------------------------------------------
struct RepConst {
  double a, b, c, d, e, f, coef_d, coef_a;
};

__global__ void rep_tables_kernel(int n0, int n1, RepConst* K, double* p1, double* den1, double* p2, double* den2) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double t0 = __ddiv_rn(1.0, (double)n1);
  double t1 = __ddiv_rn(1.0, (double)n1);
  double t2 = __ddiv_rn(1.0, (double)n0);
  double t3 = 2.0, t4 = 1.0;
  double t5 = __ddiv_rn(__dmul_rn(t3, t2), __dmul_rn(t0, t0));
  double t6 = __ddiv_rn(__dmul_rn(t4, t2), __dmul_rn(t1, t1));
  double a = __ddiv_rn(-t5, 2.0);
  double b = __dadd_rn(1.0, t5);
  double c = a;
  double d = __ddiv_rn(-t6, 2.0);
  double e = __dadd_rn(1.0, t6);
  double f = d;
  RepConst k;
  k.a = a; k.b = b; k.c = c; k.d = d; k.e = e; k.f = f;
  k.coef_d = __dadd_rn(1.0, __dmul_rn(2.0, d));
  k.coef_a = __dadd_rn(1.0, __dmul_rn(2.0, a));
  *K = k;
  double pp1 = 0.0, pp2 = 0.0;
  p1[0] = 0.0;
  p2[0] = 0.0;
  for (int j = 1; j <= n1 - 2; j++) {
    double dn1 = __dadd_rn(__dmul_rn(a, pp1), b);
    double dn2 = __dadd_rn(__dmul_rn(d, pp2), e);
    den1[j] = dn1;
    den2[j] = dn2;
    pp1 = __ddiv_rn(-c, dn1);
    pp2 = __ddiv_rn(-f, dn2);
    p1[j] = pp1;
    p2[j] = pp2;
  }
}

// sweep A: thread per column i; reads u (buf0), writes v (buf1) and q scratch (buf3)
__global__ void rep_sweepA(int n1, const double* __restrict__ u, double* __restrict__ V, double* __restrict__ Q,
                           const RepConst* __restrict__ K, const double* __restrict__ p1, const double* __restrict__ den1) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > n1 - 2) return;
  const RepConst k = *K;
  const double md = -k.d;
  V[i] = 1.0;                              // buf1[0][i]
  V[(size_t)(n1 - 1) * n1 + i] = 1.0;      // buf1[n1-1][i]
  double qprev = 1.0;                      // buf3[i][0] = buf1[0][i]
  for (int j = 1; j <= n1 - 2; j++) {
    const size_t r = (size_t)j * n1;
    const double u1 = u[r + i - 1], u2 = u[r + i], u3 = u[r + i + 1];
    double s1 = __dadd_rn(__dmul_rn(md, u1), __dmul_rn(k.coef_d, u2));
    double s2 = __dsub_rn(s1, __dmul_rn(k.f, u3));
    double s3 = __dsub_rn(s2, __dmul_rn(k.a, qprev));
    double q = __ddiv_rn(s3, den1[j]);
    Q[r + i] = q;
    qprev = q;
  }
  double vnext = 1.0;
  for (int j = n1 - 2; j >= 1; j--) {
    const size_t r = (size_t)j * n1;
    double v = __dadd_rn(__dmul_rn(p1[j], vnext), Q[r + i]);
    V[r + i] = v;
    vnext = v;
  }
}

// sweep B: thread per row i; reads v (buf1), updates u (buf0) in place
__global__ void rep_sweepB(int n1, double* __restrict__ u, const double* __restrict__ V, double* __restrict__ Q,
                           const RepConst* __restrict__ K, const double* __restrict__ p2, const double* __restrict__ den2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > n1 - 2) return;
  const RepConst k = *K;
  const double ma = -k.a;
  const size_t ri = (size_t)i * n1;
  u[ri + 0] = 1.0;
  double qprev = 1.0;  // buf3[i][0] = buf0[i][0]
  for (int j = 1; j <= n1 - 2; j++) {
    const double v1 = V[(size_t)(i - 1) * n1 + j], v2 = V[ri + j], v3 = V[(size_t)(i + 1) * n1 + j];
    double s1 = __dadd_rn(__dmul_rn(ma, v1), __dmul_rn(k.coef_a, v2));
    double s2 = __dsub_rn(s1, __dmul_rn(k.c, v3));
    double s3 = __dsub_rn(s2, __dmul_rn(k.d, qprev));
    double q = __ddiv_rn(s3, den2[j]);
    Q[ri + j] = q;
    qprev = q;
  }
  u[ri + n1 - 1] = 1.0;
  double unext = 1.0;
  for (int j = n1 - 2; j >= 1; j--) {
    double w = __dadd_rn(__dmul_rn(p2[j], unext), Q[ri + j]);
    u[ri + j] = w;
    unext = w;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// host: state, plan construction
// ---------------------------------------------------------------------------------------------------------------
namespace {

enum Mode { MODE_NONE = 0, MODE_TRIVIAL = 1, MODE_REPLICA = 2, MODE_SPECTRAL = 3 };

struct State {
  bool ready = false;
  cudaStream_t stream = nullptr;
  int n0 = 0, n1 = 0, N = 0;
  Mode mode = MODE_NONE;
  size_t nn = 0;  // n1*n1
  // common
  double* d_in = nullptr;
  double* d_out = nullptr;
  // replica
  double *d_V = nullptr, *d_Q = nullptr, *d_p1 = nullptr, *d_den1 = nullptr, *d_p2 = nullptr, *d_den2 = nullptr;
  RepConst* d_K = nullptr;
  // spectral
  int L = 0, M = 0, T = 256, ldT = 0;
  size_t smem = 0;
  FftParams fp;
  double2* d_tw = nullptr;
  int* d_perm = nullptr;
  double *d_cosT = nullptr, *d_sinT = nullptr, *d_shi = nullptr, *d_slo = nullptr, *d_chi = nullptr, *d_clo = nullptr;
  double *d_pair = nullptr, *d_kP = nullptr, *d_kQ = nullptr, *d_e1 = nullptr;
  double* d_A = nullptr;
  double2* d_B = nullptr;
  bool use_df = false;
  float4* d_twdf = nullptr;
  float2* d_Adf = nullptr;
  float4* d_Bdf = nullptr;
  double *d_T1 = nullptr, *d_pw = nullptr;
  float2 *d_T1df = nullptr, *d_pwdf = nullptr;
};
State g;

const int kMaxM = 13824;  // 16*M bytes of shared memory must fit in the per-block opt-in limit

bool smooth5(long long m) {
  for (int p : {2, 3, 5})
    while (m % p == 0) m /= p;
  return m == 1;
}
// DP-instruction cost (per packed line pair) of the line-convolution FFT pair for size M: forward + inverse stages
// plus the 6-op frequency-domain operator.  Radix order matches build_spectral_plan (5s, 3s, 4s, then a 2).
double fft_cost(int M, bool df) {
  int m = M, a2 = 0, a3 = 0, a5 = 0;
  while (m % 2 == 0) { m /= 2; a2++; }
  while (m % 3 == 0) { m /= 3; a3++; }
  while (m % 5 == 0) { m /= 5; a5++; }
  std::vector<int> rad;
  for (int i = 0; i < a5; i++) rad.push_back(5);
  for (int i = 0; i < a3; i++) rad.push_back(3);
  for (int i = 0; i < a2 / 2; i++) rad.push_back(4);
  if (a2 % 2) rad.push_back(2);
  // FP64 instruction counts (butterfly + twiddles) / FP32 instruction counts of the df64 butterflies
  const double bfD[6] = {0, 0, 8, 24, 28, 52}, bf0D[6] = {0, 0, 4, 16, 16, 36};
  const double bfF[6] = {0, 0, 88, 234, 308, 640}, bf0F[6] = {0, 0, 44, 146, 176, 464};
  const double* bf = df ? bfF : bfD;
  const double* bf0 = df ? bf0F : bf0D;
  double c = 0;
  for (size_t s = 0; s < rad.size(); s++) c += (double)(M / rad[s]) * (s == 0 ? bf0[rad[s]] : bf[rad[s]]);
  return 2.0 * c + (df ? 80.0 : 6.0) * M;
}
// cheapest 5-smooth FFT size >= minM that fits in shared memory (0 if none)
int pick_M(long long minM, bool df) {
  long long lo = std::max<long long>(minM, 2);
  int best = 0;
  double bc = 1e300;
  for (long long m = lo; m <= kMaxM; m++) {
    if (!smooth5(m)) continue;
    const double c = fft_cost((int)m, df);
    if (c < bc) { bc = c; best = (int)m; }
    if (m > 2 * lo + 64) break;
  }
  return best;
}

// The df64 (two-FP32) line passes are ~3.3x faster than the FP64 ones but carry ~1-4e-14 absolute error.  They are only
// used where the CPU baseline's own deviation from the exact result is orders of magnitude larger.  On every shape
// measured (n1 = 33..3000, n0 = 1..1000) that deviation was at least  0.1 * eps * rho * sqrt(min(n0, 2 rho)),
// rho = n1^2 / n0 (rounding noise of the (1+2a) v[i] cancellation, accumulated over the steps that do not damp it).
// We require this lower estimate to exceed 5e-13 (>= ~14x the df64 error), so the acceptance margin is >= 10x.
bool want_df(int n0, int n1) {
  if (n1 < 300) return false;
  const double eps = 2.220446049250313e-16;
  const double rho = (double)n1 * (double)n1 / (double)n0;
  const double est = 0.1 * eps * rho * std::sqrt(std::min((double)n0, 2.0 * rho));
  return est >= 5e-13;
}

// The replica reproduces the baseline bit for bit, so its error equals the baseline's (acceptance factor needed ~1);
// the spectral path is 10-1000x more accurate, so it is preferred everywhere except the few tiny shapes (n1 <= 5, i.e.
// N <= 3) where its absolute error (~5e-16) is comparable to the baseline's and the replica is cheaper.
bool want_replica(int n0, int N) { return N >= 1 && N <= 3 && n0 <= 16; }

void free_all() {
  if (g.stream) {
    cudaStreamSynchronize(g.stream);
    cudaStreamDestroy(g.stream);
    g.stream = nullptr;
  }
  auto F = [](auto*& p) {
    if (p) cudaFree((void*)p);
    p = nullptr;
  };
  F(g.d_in); F(g.d_out);
  F(g.d_V); F(g.d_Q); F(g.d_p1); F(g.d_den1); F(g.d_p2); F(g.d_den2); F(g.d_K);
  F(g.d_tw); F(g.d_perm); F(g.d_cosT); F(g.d_sinT); F(g.d_shi); F(g.d_slo); F(g.d_chi); F(g.d_clo);
  F(g.d_twdf); F(g.d_Adf); F(g.d_Bdf); F(g.d_pair); F(g.d_kP); F(g.d_kQ); F(g.d_e1); F(g.d_A); F(g.d_B); F(g.d_T1); F(g.d_pw); F(g.d_T1df); F(g.d_pwdf);
  g.ready = false;
  g.mode = MODE_NONE;
}

template <typename T>
cudaError_t upload_vec(T** dptr, const std::vector<T>& v) {
  cudaError_t e = cudaMalloc((void**)dptr, std::max<size_t>(v.size(), 1) * sizeof(T));
  if (e != cudaSuccess) return e;
  if (!v.empty()) e = cudaMemcpy(*dptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return e;
}

int build_spectral_plan() {
  const int N = g.N, L = g.L, M = g.M;
  // ---- FFT plan: radices, per-stage twiddles, digit-reversal permutation ----
  int m = M, a2 = 0, a3 = 0, a5 = 0;
  while (m % 2 == 0) { m /= 2; a2++; }
  while (m % 3 == 0) { m /= 3; a3++; }
  while (m % 5 == 0) { m /= 5; a5++; }
  std::vector<int> rad;
  for (int i = 0; i < a5; i++) rad.push_back(5);
  for (int i = 0; i < a3; i++) rad.push_back(3);
  for (int i = 0; i < a2 / 2; i++) rad.push_back(4);
  if (a2 % 2) rad.push_back(2);
  const int S = (int)rad.size();
  if (S > PB_MAX_STAGES) return 1;
  memset(&g.fp, 0, sizeof(g.fp));
  g.fp.M = M;
  g.fp.S = S;
  size_t off = 0;
  int mp = 1;
  for (int s = 0; s < S; s++) {
    g.fp.rad[s] = rad[s];
    g.fp.mprev[s] = mp;
    g.fp.twoff[s] = (int)off;
    off += (size_t)(rad[s] - 1) * mp;
    mp *= rad[s];
  }
  const long double PI = 3.14159265358979323846264338327950288L;
  std::vector<double2> tw(std::max<size_t>(off, 1));
  for (int s = 0; s < S; s++) {
    const int R = rad[s], mpv = g.fp.mprev[s], ms = R * mpv;
    for (int q = 1; q < R; q++)
      for (int n = 0; n < mpv; n++) {
        long double ang = -2.0L * PI * (long double)((long long)n * q % ms) / (long double)ms;
        tw[g.fp.twoff[s] + (size_t)(q - 1) * mpv + n] = make_double2((double)cosl(ang), (double)sinl(ang));
      }
  }
  std::vector<int> perm(M);
  for (int j = 0; j < M; j++) {
    int jj = j, p = 0;
    for (int s = S - 1; s >= 0; s--) {
      p += (jj % rad[s]) * g.fp.mprev[s];
      jj /= rad[s];
    }
    perm[j] = p;
  }
  // ---- cos table cos(pi t / L), t in [0, 2L) ; sin(pi k / (2L)) as double-double, k = 1..N ----
  std::vector<double> cosT(2 * (size_t)L), sinT(2 * (size_t)L), shi(N), slo(N), chi(N), clo(N);
  for (int t = 0; t < 2 * L; t++) {
    cosT[t] = (double)cosl(PI * (long double)t / (long double)L);
    sinT[t] = (double)sinl(PI * (long double)t / (long double)L);
  }
  for (int k = 1; k <= N; k++) {
    const long double ang = PI * (long double)k / (2.0L * (long double)L);
    long double sv = sinl(ang), cv = cosl(ang);
    double hi = (double)sv;
    shi[k - 1] = hi;
    slo[k - 1] = (double)(sv - (long double)hi);
    hi = (double)cv;
    chi[k - 1] = hi;
    clo[k - 1] = (double)(cv - (long double)hi);
  }
  PB_CK(upload_vec(&g.d_tw, tw));
  if (g.use_df) {
    std::vector<float4> twdf(tw.size());
    for (size_t i = 0; i < tw.size(); i++) {
      const float rh = (float)tw[i].x, ih = (float)tw[i].y;
      twdf[i] = make_float4(rh, (float)(tw[i].x - (double)rh), ih, (float)(tw[i].y - (double)ih));
    }
    PB_CK(upload_vec(&g.d_twdf, twdf));
    PB_CK(cudaMalloc((void**)&g.d_Adf, (size_t)2 * M * sizeof(float2)));
    PB_CK(cudaMalloc((void**)&g.d_Bdf, (size_t)2 * M * sizeof(float4)));
  }
  PB_CK(upload_vec(&g.d_perm, perm));
  PB_CK(upload_vec(&g.d_cosT, cosT));
  PB_CK(upload_vec(&g.d_sinT, sinT));
  PB_CK(upload_vec(&g.d_shi, shi));
  PB_CK(upload_vec(&g.d_slo, slo));
  PB_CK(upload_vec(&g.d_chi, chi));
  PB_CK(upload_vec(&g.d_clo, clo));
  // ---- work arrays ----
  g.ldT = (N + 1) & ~1;
  PB_CK(cudaMalloc((void**)&g.d_pair, ((size_t)6 * (N / 2) + 8) * sizeof(double)));
  PB_CK(cudaMalloc((void**)&g.d_kP, (size_t)(2 * N + 1) * sizeof(double)));
  PB_CK(cudaMalloc((void**)&g.d_kQ, (size_t)(2 * N + 1) * sizeof(double)));
  PB_CK(cudaMalloc((void**)&g.d_e1, (size_t)N * sizeof(double)));
  PB_CK(cudaMalloc((void**)&g.d_A, (size_t)2 * M * sizeof(double)));
  PB_CK(cudaMalloc((void**)&g.d_B, (size_t)2 * M * sizeof(double2)));
  if (g.use_df) {
    PB_CK(cudaMalloc((void**)&g.d_T1df, (size_t)N * g.ldT * sizeof(float2)));
    PB_CK(cudaMalloc((void**)&g.d_pwdf, (size_t)2 * N * sizeof(float2)));
  } else {
    PB_CK(cudaMalloc((void**)&g.d_T1, (size_t)N * g.ldT * sizeof(double)));
    PB_CK(cudaMalloc((void**)&g.d_pw, (size_t)2 * N * sizeof(double)));
  }
  g.smem = (size_t)M * sizeof(double2);
  g.T = M >= 8192 ? 512 : 256;
  PB_CK(cudaFuncSetAttribute(pass_kernel<0, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g.smem));
  PB_CK(cudaFuncSetAttribute(pass_kernel<1, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g.smem));
  PB_CK(cudaFuncSetAttribute(pass_kernel_df<0, 256, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g.smem));
  PB_CK(cudaFuncSetAttribute(pass_kernel_df<0, 256, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g.smem));
  PB_CK(cudaFuncSetAttribute(pass_kernel_df<1, 256, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g.smem));
  PB_CK(cudaFuncSetAttribute(pass_kernel_df<1, 256, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g.smem));
  PB_CK(cudaFuncSetAttribute(pass_kernel<0, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g.smem));
  PB_CK(cudaFuncSetAttribute(pass_kernel<1, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g.smem));
  PB_CK(cudaFuncSetAttribute(pass_kernel_df<0, 512, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g.smem));
  PB_CK(cudaFuncSetAttribute(pass_kernel_df<1, 512, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g.smem));
  PB_CK(cudaFuncSetAttribute(kernel_spectrum_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g.smem));
  PB_CK(cudaFuncSetAttribute(kernel_spectrum_kernel_df, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g.smem));
  return 0;
}



// Launch with programmatic stream serialization: the kernel may start while its predecessor is still running and must
// call pdl_wait() before consuming the predecessor's results.
template <typename... KArgs, typename... Args>
static cudaError_t launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = g.stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kernel, args...);
}

template <int MODE, bool DF>
static cudaError_t launch_pass(int blocks, int T, size_t smem, const PassArgs& a, bool pdl) {
  const dim3 grid(blocks), block(T);
  if (T <= 256) {
    if (DF) {
      // blocks resident per SM are limited by shared memory (3 above 56 KB, else 4): let registers allow exactly that
      if (smem > 57344) {
        if (pdl) return launch_pdl(pass_kernel_df<MODE, 256, 3>, grid, block, smem, a);
        pass_kernel_df<MODE, 256, 3><<<grid, block, smem, g.stream>>>(a);
      } else {
        if (pdl) return launch_pdl(pass_kernel_df<MODE, 256, 4>, grid, block, smem, a);
        pass_kernel_df<MODE, 256, 4><<<grid, block, smem, g.stream>>>(a);
      }
    } else {
      if (pdl) return launch_pdl(pass_kernel<MODE, 256>, grid, block, smem, a);
      pass_kernel<MODE, 256><<<grid, block, smem, g.stream>>>(a);
    }
  } else {
    if (DF) {
      if (pdl) return launch_pdl(pass_kernel_df<MODE, 512, 1>, grid, block, smem, a);
      pass_kernel_df<MODE, 512, 1><<<grid, block, smem, g.stream>>>(a);
    } else {
      if (pdl) return launch_pdl(pass_kernel<MODE, 512>, grid, block, smem, a);
      pass_kernel<MODE, 512><<<grid, block, smem, g.stream>>>(a);
    }
  }
  return cudaGetLastError();
}

int run_replica() {
  const int n0 = g.n0, n1 = g.n1, N = g.N;
  PB_CK(cudaMemcpyAsync(g.d_out, g.d_in, g.nn * sizeof(double), cudaMemcpyDeviceToDevice, g.stream));
  rep_tables_kernel<<<1, 1, 0, g.stream>>>(n0, n1, g.d_K, g.d_p1, g.d_den1, g.d_p2, g.d_den2);
  const int tpb = N >= 4096 ? 64 : 32;
  const int blocks = (N + tpb - 1) / tpb;
  for (int t = 1; t <= n0; t++) {
    rep_sweepA<<<blocks, tpb, 0, g.stream>>>(n1, g.d_out, g.d_V, g.d_Q, g.d_K, g.d_p1, g.d_den1);
    rep_sweepB<<<blocks, tpb, 0, g.stream>>>(n1, g.d_out, g.d_V, g.d_Q, g.d_K, g.d_p2, g.d_den2);
  }
  PB_CK(cudaGetLastError());
  return 0;
}

int run_spectral() {
  const int n0 = g.n0, n1 = g.n1, N = g.N, L = g.L, M = g.M;
  PB_MARK("start");
  mult_kernel<<<(4 * (N / 2 + 1) + 127) / 128, 128, 0, g.stream>>>(n0, n1, N, L, M, g.d_shi, g.d_slo, g.d_chi, g.d_clo, g.d_pair);
  PB_MARK("mult");
  PB_CK(launch_pdl(kappa_kernel, dim3(((L + N + 1) + 7) / 8), dim3(256), (size_t)0, N, L, (const double*)g.d_cosT,
                   (const double*)g.d_sinT, (const double*)g.d_pair, g.d_kP, g.d_kQ, g.d_e1));
  PB_MARK("kappa");
  if (g.use_df)
    PB_CK(launch_pdl(kernel_spectrum_kernel_df, dim3(2), dim3(512), g.smem, g.fp, N, (const float4*)g.d_twdf, (const int*)g.d_perm,
                     (const double*)g.d_kP, (const double*)g.d_kQ, g.d_Adf, g.d_Bdf));
  else
    PB_CK(launch_pdl(kernel_spectrum_kernel, dim3(2), dim3(512), g.smem, g.fp, N, (const double2*)g.d_tw, (const int*)g.d_perm,
                     (const double*)g.d_kP, (const double*)g.d_kQ, g.d_A, g.d_B));
  PB_MARK("kspec");
  PassArgs a;
  a.P = g.fp;
  a.N = N;
  a.n1 = n1;
  a.ldT = g.ldT;
  a.tw = g.d_tw;
  a.perm = g.d_perm;
  a.A_all = g.d_A;
  a.B_all = g.d_B;
  a.d_in = g.d_in;
  a.d_out = g.d_out;
  a.T1 = g.d_T1;
  a.pwl = g.d_pw;
  a.pwr = g.d_pw ? g.d_pw + N : nullptr;
  a.e1 = g.d_e1;
  a.tw_df = g.d_twdf;
  a.A_df = g.d_Adf;
  a.B_df = g.d_Bdf;
  a.T1df = g.d_T1df;
  a.pwldf = g.d_pwdf;
  a.pwrdf = g.d_pwdf ? g.d_pwdf + N : nullptr;
  const int nb = (N + 1) / 2;
  if (g.use_df) {
    PB_CK((launch_pass<0, true>(nb + 1, g.T, g.smem, a, true)));
    PB_MARK("pass1(rows)");
    PB_CK((launch_pass<1, true>(nb, g.T, g.smem, a, false)));
  } else {
    // FP64 passes compete with the (FP64) setup kernels for the same pipe: start early only when they are short
    PB_CK((launch_pass<0, false>(nb + 1, g.T, g.smem, a, M <= 512)));
    PB_MARK("pass1(rows)");
    PB_CK((launch_pass<1, false>(nb, g.T, g.smem, a, false)));
  }
  PB_MARK("pass2(cols)");
  PB_CK(cudaGetLastError());
  return 0;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// entry points
// ---------------------------------------------------------------------------------------------------------------
extern "C" {

int pb_setup(const pb_problem_shape* shape) {
  free_all();  // release anything left over from a previous (possibly failed) setup
  if (!shape || shape->abi_version != PB_SOLVE_ABI_VERSION || shape->dimension_count != 2 || shape->scalar_count != 0 ||
      shape->buffer_count != 4 || !shape->dimensions || !shape->buffers)
    return 1;
  const int64_t d0 = shape->dimensions[0], d1 = shape->dimensions[1];
  if (d0 <= 0 || d0 > INT_MAX || d1 <= 0 || d1 > INT_MAX) return 1;
  for (int i = 0; i < 4; i++) {
    const pb_buffer_shape& b = shape->buffers[i];
    if (b.dtype != PB_F64 || b.rank != 2 || b.shape[0] != d1 || b.shape[1] != d1) return 1;
  }
  g.n0 = (int)d0;
  g.n1 = (int)d1;
  g.N = g.n1 - 2;
  g.nn = (size_t)g.n1 * (size_t)g.n1;
  g.L = g.n1 - 1;
  PB_CK(cudaFree(0));  // make sure the context exists
  PB_CK(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
#ifdef PB_DEV_TIMING
  g_dev_stream = g.stream;
#endif
  PB_CK(cudaMalloc((void**)&g.d_in, g.nn * sizeof(double)));
  PB_CK(cudaMalloc((void**)&g.d_out, g.nn * sizeof(double)));
  if (g.N < 1) {
    g.mode = MODE_TRIVIAL;
  } else {
    g.use_df = want_df(g.n0, g.n1);
#ifdef PB_DEV_FORCE_MODE
    if (PB_DEV_FORCE_MODE == 3) g.use_df = false;
    if (PB_DEV_FORCE_MODE == 4) g.use_df = true;
#endif
    g.M = pick_M(2LL * g.N, g.use_df);
    const bool spectral_ok = g.M > 0;
    if (want_replica(g.n0, g.N) || !spectral_ok) g.mode = MODE_REPLICA;
    else g.mode = MODE_SPECTRAL;
#ifdef PB_DEV_FORCE_MODE
    if (PB_DEV_FORCE_MODE == 2) g.mode = MODE_REPLICA;
    if (PB_DEV_FORCE_MODE == 3 && spectral_ok) g.mode = MODE_SPECTRAL;
    if (PB_DEV_FORCE_MODE == 4 && spectral_ok) g.mode = MODE_SPECTRAL;
#endif
  }
  if (g.mode == MODE_REPLICA) {
    PB_CK(cudaMalloc((void**)&g.d_V, g.nn * sizeof(double)));
    PB_CK(cudaMalloc((void**)&g.d_Q, g.nn * sizeof(double)));
    PB_CK(cudaMalloc((void**)&g.d_p1, (size_t)g.n1 * sizeof(double)));
    PB_CK(cudaMalloc((void**)&g.d_den1, (size_t)g.n1 * sizeof(double)));
    PB_CK(cudaMalloc((void**)&g.d_p2, (size_t)g.n1 * sizeof(double)));
    PB_CK(cudaMalloc((void**)&g.d_den2, (size_t)g.n1 * sizeof(double)));
    PB_CK(cudaMalloc((void**)&g.d_K, sizeof(RepConst)));
  } else if (g.mode == MODE_SPECTRAL) {
    int rc = build_spectral_plan();
    if (rc) return rc;
  }
  // warm-up: load every kernel once on scratch contents (results are discarded; upload overwrites both buffers)
  if (g.mode != MODE_TRIVIAL) {
    PB_CK(cudaMemsetAsync(g.d_in, 0, g.nn * sizeof(double), g.stream));
    int rc = g.mode == MODE_REPLICA ? run_replica() : run_spectral();
    if (rc) return rc;
    PB_CK(cudaStreamSynchronize(g.stream));
    PB_CK(cudaDeviceSynchronize());
  }
#ifdef PB_DEV_TIMING
  fprintf(stderr, "  [plan] n0=%d n1=%d mode=%d M=%d df=%d\n", g.n0, g.n1, (int)g.mode, g.M, (int)g.use_df);
#endif
  g.ready = true;
  return 0;
}

int pb_upload(const pb_problem* problem, pb_buffer* buffers) {
  if (!g.ready || !problem || !buffers || !problem->dimensions || problem->dimension_count != 2) return 1;
  if (problem->dimensions[0] != g.n0 || problem->dimensions[1] != g.n1) return 1;
  if (!buffers[0].data || buffers[0].nbytes != g.nn * sizeof(double)) return 1;
  PB_CK(cudaMemcpyAsync(g.d_in, buffers[0].data, g.nn * sizeof(double), cudaMemcpyHostToDevice, g.stream));
  PB_CK(cudaMemcpyAsync(g.d_out, buffers[0].data, g.nn * sizeof(double), cudaMemcpyHostToDevice, g.stream));
  PB_CK(cudaStreamSynchronize(g.stream));
  return 0;
}

int pb_compute(void) {
  if (!g.ready) return 1;
  int rc = 0;
  if (g.mode == MODE_REPLICA) rc = run_replica();
  else if (g.mode == MODE_SPECTRAL) rc = run_spectral();
  if (rc) return rc;
  PB_CK(cudaGetLastError());
  PB_CK(cudaStreamSynchronize(g.stream));
  PB_REPORT();
  return 0;
}

int pb_download(pb_buffer* buffers) {
  if (!g.ready || !buffers || !buffers[0].data || buffers[0].nbytes != g.nn * sizeof(double)) return 1;
  PB_CK(cudaMemcpyAsync(buffers[0].data, g.d_out, g.nn * sizeof(double), cudaMemcpyDeviceToHost, g.stream));
  PB_CK(cudaStreamSynchronize(g.stream));
  return 0;
}

int pb_teardown(void) {
  free_all();
  PB_CK(cudaGetLastError());
  return 0;
}

}  // extern "C"
