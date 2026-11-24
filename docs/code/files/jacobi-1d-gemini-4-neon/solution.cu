#include "interface.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr float C_F32 = 0.33333f;
static constexpr double C_F64 = (double)0.33333f;
static constexpr float INV_72 = 1.0f / 72.0f;
static constexpr int MAX_K_BND = 8192;
static constexpr int DIRECT_MAX_N0 = 14;
static constexpr int DIRECT_KPT = 24;
static constexpr int DIRECT_WARP_ELEMS = 32 * DIRECT_KPT; // 768
static constexpr int64_t S_SINGLE_PASS_MAX = 4000000LL;
static constexpr int64_t S_STEADY_MIN = 11500000LL;

// ============================================================================
// Complex FP32 & Radix Butterflies (float2 for native LDS.64 / STS.64)
// ============================================================================

__device__ __forceinline__ float2 cadd(float2 a, float2 b) { return {a.x + b.x, a.y + b.y}; }
__device__ __forceinline__ float2 csub(float2 a, float2 b) { return {a.x - b.x, a.y - b.y}; }
__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return {fmaf(a.x, b.x, -a.y * b.y), fmaf(a.x, b.y, a.y * b.x)};
}
__device__ __forceinline__ float2 cmul_conj(float2 a, float2 b) {
    return {fmaf(a.x, b.x, a.y * b.y), fmaf(a.y, b.x, -a.x * b.y)};
}

template <int DIR>
__device__ __forceinline__ void radix4(float2 &x0, float2 &x1, float2 &x2, float2 &x3) {
    float2 a0 = cadd(x0, x2), a1 = csub(x0, x2), a2 = cadd(x1, x3), a3 = csub(x1, x3);
    float2 a3i = (DIR == -1) ? float2{a3.y, -a3.x} : float2{-a3.y, a3.x};
    x0 = cadd(a0, a2); x1 = cadd(a1, a3i); x2 = csub(a0, a2); x3 = csub(a1, a3i);
}

__device__ __forceinline__ void radix4_fwd_03(const float2 x[4], float2 &y0, float2 &y3) {
    float2 a0 = cadd(x[0], x[2]), a1 = csub(x[0], x[2]);
    float2 a2 = cadd(x[1], x[3]), a3 = csub(x[1], x[3]);
    y0 = cadd(a0, a2);
    y3 = {a1.x - a3.y, a1.y + a3.x};
}

__device__ __forceinline__ void radix4_inv_03(float2 x0, float2 x3, float2 y[4]) {
    float2 t1 = {x3.y, -x3.x};
    y[0] = cadd(x0, x3); y[2] = csub(x0, x3);
    y[1] = cadd(x0, t1); y[3] = csub(x0, t1);
}

template <int DIR>
__device__ __forceinline__ void radix16(float2 x[16]) {
    constexpr float C1 = 0.92387953251128675613f;
    constexpr float S1 = 0.38268343236508977173f;
    constexpr float SQRT1_2 = 0.70710678118654752440f;
    radix4<DIR>(x[0], x[4], x[8],  x[12]);
    radix4<DIR>(x[1], x[5], x[9],  x[13]);
    radix4<DIR>(x[2], x[6], x[10], x[14]);
    radix4<DIR>(x[3], x[7], x[11], x[15]);
    if constexpr (DIR == -1) {
        x[5]  = {fmaf(x[5].x,  C1,  x[5].y  * S1), fmaf(x[5].y,  C1, -x[5].x  * S1)};
        x[6]  = {SQRT1_2 * (x[6].x  + x[6].y),  SQRT1_2 * (x[6].y  - x[6].x)};
        x[7]  = {fmaf(x[7].x,  S1,  x[7].y  * C1), fmaf(x[7].y,  S1, -x[7].x  * C1)};
        x[9]  = {SQRT1_2 * (x[9].x  + x[9].y),  SQRT1_2 * (x[9].y  - x[9].x)};
        x[10] = {x[10].y, -x[10].x};
        x[11] = {SQRT1_2 * (x[11].y - x[11].x), -SQRT1_2 * (x[11].x + x[11].y)};
        x[13] = {fmaf(x[13].x, S1,  x[13].y * C1), fmaf(x[13].y, S1, -x[13].x * C1)};
        x[14] = {SQRT1_2 * (x[14].y - x[14].x), -SQRT1_2 * (x[14].x + x[14].y)};
        x[15] = {fmaf(-x[15].x, C1, -x[15].y * S1), fmaf(-x[15].y, C1,  x[15].x * S1)};
    } else {
        x[5]  = {fmaf(x[5].x,  C1, -x[5].y  * S1), fmaf(x[5].y,  C1,  x[5].x  * S1)};
        x[6]  = {SQRT1_2 * (x[6].x  - x[6].y),  SQRT1_2 * (x[6].x  + x[6].y)};
        x[7]  = {fmaf(x[7].x,  S1, -x[7].y  * C1), fmaf(x[7].y,  S1,  x[7].x  * C1)};
        x[9]  = {SQRT1_2 * (x[9].x  - x[9].y),  SQRT1_2 * (x[9].x  + x[9].y)};
        x[10] = {-x[10].y, x[10].x};
        x[11] = {-SQRT1_2 * (x[11].x + x[11].y), SQRT1_2 * (x[11].x - x[11].y)};
        x[13] = {fmaf(x[13].x, S1, -x[13].y * C1), fmaf(x[13].y, S1,  x[13].x * C1)};
        x[14] = {-SQRT1_2 * (x[14].x + x[14].y), SQRT1_2 * (x[14].x - x[14].y)};
        x[15] = {fmaf(-x[15].x, C1,  x[15].y * S1), fmaf(-x[15].y, C1, -x[15].x * S1)};
    }
    float2 y[16];
    y[0] = x[0];  y[4] = x[1];  y[8]  = x[2];  y[12] = x[3];  radix4<DIR>(y[0], y[4], y[8], y[12]);
    y[1] = x[4];  y[5] = x[5];  y[9]  = x[6];  y[13] = x[7];  radix4<DIR>(y[1], y[5], y[9], y[13]);
    y[2] = x[8];  y[6] = x[9];  y[10] = x[10]; y[14] = x[11]; radix4<DIR>(y[2], y[6], y[10], y[14]);
    y[3] = x[12]; y[7] = x[13]; y[11] = x[14]; y[15] = x[15]; radix4<DIR>(y[3], y[7], y[11], y[15]);
    #pragma unroll
    for (int i = 0; i < 16; i++) x[i] = y[i];
}

template <int DIR>
__device__ __forceinline__ void apply_tw16(float2 x[16], float2 w1) {
    if constexpr (DIR == +1) w1.y = -w1.y;
    float2 w2 = cmul(w1, w1), w3 = cmul(w1, w2), w4 = cmul(w2, w2);
    float2 w5 = cmul(w1, w4), w6 = cmul(w2, w4), w7 = cmul(w3, w4), w8 = cmul(w4, w4);
    x[1] = cmul(x[1], w1); x[2] = cmul(x[2], w2); x[3] = cmul(x[3], w3); x[4] = cmul(x[4], w4);
    x[5] = cmul(x[5], w5); x[6] = cmul(x[6], w6); x[7] = cmul(x[7], w7); x[8] = cmul(x[8], w8);
    x[9]  = cmul(x[9],  cmul(w8, w1)); x[10] = cmul(x[10], cmul(w8, w2));
    x[11] = cmul(x[11], cmul(w8, w3)); x[12] = cmul(x[12], cmul(w8, w4));
    x[13] = cmul(x[13], cmul(w8, w5)); x[14] = cmul(x[14], cmul(w8, w6));
    x[15] = cmul(x[15], cmul(w8, w7));
}

// Exact FP64-rounded LUT twiddles for boundary / small-grid kernel (31x lower rounding floor)
template <int DIR, int N>
__device__ __forceinline__ void apply_tw16_lut(float2 x[16], const float2 * __restrict__ tw, int step) {
    #pragma unroll
    for (int q = 1; q < 16; q++) {
        float2 w = __ldg(&tw[(q * step) & (N - 1)]);
        if constexpr (DIR == +1) x[q] = cmul_conj(x[q], w);
        else                     x[q] = cmul(x[q], w);
    }
}

__device__ __forceinline__ void radix8_fwd_07(const float2 x[8], float2 &y0, float2 &y7) {
    constexpr float SQRT1_2 = 0.70710678118654752440f;
    float2 s04 = cadd(x[0], x[4]), d04 = csub(x[0], x[4]);
    float2 s26 = cadd(x[2], x[6]), d26 = csub(x[2], x[6]);
    float2 s15 = cadd(x[1], x[5]), d15 = csub(x[1], x[5]);
    float2 s37 = cadd(x[3], x[7]), d37 = csub(x[3], x[7]);
    y0 = cadd(cadd(s04, s26), cadd(s15, s37));
    float2 e = {d04.x - d26.y, d04.y + d26.x};
    float ox = SQRT1_2 * ((d15.x - d37.x) - (d15.y + d37.y));
    float oy = SQRT1_2 * ((d15.x + d37.x) + (d15.y - d37.y));
    y7 = {e.x + ox, e.y + oy};
}

__device__ __forceinline__ void radix8_inv_07(float2 x0, float2 x7, float2 y[8]) {
    constexpr float SQRT1_2 = 0.70710678118654752440f;
    float2 t0 = x7;
    float2 t1 = {SQRT1_2 * (x7.x + x7.y), SQRT1_2 * (x7.y - x7.x)};
    float2 t2 = {x7.y, -x7.x};
    float2 t3 = {SQRT1_2 * (x7.y - x7.x), -SQRT1_2 * (x7.x + x7.y)};
    y[0] = cadd(x0, t0); y[4] = csub(x0, t0);
    y[1] = cadd(x0, t1); y[5] = csub(x0, t1);
    y[2] = cadd(x0, t2); y[6] = csub(x0, t2);
    y[3] = cadd(x0, t3); y[7] = csub(x0, t3);
}

__device__ __forceinline__ void radix16_fwd_0_15(const float2 x[16], float2 &y0, float2 &y15) {
    constexpr float C1 = 0.92387953251128675613f, S1 = 0.38268343236508977173f, SQRT1_2 = 0.70710678118654752440f;
    float2 s0 = cadd(cadd(x[0], x[8]),  cadd(x[4],  x[12]));
    float2 s1 = cadd(cadd(x[1], x[9]),  cadd(x[5],  x[13]));
    float2 s2 = cadd(cadd(x[2], x[10]), cadd(x[6],  x[14]));
    float2 s3 = cadd(cadd(x[3], x[11]), cadd(x[7],  x[15]));
    y0 = cadd(cadd(s0, s2), cadd(s1, s3));
    float2 a0 = csub(x[0], x[8]), b0 = csub(x[4], x[12]); float2 u0 = {a0.x - b0.y, a0.y + b0.x};
    float2 a1 = csub(x[1], x[9]), b1 = csub(x[5], x[13]); float2 u1 = {a1.x - b1.y, a1.y + b1.x};
    float2 a2 = csub(x[2], x[10]), b2 = csub(x[6], x[14]); float2 u2 = {a2.x - b2.y, a2.y + b2.x};
    float2 a3 = csub(x[3], x[11]), b3 = csub(x[7], x[15]); float2 u3 = {a3.x - b3.y, a3.y + b3.x};
    float2 t1 = {fmaf(u1.x, C1, -u1.y * S1), fmaf(u1.y, C1, u1.x * S1)};
    float2 t2 = {SQRT1_2 * (u2.x - u2.y), SQRT1_2 * (u2.x + u2.y)};
    float2 t3 = {fmaf(u3.x, S1, -u3.y * C1), fmaf(u3.y, S1, u3.x * C1)};
    y15 = cadd(cadd(u0, t2), cadd(t1, t3));
}

__device__ __forceinline__ void radix16_inv_0_15(float2 x0, float2 x15, float2 y[16]) {
    constexpr float C1 = 0.92387953251128675613f, S1 = 0.38268343236508977173f, SQRT1_2 = 0.70710678118654752440f;
    float2 t0 = x15;
    float2 t1 = {fmaf(x15.x, C1,  x15.y * S1), fmaf(x15.y, C1, -x15.x * S1)};
    float2 t2 = {SQRT1_2 * (x15.x + x15.y), SQRT1_2 * (x15.y - x15.x)};
    float2 t3 = {fmaf(x15.x, S1,  x15.y * C1), fmaf(x15.y, S1, -x15.x * C1)};
    float2 t4 = {x15.y, -x15.x};
    float2 t5 = {fmaf(-x15.x, S1,  x15.y * C1), fmaf(-x15.y, S1, -x15.x * C1)};
    float2 t6 = {SQRT1_2 * (x15.y - x15.x), -SQRT1_2 * (x15.x + x15.y)};
    float2 t7 = {fmaf(-x15.x, C1,  x15.y * S1), fmaf(-x15.y, C1, -x15.x * S1)};
    y[0] = cadd(x0, t0); y[8]  = csub(x0, t0);
    y[1] = cadd(x0, t1); y[9]  = csub(x0, t1);
    y[2] = cadd(x0, t2); y[10] = csub(x0, t2);
    y[3] = cadd(x0, t3); y[11] = csub(x0, t3);
    y[4] = cadd(x0, t4); y[12] = csub(x0, t4);
    y[5] = cadd(x0, t5); y[13] = csub(x0, t5);
    y[6] = cadd(x0, t6); y[14] = csub(x0, t6);
    y[7] = cadd(x0, t7); y[15] = csub(x0, t7);
}

__device__ __forceinline__ int swiz2(int i) { return i ^ ((i >> 4) & 15); }

template <int L>
__device__ __forceinline__ int drev_L(int k) {
    if constexpr (L == 1024)  return ((k & 15) << 5)  | (((k >> 4) & 15) << 1) | (k >> 8);
    if constexpr (L == 2048)  return ((k & 15) << 6)  | (((k >> 4) & 15) << 2) | (k >> 8);
    if constexpr (L == 4096)  return ((k & 15) << 7)  | (((k >> 4) & 15) << 3) | (k >> 8);
    if constexpr (L == 8192)  return ((k & 15) << 8)  | (((k >> 4) & 15) << 4) | (k >> 8);
    if constexpr (L == 16384) return ((k & 15) << 9)  | (((k >> 4) & 15) << 5) | (((k >> 8) & 1) << 4) | (k >> 9);
    if constexpr (L == 32768) return ((k & 15) << 10) | (((k >> 4) & 15) << 6) | (((k >> 8) & 3) << 4) | (k >> 10);
}

__device__ __noinline__ float odd_ext_load(const float * __restrict__ buf0, int64_t M, int64_t u) {
    if (u >= 1 && u <= M) return __ldg(&buf0[u]);
    if (u == 0 || u == M + 1) return 0.0f;
    if (u >= -M && u <= -1) return -__ldg(&buf0[-u]);
    int64_t P = 2LL * (M + 1LL);
    if (u >= M + 2 && u < P) return -__ldg(&buf0[P - u]);
    int64_t r;
    if (P <= 2000000000LL && u >= -2000000000LL && u <= 2000000000LL) {
        int ri = (int)u % (int)P;
        if (ri < 0) ri += (int)P;
        r = (int64_t)ri;
    } else {
        r = u % P;
        if (r < 0) r += P;
    }
    if (r == 0 || r == M + 1) return 0.0f;
    if (r <= M) return __ldg(&buf0[r]);
    return -__ldg(&buf0[P - r]);
}

__device__ __forceinline__ float2 eval_filter0_interior(const float2 * __restrict__ src2, int n) {
    float2 vm2 = __ldg(&src2[n - 2]);
    float2 vm1 = __ldg(&src2[n - 1]);
    float2 v0  = __ldg(&src2[n]);
    float2 vp1 = __ldg(&src2[n + 1]);
    float2 vp2 = __ldg(&src2[n + 2]);
    float y0 = fmaf(16.0f, v0.x,
               fmaf(14.0f, vm1.y + v0.y,
               fmaf( 9.0f, vm1.x + vp1.x,
               fmaf( 4.0f, vm2.y + vp1.y, vm2.x + vp2.x)))) * INV_72;
    float y1 = fmaf(16.0f, v0.y,
               fmaf(14.0f, v0.x  + vp1.x,
               fmaf( 9.0f, vm1.y + vp1.y,
               fmaf( 4.0f, vm1.x + vp2.x, vm2.y + vp2.y)))) * INV_72;
    return {y0, y1};
}

__device__ __noinline__ float2 eval_filter0_boundary(const float * __restrict__ src_in, int64_t M, int64_t u0) {
    float x0 = odd_ext_load(src_in, M, u0 - 4LL);
    float x1 = odd_ext_load(src_in, M, u0 - 3LL);
    float x2 = odd_ext_load(src_in, M, u0 - 2LL);
    float x3 = odd_ext_load(src_in, M, u0 - 1LL);
    float x4 = odd_ext_load(src_in, M, u0);
    float x5 = odd_ext_load(src_in, M, u0 + 1LL);
    float x6 = odd_ext_load(src_in, M, u0 + 2LL);
    float x7 = odd_ext_load(src_in, M, u0 + 3LL);
    float x8 = odd_ext_load(src_in, M, u0 + 4LL);
    float x9 = odd_ext_load(src_in, M, u0 + 5LL);
    float y0 = fmaf(16.0f, x4, fmaf(14.0f, x3 + x5, fmaf(9.0f, x2 + x6, fmaf(4.0f, x1 + x7, x0 + x8)))) * INV_72;
    float y1 = fmaf(16.0f, x5, fmaf(14.0f, x4 + x6, fmaf(9.0f, x3 + x7, fmaf(4.0f, x2 + x8, x1 + x9)))) * INV_72;
    return {y0, y1};
}

__device__ __forceinline__ float jstep(float a, float b, float d) {
    return __fmul_rn(C_F32, __fadd_rn(__fadd_rn(a, b), d));
}

// ============================================================================
// Mode 1A: Ultra-Fast Zero-SMEM Vectorized Kernel for n0 == 1 (bit-for-bit FP32)
// ============================================================================

__global__ void direct_n0_1_kernel(
    const float * __restrict__ buf0,
    float * __restrict__ buf1,
    float * __restrict__ out0,
    int n1)
{
    int64_t M = (int64_t)n1 - 2LL;
    int64_t g4 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t g = g4 * 4LL;
    if (g >= (int64_t)n1) return;

    if (g >= 4LL && g + 7LL < (int64_t)n1) {
        const float4 *b0_4 = (const float4 *)buf0;
        float4 vl = __ldg(&b0_4[g4 - 1]);
        float4 vc = __ldg(&b0_4[g4]);
        float4 vr = __ldg(&b0_4[g4 + 1]);

        float t_m1 = jstep(vl.z, vl.w, vc.x);
        float t_0  = jstep(vl.w, vc.x, vc.y);
        float t_1  = jstep(vc.x, vc.y, vc.z);
        float t_2  = jstep(vc.y, vc.z, vc.w);
        float t_3  = jstep(vc.z, vc.w, vr.x);
        float t_p1 = jstep(vc.w, vr.x, vr.y);

        ((float4 *)buf1)[g4] = make_float4(t_0, t_1, t_2, t_3);
        ((float4 *)out0)[g4] = make_float4(
            jstep(t_m1, t_0, t_1),
            jstep(t_0,  t_1, t_2),
            jstep(t_1,  t_2, t_3),
            jstep(t_2,  t_3, t_p1));
    } else {
        float b1_L = buf1[0], b1_R = buf1[n1 - 1];
        auto get_b1 = [&](int64_t idx) -> float {
            if (idx == 0LL) return b1_L;
            if (idx == (int64_t)n1 - 1LL) return b1_R;
            return jstep(buf0[idx - 1], buf0[idx], buf0[idx + 1]);
        };
        for (int r = 0; r < 4; r++) {
            int64_t idx = g + r;
            if (idx == 0LL) {
                out0[0] = buf0[0];
            } else if (idx == (int64_t)n1 - 1LL) {
                out0[n1 - 1] = buf0[n1 - 1];
            } else if (idx >= 1LL && idx <= M) {
                float vm = get_b1(idx - 1LL);
                float vc = get_b1(idx);
                float vp = get_b1(idx + 1LL);
                buf1[idx] = vc;
                out0[idx] = jstep(vm, vc, vp);
            }
        }
    }
}

// ============================================================================
// Mode 1B: 768-elem/warp (KPT=24) Direct Warp-Register Stencil (2 <= n0 <= 14)
//          Split PURE_INTERIOR=true (0 spills, 64 regs) + boundary kernel
// ============================================================================

template <bool PURE_INTERIOR>
__global__ void __launch_bounds__(256, 4)
direct_warp_stencil_kernel(
    const float * __restrict__ buf0,
    float * __restrict__ buf1,
    float * __restrict__ out0,
    int n0,
    int n1,
    int B,
    int H,
    int64_t w_int_lo,
    int64_t w_int_hi,
    int64_t num_warps)
{
    constexpr int KPT = DIRECT_KPT; // 24
    constexpr int WSIZE = DIRECT_WARP_ELEMS; // 768
    constexpr int WSTRIDE = WSIZE + (WSIZE >> 5); // 792
    constexpr int WARPS_PER_BLK = 8;
    __shared__ float smem[WARPS_PER_BLK * WSTRIDE];

    int warp_in_blk = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int64_t wid_local = (int64_t)blockIdx.x * WARPS_PER_BLK + warp_in_blk;

    int64_t warp_id;
    if constexpr (PURE_INTERIOR) {
        warp_id = w_int_lo + wid_local;
        if (warp_id >= w_int_hi) return;
    } else {
        int64_t n_left = w_int_lo;
        int64_t n_bnd = n_left + (num_warps - w_int_hi);
        if (wid_local >= n_bnd) return;
        warp_id = (wid_local < n_left) ? wid_local : (w_int_hi + (wid_local - n_left));
    }

    int64_t tile_out_start = warp_id * (int64_t)B;
    if (tile_out_start > (int64_t)n1 - 2LL) return;

    float *ws = smem + warp_in_blk * WSTRIDE;
    int64_t tile_load_start = tile_out_start - (int64_t)H;

    if constexpr (PURE_INTERIOR) {
        const float4 *src4 = (const float4 *)(buf0 + tile_load_start);
        #pragma unroll
        for (int j = 0; j < KPT / 4; j++) {
            float4 v = __ldg(&src4[j * 32 + lane]);
            int idx = (j * 32 + lane) * 4;
            ws[idx + 0 + ((idx + 0) >> 5)] = v.x;
            ws[idx + 1 + ((idx + 1) >> 5)] = v.y;
            ws[idx + 2 + ((idx + 2) >> 5)] = v.z;
            ws[idx + 3 + ((idx + 3) >> 5)] = v.w;
        }
        __syncwarp();

        float r[KPT];
        int base = lane * KPT;
        #pragma unroll
        for (int k = 0; k < KPT; k++) {
            int idx = base + k;
            r[k] = ws[idx + (idx >> 5)];
        }

        int total_half = 2 * n0;
        for (int s = 0; s < total_half - 1; s++) {
            float left_nb  = __shfl_up_sync(0xffffffffu, r[KPT - 1], 1);
            float right_nb = __shfl_down_sync(0xffffffffu, r[0], 1);
            float prev = left_nb;
            #pragma unroll
            for (int k = 0; k < KPT - 1; k++) {
                float cur = r[k];
                r[k] = jstep(prev, cur, r[k + 1]);
                prev = cur;
            }
            r[KPT - 1] = jstep(prev, r[KPT - 1], right_nb);
        }

        #pragma unroll
        for (int k = 0; k < KPT; k++) {
            int idx = base + k;
            ws[idx + (idx >> 5)] = r[k];
        }
        __syncwarp();
        float4 *dst1_4 = (float4 *)(buf1 + tile_out_start);
        int B4 = B >> 2;
        for (int j4 = lane; j4 < B4; j4 += 32) {
            int idx = H + j4 * 4;
            dst1_4[j4] = make_float4(
                ws[idx + 0 + ((idx + 0) >> 5)],
                ws[idx + 1 + ((idx + 1) >> 5)],
                ws[idx + 2 + ((idx + 2) >> 5)],
                ws[idx + 3 + ((idx + 3) >> 5)]);
        }

        float left_nb  = __shfl_up_sync(0xffffffffu, r[KPT - 1], 1);
        float right_nb = __shfl_down_sync(0xffffffffu, r[0], 1);
        float prev = left_nb;
        #pragma unroll
        for (int k = 0; k < KPT - 1; k++) {
            float cur = r[k];
            r[k] = jstep(prev, cur, r[k + 1]);
            prev = cur;
        }
        r[KPT - 1] = jstep(prev, r[KPT - 1], right_nb);

        __syncwarp();
        #pragma unroll
        for (int k = 0; k < KPT; k++) {
            int idx = base + k;
            ws[idx + (idx >> 5)] = r[k];
        }
        __syncwarp();
        float4 *dst0_4 = (float4 *)(out0 + tile_out_start);
        for (int j4 = lane; j4 < B4; j4 += 32) {
            int idx = H + j4 * 4;
            dst0_4[j4] = make_float4(
                ws[idx + 0 + ((idx + 0) >> 5)],
                ws[idx + 1 + ((idx + 1) >> 5)],
                ws[idx + 2 + ((idx + 2) >> 5)],
                ws[idx + 3 + ((idx + 3) >> 5)]);
        }
    } else {
        #pragma unroll
        for (int j = 0; j < KPT; j++) {
            int idx = j * 32 + lane;
            int64_t g = tile_load_start + idx;
            ws[idx + (idx >> 5)] = (g >= 0 && g < (int64_t)n1) ? buf0[g] : 0.0f;
        }
        __syncwarp();

        float r[KPT];
        int base = lane * KPT;
        #pragma unroll
        for (int k = 0; k < KPT; k++) {
            int idx = base + k;
            r[k] = ws[idx + (idx >> 5)];
        }

        float b0_L = buf0[0], b0_R = buf0[n1 - 1];
        float b1_L = buf1[0], b1_R = buf1[n1 - 1];
        int64_t g_base = tile_load_start + base;
        int64_t k_left  = 0LL - g_base;
        int64_t k_right = (int64_t)(n1 - 1) - g_base;

        int total_half = 2 * n0;
        for (int s = 0; s < total_half; s++) {
            if (s > 0) {
                float bl = (s & 1) ? b1_L : b0_L;
                float br = (s & 1) ? b1_R : b0_R;
                #pragma unroll
                for (int k = 0; k < KPT; k++) {
                    if (k == k_left)  r[k] = bl;
                    if (k == k_right) r[k] = br;
                }
            }
            float left_nb  = __shfl_up_sync(0xffffffffu, r[KPT - 1], 1);
            float right_nb = __shfl_down_sync(0xffffffffu, r[0], 1);
            float prev = left_nb;
            #pragma unroll
            for (int k = 0; k < KPT - 1; k++) {
                float cur = r[k];
                r[k] = jstep(prev, cur, r[k + 1]);
                prev = cur;
            }
            r[KPT - 1] = jstep(prev, r[KPT - 1], right_nb);

            if (s == total_half - 2) {
                #pragma unroll
                for (int k = 0; k < KPT; k++) {
                    int idx = base + k;
                    ws[idx + (idx >> 5)] = r[k];
                }
                __syncwarp();
                for (int idx = H + lane; idx < H + B; idx += 32) {
                    int64_t g = tile_load_start + idx;
                    if (g >= 1 && g <= (int64_t)n1 - 2LL) {
                        buf1[g] = ws[idx + (idx >> 5)];
                    }
                }
                __syncwarp();
            }
        }
        #pragma unroll
        for (int k = 0; k < KPT; k++) {
            int idx = base + k;
            ws[idx + (idx >> 5)] = r[k];
        }
        __syncwarp();
        for (int idx = H + lane; idx < H + B; idx += 32) {
            int64_t g = tile_load_start + idx;
            if (g >= 1 && g <= (int64_t)n1 - 2LL) {
                out0[g] = ws[idx + (idx >> 5)];
            }
        }
        if (warp_id == 0 && lane == 0) {
            out0[0] = b0_L;
            out0[n1 - 1] = b0_R;
        }
    }
}

// ============================================================================
// Mode 2 & 3: Fused-Stage-0 float2 (LDS.64/STS.64) Pruned Overlap-Save FFT
//   PURE_INTERIOR == true:  Static SMEM (L<=8192), 0-stack-frame, 0-spill fast path
//   PURE_INTERIOR == false: Boundary / small-grid path with exact LUT twiddles
// ============================================================================

template <int L, bool PRE_FILTER, bool PRUNE, bool WRITE_FINAL, bool PURE_INTERIOR>
__global__ void __launch_bounds__(L / 32, 1024 / (L / 32))
fft_overlap_save_kernel(
    const float * __restrict__ src_in,
    const float * orig_buf0,
    float * __restrict__ dst_buf1,
    float * dst_out0,
    const float2 * __restrict__ tw,
    const float4 * __restrict__ H_abc,
    const float2 * __restrict__ d_pq,
    int n1,
    int R,
    int B,
    int K_bnd,
    int m_cut,
    int64_t blk_int_lo,
    int64_t blk_int_hi)
{
    constexpr int N = L / 2;
    constexpr int T = N / 16; // threads per block = L / 32
    extern __shared__ float2 s_dyn[];
    __shared__ float2 s_stat[(L <= 8192) ? N : 1];
    float2 *s_z = (L <= 8192) ? s_stat : s_dyn;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int64_t M = (int64_t)n1 - 2LL;

    int64_t bid;
    if constexpr (PURE_INTERIOR) {
        bid = blk_int_lo + (int64_t)blockIdx.x;
    } else {
        int64_t b = (int64_t)blockIdx.x;
        bid = (b < blk_int_lo) ? b : (blk_int_hi + (b - blk_int_lo));
    }

    int64_t tile_out_start = bid * (int64_t)B;
    if constexpr (!PURE_INTERIOR) {
        if (tile_out_start > M) return;
    }

    int64_t tile_fft_start = WRITE_FINAL ? (tile_out_start - (int64_t)(R + 1)) : (tile_out_start - (int64_t)R);

    // ------------------------------------------------------------------------
    // Step 1: FUSED Global Load (+ optional 9-tap Filter0) + Forward Stage 0!
    // ------------------------------------------------------------------------
    {
        float2 x[16];
        if constexpr (PURE_INTERIOR) {
            const float2 *src2 = (const float2 *)(src_in + tile_fft_start);
            #pragma unroll
            for (int q = 0; q < 16; q++) {
                int n = tid + q * T;
                if constexpr (PRE_FILTER) x[q] = eval_filter0_interior(src2, n);
                else                      x[q] = __ldg(&src2[n]);
            }
        } else {
            constexpr int PAD_F = PRE_FILTER ? 4 : 0;
            bool interior_load = (tile_fft_start - PAD_F >= 1LL) && (tile_fft_start + (int64_t)L - 1LL + PAD_F <= M);
            if (interior_load) {
                const float2 *src2 = (const float2 *)(src_in + tile_fft_start);
                #pragma unroll
                for (int q = 0; q < 16; q++) {
                    int n = tid + q * T;
                    if constexpr (PRE_FILTER) x[q] = eval_filter0_interior(src2, n);
                    else                      x[q] = __ldg(&src2[n]);
                }
            } else {
                for (int q = 0; q < 16; q++) {
                    int64_t u0 = tile_fft_start + 2LL * (int64_t)(tid + q * T);
                    if (u0 - (int64_t)PAD_F >= 1LL && u0 + 1LL + (int64_t)PAD_F <= M) {
                        const float2 *p2 = (const float2 *)(src_in + u0);
                        if constexpr (PRE_FILTER) x[q] = eval_filter0_interior(p2, 0);
                        else                      x[q] = __ldg(p2);
                    } else {
                        if constexpr (PRE_FILTER) x[q] = eval_filter0_boundary(src_in, M, u0);
                        else                      x[q] = {odd_ext_load(src_in, M, u0), odd_ext_load(src_in, M, u0 + 1LL)};
                    }
                }
            }
        }
        radix16<-1>(x);
        if constexpr (PURE_INTERIOR) apply_tw16<-1>(x, __ldg(&tw[tid]));
        else                         apply_tw16_lut<-1, N>(x, tw, tid);
        #pragma unroll
        for (int q = 0; q < 16; q++) s_z[swiz2(tid + q * T)] = x[q];
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Step 2: Forward Stage 1 (m = N/16, radix 16, stride = N/256)
    // ------------------------------------------------------------------------
    {
        constexpr int M_CUR = N / 16;
        constexpr int STRIDE = M_CUR / 16;
        int grp = tid / STRIDE;
        int j   = tid - grp * STRIDE;
        int base = grp * M_CUR + j;
        float2 x[16];
        #pragma unroll
        for (int q = 0; q < 16; q++) x[q] = s_z[swiz2(base + q * STRIDE)];
        radix16<-1>(x);
        if constexpr (PURE_INTERIOR) apply_tw16<-1>(x, __ldg(&tw[j * 16]));
        else                         apply_tw16_lut<-1, N>(x, tw, j * 16);
        #pragma unroll
        for (int q = 0; q < 16; q++) s_z[swiz2(base + q * STRIDE)] = x[q];
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Step 2b: Optional Forward Stage 2 for L=16384 (r=2) & L=32768 (r=4)
    // ------------------------------------------------------------------------
    if constexpr (L == 16384) {
        int grp_base = (tid >> 4) * 8;
        int j = tid & 15;
        float2 w1 = __ldg(&tw[j * 256]);
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            int base = (grp_base + g) * 32 + j;
            int sw0 = swiz2(base), sw1 = swiz2(base + 16);
            float2 u = s_z[sw0], v = s_z[sw1];
            s_z[sw0] = cadd(u, v);
            s_z[sw1] = cmul(csub(u, v), w1);
        }
        __syncthreads();
    } else if constexpr (L == 32768) {
        int grp_base = (tid >> 4) * 4;
        int j = tid & 15;
        float2 w1 = __ldg(&tw[j * 256]);
        float2 w2 = cmul(w1, w1);
        float2 w3 = cmul(w1, w2);
        #pragma unroll 1
        for (int g = 0; g < 4; g++) {
            int base = (grp_base + g) * 64 + j;
            int sw0 = swiz2(base), sw1 = swiz2(base + 16), sw2 = swiz2(base + 32), sw3 = swiz2(base + 48);
            float2 x0 = s_z[sw0], x1 = s_z[sw1], x2 = s_z[sw2], x3 = s_z[sw3];
            radix4<-1>(x0, x1, x2, x3);
            s_z[sw0] = x0;
            s_z[sw1] = cmul(x1, w1);
            s_z[sw2] = cmul(x2, w2);
            s_z[sw3] = cmul(x3, w3);
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------------
    // Step 2c: Final Forward Stage (stride = 1, pruned when PRUNE == true!)
    // ------------------------------------------------------------------------
    {
        int base = tid * 16;
        if constexpr (L == 1024) {
            float2 x[16];
            #pragma unroll
            for (int q = 0; q < 16; q++) x[q] = s_z[swiz2(base + q)];
            #pragma unroll
            for (int k = 0; k < 16; k += 2) {
                float2 u = x[k], v = x[k + 1];
                x[k] = cadd(u, v); x[k + 1] = csub(u, v);
            }
            #pragma unroll
            for (int q = 0; q < 16; q++) s_z[swiz2(base + q)] = x[q];
        } else if constexpr (L == 2048) {
            if constexpr (PRUNE) {
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    int sub = base + 4 * b;
                    float2 x[4], y0, y3;
                    #pragma unroll
                    for (int q = 0; q < 4; q++) x[q] = s_z[swiz2(sub + q)];
                    radix4_fwd_03(x, y0, y3);
                    s_z[swiz2(sub + 0)] = y0; s_z[swiz2(sub + 3)] = y3;
                }
            } else {
                float2 x[16];
                #pragma unroll
                for (int q = 0; q < 16; q++) x[q] = s_z[swiz2(base + q)];
                radix4<-1>(x[0],  x[1],  x[2],  x[3]);
                radix4<-1>(x[4],  x[5],  x[6],  x[7]);
                radix4<-1>(x[8],  x[9],  x[10], x[11]);
                radix4<-1>(x[12], x[13], x[14], x[15]);
                #pragma unroll
                for (int q = 0; q < 16; q++) s_z[swiz2(base + q)] = x[q];
            }
        } else if constexpr (L == 4096) {
            // Always pruned for L == 4096!
            float2 x[8], y0, y7;
            #pragma unroll
            for (int q = 0; q < 8; q++) x[q] = s_z[swiz2(base + q)];
            radix8_fwd_07(x, y0, y7);
            s_z[swiz2(base + 0)] = y0; s_z[swiz2(base + 7)] = y7;
            #pragma unroll
            for (int q = 0; q < 8; q++) x[q] = s_z[swiz2(base + 8 + q)];
            radix8_fwd_07(x, y0, y7);
            s_z[swiz2(base + 8)] = y0; s_z[swiz2(base + 15)] = y7;
        } else {
            // Always pruned for L >= 8192 (R_LAST == 16)!
            float2 x[16], y0, y15;
            #pragma unroll
            for (int q = 0; q < 16; q++) x[q] = s_z[swiz2(base + q)];
            radix16_fwd_0_15(x, y0, y15);
            s_z[swiz2(base + 0)] = y0; s_z[swiz2(base + 15)] = y15;
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Step 3: Collapsed 4-FMA R2C + Real Filter H + C2R (in-place)
    // ------------------------------------------------------------------------
    if constexpr (PRUNE) {
        for (int k = tid; k <= m_cut; k += T) {
            if (k == 0) {
                float2 z0 = s_z[0]; float4 h0 = __ldg(&H_abc[0]);
                s_z[0] = {fmaf(h0.x, z0.x, h0.y * z0.y), fmaf(h0.x, z0.y, h0.y * z0.x)};
            } else {
                int swa = swiz2(drev_L<L>(k)), swb = swiz2(drev_L<L>(N - k));
                float2 za = s_z[swa], zb = s_z[swb];
                float4 hk = __ldg(&H_abc[k]);
                s_z[swa] = {fmaf(hk.x, za.x, hk.y * zb.y), fmaf(hk.x, za.y, hk.y * zb.x)};
                s_z[swb] = {fmaf(hk.z, zb.x, hk.y * za.y), fmaf(hk.z, zb.y, hk.y * za.x)};
            }
        }
    } else {
        for (int k = tid; k <= N / 2; k += T) {
            if (k == 0) {
                float2 z0 = s_z[0]; float4 h0 = __ldg(&H_abc[0]);
                s_z[0] = {fmaf(h0.x, z0.x, h0.y * z0.y), fmaf(h0.x, z0.y, h0.y * z0.x)};
            } else if (k == N / 2) {
                int swh = swiz2(drev_L<L>(N / 2));
                float hm = __ldg(&H_abc[N / 2]).x;
                float2 zh = s_z[swh];
                s_z[swh] = {zh.x * hm, zh.y * hm};
            } else if (k > m_cut) {
                s_z[swiz2(drev_L<L>(k))]     = {0.0f, 0.0f};
                s_z[swiz2(drev_L<L>(N - k))] = {0.0f, 0.0f};
            } else {
                int swa = swiz2(drev_L<L>(k)), swb = swiz2(drev_L<L>(N - k));
                float2 za = s_z[swa], zb = s_z[swb];
                float4 hk = __ldg(&H_abc[k]);
                s_z[swa] = {fmaf(hk.x, za.x, hk.y * zb.y), fmaf(hk.x, za.y, hk.y * zb.x)};
                s_z[swb] = {fmaf(hk.z, zb.x, hk.y * za.y), fmaf(hk.z, zb.y, hk.y * za.x)};
            }
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Step 4: First Inverse DIT Stage (stride = 1, pruned when PRUNE == true!)
    // ------------------------------------------------------------------------
    {
        int base = tid * 16;
        if constexpr (L == 1024) {
            float2 x[16];
            #pragma unroll
            for (int q = 0; q < 16; q++) x[q] = s_z[swiz2(base + q)];
            #pragma unroll
            for (int k = 0; k < 16; k += 2) {
                float2 u = x[k], v = x[k + 1];
                x[k] = cadd(u, v); x[k + 1] = csub(u, v);
            }
            #pragma unroll
            for (int q = 0; q < 16; q++) s_z[swiz2(base + q)] = x[q];
        } else if constexpr (L == 2048) {
            if constexpr (PRUNE) {
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    int sub = base + 4 * b;
                    int k_lo = ((sub >> 6) & 15) + 16 * ((sub >> 2) & 15);
                    int k_hi_neg = 256 - k_lo;
                    float2 x0 = (k_lo <= m_cut)     ? s_z[swiz2(sub + 0)] : float2{0.0f, 0.0f};
                    float2 x3 = (k_hi_neg <= m_cut) ? s_z[swiz2(sub + 3)] : float2{0.0f, 0.0f};
                    float2 y[4];
                    radix4_inv_03(x0, x3, y);
                    #pragma unroll
                    for (int q = 0; q < 4; q++) s_z[swiz2(sub + q)] = y[q];
                }
            } else {
                float2 x[16];
                #pragma unroll
                for (int q = 0; q < 16; q++) x[q] = s_z[swiz2(base + q)];
                radix4<+1>(x[0],  x[1],  x[2],  x[3]);
                radix4<+1>(x[4],  x[5],  x[6],  x[7]);
                radix4<+1>(x[8],  x[9],  x[10], x[11]);
                radix4<+1>(x[12], x[13], x[14], x[15]);
                #pragma unroll
                for (int q = 0; q < 16; q++) s_z[swiz2(base + q)] = x[q];
            }
        } else if constexpr (L == 4096) {
            #pragma unroll
            for (int b = 0; b < 2; b++) {
                int sub = base + 8 * b;
                int k_lo = ((sub >> 7) & 15) + 16 * ((sub >> 3) & 15);
                int k_hi_neg = 256 - k_lo;
                float2 x0 = (k_lo <= m_cut)     ? s_z[swiz2(sub + 0)] : float2{0.0f, 0.0f};
                float2 x7 = (k_hi_neg <= m_cut) ? s_z[swiz2(sub + 7)] : float2{0.0f, 0.0f};
                float2 y[8];
                radix8_inv_07(x0, x7, y);
                #pragma unroll
                for (int q = 0; q < 8; q++) s_z[swiz2(sub + q)] = y[q];
            }
        } else {
            int k_lo, band;
            if constexpr (L == 8192) {
                k_lo = ((base >> 8) & 15) + 16 * ((base >> 4) & 15);
                band = 256;
            } else if constexpr (L == 16384) {
                k_lo = ((base >> 9) & 15) + 16 * ((base >> 5) & 15) + 256 * ((base >> 4) & 1);
                band = 512;
            } else {
                k_lo = ((base >> 10) & 15) + 16 * ((base >> 6) & 15) + 256 * ((base >> 4) & 3);
                band = 1024;
            }
            int k_hi_neg = band - k_lo;
            float2 x0  = (k_lo <= m_cut)     ? s_z[swiz2(base + 0)]  : float2{0.0f, 0.0f};
            float2 x15 = (k_hi_neg <= m_cut) ? s_z[swiz2(base + 15)] : float2{0.0f, 0.0f};
            float2 y[16];
            radix16_inv_0_15(x0, x15, y);
            #pragma unroll
            for (int q = 0; q < 16; q++) s_z[swiz2(base + q)] = y[q];
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Step 4b: Optional Inverse DIT Stage for L=16384 (r=2) & L=32768 (r=4)
    // ------------------------------------------------------------------------
    if constexpr (L == 16384) {
        int grp_base = (tid >> 4) * 8;
        int j = tid & 15;
        float2 w1 = __ldg(&tw[j * 256]);
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            int base = (grp_base + g) * 32 + j;
            int sw0 = swiz2(base), sw1 = swiz2(base + 16);
            float2 u = s_z[sw0];
            float2 v = cmul_conj(s_z[sw1], w1);
            s_z[sw0] = cadd(u, v);
            s_z[sw1] = csub(u, v);
        }
        __syncthreads();
    } else if constexpr (L == 32768) {
        int grp_base = (tid >> 4) * 4;
        int j = tid & 15;
        float2 w1 = __ldg(&tw[j * 256]);
        w1.y = -w1.y;
        float2 w2 = cmul(w1, w1);
        float2 w3 = cmul(w1, w2);
        #pragma unroll 1
        for (int g = 0; g < 4; g++) {
            int base = (grp_base + g) * 64 + j;
            int sw0 = swiz2(base), sw1 = swiz2(base + 16), sw2 = swiz2(base + 32), sw3 = swiz2(base + 48);
            float2 x0 = s_z[sw0];
            float2 x1 = cmul(s_z[sw1], w1);
            float2 x2 = cmul(s_z[sw2], w2);
            float2 x3 = cmul(s_z[sw3], w3);
            radix4<+1>(x0, x1, x2, x3);
            s_z[sw0] = x0; s_z[sw1] = x1; s_z[sw2] = x2; s_z[sw3] = x3;
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------------
    // Step 4c: Inverse DIT Stage (m = N/16, radix 16, stride = N/256)
    // ------------------------------------------------------------------------
    {
        constexpr int M_CUR = N / 16;
        constexpr int STRIDE = M_CUR / 16;
        int grp = tid / STRIDE;
        int j   = tid - grp * STRIDE;
        int base = grp * M_CUR + j;
        float2 x[16];
        #pragma unroll
        for (int q = 0; q < 16; q++) x[q] = s_z[swiz2(base + q * STRIDE)];
        if constexpr (PURE_INTERIOR) apply_tw16<+1>(x, __ldg(&tw[j * 16]));
        else                         apply_tw16_lut<+1, N>(x, tw, j * 16);
        radix16<+1>(x);
        #pragma unroll
        for (int q = 0; q < 16; q++) s_z[swiz2(base + q * STRIDE)] = x[q];
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Step 4d: Final Inverse DIT Stage (m = N, stride = T) + Output Write!
    // ------------------------------------------------------------------------
    {
        float2 x[16];
        #pragma unroll
        for (int q = 0; q < 16; q++) x[q] = s_z[swiz2(tid + q * T)];
        if constexpr (PURE_INTERIOR) apply_tw16<+1>(x, __ldg(&tw[tid]));
        else                         apply_tw16_lut<+1, N>(x, tw, tid);
        radix16<+1>(x);
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < 16; q++) s_z[tid + q * T] = x[q];
        __syncthreads();

        if constexpr (!WRITE_FINAL) {
            if constexpr (PURE_INTERIOR) {
                const float4 *s_out4 = (const float4 *)s_z;
                float4 *dst4 = (float4 *)(dst_buf1 + tile_out_start);
                int k4_base = R >> 2;
                int B4 = B >> 2;
                for (int j4 = tid; j4 < B4; j4 += T) dst4[j4] = s_out4[k4_base + j4];
            } else {
                const float *s_out = (const float *)s_z;
                for (int j = tid; j < B; j += T) {
                    int64_t g = tile_out_start + j;
                    if (g >= 1LL && g <= M) dst_buf1[g] = s_out[R + j];
                }
            }
        } else {
            if constexpr (PURE_INTERIOR) {
                const float *s_out = (const float *)s_z;
                const float4 *s_out4 = (const float4 *)s_z;
                float4 *dst1_4 = (float4 *)(dst_buf1 + tile_out_start);
                float4 *dst0_4 = (float4 *)(dst_out0 + tile_out_start);
                int k4_base = (R + 1) >> 2;
                int B4 = B >> 2;
                for (int j4_base = (tid >> 5) * 32; j4_base < B4; j4_base += T) {
                    int j4 = j4_base + lane;
                    float4 v = (j4 < B4) ? s_out4[k4_base + j4] : make_float4(0, 0, 0, 0);
                    float vm = __shfl_up_sync(0xffffffffu, v.w, 1);
                    float vp = __shfl_down_sync(0xffffffffu, v.x, 1);
                    if (lane == 0 && j4 < B4) vm = s_out[(k4_base + j4) * 4 - 1];
                    if ((lane == 31 || j4 == B4 - 1) && j4 < B4) vp = s_out[(k4_base + j4) * 4 + 4];
                    if (j4 < B4) {
                        dst1_4[j4] = v;
                        dst0_4[j4] = make_float4(
                            jstep(vm,  v.x, v.y),
                            jstep(v.x, v.y, v.z),
                            jstep(v.y, v.z, v.w),
                            jstep(v.z, v.w, vp));
                    }
                }
            } else {
                float *s_out = (float *)s_z;
                float b0_L = orig_buf0[0], b0_R = orig_buf0[n1 - 1];
                float b1_L = dst_buf1[0],  b1_R = dst_buf1[n1 - 1];
                for (int k = R + tid; k <= R + B + 1; k += T) {
                    int64_t g = tile_fft_start + (int64_t)k;
                    if (g == 0LL) {
                        s_out[k] = b1_L;
                    } else if (g == M + 1LL) {
                        s_out[k] = b1_R;
                    } else if (g >= 1LL && g <= M) {
                        float v = s_out[k];
                        if (g <= (int64_t)K_bnd) {
                            float2 pq = __ldg(&d_pq[g]);
                            v = fmaf(b0_L, pq.x, fmaf(b1_L, pq.y, v));
                        }
                        int64_t gr = M + 1LL - g;
                        if (gr <= (int64_t)K_bnd) {
                            float2 pq = __ldg(&d_pq[gr]);
                            v = fmaf(b0_R, pq.x, fmaf(b1_R, pq.y, v));
                        }
                        s_out[k] = v;
                    }
                }
                __syncthreads();

                for (int j = tid; j < B; j += T) {
                    int64_t g = tile_out_start + (int64_t)j;
                    if (g >= 1LL && g <= M) {
                        int k = R + 1 + j;
                        float vm = s_out[k - 1], vc = s_out[k], vp = s_out[k + 1];
                        dst_buf1[g] = vc;
                        dst_out0[g] = jstep(vm, vc, vp);
                    }
                }
                if (bid == 0 && tid == 0 && dst_out0 != orig_buf0) {
                    dst_out0[0] = b0_L;
                    dst_out0[n1 - 1] = b0_R;
                }
            }
        }
    }
}

// ============================================================================
// Mode 4: Steady-State Boundary-Only Kernel (S >= 11,500,000)
// ============================================================================

__global__ void steady_boundary_kernel(
    const float * __restrict__ buf0,
    float * __restrict__ buf1,
    float * __restrict__ out0,
    const float2 * __restrict__ d_pq,
    int n1,
    int K_bnd)
{
    int64_t M = (int64_t)n1 - 2LL;
    int64_t g4 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride4 = (int64_t)gridDim.x * blockDim.x;
    int64_t n4 = ((int64_t)n1 + 3LL) >> 2;

    float b0_L = buf0[0], b0_R = buf0[n1 - 1];
    float b1_L = buf1[0], b1_R = buf1[n1 - 1];

    auto eval_b1 = [&](int64_t g) -> float {
        if (g == 0LL) return b1_L;
        if (g == M + 1LL) return b1_R;
        float v = 0.0f;
        if (g <= (int64_t)K_bnd) {
            float2 pq = __ldg(&d_pq[g]);
            v = fmaf(b0_L, pq.x, fmaf(b1_L, pq.y, v));
        }
        int64_t gr = M + 1LL - g;
        if (gr <= (int64_t)K_bnd) {
            float2 pq = __ldg(&d_pq[gr]);
            v = fmaf(b0_R, pq.x, fmaf(b1_R, pq.y, v));
        }
        return v;
    };

    float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int64_t idx4 = g4; idx4 < n4; idx4 += stride4) {
        int64_t g = idx4 * 4LL;
        if (g > (int64_t)K_bnd + 4LL && (M + 1LL - (g + 3LL)) > (int64_t)K_bnd + 4LL) {
            ((float4 *)buf1)[idx4] = zero4;
            ((float4 *)out0)[idx4] = zero4;
        } else {
            for (int r = 0; r < 4; r++) {
                int64_t gi = g + r;
                if (gi == 0LL) {
                    out0[0] = b0_L;
                } else if (gi == (int64_t)n1 - 1LL) {
                    out0[n1 - 1] = b0_R;
                } else if (gi >= 1LL && gi <= M) {
                    float vm = eval_b1(gi - 1LL);
                    float vc = eval_b1(gi);
                    float vp = eval_b1(gi + 1LL);
                    buf1[gi] = vc;
                    out0[gi] = jstep(vm, vc, vp);
                }
            }
        }
    }
}

// ============================================================================
// Host Precomputation Helpers (called only in pb_setup)
// ============================================================================

static void host_fft_f64(double *re, double *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / len;
        double wlen_r = cos(ang), wlen_i = sin(ang);
        for (int i = 0; i < n; i += len) {
            double wr = 1.0, wi = 0.0;
            for (int j = 0; j < len / 2; j++) {
                double ur = re[i + j], ui = im[i + j];
                double vr = re[i + j + len / 2] * wr - im[i + j + len / 2] * wi;
                double vi = re[i + j + len / 2] * wi + im[i + j + len / 2] * wr;
                re[i + j] = ur + vr;
                im[i + j] = ui + vi;
                re[i + j + len / 2] = ur - vr;
                im[i + j + len / 2] = ui - vi;
                double nwr = wr * wlen_r - wi * wlen_i;
                wi = wr * wlen_i + wi * wlen_r;
                wr = nwr;
            }
        }
    }
}

static int compute_pq_host(int64_t n0, int64_t M, int K_bnd, float2 *h_pq) {
    int64_t S = 2LL * n0 - 1LL;
    int K_inf = (S < (int64_t)MAX_K_BND) ? (int)S : MAX_K_BND;
    int N_fft = 32768;
    if (S <= (int64_t)MAX_K_BND) {
        N_fft = 256;
        while (N_fft <= 2 * K_inf + 2 && N_fft < 32768) N_fft <<= 1;
    }

    double *re = (double *)calloc((size_t)N_fft, sizeof(double));
    double *im = (double *)calloc((size_t)N_fft, sizeof(double));
    double *p_inf = (double *)calloc((size_t)(K_inf + 1), sizeof(double));
    double *q_inf = (double *)calloc((size_t)(K_inf + 1), sizeof(double));
    if (!re || !im || !p_inf || !q_inf) {
        free(re); free(im); free(p_inf); free(q_inf);
        return 1;
    }

    for (int m = 1; m < N_fft / 2; m++) {
        double theta = 2.0 * M_PI * (double)m / (double)N_fft;
        double lam = C_F64 * (1.0 + 2.0 * cos(theta));
        double lam2 = lam * lam;
        double s = sin(theta);
        double p_pow = (lam2 > 0.0) ? -expm1((double)n0 * log(lam2)) : 1.0;
        double q_pow = (n0 > 1) ? ((lam2 > 0.0) ? -expm1((double)(n0 - 1) * log(lam2)) : 1.0) : 0.0;
        double fp = C_F64 * s * p_pow / (1.0 - lam2);
        double fq = C_F64 * s * lam * q_pow / (1.0 - lam2);
        re[m] = fp; re[N_fft - m] = -fp;
        im[m] = fq; im[N_fft - m] = -fq;
    }

    host_fft_f64(re, im, N_fft);
    double scale = 2.0 / (double)N_fft;
    for (int k = 1; k <= K_inf; k++) {
        p_inf[k] = -im[k] * scale;
        q_inf[k] =  re[k] * scale;
    }

    h_pq[0] = make_float2(0.0f, 0.0f);
    int64_t P = 2LL * (M + 1LL);
    for (int i = 1; i <= K_bnd; i++) {
        double pi = 0.0, qi = 0.0;
        for (int64_t r = 0; ; r++) {
            int64_t idx1 = (int64_t)i + r * P;
            int64_t idx2 = (r + 1LL) * P - (int64_t)i;
            if (idx1 > K_inf && idx2 > K_inf) break;
            if (idx1 <= K_inf) { pi += p_inf[idx1]; qi += q_inf[idx1]; }
            if (idx2 <= K_inf) { pi -= p_inf[idx2]; qi -= q_inf[idx2]; }
        }
        h_pq[i] = make_float2((float)pi, (float)qi);
    }

    free(re); free(im); free(p_inf); free(q_inf);
    return 0;
}

static int compute_exact_m_cut(int L, int64_t S_steps, bool pre_filter) {
    int N = L / 2;
    double H0 = pow(3.0 * C_F64, (double)S_steps);
    double thresh = 1e-14 * H0;
    int m_cut = 0;
    int max_m = pre_filter ? (L / 4 - 1) : N;
    for (int m = 1; m <= max_m; m++) {
        double th = 2.0 * M_PI * (double)m / (double)L;
        double lam = C_F64 * (1.0 + 2.0 * cos(th));
        bool nz = false;
        if (pre_filter) {
            double f5 = cos(th * 0.5) * sin(2.0 * th) / (4.0 * sin(th * 0.5));
            double Hrem = pow(lam, (double)(S_steps - 2)) * (9.0 * C_F64 * C_F64);
            nz = (fabs(Hrem) > thresh && fabs(f5) > 0.12);
        } else {
            nz = (fabs(pow(lam, (double)S_steps)) > thresh);
        }
        if (nz) {
            int k = (m <= N / 2) ? m : (N - m);
            if (k < N / 2 && k > m_cut) m_cut = k;
        }
    }
    return m_cut;
}

static void build_H_abc(int L, int64_t S_steps, bool pre_filter, float4 *h_abc, int *m_cut_out) {
    int N = L / 2;
    double invN = 1.0 / (double)N;
    double H0 = pow(3.0 * C_F64, (double)S_steps);
    double thresh = 1e-14 * H0;
    double *H = (double *)calloc((size_t)(N + 1), sizeof(double));
    for (int m = 0; m <= N; m++) {
        double th = 2.0 * M_PI * (double)m / (double)L;
        double lam = C_F64 * (1.0 + 2.0 * cos(th));
        if (pre_filter) {
            if (m < L / 4) {
                double f5 = (m == 0) ? 1.0 : cos(th * 0.5) * sin(2.0 * th) / (4.0 * sin(th * 0.5));
                double Hrem = pow(lam, (double)(S_steps - 2)) * (9.0 * C_F64 * C_F64);
                if (fabs(Hrem) > thresh && fabs(f5) > 0.12) {
                    H[m] = Hrem / f5;
                }
            }
        } else {
            double val = pow(lam, (double)S_steps);
            if (fabs(val) > thresh) H[m] = val;
        }
    }

    {
        double SH = 0.5 * (H[0] + H[N]);
        double DH = 0.5 * (H[0] - H[N]);
        h_abc[0] = make_float4((float)(SH * invN), (float)(DH * invN), (float)(SH * invN), 0.0f);
    }
    h_abc[N / 2] = make_float4((float)(H[N / 2] * invN), 0.0f, 0.0f, 0.0f);

    int m_cut = 0;
    for (int k = 1; k < N / 2; k++) {
        double th = 2.0 * M_PI * (double)k / (double)L;
        double SH = 0.5 * (H[k] + H[N - k]);
        double DH = 0.5 * (H[k] - H[N - k]);
        double Ak = (SH - DH * sin(th)) * invN;
        double Bk = (DH * cos(th)) * invN;
        double Ck = (SH + DH * sin(th)) * invN;
        h_abc[k] = make_float4((float)Ak, (float)Bk, (float)Ck, 0.0f);
        if (H[k] != 0.0 || H[N - k] != 0.0) {
            m_cut = k;
        }
    }
    *m_cut_out = m_cut;
    free(H);
}

// ============================================================================
// Persistent Solver State
// ============================================================================

enum SolverMode {
    MODE_NONE = 0,
    MODE_NOOP = 1,
    MODE_DIRECT_N0_1 = 2,
    MODE_DIRECT_WARP = 3,
    MODE_FFT_SINGLE = 4,
    MODE_FFT_MULTI = 5,
    MODE_STEADY = 6
};

struct PassPlan {
    int L;
    int R;
    int B;
    bool pre_filter;
    bool prune;
    int64_t S_steps;
    int m_cut;
    int64_t blk_int_lo;
    int64_t blk_int_hi;
    int grid;
    bool use_split;
    float4 *d_H_abc;
};

struct SolverState {
    bool initialized;
    int n0;
    int n1;
    SolverMode mode;

    float *d_buf0;
    float *d_buf1;
    float *d_tmp;

    int direct_B;
    int direct_H;
    int64_t direct_w_lo;
    int64_t direct_w_hi;
    int64_t direct_num_warps;
    bool direct_use_split;

    int K_bnd;
    float2 *d_pq;
    float2 *d_tw;
    int tw_L;

    int num_passes;
    PassPlan passes[6];

    cudaStream_t stream_bnd;
    cudaEvent_t ev_int;
    cudaEvent_t ev_bnd;
};

static SolverState g_st = {};

static int check_buffer_shape_matches(const pb_buffer_shape *b, int64_t n1) {
    if (!b || b->dtype != PB_F32 || b->rank != 1 || b->shape[0] != n1) return 0;
    if ((uint64_t)n1 > UINT64_MAX / sizeof(float)) return 0;
    return b->nbytes == (uint64_t)n1 * sizeof(float);
}

static int check_buffer_runtime(const pb_buffer *b, int64_t n1) {
    if (!b || !b->data || b->dtype != PB_F32 || b->rank != 1 || b->shape[0] != n1) return 0;
    if ((uint64_t)n1 > UINT64_MAX / sizeof(float)) return 0;
    if (b->nbytes != (uint64_t)n1 * sizeof(float)) return 0;
    return ((uintptr_t)b->data % sizeof(float)) == 0;
}

static int align_R(int R_raw, bool write_final) {
    return write_final ? ((((R_raw + 1) + 3) & ~3) - 1) : ((R_raw + 3) & ~3);
}

static int compute_B(int L, int R, bool write_final) {
    int extra = write_final ? 2 : 0;
    return (L - 2 * R - extra) & ~3;
}

static int band_for_L(int L) {
    if (L == 2048 || L == 4096 || L == 8192) return 256;
    if (L == 16384) return 512;
    if (L == 32768) return 1024;
    return 0;
}

static int choose_fft_L(int64_t M, int R_raw, int64_t S_steps, bool write_final) {
    int R = align_R(R_raw, write_final);
    const int candidates[] = {1024, 2048, 4096, 8192, 16384, 32768};
    int best_L = 32768;
    double best_score = 1e30;

    for (int ci = 0; ci < 6; ci++) {
        int L = candidates[ci];
        int B = compute_B(L, R, write_final);
        if (B <= 0) continue;
        if (L < 32768 && B < L / 4) continue;

        int m_cut = compute_exact_m_cut(L, S_steps, true);
        int band = band_for_L(L);
        bool prune = (band > 0) && (m_cut < band);
        // L >= 4096 are always instantiated pruned; skip if not prunable
        if (L >= 4096 && !prune) continue;

        int64_t blocks = (M + (int64_t)B) / (int64_t)B;
        int blocks_per_sm = 1024 / (L / 32);
        if (blocks_per_sm > 32) blocks_per_sm = 32;
        int64_t wave_cap = 148LL * blocks_per_sm;
        int64_t waves = (blocks + wave_cap - 1) / wave_cap;
        double read_amp = (double)L / (double)B;

        // Empirically calibrated on B300 (SM 10.3) pure-interior sweep (us for 67M elems):
        double per_elem_cost;
        if (L == 1024)       per_elem_cost = 118.0 + 114.0 * read_amp;
        else if (L == 2048)  per_elem_cost = prune ? (130.0 +  90.0 * read_amp) : (120.0 + 125.0 * read_amp);
        else if (L == 4096)  per_elem_cost = 155.0 +  60.0 * read_amp;
        else if (L == 8192)  per_elem_cost = 124.0 + 100.0 * read_amp;
        else if (L == 16384) per_elem_cost =  55.0 + 200.0 * read_amp;
        else                 per_elem_cost = 128.0 + 203.0 * read_amp;

        double cost;
        if (blocks < 148) {
            // Latency-dominated regime: favor smaller L for fewer stages and better SM spread
            cost = (double)L * (prune ? 0.85 : 1.0) * (double)waves;
        } else {
            double sm_util = (double)blocks / (double)(waves * wave_cap);
            cost = per_elem_cost / (0.35 + 0.65 * sm_util);
        }
        if (cost < best_score) {
            best_score = cost;
            best_L = L;
        }
    }
    return best_L;
}

template <int L, bool PRE_FILTER, bool PRUNE, bool WRITE_FINAL, bool PURE_INTERIOR>
static cudaError_t configure_and_warm_one() {
    size_t dyn_smem = (L <= 8192) ? 0 : (size_t)L * sizeof(float);
    if (dyn_smem > 49152) {
        cudaError_t err = cudaFuncSetAttribute(
            fft_overlap_save_kernel<L, PRE_FILTER, PRUNE, WRITE_FINAL, PURE_INTERIOR>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)dyn_smem);
        if (err != cudaSuccess) return err;
    }
    cudaFuncAttributes attr;
    return cudaFuncGetAttributes(&attr, fft_overlap_save_kernel<L, PRE_FILTER, PRUNE, WRITE_FINAL, PURE_INTERIOR>);
}

template <int L, bool PRE_FILTER, bool PRUNE, bool WRITE_FINAL>
static cudaError_t configure_and_warm_pair(bool use_split) {
    cudaError_t err = configure_and_warm_one<L, PRE_FILTER, PRUNE, WRITE_FINAL, false>();
    if (err != cudaSuccess) return err;
    if (use_split) {
        err = configure_and_warm_one<L, PRE_FILTER, PRUNE, WRITE_FINAL, true>();
    }
    return err;
}

template <int L, bool PRE_FILTER, bool PRUNE, bool WRITE_FINAL>
static cudaError_t launch_fft_variant(
    const PassPlan &plan,
    const float *src_in,
    const float *orig_buf0,
    float *dst_buf1,
    float *dst_out0)
{
    size_t dyn_smem = (L <= 8192) ? 0 : (size_t)L * sizeof(float);
    int threads = L / 32;
    if (plan.use_split) {
        int n_int = (int)(plan.blk_int_hi - plan.blk_int_lo);
        int n_bnd = (int)(plan.blk_int_lo + ((int64_t)plan.grid - plan.blk_int_hi));
        cudaError_t err = cudaSuccess;
        if (n_bnd > 0) {
            fft_overlap_save_kernel<L, PRE_FILTER, PRUNE, WRITE_FINAL, false><<<n_bnd, threads, dyn_smem, g_st.stream_bnd>>>(
                src_in, orig_buf0, dst_buf1, dst_out0, g_st.d_tw, plan.d_H_abc, g_st.d_pq,
                g_st.n1, plan.R, plan.B, g_st.K_bnd, plan.m_cut, plan.blk_int_lo, plan.blk_int_hi);
            err = cudaGetLastError();
            if (err != cudaSuccess) return err;
        }
        fft_overlap_save_kernel<L, PRE_FILTER, PRUNE, WRITE_FINAL, true><<<n_int, threads, dyn_smem, 0>>>(
            src_in, orig_buf0, dst_buf1, dst_out0, g_st.d_tw, plan.d_H_abc, g_st.d_pq,
            g_st.n1, plan.R, plan.B, g_st.K_bnd, plan.m_cut, plan.blk_int_lo, plan.blk_int_hi);
        return cudaGetLastError();
    } else {
        fft_overlap_save_kernel<L, PRE_FILTER, PRUNE, WRITE_FINAL, false><<<plan.grid, threads, dyn_smem>>>(
            src_in, orig_buf0, dst_buf1, dst_out0, g_st.d_tw, plan.d_H_abc, g_st.d_pq,
            g_st.n1, plan.R, plan.B, g_st.K_bnd, plan.m_cut, (int64_t)plan.grid, (int64_t)plan.grid);
        return cudaGetLastError();
    }
}

static cudaError_t launch_fft_pass(
    const PassPlan &plan,
    bool write_final,
    const float *src_in,
    const float *orig_buf0,
    float *dst_buf1,
    float *dst_out0)
{
    if (plan.L == 1024)  return launch_fft_variant<1024,  true, false, true>(plan, src_in, orig_buf0, dst_buf1, dst_out0);
    if (plan.L == 2048) {
        if (plan.prune)  return launch_fft_variant<2048,  true, true,  true>(plan, src_in, orig_buf0, dst_buf1, dst_out0);
        else             return launch_fft_variant<2048,  true, false, true>(plan, src_in, orig_buf0, dst_buf1, dst_out0);
    }
    if (plan.L == 4096)  return launch_fft_variant<4096,  true, true,  true>(plan, src_in, orig_buf0, dst_buf1, dst_out0);
    if (plan.L == 8192)  return launch_fft_variant<8192,  true, true,  true>(plan, src_in, orig_buf0, dst_buf1, dst_out0);
    if (plan.L == 16384) return launch_fft_variant<16384, true, true,  true>(plan, src_in, orig_buf0, dst_buf1, dst_out0);
    if (plan.L == 32768) {
        if (plan.pre_filter && write_final)  return launch_fft_variant<32768, true,  true, true >(plan, src_in, orig_buf0, dst_buf1, dst_out0);
        if (plan.pre_filter && !write_final) return launch_fft_variant<32768, true,  true, false>(plan, src_in, orig_buf0, dst_buf1, dst_out0);
        if (!plan.pre_filter && write_final) return launch_fft_variant<32768, false, true, true >(plan, src_in, orig_buf0, dst_buf1, dst_out0);
        return launch_fft_variant<32768, false, true, false>(plan, src_in, orig_buf0, dst_buf1, dst_out0);
    }
    return cudaErrorInvalidValue;
}

static cudaError_t configure_pass(const PassPlan &plan, bool write_final) {
    if (plan.L == 1024)  return configure_and_warm_pair<1024,  true, false, true>(plan.use_split);
    if (plan.L == 2048)  return plan.prune ? configure_and_warm_pair<2048, true, true, true>(plan.use_split)
                                           : configure_and_warm_pair<2048, true, false, true>(plan.use_split);
    if (plan.L == 4096)  return configure_and_warm_pair<4096,  true, true,  true>(plan.use_split);
    if (plan.L == 8192)  return configure_and_warm_pair<8192,  true, true,  true>(plan.use_split);
    if (plan.L == 16384) return configure_and_warm_pair<16384, true, true,  true>(plan.use_split);
    if (plan.L == 32768) {
        if (plan.pre_filter && write_final)  return configure_and_warm_pair<32768, true,  true, true >(plan.use_split);
        if (plan.pre_filter && !write_final) return configure_and_warm_pair<32768, true,  true, false>(plan.use_split);
        if (!plan.pre_filter && write_final) return configure_and_warm_pair<32768, false, true, true >(plan.use_split);
        return configure_and_warm_pair<32768, false, true, false>(plan.use_split);
    }
    return cudaErrorInvalidValue;
}

static void compute_pass_split(PassPlan &p, int64_t M, int K_bnd, bool write_final) {
    p.grid = (int)((M + (int64_t)p.B) / (int64_t)p.B);
    int64_t pad_f = p.pre_filter ? 4LL : 0LL;
    int64_t off = write_final ? 1LL : 0LL;
    int64_t min_out = std::max<int64_t>(4LL, (int64_t)p.R + off + pad_f + 1LL);
    if (write_final) min_out = std::max<int64_t>(min_out, (int64_t)K_bnd + 2LL);
    int64_t lo = (min_out + (int64_t)p.B - 1LL) / (int64_t)p.B;

    int64_t max_out_load = M - (int64_t)p.L + 1LL - pad_f + (int64_t)p.R + off;
    int64_t max_out_store = write_final ? (M - (int64_t)K_bnd - (int64_t)p.B) : (M + 1LL - (int64_t)p.B);
    int64_t max_out = std::min(max_out_load, max_out_store);
    int64_t hi = (max_out >= 0LL) ? (max_out / (int64_t)p.B + 1LL) : 0LL;
    if (hi > (int64_t)p.grid) hi = (int64_t)p.grid;

    if (hi - lo >= 128LL) {
        p.blk_int_lo = lo;
        p.blk_int_hi = hi;
        p.use_split = true;
    } else {
        p.blk_int_lo = (int64_t)p.grid;
        p.blk_int_hi = (int64_t)p.grid;
        p.use_split = false;
    }
}

static void cleanup_state() {
    if (g_st.stream_bnd) cudaStreamDestroy(g_st.stream_bnd);
    if (g_st.ev_int)     cudaEventDestroy(g_st.ev_int);
    if (g_st.ev_bnd)     cudaEventDestroy(g_st.ev_bnd);
    if (g_st.d_buf0) cudaFree(g_st.d_buf0);
    if (g_st.d_buf1) cudaFree(g_st.d_buf1);
    if (g_st.d_tmp)  cudaFree(g_st.d_tmp);
    if (g_st.d_pq)   cudaFree(g_st.d_pq);
    if (g_st.d_tw)   cudaFree(g_st.d_tw);
    for (int i = 0; i < g_st.num_passes; i++) {
        if (g_st.passes[i].d_H_abc) cudaFree(g_st.passes[i].d_H_abc);
    }
    memset(&g_st, 0, sizeof(g_st));
}

// ============================================================================
// Five-Phase ABI Entry Points (pb-solve-v3)
// ============================================================================

extern "C" int pb_setup(const pb_problem_shape *shape) {
    if (g_st.initialized) cleanup_state();
    if (!shape || shape->abi_version != PB_SOLVE_ABI_VERSION ||
        shape->dimension_count != 2 || shape->scalar_count != 0 ||
        shape->buffer_count != 2 || !shape->dimensions || !shape->buffers) {
        return 1;
    }
    if (shape->dimensions[0] <= 0 || shape->dimensions[0] > INT_MAX) return 1;
    if (shape->dimensions[1] <= 0 || shape->dimensions[1] > INT_MAX) return 1;

    int n0 = (int)shape->dimensions[0];
    int n1 = (int)shape->dimensions[1];
    if (!check_buffer_shape_matches(&shape->buffers[0], n1)) return 1;
    if (!check_buffer_shape_matches(&shape->buffers[1], n1)) return 1;

    g_st.n0 = n0;
    g_st.n1 = n1;

    size_t buf_bytes = (size_t)n1 * sizeof(float);
    if (cudaMalloc(&g_st.d_buf0, buf_bytes) != cudaSuccess) { cleanup_state(); return 1; }
    if (cudaMalloc(&g_st.d_buf1, buf_bytes) != cudaSuccess) { cleanup_state(); return 1; }

    int64_t M = (int64_t)n1 - 2LL;
    if (M <= 0) {
        g_st.mode = MODE_NOOP;
        g_st.initialized = true;
        return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : 1;
    }

    if (cudaMalloc(&g_st.d_tmp, buf_bytes) != cudaSuccess) { cleanup_state(); return 1; }

    int64_t S = 2LL * (int64_t)n0 - 1LL;

    if (n0 == 1) {
        g_st.mode = MODE_DIRECT_N0_1;
        cudaFuncAttributes attr;
        if (cudaFuncGetAttributes(&attr, direct_n0_1_kernel) != cudaSuccess) { cleanup_state(); return 1; }
        g_st.initialized = true;
        return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : 1;
    }

    if (n0 <= DIRECT_MAX_N0 || (n1 <= DIRECT_WARP_ELEMS && n0 <= 128)) {
        g_st.mode = MODE_DIRECT_WARP;
        if (n1 <= DIRECT_WARP_ELEMS) {
            g_st.direct_H = 0;
            g_st.direct_B = DIRECT_WARP_ELEMS;
        } else {
            g_st.direct_H = (2 * n0 + 3) & ~3;
            g_st.direct_B = DIRECT_WARP_ELEMS - 2 * g_st.direct_H;
        }
        int64_t B = (int64_t)g_st.direct_B;
        int64_t H = (int64_t)g_st.direct_H;
        int64_t num_warps = (M + B) / B;
        g_st.direct_num_warps = num_warps;
        // Warp w is strictly interior iff w*B - H >= 4 AND (w+1)*B + H <= n1 - 1
        int64_t w_lo = (H + 4LL + B - 1LL) / B;
        if (w_lo < 1LL) w_lo = 1LL; // keep warp 0 in boundary kernel to copy out0[0], out0[n1-1]
        int64_t max_end = (int64_t)n1 - 1LL - H;
        int64_t w_hi = (max_end >= B) ? (max_end / B) : 0LL;
        if (w_hi > num_warps) w_hi = num_warps;

        if (w_hi - w_lo >= 64LL) {
            g_st.direct_w_lo = w_lo;
            g_st.direct_w_hi = w_hi;
            g_st.direct_use_split = true;
            if (cudaStreamCreateWithFlags(&g_st.stream_bnd, cudaStreamNonBlocking) != cudaSuccess) {
                cleanup_state(); return 1;
            }
        } else {
            g_st.direct_w_lo = num_warps;
            g_st.direct_w_hi = num_warps;
            g_st.direct_use_split = false;
        }

        cudaFuncAttributes attr;
        if (cudaFuncGetAttributes(&attr, direct_warp_stencil_kernel<false>) != cudaSuccess) { cleanup_state(); return 1; }
        if (g_st.direct_use_split) {
            if (cudaFuncGetAttributes(&attr, direct_warp_stencil_kernel<true>) != cudaSuccess) { cleanup_state(); return 1; }
        }
        g_st.initialized = true;
        return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : 1;
    }

    int K_bnd = (int)std::min<int64_t>(M, std::min<int64_t>(S, (int64_t)MAX_K_BND));
    g_st.K_bnd = K_bnd;
    float2 *h_pq = (float2 *)malloc((size_t)(K_bnd + 1) * sizeof(float2));
    if (!h_pq) { cleanup_state(); return 1; }
    if (compute_pq_host(n0, M, K_bnd, h_pq) != 0) { free(h_pq); cleanup_state(); return 1; }
    if (cudaMalloc(&g_st.d_pq, (size_t)(K_bnd + 1) * sizeof(float2)) != cudaSuccess ||
        cudaMemcpy(g_st.d_pq, h_pq, (size_t)(K_bnd + 1) * sizeof(float2), cudaMemcpyHostToDevice) != cudaSuccess) {
        free(h_pq); cleanup_state(); return 1;
    }
    free(h_pq);

    if (S >= S_STEADY_MIN) {
        g_st.mode = MODE_STEADY;
        cudaFuncAttributes attr;
        if (cudaFuncGetAttributes(&attr, steady_boundary_kernel) != cudaSuccess) { cleanup_state(); return 1; }
        g_st.initialized = true;
        return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : 1;
    }

    if (S <= S_SINGLE_PASS_MAX) {
        g_st.mode = MODE_FFT_SINGLE;
        g_st.num_passes = 1;
        PassPlan &p = g_st.passes[0];
        p.pre_filter = true;
        p.S_steps = S;
        int R_raw = (int)ceil(6.0 * sqrt((double)S));
        if ((int64_t)R_raw > S) R_raw = (int)S;
        R_raw += 4;
        p.L = choose_fft_L(M, R_raw, S, true);
        p.R = align_R(R_raw, true);
        p.B = compute_B(p.L, p.R, true);
        g_st.tw_L = p.L;
    } else {
        g_st.mode = MODE_FFT_MULTI;
        int k = (int)((S + 3200000LL - 1LL) / 3200000LL);
        if (k > 6) k = 6;
        g_st.num_passes = k;
        g_st.tw_L = 32768;
        int64_t base_s = S / k;
        int64_t extra_s = S % k;
        for (int i = 0; i < k; i++) {
            PassPlan &p = g_st.passes[i];
            p.pre_filter = (i == 0);
            p.S_steps = base_s + (i < extra_s ? 1 : 0);
            bool is_last = (i == k - 1);
            int R_raw = (int)ceil(6.0 * sqrt((double)p.S_steps)) + (p.pre_filter ? 4 : 0);
            if ((int64_t)R_raw > p.S_steps + 4) R_raw = (int)(p.S_steps + 4);
            p.L = 32768;
            p.R = align_R(R_raw, is_last);
            p.B = compute_B(p.L, p.R, is_last);
        }
    }

    int tw_N = g_st.tw_L / 2;
    float2 *h_tw = (float2 *)malloc((size_t)tw_N * sizeof(float2));
    if (!h_tw) { cleanup_state(); return 1; }
    for (int k = 0; k < tw_N; k++) {
        double ang = -2.0 * M_PI * (double)k / (double)tw_N;
        h_tw[k] = make_float2((float)cos(ang), (float)sin(ang));
    }
    if (cudaMalloc(&g_st.d_tw, (size_t)tw_N * sizeof(float2)) != cudaSuccess ||
        cudaMemcpy(g_st.d_tw, h_tw, (size_t)tw_N * sizeof(float2), cudaMemcpyHostToDevice) != cudaSuccess) {
        free(h_tw); cleanup_state(); return 1;
    }
    free(h_tw);

    for (int i = 0; i < g_st.num_passes; i++) {
        PassPlan &p = g_st.passes[i];
        bool is_last = (i == g_st.num_passes - 1);
        int N = p.L / 2;
        float4 *h_abc = (float4 *)malloc((size_t)(N / 2 + 1) * sizeof(float4));
        if (!h_abc) { cleanup_state(); return 1; }
        build_H_abc(p.L, p.S_steps, p.pre_filter, h_abc, &p.m_cut);

        int band = band_for_L(p.L);
        p.prune = (band > 0) && (p.m_cut < band);
        if (p.L >= 4096 && !p.prune) { free(h_abc); cleanup_state(); return 1; }

        compute_pass_split(p, M, g_st.K_bnd, is_last);

        if (cudaMalloc(&p.d_H_abc, (size_t)(N / 2 + 1) * sizeof(float4)) != cudaSuccess ||
            cudaMemcpy(p.d_H_abc, h_abc, (size_t)(N / 2 + 1) * sizeof(float4), cudaMemcpyHostToDevice) != cudaSuccess) {
            free(h_abc); cleanup_state(); return 1;
        }
        free(h_abc);

        if (configure_pass(p, is_last) != cudaSuccess) {
            cleanup_state(); return 1;
        }
    }

    bool any_split = false;
    for (int i = 0; i < g_st.num_passes; i++) {
        if (g_st.passes[i].use_split) any_split = true;
    }
    if (any_split) {
        if (cudaStreamCreateWithFlags(&g_st.stream_bnd, cudaStreamNonBlocking) != cudaSuccess) {
            cleanup_state(); return 1;
        }
        if (g_st.num_passes > 1) {
            if (cudaEventCreateWithFlags(&g_st.ev_int, cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&g_st.ev_bnd, cudaEventDisableTiming) != cudaSuccess) {
                cleanup_state(); return 1;
            }
        }
    }

    g_st.initialized = true;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : 1;
}

extern "C" int pb_upload(const pb_problem *problem, pb_buffer *buffers) {
    if (!g_st.initialized || !problem || !buffers) return 1;
    if (problem->abi_version != PB_SOLVE_ABI_VERSION ||
        problem->dimension_count != 2 || problem->scalar_count != 0 ||
        problem->buffer_count != 2 || !problem->dimensions) return 1;
    if (problem->dimensions[0] != g_st.n0 || problem->dimensions[1] != g_st.n1) return 1;
    if (!check_buffer_runtime(&buffers[0], g_st.n1)) return 1;
    if (!check_buffer_runtime(&buffers[1], g_st.n1)) return 1;

    uintptr_t s0 = (uintptr_t)buffers[0].data, s1 = (uintptr_t)buffers[1].data;
    if (buffers[0].nbytes > UINTPTR_MAX - s0 || buffers[1].nbytes > UINTPTR_MAX - s1) return 1;
    if (s0 < s1 + buffers[1].nbytes && s1 < s0 + buffers[0].nbytes) return 1;

    size_t nbytes = (size_t)g_st.n1 * sizeof(float);
    if (cudaMemcpy(g_st.d_buf0, buffers[0].data, nbytes, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    if (cudaMemcpy(g_st.d_buf1, buffers[1].data, nbytes, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : 1;
}

extern "C" int pb_compute(void) {
    if (!g_st.initialized) return 1;
    if (g_st.mode == MODE_NOOP) return 0;

    cudaError_t err = cudaSuccess;

    if (g_st.mode == MODE_DIRECT_N0_1) {
        int64_t n4 = ((int64_t)g_st.n1 + 3LL) >> 2;
        int grid = (int)((n4 + 255LL) / 256LL);
        direct_n0_1_kernel<<<grid, 256>>>(g_st.d_buf0, g_st.d_buf1, g_st.d_tmp, g_st.n1);
        err = cudaGetLastError();
        if (err == cudaSuccess) std::swap(g_st.d_buf0, g_st.d_tmp);
    } else if (g_st.mode == MODE_DIRECT_WARP) {
        if (g_st.direct_use_split) {
            int64_t n_bnd_w = g_st.direct_w_lo + (g_st.direct_num_warps - g_st.direct_w_hi);
            if (n_bnd_w > 0) {
                int grid_bnd = (int)((n_bnd_w + 7LL) / 8LL);
                direct_warp_stencil_kernel<false><<<grid_bnd, 256, 0, g_st.stream_bnd>>>(
                    g_st.d_buf0, g_st.d_buf1, g_st.d_tmp,
                    g_st.n0, g_st.n1, g_st.direct_B, g_st.direct_H,
                    g_st.direct_w_lo, g_st.direct_w_hi, g_st.direct_num_warps);
                err = cudaGetLastError();
            }
            if (err == cudaSuccess) {
                int64_t n_int_w = g_st.direct_w_hi - g_st.direct_w_lo;
                int grid_int = (int)((n_int_w + 7LL) / 8LL);
                direct_warp_stencil_kernel<true><<<grid_int, 256, 0, 0>>>(
                    g_st.d_buf0, g_st.d_buf1, g_st.d_tmp,
                    g_st.n0, g_st.n1, g_st.direct_B, g_st.direct_H,
                    g_st.direct_w_lo, g_st.direct_w_hi, g_st.direct_num_warps);
                err = cudaGetLastError();
            }
        } else {
            int grid = (int)((g_st.direct_num_warps + 7LL) / 8LL);
            direct_warp_stencil_kernel<false><<<grid, 256>>>(
                g_st.d_buf0, g_st.d_buf1, g_st.d_tmp,
                g_st.n0, g_st.n1, g_st.direct_B, g_st.direct_H,
                g_st.direct_num_warps, g_st.direct_num_warps, g_st.direct_num_warps);
            err = cudaGetLastError();
        }
        if (err == cudaSuccess) std::swap(g_st.d_buf0, g_st.d_tmp);
    } else if (g_st.mode == MODE_FFT_SINGLE) {
        err = launch_fft_pass(g_st.passes[0], true, g_st.d_buf0, g_st.d_buf0, g_st.d_buf1, g_st.d_tmp);
        if (err == cudaSuccess) std::swap(g_st.d_buf0, g_st.d_tmp);
    } else if (g_st.mode == MODE_FFT_MULTI) {
        float *cur_src = g_st.d_buf0;
        float *cur_dst = g_st.d_tmp;
        for (int i = 0; i < g_st.num_passes; i++) {
            bool is_last = (i == g_st.num_passes - 1);
            if (!is_last) {
                err = launch_fft_pass(g_st.passes[i], false, cur_src, g_st.d_buf0, cur_dst, nullptr);
                if (err != cudaSuccess) break;
                // Synchronize concurrent streams on GPU before next pass reads cur_dst
                if (g_st.passes[i].use_split) {
                    err = cudaEventRecord(g_st.ev_bnd, g_st.stream_bnd);
                    if (err == cudaSuccess) err = cudaStreamWaitEvent(0, g_st.ev_bnd, 0);
                }
                if (err == cudaSuccess && g_st.passes[i + 1].use_split) {
                    err = cudaEventRecord(g_st.ev_int, 0);
                    if (err == cudaSuccess) err = cudaStreamWaitEvent(g_st.stream_bnd, g_st.ev_int, 0);
                }
                if (err != cudaSuccess) break;
                std::swap(cur_src, cur_dst);
            } else {
                err = launch_fft_pass(g_st.passes[i], true, cur_src, g_st.d_buf0, g_st.d_buf1, cur_dst);
                if (err != cudaSuccess) break;
                if (cur_dst != g_st.d_buf0) std::swap(g_st.d_buf0, g_st.d_tmp);
            }
        }
    } else if (g_st.mode == MODE_STEADY) {
        int64_t n4 = ((int64_t)g_st.n1 + 3LL) >> 2;
        int blocks = (int)std::min<int64_t>((n4 + 255LL) / 256LL, 148LL * 16LL);
        if (blocks < 1) blocks = 1;
        steady_boundary_kernel<<<blocks, 256>>>(
            g_st.d_buf0, g_st.d_buf1, g_st.d_tmp,
            g_st.d_pq, g_st.n1, g_st.K_bnd);
        err = cudaGetLastError();
        if (err == cudaSuccess) std::swap(g_st.d_buf0, g_st.d_tmp);
    } else {
        return 1;
    }

    if (err != cudaSuccess) return 1;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : 1;
}

extern "C" int pb_download(pb_buffer *buffers) {
    if (!g_st.initialized || !buffers) return 1;
    if (!check_buffer_runtime(&buffers[0], g_st.n1)) return 1;
    if (!check_buffer_runtime(&buffers[1], g_st.n1)) return 1;

    uintptr_t s0 = (uintptr_t)buffers[0].data, s1 = (uintptr_t)buffers[1].data;
    if (buffers[0].nbytes > UINTPTR_MAX - s0 || buffers[1].nbytes > UINTPTR_MAX - s1) return 1;
    if (s0 < s1 + buffers[1].nbytes && s1 < s0 + buffers[0].nbytes) return 1;

    size_t nbytes = (size_t)g_st.n1 * sizeof(float);
    if (cudaMemcpy(buffers[0].data, g_st.d_buf0, nbytes, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (cudaMemcpy(buffers[1].data, g_st.d_buf1, nbytes, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : 1;
}

extern "C" int pb_teardown(void) {
    if (!g_st.initialized) return 0;
    cleanup_state();
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : 1;
}
