#include "interface.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <climits>
#include <cstdio>
#include <vector>

// ============================================================================
// Double-Float (48-bit mantissa emulated on FP32 FMA/FADD pipes)
// ============================================================================

struct __align__(8) df {
    float hi;
    float lo;
};

struct __align__(16) cdf {
    df r;
    df i;
};

__host__ __device__ __forceinline__ df df_from_double(double d) {
    float hi = static_cast<float>(d);
    float lo = static_cast<float>(d - static_cast<double>(hi));
    return {hi, lo};
}

__host__ __device__ __forceinline__ df df_neg(df a) {
    return {-a.hi, -a.lo};
}

__device__ __forceinline__ df df_renorm(df a) {
    float hi = __fadd_rn(a.hi, a.lo);
    float lo = __fsub_rn(a.lo, __fsub_rn(hi, a.hi));
    return {hi, lo};
}

__device__ __forceinline__ df df_add_u(df a, df b) {
    float s = __fadd_rn(a.hi, b.hi);
    float v = __fsub_rn(s, a.hi);
    float e = __fadd_rn(__fsub_rn(a.hi, __fsub_rn(s, v)), __fsub_rn(b.hi, v));
    e = __fadd_rn(e, __fadd_rn(a.lo, b.lo));
    return {s, e};
}

__device__ __forceinline__ df df_sub_u(df a, df b) {
    return df_add_u(a, {-b.hi, -b.lo});
}

__device__ __forceinline__ df df_add(df a, df b) {
    return df_renorm(df_add_u(a, b));
}

__device__ __forceinline__ df df_sub(df a, df b) {
    return df_renorm(df_sub_u(a, b));
}

__device__ __forceinline__ df df_mul(df a, df b) {
    float p = __fmul_rn(a.hi, b.hi);
    float e = __fmaf_rn(a.hi, b.hi, -p);
    e = __fmaf_rn(a.hi, b.lo, e);
    e = __fmaf_rn(a.lo, b.hi, e);
    float hi = __fadd_rn(p, e);
    float lo = __fsub_rn(e, __fsub_rn(hi, p));
    return {hi, lo};
}

__device__ __forceinline__ df df_mul_f(df a, float b) {
    float p = __fmul_rn(a.hi, b);
    float e = __fmaf_rn(a.hi, b, -p);
    e = __fmaf_rn(a.lo, b, e);
    float hi = __fadd_rn(p, e);
    float lo = __fsub_rn(e, __fsub_rn(hi, p));
    return {hi, lo};
}

__device__ __forceinline__ df df_rcp(df a) {
    float q1 = __fdividef(1.0f, a.hi);
    df r = df_sub({1.0f, 0.0f}, df_mul_f(a, q1));
    float q2 = __fmul_rn(r.hi, q1);
    return df_renorm({q1, q2});
}

__device__ __forceinline__ cdf cdf_add_u(cdf a, cdf b) {
    return {df_add_u(a.r, b.r), df_add_u(a.i, b.i)};
}

__device__ __forceinline__ cdf cdf_sub_u(cdf a, cdf b) {
    return {df_sub_u(a.r, b.r), df_sub_u(a.i, b.i)};
}

__device__ __forceinline__ cdf cdf_add(cdf a, cdf b) {
    return {df_add(a.r, b.r), df_add(a.i, b.i)};
}

__device__ __forceinline__ cdf cdf_sub(cdf a, cdf b) {
    return {df_sub(a.r, b.r), df_sub(a.i, b.i)};
}

__device__ __forceinline__ cdf cdf_mul_negi(cdf a) {
    return {a.i, df_neg(a.r)};
}

__device__ __forceinline__ cdf cdf_mul_posi(cdf a) {
    return {df_neg(a.i), a.r};
}

__device__ __forceinline__ cdf cdf_conj(cdf a) {
    return {a.r, df_neg(a.i)};
}

__device__ __forceinline__ cdf ldg_cdf(const cdf* p) {
    float4 v = __ldg(reinterpret_cast<const float4*>(p));
    return {{v.x, v.y}, {v.z, v.w}};
}

__device__ __forceinline__ df ldg_df(const df* p) {
    float2 v = __ldg(reinterpret_cast<const float2*>(p));
    return {v.x, v.y};
}

__device__ __forceinline__ void st_cdf(cdf* p, cdf c) {
    *reinterpret_cast<float4*>(p) = make_float4(c.r.hi, c.r.lo, c.i.hi, c.i.lo);
}

__device__ __forceinline__ void st_df(df* p, df d) {
    *reinterpret_cast<float2*>(p) = make_float2(d.hi, d.lo);
}

__device__ __forceinline__ cdf cdf_mul(cdf a, cdf b) {
    float p1 = __fmul_rn(a.r.hi, b.r.hi);
    float e1 = __fmaf_rn(a.r.hi, b.r.hi, -p1);
    float p2 = __fmul_rn(a.i.hi, b.i.hi);
    float e2 = __fmaf_rn(a.i.hi, b.i.hi, -p2);
    float s = __fsub_rn(p1, p2);
    float v = __fsub_rn(s, p1);
    float e = __fadd_rn(__fsub_rn(p1, __fsub_rn(s, v)), __fsub_rn(-p2, v));
    e = __fadd_rn(e, __fsub_rn(e1, e2));
    e = __fmaf_rn(a.r.hi, b.r.lo, e);
    e = __fmaf_rn(a.r.lo, b.r.hi, e);
    e = __fmaf_rn(-a.i.hi, b.i.lo, e);
    e = __fmaf_rn(-a.i.lo, b.i.hi, e);
    float rhi = __fadd_rn(s, e);
    float rlo = __fsub_rn(e, __fsub_rn(rhi, s));

    float q1 = __fmul_rn(a.r.hi, b.i.hi);
    float f1 = __fmaf_rn(a.r.hi, b.i.hi, -q1);
    float q2 = __fmul_rn(a.i.hi, b.r.hi);
    float f2 = __fmaf_rn(a.i.hi, b.r.hi, -q2);
    float t = __fadd_rn(q1, q2);
    float u = __fsub_rn(t, q1);
    float f = __fadd_rn(__fsub_rn(q1, __fsub_rn(t, u)), __fsub_rn(q2, u));
    f = __fadd_rn(f, __fadd_rn(f1, f2));
    f = __fmaf_rn(a.r.hi, b.i.lo, f);
    f = __fmaf_rn(a.r.lo, b.i.hi, f);
    f = __fmaf_rn(a.i.hi, b.r.lo, f);
    f = __fmaf_rn(a.i.lo, b.r.hi, f);
    float ihi = __fadd_rn(t, f);
    float ilo = __fsub_rn(f, __fsub_rn(ihi, t));

    return {{rhi, rlo}, {ihi, ilo}};
}

// ============================================================================
// FFT Radix-3 / Radix-4 / Radix-5 / Radix-8 Butterflies (Double-Float)
// ============================================================================

template <bool INV>
__device__ __forceinline__ void radix3_bfly(cdf& x0, cdf& x1, cdf& x2) {
    const df S3_2 = {0.86602538824081420898f, 1.5543624437692625e-08f};
    cdf t1 = cdf_add_u(x1, x2);
    cdf t2 = cdf_sub_u(x1, x2);
    cdf y0 = cdf_add(x0, t1);
    cdf m1 = {df_sub_u(x0.r, df_mul_f(t1.r, 0.5f)), df_sub_u(x0.i, df_mul_f(t1.i, 0.5f))};
    df a = df_mul(df_renorm(t2.i), S3_2);
    df b = df_mul(df_renorm(t2.r), S3_2);
    cdf m2 = INV ? cdf{df_neg(a), b} : cdf{a, df_neg(b)};
    x0 = y0;
    x1 = cdf_add(m1, m2);
    x2 = cdf_sub(m1, m2);
}

template <bool INV>
__device__ __forceinline__ void radix5_bfly(cdf& x0, cdf& x1, cdf& x2, cdf& x3, cdf& x4) {
    const df C1 = { 0.30901700258255004883f, -8.207602624727160e-09f};
    const df C2 = {-0.80901700258255004883f,  8.207602624727160e-09f};
    const df S1 = { 0.95105654001235961914f, -2.371720604701557e-08f};
    const df S2 = { 0.58778524398803710938f,  8.304436019795410e-09f};

    cdf t1 = cdf_add(x1, x4), t3 = cdf_sub(x1, x4);
    cdf t2 = cdf_add(x2, x3), t4 = cdf_sub(x2, x3);

    cdf m1 = {df_add(x0.r, df_add(df_mul(t1.r, C1), df_mul(t2.r, C2))),
              df_add(x0.i, df_add(df_mul(t1.i, C1), df_mul(t2.i, C2)))};
    cdf n1 = {df_add(df_mul(t3.r, S1), df_mul(t4.r, S2)),
              df_add(df_mul(t3.i, S1), df_mul(t4.i, S2))};
    cdf j1 = INV ? cdf_mul_posi(n1) : cdf_mul_negi(n1);
    x1 = cdf_add(m1, j1);
    x4 = cdf_sub(m1, j1);

    cdf m2 = {df_add(x0.r, df_add(df_mul(t1.r, C2), df_mul(t2.r, C1))),
              df_add(x0.i, df_add(df_mul(t1.i, C2), df_mul(t2.i, C1)))};
    cdf n2 = {df_sub(df_mul(t3.r, S2), df_mul(t4.r, S1)),
              df_sub(df_mul(t3.i, S2), df_mul(t4.i, S1))};
    cdf j2 = INV ? cdf_mul_posi(n2) : cdf_mul_negi(n2);
    x0 = cdf_add(x0, cdf_add(t1, t2));
    x2 = cdf_add(m2, j2);
    x3 = cdf_sub(m2, j2);
}

template <bool INV, bool RENORM_OUT = true>
__device__ __forceinline__ void radix4_bfly(cdf& a0, cdf& a1, cdf& a2, cdf& a3) {
    cdf t0 = cdf_add_u(a0, a2);
    cdf t1 = cdf_sub_u(a0, a2);
    cdf t2 = cdf_add_u(a1, a3);
    cdf t3 = cdf_sub_u(a1, a3);
    cdf t3i = INV ? cdf_mul_posi(t3) : cdf_mul_negi(t3);
    if constexpr (RENORM_OUT) {
        a0 = cdf_add(t0, t2);
        a2 = cdf_sub(t0, t2);
        a1 = cdf_add(t1, t3i);
        a3 = cdf_sub(t1, t3i);
    } else {
        a0 = cdf_add_u(t0, t2);
        a2 = cdf_sub_u(t0, t2);
        a1 = cdf_add_u(t1, t3i);
        a3 = cdf_sub_u(t1, t3i);
    }
}

template <bool INV, bool RENORM_OUT = true>
__device__ __forceinline__ void radix8_bfly(cdf& x0, cdf& x1, cdf& x2, cdf& x3,
                                            cdf& x4, cdf& x5, cdf& x6, cdf& x7) {
    cdf a0 = cdf_add_u(x0, x4), a4 = cdf_sub_u(x0, x4);
    cdf a1 = cdf_add_u(x1, x5), a5 = cdf_sub_u(x1, x5);
    cdf a2 = cdf_add_u(x2, x6), a6 = cdf_sub_u(x2, x6);
    cdf a3 = cdf_add_u(x3, x7), a7 = cdf_sub_u(x3, x7);

    const df SQRT1_2 = {0.70710676908493041992f, 1.2101617104475617e-08f};
    if constexpr (!INV) {
        df r5 = df_mul(df_add_u(a5.r, a5.i), SQRT1_2);
        df i5 = df_mul(df_sub_u(a5.i, a5.r), SQRT1_2);
        a5 = {r5, i5};
        a6 = cdf_mul_negi(a6);
        df r7 = df_mul(df_sub_u(a7.i, a7.r), SQRT1_2);
        df i7 = df_mul(df_neg(df_add_u(a7.r, a7.i)), SQRT1_2);
        a7 = {r7, i7};
    } else {
        df r5 = df_mul(df_sub_u(a5.r, a5.i), SQRT1_2);
        df i5 = df_mul(df_add_u(a5.r, a5.i), SQRT1_2);
        a5 = {r5, i5};
        a6 = cdf_mul_posi(a6);
        df r7 = df_mul(df_neg(df_add_u(a7.r, a7.i)), SQRT1_2);
        df i7 = df_mul(df_sub_u(a7.r, a7.i), SQRT1_2);
        a7 = {r7, i7};
    }

    radix4_bfly<INV, RENORM_OUT>(a0, a1, a2, a3);
    radix4_bfly<INV, RENORM_OUT>(a4, a5, a6, a7);
    x0 = a0; x1 = a4; x2 = a1; x3 = a5;
    x4 = a2; x5 = a6; x6 = a3; x7 = a7;
}

// ============================================================================
// Spectral Evolution Helper (Exact 48-bit Geometric Series, Zero Cancellation)
// ============================================================================

__device__ __forceinline__ df spectral_evolve_mode(
    int n0,
    df u0,
    df sin2_p, df sin2_q,
    df sp1_p, df sp1_q,
    df tb0_p, df tb1_p,
    df lr0_q, df lr1_q
) {
    df delta = df_add(sin2_p, sin2_q);
    df lam = df_sub({1.0f, 0.0f}, delta);
    df eta = df_mul(delta, df_sub({2.0f, 0.0f}, delta));
    df lam2 = df_mul(lam, lam);

    df p_res = {0.0f, 0.0f};
    df p_base = eta;
    df pow_res = {1.0f, 0.0f};
    df pow_base = lam2;
    uint32_t k = static_cast<uint32_t>(n0 - 1);
    while (k > 0u) {
        if (k & 1u) {
            p_res = df_sub(df_add(p_res, p_base), df_mul(p_res, p_base));
            pow_res = df_mul(pow_res, pow_base);
        }
        k >>= 1;
        if (k > 0u) {
            p_base = df_mul(p_base, df_sub({2.0f, 0.0f}, p_base));
            pow_base = df_mul(pow_base, pow_base);
        }
    }

    df p_n0 = df_sub(df_add(eta, p_res), df_mul(eta, p_res));
    df inv_eta = df_rcp(eta);

    df alpha = df_mul(lam, pow_res);
    df beta  = df_mul(p_n0, inv_eta);
    df gamma = df_mul(lam, df_mul(p_res, inv_eta));

    df F0 = df_add(df_mul(sp1_p, tb0_p), df_mul(sp1_q, lr0_q));
    df F1 = df_add(df_mul(sp1_p, tb1_p), df_mul(sp1_q, lr1_q));

    df res = df_mul(alpha, u0);
    res = df_add(res, df_mul(beta, F0));
    res = df_add(res, df_mul(gamma, F1));
    return res;
}

// ============================================================================
// Warp-Shuffle Block Prefix Sum for Two Double-Float Arrays Simultaneously
// ============================================================================

template <int BLOCK_T>
__device__ __noinline__ void df_prefix_sum_pair_smem(df* s1, df* s2, int n, bool has2) {
    const int tx = threadIdx.x;
    const int lane = tx & 31;
    const int wid = tx >> 5;
    constexpr int NWARPS = BLOCK_T / 32;

    int chunk = (n + BLOCK_T - 1) / BLOCK_T;
    int start = tx * chunk;
    int end = min(n, start + chunk);

    df acc1 = {0.0f, 0.0f}, acc2 = {0.0f, 0.0f};
    for (int i = start; i < end; ++i) {
        acc1 = df_add(acc1, s1[i]);
        s1[i] = acc1;
        if (has2) {
            acc2 = df_add(acc2, s2[i]);
            s2[i] = acc2;
        }
    }

    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        df o1 = {__shfl_up_sync(0xffffffffu, acc1.hi, offset),
                 __shfl_up_sync(0xffffffffu, acc1.lo, offset)};
        if (lane >= offset) acc1 = df_add(acc1, o1);
        if (has2) {
            df o2 = {__shfl_up_sync(0xffffffffu, acc2.hi, offset),
                     __shfl_up_sync(0xffffffffu, acc2.lo, offset)};
            if (lane >= offset) acc2 = df_add(acc2, o2);
        }
    }

    __shared__ df warp_sums1[32], warp_sums2[32];
    if (lane == 31) {
        warp_sums1[wid] = acc1;
        if (has2) warp_sums2[wid] = acc2;
    }
    __syncthreads();

    if (wid == 0) {
        df w1 = (lane < NWARPS) ? warp_sums1[lane] : df{0.0f, 0.0f};
        df w2 = (has2 && lane < NWARPS) ? warp_sums2[lane] : df{0.0f, 0.0f};
        #pragma unroll
        for (int offset = 1; offset < NWARPS; offset <<= 1) {
            df o1 = {__shfl_up_sync(0xffffffffu, w1.hi, offset),
                     __shfl_up_sync(0xffffffffu, w1.lo, offset)};
            if (lane >= offset) w1 = df_add(w1, o1);
            if (has2) {
                df o2 = {__shfl_up_sync(0xffffffffu, w2.hi, offset),
                         __shfl_up_sync(0xffffffffu, w2.lo, offset)};
                if (lane >= offset) w2 = df_add(w2, o2);
            }
        }
        if (lane < NWARPS) {
            warp_sums1[lane] = w1;
            if (has2) warp_sums2[lane] = w2;
        }
    }
    __syncthreads();

    df prev1 = {__shfl_up_sync(0xffffffffu, acc1.hi, 1),
                __shfl_up_sync(0xffffffffu, acc1.lo, 1)};
    df carry1 = (lane > 0) ? prev1 : df{0.0f, 0.0f};
    if (wid > 0) carry1 = df_add(carry1, warp_sums1[wid - 1]);

    df carry2 = {0.0f, 0.0f};
    if (has2) {
        df prev2 = {__shfl_up_sync(0xffffffffu, acc2.hi, 1),
                    __shfl_up_sync(0xffffffffu, acc2.lo, 1)};
        carry2 = (lane > 0) ? prev2 : df{0.0f, 0.0f};
        if (wid > 0) carry2 = df_add(carry2, warp_sums2[wid - 1]);
    }

    if (tx > 0 && start < n) {
        for (int i = start; i < end; ++i) {
            s1[i] = df_add(s1[i], carry1);
            if (has2) s2[i] = df_add(s2[i], carry2);
        }
    }
    __syncthreads();
}

// ============================================================================
// Engine 1: Single-Block 100%-SMEM Direct Stencil (tiny n1, small n0)
// Zero integer divisions (2D thread block!)
// ============================================================================

__global__ void __launch_bounds__(512)
direct_smem1_kernel(
    int n0, int n1,
    float* __restrict__ d_buf0,
    float* __restrict__ d_buf1
) {
    extern __shared__ float smem[];
    const int stride = n1 + 1; // +1 bank-conflict avoidance
    const int plane = n1 * stride;
    float* s0 = smem;
    float* s1 = smem + plane;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockDim.x;
    const int by = blockDim.y;

    for (int i = ty; i < n1; i += by) {
        for (int j = tx; j < n1; j += bx) {
            s0[i * stride + j] = d_buf0[i * n1 + j];
            s1[i * stride + j] = d_buf1[i * n1 + j];
        }
    }
    __syncthreads();

    for (int t = 0; t < n0; ++t) {
        for (int i = 1 + ty; i < n1 - 1; i += by) {
            int row = i * stride;
            for (int j = 1 + tx; j < n1 - 1; j += bx) {
                int pos = row + j;
                s1[pos] = 0.2f * (s0[pos] + s0[pos - 1] + s0[pos + 1] + s0[pos + stride] + s0[pos - stride]);
            }
        }
        __syncthreads();
        for (int i = 1 + ty; i < n1 - 1; i += by) {
            int row = i * stride;
            for (int j = 1 + tx; j < n1 - 1; j += bx) {
                int pos = row + j;
                s0[pos] = 0.2f * (s1[pos] + s1[pos - 1] + s1[pos + 1] + s1[pos + stride] + s1[pos - stride]);
            }
        }
        __syncthreads();
    }

    for (int i = 1 + ty; i < n1 - 1; i += by) {
        for (int j = 1 + tx; j < n1 - 1; j += bx) {
            d_buf0[i * n1 + j] = s0[i * stride + j];
            d_buf1[i * n1 + j] = s1[i * stride + j];
        }
    }
}

// ============================================================================
// Engine 2: Multi-Step 2D Overlapped SMEM Temporally-Blocked Direct Stencil
// ============================================================================

__global__ void copy_border_kernel(int n1, const float* __restrict__ src, float* __restrict__ dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n1) {
        dst[idx] = src[idx];
        dst[static_cast<size_t>(n1 - 1) * n1 + idx] = src[static_cast<size_t>(n1 - 1) * n1 + idx];
        dst[static_cast<size_t>(idx) * n1] = src[static_cast<size_t>(idx) * n1];
        dst[static_cast<size_t>(idx) * n1 + (n1 - 1)] = src[static_cast<size_t>(idx) * n1 + (n1 - 1)];
    }
}

template <int IN_X, int IN_Y, int K, int BX, int BY, bool SAVE_PREV>
__global__ void __launch_bounds__(BX * BY)
jacobi_2d_smem_kernel(
    int n1,
    const float* __restrict__ d_src,
    const float* __restrict__ d_b0_orig,
    const float* __restrict__ d_b1_orig,
    float* __restrict__ d_dst,
    float* __restrict__ d_dst_prev
) {
    constexpr int OUT_X = IN_X - 2 * K;
    constexpr int OUT_Y = IN_Y - 2 * K;
    constexpr int SMEM_STRIDE = IN_X + 1;

    extern __shared__ float smem_raw[];
    float* smem[2] = {smem_raw, smem_raw + IN_Y * SMEM_STRIDE};

    const int ox0 = 1 + blockIdx.x * OUT_X;
    const int oy0 = 1 + blockIdx.y * OUT_Y;
    const int ix0 = ox0 - K;
    const int iy0 = oy0 - K;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const bool is_interior = (ix0 >= 1 && ix0 + IN_X <= n1 - 1 &&
                              iy0 >= 1 && iy0 + IN_Y <= n1 - 1);

    if (is_interior) {
        #pragma unroll
        for (int ly = ty; ly < IN_Y; ly += BY) {
            const float* row_ptr = d_src + static_cast<size_t>(iy0 + ly) * n1 + ix0;
            #pragma unroll
            for (int lx = tx; lx < IN_X; lx += BX) {
                smem[0][ly * SMEM_STRIDE + lx] = __ldg(row_ptr + lx);
            }
        }
    } else {
        for (int ly = ty; ly < IN_Y; ly += BY) {
            int gy = iy0 + ly;
            bool vy = (gy >= 0 && gy < n1);
            const float* row_ptr = vy ? (d_src + static_cast<size_t>(gy) * n1) : nullptr;
            for (int lx = tx; lx < IN_X; lx += BX) {
                int gx = ix0 + lx;
                smem[0][ly * SMEM_STRIDE + lx] = (vy && gx >= 0 && gx < n1) ? __ldg(row_ptr + gx) : 0.0f;
            }
        }
    }
    __syncthreads();

    int cur = 0;
    if (is_interior) {
        #pragma unroll 2
        for (int s = 0; s < K; ++s) {
            int nxt = cur ^ 1;
            int lo = s + 1;
            int hi_y = IN_Y - 1 - s;
            int hi_x = IN_X - 1 - s;
            const float* __restrict__ in_s = smem[cur];
            float* __restrict__ out_s = smem[nxt];

            for (int ly = lo + ty; ly < hi_y; ly += BY) {
                int base = ly * SMEM_STRIDE;
                #pragma unroll
                for (int lx = lo + tx; lx < hi_x; lx += BX) {
                    int idx = base + lx;
                    float val = 0.2f * (in_s[idx] + in_s[idx - 1] + in_s[idx + 1]
                                      + in_s[idx + SMEM_STRIDE] + in_s[idx - SMEM_STRIDE]);
                    out_s[idx] = val;
                    if constexpr (SAVE_PREV) {
                        if (s == K - 2 && ly >= K && ly < IN_Y - K && lx >= K && lx < IN_X - K) {
                            d_dst_prev[static_cast<size_t>(iy0 + ly) * n1 + (ix0 + lx)] = val;
                        }
                    }
                }
            }
            __syncthreads();
            cur = nxt;
        }
        const float* __restrict__ fin = smem[cur];
        for (int ly = K + ty; ly < IN_Y - K; ly += BY) {
            float* dst_row = d_dst + static_cast<size_t>(iy0 + ly) * n1 + ix0;
            #pragma unroll
            for (int lx = K + tx; lx < IN_X - K; lx += BX) {
                dst_row[lx] = fin[ly * SMEM_STRIDE + lx];
            }
        }
    } else {
        for (int s = 0; s < K; ++s) {
            int nxt = cur ^ 1;
            int lo = s + 1;
            int hi_y = IN_Y - 1 - s;
            int hi_x = IN_X - 1 - s;
            const float* __restrict__ in_s = smem[cur];
            float* __restrict__ out_s = smem[nxt];
            const float* bdy_src = ((s & 1) == 0) ? d_b1_orig : d_b0_orig;

            for (int ly = lo + ty; ly < hi_y; ly += BY) {
                int gy = iy0 + ly;
                int base = ly * SMEM_STRIDE;
                for (int lx = lo + tx; lx < hi_x; lx += BX) {
                    int gx = ix0 + lx;
                    float val = 0.0f;
                    if (gy >= 1 && gy < n1 - 1 && gx >= 1 && gx < n1 - 1) {
                        int idx = base + lx;
                        val = 0.2f * (in_s[idx] + in_s[idx - 1] + in_s[idx + 1]
                                    + in_s[idx + SMEM_STRIDE] + in_s[idx - SMEM_STRIDE]);
                    } else if (gy >= 0 && gy < n1 && gx >= 0 && gx < n1) {
                        val = __ldg(bdy_src + static_cast<size_t>(gy) * n1 + gx);
                    }
                    out_s[base + lx] = val;
                    if constexpr (SAVE_PREV) {
                        if (s == K - 2 && ly >= K && ly < IN_Y - K && lx >= K && lx < IN_X - K &&
                            gy >= 1 && gy < n1 - 1 && gx >= 1 && gx < n1 - 1) {
                            d_dst_prev[static_cast<size_t>(gy) * n1 + gx] = val;
                        }
                    }
                }
            }
            __syncthreads();
            cur = nxt;
        }
        const float* __restrict__ fin = smem[cur];
        for (int ly = K + ty; ly < IN_Y - K; ly += BY) {
            int gy = iy0 + ly;
            if (gy >= 1 && gy < n1 - 1) {
                for (int lx = K + tx; lx < IN_X - K; lx += BX) {
                    int gx = ix0 + lx;
                    if (gx >= 1 && gx < n1 - 1) {
                        d_dst[static_cast<size_t>(gy) * n1 + gx] = fin[ly * SMEM_STRIDE + lx];
                    }
                }
            }
        }
    }
}

// ============================================================================
// Common Spectral Utilities: Coalesced Transpose, 2D Spectral Evolve, Final Stencil
// ============================================================================

__global__ void __launch_bounds__(256)
transpose_df_kernel(int M, const df* __restrict__ src, df* __restrict__ dst) {
    __shared__ df tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        if (x < M && (y + k) < M) {
            tile[threadIdx.y + k][threadIdx.x] = ldg_df(src + static_cast<size_t>(y + k) * M + x);
        }
    }
    __syncthreads();

    int ox = blockIdx.y * 32 + threadIdx.x;
    int oy = blockIdx.x * 32 + threadIdx.y;
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        if (ox < M && (oy + k) < M) {
            st_df(dst + static_cast<size_t>(oy + k) * M + ox, tile[threadIdx.x][threadIdx.y + k]);
        }
    }
}

__global__ void __launch_bounds__(256)
spectral_evolve_2d_kernel(
    int n0, int M,
    df* __restrict__ d_work,
    const df* __restrict__ d_bdry_hat,
    const df* __restrict__ d_sin2_half,
    const df* __restrict__ d_sp1
) {
    int p = blockIdx.x * 32 + threadIdx.x;
    int q = blockIdx.y * 8 + threadIdx.y;
    if (p >= M || q >= M) return;

    const df* t0_h = d_bdry_hat + 0 * M;
    const df* b0_h = d_bdry_hat + 1 * M;
    const df* l0_h = d_bdry_hat + 2 * M;
    const df* r0_h = d_bdry_hat + 3 * M;
    const df* t1_h = d_bdry_hat + 4 * M;
    const df* b1_h = d_bdry_hat + 5 * M;
    const df* l1_h = d_bdry_hat + 6 * M;
    const df* r1_h = d_bdry_hat + 7 * M;

    df sin2_p = ldg_df(d_sin2_half + p), sin2_q = ldg_df(d_sin2_half + q);
    df sp1_p  = ldg_df(d_sp1 + p),       sp1_q  = ldg_df(d_sp1 + q);
    df t0_q = ldg_df(t0_h + q), b0_q = ldg_df(b0_h + q);
    df t1_q = ldg_df(t1_h + q), b1_q = ldg_df(b1_h + q);
    df l0_p = ldg_df(l0_h + p), r0_p = ldg_df(r0_h + p);
    df l1_p = ldg_df(l1_h + p), r1_p = ldg_df(r1_h + p);

    df tb0 = (p & 1) ? df_sub(t0_q, b0_q) : df_add(t0_q, b0_q);
    df tb1 = (p & 1) ? df_sub(t1_q, b1_q) : df_add(t1_q, b1_q);
    df lr0 = (q & 1) ? df_sub(l0_p, r0_p) : df_add(l0_p, r0_p);
    df lr1 = (q & 1) ? df_sub(l1_p, r1_p) : df_add(l1_p, r1_p);

    size_t idx = static_cast<size_t>(q) * M + p;
    df u0 = ldg_df(d_work + idx);
    df ev = spectral_evolve_mode(n0, u0, sin2_p, sin2_q, sp1_p, sp1_q, tb0, tb1, lr0, lr1);
    st_df(d_work + idx, ev);
}

__global__ void __launch_bounds__(256)
final_stencil_df_kernel(
    int M, int n1,
    const df* __restrict__ d_b1_int,
    const float* __restrict__ d_buf1,
    float* __restrict__ d_buf0
) {
    __shared__ df s_tile[18][34];
    const int bx = blockIdx.x * 32;
    const int by = blockIdx.y * 16;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    auto load_elem = [&](int iy, int ix) -> df {
        if (iy >= 0 && iy < M && ix >= 0 && ix < M) {
            return ldg_df(d_b1_int + static_cast<size_t>(iy) * M + ix);
        }
        int gy = iy + 1, gx = ix + 1;
        if (gy >= 0 && gy < n1 && gx >= 0 && gx < n1) {
            return {__ldg(d_buf1 + static_cast<size_t>(gy) * n1 + gx), 0.0f};
        }
        return {0.0f, 0.0f};
    };

    for (int ly = ty; ly < 18; ly += 8) {
        int iy = by + ly - 1;
        for (int lx = tx; lx < 34; lx += 32) {
            int ix = bx + lx - 1;
            s_tile[ly][lx] = load_elem(iy, ix);
        }
    }
    __syncthreads();

    const df INV5 = {0.20000000298023223877f, -2.9802322276673015e-09f};
    int j = bx + tx;
    if (j < M) {
        for (int ly = ty; ly < 16; ly += 8) {
            int i = by + ly;
            if (i < M) {
                df c = s_tile[ly + 1][tx + 1];
                df l = s_tile[ly + 1][tx];
                df r = s_tile[ly + 1][tx + 2];
                df u = s_tile[ly][tx + 1];
                df d = s_tile[ly + 2][tx + 1];
                df sum = df_add(df_add(df_add(c, l), df_add(r, d)), u);
                df val = df_mul(sum, INV5);
                d_buf0[static_cast<size_t>(i + 1) * n1 + (j + 1)] = __fadd_rn(val.hi, val.lo);
            }
        }
    }
}

// ============================================================================
// Engine 4: Unified Single-Block SMEM Bluestein 1D DST-I
// pass_kind: 0 = row fwd (f32 + 4 boundary blocks -> df), 1 = df -> df, 2 = row inv (df -> df + f32)
// Interleaved 128-bit SMEM (LDS.128/STS.128, 0 bank conflicts) + SpecArgs in const memory -> 0 spills!
// ============================================================================

#define S2IDX_C(i) ((i) + ((i) >> 3))

enum F1Mode {
    F1_13 = 0,
    F1_5 = 1,
    F1_ALL = 2
};

struct SpecArgs {
    int pass_kind, P, F1, F2, M_R8, M, n1;
    int P_div_F1, tw_off_f2, r8_off_start, r8_off_end, n_row_blocks, fuse_r2;
    int compact_smem;
    const float* d_buf0;
    float* d_buf1;
    const df* d_in_df;
    df* d_out_df;
    df* d_bdry_hat;
    const df* d_sin_k;
    const cdf* d_chirp;
    const cdf* d_tw_stages;
    const cdf* d_bhat_t;
    const int* d_inv_perm;
    df scale_dst;
};

template <int BLOCK_T, int MIN_BLOCKS, int F1_MODE>
__global__ void __launch_bounds__(BLOCK_T, MIN_BLOCKS)
bluestein_1d_kernel(SpecArgs a) {
    extern __shared__ cdf smem_c[];
    auto ld = [&](int idx) -> cdf { return smem_c[S2IDX_C(idx)]; };
    auto st = [&](int idx, cdf v) { smem_c[S2IDX_C(idx)] = v; };

    // 1. Load & Pre-chirp (with fused zero-pruned R2 when F1==1 && F2==2)
    {
        const int L = a.M + 1;
        const int tx = threadIdx.x;
        const bool is_bdry = (a.pass_kind == 0 && blockIdx.x >= a.n_row_blocks);
        const int row0 = blockIdx.x * 2;
        const int row1 = row0 + 1;
        const bool has1 = is_bdry ? true : (row1 < a.M);

        const float* pf0 = nullptr;
        const float* pf1 = nullptr;
        int step = 1;
        const df* pd0 = nullptr;
        const df* pd1 = nullptr;

        if (a.pass_kind == 0) {
            if (!is_bdry) {
                pf0 = a.d_buf0 + static_cast<size_t>(row0 + 1) * a.n1 + 1;
                pf1 = has1 ? (a.d_buf0 + static_cast<size_t>(row1 + 1) * a.n1 + 1) : nullptr;
            } else {
                int bpair = blockIdx.x - a.n_row_blocks;
                const float* buf = (bpair < 2) ? a.d_buf0 : a.d_buf1;
                if ((bpair & 1) == 0) {
                    pf0 = buf + 1;
                    pf1 = buf + static_cast<size_t>(a.n1 - 1) * a.n1 + 1;
                } else {
                    pf0 = buf + static_cast<size_t>(a.n1);
                    pf1 = buf + static_cast<size_t>(a.n1) + (a.n1 - 1);
                    step = a.n1;
                }
            }
        } else {
            pd0 = a.d_in_df + static_cast<size_t>(row0) * a.M;
            pd1 = has1 ? (a.d_in_df + static_cast<size_t>(row1) * a.M) : nullptr;
        }

        const bool use_fuse_r2 = (F1_MODE != F1_5) && (a.fuse_r2 != 0);
        const int lim = use_fuse_r2 ? (a.P >> 1) : a.P;
        for (int k = tx; k < lim; k += BLOCK_T) {
            cdf val = {{0.0f, 0.0f}, {0.0f, 0.0f}};
            if (k >= 1 && k < L) {
                df s = ldg_df(a.d_sin_k + k);
                df y0, y1 = {0.0f, 0.0f};
                if (a.pass_kind == 0) {
                    float x0a = __ldg(pf0 + static_cast<size_t>(k - 1) * step);
                    float x0b = __ldg(pf0 + static_cast<size_t>(L - k - 1) * step);
                    y0 = df_add(df_mul(s, df_add_u({x0a, 0.0f}, {x0b, 0.0f})),
                                df_mul_f(df_sub_u({x0a, 0.0f}, {x0b, 0.0f}), 0.5f));
                    if (has1) {
                        float x1a = __ldg(pf1 + static_cast<size_t>(k - 1) * step);
                        float x1b = __ldg(pf1 + static_cast<size_t>(L - k - 1) * step);
                        y1 = df_add(df_mul(s, df_add_u({x1a, 0.0f}, {x1b, 0.0f})),
                                    df_mul_f(df_sub_u({x1a, 0.0f}, {x1b, 0.0f}), 0.5f));
                    }
                } else {
                    df x0a = ldg_df(pd0 + k - 1), x0b = ldg_df(pd0 + L - k - 1);
                    y0 = df_add(df_mul(s, df_add_u(x0a, x0b)), df_mul_f(df_sub_u(x0a, x0b), 0.5f));
                    if (has1) {
                        df x1a = ldg_df(pd1 + k - 1), x1b = ldg_df(pd1 + L - k - 1);
                        y1 = df_add(df_mul(s, df_add_u(x1a, x1b)), df_mul_f(df_sub_u(x1a, x1b), 0.5f));
                    }
                }
                val = cdf_mul({y0, y1}, ldg_cdf(a.d_chirp + k));
            }
            st(k, val);
            if constexpr (F1_MODE != F1_5) {
                if (a.fuse_r2) {
                    st(k + lim, (k >= 1 && k < L) ? cdf_mul(val, ldg_cdf(a.d_tw_stages + k)) : val);
                }
            }
        }
        __syncthreads();
    }

    // 2. Forward DIF F1/F2 stages
    if (F1_MODE == F1_5 || !a.fuse_r2) {
        if constexpr (F1_MODE != F1_5) {
            if (a.F1 == 3) {
                for (int j = threadIdx.x; j < a.P_div_F1; j += BLOCK_T) {
                    cdf x0 = ld(j), x1 = ld(j + a.P_div_F1), x2 = ld(j + 2 * a.P_div_F1);
                    radix3_bfly<false>(x0, x1, x2);
                    st(j, x0);
                    st(j + a.P_div_F1,     cdf_mul(x1, ldg_cdf(a.d_tw_stages + j)));
                    st(j + 2 * a.P_div_F1, cdf_mul(x2, ldg_cdf(a.d_tw_stages + a.P_div_F1 + j)));
                }
                __syncthreads();
            }
        }
        if constexpr (F1_MODE != F1_13) {
            if (a.F1 == 5) {
                for (int j = threadIdx.x; j < a.P_div_F1; j += BLOCK_T) {
                    cdf x0 = ld(j), x1 = ld(j + a.P_div_F1), x2 = ld(j + 2 * a.P_div_F1);
                    cdf x3 = ld(j + 3 * a.P_div_F1), x4 = ld(j + 4 * a.P_div_F1);
                    radix5_bfly<false>(x0, x1, x2, x3, x4);
                    st(j, x0);
                    st(j + a.P_div_F1,     cdf_mul(x1, ldg_cdf(a.d_tw_stages + 0 * a.P_div_F1 + j)));
                    st(j + 2 * a.P_div_F1, cdf_mul(x2, ldg_cdf(a.d_tw_stages + 1 * a.P_div_F1 + j)));
                    st(j + 3 * a.P_div_F1, cdf_mul(x3, ldg_cdf(a.d_tw_stages + 2 * a.P_div_F1 + j)));
                    st(j + 4 * a.P_div_F1, cdf_mul(x4, ldg_cdf(a.d_tw_stages + 3 * a.P_div_F1 + j)));
                }
                __syncthreads();
            }
        }

        if (a.F2 == 2) {
            const int sh = __ffs(a.M_R8) - 1, mask = a.M_R8 - 1;
            const int n_bfly = a.P >> 1;
            for (int b = threadIdx.x; b < n_bfly; b += BLOCK_T) {
                int blk = b >> sh, j = b & mask, base = blk * a.P_div_F1 + j;
                cdf x0 = ld(base), x1 = ld(base + a.M_R8);
                st(base, cdf_add(x0, x1));
                st(base + a.M_R8, cdf_mul(cdf_sub_u(x0, x1), ldg_cdf(a.d_tw_stages + a.tw_off_f2 + j)));
            }
            __syncthreads();
        } else if (a.F2 == 4) {
            const int sh = __ffs(a.M_R8) - 1, mask = a.M_R8 - 1;
            const int n_bfly = a.P >> 2;
            for (int b = threadIdx.x; b < n_bfly; b += BLOCK_T) {
                int blk = b >> sh, j = b & mask, base = blk * a.P_div_F1 + j;
                cdf x0 = ld(base), x1 = ld(base + a.M_R8);
                cdf x2 = ld(base + 2 * a.M_R8), x3 = ld(base + 3 * a.M_R8);
                radix4_bfly<false, false>(x0, x1, x2, x3);
                st(base, {df_renorm(x0.r), df_renorm(x0.i)});
                st(base + a.M_R8,     cdf_mul(x1, ldg_cdf(a.d_tw_stages + a.tw_off_f2 + 0 * a.M_R8 + j)));
                st(base + 2 * a.M_R8, cdf_mul(x2, ldg_cdf(a.d_tw_stages + a.tw_off_f2 + 1 * a.M_R8 + j)));
                st(base + 3 * a.M_R8, cdf_mul(x3, ldg_cdf(a.d_tw_stages + a.tw_off_f2 + 2 * a.M_R8 + j)));
            }
            __syncthreads();
        }
    }

    // Forward DIF radix-8 stages
    {
        int off = a.r8_off_start;
        #pragma unroll 1
        for (int m = a.M_R8; m > 8; m >>= 3) {
            int stride = m >> 3;
            int sh = __ffs(stride) - 1, mask = stride - 1;
            const cdf* tw = a.d_tw_stages + off;
            for (int b = threadIdx.x; b < (a.P >> 3); b += BLOCK_T) {
                int blk = b >> sh, j = b & mask, base = (blk << (sh + 3)) + j;
                cdf x0 = ld(base + 0 * stride), x1 = ld(base + 1 * stride);
                cdf x2 = ld(base + 2 * stride), x3 = ld(base + 3 * stride);
                cdf x4 = ld(base + 4 * stride), x5 = ld(base + 5 * stride);
                cdf x6 = ld(base + 6 * stride), x7 = ld(base + 7 * stride);
                radix8_bfly<false, false>(x0, x1, x2, x3, x4, x5, x6, x7);
                st(base + 0 * stride, {df_renorm(x0.r), df_renorm(x0.i)});
                st(base + 1 * stride, cdf_mul(x1, ldg_cdf(tw + 0 * stride + j)));
                st(base + 2 * stride, cdf_mul(x2, ldg_cdf(tw + 1 * stride + j)));
                st(base + 3 * stride, cdf_mul(x3, ldg_cdf(tw + 2 * stride + j)));
                st(base + 4 * stride, cdf_mul(x4, ldg_cdf(tw + 3 * stride + j)));
                st(base + 5 * stride, cdf_mul(x5, ldg_cdf(tw + 4 * stride + j)));
                st(base + 6 * stride, cdf_mul(x6, ldg_cdf(tw + 5 * stride + j)));
                st(base + 7 * stride, cdf_mul(x7, ldg_cdf(tw + 6 * stride + j)));
            }
            __syncthreads();
            off += 7 * stride;
        }
    }

    // 3. Innermost fused radix-8: FWD + MUL(d_bhat_t) + INV (contiguous p[0..7]!)
    {
        const int n_in = a.P >> 3;
        for (int b = threadIdx.x; b < n_in; b += BLOCK_T) {
            cdf* p = smem_c + b * 9;
            cdf x0 = p[0], x1 = p[1], x2 = p[2], x3 = p[3];
            cdf x4 = p[4], x5 = p[5], x6 = p[6], x7 = p[7];
            radix8_bfly<false, false>(x0, x1, x2, x3, x4, x5, x6, x7);
            x0 = cdf_mul(x0, ldg_cdf(a.d_bhat_t + 0 * n_in + b));
            x1 = cdf_mul(x1, ldg_cdf(a.d_bhat_t + 1 * n_in + b));
            x2 = cdf_mul(x2, ldg_cdf(a.d_bhat_t + 2 * n_in + b));
            x3 = cdf_mul(x3, ldg_cdf(a.d_bhat_t + 3 * n_in + b));
            x4 = cdf_mul(x4, ldg_cdf(a.d_bhat_t + 4 * n_in + b));
            x5 = cdf_mul(x5, ldg_cdf(a.d_bhat_t + 5 * n_in + b));
            x6 = cdf_mul(x6, ldg_cdf(a.d_bhat_t + 6 * n_in + b));
            x7 = cdf_mul(x7, ldg_cdf(a.d_bhat_t + 7 * n_in + b));
            radix8_bfly<true, true>(x0, x1, x2, x3, x4, x5, x6, x7);
            p[0] = x0; p[1] = x1; p[2] = x2; p[3] = x3;
            p[4] = x4; p[5] = x5; p[6] = x6; p[7] = x7;
        }
        __syncthreads();
    }

    // 4. Inverse DIT radix-8 stages
    {
        int off = a.r8_off_end;
        #pragma unroll 1
        for (int m = 64; m <= a.M_R8; m <<= 3) {
            int stride = m >> 3;
            off -= 7 * stride;
            int sh = __ffs(stride) - 1, mask = stride - 1;
            const cdf* tw = a.d_tw_stages + off;
            for (int b = threadIdx.x; b < (a.P >> 3); b += BLOCK_T) {
                int blk = b >> sh, j = b & mask, base = (blk << (sh + 3)) + j;
                cdf x0 = ld(base + 0 * stride);
                cdf x1 = cdf_mul(ld(base + 1 * stride), cdf_conj(ldg_cdf(tw + 0 * stride + j)));
                cdf x2 = cdf_mul(ld(base + 2 * stride), cdf_conj(ldg_cdf(tw + 1 * stride + j)));
                cdf x3 = cdf_mul(ld(base + 3 * stride), cdf_conj(ldg_cdf(tw + 2 * stride + j)));
                cdf x4 = cdf_mul(ld(base + 4 * stride), cdf_conj(ldg_cdf(tw + 3 * stride + j)));
                cdf x5 = cdf_mul(ld(base + 5 * stride), cdf_conj(ldg_cdf(tw + 4 * stride + j)));
                cdf x6 = cdf_mul(ld(base + 6 * stride), cdf_conj(ldg_cdf(tw + 5 * stride + j)));
                cdf x7 = cdf_mul(ld(base + 7 * stride), cdf_conj(ldg_cdf(tw + 6 * stride + j)));
                radix8_bfly<true, true>(x0, x1, x2, x3, x4, x5, x6, x7);
                st(base + 0 * stride, x0); st(base + 1 * stride, x1);
                st(base + 2 * stride, x2); st(base + 3 * stride, x3);
                st(base + 4 * stride, x4); st(base + 5 * stride, x5);
                st(base + 6 * stride, x6); st(base + 7 * stride, x7);
            }
            __syncthreads();
        }
    }

    // 5. Inverse DIT F2/F1 + post-chirp
    {
        const int L = a.M + 1;
        if (F1_MODE != F1_5 && a.fuse_r2) {
            const int halfP = a.P >> 1;
            for (int m = threadIdx.x; m < L; m += BLOCK_T) {
                cdf x0 = ld(m);
                cdf x1 = cdf_mul(ld(m + halfP), cdf_conj(ldg_cdf(a.d_tw_stages + m)));
                st(m, cdf_mul(cdf_add_u(x0, x1), ldg_cdf(a.d_chirp + m)));
            }
            __syncthreads();
        } else {
            if (a.F2 == 2) {
                const int sh = __ffs(a.M_R8) - 1, mask = a.M_R8 - 1;
                const int n_bfly = a.P >> 1;
                for (int b = threadIdx.x; b < n_bfly; b += BLOCK_T) {
                    int blk = b >> sh, j = b & mask, base = blk * a.P_div_F1 + j;
                    cdf x0 = ld(base);
                    cdf x1 = cdf_mul(ld(base + a.M_R8), cdf_conj(ldg_cdf(a.d_tw_stages + a.tw_off_f2 + j)));
                    st(base, cdf_add(x0, x1));
                    st(base + a.M_R8, cdf_sub(x0, x1));
                }
                __syncthreads();
            } else if (a.F2 == 4) {
                const int sh = __ffs(a.M_R8) - 1, mask = a.M_R8 - 1;
                const int n_bfly = a.P >> 2;
                for (int b = threadIdx.x; b < n_bfly; b += BLOCK_T) {
                    int blk = b >> sh, j = b & mask, base = blk * a.P_div_F1 + j;
                    cdf x0 = ld(base);
                    cdf x1 = cdf_mul(ld(base + a.M_R8),     cdf_conj(ldg_cdf(a.d_tw_stages + a.tw_off_f2 + 0 * a.M_R8 + j)));
                    cdf x2 = cdf_mul(ld(base + 2 * a.M_R8), cdf_conj(ldg_cdf(a.d_tw_stages + a.tw_off_f2 + 1 * a.M_R8 + j)));
                    cdf x3 = cdf_mul(ld(base + 3 * a.M_R8), cdf_conj(ldg_cdf(a.d_tw_stages + a.tw_off_f2 + 2 * a.M_R8 + j)));
                    radix4_bfly<true, true>(x0, x1, x2, x3);
                    st(base, x0); st(base + a.M_R8, x1); st(base + 2 * a.M_R8, x2); st(base + 3 * a.M_R8, x3);
                }
                __syncthreads();
            }
            if constexpr (F1_MODE != F1_5) {
                if (a.F1 == 3) {
                    for (int j = threadIdx.x; j < a.P_div_F1; j += BLOCK_T) {
                        cdf x0 = ld(j);
                        cdf x1 = cdf_mul(ld(j + a.P_div_F1),     cdf_conj(ldg_cdf(a.d_tw_stages + j)));
                        cdf x2 = cdf_mul(ld(j + 2 * a.P_div_F1), cdf_conj(ldg_cdf(a.d_tw_stages + a.P_div_F1 + j)));
                        radix3_bfly<true>(x0, x1, x2);
                        st(j, x0);
                        if (j + a.P_div_F1 < L) st(j + a.P_div_F1, x1);
                    }
                    __syncthreads();
                }
            }
            if constexpr (F1_MODE != F1_13) {
                if (a.F1 == 5) {
                    for (int j = threadIdx.x; j < a.P_div_F1; j += BLOCK_T) {
                        cdf x0 = ld(j);
                        cdf x1 = cdf_mul(ld(j + a.P_div_F1),     cdf_conj(ldg_cdf(a.d_tw_stages + 0 * a.P_div_F1 + j)));
                        cdf x2 = cdf_mul(ld(j + 2 * a.P_div_F1), cdf_conj(ldg_cdf(a.d_tw_stages + 1 * a.P_div_F1 + j)));
                        cdf x3 = cdf_mul(ld(j + 3 * a.P_div_F1), cdf_conj(ldg_cdf(a.d_tw_stages + 2 * a.P_div_F1 + j)));
                        cdf x4 = cdf_mul(ld(j + 4 * a.P_div_F1), cdf_conj(ldg_cdf(a.d_tw_stages + 3 * a.P_div_F1 + j)));
                        radix5_bfly<true>(x0, x1, x2, x3, x4);
                        st(j, x0);
                        if (j + a.P_div_F1 < L) st(j + a.P_div_F1, x1);
                        if (j + 2 * a.P_div_F1 < L) st(j + 2 * a.P_div_F1, x2);
                    }
                    __syncthreads();
                }
            }
            for (int m = threadIdx.x; m < L; m += BLOCK_T) {
                st(m, cdf_mul(ld(m), ldg_cdf(a.d_chirp + m)));
            }
            __syncthreads();
        }
    }

    // 6. Extract DST-I + parallel prefix sum + coalesced stores
    {
        const int L = a.M + 1;
        const int tx = threadIdx.x;
        const bool is_bdry = (a.pass_kind == 0 && blockIdx.x >= a.n_row_blocks);
        const int row0 = blockIdx.x * 2;
        const int row1 = row0 + 1;
        const bool has1 = is_bdry ? true : (row1 < a.M);
        const int n_odd = (a.M + 1) >> 1;

        df* free_df = reinterpret_cast<df*>(smem_c + S2IDX_C(L) + 2);
        df* r1_buf = free_df;
        df* i1_buf = r1_buf + n_odd;
        df* r2_buf = i1_buf + n_odd;
        df* i2_buf = r2_buf + n_odd;

        for (int m = tx; m <= (L >> 1); m += BLOCK_T) {
            if (m == 0) {
                cdf z0 = smem_c[0];
                r1_buf[0] = df_mul_f(z0.r, 0.5f);
                r2_buf[0] = df_mul_f(z0.i, 0.5f);
            } else {
                cdf zm = ld(m), zc = ld(L - m);
                df R1 = df_mul_f(df_add_u(zm.r, zc.r), 0.5f);
                df I1 = df_mul_f(df_sub_u(zm.i, zc.i), 0.5f);
                df R2 = df_mul_f(df_add_u(zm.i, zc.i), 0.5f);
                df I2 = df_mul_f(df_sub_u(zc.r, zm.r), 0.5f);
                if (2 * m <= a.M) {
                    i1_buf[m - 1] = df_neg(I1);
                    if (has1) i2_buf[m - 1] = df_neg(I2);
                }
                if (m < n_odd) {
                    r1_buf[m] = R1;
                    if (has1) r2_buf[m] = R2;
                }
            }
        }
        __syncthreads();

        df_prefix_sum_pair_smem<BLOCK_T>(r1_buf, r2_buf, n_odd, has1);

        df* out0 = is_bdry ? (a.d_bdry_hat + static_cast<size_t>((blockIdx.x - a.n_row_blocks) * 2) * a.M)
                           : (a.d_out_df + static_cast<size_t>(row0) * a.M);
        df* out1 = is_bdry ? (a.d_bdry_hat + static_cast<size_t>((blockIdx.x - a.n_row_blocks) * 2 + 1) * a.M)
                           : (has1 ? (a.d_out_df + static_cast<size_t>(row1) * a.M) : nullptr);
        float* fout0 = (a.pass_kind == 2) ? (a.d_buf1 + static_cast<size_t>(row0 + 1) * a.n1 + 1) : nullptr;
        float* fout1 = (a.pass_kind == 2 && has1) ? (a.d_buf1 + static_cast<size_t>(row1 + 1) * a.n1 + 1) : nullptr;

        for (int p = tx; p < a.M; p += BLOCK_T) {
            int h = p >> 1;
            df v0 = df_mul((p & 1) ? i1_buf[h] : r1_buf[h], a.scale_dst);
            st_df(out0 + p, v0);
            if (a.pass_kind == 2) fout0[p] = __fadd_rn(v0.hi, v0.lo);
            if (has1) {
                df v1 = df_mul((p & 1) ? i2_buf[h] : r2_buf[h], a.scale_dst);
                st_df(out1 + p, v1);
                if (a.pass_kind == 2) fout1[p] = __fadd_rn(v1.hi, v1.lo);
            }
        }
    }
}

// ============================================================================
// Engine 3: Unified Single-Block SMEM Smooth-L 1D DST-I
// Covers ALL smooth L <= 8192 (including n1 = 4097, 6145, 8193) with 0 spills!
// ============================================================================

template <int BLOCK_T, int MIN_BLOCKS, int F1_MODE>
__global__ void __launch_bounds__(BLOCK_T, MIN_BLOCKS)
smooth_1d_kernel(SpecArgs a) {
    extern __shared__ cdf smem_c[];
    auto ld = [&](int idx) -> cdf { return smem_c[S2IDX_C(idx)]; };
    auto st = [&](int idx, cdf v) { smem_c[S2IDX_C(idx)] = v; };

    // 1. Load & DST-I pre-processing
    {
        const int L = a.P;
        const int tx = threadIdx.x;
        const bool is_bdry = (a.pass_kind == 0 && blockIdx.x >= a.n_row_blocks);
        const int row0 = blockIdx.x * 2;
        const int row1 = row0 + 1;
        const bool has1 = is_bdry ? true : (row1 < a.M);

        const float* pf0 = nullptr;
        const float* pf1 = nullptr;
        int step = 1;
        const df* pd0 = nullptr;
        const df* pd1 = nullptr;

        if (a.pass_kind == 0) {
            if (!is_bdry) {
                pf0 = a.d_buf0 + static_cast<size_t>(row0 + 1) * a.n1 + 1;
                pf1 = has1 ? (a.d_buf0 + static_cast<size_t>(row1 + 1) * a.n1 + 1) : nullptr;
            } else {
                int bpair = blockIdx.x - a.n_row_blocks;
                const float* buf = (bpair < 2) ? a.d_buf0 : a.d_buf1;
                if ((bpair & 1) == 0) {
                    pf0 = buf + 1;
                    pf1 = buf + static_cast<size_t>(a.n1 - 1) * a.n1 + 1;
                } else {
                    pf0 = buf + static_cast<size_t>(a.n1);
                    pf1 = buf + static_cast<size_t>(a.n1) + (a.n1 - 1);
                    step = a.n1;
                }
            }
        } else {
            pd0 = a.d_in_df + static_cast<size_t>(row0) * a.M;
            pd1 = has1 ? (a.d_in_df + static_cast<size_t>(row1) * a.M) : nullptr;
        }

        for (int k = tx; k < L; k += BLOCK_T) {
            df y0 = {0.0f, 0.0f}, y1 = {0.0f, 0.0f};
            if (k >= 1) {
                df s = ldg_df(a.d_sin_k + k);
                if (a.pass_kind == 0) {
                    float x0a = __ldg(pf0 + static_cast<size_t>(k - 1) * step);
                    float x0b = __ldg(pf0 + static_cast<size_t>(L - k - 1) * step);
                    y0 = df_add(df_mul(s, df_add_u({x0a, 0.0f}, {x0b, 0.0f})),
                                df_mul_f(df_sub_u({x0a, 0.0f}, {x0b, 0.0f}), 0.5f));
                    if (has1) {
                        float x1a = __ldg(pf1 + static_cast<size_t>(k - 1) * step);
                        float x1b = __ldg(pf1 + static_cast<size_t>(L - k - 1) * step);
                        y1 = df_add(df_mul(s, df_add_u({x1a, 0.0f}, {x1b, 0.0f})),
                                    df_mul_f(df_sub_u({x1a, 0.0f}, {x1b, 0.0f}), 0.5f));
                    }
                } else {
                    df x0a = ldg_df(pd0 + k - 1), x0b = ldg_df(pd0 + L - k - 1);
                    y0 = df_add(df_mul(s, df_add_u(x0a, x0b)), df_mul_f(df_sub_u(x0a, x0b), 0.5f));
                    if (has1) {
                        df x1a = ldg_df(pd1 + k - 1), x1b = ldg_df(pd1 + L - k - 1);
                        y1 = df_add(df_mul(s, df_add_u(x1a, x1b)), df_mul_f(df_sub_u(x1a, x1b), 0.5f));
                    }
                }
            }
            st(k, {y0, y1});
        }
        __syncthreads();
    }

    // 2. Forward DIF FFT of length L
    if constexpr (F1_MODE != F1_5) {
        if (a.F1 == 3) {
            const int sc = a.P_div_F1 + (a.P_div_F1 >> 3);
            for (int j = threadIdx.x; j < a.P_div_F1; j += BLOCK_T) {
                cdf* p = smem_c + S2IDX_C(j);
                cdf x0 = p[0], x1 = p[sc], x2 = p[2 * sc];
                radix3_bfly<false>(x0, x1, x2);
                p[0] = x0;
                p[sc]     = cdf_mul(x1, ldg_cdf(a.d_tw_stages + j));
                p[2 * sc] = cdf_mul(x2, ldg_cdf(a.d_tw_stages + a.P_div_F1 + j));
            }
            __syncthreads();
        }
    }
    if constexpr (F1_MODE != F1_13) {
        if (a.F1 == 5) {
            const int sc = a.P_div_F1 + (a.P_div_F1 >> 3);
            for (int j = threadIdx.x; j < a.P_div_F1; j += BLOCK_T) {
                cdf* p = smem_c + S2IDX_C(j);
                cdf x0 = p[0], x1 = p[sc], x2 = p[2 * sc], x3 = p[3 * sc], x4 = p[4 * sc];
                radix5_bfly<false>(x0, x1, x2, x3, x4);
                p[0] = x0;
                p[sc]     = cdf_mul(x1, ldg_cdf(a.d_tw_stages + 0 * a.P_div_F1 + j));
                p[2 * sc] = cdf_mul(x2, ldg_cdf(a.d_tw_stages + 1 * a.P_div_F1 + j));
                p[3 * sc] = cdf_mul(x3, ldg_cdf(a.d_tw_stages + 2 * a.P_div_F1 + j));
                p[4 * sc] = cdf_mul(x4, ldg_cdf(a.d_tw_stages + 3 * a.P_div_F1 + j));
            }
            __syncthreads();
        }
    }

    if (a.F2 == 2) {
        const int sh = __ffs(a.M_R8) - 1, mask = a.M_R8 - 1;
        const int sc = a.M_R8 + (a.M_R8 >> 3);
        const int n_bfly = a.P >> 1;
        for (int b = threadIdx.x; b < n_bfly; b += BLOCK_T) {
            int blk = b >> sh, j = b & mask, base = blk * a.P_div_F1 + j;
            cdf* p = smem_c + S2IDX_C(base);
            cdf x0 = p[0], x1 = p[sc];
            p[0] = cdf_add(x0, x1);
            p[sc] = cdf_mul(cdf_sub_u(x0, x1), ldg_cdf(a.d_tw_stages + a.tw_off_f2 + j));
        }
        __syncthreads();
    } else if (a.F2 == 4) {
        const int sh = __ffs(a.M_R8) - 1, mask = a.M_R8 - 1;
        const int sc = a.M_R8 + (a.M_R8 >> 3);
        const int n_bfly = a.P >> 2;
        for (int b = threadIdx.x; b < n_bfly; b += BLOCK_T) {
            int blk = b >> sh, j = b & mask, base = blk * a.P_div_F1 + j;
            cdf* p = smem_c + S2IDX_C(base);
            cdf x0 = p[0], x1 = p[sc], x2 = p[2 * sc], x3 = p[3 * sc];
            radix4_bfly<false, false>(x0, x1, x2, x3);
            p[0] = {df_renorm(x0.r), df_renorm(x0.i)};
            p[sc]     = cdf_mul(x1, ldg_cdf(a.d_tw_stages + a.tw_off_f2 + 0 * a.M_R8 + j));
            p[2 * sc] = cdf_mul(x2, ldg_cdf(a.d_tw_stages + a.tw_off_f2 + 1 * a.M_R8 + j));
            p[3 * sc] = cdf_mul(x3, ldg_cdf(a.d_tw_stages + a.tw_off_f2 + 2 * a.M_R8 + j));
        }
        __syncthreads();
    }

    {
        const int n_bfly = a.P >> 3;
        int off = a.r8_off_start;
        #pragma unroll 1
        for (int m = a.M_R8; m > 8; m >>= 3) {
            int stride = m >> 3;
            int sc = stride + (stride >> 3);
            int sh = __ffs(stride) - 1, mask = stride - 1;
            const cdf* tw = a.d_tw_stages + off;
            for (int b = threadIdx.x; b < n_bfly; b += BLOCK_T) {
                int blk = b >> sh, j = b & mask, base = (blk << (sh + 3)) + j;
                cdf* p = smem_c + S2IDX_C(base);
                cdf x0 = p[0],      x1 = p[1 * sc], x2 = p[2 * sc], x3 = p[3 * sc];
                cdf x4 = p[4 * sc], x5 = p[5 * sc], x6 = p[6 * sc], x7 = p[7 * sc];
                radix8_bfly<false, false>(x0, x1, x2, x3, x4, x5, x6, x7);
                p[0]      = {df_renorm(x0.r), df_renorm(x0.i)};
                p[1 * sc] = cdf_mul(x1, ldg_cdf(tw + 0 * stride + j));
                p[2 * sc] = cdf_mul(x2, ldg_cdf(tw + 1 * stride + j));
                p[3 * sc] = cdf_mul(x3, ldg_cdf(tw + 2 * stride + j));
                p[4 * sc] = cdf_mul(x4, ldg_cdf(tw + 3 * stride + j));
                p[5 * sc] = cdf_mul(x5, ldg_cdf(tw + 4 * stride + j));
                p[6 * sc] = cdf_mul(x6, ldg_cdf(tw + 5 * stride + j));
                p[7 * sc] = cdf_mul(x7, ldg_cdf(tw + 6 * stride + j));
            }
            __syncthreads();
            off += 7 * stride;
        }
        for (int b = threadIdx.x; b < n_bfly; b += BLOCK_T) {
            cdf* p = smem_c + b * 9;
            cdf x0 = p[0], x1 = p[1], x2 = p[2], x3 = p[3];
            cdf x4 = p[4], x5 = p[5], x6 = p[6], x7 = p[7];
            radix8_bfly<false, true>(x0, x1, x2, x3, x4, x5, x6, x7);
            p[0] = x0; p[1] = x1; p[2] = x2; p[3] = x3;
            p[4] = x4; p[5] = x5; p[6] = x6; p[7] = x7;
        }
        __syncthreads();
    }

    // 3a. Extract DST-I (scoped so out0/out1/fout0/fout1 are NEVER live across prefix sum call!)
    {
        const int L = a.P;
        const int M = a.M;
        const int tx = threadIdx.x;
        const bool is_bdry = (a.pass_kind == 0 && blockIdx.x >= a.n_row_blocks);
        const int row0 = blockIdx.x * 2;
        const int row1 = row0 + 1;
        const bool has1 = is_bdry ? true : (row1 < M);
        const int n_odd = (M + 1) >> 1;
        const int LS_C = L + (L >> 3) + 4;

        df* extra_df = reinterpret_cast<df*>(smem_c + LS_C);
        df* r1_buf = extra_df;
        df* i1_buf = a.compact_smem ? nullptr : (r1_buf + n_odd);
        df* r2_buf = a.compact_smem ? (r1_buf + n_odd) : (r1_buf + M);
        df* i2_buf = a.compact_smem ? nullptr : (r2_buf + n_odd);

        df* out0 = a.compact_smem ? (is_bdry ? (a.d_bdry_hat + static_cast<size_t>((blockIdx.x - a.n_row_blocks) * 2) * M)
                                             : (a.d_out_df + static_cast<size_t>(row0) * M)) : nullptr;
        df* out1 = a.compact_smem ? (is_bdry ? (a.d_bdry_hat + static_cast<size_t>((blockIdx.x - a.n_row_blocks) * 2 + 1) * M)
                                             : (has1 ? (a.d_out_df + static_cast<size_t>(row1) * M) : nullptr)) : nullptr;
        float* fout0 = (a.compact_smem && a.pass_kind == 2) ? (a.d_buf1 + static_cast<size_t>(row0 + 1) * a.n1 + 1) : nullptr;
        float* fout1 = (a.compact_smem && a.pass_kind == 2 && has1) ? (a.d_buf1 + static_cast<size_t>(row1 + 1) * a.n1 + 1) : nullptr;

        for (int m = tx; m <= (L >> 1); m += BLOCK_T) {
            if (m == 0) {
                cdf z0 = smem_c[0];
                r1_buf[0] = df_mul_f(z0.r, 0.5f);
                r2_buf[0] = df_mul_f(z0.i, 0.5f);
            } else {
                int pm = __ldg(a.d_inv_perm + m);
                int pc = __ldg(a.d_inv_perm + L - m);
                cdf zm = ld(pm), zc = ld(pc);
                df R1 = df_mul_f(df_add_u(zm.r, zc.r), 0.5f);
                df I1 = df_mul_f(df_sub_u(zm.i, zc.i), 0.5f);
                df R2 = df_mul_f(df_add_u(zm.i, zc.i), 0.5f);
                df I2 = df_mul_f(df_sub_u(zc.r, zm.r), 0.5f);
                if (2 * m <= M) {
                    if (!a.compact_smem) {
                        i1_buf[m - 1] = df_neg(I1);
                        if (has1) i2_buf[m - 1] = df_neg(I2);
                    } else {
                        df v0 = df_mul(df_neg(I1), a.scale_dst);
                        st_df(out0 + (2 * m - 1), v0);
                        if (a.pass_kind == 2) fout0[2 * m - 1] = __fadd_rn(v0.hi, v0.lo);
                        if (has1) {
                            df v1 = df_mul(df_neg(I2), a.scale_dst);
                            st_df(out1 + (2 * m - 1), v1);
                            if (a.pass_kind == 2) fout1[2 * m - 1] = __fadd_rn(v1.hi, v1.lo);
                        }
                    }
                }
                if (m < n_odd) {
                    r1_buf[m] = R1;
                    if (has1) r2_buf[m] = R2;
                }
            }
        }
        __syncthreads();
    }

    // 3b. Prefix sum (minimal live registers across __noinline__ call -> 0 spills!)
    {
        const int M = a.M;
        const int n_odd = (M + 1) >> 1;
        const int LS_C = a.P + (a.P >> 3) + 4;
        const bool is_bdry = (a.pass_kind == 0 && blockIdx.x >= a.n_row_blocks);
        const bool has1 = is_bdry ? true : ((blockIdx.x * 2 + 1) < M);
        df* r1_buf = reinterpret_cast<df*>(smem_c + LS_C);
        df* r2_buf = a.compact_smem ? (r1_buf + n_odd) : (r1_buf + M);
        df_prefix_sum_pair_smem<BLOCK_T>(r1_buf, r2_buf, n_odd, has1);
    }

    // 3c. Coalesced stores
    {
        const int M = a.M;
        const int tx = threadIdx.x;
        const bool is_bdry = (a.pass_kind == 0 && blockIdx.x >= a.n_row_blocks);
        const int row0 = blockIdx.x * 2;
        const int row1 = row0 + 1;
        const bool has1 = is_bdry ? true : (row1 < M);
        const int n_odd = (M + 1) >> 1;
        const int LS_C = a.P + (a.P >> 3) + 4;

        df* r1_buf = reinterpret_cast<df*>(smem_c + LS_C);
        df* i1_buf = a.compact_smem ? nullptr : (r1_buf + n_odd);
        df* r2_buf = a.compact_smem ? (r1_buf + n_odd) : (r1_buf + M);
        df* i2_buf = a.compact_smem ? nullptr : (r2_buf + n_odd);

        df* out0 = is_bdry ? (a.d_bdry_hat + static_cast<size_t>((blockIdx.x - a.n_row_blocks) * 2) * M)
                           : (a.d_out_df + static_cast<size_t>(row0) * M);
        df* out1 = is_bdry ? (a.d_bdry_hat + static_cast<size_t>((blockIdx.x - a.n_row_blocks) * 2 + 1) * M)
                           : (has1 ? (a.d_out_df + static_cast<size_t>(row1) * M) : nullptr);
        float* fout0 = (a.pass_kind == 2) ? (a.d_buf1 + static_cast<size_t>(row0 + 1) * a.n1 + 1) : nullptr;
        float* fout1 = (a.pass_kind == 2 && has1) ? (a.d_buf1 + static_cast<size_t>(row1 + 1) * a.n1 + 1) : nullptr;

        if (!a.compact_smem) {
            for (int p = tx; p < M; p += BLOCK_T) {
                int h = p >> 1;
                df v0 = df_mul((p & 1) ? i1_buf[h] : r1_buf[h], a.scale_dst);
                st_df(out0 + p, v0);
                if (a.pass_kind == 2) fout0[p] = __fadd_rn(v0.hi, v0.lo);
                if (has1) {
                    df v1 = df_mul((p & 1) ? i2_buf[h] : r2_buf[h], a.scale_dst);
                    st_df(out1 + p, v1);
                    if (a.pass_kind == 2) fout1[p] = __fadd_rn(v1.hi, v1.lo);
                }
            }
        } else {
            for (int m = tx; m < n_odd; m += BLOCK_T) {
                df v0 = df_mul(r1_buf[m], a.scale_dst);
                st_df(out0 + (2 * m), v0);
                if (a.pass_kind == 2) fout0[2 * m] = __fadd_rn(v0.hi, v0.lo);
                if (has1) {
                    df v1 = df_mul(r2_buf[m], a.scale_dst);
                    st_df(out1 + (2 * m), v1);
                    if (a.pass_kind == 2) fout1[2 * m] = __fadd_rn(v1.hi, v1.lo);
                }
            }
        }
    }
}

// ============================================================================
// Engine 5: L2-Batched Bailey 4-Step Bluestein 2D DST-I (n1 > 6145 up to 92681)
// Interleaved 128-bit SMEM (LDS.128/STS.128, 0 bank conflicts), fused zero-pruned R=128,
// single-load Step 4, and L2-resident batch sizing!
// ============================================================================

struct BaileyArgs {
    int R, RF1, RF2, R_MR8, C, M, total_rows, in_stride, in_offset, pair_start;
    int R_div_F1, tw_off_f2, r8_off_start, r8_off_end, STRIDE, fuse_r2;
    const void* d_in_raw;
    cdf* d_scratch;
    cdf* d_Z;
    const df* d_sin_k;
    const cdf* d_chirp;
    const cdf* d_tw_R;
    const cdf* d_tw_bailey;
};

__global__ void pack_boundaries_kernel(
    int M, int n1,
    const float* __restrict__ d_buf0,
    const float* __restrict__ d_buf1,
    float* __restrict__ d_bdry_raw
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < M) {
        d_bdry_raw[0 * M + k] = d_buf0[k + 1];
        d_bdry_raw[1 * M + k] = d_buf0[static_cast<size_t>(n1 - 1) * n1 + (k + 1)];
        d_bdry_raw[2 * M + k] = d_buf0[static_cast<size_t>(k + 1) * n1];
        d_bdry_raw[3 * M + k] = d_buf0[static_cast<size_t>(k + 1) * n1 + (n1 - 1)];
        d_bdry_raw[4 * M + k] = d_buf1[k + 1];
        d_bdry_raw[5 * M + k] = d_buf1[static_cast<size_t>(n1 - 1) * n1 + (k + 1)];
        d_bdry_raw[6 * M + k] = d_buf1[static_cast<size_t>(k + 1) * n1];
        d_bdry_raw[7 * M + k] = d_buf1[static_cast<size_t>(k + 1) * n1 + (n1 - 1)];
    }
}

template <bool IN_F32>
__global__ void __launch_bounds__(128, 6)
bailey_step1_kernel(BaileyArgs a) {
    extern __shared__ cdf smem_c[];
    const int tid = threadIdx.x;
    const int c_sub = tid & 7;
    const int r_sub = tid >> 3;
    const int c = blockIdx.x * 8 + c_sub;

    // 1. Load + pre-chirp (+ fused zero-pruned R2 when RF1==1 && RF2==2)
    {
        const int row0 = (a.pair_start + blockIdx.y) * 2;
        const int row1 = row0 + 1;
        const bool has1 = (row1 < a.total_rows);
        const int L = a.M + 1;
        cdf* sc = smem_c + c_sub * a.STRIDE;
        const int rlim = a.fuse_r2 ? (a.R >> 1) : a.R;

        for (int r = r_sub; r < rlim; r += 16) {
            int k = r * a.C + c;
            cdf val = {{0.0f, 0.0f}, {0.0f, 0.0f}};
            if (k >= 1 && k < L) {
                df s = ldg_df(a.d_sin_k + k);
                df y0, y1 = {0.0f, 0.0f};
                if constexpr (IN_F32) {
                    const float* base = reinterpret_cast<const float*>(a.d_in_raw);
                    const float* p0 = base + static_cast<size_t>(row0) * a.in_stride + a.in_offset;
                    float x0a = __ldg(p0 + k - 1), x0b = __ldg(p0 + L - k - 1);
                    y0 = df_add(df_mul(s, df_add_u({x0a, 0.0f}, {x0b, 0.0f})),
                                df_mul_f(df_sub_u({x0a, 0.0f}, {x0b, 0.0f}), 0.5f));
                    if (has1) {
                        const float* p1 = base + static_cast<size_t>(row1) * a.in_stride + a.in_offset;
                        float x1a = __ldg(p1 + k - 1), x1b = __ldg(p1 + L - k - 1);
                        y1 = df_add(df_mul(s, df_add_u({x1a, 0.0f}, {x1b, 0.0f})),
                                    df_mul_f(df_sub_u({x1a, 0.0f}, {x1b, 0.0f}), 0.5f));
                    }
                } else {
                    const df* base = reinterpret_cast<const df*>(a.d_in_raw);
                    const df* p0 = base + static_cast<size_t>(row0) * a.in_stride + a.in_offset;
                    df x0a = ldg_df(p0 + k - 1), x0b = ldg_df(p0 + L - k - 1);
                    y0 = df_add(df_mul(s, df_add_u(x0a, x0b)), df_mul_f(df_sub_u(x0a, x0b), 0.5f));
                    if (has1) {
                        const df* p1 = base + static_cast<size_t>(row1) * a.in_stride + a.in_offset;
                        df x1a = ldg_df(p1 + k - 1), x1b = ldg_df(p1 + L - k - 1);
                        y1 = df_add(df_mul(s, df_add_u(x1a, x1b)), df_mul_f(df_sub_u(x1a, x1b), 0.5f));
                    }
                }
                val = cdf_mul({y0, y1}, ldg_cdf(a.d_chirp + k));
            }
            sc[S2IDX_C(r)] = val;
            if (a.fuse_r2) {
                sc[S2IDX_C(r + rlim)] = (k >= 1 && k < L) ? cdf_mul(val, ldg_cdf(a.d_tw_R + r)) : val;
            }
        }
        __syncthreads();
    }

    // 2. Column length-R forward DIF FFT (16 threads per column)
    {
        const int g = tid >> 4;
        const int lid = tid & 15;
        cdf* sc = smem_c + g * a.STRIDE;

        if (!a.fuse_r2) {
            if (a.RF1 == 3) {
                const int step = a.R_div_F1 + (a.R_div_F1 >> 3);
                for (int j = lid; j < a.R_div_F1; j += 16) {
                    cdf* p = sc + S2IDX_C(j);
                    cdf x0 = p[0], x1 = p[step], x2 = p[2 * step];
                    radix3_bfly<false>(x0, x1, x2);
                    p[0] = x0;
                    p[step]     = cdf_mul(x1, ldg_cdf(a.d_tw_R + j));
                    p[2 * step] = cdf_mul(x2, ldg_cdf(a.d_tw_R + a.R_div_F1 + j));
                }
                __syncwarp();
            }
            if (a.RF2 == 2) {
                const int sh = __ffs(a.R_MR8) - 1, mask = a.R_MR8 - 1, nb = a.R >> 1;
                const int step = a.R_MR8 + (a.R_MR8 >> 3);
                for (int b = lid; b < nb; b += 16) {
                    int blk = b >> sh, j = b & mask, base = blk * a.R_div_F1 + j;
                    cdf* p = sc + S2IDX_C(base);
                    cdf x0 = p[0], x1 = p[step];
                    p[0] = cdf_add(x0, x1);
                    p[step] = cdf_mul(cdf_sub_u(x0, x1), ldg_cdf(a.d_tw_R + a.tw_off_f2 + j));
                }
                __syncwarp();
            } else if (a.RF2 == 4) {
                const int sh = __ffs(a.R_MR8) - 1, mask = a.R_MR8 - 1, nb = a.R >> 2;
                const int step = a.R_MR8 + (a.R_MR8 >> 3);
                for (int b = lid; b < nb; b += 16) {
                    int blk = b >> sh, j = b & mask, base = blk * a.R_div_F1 + j;
                    cdf* p = sc + S2IDX_C(base);
                    cdf x0 = p[0], x1 = p[step], x2 = p[2 * step], x3 = p[3 * step];
                    radix4_bfly<false, false>(x0, x1, x2, x3);
                    p[0] = {df_renorm(x0.r), df_renorm(x0.i)};
                    p[step]     = cdf_mul(x1, ldg_cdf(a.d_tw_R + a.tw_off_f2 + 0 * a.R_MR8 + j));
                    p[2 * step] = cdf_mul(x2, ldg_cdf(a.d_tw_R + a.tw_off_f2 + 1 * a.R_MR8 + j));
                    p[3 * step] = cdf_mul(x3, ldg_cdf(a.d_tw_R + a.tw_off_f2 + 2 * a.R_MR8 + j));
                }
                __syncwarp();
            }
        }

        const int nb = a.R >> 3;
        int off = a.r8_off_start;
        #pragma unroll 1
        for (int m = a.R_MR8; m > 8; m >>= 3) {
            int stride = m >> 3;
            int step = stride + (stride >> 3);
            int sh = __ffs(stride) - 1, mask = stride - 1;
            const cdf* tw = a.d_tw_R + off;
            for (int b = lid; b < nb; b += 16) {
                int blk = b >> sh, j = b & mask, base = (blk << (sh + 3)) + j;
                cdf* p = sc + S2IDX_C(base);
                cdf x0 = p[0],        x1 = p[1 * step], x2 = p[2 * step], x3 = p[3 * step];
                cdf x4 = p[4 * step], x5 = p[5 * step], x6 = p[6 * step], x7 = p[7 * step];
                radix8_bfly<false, false>(x0, x1, x2, x3, x4, x5, x6, x7);
                p[0]        = {df_renorm(x0.r), df_renorm(x0.i)};
                p[1 * step] = cdf_mul(x1, ldg_cdf(tw + 0 * stride + j));
                p[2 * step] = cdf_mul(x2, ldg_cdf(tw + 1 * stride + j));
                p[3 * step] = cdf_mul(x3, ldg_cdf(tw + 2 * stride + j));
                p[4 * step] = cdf_mul(x4, ldg_cdf(tw + 3 * stride + j));
                p[5 * step] = cdf_mul(x5, ldg_cdf(tw + 4 * stride + j));
                p[6 * step] = cdf_mul(x6, ldg_cdf(tw + 5 * stride + j));
                p[7 * step] = cdf_mul(x7, ldg_cdf(tw + 6 * stride + j));
            }
            __syncwarp();
            off += 7 * stride;
        }
        for (int b = lid; b < nb; b += 16) {
            cdf* p = sc + b * 9;
            cdf x0 = p[0], x1 = p[1], x2 = p[2], x3 = p[3];
            cdf x4 = p[4], x5 = p[5], x6 = p[6], x7 = p[7];
            radix8_bfly<false, true>(x0, x1, x2, x3, x4, x5, x6, x7);
            p[0] = x0; p[1] = x1; p[2] = x2; p[3] = x3;
            p[4] = x4; p[5] = x5; p[6] = x6; p[7] = x7;
        }
        __syncthreads();
    }

    // 3. Apply Bailey twiddles & coalesced store to d_scratch
    cdf* dst_base = a.d_scratch + static_cast<size_t>(blockIdx.y) * (static_cast<size_t>(a.R) * a.C);
    cdf* sc = smem_c + c_sub * a.STRIDE;
    for (int kr = r_sub; kr < a.R; kr += 16) {
        cdf v = cdf_mul(sc[S2IDX_C(kr)], ldg_cdf(a.d_tw_bailey + static_cast<size_t>(kr) * a.C + c));
        st_cdf(dst_base + static_cast<size_t>(kr) * a.C + c, v);
    }
}

// Step 2: C is ALWAYS 128 (2*64), 256 (4*64), or 512 (8*64) -> F1 == 1 ALWAYS!
// Interleaved 128-bit SMEM + fused innermost r8(8) fwd + bhat mul + r8(8) inv!
template <int C, int CF2, int GRP_T, int GROUPS>
__global__ void __launch_bounds__(GRP_T * GROUPS, 3)
bailey_step2_kernel(
    int R, int total_rows,
    cdf* __restrict__ d_scratch,
    const cdf* __restrict__ d_tw_C,
    const cdf* __restrict__ d_bhat,
    const cdf* __restrict__ d_tw_bailey
) {
    constexpr int STRIDE = C + (C >> 3) + 1;
    constexpr int C_MR8 = C / CF2;
    extern __shared__ cdf smem_c[];

    const int tid = threadIdx.x;
    const int g = tid / GRP_T;
    const int lid = tid - g * GRP_T;
    const int row_idx = blockIdx.x * GROUPS + g;
    if (row_idx >= total_rows) return;
    const int kr = row_idx % R;

    cdf* row_ptr = d_scratch + static_cast<size_t>(row_idx) * C;
    cdf* sc = smem_c + g * STRIDE;

    auto sync_grp = [&]() {
        if constexpr (GRP_T <= 32) __syncwarp();
        else __syncthreads();
    };

    for (int c = lid; c < C; c += GRP_T) {
        sc[S2IDX_C(c)] = ldg_cdf(row_ptr + c);
    }
    sync_grp();

    // Forward outer stages down to m = 8
    int off = 0;
    if constexpr (CF2 == 2) {
        constexpr int STEP64 = 72;
        for (int j = lid; j < 64; j += GRP_T) {
            cdf* p = sc + S2IDX_C(j);
            cdf x0 = p[0], x1 = p[STEP64];
            p[0] = cdf_add(x0, x1);
            p[STEP64] = cdf_mul(cdf_sub_u(x0, x1), ldg_cdf(d_tw_C + j));
        }
        sync_grp();
        off = 64;
    } else if constexpr (CF2 == 4) {
        constexpr int STEP64 = 72;
        for (int j = lid; j < 64; j += GRP_T) {
            cdf* p = sc + S2IDX_C(j);
            cdf x0 = p[0], x1 = p[STEP64], x2 = p[2 * STEP64], x3 = p[3 * STEP64];
            radix4_bfly<false, false>(x0, x1, x2, x3);
            p[0] = {df_renorm(x0.r), df_renorm(x0.i)};
            p[STEP64]     = cdf_mul(x1, ldg_cdf(d_tw_C + 0 * 64 + j));
            p[2 * STEP64] = cdf_mul(x2, ldg_cdf(d_tw_C + 1 * 64 + j));
            p[3 * STEP64] = cdf_mul(x3, ldg_cdf(d_tw_C + 2 * 64 + j));
        }
        sync_grp();
        off = 192;
    }
    constexpr int nb = C >> 3;
    #pragma unroll 1
    for (int m = C_MR8; m > 8; m >>= 3) {
        int stride = m >> 3;
        int step = stride + (stride >> 3);
        int sh = __ffs(stride) - 1, mask = stride - 1;
        const cdf* tw = d_tw_C + off;
        for (int b = lid; b < nb; b += GRP_T) {
            int blk = b >> sh, j = b & mask, base = (blk << (sh + 3)) + j;
            cdf* p = sc + S2IDX_C(base);
            cdf x0 = p[0],        x1 = p[1 * step], x2 = p[2 * step], x3 = p[3 * step];
            cdf x4 = p[4 * step], x5 = p[5 * step], x6 = p[6 * step], x7 = p[7 * step];
            radix8_bfly<false, false>(x0, x1, x2, x3, x4, x5, x6, x7);
            p[0]        = {df_renorm(x0.r), df_renorm(x0.i)};
            p[1 * step] = cdf_mul(x1, ldg_cdf(tw + 0 * stride + j));
            p[2 * step] = cdf_mul(x2, ldg_cdf(tw + 1 * stride + j));
            p[3 * step] = cdf_mul(x3, ldg_cdf(tw + 2 * stride + j));
            p[4 * step] = cdf_mul(x4, ldg_cdf(tw + 3 * stride + j));
            p[5 * step] = cdf_mul(x5, ldg_cdf(tw + 4 * stride + j));
            p[6 * step] = cdf_mul(x6, ldg_cdf(tw + 5 * stride + j));
            p[7 * step] = cdf_mul(x7, ldg_cdf(tw + 6 * stride + j));
        }
        sync_grp();
        off += 7 * stride;
    }

    // FUSED innermost r8(8) fwd + bhat mul + r8(8) inv in registers! (contiguous p[0..7]!)
    const cdf* bh = d_bhat + static_cast<size_t>(kr) * C;
    for (int b = lid; b < nb; b += GRP_T) {
        int base = b << 3;
        cdf* p = sc + b * 9;
        cdf x0 = p[0], x1 = p[1], x2 = p[2], x3 = p[3];
        cdf x4 = p[4], x5 = p[5], x6 = p[6], x7 = p[7];
        radix8_bfly<false, false>(x0, x1, x2, x3, x4, x5, x6, x7);
        x0 = cdf_mul(x0, ldg_cdf(bh + base + 0)); x1 = cdf_mul(x1, ldg_cdf(bh + base + 1));
        x2 = cdf_mul(x2, ldg_cdf(bh + base + 2)); x3 = cdf_mul(x3, ldg_cdf(bh + base + 3));
        x4 = cdf_mul(x4, ldg_cdf(bh + base + 4)); x5 = cdf_mul(x5, ldg_cdf(bh + base + 5));
        x6 = cdf_mul(x6, ldg_cdf(bh + base + 6)); x7 = cdf_mul(x7, ldg_cdf(bh + base + 7));
        radix8_bfly<true, true>(x0, x1, x2, x3, x4, x5, x6, x7);
        p[0] = x0; p[1] = x1; p[2] = x2; p[3] = x3;
        p[4] = x4; p[5] = x5; p[6] = x6; p[7] = x7;
    }
    sync_grp();

    // Inverse outer stages
    #pragma unroll 1
    for (int m = 64; m <= C_MR8; m <<= 3) {
        int stride = m >> 3;
        int step = stride + (stride >> 3);
        off -= 7 * stride;
        int sh = __ffs(stride) - 1, mask = stride - 1;
        const cdf* tw = d_tw_C + off;
        for (int b = lid; b < nb; b += GRP_T) {
            int blk = b >> sh, j = b & mask, base = (blk << (sh + 3)) + j;
            cdf* p = sc + S2IDX_C(base);
            cdf x0 = p[0];
            cdf x1 = cdf_mul(p[1 * step], cdf_conj(ldg_cdf(tw + 0 * stride + j)));
            cdf x2 = cdf_mul(p[2 * step], cdf_conj(ldg_cdf(tw + 1 * stride + j)));
            cdf x3 = cdf_mul(p[3 * step], cdf_conj(ldg_cdf(tw + 2 * stride + j)));
            cdf x4 = cdf_mul(p[4 * step], cdf_conj(ldg_cdf(tw + 3 * stride + j)));
            cdf x5 = cdf_mul(p[5 * step], cdf_conj(ldg_cdf(tw + 4 * stride + j)));
            cdf x6 = cdf_mul(p[6 * step], cdf_conj(ldg_cdf(tw + 5 * stride + j)));
            cdf x7 = cdf_mul(p[7 * step], cdf_conj(ldg_cdf(tw + 6 * stride + j)));
            radix8_bfly<true, true>(x0, x1, x2, x3, x4, x5, x6, x7);
            p[0]        = x0; p[1 * step] = x1; p[2 * step] = x2; p[3 * step] = x3;
            p[4 * step] = x4; p[5 * step] = x5; p[6 * step] = x6; p[7 * step] = x7;
        }
        sync_grp();
    }
    if constexpr (CF2 == 2) {
        constexpr int STEP64 = 72;
        for (int j = lid; j < 64; j += GRP_T) {
            cdf* p = sc + S2IDX_C(j);
            cdf x0 = p[0];
            cdf x1 = cdf_mul(p[STEP64], cdf_conj(ldg_cdf(d_tw_C + j)));
            p[0] = cdf_add(x0, x1);
            p[STEP64] = cdf_sub(x0, x1);
        }
        sync_grp();
    } else if constexpr (CF2 == 4) {
        constexpr int STEP64 = 72;
        for (int j = lid; j < 64; j += GRP_T) {
            cdf* p = sc + S2IDX_C(j);
            cdf x0 = p[0];
            cdf x1 = cdf_mul(p[STEP64],     cdf_conj(ldg_cdf(d_tw_C + 0 * 64 + j)));
            cdf x2 = cdf_mul(p[2 * STEP64], cdf_conj(ldg_cdf(d_tw_C + 1 * 64 + j)));
            cdf x3 = cdf_mul(p[3 * STEP64], cdf_conj(ldg_cdf(d_tw_C + 2 * 64 + j)));
            radix4_bfly<true, true>(x0, x1, x2, x3);
            p[0] = x0; p[STEP64] = x1; p[2 * STEP64] = x2; p[3 * STEP64] = x3;
        }
        sync_grp();
    }

    const cdf* twb = d_tw_bailey + static_cast<size_t>(kr) * C;
    for (int c = lid; c < C; c += GRP_T) {
        cdf v = cdf_mul(sc[S2IDX_C(c)], cdf_conj(ldg_cdf(twb + c)));
        st_cdf(row_ptr + c, v);
    }
}

__global__ void __launch_bounds__(128, 6)
bailey_step3_kernel(BaileyArgs a) {
    extern __shared__ cdf smem_c[];
    const int tid = threadIdx.x;
    const int c_sub = tid & 7;
    const int r_sub = tid >> 3;
    const int c = blockIdx.x * 8 + c_sub;

    // 1. Load from d_scratch into transposed SMEM
    {
        const cdf* src_base = a.d_scratch + static_cast<size_t>(blockIdx.y) * (static_cast<size_t>(a.R) * a.C);
        cdf* sc = smem_c + c_sub * a.STRIDE;
        for (int kr = r_sub; kr < a.R; kr += 16) {
            sc[S2IDX_C(kr)] = ldg_cdf(src_base + static_cast<size_t>(kr) * a.C + c);
        }
        __syncthreads();
    }

    // 2. Column length-R inverse DIT FFT (16 threads per column)
    {
        const int g = tid >> 4;
        const int lid = tid & 15;
        cdf* sc = smem_c + g * a.STRIDE;

        const int nb = a.R >> 3;
        for (int b = lid; b < nb; b += 16) {
            cdf* p = sc + b * 9;
            cdf x0 = p[0], x1 = p[1], x2 = p[2], x3 = p[3];
            cdf x4 = p[4], x5 = p[5], x6 = p[6], x7 = p[7];
            radix8_bfly<true, true>(x0, x1, x2, x3, x4, x5, x6, x7);
            p[0] = x0; p[1] = x1; p[2] = x2; p[3] = x3;
            p[4] = x4; p[5] = x5; p[6] = x6; p[7] = x7;
        }
        __syncwarp();

        int off = a.r8_off_end;
        #pragma unroll 1
        for (int m = 64; m <= a.R_MR8; m <<= 3) {
            int stride = m >> 3;
            int step = stride + (stride >> 3);
            off -= 7 * stride;
            int sh = __ffs(stride) - 1, mask = stride - 1;
            const cdf* tw = a.d_tw_R + off;
            for (int b = lid; b < nb; b += 16) {
                int blk = b >> sh, j = b & mask, base = (blk << (sh + 3)) + j;
                cdf* p = sc + S2IDX_C(base);
                cdf x0 = p[0];
                cdf x1 = cdf_mul(p[1 * step], cdf_conj(ldg_cdf(tw + 0 * stride + j)));
                cdf x2 = cdf_mul(p[2 * step], cdf_conj(ldg_cdf(tw + 1 * stride + j)));
                cdf x3 = cdf_mul(p[3 * step], cdf_conj(ldg_cdf(tw + 2 * stride + j)));
                cdf x4 = cdf_mul(p[4 * step], cdf_conj(ldg_cdf(tw + 3 * stride + j)));
                cdf x5 = cdf_mul(p[5 * step], cdf_conj(ldg_cdf(tw + 4 * stride + j)));
                cdf x6 = cdf_mul(p[6 * step], cdf_conj(ldg_cdf(tw + 5 * stride + j)));
                cdf x7 = cdf_mul(p[7 * step], cdf_conj(ldg_cdf(tw + 6 * stride + j)));
                radix8_bfly<true, true>(x0, x1, x2, x3, x4, x5, x6, x7);
                p[0]        = x0; p[1 * step] = x1; p[2 * step] = x2; p[3 * step] = x3;
                p[4 * step] = x4; p[5 * step] = x5; p[6 * step] = x6; p[7 * step] = x7;
            }
            __syncwarp();
        }

        if (!a.fuse_r2) {
            if (a.RF2 == 2) {
                const int sh = __ffs(a.R_MR8) - 1, mask = a.R_MR8 - 1, nb2 = a.R >> 1;
                const int step = a.R_MR8 + (a.R_MR8 >> 3);
                for (int b = lid; b < nb2; b += 16) {
                    int blk = b >> sh, j = b & mask, base = blk * a.R_div_F1 + j;
                    cdf* p = sc + S2IDX_C(base);
                    cdf x0 = p[0];
                    cdf x1 = cdf_mul(p[step], cdf_conj(ldg_cdf(a.d_tw_R + a.tw_off_f2 + j)));
                    p[0] = cdf_add(x0, x1);
                    p[step] = cdf_sub(x0, x1);
                }
                __syncwarp();
            } else if (a.RF2 == 4) {
                const int sh = __ffs(a.R_MR8) - 1, mask = a.R_MR8 - 1, nb4 = a.R >> 2;
                const int step = a.R_MR8 + (a.R_MR8 >> 3);
                for (int b = lid; b < nb4; b += 16) {
                    int blk = b >> sh, j = b & mask, base = blk * a.R_div_F1 + j;
                    cdf* p = sc + S2IDX_C(base);
                    cdf x0 = p[0];
                    cdf x1 = cdf_mul(p[1 * step], cdf_conj(ldg_cdf(a.d_tw_R + a.tw_off_f2 + 0 * a.R_MR8 + j)));
                    cdf x2 = cdf_mul(p[2 * step], cdf_conj(ldg_cdf(a.d_tw_R + a.tw_off_f2 + 1 * a.R_MR8 + j)));
                    cdf x3 = cdf_mul(p[3 * step], cdf_conj(ldg_cdf(a.d_tw_R + a.tw_off_f2 + 2 * a.R_MR8 + j)));
                    radix4_bfly<true, true>(x0, x1, x2, x3);
                    p[0] = x0; p[1 * step] = x1; p[2 * step] = x2; p[3 * step] = x3;
                }
                __syncwarp();
            }
            if (a.RF1 == 3) {
                const int step = a.R_div_F1 + (a.R_div_F1 >> 3);
                for (int j = lid; j < a.R_div_F1; j += 16) {
                    cdf* p = sc + S2IDX_C(j);
                    cdf x0 = p[0];
                    cdf x1 = cdf_mul(p[1 * step], cdf_conj(ldg_cdf(a.d_tw_R + j)));
                    cdf x2 = cdf_mul(p[2 * step], cdf_conj(ldg_cdf(a.d_tw_R + a.R_div_F1 + j)));
                    radix3_bfly<true>(x0, x1, x2);
                    p[0] = x0; p[1 * step] = x1; p[2 * step] = x2;
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }

    // 3. Post-chirp (+ fused zero-pruned R2 when RF1==1 && RF2==2) & store to d_Z
    const int L = a.M + 1;
    cdf* z_base = a.d_Z + static_cast<size_t>(blockIdx.y) * L;
    cdf* sc = smem_c + c_sub * a.STRIDE;
    if (a.fuse_r2) {
        const int halfR = a.R >> 1;
        for (int r = r_sub; r < halfR; r += 16) {
            int m = r * a.C + c;
            if (m < L) {
                cdf x0 = sc[S2IDX_C(r)];
                cdf x1 = cdf_mul(sc[S2IDX_C(r + halfR)], cdf_conj(ldg_cdf(a.d_tw_R + r)));
                cdf v = cdf_mul(cdf_add_u(x0, x1), ldg_cdf(a.d_chirp + m));
                st_cdf(z_base + m, v);
            }
        }
    } else {
        for (int r = r_sub; r < a.R; r += 16) {
            int m = r * a.C + c;
            if (m < L) {
                cdf v = cdf_mul(sc[S2IDX_C(r)], ldg_cdf(a.d_chirp + m));
                st_cdf(z_base + m, v);
            }
        }
    }
}

template <bool WRITE_F32_ALSO>
__global__ void __launch_bounds__(512, 2)
bailey_step4_extract_kernel(
    int M, int total_rows, int n1, int pair_start,
    const cdf* __restrict__ d_Z,
    df* __restrict__ d_out_df,
    float* __restrict__ d_out_f32,
    df scale_dst
) {
    constexpr int CHUNK = 2048;
    extern __shared__ df smem_df[];
    df* s_r1 = smem_df;
    df* s_i1 = s_r1 + CHUNK;
    df* s_r2 = s_i1 + CHUNK;
    df* s_i2 = s_r2 + CHUNK;

    const int L = M + 1;
    const int n_odd = (M + 1) >> 1;
    const int bp = blockIdx.x;
    const int row0 = (pair_start + bp) * 2;
    const int row1 = row0 + 1;
    const bool has1 = (row1 < total_rows);

    const cdf* Z = d_Z + static_cast<size_t>(bp) * L;
    df* out0_df = d_out_df + static_cast<size_t>(row0) * M;
    df* out1_df = has1 ? (d_out_df + static_cast<size_t>(row1) * M) : nullptr;
    float* out0_f = WRITE_F32_ALSO ? (d_out_f32 + static_cast<size_t>(row0 + 1) * n1 + 1) : nullptr;
    float* out1_f = (WRITE_F32_ALSO && has1) ? (d_out_f32 + static_cast<size_t>(row1 + 1) * n1 + 1) : nullptr;

    const int tx = threadIdx.x;
    df carry1 = {0.0f, 0.0f}, carry2 = {0.0f, 0.0f};

    for (int base = 0; base < n_odd; base += CHUNK) {
        int cnt = min(CHUNK, n_odd - base);
        for (int k = tx; k < cnt; k += 512) {
            int m = base + k;
            if (m == 0) {
                cdf z0 = ldg_cdf(Z);
                s_r1[0] = df_mul_f(z0.r, 0.5f);
                if (has1) s_r2[0] = df_mul_f(z0.i, 0.5f);
            } else {
                cdf zm = ldg_cdf(Z + m), zc = ldg_cdf(Z + L - m);
                s_r1[k] = df_mul_f(df_add_u(zm.r, zc.r), 0.5f);
                if (has1) s_r2[k] = df_mul_f(df_add_u(zm.i, zc.i), 0.5f);
                if (k > 0 && 2 * m <= M) {
                    s_i1[k - 1] = df_neg(df_mul_f(df_sub_u(zm.i, zc.i), 0.5f));
                    if (has1) s_i2[k - 1] = df_neg(df_mul_f(df_sub_u(zc.r, zm.r), 0.5f));
                }
            }
        }
        if (tx == 0) {
            int m_end = base + cnt;
            if (2 * m_end <= M) {
                cdf zm = ldg_cdf(Z + m_end), zc = ldg_cdf(Z + L - m_end);
                s_i1[cnt - 1] = df_neg(df_mul_f(df_sub_u(zm.i, zc.i), 0.5f));
                if (has1) s_i2[cnt - 1] = df_neg(df_mul_f(df_sub_u(zc.r, zm.r), 0.5f));
            }
        }
        __syncthreads();

        df_prefix_sum_pair_smem<512>(s_r1, s_r2, cnt, has1);

        int p_start = 2 * base;
        int p_end = min(M, 2 * (base + cnt));
        int p_cnt = p_end - p_start;
        for (int t = tx; t < p_cnt; t += 512) {
            int p = p_start + t;
            int k = t >> 1;
            df v0 = (t & 1) ? s_i1[k] : df_add(s_r1[k], carry1);
            v0 = df_mul(v0, scale_dst);
            st_df(out0_df + p, v0);
            if constexpr (WRITE_F32_ALSO) out0_f[p] = __fadd_rn(v0.hi, v0.lo);
            if (has1) {
                df v1 = (t & 1) ? s_i2[k] : df_add(s_r2[k], carry2);
                v1 = df_mul(v1, scale_dst);
                st_df(out1_df + p, v1);
                if constexpr (WRITE_F32_ALSO) out1_f[p] = __fadd_rn(v1.hi, v1.lo);
            }
        }

        carry1 = df_add(carry1, s_r1[cnt - 1]);
        if (has1) carry2 = df_add(carry2, s_r2[cnt - 1]);
        __syncthreads();
    }
}

// ============================================================================
// Host Table Builders & Plan Management
// ============================================================================

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

enum ExecMode {
    MODE_UNINIT = 0,
    MODE_NOOP = 1,
    MODE_DIRECT_SMEM1 = 2,
    MODE_DIRECT_2D = 3,
    MODE_SPECTRAL_SMOOTH = 4,
    MODE_SPECTRAL_BLUESTEIN = 5,
    MODE_SPECTRAL_BAILEY = 6
};

struct SpecPlan {
    int P;
    int F1;
    int F2;
    int BT;
};

static const SpecPlan SINGLE_PLANS[] = {
    {   64, 1, 1,  128},
    {  128, 1, 2,  128},
    {  192, 3, 1,  128},
    {  256, 1, 4,  128},
    {  320, 5, 1,  128},
    {  384, 3, 2,  128},
    {  512, 1, 1,  128},
    {  640, 5, 2,  128},
    {  768, 3, 4,  128},
    { 1024, 1, 2,  256},
    { 1280, 5, 4,  256},
    { 1536, 3, 1,  256},
    { 2048, 1, 4,  256},
    { 2560, 5, 1,  256},
    { 3072, 3, 2,  512},
    { 4096, 1, 1,  512},
    { 5120, 5, 2,  512},
    { 6144, 3, 4,  512},
    { 8192, 1, 2, 1024},
    {10240, 5, 4, 1024},
    {12288, 3, 1, 1024},
};

struct BaileyPlan {
    int P, R, RF1, RF2, C, CF2;
};

static const BaileyPlan BAILEY_PLANS[] = {
    { 16384, 128, 1, 2, 128, 2},
    { 24576, 192, 3, 1, 128, 2},
    { 32768, 256, 1, 4, 128, 2},
    { 49152, 192, 3, 1, 256, 4},
    { 65536, 256, 1, 4, 256, 4},
    { 98304, 384, 3, 2, 256, 4},
    {131072, 256, 1, 4, 512, 1},
    {196608, 384, 3, 2, 512, 1},
};

static std::vector<int> build_dif_perm(int N, int F1, int F2) {
    std::vector<int> radices;
    if (F1 > 1) radices.push_back(F1);
    if (F2 > 1) radices.push_back(F2);
    int m = N / (F1 * F2);
    while (m > 1) {
        radices.push_back(8);
        m /= 8;
    }
    int d = static_cast<int>(radices.size());
    std::vector<int> perm(N);
    for (int idx = 0; idx < N; ++idx) {
        int tmp = idx;
        int q[16];
        for (int i = d - 1; i >= 0; --i) {
            q[i] = tmp % radices[i];
            tmp /= radices[i];
        }
        int k = 0;
        for (int i = d - 1; i >= 0; --i) {
            k = k * radices[i] + q[i];
        }
        perm[idx] = k;
    }
    return perm;
}

static std::vector<cdf> build_stage_twiddles(int N, int F1, int F2) {
    std::vector<cdf> tw;
    auto push_tw = [&](int radix, int m) {
        int stride = m / radix;
        for (int r = 1; r < radix; ++r) {
            for (int j = 0; j < stride; ++j) {
                double ang = -2.0 * PI * static_cast<double>(r * j) / static_cast<double>(m);
                tw.push_back({df_from_double(cos(ang)), df_from_double(sin(ang))});
            }
        }
    };
    if (F1 > 1) push_tw(F1, N);
    if (F2 > 1) push_tw(F2, N / F1);
    int M_R8 = N / (F1 * F2);
    for (int m = M_R8; m > 8; m >>= 3) push_tw(8, m);
    return tw;
}

static void host_dft_dif_exact(int N, int F1, int F2, std::vector<double>& br, std::vector<double>& bi) {
    auto dft_small = [&](int radix, const int* idx) {
        double xr[8], xi[8], yr[8], yi[8];
        for (int k = 0; k < radix; ++k) { xr[k] = br[idx[k]]; xi[k] = bi[idx[k]]; }
        for (int q = 0; q < radix; ++q) {
            double sr = 0.0, si = 0.0;
            for (int n = 0; n < radix; ++n) {
                double a = -2.0 * PI * static_cast<double>(q * n) / static_cast<double>(radix);
                double c = cos(a), s = sin(a);
                sr += xr[n] * c - xi[n] * s;
                si += xr[n] * s + xi[n] * c;
            }
            yr[q] = sr; yi[q] = si;
        }
        for (int k = 0; k < radix; ++k) { br[idx[k]] = yr[k]; bi[idx[k]] = yi[k]; }
    };
    auto apply_dif = [&](int radix, int m) {
        int stride = m / radix;
        int idx[8];
        for (int blk = 0; blk < N / m; ++blk) {
            for (int j = 0; j < stride; ++j) {
                int base = blk * m + j;
                for (int r = 0; r < radix; ++r) idx[r] = base + r * stride;
                dft_small(radix, idx);
                for (int r = 1; r < radix; ++r) {
                    double ang = -2.0 * PI * static_cast<double>(r * j) / static_cast<double>(m);
                    double c = cos(ang), s = sin(ang);
                    double tr = br[idx[r]] * c - bi[idx[r]] * s;
                    double ti = br[idx[r]] * s + bi[idx[r]] * c;
                    br[idx[r]] = tr; bi[idx[r]] = ti;
                }
            }
        }
    };
    if (F1 > 1) apply_dif(F1, N);
    if (F2 > 1) apply_dif(F2, N / F1);
    int M_R8 = N / (F1 * F2);
    for (int m = M_R8; m > 8; m >>= 3) apply_dif(8, m);
    for (int b = 0; b < N / 8; ++b) {
        int idx[8];
        for (int r = 0; r < 8; ++r) idx[r] = b * 8 + r;
        dft_small(8, idx);
    }
}

static void host_fft_natural(int P, std::vector<double>& re, std::vector<double>& im) {
    std::vector<int> factors;
    int tmp = P;
    while (tmp % 5 == 0) { factors.push_back(5); tmp /= 5; }
    while (tmp % 3 == 0) { factors.push_back(3); tmp /= 3; }
    while (tmp % 2 == 0) { factors.push_back(2); tmp /= 2; }

    int stride = P;
    for (int rad : factors) {
        int m = stride;
        stride /= rad;
        int nblocks = P / m;
        for (int blk = 0; blk < nblocks; ++blk) {
            for (int j = 0; j < stride; ++j) {
                int base = blk * m + j;
                double xr[5], xi[5];
                for (int r = 0; r < rad; ++r) {
                    xr[r] = re[base + r * stride];
                    xi[r] = im[base + r * stride];
                }
                for (int q = 0; q < rad; ++q) {
                    double sr = 0.0, si = 0.0;
                    for (int n = 0; n < rad; ++n) {
                        double a = -2.0 * PI * static_cast<double>(q * n) / static_cast<double>(rad);
                        double c = cos(a), s = sin(a);
                        sr += xr[n] * c - xi[n] * s;
                        si += xr[n] * s + xi[n] * c;
                    }
                    if (q > 0 && j > 0) {
                        double ang = -2.0 * PI * static_cast<double>(q * j) / static_cast<double>(m);
                        double c = cos(ang), s = sin(ang);
                        double tr = sr * c - si * s;
                        double ti = sr * s + si * c;
                        sr = tr; si = ti;
                    }
                    re[base + q * stride] = sr;
                    im[base + q * stride] = si;
                }
            }
        }
    }
    std::vector<double> out_r(P), out_i(P);
    int d = static_cast<int>(factors.size());
    for (int idx = 0; idx < P; ++idx) {
        int t = idx;
        int q[32];
        for (int i = d - 1; i >= 0; --i) {
            q[i] = t % factors[i];
            t /= factors[i];
        }
        int k = 0;
        for (int i = d - 1; i >= 0; --i) {
            k = k * factors[i] + q[i];
        }
        out_r[k] = re[idx];
        out_i[k] = im[idx];
    }
    re.swap(out_r);
    im.swap(out_i);
}

struct SolverState {
    bool initialized = false;
    int n0 = 0;
    int n1 = 0;
    size_t elem_count = 0;
    size_t buf_bytes = 0;
    ExecMode mode = MODE_UNINIT;

    float* d_buf0 = nullptr;
    float* d_buf1 = nullptr;
    float* d_alt = nullptr;

    int M = 0;
    int L = 0;
    df scale_dst = {0.0f, 0.0f};
    df* d_work0 = nullptr;
    df* d_work1 = nullptr;
    df* d_bdry_hat = nullptr;
    df* d_sin_k = nullptr;
    df* d_sin2_half = nullptr;
    df* d_sp1 = nullptr;

    SpecPlan spec_plan = {0, 0, 0, 0};
    cdf* d_chirp = nullptr;
    cdf* d_tw_stages = nullptr;
    cdf* d_bhat_t = nullptr;
    int* d_inv_perm = nullptr;

    BaileyPlan bailey_plan = {0, 0, 0, 0, 0, 0};
    int bailey_batch = 0;
    cdf* d_tw_R = nullptr;
    cdf* d_tw_C = nullptr;
    cdf* d_tw_bailey = nullptr;
    cdf* d_bhat_bailey = nullptr;
    cdf* d_scratch = nullptr;
    cdf* d_Z = nullptr;
    float* d_bdry_raw = nullptr;
};

static SolverState g;

static void free_all() {
    auto safe_free = [](auto*& p) {
        if (p) { cudaFree(p); p = nullptr; }
    };
    safe_free(g.d_buf0);
    safe_free(g.d_buf1);
    safe_free(g.d_alt);
    safe_free(g.d_work0);
    safe_free(g.d_work1);
    safe_free(g.d_bdry_hat);
    safe_free(g.d_sin_k);
    safe_free(g.d_sin2_half);
    safe_free(g.d_sp1);
    safe_free(g.d_chirp);
    safe_free(g.d_tw_stages);
    safe_free(g.d_bhat_t);
    safe_free(g.d_inv_perm);
    safe_free(g.d_tw_R);
    safe_free(g.d_tw_C);
    safe_free(g.d_tw_bailey);
    safe_free(g.d_bhat_bailey);
    safe_free(g.d_scratch);
    safe_free(g.d_Z);
    safe_free(g.d_bdry_raw);
    g = SolverState{};
}

#define CUDA_OK(call) do { if ((call) != cudaSuccess) return 1; } while (0)

static bool check_buffer_shape(const pb_buffer_shape* b, int64_t n1) {
    if (!b || b->dtype != PB_F32 || b->rank != 2) return false;
    if (b->shape[0] != n1 || b->shape[1] != n1) return false;
    uint64_t count = static_cast<uint64_t>(n1) * static_cast<uint64_t>(n1);
    if (b->nbytes != count * sizeof(float)) return false;
    return true;
}

static bool check_buffer_live(const pb_buffer* b, int64_t n1) {
    if (!b || !b->data || b->dtype != PB_F32 || b->rank != 2) return false;
    if (b->shape[0] != n1 || b->shape[1] != n1) return false;
    uint64_t count = static_cast<uint64_t>(n1) * static_cast<uint64_t>(n1);
    if (b->nbytes != count * sizeof(float)) return false;
    return (reinterpret_cast<uintptr_t>(b->data) % sizeof(float)) == 0;
}

template <int IN_X, int IN_Y, int K, int BX, int BY>
static cudaError_t set_direct2d_attr() {
    int smem = 2 * IN_Y * (IN_X + 1) * static_cast<int>(sizeof(float));
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(
            jacobi_2d_smem_kernel<IN_X, IN_Y, K, BX, BY, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
        e = cudaFuncSetAttribute(
            jacobi_2d_smem_kernel<IN_X, IN_Y, K, BX, BY, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

template <int IN_X, int IN_Y, int K, int BX, int BY, bool SAVE_PREV>
static void launch_direct2d(int n1, const float* src, const float* b0, const float* b1,
                            float* dst, float* dst_prev) {
    constexpr int OUT_X = IN_X - 2 * K;
    constexpr int OUT_Y = IN_Y - 2 * K;
    dim3 blk(BX, BY);
    dim3 grd(((n1 - 2) + OUT_X - 1) / OUT_X, ((n1 - 2) + OUT_Y - 1) / OUT_Y);
    size_t smem = 2 * IN_Y * (IN_X + 1) * sizeof(float);
    jacobi_2d_smem_kernel<IN_X, IN_Y, K, BX, BY, SAVE_PREV><<<grd, blk, smem>>>(
        n1, src, b0, b1, dst, dst_prev);
}

static int run_direct_2d() {
    const int n1 = g.n1;
    copy_border_kernel<<<(n1 + 255) / 256, 256>>>(n1, g.d_buf0, g.d_alt);

    int64_t rem = 2LL * g.n0;
    float* cur = g.d_buf0;
    float* nxt = g.d_alt;
    const float* b0_orig = g.d_buf0;
    const float* b1_orig = g.d_buf1;

    while (rem > 16) {
        if (n1 >= 256) {
            launch_direct2d<96, 96, 16, 32, 16, false>(n1, cur, b0_orig, b1_orig, nxt, nullptr);
            rem -= 16;
        } else {
            launch_direct2d<64, 64, 8, 32, 8, false>(n1, cur, b0_orig, b1_orig, nxt, nullptr);
            rem -= 8;
        }
        std::swap(cur, nxt);
    }

    if (rem == 16) {
        launch_direct2d<96, 96, 16, 32, 16, true>(n1, cur, b0_orig, b1_orig, nxt, g.d_buf1);
        std::swap(cur, nxt);
    } else if (rem == 14) {
        launch_direct2d<64, 64, 12, 32, 8, false>(n1, cur, b0_orig, b1_orig, nxt, nullptr);
        std::swap(cur, nxt);
        launch_direct2d<64, 32, 2, 32, 8, true>(n1, cur, b0_orig, b1_orig, nxt, g.d_buf1);
        std::swap(cur, nxt);
    } else if (rem == 12) {
        launch_direct2d<64, 64, 12, 32, 8, true>(n1, cur, b0_orig, b1_orig, nxt, g.d_buf1);
        std::swap(cur, nxt);
    } else if (rem == 10) {
        launch_direct2d<64, 64, 8, 32, 8, false>(n1, cur, b0_orig, b1_orig, nxt, nullptr);
        std::swap(cur, nxt);
        launch_direct2d<64, 32, 2, 32, 8, true>(n1, cur, b0_orig, b1_orig, nxt, g.d_buf1);
        std::swap(cur, nxt);
    } else if (rem == 8) {
        launch_direct2d<64, 64, 8, 32, 8, true>(n1, cur, b0_orig, b1_orig, nxt, g.d_buf1);
        std::swap(cur, nxt);
    } else if (rem == 6) {
        launch_direct2d<64, 32, 4, 32, 8, false>(n1, cur, b0_orig, b1_orig, nxt, nullptr);
        std::swap(cur, nxt);
        launch_direct2d<64, 32, 2, 32, 8, true>(n1, cur, b0_orig, b1_orig, nxt, g.d_buf1);
        std::swap(cur, nxt);
    } else if (rem == 4) {
        launch_direct2d<64, 32, 4, 32, 8, true>(n1, cur, b0_orig, b1_orig, nxt, g.d_buf1);
        std::swap(cur, nxt);
    } else if (rem == 2) {
        launch_direct2d<64, 32, 2, 32, 8, true>(n1, cur, b0_orig, b1_orig, nxt, g.d_buf1);
        std::swap(cur, nxt);
    }

    if (cur != g.d_buf0) {
        std::swap(g.d_buf0, g.d_alt);
    }
    return 0;
}

static SpecArgs make_spec_args(int pass_kind, bool is_smooth, const df* d_in_df, df* d_out_df) {
    const int P = g.spec_plan.P;
    const int F1 = g.spec_plan.F1;
    const int F2 = g.spec_plan.F2;
    const int M_R8 = P / (F1 * F2);
    const int P_div_F1 = F2 * M_R8;
    const int tw_off_f2 = (F1 > 1) ? (F1 - 1) * P_div_F1 : 0;
    const int r8_off_start = tw_off_f2 + ((F2 > 1) ? (F2 - 1) * M_R8 : 0);
    const int r8_off_end = r8_off_start + (M_R8 - 8);
    SpecArgs a{};
    a.pass_kind = pass_kind;
    a.P = P;
    a.F1 = F1;
    a.F2 = F2;
    a.M_R8 = M_R8;
    a.M = g.M;
    a.n1 = g.n1;
    a.P_div_F1 = P_div_F1;
    a.tw_off_f2 = tw_off_f2;
    a.r8_off_start = r8_off_start;
    a.r8_off_end = r8_off_end;
    a.n_row_blocks = (g.M + 1) >> 1;
    a.fuse_r2 = (!is_smooth && F1 == 1 && F2 == 2) ? 1 : 0;
    a.compact_smem = (is_smooth && (P == 4096 || P == 8192)) ? 1 : 0;
    a.d_buf0 = (pass_kind == 0) ? g.d_buf0 : nullptr;
    a.d_buf1 = (pass_kind == 0 || pass_kind == 2) ? g.d_buf1 : nullptr;
    a.d_in_df = d_in_df;
    a.d_out_df = d_out_df;
    a.d_bdry_hat = (pass_kind == 0) ? g.d_bdry_hat : nullptr;
    a.d_sin_k = g.d_sin_k;
    a.d_chirp = g.d_chirp;
    a.d_tw_stages = g.d_tw_stages;
    a.d_bhat_t = g.d_bhat_t;
    a.d_inv_perm = g.d_inv_perm;
    a.scale_dst = g.scale_dst;
    return a;
}

template <int BT, int MB, int F1_MODE>
static void launch_bluestein_bt() {
    const int P = g.spec_plan.P;
    const size_t smem = static_cast<size_t>(P + (P >> 3) + 4) * sizeof(cdf);
    const int n_pairs = (g.M + 1) / 2;
    dim3 t_blk(32, 8);
    dim3 t_grd((g.M + 31) / 32, (g.M + 31) / 32);

    // 1. Row Fwd + 8 boundaries
    bluestein_1d_kernel<BT, MB, F1_MODE><<<n_pairs + 4, BT, smem>>>(
        make_spec_args(0, false, nullptr, g.d_work0));

    transpose_df_kernel<<<t_grd, t_blk>>>(g.M, g.d_work0, g.d_work1);

    // 2. Col Fwd
    bluestein_1d_kernel<BT, MB, F1_MODE><<<n_pairs, BT, smem>>>(
        make_spec_args(1, false, g.d_work1, g.d_work0));

    // 3. Spectral Evolution in-place on d_work0
    spectral_evolve_2d_kernel<<<dim3((g.M + 31) / 32, (g.M + 7) / 8), dim3(32, 8)>>>(
        g.n0, g.M, g.d_work0, g.d_bdry_hat, g.d_sin2_half, g.d_sp1);

    // 4. Col Inv
    bluestein_1d_kernel<BT, MB, F1_MODE><<<n_pairs, BT, smem>>>(
        make_spec_args(1, false, g.d_work0, g.d_work1));

    transpose_df_kernel<<<t_grd, t_blk>>>(g.M, g.d_work1, g.d_work0);

    // 5. Row Inv -> d_buf1 (f32 interior) + d_work1 (df interior)
    bluestein_1d_kernel<BT, MB, F1_MODE><<<n_pairs, BT, smem>>>(
        make_spec_args(2, false, g.d_work0, g.d_work1));

    // 6. Final 1-step df stencil -> d_buf0 (f32 interior)
    dim3 s_blk(32, 8);
    dim3 s_grd((g.M + 31) / 32, (g.M + 15) / 16);
    final_stencil_df_kernel<<<s_grd, s_blk>>>(g.M, g.n1, g.d_work1, g.d_buf1, g.d_buf0);
}

static void launch_bluestein_pipeline() {
    const bool r5 = (g.spec_plan.F1 == 5);
    switch (g.spec_plan.BT) {
        case  128: launch_bluestein_bt< 128, 4, F1_ALL>(); break;
        case  256: launch_bluestein_bt< 256, 3, F1_ALL>(); break;
        case  512: if (r5) launch_bluestein_bt< 512, 2, F1_5>(); else launch_bluestein_bt< 512, 2, F1_13>(); break;
        case 1024: if (r5) launch_bluestein_bt<1024, 1, F1_5>(); else launch_bluestein_bt<1024, 1, F1_13>(); break;
        default: break;
    }
}

template <int BT, int MB, int F1_MODE>
static void launch_smooth_bt() {
    const int L = g.spec_plan.P;
    const int M = L - 1;
    const int LS_C = L + (L >> 3) + 4;
    const bool compact = (L == 4096 || L == 8192);
    const int extra_df = compact ? (2 * ((M + 1) / 2)) : (2 * M);
    const size_t smem = static_cast<size_t>(LS_C) * sizeof(cdf) + static_cast<size_t>(extra_df) * sizeof(df);
    const int n_pairs = (M + 1) / 2;
    dim3 t_blk(32, 8);
    dim3 t_grd((M + 31) / 32, (M + 31) / 32);

    smooth_1d_kernel<BT, MB, F1_MODE><<<n_pairs + 4, BT, smem>>>(
        make_spec_args(0, true, nullptr, g.d_work0));

    transpose_df_kernel<<<t_grd, t_blk>>>(M, g.d_work0, g.d_work1);

    smooth_1d_kernel<BT, MB, F1_MODE><<<n_pairs, BT, smem>>>(
        make_spec_args(1, true, g.d_work1, g.d_work0));

    spectral_evolve_2d_kernel<<<dim3((M + 31) / 32, (M + 7) / 8), dim3(32, 8)>>>(
        g.n0, M, g.d_work0, g.d_bdry_hat, g.d_sin2_half, g.d_sp1);

    smooth_1d_kernel<BT, MB, F1_MODE><<<n_pairs, BT, smem>>>(
        make_spec_args(1, true, g.d_work0, g.d_work1));

    transpose_df_kernel<<<t_grd, t_blk>>>(M, g.d_work1, g.d_work0);

    smooth_1d_kernel<BT, MB, F1_MODE><<<n_pairs, BT, smem>>>(
        make_spec_args(2, true, g.d_work0, g.d_work1));

    dim3 s_blk(32, 8);
    dim3 s_grd((M + 31) / 32, (M + 15) / 16);
    final_stencil_df_kernel<<<s_grd, s_blk>>>(M, g.n1, g.d_work1, g.d_buf1, g.d_buf0);
}

static void launch_smooth_pipeline() {
    const int L = g.spec_plan.P;
    const bool r5 = (g.spec_plan.F1 == 5);
    const int bt = (L >= 5120) ? 1024 : g.spec_plan.BT;
    switch (bt) {
        case  128: launch_smooth_bt< 128, 4, F1_ALL>(); break;
        case  256: launch_smooth_bt< 256, 3, F1_ALL>(); break;
        case  512: launch_smooth_bt< 512, 2, F1_13>(); break;
        case 1024: if (r5) launch_smooth_bt<1024, 1, F1_5>(); else launch_smooth_bt<1024, 1, F1_13>(); break;
        default: break;
    }
}

static void bailey_1d_pass(int total_rows, int in_stride, int in_offset,
                           bool in_is_f32, const void* d_in_raw,
                           df* d_out_df, bool write_f32_also, float* d_out_f32) {
    const BaileyPlan& bp = g.bailey_plan;
    const int R = bp.R, C = bp.C;
    const int R_MR8 = R / (bp.RF1 * bp.RF2);
    const int R_div_F1 = bp.RF2 * R_MR8;
    const int tw_off_f2 = (bp.RF1 > 1) ? (bp.RF1 - 1) * R_div_F1 : 0;
    const int r8_off_start = tw_off_f2 + ((bp.RF2 > 1) ? (bp.RF2 - 1) * R_MR8 : 0);
    const int r8_off_end = r8_off_start + (R_MR8 - 8);
    const int STRIDE_R = R + (R >> 3) + 1;
    const int total_pairs = (total_rows + 1) / 2;
    const int B = g.bailey_batch;

    const size_t smem_s13 = 8ULL * STRIDE_R * sizeof(cdf);
    const size_t smem_s4 = 4ULL * 2048 * sizeof(df);

    BaileyArgs ba{};
    ba.R = R; ba.RF1 = bp.RF1; ba.RF2 = bp.RF2; ba.R_MR8 = R_MR8;
    ba.C = C; ba.M = g.M; ba.total_rows = total_rows;
    ba.in_stride = in_stride; ba.in_offset = in_offset;
    ba.R_div_F1 = R_div_F1; ba.tw_off_f2 = tw_off_f2;
    ba.r8_off_start = r8_off_start; ba.r8_off_end = r8_off_end;
    ba.STRIDE = STRIDE_R;
    ba.fuse_r2 = (bp.RF1 == 1 && bp.RF2 == 2) ? 1 : 0;
    ba.d_in_raw = d_in_raw;
    ba.d_scratch = g.d_scratch;
    ba.d_Z = g.d_Z;
    ba.d_sin_k = g.d_sin_k;
    ba.d_chirp = g.d_chirp;
    ba.d_tw_R = g.d_tw_R;
    ba.d_tw_bailey = g.d_tw_bailey;

    for (int p0 = 0; p0 < total_pairs; p0 += B) {
        int cur_b = std::min(B, total_pairs - p0);
        ba.pair_start = p0;
        dim3 grd1(C / 8, cur_b);
        if (in_is_f32) {
            bailey_step1_kernel<true><<<grd1, 128, smem_s13>>>(ba);
        } else {
            bailey_step1_kernel<false><<<grd1, 128, smem_s13>>>(ba);
        }

        int batch_rows = cur_b * R;
        if (C == 128) {
            size_t smem_s2 = 16ULL * (128 + 16 + 1) * sizeof(cdf);
            bailey_step2_kernel<128, 2, 16, 16><<<(batch_rows + 15) / 16, 256, smem_s2>>>(
                R, batch_rows, g.d_scratch, g.d_tw_C, g.d_bhat_bailey, g.d_tw_bailey);
        } else if (C == 256) {
            size_t smem_s2 = 8ULL * (256 + 32 + 1) * sizeof(cdf);
            bailey_step2_kernel<256, 4, 32, 8><<<(batch_rows + 7) / 8, 256, smem_s2>>>(
                R, batch_rows, g.d_scratch, g.d_tw_C, g.d_bhat_bailey, g.d_tw_bailey);
        } else {
            size_t smem_s2 = 4ULL * (512 + 64 + 1) * sizeof(cdf);
            bailey_step2_kernel<512, 1, 64, 4><<<(batch_rows + 3) / 4, 256, smem_s2>>>(
                R, batch_rows, g.d_scratch, g.d_tw_C, g.d_bhat_bailey, g.d_tw_bailey);
        }

        bailey_step3_kernel<<<grd1, 128, smem_s13>>>(ba);

        if (write_f32_also) {
            bailey_step4_extract_kernel<true><<<cur_b, 512, smem_s4>>>(
                g.M, total_rows, g.n1, p0, g.d_Z, d_out_df, d_out_f32, g.scale_dst);
        } else {
            bailey_step4_extract_kernel<false><<<cur_b, 512, smem_s4>>>(
                g.M, total_rows, g.n1, p0, g.d_Z, d_out_df, nullptr, g.scale_dst);
        }
    }
}

static void launch_bailey_pipeline() {
    dim3 t_blk(32, 8);
    dim3 t_grd((g.M + 31) / 32, (g.M + 31) / 32);

    pack_boundaries_kernel<<<(g.M + 255) / 256, 256>>>(g.M, g.n1, g.d_buf0, g.d_buf1, g.d_bdry_raw);
    bailey_1d_pass(8, g.M, 0, true, g.d_bdry_raw, g.d_bdry_hat, false, nullptr);

    bailey_1d_pass(g.M, g.n1, g.n1 + 1, true, g.d_buf0, g.d_work0, false, nullptr);
    transpose_df_kernel<<<t_grd, t_blk>>>(g.M, g.d_work0, g.d_work1);
    bailey_1d_pass(g.M, g.M, 0, false, g.d_work1, g.d_work0, false, nullptr);

    spectral_evolve_2d_kernel<<<dim3((g.M + 31) / 32, (g.M + 7) / 8), dim3(32, 8)>>>(
        g.n0, g.M, g.d_work0, g.d_bdry_hat, g.d_sin2_half, g.d_sp1);

    bailey_1d_pass(g.M, g.M, 0, false, g.d_work0, g.d_work1, false, nullptr);
    transpose_df_kernel<<<t_grd, t_blk>>>(g.M, g.d_work1, g.d_work0);
    bailey_1d_pass(g.M, g.M, 0, false, g.d_work0, g.d_work1, true, g.d_buf1);

    dim3 s_blk(32, 8);
    dim3 s_grd((g.M + 31) / 32, (g.M + 15) / 16);
    final_stencil_df_kernel<<<s_grd, s_blk>>>(g.M, g.n1, g.d_work1, g.d_buf1, g.d_buf0);
}

static int setup_spectral_common(int M) {
    int L = M + 1;
    g.M = M;
    g.L = L;
    g.scale_dst = df_from_double(sqrt(2.0 / static_cast<double>(L)));

    std::vector<df> h_sin(L), h_sin2_half(M), h_sp1(M);
    double sc = sqrt(2.0 / static_cast<double>(L));
    for (int k = 0; k < L; ++k) {
        h_sin[k] = df_from_double(sin(PI * static_cast<double>(k) / static_cast<double>(L)));
    }
    for (int p = 0; p < M; ++p) {
        double s_half = sin(PI * static_cast<double>(p + 1) / (2.0 * static_cast<double>(L)));
        h_sin2_half[p] = df_from_double(0.8 * s_half * s_half);
        h_sp1[p] = df_from_double(0.2 * sc * sin(PI * static_cast<double>(p + 1) / static_cast<double>(L)));
    }

    size_t work_bytes = static_cast<size_t>(M) * M * sizeof(df);
    CUDA_OK(cudaMalloc(&g.d_work0, work_bytes));
    CUDA_OK(cudaMalloc(&g.d_work1, work_bytes));
    CUDA_OK(cudaMalloc(&g.d_bdry_hat, 8ULL * M * sizeof(df)));
    CUDA_OK(cudaMalloc(&g.d_sin_k, static_cast<size_t>(L) * sizeof(df)));
    CUDA_OK(cudaMalloc(&g.d_sin2_half, static_cast<size_t>(M) * sizeof(df)));
    CUDA_OK(cudaMalloc(&g.d_sp1, static_cast<size_t>(M) * sizeof(df)));

    CUDA_OK(cudaMemcpy(g.d_sin_k, h_sin.data(), static_cast<size_t>(L) * sizeof(df), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g.d_sin2_half, h_sin2_half.data(), static_cast<size_t>(M) * sizeof(df), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g.d_sp1, h_sp1.data(), static_cast<size_t>(M) * sizeof(df), cudaMemcpyHostToDevice));
    return 0;
}

static int setup_smooth(const SpecPlan& sp) {
    g.spec_plan = sp;
    if (setup_spectral_common(g.n1 - 2) != 0) return 1;

    auto tw = build_stage_twiddles(sp.P, sp.F1, sp.F2);
    auto perm = build_dif_perm(sp.P, sp.F1, sp.F2);
    std::vector<int> inv_perm(sp.P);
    for (int i = 0; i < sp.P; ++i) inv_perm[perm[i]] = i;

    CUDA_OK(cudaMalloc(&g.d_tw_stages, std::max<size_t>(1, tw.size()) * sizeof(cdf)));
    if (!tw.empty()) {
        CUDA_OK(cudaMemcpy(g.d_tw_stages, tw.data(), tw.size() * sizeof(cdf), cudaMemcpyHostToDevice));
    }
    CUDA_OK(cudaMalloc(&g.d_inv_perm, static_cast<size_t>(sp.P) * sizeof(int)));
    CUDA_OK(cudaMemcpy(g.d_inv_perm, inv_perm.data(), static_cast<size_t>(sp.P) * sizeof(int), cudaMemcpyHostToDevice));

    int L = sp.P;
    int M = L - 1;
    int LS_C = L + (L >> 3) + 4;
    bool compact = (L == 4096 || L == 8192);
    int extra_df = compact ? (2 * ((M + 1) / 2)) : (2 * M);
    int smem = static_cast<int>(static_cast<size_t>(LS_C) * sizeof(cdf) + static_cast<size_t>(extra_df) * sizeof(df));
    if (smem > 48 * 1024) {
        cudaError_t e = cudaSuccess;
        bool r5 = (sp.F1 == 5);
        int bt = (L >= 5120) ? 1024 : sp.BT;
        if (bt == 128)      e = cudaFuncSetAttribute(smooth_1d_kernel< 128, 4, F1_ALL>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        else if (bt == 256) e = cudaFuncSetAttribute(smooth_1d_kernel< 256, 3, F1_ALL>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        else if (bt == 512) e = cudaFuncSetAttribute(smooth_1d_kernel< 512, 2, F1_13>,  cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        else                e = r5 ? cudaFuncSetAttribute(smooth_1d_kernel<1024, 1, F1_5>,  cudaFuncAttributeMaxDynamicSharedMemorySize, smem)
                                   : cudaFuncSetAttribute(smooth_1d_kernel<1024, 1, F1_13>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return 1;
    }
    return 0;
}

static int setup_bluestein(const SpecPlan& sp) {
    g.spec_plan = sp;
    if (setup_spectral_common(g.n1 - 2) != 0) return 1;

    int L = g.L, P = sp.P;
    std::vector<cdf> h_chirp(L);
    std::vector<double> br(P, 0.0), bi(P, 0.0);
    for (int k = 0; k < L; ++k) {
        long long kk = (static_cast<long long>(k) * k) % (2LL * L);
        double ang = -PI * static_cast<double>(kk) / static_cast<double>(L);
        h_chirp[k] = {df_from_double(cos(ang)), df_from_double(sin(ang))};
        br[k] = cos(-ang);
        bi[k] = sin(-ang);
        if (k > 0) { br[P - k] = br[k]; bi[P - k] = bi[k]; }
    }

    auto tw = build_stage_twiddles(P, sp.F1, sp.F2);
    host_dft_dif_exact(P, sp.F1, sp.F2, br, bi);

    int n_in = P / 8;
    double invP = 1.0 / static_cast<double>(P);
    std::vector<cdf> h_bhat_t(P);
    for (int b = 0; b < n_in; ++b) {
        for (int r = 0; r < 8; ++r) {
            int k = b * 8 + r;
            h_bhat_t[r * n_in + b] = {df_from_double(br[k] * invP), df_from_double(bi[k] * invP)};
        }
    }

    CUDA_OK(cudaMalloc(&g.d_chirp, static_cast<size_t>(L) * sizeof(cdf)));
    CUDA_OK(cudaMalloc(&g.d_tw_stages, std::max<size_t>(1, tw.size()) * sizeof(cdf)));
    CUDA_OK(cudaMalloc(&g.d_bhat_t, static_cast<size_t>(P) * sizeof(cdf)));

    CUDA_OK(cudaMemcpy(g.d_chirp, h_chirp.data(), static_cast<size_t>(L) * sizeof(cdf), cudaMemcpyHostToDevice));
    if (!tw.empty()) {
        CUDA_OK(cudaMemcpy(g.d_tw_stages, tw.data(), tw.size() * sizeof(cdf), cudaMemcpyHostToDevice));
    }
    CUDA_OK(cudaMemcpy(g.d_bhat_t, h_bhat_t.data(), static_cast<size_t>(P) * sizeof(cdf), cudaMemcpyHostToDevice));

    int smem = static_cast<int>(static_cast<size_t>(P + (P >> 3) + 4) * sizeof(cdf));
    if (smem > 48 * 1024) {
        cudaError_t e = cudaSuccess;
        bool r5 = (sp.F1 == 5);
        if (sp.BT == 128)      e = cudaFuncSetAttribute(bluestein_1d_kernel< 128, 4, F1_ALL>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        else if (sp.BT == 256) e = cudaFuncSetAttribute(bluestein_1d_kernel< 256, 3, F1_ALL>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        else if (sp.BT == 512) e = r5 ? cudaFuncSetAttribute(bluestein_1d_kernel< 512, 2, F1_5>,  cudaFuncAttributeMaxDynamicSharedMemorySize, smem)
                                      : cudaFuncSetAttribute(bluestein_1d_kernel< 512, 2, F1_13>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        else                   e = r5 ? cudaFuncSetAttribute(bluestein_1d_kernel<1024, 1, F1_5>,  cudaFuncAttributeMaxDynamicSharedMemorySize, smem)
                                      : cudaFuncSetAttribute(bluestein_1d_kernel<1024, 1, F1_13>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return 1;
    }
    return 0;
}

static int setup_bailey(const BaileyPlan& bp) {
    g.bailey_plan = bp;
    if (setup_spectral_common(g.n1 - 2) != 0) return 1;

    int L = g.L, P = bp.P, R = bp.R, C = bp.C;
    auto perm_R = build_dif_perm(R, bp.RF1, bp.RF2);
    auto perm_C = build_dif_perm(C, 1, bp.CF2);
    auto tw_R = build_stage_twiddles(R, bp.RF1, bp.RF2);
    auto tw_C = build_stage_twiddles(C, 1, bp.CF2);

    std::vector<cdf> h_chirp(L), h_tw_bailey(static_cast<size_t>(R) * C), h_bhat(static_cast<size_t>(R) * C);
    std::vector<double> br(P, 0.0), bi(P, 0.0);
    for (int k = 0; k < L; ++k) {
        long long kk = (static_cast<long long>(k) * k) % (2LL * L);
        double ang = -PI * static_cast<double>(kk) / static_cast<double>(L);
        h_chirp[k] = {df_from_double(cos(ang)), df_from_double(sin(ang))};
        br[k] = cos(-ang);
        bi[k] = sin(-ang);
        if (k > 0) { br[P - k] = br[k]; bi[P - k] = bi[k]; }
    }

    for (int kr_rev = 0; kr_rev < R; ++kr_rev) {
        int kr = perm_R[kr_rev];
        for (int c = 0; c < C; ++c) {
            double a = -2.0 * PI * static_cast<double>(static_cast<long long>(kr) * c) / static_cast<double>(P);
            h_tw_bailey[static_cast<size_t>(kr_rev) * C + c] = {df_from_double(cos(a)), df_from_double(sin(a))};
        }
    }

    host_fft_natural(P, br, bi);
    double invP = 1.0 / static_cast<double>(P);
    for (int kr_rev = 0; kr_rev < R; ++kr_rev) {
        int kr = perm_R[kr_rev];
        for (int kc_rev = 0; kc_rev < C; ++kc_rev) {
            int kc = perm_C[kc_rev];
            int freq = kc * R + kr;
            h_bhat[static_cast<size_t>(kr_rev) * C + kc_rev] = {
                df_from_double(br[freq] * invP), df_from_double(bi[freq] * invP)
            };
        }
    }

    // Keep d_scratch (<=38MB) + d_Z (<=19MB) + in/out rows (<38MB) comfortably inside B300's 126.5MB L2!
    size_t bytes_per_pair = static_cast<size_t>(P) * sizeof(cdf);
    int B = static_cast<int>((38ULL * 1024 * 1024) / bytes_per_pair);
    B = std::max(4, std::min(B, (g.M + 1) / 2));
    g.bailey_batch = B;

    CUDA_OK(cudaMalloc(&g.d_chirp, static_cast<size_t>(L) * sizeof(cdf)));
    CUDA_OK(cudaMalloc(&g.d_tw_R, std::max<size_t>(1, tw_R.size()) * sizeof(cdf)));
    CUDA_OK(cudaMalloc(&g.d_tw_C, std::max<size_t>(1, tw_C.size()) * sizeof(cdf)));
    CUDA_OK(cudaMalloc(&g.d_tw_bailey, static_cast<size_t>(R) * C * sizeof(cdf)));
    CUDA_OK(cudaMalloc(&g.d_bhat_bailey, static_cast<size_t>(R) * C * sizeof(cdf)));
    CUDA_OK(cudaMalloc(&g.d_scratch, static_cast<size_t>(B) * P * sizeof(cdf)));
    CUDA_OK(cudaMalloc(&g.d_Z, static_cast<size_t>(B) * L * sizeof(cdf)));
    CUDA_OK(cudaMalloc(&g.d_bdry_raw, 8ULL * g.M * sizeof(float)));

    CUDA_OK(cudaMemcpy(g.d_chirp, h_chirp.data(), static_cast<size_t>(L) * sizeof(cdf), cudaMemcpyHostToDevice));
    if (!tw_R.empty()) CUDA_OK(cudaMemcpy(g.d_tw_R, tw_R.data(), tw_R.size() * sizeof(cdf), cudaMemcpyHostToDevice));
    if (!tw_C.empty()) CUDA_OK(cudaMemcpy(g.d_tw_C, tw_C.data(), tw_C.size() * sizeof(cdf), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g.d_tw_bailey, h_tw_bailey.data(), static_cast<size_t>(R) * C * sizeof(cdf), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g.d_bhat_bailey, h_bhat.data(), static_cast<size_t>(R) * C * sizeof(cdf), cudaMemcpyHostToDevice));

    int smem_s13 = static_cast<int>(8ULL * (R + (R >> 3) + 1) * sizeof(cdf));
    if (smem_s13 > 48 * 1024) {
        CUDA_OK(cudaFuncSetAttribute(bailey_step1_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_s13));
        CUDA_OK(cudaFuncSetAttribute(bailey_step1_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_s13));
        CUDA_OK(cudaFuncSetAttribute(bailey_step3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_s13));
    }
    int smem_s4 = static_cast<int>(4ULL * 2048 * sizeof(df));
    CUDA_OK(cudaFuncSetAttribute(bailey_step4_extract_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_s4));
    CUDA_OK(cudaFuncSetAttribute(bailey_step4_extract_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_s4));
    return 0;
}

static int setup_direct_2d() {
    CUDA_OK(cudaMalloc(&g.d_alt, g.buf_bytes));
    if (set_direct2d_attr<96, 96, 16, 32, 16>() != cudaSuccess) return 1;
    if (set_direct2d_attr<64, 64, 12, 32, 8>() != cudaSuccess) return 1;
    if (set_direct2d_attr<64, 64, 8, 32, 8>() != cudaSuccess) return 1;
    if (set_direct2d_attr<64, 32, 4, 32, 8>() != cudaSuccess) return 1;
    if (set_direct2d_attr<64, 32, 2, 32, 8>() != cudaSuccess) return 1;
    return 0;
}

} // anonymous namespace

// ============================================================================
// Five-Phase C-Linkage Entry Points (pb-solve-v3)
// ============================================================================

extern "C" {

int pb_setup(const pb_problem_shape *shape) {
    if (g.initialized) free_all();
    if (!shape || shape->abi_version != PB_SOLVE_ABI_VERSION ||
        shape->dimension_count != 2 || shape->scalar_count != 0 ||
        shape->buffer_count != 2 || !shape->dimensions || !shape->buffers) {
        return 1;
    }
    int64_t dn0 = shape->dimensions[0];
    int64_t dn1 = shape->dimensions[1];
    if (dn0 <= 0 || dn0 > INT_MAX || dn1 <= 0 || dn1 > INT_MAX) return 1;
    if (!check_buffer_shape(&shape->buffers[0], dn1) ||
        !check_buffer_shape(&shape->buffers[1], dn1)) {
        return 1;
    }

    g.n0 = static_cast<int>(dn0);
    g.n1 = static_cast<int>(dn1);
    g.elem_count = static_cast<size_t>(dn1) * static_cast<size_t>(dn1);
    g.buf_bytes = g.elem_count * sizeof(float);

    if (cudaMalloc(&g.d_buf0, g.buf_bytes) != cudaSuccess ||
        cudaMalloc(&g.d_buf1, g.buf_bytes) != cudaSuccess) {
        free_all();
        return 1;
    }

    if (g.n1 <= 2) {
        g.mode = MODE_NOOP;
        g.initialized = true;
        return cudaDeviceSynchronize() == cudaSuccess ? 0 : 1;
    }

    const int M = g.n1 - 2;
    const int L = M + 1;

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t spectral_need = 2ULL * static_cast<size_t>(M) * M * sizeof(df) + 128ULL * 1024 * 1024;
    bool spectral_fits = (free_mem > spectral_need + 256ULL * 1024 * 1024);

    int smooth_idx = -1;
    constexpr int N_SINGLE = static_cast<int>(sizeof(SINGLE_PLANS) / sizeof(SINGLE_PLANS[0]));
    for (int i = 0; i < N_SINGLE; ++i) {
        if (SINGLE_PLANS[i].P == L && L <= 8192) { smooth_idx = i; break; }
    }
    int bluestein_idx = -1;
    for (int i = 0; i < N_SINGLE; ++i) {
        if (SINGLE_PLANS[i].P >= 2 * L - 1) { bluestein_idx = i; break; }
    }
    int bailey_idx = -1;
    if (bluestein_idx < 0) {
        constexpr int N_BAILEY = static_cast<int>(sizeof(BAILEY_PLANS) / sizeof(BAILEY_PLANS[0]));
        for (int i = 0; i < N_BAILEY; ++i) {
            if (BAILEY_PLANS[i].P >= 2 * L - 1) { bailey_idx = i; break; }
        }
    }

    // Calibrated B300 performance model (includes per-pass launch overhead!)
    if ((g.n1 <= 48 && g.n0 <= 64) || (g.n1 <= 96 && g.n0 <= 16)) {
        g.mode = MODE_DIRECT_SMEM1;
        size_t smem = 2ULL * g.n1 * (g.n1 + 1) * sizeof(float);
        if (smem > 48 * 1024) {
            if (cudaFuncSetAttribute(direct_smem1_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)) != cudaSuccess) {
                free_all();
                return 1;
            }
        }
    } else if (!spectral_fits || (smooth_idx < 0 && bluestein_idx < 0 && bailey_idx < 0)) {
        g.mode = MODE_DIRECT_2D;
    } else {
        double k_step = (g.n1 >= 256) ? 16.0 : 8.0;
        double passes = ceil((2.0 * g.n0) / k_step);
        double direct_rate = (g.n1 < 1800) ? 1.8e8 : 3.5e8;
        double est_direct_ms = passes * 0.018 + (static_cast<double>(g.n0) * M * M) / direct_rate;

        double est_spec_ms = 1e18;
        ExecMode cand_mode = MODE_DIRECT_2D;
        if (smooth_idx >= 0) {
            int Lp = SINGLE_PLANS[smooth_idx].P;
            est_spec_ms = 0.06 + 6.8e-9 * static_cast<double>(M) * Lp * log2(static_cast<double>(Lp));
            cand_mode = MODE_SPECTRAL_SMOOTH;
        } else if (bluestein_idx >= 0) {
            int Pp = SINGLE_PLANS[bluestein_idx].P;
            est_spec_ms = 0.06 + 7.2e-9 * static_cast<double>(M) * Pp * log2(static_cast<double>(Pp));
            cand_mode = MODE_SPECTRAL_BLUESTEIN;
        } else if (bailey_idx >= 0) {
            int Pp = BAILEY_PLANS[bailey_idx].P;
            est_spec_ms = 0.25 + 9.2e-9 * static_cast<double>(M) * Pp * log2(static_cast<double>(Pp));
            cand_mode = MODE_SPECTRAL_BAILEY;
        }

        g.mode = (est_spec_ms < est_direct_ms) ? cand_mode : MODE_DIRECT_2D;
    }

    int rc = 0;
    if (g.mode == MODE_DIRECT_2D) {
        rc = setup_direct_2d();
    } else if (g.mode == MODE_SPECTRAL_SMOOTH) {
        rc = setup_smooth(SINGLE_PLANS[smooth_idx]);
    } else if (g.mode == MODE_SPECTRAL_BLUESTEIN) {
        rc = setup_bluestein(SINGLE_PLANS[bluestein_idx]);
    } else if (g.mode == MODE_SPECTRAL_BAILEY) {
        rc = setup_bailey(BAILEY_PLANS[bailey_idx]);
    }

    if (rc != 0 || cudaDeviceSynchronize() != cudaSuccess) {
        free_all();
        return 1;
    }
    g.initialized = true;
    return 0;
}

int pb_upload(const pb_problem *problem, pb_buffer *buffers) {
    if (!g.initialized || !problem || !buffers) return 1;
    if (problem->abi_version != PB_SOLVE_ABI_VERSION ||
        problem->dimension_count != 2 || problem->scalar_count != 0 ||
        problem->buffer_count != 2 || !problem->dimensions) {
        return 1;
    }
    if (problem->dimensions[0] != g.n0 || problem->dimensions[1] != g.n1) return 1;
    if (!check_buffer_live(&buffers[0], g.n1) || !check_buffer_live(&buffers[1], g.n1)) return 1;

    uintptr_t s0 = reinterpret_cast<uintptr_t>(buffers[0].data);
    uintptr_t s1 = reinterpret_cast<uintptr_t>(buffers[1].data);
    if (buffers[0].nbytes > UINTPTR_MAX - s0 || buffers[1].nbytes > UINTPTR_MAX - s1) return 1;
    if (s0 < s1 + buffers[1].nbytes && s1 < s0 + buffers[0].nbytes) return 1;

    CUDA_OK(cudaMemcpy(g.d_buf0, buffers[0].data, g.buf_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g.d_buf1, buffers[1].data, g.buf_bytes, cudaMemcpyHostToDevice));
    return 0;
}

int pb_compute(void) {
    if (!g.initialized) return 1;

    if (g.mode == MODE_NOOP) {
        return 0;
    } else if (g.mode == MODE_DIRECT_SMEM1) {
        size_t smem = 2ULL * g.n1 * (g.n1 + 1) * sizeof(float);
        dim3 blk = (g.n1 <= 16) ? dim3(16, 8) : ((g.n1 <= 32) ? dim3(32, 8) : dim3(32, 16));
        direct_smem1_kernel<<<1, blk, smem>>>(g.n0, g.n1, g.d_buf0, g.d_buf1);
    } else if (g.mode == MODE_DIRECT_2D) {
        if (run_direct_2d() != 0) return 1;
    } else if (g.mode == MODE_SPECTRAL_SMOOTH) {
        launch_smooth_pipeline();
    } else if (g.mode == MODE_SPECTRAL_BLUESTEIN) {
        launch_bluestein_pipeline();
    } else if (g.mode == MODE_SPECTRAL_BAILEY) {
        launch_bailey_pipeline();
    } else {
        return 1;
    }

    if (cudaGetLastError() != cudaSuccess) return 1;
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    return 0;
}

int pb_download(pb_buffer *buffers) {
    if (!g.initialized || !buffers) return 1;
    if (!check_buffer_live(&buffers[0], g.n1) || !check_buffer_live(&buffers[1], g.n1)) return 1;

    uintptr_t s0 = reinterpret_cast<uintptr_t>(buffers[0].data);
    uintptr_t s1 = reinterpret_cast<uintptr_t>(buffers[1].data);
    if (buffers[0].nbytes > UINTPTR_MAX - s0 || buffers[1].nbytes > UINTPTR_MAX - s1) return 1;
    if (s0 < s1 + buffers[1].nbytes && s1 < s0 + buffers[0].nbytes) return 1;

    CUDA_OK(cudaMemcpy(buffers[0].data, g.d_buf0, g.buf_bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(buffers[1].data, g.d_buf1, g.buf_bytes, cudaMemcpyDeviceToHost));
    return 0;
}

int pb_teardown(void) {
    free_all();
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

} // extern "C"
