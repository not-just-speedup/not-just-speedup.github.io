#include "interface.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <climits>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace {

static constexpr double kPi = 3.141592653589793238462643383279502884;

static inline int align8(int x) { return (x + 7) & ~7; }
static inline int align16(int x) { return (x + 15) & ~15; }

#define CUDA_TRY(expr) do { cudaError_t _err = (expr); if (_err != cudaSuccess) return 1; } while (0)
#define CUBLAS_TRY(expr) do { cublasStatus_t _st = (expr); if (_st != CUBLAS_STATUS_SUCCESS) return 1; } while (0)

// ============================================================================
// Double-Float (float2) Exact Arithmetic Helpers
// ============================================================================
__device__ __forceinline__ float2 two_sum(float a, float b) {
    float s = __fadd_rn(a, b);
    float v = __fsub_rn(s, a);
    float e = __fadd_rn(__fsub_rn(a, __fsub_rn(s, v)), __fsub_rn(b, v));
    return make_float2(s, e);
}

__device__ __forceinline__ float2 quick_two_sum(float a, float b) {
    float s = __fadd_rn(a, b);
    float e = __fadd_rn(b, __fsub_rn(a, s));
    return make_float2(s, e);
}

__device__ __forceinline__ float2 df_add_f(float2 a, float b) {
    float2 t = two_sum(a.x, b);
    return quick_two_sum(t.x, __fadd_rn(t.y, a.y));
}

__device__ __forceinline__ float2 df_add(float2 a, float2 b) {
    float2 t = two_sum(a.x, b.x);
    return quick_two_sum(t.x, __fadd_rn(t.y, __fadd_rn(a.y, b.y)));
}

__device__ __forceinline__ float2 df_sub(float2 a, float2 b) {
    return df_add(a, make_float2(-b.x, -b.y));
}

__device__ __forceinline__ float2 df_mul(float2 a, float2 b) {
    float p = __fmul_rn(a.x, b.x);
    float e = __fmaf_rn(a.x, b.x, -p);
    float lo = __fmaf_rn(a.x, b.y, __fmaf_rn(a.y, b.x, e));
    return quick_two_sum(p, lo);
}

__device__ __forceinline__ float2 df_fma_lazy(float2 a, float2 b, float2 acc) {
    float p = __fmul_rn(a.x, b.x);
    float e = __fmaf_rn(a.x, b.x, -p);
    float pl = __fmaf_rn(a.x, b.y, __fmaf_rn(a.y, b.x, e));
    float s = __fadd_rn(acc.x, p);
    float v = __fsub_rn(s, acc.x);
    float se = __fadd_rn(__fsub_rn(acc.x, __fsub_rn(s, v)), __fsub_rn(p, v));
    return make_float2(s, __fadd_rn(acc.y, __fadd_rn(se, pl)));
}

__device__ __forceinline__ float2 d2df(double x) {
    float hi = static_cast<float>(x);
    return make_float2(hi, static_cast<float>(x - static_cast<double>(hi)));
}

// ============================================================================
// PATH 1: Direct Stencil Kernels (Small n0 / Tiny n1 / OOM Fallback)
// ============================================================================
__device__ __forceinline__ float stencil7_exact(float c, float xm, float xp,
                                                float ym, float yp,
                                                float zm, float zp) {
    float dx = (xp - 2.0f * c) + xm;
    float dy = (yp - 2.0f * c) + ym;
    float dz = (zp - 2.0f * c) + zm;
    return fmaf(0.125f, (dx + dy) + dz, c);
}

__global__ void stencil_coop_l2_kernel(int n0, int n1,
                                       float* __restrict__ b0,
                                       float* __restrict__ b1) {
    cg::grid_group grid = cg::this_grid();
    int m = n1 - 2;
    size_t s1 = static_cast<size_t>(n1);
    size_t s2 = s1 * s1;
    size_t nrows = static_cast<size_t>(m) * m;

    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    size_t warp_id = tid >> 5;
    int lane = static_cast<int>(tid & 31);
    size_t nwarps = stride >> 5;

    for (int t = 0; t < n0; ++t) {
        for (size_t row = warp_id; row < nrows; row += nwarps) {
            int i = static_cast<int>(row / m) + 1;
            int j = static_cast<int>(row - static_cast<size_t>(i - 1) * m) + 1;
            size_t base = static_cast<size_t>(i) * s2 + static_cast<size_t>(j) * s1;
            for (int k = lane + 1; k < n1 - 1; k += 32) {
                size_t p = base + k;
                float c = __ldg(b0 + p);
                b1[p] = stencil7_exact(c, __ldg(b0 + p - 1), __ldg(b0 + p + 1),
                                       __ldg(b0 + p - s1), __ldg(b0 + p + s1),
                                       __ldg(b0 + p - s2), __ldg(b0 + p + s2));
            }
        }
        grid.sync();
        for (size_t row = warp_id; row < nrows; row += nwarps) {
            int i = static_cast<int>(row / m) + 1;
            int j = static_cast<int>(row - static_cast<size_t>(i - 1) * m) + 1;
            size_t base = static_cast<size_t>(i) * s2 + static_cast<size_t>(j) * s1;
            for (int k = lane + 1; k < n1 - 1; k += 32) {
                size_t p = base + k;
                float c = __ldg(b1 + p);
                b0[p] = stencil7_exact(c, __ldg(b1 + p - 1), __ldg(b1 + p + 1),
                                       __ldg(b1 + p - s1), __ldg(b1 + p + s1),
                                       __ldg(b1 + p - s2), __ldg(b1 + p + s2));
            }
        }
        grid.sync();
    }
}

__global__ void stencil_1step_2j_4k_kernel(int n1, int z_chunk,
                                           const float* __restrict__ in,
                                           float* __restrict__ out) {
    int k0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4 + 1;
    int j0 = (blockIdx.y * blockDim.y + threadIdx.y) * 2 + 1;
    int i0 = blockIdx.z * z_chunk + 1;
    if (k0 >= n1 - 1 || j0 >= n1 - 1) return;
    int i1 = min(i0 + z_chunk, n1 - 1);
    size_t s1 = static_cast<size_t>(n1);
    size_t s2 = s1 * s1;

    if (k0 + 3 < n1 - 1 && j0 + 1 < n1 - 1) {
        size_t b0 = static_cast<size_t>(i0 - 1) * s2 + static_cast<size_t>(j0) * s1 + k0;
        size_t b1 = b0 + s1;
        float pa0 = __ldg(in + b0), pa1 = __ldg(in + b0 + 1), pa2 = __ldg(in + b0 + 2), pa3 = __ldg(in + b0 + 3);
        float pb0 = __ldg(in + b1), pb1 = __ldg(in + b1 + 1), pb2 = __ldg(in + b1 + 2), pb3 = __ldg(in + b1 + 3);
        float ca0 = __ldg(in + b0 + s2), ca1 = __ldg(in + b0 + s2 + 1), ca2 = __ldg(in + b0 + s2 + 2), ca3 = __ldg(in + b0 + s2 + 3);
        float cb0 = __ldg(in + b1 + s2), cb1 = __ldg(in + b1 + s2 + 1), cb2 = __ldg(in + b1 + s2 + 2), cb3 = __ldg(in + b1 + s2 + 3);

        for (int i = i0; i < i1; ++i) {
            size_t p0 = static_cast<size_t>(i) * s2 + static_cast<size_t>(j0) * s1 + k0;
            size_t p1 = p0 + s1;
            float na0 = __ldg(in + p0 + s2), na1 = __ldg(in + p0 + s2 + 1), na2 = __ldg(in + p0 + s2 + 2), na3 = __ldg(in + p0 + s2 + 3);
            float nb0 = __ldg(in + p1 + s2), nb1 = __ldg(in + p1 + s2 + 1), nb2 = __ldg(in + p1 + s2 + 2), nb3 = __ldg(in + p1 + s2 + 3);
            float xma = __ldg(in + p0 - 1), xpa = __ldg(in + p0 + 4);
            float xmb = __ldg(in + p1 - 1), xpb = __ldg(in + p1 + 4);
            float ym0 = __ldg(in + p0 - s1), ym1 = __ldg(in + p0 - s1 + 1), ym2 = __ldg(in + p0 - s1 + 2), ym3 = __ldg(in + p0 - s1 + 3);
            float yp0 = __ldg(in + p1 + s1), yp1 = __ldg(in + p1 + s1 + 1), yp2 = __ldg(in + p1 + s1 + 2), yp3 = __ldg(in + p1 + s1 + 3);

            out[p0 + 0] = stencil7_exact(ca0, xma, ca1, ym0, cb0, pa0, na0);
            out[p0 + 1] = stencil7_exact(ca1, ca0, ca2, ym1, cb1, pa1, na1);
            out[p0 + 2] = stencil7_exact(ca2, ca1, ca3, ym2, cb2, pa2, na2);
            out[p0 + 3] = stencil7_exact(ca3, ca2, xpa, ym3, cb3, pa3, na3);

            out[p1 + 0] = stencil7_exact(cb0, xmb, cb1, ca0, yp0, pb0, nb0);
            out[p1 + 1] = stencil7_exact(cb1, cb0, cb2, ca1, yp1, pb1, nb1);
            out[p1 + 2] = stencil7_exact(cb2, cb1, cb3, ca2, yp2, pb2, nb2);
            out[p1 + 3] = stencil7_exact(cb3, cb2, xpb, ca3, yp3, pb3, nb3);

            pa0 = ca0; pa1 = ca1; pa2 = ca2; pa3 = ca3;
            pb0 = cb0; pb1 = cb1; pb2 = cb2; pb3 = cb3;
            ca0 = na0; ca1 = na1; ca2 = na2; ca3 = na3;
            cb0 = nb0; cb1 = nb1; cb2 = nb2; cb3 = nb3;
        }
    } else {
        int j_end = min(j0 + 2, n1 - 1);
        int k_end = min(k0 + 4, n1 - 1);
        for (int j = j0; j < j_end; ++j) {
            for (int k = k0; k < k_end; ++k) {
                size_t base = static_cast<size_t>(i0 - 1) * s2 + static_cast<size_t>(j) * s1 + k;
                float prev = __ldg(in + base), curr = __ldg(in + base + s2);
                for (int i = i0; i < i1; ++i) {
                    size_t p = static_cast<size_t>(i) * s2 + static_cast<size_t>(j) * s1 + k;
                    float next = __ldg(in + p + s2);
                    out[p] = stencil7_exact(curr, __ldg(in + p - 1), __ldg(in + p + 1),
                                            __ldg(in + p - s1), __ldg(in + p + s1), prev, next);
                    prev = curr; curr = next;
                }
            }
        }
    }
}

__global__ void stencil_1step_2j_aligned4_kernel(int n1, int z_chunk,
                                                 const float* __restrict__ in,
                                                 float* __restrict__ out) {
    int k4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int j0 = (blockIdx.y * blockDim.y + threadIdx.y) * 2 + 1;
    int i0 = blockIdx.z * z_chunk + 1;
    if (k4 >= n1 - 1 || j0 >= n1 - 1) return;
    int i1 = min(i0 + z_chunk, n1 - 1);
    size_t s1 = static_cast<size_t>(n1);
    size_t s2 = s1 * s1;

    if (k4 >= 1 && k4 + 3 <= n1 - 2 && j0 + 1 < n1 - 1) {
        size_t b0 = static_cast<size_t>(i0 - 1) * s2 + static_cast<size_t>(j0) * s1 + k4;
        size_t b1 = b0 + s1;
        float4 pa = __ldg(reinterpret_cast<const float4*>(in + b0));
        float4 pb = __ldg(reinterpret_cast<const float4*>(in + b1));
        float4 ca = __ldg(reinterpret_cast<const float4*>(in + b0 + s2));
        float4 cb = __ldg(reinterpret_cast<const float4*>(in + b1 + s2));
        for (int i = i0; i < i1; ++i) {
            size_t p0 = static_cast<size_t>(i) * s2 + static_cast<size_t>(j0) * s1 + k4;
            size_t p1 = p0 + s1;
            float4 na = __ldg(reinterpret_cast<const float4*>(in + p0 + s2));
            float4 nb = __ldg(reinterpret_cast<const float4*>(in + p1 + s2));
            float4 ym = __ldg(reinterpret_cast<const float4*>(in + p0 - s1));
            float4 yp = __ldg(reinterpret_cast<const float4*>(in + p1 + s1));
            float xma = __ldg(in + p0 - 1), xpa = __ldg(in + p0 + 4);
            float xmb = __ldg(in + p1 - 1), xpb = __ldg(in + p1 + 4);
            float4 ra, rb;
            ra.x = stencil7_exact(ca.x, xma,  ca.y, ym.x, cb.x, pa.x, na.x);
            ra.y = stencil7_exact(ca.y, ca.x, ca.z, ym.y, cb.y, pa.y, na.y);
            ra.z = stencil7_exact(ca.z, ca.y, ca.w, ym.z, cb.z, pa.z, na.z);
            ra.w = stencil7_exact(ca.w, ca.z, xpa,  ym.w, cb.w, pa.w, na.w);
            rb.x = stencil7_exact(cb.x, xmb,  cb.y, ca.x, yp.x, pb.x, nb.x);
            rb.y = stencil7_exact(cb.y, cb.x, cb.z, ca.y, yp.y, pb.y, nb.y);
            rb.z = stencil7_exact(cb.z, cb.y, cb.w, ca.z, yp.z, pb.z, nb.z);
            rb.w = stencil7_exact(cb.w, cb.z, xpb,  ca.w, yp.w, pb.w, nb.w);
            *reinterpret_cast<float4*>(out + p0) = ra;
            *reinterpret_cast<float4*>(out + p1) = rb;
            pa = ca; pb = cb; ca = na; cb = nb;
        }
    } else {
        int ks = max(1, k4), ke = min(n1 - 1, k4 + 4);
        int je = min(j0 + 2, n1 - 1);
        for (int j = j0; j < je; ++j) {
            for (int k = ks; k < ke; ++k) {
                size_t base = static_cast<size_t>(i0 - 1) * s2 + static_cast<size_t>(j) * s1 + k;
                float prev = __ldg(in + base), curr = __ldg(in + base + s2);
                for (int i = i0; i < i1; ++i) {
                    size_t p = static_cast<size_t>(i) * s2 + static_cast<size_t>(j) * s1 + k;
                    float next = __ldg(in + p + s2);
                    out[p] = stencil7_exact(curr, __ldg(in + p - 1), __ldg(in + p + 1),
                                            __ldg(in + p - s1), __ldg(in + p + s1), prev, next);
                    prev = curr; curr = next;
                }
            }
        }
    }
}

// ============================================================================
// PATH 2: Analytic Harmonic + Mode-Truncated Spectral Solver (O(1) in n0)
// ============================================================================
__global__ void max_abs_f32_kernel(size_t n, const float* __restrict__ x, float* __restrict__ d_max) {
    __shared__ float sm[256];
    float mx = 0.0f;
    size_t n4 = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n4; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        float4 v = x4[i];
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    if (blockIdx.x == 0 && threadIdx.x < (n & 3)) {
        mx = fmaxf(mx, fabsf(x[(n4 << 2) + threadIdx.x]));
    }
    sm[threadIdx.x] = mx;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s && sm[threadIdx.x + s] > sm[threadIdx.x]) sm[threadIdx.x] = sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(reinterpret_cast<int*>(d_max), __float_as_int(sm[0]));
}

__global__ void max_abs_df_kernel(size_t n, const float2* __restrict__ x, float* __restrict__ d_max) {
    __shared__ float sm[256];
    float mx = 0.0f;
    size_t n2 = n >> 1;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n2; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        float4 v = x4[i];
        mx = fmaxf(mx, fmaxf(fabsf(v.x) + fabsf(v.y), fabsf(v.z) + fabsf(v.w)));
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && (n & 1)) {
        float2 v = x[n - 1];
        mx = fmaxf(mx, fabsf(v.x) + fabsf(v.y));
    }
    sm[threadIdx.x] = mx;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s && sm[threadIdx.x + s] > sm[threadIdx.x]) sm[threadIdx.x] = sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(reinterpret_cast<int*>(d_max), __float_as_int(sm[0]));
}

__global__ void compute_pow2_scale_kernel(const float* __restrict__ d_max,
                                          float* __restrict__ d_scale,
                                          float safety_mult) {
    float mx = d_max[0] * safety_mult;
    if (mx == 0.0f) {
        d_scale[0] = 0.0f;
        d_scale[1] = 0.0f;
    } else {
        int e;
        frexpf(mx, &e);
        int ep = max(-124, min(124, e + 1));
        d_scale[0] = ldexpf(1.0f, -ep);
        d_scale[1] = ldexpf(1.0f,  ep);
    }
}

__global__ void build_dst_ozaki_stack_kernel(int L, int M, int K, int Mp, int Kp,
                                             int S, int S_B, int beta,
                                             int parity, int transpose,
                                             double sc, double inv_mu,
                                             __nv_bfloat16* __restrict__ d_Astack) {
    int rows = S * Mp;
    size_t total = static_cast<size_t>(M) * K;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int m = static_cast<int>(idx / K);
    int k = static_cast<int>(idx - static_cast<size_t>(m) * K);
    int mode_idx = transpose ? k : m;
    int sp_idx   = transpose ? m : k;
    int p = 2 * mode_idx + parity;
    double val = sc * sin(kPi * static_cast<double>(p + 1) * static_cast<double>(sp_idx + 1) / static_cast<double>(L));
    double r = val * inv_mu;
    __nv_bfloat16 qs[7];
    double mul = static_cast<double>(1 << beta);
    for (int s = 0; s < S; ++s) {
        double scaled = r * mul;
        double q = rint(scaled);
        qs[s] = __float2bfloat16(static_cast<float>(q));
        r = scaled - q;
    }
    for (int d = 0; d < S; ++d) {
        for (int blk = 0; blk <= d && blk < S_B; ++blk) {
            int s = d - blk;
            size_t pos = static_cast<size_t>(blk * Kp + k) * rows + static_cast<size_t>(d) * Mp + m;
            d_Astack[pos] = qs[s];
        }
    }
}

__global__ void build_dst_df64_kernel(int L, int n_sp, int n_modes, int parity,
                                      double sc, float2* __restrict__ out) {
    size_t total = static_cast<size_t>(n_sp) * n_modes;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int q = static_cast<int>(idx / n_modes);
    int r = static_cast<int>(idx - static_cast<size_t>(q) * n_modes);
    int p = 2 * r + parity;
    double val = sc * sin(kPi * static_cast<double>(p + 1) * static_cast<double>(q + 1) / static_cast<double>(L));
    out[idx] = d2df(val);
}

struct OzakiMatrixStacked {
    int M = 0, K = 0, Mp = 0, Kp = 0, S = 5, S_B = 5, beta = 7;
    double mu_A = 1.0;
    __nv_bfloat16* d_Astack = nullptr;

    static size_t estimate_bytes(int n_modes, int n_sp, bool transpose, bool high_prec, bool fp32_B) {
        int M_ = transpose ? n_sp : n_modes;
        int K_ = transpose ? n_modes : n_sp;
        if (M_ == 0 || K_ == 0) return 0;
        int Mp_ = align8(M_), Kp_ = align8(K_);
        int beta_ = (K_ <= 512) ? 7 : 6;
        int S_ = (K_ <= 512) ? (high_prec ? 6 : 5) : (high_prec ? 7 : 6);
        int SB_ = fp32_B ? ((beta_ == 7) ? 4 : 5) : S_;
        return static_cast<size_t>(S_) * Mp_ * SB_ * Kp_ * sizeof(__nv_bfloat16);
    }

    int init_dst(int L, int n_modes, int n_sp, int parity, bool transpose,
                 bool high_prec, bool fp32_B, cudaStream_t stream) {
        M = transpose ? n_sp : n_modes;
        K = transpose ? n_modes : n_sp;
        if (M == 0 || K == 0) return 0;
        Mp = align8(M);
        Kp = align8(K);
        if (K <= 512) { S = high_prec ? 6 : 5; beta = 7; }
        else          { S = high_prec ? 7 : 6; beta = 6; }
        S_B = fp32_B ? ((beta == 7) ? 4 : 5) : S;

        double sc = sqrt(2.0 / static_cast<double>(L));
        int e;
        frexp(sc, &e);
        mu_A = ldexp(1.0, e + 1);
        double inv_mu = 1.0 / mu_A;

        size_t sz = static_cast<size_t>(S) * Mp * S_B * Kp;
        CUDA_TRY(cudaMalloc(&d_Astack, sz * sizeof(__nv_bfloat16)));
        CUDA_TRY(cudaMemsetAsync(d_Astack, 0, sz * sizeof(__nv_bfloat16), stream));
        size_t total = static_cast<size_t>(M) * K;
        build_dst_ozaki_stack_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            L, M, K, Mp, Kp, S, S_B, beta, parity, transpose ? 1 : 0, sc, inv_mu, d_Astack);
        CUDA_TRY(cudaGetLastError());
        return 0;
    }

    int destroy() {
        int rc = 0;
        if (d_Astack) {
            if (cudaFree(d_Astack) != cudaSuccess) rc = 1;
            d_Astack = nullptr;
        }
        M = K = Mp = Kp = 0;
        return rc;
    }
};

template<int S_B, int BETA>
__global__ void slice_df_vec8_kernel(int K, int Kp, int n0, int nc,
                                     const float2* __restrict__ B,
                                     __nv_bfloat16* __restrict__ B_s,
                                     const float* __restrict__ d_scale) {
    int Kp8 = Kp >> 3;
    size_t total = static_cast<size_t>(nc) * Kp8;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int r_idx = static_cast<int>(idx / Kp8);
    int k8 = static_cast<int>(idx - static_cast<size_t>(r_idx) * Kp8);
    int k0 = k8 << 3;
    float inv_mu = d_scale[0];
    constexpr float MUL = static_cast<float>(1 << BETA);
    float rh[8], rl[8];
    const float2* row_ptr = B + static_cast<size_t>(n0 + r_idx) * K + k0;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        if (k0 + u < K) {
            float2 v = row_ptr[u];
            rh[u] = v.x * inv_mu;
            rl[u] = v.y * inv_mu;
        } else {
            rh[u] = 0.0f;
            rl[u] = 0.0f;
        }
    }
    uint4* out_u4 = reinterpret_cast<uint4*>(B_s + static_cast<size_t>(r_idx) * (S_B * Kp));
    #pragma unroll
    for (int s = 0; s < S_B; ++s) {
        __nv_bfloat16 bvals[8];
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            rh[u] *= MUL;
            rl[u] *= MUL;
            float q = rintf(rh[u] + rl[u]);
            float2 t = two_sum(rh[u], -q);
            float2 nw = quick_two_sum(t.x, __fadd_rn(t.y, rl[u]));
            rh[u] = nw.x;
            rl[u] = nw.y;
            bvals[u] = __float2bfloat16(q);
        }
        out_u4[s * Kp8 + k8] = *reinterpret_cast<uint4*>(bvals);
    }
}

template<int S, int BETA>
__device__ __forceinline__ float2 combine_stacked_at(const float* __restrict__ row_ptr,
                                                     int Mp, int m,
                                                     float base_sc, float d_sc1) {
    constexpr float INV_MUL = 1.0f / static_cast<float>(1 << BETA);
    float2 acc = make_float2(row_ptr[(S - 1) * Mp + m], 0.0f);
    #pragma unroll
    for (int d = S - 2; d >= 0; --d) {
        acc.x *= INV_MUL;
        acc.y *= INV_MUL;
        acc = df_add_f(acc, row_ptr[d * Mp + m]);
    }
    return make_float2((acc.x * base_sc) * d_sc1, (acc.y * base_sc) * d_sc1);
}

template<int S, int BETA>
__global__ void combine_stacked_vec4_kernel(int M, int Mp, int n0, int nc,
                                            const float* __restrict__ C_stack,
                                            float2* __restrict__ out,
                                            float base_scale,
                                            const float* __restrict__ d_scale) {
    int Mp4 = Mp >> 2;
    size_t total = static_cast<size_t>(nc) * Mp4;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int r_idx = static_cast<int>(idx / Mp4);
    int m4 = static_cast<int>(idx - static_cast<size_t>(r_idx) * Mp4);
    int m0 = m4 << 2;
    if (m0 >= M) return;
    const float4* row4 = reinterpret_cast<const float4*>(C_stack + static_cast<size_t>(r_idx) * (S * Mp));
    constexpr float INV_MUL = 1.0f / static_cast<float>(1 << BETA);
    float4 vl = row4[(S - 1) * Mp4 + m4];
    float2 a0 = make_float2(vl.x, 0.0f), a1 = make_float2(vl.y, 0.0f);
    float2 a2 = make_float2(vl.z, 0.0f), a3 = make_float2(vl.w, 0.0f);
    #pragma unroll
    for (int d = S - 2; d >= 0; --d) {
        float4 vd = row4[d * Mp4 + m4];
        a0.x *= INV_MUL; a0.y *= INV_MUL; a0 = df_add_f(a0, vd.x);
        a1.x *= INV_MUL; a1.y *= INV_MUL; a1 = df_add_f(a1, vd.y);
        a2.x *= INV_MUL; a2.y *= INV_MUL; a2 = df_add_f(a2, vd.z);
        a3.x *= INV_MUL; a3.y *= INV_MUL; a3 = df_add_f(a3, vd.w);
    }
    float sc1 = d_scale[1];
    float2* dst = out + static_cast<size_t>(n0 + r_idx) * M + m0;
    dst[0] = make_float2((a0.x * base_scale) * sc1, (a0.y * base_scale) * sc1);
    if (m0 + 1 < M) dst[1] = make_float2((a1.x * base_scale) * sc1, (a1.y * base_scale) * sc1);
    if (m0 + 2 < M) dst[2] = make_float2((a2.x * base_scale) * sc1, (a2.y * base_scale) * sc1);
    if (m0 + 3 < M) dst[3] = make_float2((a3.x * base_scale) * sc1, (a3.y * base_scale) * sc1);
}

static int ozaki_stacked_df(cublasHandle_t h, const OzakiMatrixStacked& A, int N,
                            const float2* d_B, float2* d_C,
                            __nv_bfloat16* d_Bs, float* d_Cstack,
                            float* d_max, float* d_scale, int max_chunk_n, cudaStream_t stream) {
    if (A.M == 0 || A.K == 0 || N == 0) return 0;
    size_t szB = static_cast<size_t>(N) * A.K;
    int max_blk = static_cast<int>(std::max<size_t>(1, std::min<size_t>(1024, (szB + 255) / 256)));
    CUDA_TRY(cudaMemsetAsync(d_max, 0, sizeof(float), stream));
    max_abs_df_kernel<<<max_blk, 256, 0, stream>>>(szB, d_B, d_max);
    compute_pow2_scale_kernel<<<1, 1, 0, stream>>>(d_max, d_scale, 1.0f);

    float alpha = 1.0f, beta_z = 0.0f;
    float base_scale = static_cast<float>(A.mu_A * ldexp(1.0, -2 * A.beta));
    int n_chunk = std::min(N, max_chunk_n);
    int S = A.S;
    for (int n0 = 0; n0 < N; n0 += n_chunk) {
        int nc = std::min(n_chunk, N - n0);
        size_t b8 = static_cast<size_t>(nc) * (A.Kp >> 3);
        size_t c4 = static_cast<size_t>(nc) * (A.Mp >> 2);
        if (A.beta == 7 && S == 5)      slice_df_vec8_kernel<5, 7><<<(b8 + 255) / 256, 256, 0, stream>>>(A.K, A.Kp, n0, nc, d_B, d_Bs, d_scale);
        else if (A.beta == 7 && S == 6) slice_df_vec8_kernel<6, 7><<<(b8 + 255) / 256, 256, 0, stream>>>(A.K, A.Kp, n0, nc, d_B, d_Bs, d_scale);
        else if (A.beta == 6 && S == 6) slice_df_vec8_kernel<6, 6><<<(b8 + 255) / 256, 256, 0, stream>>>(A.K, A.Kp, n0, nc, d_B, d_Bs, d_scale);
        else                            slice_df_vec8_kernel<7, 6><<<(b8 + 255) / 256, 256, 0, stream>>>(A.K, A.Kp, n0, nc, d_B, d_Bs, d_scale);

        CUBLAS_TRY(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, S * A.Mp, nc, A.S_B * A.Kp, &alpha,
                                A.d_Astack, CUDA_R_16BF, S * A.Mp, d_Bs, CUDA_R_16BF, A.S_B * A.Kp,
                                &beta_z, d_Cstack, CUDA_R_32F, S * A.Mp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        if (A.beta == 7 && S == 5)      combine_stacked_vec4_kernel<5, 7><<<(c4 + 255) / 256, 256, 0, stream>>>(A.M, A.Mp, n0, nc, d_Cstack, d_C, base_scale, d_scale);
        else if (A.beta == 7 && S == 6) combine_stacked_vec4_kernel<6, 7><<<(c4 + 255) / 256, 256, 0, stream>>>(A.M, A.Mp, n0, nc, d_Cstack, d_C, base_scale, d_scale);
        else if (A.beta == 6 && S == 6) combine_stacked_vec4_kernel<6, 6><<<(c4 + 255) / 256, 256, 0, stream>>>(A.M, A.Mp, n0, nc, d_Cstack, d_C, base_scale, d_scale);
        else                            combine_stacked_vec4_kernel<7, 6><<<(c4 + 255) / 256, 256, 0, stream>>>(A.M, A.Mp, n0, nc, d_Cstack, d_C, base_scale, d_scale);
    }
    return 0;
}

template<int Q_HALF, int KT>
__global__ void fused_eo_df64_inv_r_to_TeTo_kernel(int m, int Q, int ns3,
                                                   const float2* __restrict__ Se_df,
                                                   const float2* __restrict__ So_df,
                                                   const float2* __restrict__ R_e,
                                                   const float2* __restrict__ R_o,
                                                   float2* __restrict__ Te,
                                                   float2* __restrict__ To) {
    int h_sym = (m + 1) / 2, h_anti = m / 2;
    int z = blockIdx.x * 32 + threadIdx.x;
    int row0 = blockIdx.y * 32;
    int tx = threadIdx.x, ty = threadIdx.y;
    int total_rows = ns3 * Q;

    __shared__ float2 sm_Se[KT][33];
    __shared__ float2 sm_So[KT][33];
    __shared__ float2 sm_Re[32][KT + 1];
    __shared__ float2 sm_Ro[32][KT + 1];

    float2 ue[4], uo[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) { ue[u] = make_float2(0.0f, 0.0f); uo[u] = make_float2(0.0f, 0.0f); }

    int tid = ty * 32 + tx;
    #pragma unroll
    for (int q0 = 0; q0 < Q_HALF; q0 += KT) {
        for (int idx = tid; idx < 32 * KT; idx += 256) {
            int zl = idx / KT, ql = idx - zl * KT;
            int zg = blockIdx.x * 32 + zl;
            sm_Se[ql][zl] = (zg < h_sym)  ? Se_df[static_cast<size_t>(zg) * Q_HALF + (q0 + ql)] : make_float2(0.0f, 0.0f);
            sm_So[ql][zl] = (zg < h_anti) ? So_df[static_cast<size_t>(zg) * Q_HALF + (q0 + ql)] : make_float2(0.0f, 0.0f);
            int rl = zl, rg = row0 + rl;
            sm_Re[rl][ql] = (rg < total_rows) ? R_e[static_cast<size_t>(rg) * Q_HALF + (q0 + ql)] : make_float2(0.0f, 0.0f);
            sm_Ro[rl][ql] = (rg < total_rows) ? R_o[static_cast<size_t>(rg) * Q_HALF + (q0 + ql)] : make_float2(0.0f, 0.0f);
        }
        __syncthreads();
        if (z < h_sym) {
            #pragma unroll
            for (int ql = 0; ql < KT; ++ql) {
                float2 se = sm_Se[ql][tx], so = sm_So[ql][tx];
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    int rl = ty + u * 8;
                    ue[u] = df_fma_lazy(se, sm_Re[rl][ql], ue[u]);
                    uo[u] = df_fma_lazy(so, sm_Ro[rl][ql], uo[u]);
                }
            }
        }
        if (q0 + KT < Q_HALF) __syncthreads();
    }
    if (z >= h_sym) return;
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        int rg = row0 + ty + u * 8;
        if (rg >= total_rows) continue;
        int as = rg / Q, q = rg - as * Q;
        float2* dst = ((q & 1) == 0) ? Te : To;
        size_t base = (static_cast<size_t>(as) * Q_HALF + (q >> 1)) * m;
        float2 eu = quick_two_sum(ue[u].x, ue[u].y);
        if (z < h_anti) {
            float2 ov = quick_two_sum(uo[u].x, uo[u].y);
            dst[base + z] = df_add(eu, ov);
            dst[base + (m - 1 - z)] = df_sub(eu, ov);
        } else {
            dst[base + z] = eu;
        }
    }
}

template<int Q_HALF, int KT>
__global__ void fused_eo_df64_q_accum_tiled_kernel(int m, int ns, int axis,
                                                   const int* __restrict__ d_xlist,
                                                   const float2* __restrict__ Se_df,
                                                   const float2* __restrict__ So_df,
                                                   const float2* __restrict__ Te,
                                                   const float2* __restrict__ To,
                                                   float2* __restrict__ v_df,
                                                   float2* __restrict__ ax2_tmp) {
    int h_sym = (m + 1) / 2, h_anti = m / 2;
    int z = blockIdx.x * 32 + threadIdx.x;
    int y0 = blockIdx.y * 32;
    int s = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;
    int as = axis * ns + s;

    __shared__ float2 sm_Se[32][KT + 1];
    __shared__ float2 sm_So[32][KT + 1];
    __shared__ float2 sm_Te[KT][33];
    __shared__ float2 sm_To[KT][33];

    float2 ue[4], uo[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) { ue[u] = make_float2(0.0f, 0.0f); uo[u] = make_float2(0.0f, 0.0f); }

    int tid = ty * 32 + tx;
    const float2* Te_sl = Te + static_cast<size_t>(as) * Q_HALF * m;
    const float2* To_sl = To + static_cast<size_t>(as) * Q_HALF * m;

    #pragma unroll
    for (int q0 = 0; q0 < Q_HALF; q0 += KT) {
        for (int idx = tid; idx < 32 * KT; idx += 256) {
            int yl = idx / KT, ql = idx - yl * KT;
            int yg = y0 + yl;
            sm_Se[yl][ql] = (yg < h_sym)  ? Se_df[static_cast<size_t>(yg) * Q_HALF + (q0 + ql)] : make_float2(0.0f, 0.0f);
            sm_So[yl][ql] = (yg < h_anti) ? So_df[static_cast<size_t>(yg) * Q_HALF + (q0 + ql)] : make_float2(0.0f, 0.0f);
        }
        for (int ql = ty; ql < KT; ql += 8) {
            sm_Te[ql][tx] = (z < m) ? Te_sl[static_cast<size_t>(q0 + ql) * m + z] : make_float2(0.0f, 0.0f);
            sm_To[ql][tx] = (z < m) ? To_sl[static_cast<size_t>(q0 + ql) * m + z] : make_float2(0.0f, 0.0f);
        }
        __syncthreads();
        if (z < m) {
            #pragma unroll
            for (int ql = 0; ql < KT; ++ql) {
                float2 te = sm_Te[ql][tx], to = sm_To[ql][tx];
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    int yl = ty + u * 8;
                    ue[u] = df_fma_lazy(sm_Se[yl][ql], te, ue[u]);
                    uo[u] = df_fma_lazy(sm_So[yl][ql], to, uo[u]);
                }
            }
        }
        if (q0 + KT < Q_HALF) __syncthreads();
    }
    if (z >= m) return;
    int x = d_xlist[s] - 1;
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        int yg = y0 + ty + u * 8;
        if (yg >= h_sym) continue;
        float2 eu = quick_two_sum(ue[u].x, ue[u].y);
        if (yg < h_anti) {
            float2 ov = quick_two_sum(uo[u].x, uo[u].y);
            float2 v1 = df_add(eu, ov), v2 = df_sub(eu, ov);
            int y2 = m - 1 - yg;
            if (axis == 0) {
                size_t p1 = (static_cast<size_t>(x) * m + yg) * m + z;
                size_t p2 = (static_cast<size_t>(x) * m + y2) * m + z;
                v_df[p1] = df_add(v_df[p1], v1);
                v_df[p2] = df_add(v_df[p2], v2);
            } else if (axis == 1) {
                size_t p1 = (static_cast<size_t>(yg) * m + x) * m + z;
                size_t p2 = (static_cast<size_t>(y2) * m + x) * m + z;
                v_df[p1] = df_add(v_df[p1], v1);
                v_df[p2] = df_add(v_df[p2], v2);
            } else {
                ax2_tmp[(static_cast<size_t>(s) * m + yg) * m + z] = v1;
                ax2_tmp[(static_cast<size_t>(s) * m + y2) * m + z] = v2;
            }
        } else {
            if (axis == 0) {
                size_t p = (static_cast<size_t>(x) * m + yg) * m + z;
                v_df[p] = df_add(v_df[p], eu);
            } else if (axis == 1) {
                size_t p = (static_cast<size_t>(yg) * m + x) * m + z;
                v_df[p] = df_add(v_df[p], eu);
            } else {
                ax2_tmp[(static_cast<size_t>(s) * m + yg) * m + z] = eu;
            }
        }
    }
}

__global__ void finalize_fast_v2_kernel(int n1, int m, int z_chunk,
                                        const float2* __restrict__ v_df,
                                        float* __restrict__ buf1,
                                        float* __restrict__ buf0) {
    int k0 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int j0 = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    int i0 = blockIdx.z * z_chunk;
    if (k0 >= m || j0 >= m) return;
    int i1 = min(i0 + z_chunk, m);
    size_t s1 = static_cast<size_t>(n1), s2 = s1 * s1;
    size_t m1 = static_cast<size_t>(m), m2 = m1 * m1;

    auto get_v = [&](int ii, int jj, int kk) -> float2 {
        if (ii >= 0 && ii < m && jj >= 0 && jj < m && kk >= 0 && kk < m)
            return __ldg(v_df + static_cast<size_t>(ii) * m2 + static_cast<size_t>(jj) * m1 + kk);
        return make_float2(__ldg(buf1 + static_cast<size_t>(ii + 1) * s2 +
                                        static_cast<size_t>(jj + 1) * s1 + (kk + 1)), 0.0f);
    };

    if (j0 >= 1 && j0 + 2 < m && k0 >= 1 && k0 + 2 < m) {
        float2 pa0 = get_v(i0 - 1, j0,     k0), pa1 = get_v(i0 - 1, j0,     k0 + 1);
        float2 pb0 = get_v(i0 - 1, j0 + 1, k0), pb1 = get_v(i0 - 1, j0 + 1, k0 + 1);
        float2 ca0 = __ldg(v_df + static_cast<size_t>(i0) * m2 + static_cast<size_t>(j0) * m1 + k0);
        float2 ca1 = __ldg(v_df + static_cast<size_t>(i0) * m2 + static_cast<size_t>(j0) * m1 + k0 + 1);
        float2 cb0 = __ldg(v_df + static_cast<size_t>(i0) * m2 + static_cast<size_t>(j0 + 1) * m1 + k0);
        float2 cb1 = __ldg(v_df + static_cast<size_t>(i0) * m2 + static_cast<size_t>(j0 + 1) * m1 + k0 + 1);
        int i_fast_end = min(i1, m - 1);
        for (int i = i0; i < i1; ++i) {
            size_t vp0 = static_cast<size_t>(i) * m2 + static_cast<size_t>(j0) * m1 + k0;
            size_t vp1 = vp0 + m1;
            float2 na0, na1, nb0, nb1;
            if (i < i_fast_end) {
                na0 = __ldg(v_df + vp0 + m2); na1 = __ldg(v_df + vp0 + m2 + 1);
                nb0 = __ldg(v_df + vp1 + m2); nb1 = __ldg(v_df + vp1 + m2 + 1);
            } else {
                na0 = get_v(i + 1, j0,     k0); na1 = get_v(i + 1, j0,     k0 + 1);
                nb0 = get_v(i + 1, j0 + 1, k0); nb1 = get_v(i + 1, j0 + 1, k0 + 1);
            }
            float2 xma = __ldg(v_df + vp0 - 1), xpa = __ldg(v_df + vp0 + 2);
            float2 xmb = __ldg(v_df + vp1 - 1), xpb = __ldg(v_df + vp1 + 2);
            float2 ym0 = __ldg(v_df + vp0 - m1), ym1 = __ldg(v_df + vp0 - m1 + 1);
            float2 yp0 = __ldg(v_df + vp1 + m1), yp1 = __ldg(v_df + vp1 + m1 + 1);
            size_t gp0 = static_cast<size_t>(i + 1) * s2 + static_cast<size_t>(j0 + 1) * s1 + (k0 + 1);
            size_t gp1 = gp0 + s1;
            buf1[gp0] = __fadd_rn(ca0.x, ca0.y); buf1[gp0 + 1] = __fadd_rn(ca1.x, ca1.y);
            buf1[gp1] = __fadd_rn(cb0.x, cb0.y); buf1[gp1 + 1] = __fadd_rn(cb1.x, cb1.y);
            auto sten = [](float2 c, float2 p, float2 n, float2 ym, float2 yp, float2 xm, float2 xp) -> float {
                float2 s = df_add(df_add(p, n), df_add(df_add(ym, yp), df_add(xm, xp)));
                float2 r = df_add(make_float2(0.125f * s.x, 0.125f * s.y), make_float2(0.25f * c.x, 0.25f * c.y));
                return __fadd_rn(r.x, r.y);
            };
            buf0[gp0]     = sten(ca0, pa0, na0, ym0, cb0, xma, ca1);
            buf0[gp0 + 1] = sten(ca1, pa1, na1, ym1, cb1, ca0, xpa);
            buf0[gp1]     = sten(cb0, pb0, nb0, ca0, yp0, xmb, cb1);
            buf0[gp1 + 1] = sten(cb1, pb1, nb1, ca1, yp1, cb0, xpb);
            pa0 = ca0; pa1 = ca1; pb0 = cb0; pb1 = cb1;
            ca0 = na0; ca1 = na1; cb0 = nb0; cb1 = nb1;
        }
    } else {
        int je = min(j0 + 2, m), ke = min(k0 + 2, m);
        for (int j = j0; j < je; ++j) {
            for (int k = k0; k < ke; ++k) {
                float2 p = get_v(i0 - 1, j, k), c = get_v(i0, j, k);
                for (int i = i0; i < i1; ++i) {
                    float2 n = get_v(i + 1, j, k);
                    size_t gp = static_cast<size_t>(i + 1) * s2 + static_cast<size_t>(j + 1) * s1 + (k + 1);
                    buf1[gp] = __fadd_rn(c.x, c.y);
                    float2 s = df_add(df_add(p, n),
                                      df_add(df_add(get_v(i, j - 1, k), get_v(i, j + 1, k)),
                                             df_add(get_v(i, j, k - 1), get_v(i, j, k + 1))));
                    float2 r = df_add(make_float2(0.125f * s.x, 0.125f * s.y), make_float2(0.25f * c.x, 0.25f * c.y));
                    buf0[gp] = __fadd_rn(r.x, r.y);
                    p = c; c = n;
                }
            }
        }
    }
}

template<int S_B, int BETA>
__global__ void fused_extract_u0_vec8_kernel(int n1, int m, int Kp_e, int Kp_o, int b0, int nc,
                                             const float* __restrict__ buf0,
                                             __nv_bfloat16* __restrict__ Bs_sym,
                                             __nv_bfloat16* __restrict__ Bs_anti,
                                             const float* __restrict__ d_scale) {
    int h_sym = (m + 1) / 2, h_anti = m / 2, Kpe8 = Kp_e >> 3, Kpo8 = Kp_o >> 3;
    size_t total = static_cast<size_t>(nc) * Kpe8;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int r_idx = static_cast<int>(idx / Kpe8);
    int q8 = static_cast<int>(idx - static_cast<size_t>(r_idx) * Kpe8);
    int q0 = q8 << 3;
    int b = b0 + r_idx, i = b / m, j = b - i * m;
    size_t row = (static_cast<size_t>(i + 1) * n1 + (j + 1)) * n1 + 1;
    float inv_mu = d_scale[0];
    constexpr float MUL = static_cast<float>(1 << BETA);
    float sh[8], sl[8], ah[8], al[8];
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        int q = q0 + u;
        if (q < h_sym) {
            float a = buf0[row + q];
            if (q < h_anti) {
                float c = buf0[row + (m - 1 - q)];
                float2 sv = two_sum(a, c), av = two_sum(a, -c);
                sh[u] = sv.x * inv_mu; sl[u] = sv.y * inv_mu;
                ah[u] = av.x * inv_mu; al[u] = av.y * inv_mu;
            } else {
                sh[u] = a * inv_mu; sl[u] = 0.0f; ah[u] = 0.0f; al[u] = 0.0f;
            }
        } else {
            sh[u] = sl[u] = ah[u] = al[u] = 0.0f;
        }
    }
    uint4* out_e = reinterpret_cast<uint4*>(Bs_sym + static_cast<size_t>(r_idx) * (S_B * Kp_e));
    #pragma unroll
    for (int s = 0; s < S_B; ++s) {
        __nv_bfloat16 bv[8];
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            sh[u] *= MUL; sl[u] *= MUL;
            float qq = rintf(sh[u] + sl[u]);
            float2 t = two_sum(sh[u], -qq);
            float2 nw = quick_two_sum(t.x, __fadd_rn(t.y, sl[u]));
            sh[u] = nw.x; sl[u] = nw.y;
            bv[u] = __float2bfloat16(qq);
        }
        out_e[s * Kpe8 + q8] = *reinterpret_cast<uint4*>(bv);
    }
    if (q8 < Kpo8) {
        uint4* out_o = reinterpret_cast<uint4*>(Bs_anti + static_cast<size_t>(r_idx) * (S_B * Kp_o));
        #pragma unroll
        for (int s = 0; s < S_B; ++s) {
            __nv_bfloat16 bv[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                ah[u] *= MUL; al[u] *= MUL;
                float qq = rintf(ah[u] + al[u]);
                float2 t = two_sum(ah[u], -qq);
                float2 nw = quick_two_sum(t.x, __fadd_rn(t.y, al[u]));
                ah[u] = nw.x; al[u] = nw.y;
                bv[u] = __float2bfloat16(qq);
            }
            out_o[s * Kpo8 + q8] = *reinterpret_cast<uint4*>(bv);
        }
    }
}

static int ozaki_fwd_u0_stacked(cublasHandle_t h0, cublasHandle_t h1, int n1, int m,
                                const OzakiMatrixStacked& Ae, const OzakiMatrixStacked& Ao,
                                const float* d_buf0, float2* d_Ce, float2* d_Co,
                                __nv_bfloat16* d_Bs_e, __nv_bfloat16* d_Bs_o,
                                float* d_Cse, float* d_Cso,
                                float* d_max, float* d_scale, int max_chunk_n,
                                cudaStream_t s0, cudaStream_t s1,
                                cudaEvent_t ev_fork, cudaEvent_t ev_join) {
    size_t sz0 = static_cast<size_t>(n1) * n1 * n1;
    int max_blk = static_cast<int>(std::max<size_t>(1, std::min<size_t>(1024, (sz0 + 255) / 256)));
    CUDA_TRY(cudaMemsetAsync(d_max, 0, sizeof(float), s0));
    max_abs_f32_kernel<<<max_blk, 256, 0, s0>>>(sz0, d_buf0, d_max);
    compute_pow2_scale_kernel<<<1, 1, 0, s0>>>(d_max, d_scale, 2.0f);

    int N = m * m, S = Ae.S, beta = Ae.beta, S_B = Ae.S_B;
    float alpha = 1.0f, beta_z = 0.0f;
    float base_e = static_cast<float>(Ae.mu_A * ldexp(1.0, -2 * beta));
    float base_o = static_cast<float>(Ao.mu_A * ldexp(1.0, -2 * beta));
    int n_chunk = std::min(N, max_chunk_n);
    for (int b0 = 0; b0 < N; b0 += n_chunk) {
        int nc = std::min(n_chunk, N - b0);
        size_t tot8 = static_cast<size_t>(nc) * (Ae.Kp >> 3);
        if (beta == 7) fused_extract_u0_vec8_kernel<4, 7><<<(tot8 + 255) / 256, 256, 0, s0>>>(n1, m, Ae.Kp, Ao.Kp, b0, nc, d_buf0, d_Bs_e, d_Bs_o, d_scale);
        else           fused_extract_u0_vec8_kernel<5, 6><<<(tot8 + 255) / 256, 256, 0, s0>>>(n1, m, Ae.Kp, Ao.Kp, b0, nc, d_buf0, d_Bs_e, d_Bs_o, d_scale);

        CUDA_TRY(cudaEventRecord(ev_fork, s0));
        CUDA_TRY(cudaStreamWaitEvent(s1, ev_fork, 0));

        CUBLAS_TRY(cublasGemmEx(h0, CUBLAS_OP_N, CUBLAS_OP_N, S * Ae.Mp, nc, S_B * Ae.Kp, &alpha,
                                Ae.d_Astack, CUDA_R_16BF, S * Ae.Mp, d_Bs_e, CUDA_R_16BF, S_B * Ae.Kp,
                                &beta_z, d_Cse, CUDA_R_32F, S * Ae.Mp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        size_t c4e = static_cast<size_t>(nc) * (Ae.Mp >> 2);
        if (beta == 7 && S == 5)      combine_stacked_vec4_kernel<5, 7><<<(c4e + 255) / 256, 256, 0, s0>>>(Ae.M, Ae.Mp, b0, nc, d_Cse, d_Ce, base_e, d_scale);
        else if (beta == 7 && S == 6) combine_stacked_vec4_kernel<6, 7><<<(c4e + 255) / 256, 256, 0, s0>>>(Ae.M, Ae.Mp, b0, nc, d_Cse, d_Ce, base_e, d_scale);
        else if (beta == 6 && S == 6) combine_stacked_vec4_kernel<6, 6><<<(c4e + 255) / 256, 256, 0, s0>>>(Ae.M, Ae.Mp, b0, nc, d_Cse, d_Ce, base_e, d_scale);
        else                          combine_stacked_vec4_kernel<7, 6><<<(c4e + 255) / 256, 256, 0, s0>>>(Ae.M, Ae.Mp, b0, nc, d_Cse, d_Ce, base_e, d_scale);

        if (Ao.M > 0 && Ao.K > 0) {
            CUBLAS_TRY(cublasGemmEx(h1, CUBLAS_OP_N, CUBLAS_OP_N, S * Ao.Mp, nc, S_B * Ao.Kp, &alpha,
                                    Ao.d_Astack, CUDA_R_16BF, S * Ao.Mp, d_Bs_o, CUDA_R_16BF, S_B * Ao.Kp,
                                    &beta_z, d_Cso, CUDA_R_32F, S * Ao.Mp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            size_t c4o = static_cast<size_t>(nc) * (Ao.Mp >> 2);
            if (beta == 7 && S == 5)      combine_stacked_vec4_kernel<5, 7><<<(c4o + 255) / 256, 256, 0, s1>>>(Ao.M, Ao.Mp, b0, nc, d_Cso, d_Co, base_o, d_scale);
            else if (beta == 7 && S == 6) combine_stacked_vec4_kernel<6, 7><<<(c4o + 255) / 256, 256, 0, s1>>>(Ao.M, Ao.Mp, b0, nc, d_Cso, d_Co, base_o, d_scale);
            else if (beta == 6 && S == 6) combine_stacked_vec4_kernel<6, 6><<<(c4o + 255) / 256, 256, 0, s1>>>(Ao.M, Ao.Mp, b0, nc, d_Cso, d_Co, base_o, d_scale);
            else                          combine_stacked_vec4_kernel<7, 6><<<(c4o + 255) / 256, 256, 0, s1>>>(Ao.M, Ao.Mp, b0, nc, d_Cso, d_Co, base_o, d_scale);
        }
        CUDA_TRY(cudaEventRecord(ev_join, s1));
        CUDA_TRY(cudaStreamWaitEvent(s0, ev_join, 0));
    }
    return 0;
}

static int ozaki_eo_pair_stacked(cublasHandle_t h0, cublasHandle_t h1,
                                 const OzakiMatrixStacked& Ae, const OzakiMatrixStacked& Ao, int N,
                                 const float2* d_Be, const float2* d_Bo,
                                 float2* d_Ce, float2* d_Co,
                                 __nv_bfloat16* d_Bs_e, __nv_bfloat16* d_Bs_o,
                                 float* d_Cse, float* d_Cso,
                                 float* d_max_e, float* d_max_o,
                                 float* d_sc_e, float* d_sc_o,
                                 int max_chunk_n, cudaStream_t s0, cudaStream_t s1,
                                 cudaEvent_t ev_fork, cudaEvent_t ev_join) {
    CUDA_TRY(cudaEventRecord(ev_fork, s0));
    CUDA_TRY(cudaStreamWaitEvent(s1, ev_fork, 0));
    if (ozaki_stacked_df(h0, Ae, N, d_Be, d_Ce, d_Bs_e, d_Cse, d_max_e, d_sc_e, max_chunk_n, s0) != 0) return 1;
    if (ozaki_stacked_df(h1, Ao, N, d_Bo, d_Co, d_Bs_o, d_Cso, d_max_o, d_sc_o, max_chunk_n, s1) != 0) return 1;
    CUDA_TRY(cudaEventRecord(ev_join, s1));
    CUDA_TRY(cudaStreamWaitEvent(s0, ev_join, 0));
    return 0;
}

template<int S, int BETA>
__global__ void fused_combine_inv_k_stacked_kernel(int m, int Mp_e, int Mp_o, int b0, int nc,
                                                   const float* __restrict__ Ce_st,
                                                   const float* __restrict__ Co_st,
                                                   float base_e, float base_o,
                                                   const float* __restrict__ sc_e,
                                                   const float* __restrict__ sc_o,
                                                   float2* __restrict__ v_df) {
    int h_sym = (m + 1) / 2, h_anti = m / 2;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= h_sym || r_idx >= nc) return;
    float2 u = combine_stacked_at<S, BETA>(Ce_st + static_cast<size_t>(r_idx) * (S * Mp_e), Mp_e, k, base_e, sc_e[1]);
    size_t b = static_cast<size_t>(b0 + r_idx);
    if (k < h_anti) {
        float2 v = (Mp_o > 0) ? combine_stacked_at<S, BETA>(Co_st + static_cast<size_t>(r_idx) * (S * Mp_o), Mp_o, k, base_o, sc_o[1])
                              : make_float2(0.0f, 0.0f);
        v_df[b * m + k] = df_add(u, v);
        v_df[b * m + (m - 1 - k)] = df_sub(u, v);
    } else {
        v_df[b * m + k] = u;
    }
}

static int ozaki_inv_k_stacked_fused(cublasHandle_t h0, cublasHandle_t h1, int m,
                                     const OzakiMatrixStacked& Ae, const OzakiMatrixStacked& Ao,
                                     const float2* d_Be, const float2* d_Bo, float2* d_vdf,
                                     __nv_bfloat16* d_Bs_e, __nv_bfloat16* d_Bs_o,
                                     float* d_Cse, float* d_Cso,
                                     float* d_max_e, float* d_max_o,
                                     float* d_sc_e, float* d_sc_o,
                                     int max_chunk_n, cudaStream_t s0, cudaStream_t s1,
                                     cudaEvent_t ev_fork, cudaEvent_t ev_join) {
    int N = m * m;
    CUDA_TRY(cudaEventRecord(ev_fork, s0));
    CUDA_TRY(cudaStreamWaitEvent(s1, ev_fork, 0));

    size_t szBe = static_cast<size_t>(N) * Ae.K;
    int blk_e = static_cast<int>(std::max<size_t>(1, std::min<size_t>(1024, (szBe + 255) / 256)));
    CUDA_TRY(cudaMemsetAsync(d_max_e, 0, sizeof(float), s0));
    max_abs_df_kernel<<<blk_e, 256, 0, s0>>>(szBe, d_Be, d_max_e);
    compute_pow2_scale_kernel<<<1, 1, 0, s0>>>(d_max_e, d_sc_e, 1.0f);

    if (Ao.M > 0 && Ao.K > 0) {
        size_t szBo = static_cast<size_t>(N) * Ao.K;
        int blk_o = static_cast<int>(std::max<size_t>(1, std::min<size_t>(1024, (szBo + 255) / 256)));
        CUDA_TRY(cudaMemsetAsync(d_max_o, 0, sizeof(float), s1));
        max_abs_df_kernel<<<blk_o, 256, 0, s1>>>(szBo, d_Bo, d_max_o);
        compute_pow2_scale_kernel<<<1, 1, 0, s1>>>(d_max_o, d_sc_o, 1.0f);
    }

    int S = Ae.S, beta = Ae.beta;
    float alpha = 1.0f, beta_z = 0.0f;
    float base_e = static_cast<float>(Ae.mu_A * ldexp(1.0, -2 * beta));
    float base_o = static_cast<float>(Ao.mu_A * ldexp(1.0, -2 * beta));
    int n_chunk = std::min(N, max_chunk_n), h_sym = (m + 1) / 2;
    dim3 blk16(16, 16);

    for (int b0 = 0; b0 < N; b0 += n_chunk) {
        int nc = std::min(n_chunk, N - b0);
        size_t b8e = static_cast<size_t>(nc) * (Ae.Kp >> 3);
        if (beta == 7 && S == 5)      slice_df_vec8_kernel<5, 7><<<(b8e + 255) / 256, 256, 0, s0>>>(Ae.K, Ae.Kp, b0, nc, d_Be, d_Bs_e, d_sc_e);
        else if (beta == 7 && S == 6) slice_df_vec8_kernel<6, 7><<<(b8e + 255) / 256, 256, 0, s0>>>(Ae.K, Ae.Kp, b0, nc, d_Be, d_Bs_e, d_sc_e);
        else if (beta == 6 && S == 6) slice_df_vec8_kernel<6, 6><<<(b8e + 255) / 256, 256, 0, s0>>>(Ae.K, Ae.Kp, b0, nc, d_Be, d_Bs_e, d_sc_e);
        else                          slice_df_vec8_kernel<7, 6><<<(b8e + 255) / 256, 256, 0, s0>>>(Ae.K, Ae.Kp, b0, nc, d_Be, d_Bs_e, d_sc_e);
        CUBLAS_TRY(cublasGemmEx(h0, CUBLAS_OP_N, CUBLAS_OP_N, S * Ae.Mp, nc, Ae.S_B * Ae.Kp, &alpha,
                                Ae.d_Astack, CUDA_R_16BF, S * Ae.Mp, d_Bs_e, CUDA_R_16BF, Ae.S_B * Ae.Kp,
                                &beta_z, d_Cse, CUDA_R_32F, S * Ae.Mp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        if (Ao.M > 0 && Ao.K > 0) {
            size_t b8o = static_cast<size_t>(nc) * (Ao.Kp >> 3);
            if (beta == 7 && S == 5)      slice_df_vec8_kernel<5, 7><<<(b8o + 255) / 256, 256, 0, s1>>>(Ao.K, Ao.Kp, b0, nc, d_Bo, d_Bs_o, d_sc_o);
            else if (beta == 7 && S == 6) slice_df_vec8_kernel<6, 7><<<(b8o + 255) / 256, 256, 0, s1>>>(Ao.K, Ao.Kp, b0, nc, d_Bo, d_Bs_o, d_sc_o);
            else if (beta == 6 && S == 6) slice_df_vec8_kernel<6, 6><<<(b8o + 255) / 256, 256, 0, s1>>>(Ao.K, Ao.Kp, b0, nc, d_Bo, d_Bs_o, d_sc_o);
            else                          slice_df_vec8_kernel<7, 6><<<(b8o + 255) / 256, 256, 0, s1>>>(Ao.K, Ao.Kp, b0, nc, d_Bo, d_Bs_o, d_sc_o);
            CUBLAS_TRY(cublasGemmEx(h1, CUBLAS_OP_N, CUBLAS_OP_N, S * Ao.Mp, nc, Ao.S_B * Ao.Kp, &alpha,
                                    Ao.d_Astack, CUDA_R_16BF, S * Ao.Mp, d_Bs_o, CUDA_R_16BF, Ao.S_B * Ao.Kp,
                                    &beta_z, d_Cso, CUDA_R_32F, S * Ao.Mp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
        CUDA_TRY(cudaEventRecord(ev_join, s1));
        CUDA_TRY(cudaStreamWaitEvent(s0, ev_join, 0));

        dim3 grd((h_sym + 15) / 16, (nc + 15) / 16);
        if (beta == 7 && S == 5)      fused_combine_inv_k_stacked_kernel<5, 7><<<grd, blk16, 0, s0>>>(m, Ae.Mp, Ao.Mp, b0, nc, d_Cse, d_Cso, base_e, base_o, d_sc_e, d_sc_o, d_vdf);
        else if (beta == 7 && S == 6) fused_combine_inv_k_stacked_kernel<6, 7><<<grd, blk16, 0, s0>>>(m, Ae.Mp, Ao.Mp, b0, nc, d_Cse, d_Cso, base_e, base_o, d_sc_e, d_sc_o, d_vdf);
        else if (beta == 6 && S == 6) fused_combine_inv_k_stacked_kernel<6, 6><<<grd, blk16, 0, s0>>>(m, Ae.Mp, Ao.Mp, b0, nc, d_Cse, d_Cso, base_e, base_o, d_sc_e, d_sc_o, d_vdf);
        else                          fused_combine_inv_k_stacked_kernel<7, 6><<<grd, blk16, 0, s0>>>(m, Ae.Mp, Ao.Mp, b0, nc, d_Cse, d_Cso, base_e, base_o, d_sc_e, d_sc_o, d_vdf);

        if (b0 + n_chunk < N) {
            CUDA_TRY(cudaEventRecord(ev_fork, s0));
            CUDA_TRY(cudaStreamWaitEvent(s1, ev_fork, 0));
        }
    }
    return 0;
}

template<int S, int BETA>
__global__ void fused_combine_bucket_stacked_kernel(
    int m, int Mp_e, int Mp_o, int ns_total, int as0, int sc_slices,
    const int* __restrict__ d_xlist,
    const float* __restrict__ Ce_st, const float* __restrict__ Co_st,
    float base_e, float base_o,
    const float* __restrict__ sc_e, const float* __restrict__ sc_o,
    float2* __restrict__ v_df, float2* __restrict__ ax2_tmp) {
    int h_anti = m / 2;
    int y0 = blockIdx.x * 16, z0 = blockIdx.y * 16, sl = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;
    int as = as0 + sl, axis = as / ns_total, s = as - axis * ns_total;
    float sc1e = sc_e[1], sc1o = (Mp_o > 0) ? sc_o[1] : 0.0f;
    __shared__ float2 sm[16][17];
    int y = y0 + tx, z = z0 + ty;
    if (y < m && z < m) {
        size_t row = static_cast<size_t>(sl) * m + z;
        const float* re = Ce_st + row * (S * Mp_e);
        const float* ro = Co_st + row * (S * Mp_o);
        float2 val;
        if (y < h_anti) {
            float2 u = combine_stacked_at<S, BETA>(re, Mp_e, y, base_e, sc1e);
            float2 v = (Mp_o > 0) ? combine_stacked_at<S, BETA>(ro, Mp_o, y, base_o, sc1o) : make_float2(0.0f, 0.0f);
            val = df_add(u, v);
        } else if (y == h_anti && (m & 1)) {
            val = combine_stacked_at<S, BETA>(re, Mp_e, h_anti, base_e, sc1e);
        } else {
            int y2 = m - 1 - y;
            float2 u = combine_stacked_at<S, BETA>(re, Mp_e, y2, base_e, sc1e);
            float2 v = (Mp_o > 0) ? combine_stacked_at<S, BETA>(ro, Mp_o, y2, base_o, sc1o) : make_float2(0.0f, 0.0f);
            val = df_sub(u, v);
        }
        sm[tx][ty] = val;
    }
    __syncthreads();
    int zw = z0 + tx, yw = y0 + ty;
    if (zw < m && yw < m) {
        float2 val = sm[ty][tx];
        int x = d_xlist[s] - 1;
        if (axis == 0) {
            size_t p = (static_cast<size_t>(x) * m + yw) * m + zw;
            v_df[p] = df_add(v_df[p], val);
        } else if (axis == 1) {
            size_t p = (static_cast<size_t>(yw) * m + x) * m + zw;
            v_df[p] = df_add(v_df[p], val);
        } else {
            ax2_tmp[(static_cast<size_t>(s) * m + yw) * m + zw] = val;
        }
    }
}

static int ozaki_bucket_q_stacked_fused(cublasHandle_t h0, cublasHandle_t h1, int m, int ns_total,
                                        const int* d_xlist,
                                        const OzakiMatrixStacked& Ae, const OzakiMatrixStacked& Ao,
                                        const float2* d_Qe, const float2* d_Qo,
                                        float2* d_vdf, float2* d_ax2,
                                        __nv_bfloat16* d_Bs_e, __nv_bfloat16* d_Bs_o,
                                        float* d_Cse, float* d_Cso,
                                        float* d_max_e, float* d_max_o,
                                        float* d_sc_e, float* d_sc_o,
                                        int max_chunk_n, cudaStream_t s0, cudaStream_t s1,
                                        cudaEvent_t ev_fork, cudaEvent_t ev_join) {
    int ns3 = 3 * ns_total, N = ns3 * m;
    CUDA_TRY(cudaEventRecord(ev_fork, s0));
    CUDA_TRY(cudaStreamWaitEvent(s1, ev_fork, 0));

    size_t szQe = static_cast<size_t>(N) * Ae.K;
    int blk_e = static_cast<int>(std::max<size_t>(1, std::min<size_t>(1024, (szQe + 255) / 256)));
    CUDA_TRY(cudaMemsetAsync(d_max_e, 0, sizeof(float), s0));
    max_abs_df_kernel<<<blk_e, 256, 0, s0>>>(szQe, d_Qe, d_max_e);
    compute_pow2_scale_kernel<<<1, 1, 0, s0>>>(d_max_e, d_sc_e, 1.0f);

    if (Ao.M > 0 && Ao.K > 0) {
        size_t szQo = static_cast<size_t>(N) * Ao.K;
        int blk_o = static_cast<int>(std::max<size_t>(1, std::min<size_t>(1024, (szQo + 255) / 256)));
        CUDA_TRY(cudaMemsetAsync(d_max_o, 0, sizeof(float), s1));
        max_abs_df_kernel<<<blk_o, 256, 0, s1>>>(szQo, d_Qo, d_max_o);
        compute_pow2_scale_kernel<<<1, 1, 0, s1>>>(d_max_o, d_sc_o, 1.0f);
    }

    int S = Ae.S, beta = Ae.beta;
    float alpha = 1.0f, beta_z = 0.0f;
    float base_e = static_cast<float>(Ae.mu_A * ldexp(1.0, -2 * beta));
    float base_o = static_cast<float>(Ao.mu_A * ldexp(1.0, -2 * beta));
    int sc_max = std::max(1, max_chunk_n / m);
    dim3 blk16(16, 16);

    for (int axis = 0; axis < 3; ++axis) {
        int as_begin = axis * ns_total, as_end = as_begin + ns_total;
        for (int as0 = as_begin; as0 < as_end; as0 += sc_max) {
            int sc_slices = std::min(sc_max, as_end - as0);
            int n0 = as0 * m, nc = sc_slices * m;
            size_t b8e = static_cast<size_t>(nc) * (Ae.Kp >> 3);
            if (beta == 7 && S == 5)      slice_df_vec8_kernel<5, 7><<<(b8e + 255) / 256, 256, 0, s0>>>(Ae.K, Ae.Kp, n0, nc, d_Qe, d_Bs_e, d_sc_e);
            else if (beta == 7 && S == 6) slice_df_vec8_kernel<6, 7><<<(b8e + 255) / 256, 256, 0, s0>>>(Ae.K, Ae.Kp, n0, nc, d_Qe, d_Bs_e, d_sc_e);
            else if (beta == 6 && S == 6) slice_df_vec8_kernel<6, 6><<<(b8e + 255) / 256, 256, 0, s0>>>(Ae.K, Ae.Kp, n0, nc, d_Qe, d_Bs_e, d_sc_e);
            else                          slice_df_vec8_kernel<7, 6><<<(b8e + 255) / 256, 256, 0, s0>>>(Ae.K, Ae.Kp, n0, nc, d_Qe, d_Bs_e, d_sc_e);
            CUBLAS_TRY(cublasGemmEx(h0, CUBLAS_OP_N, CUBLAS_OP_N, S * Ae.Mp, nc, Ae.S_B * Ae.Kp, &alpha,
                                    Ae.d_Astack, CUDA_R_16BF, S * Ae.Mp, d_Bs_e, CUDA_R_16BF, Ae.S_B * Ae.Kp,
                                    &beta_z, d_Cse, CUDA_R_32F, S * Ae.Mp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

            if (Ao.M > 0 && Ao.K > 0) {
                size_t b8o = static_cast<size_t>(nc) * (Ao.Kp >> 3);
                if (beta == 7 && S == 5)      slice_df_vec8_kernel<5, 7><<<(b8o + 255) / 256, 256, 0, s1>>>(Ao.K, Ao.Kp, n0, nc, d_Qo, d_Bs_o, d_sc_o);
                else if (beta == 7 && S == 6) slice_df_vec8_kernel<6, 7><<<(b8o + 255) / 256, 256, 0, s1>>>(Ao.K, Ao.Kp, n0, nc, d_Qo, d_Bs_o, d_sc_o);
                else if (beta == 6 && S == 6) slice_df_vec8_kernel<6, 6><<<(b8o + 255) / 256, 256, 0, s1>>>(Ao.K, Ao.Kp, n0, nc, d_Qo, d_Bs_o, d_sc_o);
                else                          slice_df_vec8_kernel<7, 6><<<(b8o + 255) / 256, 256, 0, s1>>>(Ao.K, Ao.Kp, n0, nc, d_Qo, d_Bs_o, d_sc_o);
                CUBLAS_TRY(cublasGemmEx(h1, CUBLAS_OP_N, CUBLAS_OP_N, S * Ao.Mp, nc, Ao.S_B * Ao.Kp, &alpha,
                                        Ao.d_Astack, CUDA_R_16BF, S * Ao.Mp, d_Bs_o, CUDA_R_16BF, Ao.S_B * Ao.Kp,
                                        &beta_z, d_Cso, CUDA_R_32F, S * Ao.Mp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
            CUDA_TRY(cudaEventRecord(ev_join, s1));
            CUDA_TRY(cudaStreamWaitEvent(s0, ev_join, 0));

            dim3 grd((m + 15) / 16, (m + 15) / 16, sc_slices);
            if (beta == 7 && S == 5)      fused_combine_bucket_stacked_kernel<5, 7><<<grd, blk16, 0, s0>>>(m, Ae.Mp, Ao.Mp, ns_total, as0, sc_slices, d_xlist, d_Cse, d_Cso, base_e, base_o, d_sc_e, d_sc_o, d_vdf, d_ax2);
            else if (beta == 7 && S == 6) fused_combine_bucket_stacked_kernel<6, 7><<<grd, blk16, 0, s0>>>(m, Ae.Mp, Ao.Mp, ns_total, as0, sc_slices, d_xlist, d_Cse, d_Cso, base_e, base_o, d_sc_e, d_sc_o, d_vdf, d_ax2);
            else if (beta == 6 && S == 6) fused_combine_bucket_stacked_kernel<6, 6><<<grd, blk16, 0, s0>>>(m, Ae.Mp, Ao.Mp, ns_total, as0, sc_slices, d_xlist, d_Cse, d_Cso, base_e, base_o, d_sc_e, d_sc_o, d_vdf, d_ax2);
            else                          fused_combine_bucket_stacked_kernel<7, 6><<<grd, blk16, 0, s0>>>(m, Ae.Mp, Ao.Mp, ns_total, as0, sc_slices, d_xlist, d_Cse, d_Cso, base_e, base_o, d_sc_e, d_sc_o, d_vdf, d_ax2);

            if (axis < 2 || as0 + sc_max < as_end) {
                CUDA_TRY(cudaEventRecord(ev_fork, s0));
                CUDA_TRY(cudaStreamWaitEvent(s1, ev_fork, 0));
            }
        }
    }
    return 0;
}

__global__ void fwd_k_to_j_kernel(int m, int K,
                                  const float2* __restrict__ Ce, const float2* __restrict__ Co,
                                  float2* __restrict__ B2_sym, float2* __restrict__ B2_anti) {
    int h_sym = (m + 1) / 2, h_anti = m / 2, Ke = (K + 1) / 2, Ko = K / 2;
    int q0 = blockIdx.x * 16, r0 = blockIdx.y * 16, i = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ float2 sm_a[16][17], sm_c[16][17];
    int r = r0 + tx, q = q0 + ty;
    if (r < K && q < h_sym) {
        const float2* src = ((r & 1) == 0) ? Ce : Co;
        int st = ((r & 1) == 0) ? Ke : Ko, rm = r >> 1;
        sm_a[ty][tx] = src[(static_cast<size_t>(i) * m + q) * st + rm];
        sm_c[ty][tx] = (q < h_anti) ? src[(static_cast<size_t>(i) * m + (m - 1 - q)) * st + rm] : make_float2(0.0f, 0.0f);
    }
    __syncthreads();
    int qw = q0 + tx, rw = r0 + ty;
    if (rw < K && qw < h_sym) {
        float2 a = sm_a[tx][ty];
        size_t row2 = static_cast<size_t>(i) * K + rw;
        if (qw < h_anti) {
            float2 c = sm_c[tx][ty];
            B2_sym[row2 * h_sym + qw] = df_add(a, c);
            B2_anti[row2 * h_anti + qw] = df_sub(a, c);
        } else {
            B2_sym[row2 * h_sym + qw] = a;
        }
    }
}

__global__ void fwd_j_to_i_kernel(int m, int K,
                                  const float2* __restrict__ Ce, const float2* __restrict__ Co,
                                  float2* __restrict__ B3_sym, float2* __restrict__ B3_anti) {
    int h_sym = (m + 1) / 2, h_anti = m / 2, Ke = (K + 1) / 2, Ko = K / 2;
    size_t total = static_cast<size_t>(K) * K * h_sym;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int p = static_cast<int>(idx % h_sym);
    size_t b3 = idx / h_sym;
    int q = static_cast<int>(b3 / K);
    int r = static_cast<int>(b3 - static_cast<size_t>(q) * K);
    const float2* src = ((q & 1) == 0) ? Ce : Co;
    int st = ((q & 1) == 0) ? Ke : Ko, qm = q >> 1;
    float2 a = src[(static_cast<size_t>(p) * K + r) * st + qm];
    if (p < h_anti) {
        float2 c = src[(static_cast<size_t>(m - 1 - p) * K + r) * st + qm];
        B3_sym[b3 * h_sym + p] = df_add(a, c);
        B3_anti[b3 * h_anti + p] = df_sub(a, c);
    } else {
        B3_sym[b3 * h_sym + p] = a;
    }
}

__global__ void spectral_update_K3_mfirst_kernel(int m, int K, int n0,
                                                 const double* __restrict__ s1,
                                                 const double* __restrict__ sin2_h,
                                                 const double* __restrict__ faces_hat,
                                                 const float2* __restrict__ u0_e,
                                                 const float2* __restrict__ u0_o,
                                                 float2* __restrict__ w_e,
                                                 float2* __restrict__ w_o) {
    int Ke = (K + 1) / 2, Ko = K / 2;
    size_t total = static_cast<size_t>(K) * K * K;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int p = static_cast<int>(idx % K);
    size_t b3 = idx / K;
    int q = static_cast<int>(b3 / K);
    int r = static_cast<int>(b3 - static_cast<size_t>(q) * K);
    int pm = p >> 1;
    float2 u0_df = ((p & 1) == 0) ? u0_e[b3 * Ke + pm] : u0_o[b3 * Ko + pm];
    double u0_val = static_cast<double>(u0_df.x) + static_cast<double>(u0_df.y);
    size_t m2 = static_cast<size_t>(m) * m;
    double sp = ((p & 1) == 0) ? 1.0 : -1.0;
    double sq = ((q & 1) == 0) ? 1.0 : -1.0;
    double sr = ((r & 1) == 0) ? 1.0 : -1.0;
    double b0h = 0.125 * (s1[p] * (faces_hat[0 * m2 + static_cast<size_t>(q) * m + r] + sp * faces_hat[1 * m2 + static_cast<size_t>(q) * m + r])
                        + s1[q] * (faces_hat[2 * m2 + static_cast<size_t>(p) * m + r] + sq * faces_hat[3 * m2 + static_cast<size_t>(p) * m + r])
                        + s1[r] * (faces_hat[4 * m2 + static_cast<size_t>(p) * m + q] + sr * faces_hat[5 * m2 + static_cast<size_t>(p) * m + q]));
    double b1h = 0.125 * (s1[p] * (faces_hat[6 * m2 + static_cast<size_t>(q) * m + r] + sp * faces_hat[7 * m2 + static_cast<size_t>(q) * m + r])
                        + s1[q] * (faces_hat[8 * m2 + static_cast<size_t>(p) * m + r] + sq * faces_hat[9 * m2 + static_cast<size_t>(p) * m + r])
                        + s1[r] * (faces_hat[10 * m2 + static_cast<size_t>(p) * m + q] + sr * faces_hat[11 * m2 + static_cast<size_t>(p) * m + q]));
    double delta = 0.5 * (sin2_h[p] + sin2_h[q] + sin2_h[r]);
    double lam = 1.0 - delta;
    double u_inf_h = (lam * b0h + b1h) / (delta * (2.0 - delta));
    double exp_n = 2.0 * static_cast<double>(n0) - 1.0;
    double lam_pow;
    if (delta < 0.5) {
        lam_pow = exp(exp_n * log1p(-delta));
    } else if (lam == 0.0) {
        lam_pow = 0.0;
    } else {
        double pval = exp(exp_n * log(fabs(lam)));
        lam_pow = (lam < 0.0) ? -pval : pval;
    }
    float2 wf = d2df(lam_pow * (u0_val - u_inf_h));
    if ((p & 1) == 0) w_e[b3 * Ke + pm] = wf;
    else              w_o[b3 * Ko + pm] = wf;
}

__global__ void inv_i_to_j_mfirst_kernel(int m, int K,
                                         const float2* __restrict__ U_i, const float2* __restrict__ V_i,
                                         float2* __restrict__ J_e, float2* __restrict__ J_o) {
    int h_sym = (m + 1) / 2, h_anti = m / 2, Ke = (K + 1) / 2, Ko = K / 2;
    size_t total = static_cast<size_t>(m) * K * K;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int q = static_cast<int>(idx % K);
    size_t ir = idx / K;
    int i = static_cast<int>(ir / K);
    int r = static_cast<int>(ir - static_cast<size_t>(i) * K);
    size_t b3 = static_cast<size_t>(q) * K + r;
    float2 val;
    if (i < h_anti) val = df_add(U_i[b3 * h_sym + i], V_i[b3 * h_anti + i]);
    else if (i == h_anti && (m & 1)) val = U_i[b3 * h_sym + h_anti];
    else { int i2 = m - 1 - i; val = df_sub(U_i[b3 * h_sym + i2], V_i[b3 * h_anti + i2]); }
    if ((q & 1) == 0) J_e[ir * Ke + (q >> 1)] = val;
    else              J_o[ir * Ko + (q >> 1)] = val;
}

__global__ void inv_j_to_k_mfirst_kernel(int m, int K,
                                         const float2* __restrict__ U_j, const float2* __restrict__ V_j,
                                         float2* __restrict__ K_e, float2* __restrict__ K_o) {
    int h_sym = (m + 1) / 2, h_anti = m / 2, Ke = (K + 1) / 2, Ko = K / 2;
    int j0 = blockIdx.x * 16, r0 = blockIdx.y * 16, i = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ float2 sm[16][17];
    int j = j0 + tx, r = r0 + ty;
    if (j < m && r < K) {
        size_t row = static_cast<size_t>(i) * K + r;
        float2 val;
        if (j < h_anti) val = df_add(U_j[row * h_sym + j], V_j[row * h_anti + j]);
        else if (j == h_anti && (m & 1)) val = U_j[row * h_sym + h_anti];
        else { int j2 = m - 1 - j; val = df_sub(U_j[row * h_sym + j2], V_j[row * h_anti + j2]); }
        sm[tx][ty] = val;
    }
    __syncthreads();
    int rw = r0 + tx, jw = j0 + ty;
    if (rw < K && jw < m) {
        size_t ro = static_cast<size_t>(i) * m + jw;
        float2 val = sm[ty][tx];
        if ((rw & 1) == 0) K_e[ro * Ke + (rw >> 1)] = val;
        else               K_o[ro * Ko + (rw >> 1)] = val;
    }
}

__global__ void extract_6faces_kernel(int n1, int m, const float* __restrict__ buf,
                                      float2* __restrict__ faces_df) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;
    if (a >= m || b >= m) return;
    int axis = f >> 1;
    int side = (f & 1) ? (n1 - 1) : 0;
    int i = (axis == 0) ? side : (a + 1);
    int j = (axis == 1) ? side : ((axis == 0) ? (a + 1) : (b + 1));
    int k = (axis == 2) ? side : (b + 1);
    faces_df[(static_cast<size_t>(f) * m + a) * m + b] =
        make_float2(buf[(static_cast<size_t>(i) * n1 + j) * n1 + k], 0.0f);
}

__global__ void split_last_mfirst_kernel(int B, int m, const float2* __restrict__ in,
                                         float2* __restrict__ B_sym, float2* __restrict__ B_anti) {
    int h_sym = (m + 1) / 2, h_anti = m / 2;
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= h_sym || b >= B) return;
    float2 a = in[static_cast<size_t>(b) * m + q];
    if (q < h_anti) {
        float2 c = in[static_cast<size_t>(b) * m + (m - 1 - q)];
        B_sym[static_cast<size_t>(b) * h_sym + q] = df_add(a, c);
        B_anti[static_cast<size_t>(b) * h_anti + q] = df_sub(a, c);
    } else {
        B_sym[static_cast<size_t>(b) * h_sym + q] = a;
    }
}

__global__ void merge_face_split_a_mfirst_kernel(int nfaces, int m,
                                                 const float2* __restrict__ Ce,
                                                 const float2* __restrict__ Co,
                                                 float2* __restrict__ B2_sym,
                                                 float2* __restrict__ B2_anti) {
    int h_sym = (m + 1) / 2, h_anti = m / 2;
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= h_sym || b >= nfaces * m) return;
    int f = b / m, r = b - f * m;
    const float2* src = ((r & 1) == 0) ? Ce : Co;
    int st = ((r & 1) == 0) ? h_sym : h_anti, rm = r >> 1;
    float2 v1 = src[(static_cast<size_t>(f) * m + a) * st + rm];
    if (a < h_anti) {
        float2 v2 = src[(static_cast<size_t>(f) * m + (m - 1 - a)) * st + rm];
        B2_sym[static_cast<size_t>(b) * h_sym + a] = df_add(v1, v2);
        B2_anti[static_cast<size_t>(b) * h_anti + a] = df_sub(v1, v2);
    } else {
        B2_sym[static_cast<size_t>(b) * h_sym + a] = v1;
    }
}

__global__ void merge_face_to_f64_mfirst_kernel(int nfaces, int m,
                                                const float2* __restrict__ Ce,
                                                const float2* __restrict__ Co,
                                                double* __restrict__ out) {
    int h_sym = (m + 1) / 2, h_anti = m / 2;
    size_t total = static_cast<size_t>(nfaces) * m * m;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int r = static_cast<int>(idx % m);
    int q = static_cast<int>((idx / m) % m);
    int f = static_cast<int>(idx / (static_cast<size_t>(m) * m));
    int b = f * m + r;
    float2 v = ((q & 1) == 0) ? Ce[static_cast<size_t>(b) * h_sym + (q >> 1)]
                              : Co[static_cast<size_t>(b) * h_anti + (q >> 1)];
    out[idx] = static_cast<double>(v.x) + static_cast<double>(v.y);
}

__global__ void init_harmonic_2d_kernel(int m, int L,
                                        const double* __restrict__ sin2_h,
                                        const double* __restrict__ cos_t,
                                        double* __restrict__ g_out, double* __restrict__ eg_out, double* __restrict__ inv_dg_out,
                                        double* __restrict__ e_out, double* __restrict__ ee_out, double* __restrict__ inv_de_out) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= m || r >= m) return;
    size_t idx = static_cast<size_t>(q) * m + r;
    double g = acosh(1.0 + 2.0 * (sin2_h[q] + sin2_h[r]));
    g_out[idx] = g;
    eg_out[idx] = exp(-g);
    inv_dg_out[idx] = 1.0 / (-expm1(-2.0 * g * static_cast<double>(L)));
    double e = acosh(5.0 + cos_t[q] + cos_t[r]);
    e_out[idx] = e;
    ee_out[idx] = exp(-e);
    inv_de_out[idx] = 1.0 / (-expm1(-2.0 * e * static_cast<double>(L)));
}

__global__ void build_harmonic_bucket_df64_kernel(
    int m, int L, int Q, int d_lo, int ns_near, int ns_total,
    const double* __restrict__ g_2d, const double* __restrict__ eg_2d, const double* __restrict__ inv_dg_2d,
    const double* __restrict__ e_2d, const double* __restrict__ ee_2d, const double* __restrict__ inv_de_2d,
    const double* __restrict__ faces_hat,
    float2* __restrict__ R_e, float2* __restrict__ R_o) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= Q || r >= Q) return;
    int Qe = (Q + 1) / 2, Qo = Q / 2;
    size_t qr = static_cast<size_t>(q) * m + r;
    size_t m2 = static_cast<size_t>(m) * m;
    float2 s0[3], sL[3], d0[3], dL[3];
    bool need_d = (d_lo <= 18);
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        double b00 = faces_hat[(2 * a + 0) * m2 + qr], b0L = faces_hat[(2 * a + 1) * m2 + qr];
        double b10 = faces_hat[(6 + 2 * a + 0) * m2 + qr], b1L = faces_hat[(6 + 2 * a + 1) * m2 + qr];
        s0[a] = d2df(0.5 * (b00 + b10));
        sL[a] = d2df(0.5 * (b0L + b1L));
        if (need_d) {
            d0[a] = d2df(0.5 * (b00 - b10));
            dL[a] = d2df(0.5 * (b0L - b1L));
        }
    }
    double g = g_2d[qr], eg_d = eg_2d[qr], idg_d = inv_dg_2d[qr];
    float2 eg = d2df(eg_d), inv_eg = d2df(1.0 / eg_d);
    float2 e_d    = d2df(exp(-g * static_cast<double>(d_lo)) * idg_d);
    float2 e_LpD  = d2df(exp(-g * static_cast<double>(L + d_lo)) * idg_d);
    float2 e_Lmd  = d2df(exp(-g * static_cast<double>(L - d_lo)) * idg_d);
    float2 e_2Lmd = d2df(exp(-g * static_cast<double>(2 * L - d_lo)) * idg_d);

    float2 ee = make_float2(0.0f, 0.0f), inv_ee = make_float2(1.0f, 0.0f);
    float2 f_d = make_float2(0.0f, 0.0f), f_LpD = make_float2(0.0f, 0.0f);
    float2 f_Lmd = make_float2(0.0f, 0.0f), f_2Lmd = make_float2(0.0f, 0.0f);
    if (need_d) {
        double e = e_2d[qr], ee_d = ee_2d[qr], ide_d = inv_de_2d[qr];
        ee = d2df(ee_d);
        inv_ee = d2df(1.0 / ee_d);
        f_d    = d2df(exp(-e * static_cast<double>(d_lo)) * ide_d);
        f_LpD  = d2df(exp(-e * static_cast<double>(L + d_lo)) * ide_d);
        f_Lmd  = d2df(exp(-e * static_cast<double>(L - d_lo)) * ide_d);
        f_2Lmd = d2df(exp(-e * static_cast<double>(2 * L - d_lo)) * ide_d);
    }
    int rm = r >> 1;
    bool is_even = ((r & 1) == 0);
    float2* dst = is_even ? R_e : R_o;
    int dst_st = is_even ? Qe : Qo;
    for (int s = 0; s < ns_near; ++s) {
        int d = d_lo + s;
        float2 hs_near = df_sub(e_d, e_2Lmd);
        float2 hs_far  = df_sub(e_Lmd, e_LpD);
        float2 hd_near = make_float2(0.0f, 0.0f), hd_far = make_float2(0.0f, 0.0f);
        if (need_d && d <= 18) {
            hd_near = df_sub(f_d, f_2Lmd);
            hd_far  = df_sub(f_Lmd, f_LpD);
            if (((d - 1) & 1) != 0)     { hd_near.x = -hd_near.x; hd_near.y = -hd_near.y; }
            if (((L - d - 1) & 1) != 0) { hd_far.x  = -hd_far.x;  hd_far.y  = -hd_far.y;  }
        }
        bool has_far = (2 * d < L);
        int s_far_idx = ns_total - 1 - s;
        #pragma unroll
        for (int a = 0; a < 3; ++a) {
            float2 v1 = df_add(df_mul(hs_near, s0[a]), df_mul(hs_far, sL[a]));
            if (need_d && d <= 18) v1 = df_add(v1, df_add(df_mul(hd_near, d0[a]), df_mul(hd_far, dL[a])));
            dst[((static_cast<size_t>(a) * ns_total + s) * Q + q) * dst_st + rm] = v1;
            if (has_far) {
                float2 v2 = df_add(df_mul(hs_far, s0[a]), df_mul(hs_near, sL[a]));
                if (need_d && d <= 18) v2 = df_add(v2, df_add(df_mul(hd_far, d0[a]), df_mul(hd_near, dL[a])));
                dst[((static_cast<size_t>(a) * ns_total + s_far_idx) * Q + q) * dst_st + rm] = v2;
            }
        }
        e_d = df_mul(e_d, eg); e_LpD = df_mul(e_LpD, eg);
        e_Lmd = df_mul(e_Lmd, inv_eg); e_2Lmd = df_mul(e_2Lmd, inv_eg);
        if (need_d && d < 18) {
            f_d = df_mul(f_d, ee); f_LpD = df_mul(f_LpD, ee);
            f_Lmd = df_mul(f_Lmd, inv_ee); f_2Lmd = df_mul(f_2Lmd, inv_ee);
        }
    }
}

__global__ void bucket_inv_r_to_q_mfirst_kernel(int m, int Q, int ns3,
                                                const float2* __restrict__ U_r,
                                                const float2* __restrict__ V_r,
                                                float2* __restrict__ Q_e,
                                                float2* __restrict__ Q_o) {
    int h_sym = (m + 1) / 2, h_anti = m / 2, Qe = (Q + 1) / 2, Qo = Q / 2;
    int z0 = blockIdx.x * 16, q0 = blockIdx.y * 16, as = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ float2 sm[16][17];
    int z = z0 + tx, q = q0 + ty;
    if (z < m && q < Q) {
        size_t row = static_cast<size_t>(as) * Q + q;
        float2 val;
        if (z < h_anti) val = df_add(U_r[row * h_sym + z], V_r[row * h_anti + z]);
        else if (z == h_anti && (m & 1)) val = U_r[row * h_sym + h_anti];
        else { int z2 = m - 1 - z; val = df_sub(U_r[row * h_sym + z2], V_r[row * h_anti + z2]); }
        sm[tx][ty] = val;
    }
    __syncthreads();
    int qw = q0 + tx, zw = z0 + ty;
    if (qw < Q && zw < m) {
        size_t ro = static_cast<size_t>(as) * m + zw;
        float2 val = sm[ty][tx];
        if ((qw & 1) == 0) Q_e[ro * Qe + (qw >> 1)] = val;
        else               Q_o[ro * Qo + (qw >> 1)] = val;
    }
}

__global__ void ax2_transpose_add_32x32_kernel(int m, int s_off, int k_start, int cnt,
                                               const float2* __restrict__ ax2_tmp,
                                               float2* __restrict__ v_df) {
    int j0 = blockIdx.x * 32, dk0 = blockIdx.y * 32, i = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ float2 sm[32][33];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        int dk = dk0 + ty + u * 8, j = j0 + tx;
        if (j < m && dk < cnt) sm[ty + u * 8][tx] = ax2_tmp[(static_cast<size_t>(s_off + dk) * m + i) * m + j];
    }
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        int jw = j0 + ty + u * 8, dkw = dk0 + tx;
        if (dkw < cnt && jw < m) {
            size_t p = (static_cast<size_t>(i) * m + jw) * m + (k_start + dkw);
            v_df[p] = df_add(v_df[p], sm[tx][ty + u * 8]);
        }
    }
}

// ============================================================================
// Persistent Solver State
// ============================================================================
struct QLevelMats {
    int Q = 0;
    OzakiMatrixStacked inv_even, inv_odd;
    float2 *d_Se_df = nullptr, *d_So_df = nullptr;
    int destroy() {
        int rc = 0;
        rc |= inv_even.destroy();
        rc |= inv_odd.destroy();
        if (d_Se_df) { if (cudaFree(d_Se_df) != cudaSuccess) rc = 1; d_Se_df = nullptr; }
        if (d_So_df) { if (cudaFree(d_So_df) != cudaSuccess) rc = 1; d_So_df = nullptr; }
        Q = 0;
        return rc;
    }
};

struct HarmonicBucket {
    int q_idx = 0, Q = 0, d_lo = 0, d_hi = 0, ns_near = 0, ns_far = 0, ns_total = 0;
    int* d_xlist = nullptr;
};

enum SolverMode {
    MODE_NONE = 0,
    MODE_NOOP = 1,
    MODE_DIRECT_COOP = 2,
    MODE_DIRECT_2J = 3,
    MODE_SPECTRAL = 4
};

struct SolverState {
    bool initialized = false;
    bool uploaded = false;
    int n0 = 0, n1 = 0, m = 0, L = 0;
    uint64_t nelems = 0;
    uint64_t nbytes = 0;
    SolverMode mode = MODE_NONE;

    float* d_buf0 = nullptr;
    float* d_buf1 = nullptr;
    cudaStream_t s0 = nullptr;
    cudaStream_t s1 = nullptr;
    cudaEvent_t ev_fork = nullptr;
    cudaEvent_t ev_join = nullptr;

    // Direct Path 1 state
    int coop_blocks = 0;
    bool direct_aligned4 = false;
    int direct_zc = 32;
    cudaGraph_t direct_graph = nullptr;
    cudaGraphExec_t direct_graph_exec = nullptr;

    // Spectral Path 2 state
    int K = 0, h_sym = 0, h_anti = 0, K_even = 0, K_odd = 0;
    int max_chunk_n = 0;
    cublasHandle_t h0 = nullptr;
    cublasHandle_t h1 = nullptr;
    cudaGraph_t spec_graph = nullptr;
    cudaGraphExec_t spec_graph_exec = nullptr;

    double *d_s1 = nullptr, *d_sin2_h = nullptr, *d_cos_t = nullptr;
    double *d_g2d = nullptr, *d_eg2d = nullptr, *d_idg2d = nullptr;
    double *d_e2d = nullptr, *d_ee2d = nullptr, *d_ide2d = nullptr;
    OzakiMatrixStacked fwd_full_e, fwd_full_o, fwd_K1_e, fwd_K1_o;
    OzakiMatrixStacked fwd_K_e, fwd_K_o, inv_K_e, inv_K_o;
    std::vector<QLevelMats> q_mats;
    std::vector<HarmonicBucket> buckets;
    float2 *d_wA = nullptr, *d_wB = nullptr, *d_wC = nullptr, *d_wD = nullptr;
    float2 *d_vdf = nullptr, *d_ax2 = nullptr;
    double *d_faces_hat = nullptr;
    __nv_bfloat16 *d_Bs_e = nullptr, *d_Bs_o = nullptr;
    float *d_Cse = nullptr, *d_Cso = nullptr;
    float *d_max_e = nullptr, *d_max_o = nullptr;
    float *d_sc_e = nullptr, *d_sc_o = nullptr;
};

static SolverState g_st;

template<typename T>
static int free_dev(T*& ptr) {
    if (!ptr) return 0;
    cudaError_t e = cudaFree(ptr);
    ptr = nullptr;
    return (e == cudaSuccess) ? 0 : 1;
}

static int destroy_spectral_resources(SolverState& st) {
    int rc = 0;
    if (st.spec_graph_exec) { if (cudaGraphExecDestroy(st.spec_graph_exec) != cudaSuccess) rc = 1; st.spec_graph_exec = nullptr; }
    if (st.spec_graph)      { if (cudaGraphDestroy(st.spec_graph) != cudaSuccess) rc = 1;          st.spec_graph = nullptr; }
    if (st.h0)              { if (cublasDestroy(st.h0) != CUBLAS_STATUS_SUCCESS) rc = 1;           st.h0 = nullptr; }
    if (st.h1)              { if (cublasDestroy(st.h1) != CUBLAS_STATUS_SUCCESS) rc = 1;           st.h1 = nullptr; }
    if (st.ev_fork)         { if (cudaEventDestroy(st.ev_fork) != cudaSuccess) rc = 1;             st.ev_fork = nullptr; }
    if (st.ev_join)         { if (cudaEventDestroy(st.ev_join) != cudaSuccess) rc = 1;             st.ev_join = nullptr; }
    if (st.s1)              { if (cudaStreamDestroy(st.s1) != cudaSuccess) rc = 1;                 st.s1 = nullptr; }

    rc |= free_dev(st.d_s1);    rc |= free_dev(st.d_sin2_h); rc |= free_dev(st.d_cos_t);
    rc |= free_dev(st.d_g2d);   rc |= free_dev(st.d_eg2d);   rc |= free_dev(st.d_idg2d);
    rc |= free_dev(st.d_e2d);   rc |= free_dev(st.d_ee2d);   rc |= free_dev(st.d_ide2d);

    rc |= st.fwd_full_e.destroy(); rc |= st.fwd_full_o.destroy();
    rc |= st.fwd_K1_e.destroy();   rc |= st.fwd_K1_o.destroy();
    rc |= st.fwd_K_e.destroy();    rc |= st.fwd_K_o.destroy();
    rc |= st.inv_K_e.destroy();    rc |= st.inv_K_o.destroy();

    for (auto& qm : st.q_mats) rc |= qm.destroy();
    st.q_mats.clear();
    for (auto& bk : st.buckets) rc |= free_dev(bk.d_xlist);
    st.buckets.clear();

    rc |= free_dev(st.d_wA); rc |= free_dev(st.d_wB); rc |= free_dev(st.d_wC); rc |= free_dev(st.d_wD);
    rc |= free_dev(st.d_vdf); rc |= free_dev(st.d_ax2); rc |= free_dev(st.d_faces_hat);
    rc |= free_dev(st.d_Bs_e); rc |= free_dev(st.d_Bs_o);
    rc |= free_dev(st.d_Cse); rc |= free_dev(st.d_Cso);
    rc |= free_dev(st.d_max_e); rc |= free_dev(st.d_max_o);
    rc |= free_dev(st.d_sc_e); rc |= free_dev(st.d_sc_o);
    return rc;
}

static int teardown_all(SolverState& st) {
    int rc = 0;
    rc |= destroy_spectral_resources(st);
    if (st.direct_graph_exec) { if (cudaGraphExecDestroy(st.direct_graph_exec) != cudaSuccess) rc = 1; st.direct_graph_exec = nullptr; }
    if (st.direct_graph)      { if (cudaGraphDestroy(st.direct_graph) != cudaSuccess) rc = 1;          st.direct_graph = nullptr; }
    if (st.s0)                { if (cudaStreamDestroy(st.s0) != cudaSuccess) rc = 1;                   st.s0 = nullptr; }
    rc |= free_dev(st.d_buf0);
    rc |= free_dev(st.d_buf1);
    st.initialized = false;
    st.uploaded = false;
    st.mode = MODE_NONE;
    st.n0 = st.n1 = st.m = st.L = 0;
    st.nelems = st.nbytes = 0;
    return rc;
}

static int enqueue_spectral(SolverState& st) {
    int n0 = st.n0, n1 = st.n1, m = st.m, L = st.L, K = st.K;
    int h_sym = st.h_sym;
    size_t m2 = static_cast<size_t>(m) * m;
    dim3 blk16(16, 16);

    // 1. 2D DST-I of all 12 boundary faces
    dim3 grd_f((m + 15) / 16, (m + 15) / 16, 6);
    extract_6faces_kernel<<<grd_f, blk16, 0, st.s0>>>(n1, m, st.d_buf0, st.d_wA);
    extract_6faces_kernel<<<grd_f, blk16, 0, st.s0>>>(n1, m, st.d_buf1, st.d_wA + 6 * m2);
    int Bf = 12 * m;
    dim3 grd_sf((h_sym + 15) / 16, (Bf + 15) / 16);
    split_last_mfirst_kernel<<<grd_sf, blk16, 0, st.s0>>>(Bf, m, st.d_wA, st.d_wB, st.d_wC);
    if (ozaki_eo_pair_stacked(st.h0, st.h1, st.fwd_full_e, st.fwd_full_o, Bf,
                              st.d_wB, st.d_wC, st.d_wA, st.d_wD,
                              st.d_Bs_e, st.d_Bs_o, st.d_Cse, st.d_Cso,
                              st.d_max_e, st.d_max_o, st.d_sc_e, st.d_sc_o,
                              st.max_chunk_n, st.s0, st.s1, st.ev_fork, st.ev_join) != 0) return 1;
    merge_face_split_a_mfirst_kernel<<<grd_sf, blk16, 0, st.s0>>>(12, m, st.d_wA, st.d_wD, st.d_wB, st.d_wC);
    if (ozaki_eo_pair_stacked(st.h0, st.h1, st.fwd_full_e, st.fwd_full_o, Bf,
                              st.d_wB, st.d_wC, st.d_wA, st.d_wD,
                              st.d_Bs_e, st.d_Bs_o, st.d_Cse, st.d_Cso,
                              st.d_max_e, st.d_max_o, st.d_sc_e, st.d_sc_o,
                              st.max_chunk_n, st.s0, st.s1, st.ev_fork, st.ev_join) != 0) return 1;
    merge_face_to_f64_mfirst_kernel<<<(12 * m2 + 255) / 256, 256, 0, st.s0>>>(12, m, st.d_wA, st.d_wD, st.d_faces_hat);

    // 2. Mode-truncated 3D Forward DST-I (m^3 -> K^3)
    if (ozaki_fwd_u0_stacked(st.h0, st.h1, n1, m, st.fwd_K1_e, st.fwd_K1_o,
                             st.d_buf0, st.d_wC, st.d_wD,
                             st.d_Bs_e, st.d_Bs_o, st.d_Cse, st.d_Cso,
                             st.d_max_e, st.d_sc_e, st.max_chunk_n,
                             st.s0, st.s1, st.ev_fork, st.ev_join) != 0) return 1;
    fwd_k_to_j_kernel<<<dim3((h_sym + 15) / 16, (K + 15) / 16, m), blk16, 0, st.s0>>>(m, K, st.d_wC, st.d_wD, st.d_wA, st.d_wB);
    int b2 = m * K;
    if (ozaki_eo_pair_stacked(st.h0, st.h1, st.fwd_K_e, st.fwd_K_o, b2,
                              st.d_wA, st.d_wB, st.d_wC, st.d_wD,
                              st.d_Bs_e, st.d_Bs_o, st.d_Cse, st.d_Cso,
                              st.d_max_e, st.d_max_o, st.d_sc_e, st.d_sc_o,
                              st.max_chunk_n, st.s0, st.s1, st.ev_fork, st.ev_join) != 0) return 1;
    int b3 = K * K;
    fwd_j_to_i_kernel<<<(static_cast<size_t>(b3) * h_sym + 255) / 256, 256, 0, st.s0>>>(m, K, st.d_wC, st.d_wD, st.d_wA, st.d_wB);
    if (ozaki_eo_pair_stacked(st.h0, st.h1, st.fwd_K_e, st.fwd_K_o, b3,
                              st.d_wA, st.d_wB, st.d_wC, st.d_wD,
                              st.d_Bs_e, st.d_Bs_o, st.d_Cse, st.d_Cso,
                              st.d_max_e, st.d_max_o, st.d_sc_e, st.d_sc_o,
                              st.max_chunk_n, st.s0, st.s1, st.ev_fork, st.ev_join) != 0) return 1;

    // 3. Exact Spectral Update on K^3 modes: A^(2*n0 - 1) * (u0 - u_inf)
    spectral_update_K3_mfirst_kernel<<<(static_cast<size_t>(K) * K * K + 255) / 256, 256, 0, st.s0>>>(
        m, K, n0, st.d_s1, st.d_sin2_h, st.d_faces_hat, st.d_wC, st.d_wD, st.d_wA, st.d_wB);

    // 4. Mode-truncated 3D Inverse DST-I (K^3 -> m^3 into d_vdf)
    if (ozaki_eo_pair_stacked(st.h0, st.h1, st.inv_K_e, st.inv_K_o, b3,
                              st.d_wA, st.d_wB, st.d_wC, st.d_wD,
                              st.d_Bs_e, st.d_Bs_o, st.d_Cse, st.d_Cso,
                              st.d_max_e, st.d_max_o, st.d_sc_e, st.d_sc_o,
                              st.max_chunk_n, st.s0, st.s1, st.ev_fork, st.ev_join) != 0) return 1;
    inv_i_to_j_mfirst_kernel<<<(static_cast<size_t>(m) * K * K + 255) / 256, 256, 0, st.s0>>>(m, K, st.d_wC, st.d_wD, st.d_wA, st.d_wB);
    if (ozaki_eo_pair_stacked(st.h0, st.h1, st.inv_K_e, st.inv_K_o, b2,
                              st.d_wA, st.d_wB, st.d_wC, st.d_wD,
                              st.d_Bs_e, st.d_Bs_o, st.d_Cse, st.d_Cso,
                              st.d_max_e, st.d_max_o, st.d_sc_e, st.d_sc_o,
                              st.max_chunk_n, st.s0, st.s1, st.ev_fork, st.ev_join) != 0) return 1;
    inv_j_to_k_mfirst_kernel<<<dim3((m + 15) / 16, (K + 15) / 16, m), blk16, 0, st.s0>>>(m, K, st.d_wC, st.d_wD, st.d_wA, st.d_wB);
    if (ozaki_inv_k_stacked_fused(st.h0, st.h1, m, st.inv_K_e, st.inv_K_o,
                                  st.d_wA, st.d_wB, st.d_vdf,
                                  st.d_Bs_e, st.d_Bs_o, st.d_Cse, st.d_Cso,
                                  st.d_max_e, st.d_max_o, st.d_sc_e, st.d_sc_o,
                                  st.max_chunk_n, st.s0, st.s1, st.ev_fork, st.ev_join) != 0) return 1;

    // 5. Analytic Harmonic Steady-State v_inf accumulated into d_vdf
    for (const auto& bk : st.buckets) {
        const auto& qm = st.q_mats[bk.q_idx];
        int Q = bk.Q, ns = bk.ns_total, ns3 = 3 * ns, B_in = ns3 * Q;
        build_harmonic_bucket_df64_kernel<<<dim3((Q + 15) / 16, (Q + 15) / 16), blk16, 0, st.s0>>>(
            m, L, Q, bk.d_lo, bk.ns_near, ns,
            st.d_g2d, st.d_eg2d, st.d_idg2d, st.d_e2d, st.d_ee2d, st.d_ide2d,
            st.d_faces_hat, st.d_wA, st.d_wB);

        if (qm.d_Se_df != nullptr) {
            dim3 blk_eo(32, 8);
            dim3 grd_r((h_sym + 31) / 32, (B_in + 31) / 32);
            dim3 grd_q((m + 31) / 32, (h_sym + 31) / 32, ns);
            if (Q == 16) {
                fused_eo_df64_inv_r_to_TeTo_kernel<8, 8><<<grd_r, blk_eo, 0, st.s0>>>(
                    m, Q, ns3, qm.d_Se_df, qm.d_So_df, st.d_wA, st.d_wB, st.d_wC, st.d_wD);
                for (int axis = 0; axis < 3; ++axis)
                    fused_eo_df64_q_accum_tiled_kernel<8, 8><<<grd_q, blk_eo, 0, st.s0>>>(
                        m, ns, axis, bk.d_xlist, qm.d_Se_df, qm.d_So_df, st.d_wC, st.d_wD, st.d_vdf, st.d_ax2);
            } else if (Q == 24) {
                fused_eo_df64_inv_r_to_TeTo_kernel<12, 12><<<grd_r, blk_eo, 0, st.s0>>>(
                    m, Q, ns3, qm.d_Se_df, qm.d_So_df, st.d_wA, st.d_wB, st.d_wC, st.d_wD);
                for (int axis = 0; axis < 3; ++axis)
                    fused_eo_df64_q_accum_tiled_kernel<12, 12><<<grd_q, blk_eo, 0, st.s0>>>(
                        m, ns, axis, bk.d_xlist, qm.d_Se_df, qm.d_So_df, st.d_wC, st.d_wD, st.d_vdf, st.d_ax2);
            } else {
                fused_eo_df64_inv_r_to_TeTo_kernel<16, 16><<<grd_r, blk_eo, 0, st.s0>>>(
                    m, Q, ns3, qm.d_Se_df, qm.d_So_df, st.d_wA, st.d_wB, st.d_wC, st.d_wD);
                for (int axis = 0; axis < 3; ++axis)
                    fused_eo_df64_q_accum_tiled_kernel<16, 16><<<grd_q, blk_eo, 0, st.s0>>>(
                        m, ns, axis, bk.d_xlist, qm.d_Se_df, qm.d_So_df, st.d_wC, st.d_wD, st.d_vdf, st.d_ax2);
            }
        } else {
            if (ozaki_eo_pair_stacked(st.h0, st.h1, qm.inv_even, qm.inv_odd, B_in,
                                      st.d_wA, st.d_wB, st.d_wC, st.d_wD,
                                      st.d_Bs_e, st.d_Bs_o, st.d_Cse, st.d_Cso,
                                      st.d_max_e, st.d_max_o, st.d_sc_e, st.d_sc_o,
                                      st.max_chunk_n, st.s0, st.s1, st.ev_fork, st.ev_join) != 0) return 1;
            bucket_inv_r_to_q_mfirst_kernel<<<dim3((m + 15) / 16, (Q + 15) / 16, ns3), blk16, 0, st.s0>>>(
                m, Q, ns3, st.d_wC, st.d_wD, st.d_wA, st.d_wB);
            if (ozaki_bucket_q_stacked_fused(st.h0, st.h1, m, ns, bk.d_xlist, qm.inv_even, qm.inv_odd,
                                             st.d_wA, st.d_wB, st.d_vdf, st.d_ax2,
                                             st.d_Bs_e, st.d_Bs_o, st.d_Cse, st.d_Cso,
                                             st.d_max_e, st.d_max_o, st.d_sc_e, st.d_sc_o,
                                             st.max_chunk_n, st.s0, st.s1, st.ev_fork, st.ev_join) != 0) return 1;
        }

        dim3 blk_tr(32, 8);
        ax2_transpose_add_32x32_kernel<<<dim3((m + 31) / 32, (bk.ns_near + 31) / 32, m), blk_tr, 0, st.s0>>>(
            m, 0, bk.d_lo - 1, bk.ns_near, st.d_ax2, st.d_vdf);
        if (bk.ns_far > 0) {
            int far_lo = std::max(bk.d_hi + 1, L - bk.d_hi);
            ax2_transpose_add_32x32_kernel<<<dim3((m + 31) / 32, (bk.ns_far + 31) / 32, m), blk_tr, 0, st.s0>>>(
                m, bk.ns_near, far_lo - 1, bk.ns_far, st.d_ax2, st.d_vdf);
        }
    }

    // 6. Finalize: round d_vdf -> buf1 interior and apply final DF64 stencil -> buf0 interior
    int zc = (m <= 512) ? 16 : 32;
    finalize_fast_v2_kernel<<<dim3(((m + 1) / 2 + 31) / 32, ((m + 1) / 2 + 7) / 8, (m + zc - 1) / zc),
                              dim3(32, 8), 0, st.s0>>>(n1, m, zc, st.d_vdf, st.d_buf1, st.d_buf0);
    CUDA_TRY(cudaGetLastError());
    return 0;
}

static int setup_spectral(SolverState& st) {
    int n0 = st.n0, m = st.m, L = st.L;
    st.h_sym = (m + 1) / 2;
    st.h_anti = m / 2;

    double arg = 1.0 - exp(-28.0 / (2.0 * static_cast<double>(n0) - 1.0));
    if (n0 < 20 || 2.0 * arg >= 1.0) st.K = m;
    else st.K = std::min(m, align16(static_cast<int>(ceil((2.0 * L / kPi) * asin(sqrt(2.0 * arg)))) + 2));
    st.K_even = (st.K + 1) / 2;
    st.K_odd = st.K / 2;

    const int cand_q[] = {16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536};
    std::vector<int> q_levels;
    for (int q : cand_q) if (q < m) q_levels.push_back(q);
    q_levels.push_back(m);

    int depth_max = (m + 1) / 2;
    std::vector<int> d_qi(depth_max + 1);
    for (int d = 1; d <= depth_max; ++d) {
        double sh = sinh(24.0 / (2.0 * d));
        int reqQ = (sh >= 1.0) ? m : std::min(m, static_cast<int>(ceil((2.0 * L / kPi) * asin(sh))) + 2);
        if (d <= 14) reqQ = m;
        size_t bi = 0;
        while (bi + 1 < q_levels.size() && q_levels[bi] < reqQ) ++bi;
        d_qi[d] = static_cast<int>(bi);
    }

    int max_d_step = (m > 1024) ? 16 : 64;
    size_t max_b_elems = 0;
    int max_ns_total = 0;
    std::vector<HarmonicBucket> tmp_buckets;
    std::vector<std::vector<int>> tmp_xlists;
    for (int d = 1; d <= depth_max; ) {
        int qi = d_qi[d], Q = q_levels[qi], d_lo = d;
        while (d <= depth_max && d_qi[d] == qi && (d - d_lo) < max_d_step) ++d;
        int d_hi = d - 1;
        HarmonicBucket bk;
        bk.q_idx = qi; bk.Q = Q; bk.d_lo = d_lo; bk.d_hi = d_hi; bk.ns_near = d_hi - d_lo + 1;
        std::vector<int> xl;
        for (int x = d_lo; x <= d_hi; ++x) xl.push_back(x);
        int far_lo = std::max(d_hi + 1, L - d_hi), far_hi = L - d_lo;
        bk.ns_far = std::max(0, far_hi - far_lo + 1);
        for (int x = far_lo; x <= far_hi; ++x) xl.push_back(x);
        bk.ns_total = static_cast<int>(xl.size());
        max_ns_total = std::max(max_ns_total, bk.ns_total);
        int Qe = (Q + 1) / 2;
        size_t b_elems = std::max({static_cast<size_t>(3) * bk.ns_total * Q * st.h_sym,
                                   static_cast<size_t>(3) * bk.ns_total * m * Qe,
                                   static_cast<size_t>(3) * bk.ns_total * Q * Qe});
        max_b_elems = std::max(max_b_elems, b_elems);
        tmp_buckets.push_back(bk);
        tmp_xlists.push_back(std::move(xl));
    }

    size_t m2 = static_cast<size_t>(m) * m;
    size_t m3 = m2 * m;
    size_t sz_ab = std::max({static_cast<size_t>(st.K_even) * m2,
                             static_cast<size_t>(st.K) * m * st.h_sym,
                             max_b_elems,
                             static_cast<size_t>(12) * m2,
                             static_cast<size_t>(16)});
    int max_chunk_n = std::min(65536, std::max({12 * m, static_cast<int>(std::min<size_t>(m2, 65536)),
                                                3 * max_ns_total * m, 1}));
    int max_mk = align8(std::max({st.h_sym, st.K_even, 1}));
    int max_S = (st.h_sym <= 512) ? 6 : 7;

    // Estimate total device memory needed for Path 2 before allocating
    size_t est_bytes = 0;
    est_bytes += 4 * sz_ab * sizeof(float2);
    est_bytes += m3 * sizeof(float2);
    est_bytes += static_cast<size_t>(max_ns_total) * m2 * sizeof(float2);
    est_bytes += static_cast<size_t>(18) * m2 * sizeof(double); // g2d..ide2d + faces_hat
    est_bytes += 2 * static_cast<size_t>(max_S) * max_mk * max_chunk_n * (sizeof(__nv_bfloat16) + sizeof(float));
    est_bytes += OzakiMatrixStacked::estimate_bytes(st.h_sym, st.h_sym, false, true, false);
    est_bytes += OzakiMatrixStacked::estimate_bytes(st.h_anti, st.h_anti, false, true, false);
    est_bytes += OzakiMatrixStacked::estimate_bytes(st.K_even, st.h_sym, false, false, true);
    est_bytes += OzakiMatrixStacked::estimate_bytes(st.K_odd, st.h_anti, false, false, true);
    est_bytes += OzakiMatrixStacked::estimate_bytes(st.K_even, st.h_sym, false, false, false);
    est_bytes += OzakiMatrixStacked::estimate_bytes(st.K_odd, st.h_anti, false, false, false);
    est_bytes += OzakiMatrixStacked::estimate_bytes(st.K_even, st.h_sym, true, false, false);
    est_bytes += OzakiMatrixStacked::estimate_bytes(st.K_odd, st.h_anti, true, false, false);
    for (int Q : q_levels) {
        int Qe = (Q + 1) / 2, Qo = Q / 2;
        bool use_eo = (Q == 16 || Q == 24 || Q == 32) && Q <= m;
        if (use_eo) {
            est_bytes += (static_cast<size_t>(st.h_sym) * Qe + static_cast<size_t>(st.h_anti) * Qo) * sizeof(float2);
        } else {
            bool hp = (Q > 256 || m <= 256);
            est_bytes += OzakiMatrixStacked::estimate_bytes(Qe, st.h_sym, true, hp, false);
            est_bytes += OzakiMatrixStacked::estimate_bytes(Qo, st.h_anti, true, hp, false);
        }
    }

    size_t free_mem = 0, total_mem = 0;
    CUDA_TRY(cudaMemGetInfo(&free_mem, &total_mem));
    size_t safety_margin = (1ULL << 30); // 1 GiB headroom for cuBLAS workspaces & graph
    if (est_bytes + safety_margin >= free_mem) {
        return 2; // Signal caller to fall back to Path 1
    }

    st.max_chunk_n = max_chunk_n;
    CUDA_TRY(cudaStreamCreate(&st.s1));
    CUDA_TRY(cudaEventCreateWithFlags(&st.ev_fork, cudaEventDisableTiming));
    CUDA_TRY(cudaEventCreateWithFlags(&st.ev_join, cudaEventDisableTiming));
    CUBLAS_TRY(cublasCreate(&st.h0));
    CUBLAS_TRY(cublasCreate(&st.h1));
    CUBLAS_TRY(cublasSetStream(st.h0, st.s0));
    CUBLAS_TRY(cublasSetStream(st.h1, st.s1));

    double sc = sqrt(2.0 / static_cast<double>(L));
    std::vector<double> h_s1(m), h_sin2(m), h_cos(m);
    for (int p = 0; p < m; ++p) {
        h_s1[p] = sc * sin(kPi * (p + 1) / static_cast<double>(L));
        double sh = sin(0.5 * kPi * (p + 1) / static_cast<double>(L));
        h_sin2[p] = sh * sh;
        h_cos[p] = cos(kPi * (p + 1) / static_cast<double>(L));
    }
    CUDA_TRY(cudaMalloc(&st.d_s1, static_cast<size_t>(m) * sizeof(double)));
    CUDA_TRY(cudaMalloc(&st.d_sin2_h, static_cast<size_t>(m) * sizeof(double)));
    CUDA_TRY(cudaMalloc(&st.d_cos_t, static_cast<size_t>(m) * sizeof(double)));
    CUDA_TRY(cudaMemcpyAsync(st.d_s1, h_s1.data(), static_cast<size_t>(m) * sizeof(double), cudaMemcpyHostToDevice, st.s0));
    CUDA_TRY(cudaMemcpyAsync(st.d_sin2_h, h_sin2.data(), static_cast<size_t>(m) * sizeof(double), cudaMemcpyHostToDevice, st.s0));
    CUDA_TRY(cudaMemcpyAsync(st.d_cos_t, h_cos.data(), static_cast<size_t>(m) * sizeof(double), cudaMemcpyHostToDevice, st.s0));

    CUDA_TRY(cudaMalloc(&st.d_g2d, m2 * sizeof(double)));
    CUDA_TRY(cudaMalloc(&st.d_eg2d, m2 * sizeof(double)));
    CUDA_TRY(cudaMalloc(&st.d_idg2d, m2 * sizeof(double)));
    CUDA_TRY(cudaMalloc(&st.d_e2d, m2 * sizeof(double)));
    CUDA_TRY(cudaMalloc(&st.d_ee2d, m2 * sizeof(double)));
    CUDA_TRY(cudaMalloc(&st.d_ide2d, m2 * sizeof(double)));
    init_harmonic_2d_kernel<<<dim3((m + 15) / 16, (m + 15) / 16), dim3(16, 16), 0, st.s0>>>(
        m, L, st.d_sin2_h, st.d_cos_t, st.d_g2d, st.d_eg2d, st.d_idg2d, st.d_e2d, st.d_ee2d, st.d_ide2d);
    CUDA_TRY(cudaGetLastError());

    if (st.fwd_full_e.init_dst(L, st.h_sym, st.h_sym, 0, false, true, false, st.s0) != 0) return 1;
    if (st.fwd_full_o.init_dst(L, st.h_anti, st.h_anti, 1, false, true, false, st.s0) != 0) return 1;
    if (st.fwd_K1_e.init_dst(L, st.K_even, st.h_sym, 0, false, false, true, st.s0) != 0) return 1;
    if (st.fwd_K1_o.init_dst(L, st.K_odd, st.h_anti, 1, false, false, true, st.s0) != 0) return 1;
    if (st.fwd_K_e.init_dst(L, st.K_even, st.h_sym, 0, false, false, false, st.s0) != 0) return 1;
    if (st.fwd_K_o.init_dst(L, st.K_odd, st.h_anti, 1, false, false, false, st.s0) != 0) return 1;
    if (st.inv_K_e.init_dst(L, st.K_even, st.h_sym, 0, true, false, false, st.s0) != 0) return 1;
    if (st.inv_K_o.init_dst(L, st.K_odd, st.h_anti, 1, true, false, false, st.s0) != 0) return 1;

    st.q_mats.resize(q_levels.size());
    for (size_t qi = 0; qi < q_levels.size(); ++qi) {
        int Q = q_levels[qi], Qe = (Q + 1) / 2, Qo = Q / 2;
        st.q_mats[qi].Q = Q;
        bool use_eo = (Q == 16 || Q == 24 || Q == 32) && Q <= m;
        if (use_eo) {
            CUDA_TRY(cudaMalloc(&st.q_mats[qi].d_Se_df, static_cast<size_t>(st.h_sym) * Qe * sizeof(float2)));
            CUDA_TRY(cudaMalloc(&st.q_mats[qi].d_So_df, static_cast<size_t>(st.h_anti) * Qo * sizeof(float2)));
            build_dst_df64_kernel<<<(static_cast<size_t>(st.h_sym) * Qe + 255) / 256, 256, 0, st.s0>>>(
                L, st.h_sym, Qe, 0, sc, st.q_mats[qi].d_Se_df);
            if (st.h_anti > 0 && Qo > 0) {
                build_dst_df64_kernel<<<(static_cast<size_t>(st.h_anti) * Qo + 255) / 256, 256, 0, st.s0>>>(
                    L, st.h_anti, Qo, 1, sc, st.q_mats[qi].d_So_df);
            }
            CUDA_TRY(cudaGetLastError());
        } else {
            bool hp = (Q > 256 || m <= 256);
            if (st.q_mats[qi].inv_even.init_dst(L, Qe, st.h_sym, 0, true, hp, false, st.s0) != 0) return 1;
            if (st.q_mats[qi].inv_odd.init_dst(L, Qo, st.h_anti, 1, true, hp, false, st.s0) != 0) return 1;
        }
    }

    st.buckets = std::move(tmp_buckets);
    for (size_t bi = 0; bi < st.buckets.size(); ++bi) {
        size_t xbytes = tmp_xlists[bi].size() * sizeof(int);
        CUDA_TRY(cudaMalloc(&st.buckets[bi].d_xlist, xbytes));
        CUDA_TRY(cudaMemcpyAsync(st.buckets[bi].d_xlist, tmp_xlists[bi].data(), xbytes, cudaMemcpyHostToDevice, st.s0));
    }

    CUDA_TRY(cudaMalloc(&st.d_wA, sz_ab * sizeof(float2)));
    CUDA_TRY(cudaMalloc(&st.d_wB, sz_ab * sizeof(float2)));
    CUDA_TRY(cudaMalloc(&st.d_wC, sz_ab * sizeof(float2)));
    CUDA_TRY(cudaMalloc(&st.d_wD, sz_ab * sizeof(float2)));
    CUDA_TRY(cudaMalloc(&st.d_vdf, m3 * sizeof(float2)));
    CUDA_TRY(cudaMalloc(&st.d_ax2, static_cast<size_t>(max_ns_total) * m2 * sizeof(float2)));
    CUDA_TRY(cudaMalloc(&st.d_faces_hat, static_cast<size_t>(12) * m2 * sizeof(double)));
    CUDA_TRY(cudaMalloc(&st.d_max_e, sizeof(float)));
    CUDA_TRY(cudaMalloc(&st.d_max_o, sizeof(float)));
    CUDA_TRY(cudaMalloc(&st.d_sc_e, 2 * sizeof(float)));
    CUDA_TRY(cudaMalloc(&st.d_sc_o, 2 * sizeof(float)));

    size_t sz_bs = static_cast<size_t>(max_S) * max_mk * max_chunk_n * sizeof(__nv_bfloat16);
    size_t sz_cs = static_cast<size_t>(max_S) * max_mk * max_chunk_n * sizeof(float);
    CUDA_TRY(cudaMalloc(&st.d_Bs_e, sz_bs));
    CUDA_TRY(cudaMalloc(&st.d_Bs_o, sz_bs));
    CUDA_TRY(cudaMalloc(&st.d_Cse, sz_cs));
    CUDA_TRY(cudaMalloc(&st.d_Cso, sz_cs));

    // Zero device buffers for safe warmup & graph capture (pb_upload overwrites them before compute)
    CUDA_TRY(cudaMemsetAsync(st.d_buf0, 0, st.nbytes, st.s0));
    CUDA_TRY(cudaMemsetAsync(st.d_buf1, 0, st.nbytes, st.s0));

    // Warm cuBLAS workspaces and kernels before capturing CUDA Graph
    if (enqueue_spectral(st) != 0) return 1;
    CUDA_TRY(cudaStreamSynchronize(st.s0));

    CUDA_TRY(cudaStreamBeginCapture(st.s0, cudaStreamCaptureModeThreadLocal));
    int enq_rc = enqueue_spectral(st);
    cudaError_t end_err = cudaStreamEndCapture(st.s0, &st.spec_graph);
    if (enq_rc != 0 || end_err != cudaSuccess) return 1;
    CUDA_TRY(cudaGraphInstantiate(&st.spec_graph_exec, st.spec_graph, nullptr, nullptr, 0));

    // Warm graph launch once on zeroed buffers
    CUDA_TRY(cudaGraphLaunch(st.spec_graph_exec, st.s0));
    CUDA_TRY(cudaStreamSynchronize(st.s0));
    return 0;
}

static int setup_direct(SolverState& st) {
    int n0 = st.n0, n1 = st.n1, m = st.m;
    int dev = 0;
    CUDA_TRY(cudaGetDevice(&dev));
    int coop_ok = 0;
    CUDA_TRY(cudaDeviceGetAttribute(&coop_ok, cudaDevAttrCooperativeLaunch, dev));

    if (coop_ok && n1 <= 128) {
        int sm_count = 0, max_blk_per_sm = 0;
        CUDA_TRY(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
        CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk_per_sm, stencil_coop_l2_kernel, 256, 0));
        int max_coop = max_blk_per_sm * sm_count;
        if (max_coop > 0) {
            size_t nrows = static_cast<size_t>(m) * m;
            size_t needed = (nrows + 7) / 8; // 8 warps per 256-thread block
            st.coop_blocks = static_cast<int>(std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(max_coop), needed)));
            st.mode = MODE_DIRECT_COOP;
            // Warm kernel with 0 steps? Run 1 step on zeroed buffers to ensure JIT/module loaded
            CUDA_TRY(cudaMemsetAsync(st.d_buf0, 0, st.nbytes, st.s0));
            CUDA_TRY(cudaMemsetAsync(st.d_buf1, 0, st.nbytes, st.s0));
            int n0_warm = 1;
            void* args[] = {&n0_warm, &st.n1, &st.d_buf0, &st.d_buf1};
            CUDA_TRY(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(stencil_coop_l2_kernel),
                                                 dim3(st.coop_blocks), dim3(256), args, 0, st.s0));
            CUDA_TRY(cudaStreamSynchronize(st.s0));
            return 0;
        }
    }

    st.mode = MODE_DIRECT_2J;
    st.direct_aligned4 = ((n1 & 3) == 0);
    st.direct_zc = (n1 <= 200) ? 8 : 32;

    dim3 blk(32, 4);
    dim3 grd = st.direct_aligned4
        ? dim3(((n1 + 3) / 4 + 31) / 32, ((m + 1) / 2 + 3) / 4, (m + st.direct_zc - 1) / st.direct_zc)
        : dim3(((m + 3) / 4 + 31) / 32, ((m + 1) / 2 + 3) / 4, (m + st.direct_zc - 1) / st.direct_zc);

    CUDA_TRY(cudaMemsetAsync(st.d_buf0, 0, st.nbytes, st.s0));
    CUDA_TRY(cudaMemsetAsync(st.d_buf1, 0, st.nbytes, st.s0));

    // Warm kernel
    if (st.direct_aligned4) stencil_1step_2j_aligned4_kernel<<<grd, blk, 0, st.s0>>>(n1, st.direct_zc, st.d_buf0, st.d_buf1);
    else                    stencil_1step_2j_4k_kernel<<<grd, blk, 0, st.s0>>>(n1, st.direct_zc, st.d_buf0, st.d_buf1);
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaStreamSynchronize(st.s0));

    // Capture full CUDA graph when n0 <= 64 to eliminate inter-step launch overhead
    if (n0 <= 64) {
        CUDA_TRY(cudaStreamBeginCapture(st.s0, cudaStreamCaptureModeThreadLocal));
        for (int t = 0; t < n0; ++t) {
            if (st.direct_aligned4) {
                stencil_1step_2j_aligned4_kernel<<<grd, blk, 0, st.s0>>>(n1, st.direct_zc, st.d_buf0, st.d_buf1);
                stencil_1step_2j_aligned4_kernel<<<grd, blk, 0, st.s0>>>(n1, st.direct_zc, st.d_buf1, st.d_buf0);
            } else {
                stencil_1step_2j_4k_kernel<<<grd, blk, 0, st.s0>>>(n1, st.direct_zc, st.d_buf0, st.d_buf1);
                stencil_1step_2j_4k_kernel<<<grd, blk, 0, st.s0>>>(n1, st.direct_zc, st.d_buf1, st.d_buf0);
            }
        }
        CUDA_TRY(cudaStreamEndCapture(st.s0, &st.direct_graph));
        CUDA_TRY(cudaGraphInstantiate(&st.direct_graph_exec, st.direct_graph, nullptr, nullptr, 0));
        CUDA_TRY(cudaGraphLaunch(st.direct_graph_exec, st.s0));
        CUDA_TRY(cudaStreamSynchronize(st.s0));
    }
    return 0;
}

static bool should_try_spectral(int n0, int n1) {
    if (n1 <= 2)   return false;
    if (n1 <= 8)   return n0 >= 250;
    if (n1 <= 32)  return n0 >= 90;
    if (n1 <= 64)  return n0 >= 80;
    if (n1 <= 128) return n0 >= 50;
    if (n1 <= 200) return n0 >= 42;
    if (n1 <= 256) return n0 >= 32;
    return n0 >= 28;
}

static bool validate_buffer_shape(const pb_buffer_shape* b, int64_t n1, uint64_t expected_nbytes) {
    if (!b || b->dtype != PB_F32 || b->rank != 3) return false;
    if (b->shape[0] != n1 || b->shape[1] != n1 || b->shape[2] != n1) return false;
    return b->nbytes == expected_nbytes;
}

static bool validate_host_buffers(const pb_buffer* buffers, int64_t n1, uint64_t expected_nbytes) {
    if (!buffers) return false;
    for (int i = 0; i < 2; ++i) {
        const pb_buffer& b = buffers[i];
        if (!b.data || b.dtype != PB_F32 || b.rank != 3) return false;
        if (b.shape[0] != n1 || b.shape[1] != n1 || b.shape[2] != n1) return false;
        if (b.nbytes != expected_nbytes) return false;
        uintptr_t addr = reinterpret_cast<uintptr_t>(b.data);
        if (addr % sizeof(float) != 0) return false;
        if (b.nbytes > UINTPTR_MAX - addr) return false;
    }
    uintptr_t s0 = reinterpret_cast<uintptr_t>(buffers[0].data);
    uintptr_t s1 = reinterpret_cast<uintptr_t>(buffers[1].data);
    if (s0 < s1 + buffers[1].nbytes && s1 < s0 + buffers[0].nbytes) return false;
    return true;
}

} // namespace

extern "C" {

int pb_setup(const pb_problem_shape *shape) {
    if (g_st.initialized) return 1;
    if (!shape || shape->abi_version != PB_SOLVE_ABI_VERSION ||
        shape->dimension_count != 2 || shape->scalar_count != 0 ||
        shape->buffer_count != 2 || !shape->dimensions || !shape->buffers) {
        return 1;
    }
    if (shape->dimensions[0] <= 0 || shape->dimensions[0] > INT_MAX) return 1;
    if (shape->dimensions[1] <= 0 || shape->dimensions[1] > INT_MAX) return 1;

    int n0 = static_cast<int>(shape->dimensions[0]);
    int n1 = static_cast<int>(shape->dimensions[1]);
    uint64_t u1 = static_cast<uint64_t>(n1);
    if (u1 > UINT64_MAX / u1) return 1;
    uint64_t u2 = u1 * u1;
    if (u2 > UINT64_MAX / u1) return 1;
    uint64_t count = u2 * u1;
    if (count > UINT64_MAX / sizeof(float)) return 1;
    uint64_t nbytes = count * sizeof(float);

    if (!validate_buffer_shape(&shape->buffers[0], n1, nbytes)) return 1;
    if (!validate_buffer_shape(&shape->buffers[1], n1, nbytes)) return 1;

    g_st.n0 = n0;
    g_st.n1 = n1;
    g_st.m = n1 - 2;
    g_st.L = n1 - 1;
    g_st.nelems = count;
    g_st.nbytes = nbytes;

    if (cudaStreamCreate(&g_st.s0) != cudaSuccess ||
        cudaMalloc(&g_st.d_buf0, nbytes) != cudaSuccess ||
        cudaMalloc(&g_st.d_buf1, nbytes) != cudaSuccess) {
        teardown_all(g_st);
        return 1;
    }

    if (n1 <= 2) {
        g_st.mode = MODE_NOOP;
        if (cudaStreamSynchronize(g_st.s0) != cudaSuccess) {
            teardown_all(g_st);
            return 1;
        }
        g_st.initialized = true;
        g_st.uploaded = false;
        return 0;
    }

    bool ok = false;
    if (should_try_spectral(n0, n1)) {
        int sp_rc = setup_spectral(g_st);
        if (sp_rc == 0) {
            g_st.mode = MODE_SPECTRAL;
            ok = true;
        } else {
            // Clean up any partial spectral allocations and fall back to Path 1
            destroy_spectral_resources(g_st);
            cudaGetLastError(); // clear any OOM sticky error before fallback
        }
    }

    if (!ok) {
        if (setup_direct(g_st) != 0) {
            teardown_all(g_st);
            return 1;
        }
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        teardown_all(g_st);
        return 1;
    }
    g_st.initialized = true;
    g_st.uploaded = false;
    return 0;
}

int pb_upload(const pb_problem *problem, pb_buffer *buffers) {
    if (!g_st.initialized || !problem || !buffers) return 1;
    if (problem->abi_version != PB_SOLVE_ABI_VERSION ||
        problem->dimension_count != 2 || problem->scalar_count != 0 ||
        problem->buffer_count != 2 || !problem->dimensions) {
        return 1;
    }
    if (problem->dimensions[0] != g_st.n0 || problem->dimensions[1] != g_st.n1) return 1;
    if (!validate_host_buffers(buffers, g_st.n1, g_st.nbytes)) return 1;

    CUDA_TRY(cudaMemcpyAsync(g_st.d_buf0, buffers[0].data, g_st.nbytes, cudaMemcpyHostToDevice, g_st.s0));
    CUDA_TRY(cudaMemcpyAsync(g_st.d_buf1, buffers[1].data, g_st.nbytes, cudaMemcpyHostToDevice, g_st.s0));
    CUDA_TRY(cudaStreamSynchronize(g_st.s0));
    g_st.uploaded = true;
    return 0;
}

int pb_compute(void) {
    if (!g_st.initialized || !g_st.uploaded) return 1;

    switch (g_st.mode) {
        case MODE_NOOP:
            return 0;

        case MODE_DIRECT_COOP: {
            void* args[] = {&g_st.n0, &g_st.n1, &g_st.d_buf0, &g_st.d_buf1};
            CUDA_TRY(cudaLaunchCooperativeKernel(
                reinterpret_cast<void*>(stencil_coop_l2_kernel),
                dim3(g_st.coop_blocks), dim3(256), args, 0, g_st.s0));
            CUDA_TRY(cudaStreamSynchronize(g_st.s0));
            return 0;
        }

        case MODE_DIRECT_2J: {
            if (g_st.direct_graph_exec) {
                CUDA_TRY(cudaGraphLaunch(g_st.direct_graph_exec, g_st.s0));
            } else {
                int n1 = g_st.n1, m = g_st.m, zc = g_st.direct_zc;
                dim3 blk(32, 4);
                dim3 grd = g_st.direct_aligned4
                    ? dim3(((n1 + 3) / 4 + 31) / 32, ((m + 1) / 2 + 3) / 4, (m + zc - 1) / zc)
                    : dim3(((m + 3) / 4 + 31) / 32, ((m + 1) / 2 + 3) / 4, (m + zc - 1) / zc);
                for (int t = 0; t < g_st.n0; ++t) {
                    if (g_st.direct_aligned4) {
                        stencil_1step_2j_aligned4_kernel<<<grd, blk, 0, g_st.s0>>>(n1, zc, g_st.d_buf0, g_st.d_buf1);
                        stencil_1step_2j_aligned4_kernel<<<grd, blk, 0, g_st.s0>>>(n1, zc, g_st.d_buf1, g_st.d_buf0);
                    } else {
                        stencil_1step_2j_4k_kernel<<<grd, blk, 0, g_st.s0>>>(n1, zc, g_st.d_buf0, g_st.d_buf1);
                        stencil_1step_2j_4k_kernel<<<grd, blk, 0, g_st.s0>>>(n1, zc, g_st.d_buf1, g_st.d_buf0);
                    }
                }
                CUDA_TRY(cudaGetLastError());
            }
            CUDA_TRY(cudaStreamSynchronize(g_st.s0));
            return 0;
        }

        case MODE_SPECTRAL: {
            CUDA_TRY(cudaGraphLaunch(g_st.spec_graph_exec, g_st.s0));
            CUDA_TRY(cudaStreamSynchronize(g_st.s0));
            return 0;
        }

        default:
            return 1;
    }
}

int pb_download(pb_buffer *buffers) {
    if (!g_st.initialized || !g_st.uploaded || !buffers) return 1;
    if (!validate_host_buffers(buffers, g_st.n1, g_st.nbytes)) return 1;

    CUDA_TRY(cudaMemcpyAsync(buffers[0].data, g_st.d_buf0, g_st.nbytes, cudaMemcpyDeviceToHost, g_st.s0));
    CUDA_TRY(cudaMemcpyAsync(buffers[1].data, g_st.d_buf1, g_st.nbytes, cudaMemcpyDeviceToHost, g_st.s0));
    CUDA_TRY(cudaStreamSynchronize(g_st.s0));
    return 0;
}

int pb_teardown(void) {
    if (!g_st.initialized) {
        return teardown_all(g_st);
    }
    return teardown_all(g_st);
}

} // extern "C"
