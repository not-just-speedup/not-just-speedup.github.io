// Shared emulation / baseline code (included after the candidate source).
#pragma once
#include <vector>
#include <cmath>
// ---------------- baseline port (compute.inc), templated on precision ----------------
template <typename T>
static void baseline(long long n0, long long n1, T s0, const float* x, T* out, T* b2, T* b3) {
  T k = (T(1) - std::exp(-s0)) * (T(1) - std::exp(-s0)) / (T(1) + T(2) * s0 * std::exp(-s0) - std::exp(T(2) * s0));
  T a1 = k, a5 = k;
  T a2 = k * std::exp(-s0) * (s0 - T(1)), a6 = a2;
  T a3 = k * std::exp(-s0) * (s0 + T(1)), a7 = a3;
  T a4 = -k * std::exp(T(-2) * s0), a8 = a4;
  T b1 = std::pow(T(2), -s0);
  T bb2 = -std::exp(T(-2) * s0);
  T c1 = 1, c2 = 1;
  std::vector<T> xin(n0 * n1);
  for (long long e = 0; e < n0 * n1; ++e) xin[e] = (T)x[e];
  for (long long i = 0; i < n0; ++i) {
    T t0 = 0, t1 = 0, t2 = 0;
    for (long long j = 0; j < n1; ++j) {
      b2[i * n1 + j] = a1 * xin[i * n1 + j] + a2 * t2 + b1 * t0 + bb2 * t1;
      t2 = xin[i * n1 + j]; t1 = t0; t0 = b2[i * n1 + j];
    }
  }
  for (long long i = 0; i < n0; ++i) {
    T t4 = 0, t5 = 0, t6 = 0, t7 = 0;
    for (long long j = n1 - 1; j >= 0; --j) {
      b3[i * n1 + j] = a3 * t6 + a4 * t7 + b1 * t4 + bb2 * t5;
      t7 = t6; t6 = xin[i * n1 + j]; t5 = t4; t4 = b3[i * n1 + j];
    }
  }
  for (long long e = 0; e < n0 * n1; ++e) out[e] = c1 * (b2[e] + b3[e]);
  for (long long j = 0; j < n1; ++j) {
    T t3 = 0, t0 = 0, t1 = 0;
    for (long long i = 0; i < n0; ++i) {
      b2[i * n1 + j] = a5 * out[i * n1 + j] + a6 * t3 + b1 * t0 + bb2 * t1;
      t3 = out[i * n1 + j]; t1 = t0; t0 = b2[i * n1 + j];
    }
  }
  for (long long j = 0; j < n1; ++j) {
    T t8 = 0, t9 = 0, t4 = 0, t5 = 0;
    for (long long i = n0 - 1; i >= 0; --i) {
      b3[i * n1 + j] = a7 * t8 + a8 * t9 + b1 * t4 + bb2 * t5;
      t9 = t8; t8 = out[i * n1 + j]; t5 = t4; t4 = b3[i * n1 + j];
    }
  }
  for (long long e = 0; e < n0 * n1; ++e) out[e] = c2 * (b2[e] + b3[e]);
}

// ---------------- emulation of row_kernel ----------------
static void emu_row(const float* x, float* h, long long n0, long long n1, const Coef& c, int W) {
  const int NQ = 32 + 2 * W;
  std::vector<float> tile(NQ * TS);
  float2 A[32], o[32], E[32], Eb[32];
  float z[32][32], xp[32];
  auto welem = [&](int base_piece, int e) -> float { return tile[(base_piece + (e >> 5)) * TS + (e & 31)]; };
  const long long G = (n1 + 1023) / 1024;
  for (long long i = 0; i < n0; ++i) {
    for (long long g = 0; g < G; ++g) {
      const long long j0 = g * 1024;
      for (int q = 0; q < NQ; ++q)
        for (int lane = 0; lane < 32; ++lane) {
          long long j = j0 - 32 * W + 32 * q + lane;
          tile[q * TS + lane] = (j >= 0 && j < n1) ? x[i * n1 + j] : 0.f;
        }
      // ---- anti-causal window: lane t owns [Wt, Wt+W) of the right window (base piece 32+W)
      for (int lane = 0; lane < 32; ++lane) {
        const int e0 = W * lane, e1 = e0 + W, e2 = e0 + W + 1;
        float xn1 = (e1 < 32 * W) ? welem(32 + W, e1) : 0.f;
        float xn2 = (e2 < 32 * W) ? welem(32 + W, e2) : 0.f;
        float t4 = 0.f, t5 = 0.f;
        for (int m = W - 1; m >= 0; --m) {
          float xk = welem(32 + W, e0 + m);
          float v = step_bwd(xn1, xn2, t4, t5, c);
          t5 = t4; t4 = v; xn2 = xn1; xn1 = xk;
        }
        A[lane] = make_float2(t4, t5);
      }
      for (int s = 0; s < 5; ++s) {
        int d = 1 << s;
        for (int lane = 0; lane < 32; ++lane) o[lane] = (lane + d < 32) ? A[lane + d] : A[lane];
        for (int lane = 0; lane < 32; ++lane) if (lane + d < 32) A[lane] = mat_acc(c.Mw[s], o[lane], A[lane]);
      }
      const float2 Sr = A[0];
      // main anti-causal local pass
      for (int lane = 0; lane < 32; ++lane) {
        const float* xc = &tile[(W + lane) * TS];
        float t4 = 0.f, t5 = 0.f, n1v = xc[TS], n2v = xc[TS + 1];
        for (int k = 31; k >= 0; --k) {
          float xk = xc[k];
          float v = step_bwd(n1v, n2v, t4, t5, c);
          z[lane][k] = v;
          t5 = t4; t4 = v; n2v = n1v; n1v = xk;
        }
        A[lane] = make_float2(t4, t5);
      }
      A[31] = mat_acc(c.Mp[0], Sr, A[31]);
      for (int s = 0; s < 5; ++s) {
        int d = 1 << s;
        for (int lane = 0; lane < 32; ++lane) o[lane] = (lane + d < 32) ? A[lane + d] : A[lane];
        for (int lane = 0; lane < 32; ++lane) if (lane + d < 32) A[lane] = mat_acc(c.Mp[s], o[lane], A[lane]);
      }
      for (int lane = 0; lane < 32; ++lane) Eb[lane] = (lane == 31) ? Sr : A[lane + 1];
      // ---- causal window: lane t owns [Wt, Wt+W) of the left window (base piece 0)
      for (int lane = 0; lane < 32; ++lane) {
        const int e0 = W * lane;
        float xp0 = (e0 > 0) ? welem(0, e0 - 1) : 0.f;
        float t0 = 0.f, t1 = 0.f, t2 = xp0;
        for (int m = 0; m < W; ++m) {
          float xk = welem(0, e0 + m);
          float v = step_fwd(xk, t2, t0, t1, c);
          t2 = xk; t1 = t0; t0 = v;
        }
        A[lane] = make_float2(t0, t1);
      }
      for (int s = 0; s < 5; ++s) {
        int d = 1 << s;
        for (int lane = 0; lane < 32; ++lane) o[lane] = (lane >= d) ? A[lane - d] : A[lane];
        for (int lane = 0; lane < 32; ++lane) if (lane >= d) A[lane] = mat_acc(c.Mw[s], o[lane], A[lane]);
      }
      const float2 Sl = A[31];
      // main causal local pass, accumulated onto z
      for (int lane = 0; lane < 32; ++lane) {
        const float* xc = &tile[(W + lane) * TS];
        xp[lane] = xc[-TS + 31];
        float t0 = 0.f, t1 = 0.f, t2 = xp[lane];
        for (int k = 0; k < 32; ++k) {
          float xk = xc[k];
          float v = step_fwd(xk, t2, t0, t1, c);
          t2 = xk; t1 = t0; t0 = v;
          z[lane][k] = z[lane][k] + v;
        }
        A[lane] = make_float2(t0, t1);
      }
      A[0] = mat_acc(c.Mp[0], Sl, A[0]);
      for (int s = 0; s < 5; ++s) {
        int d = 1 << s;
        for (int lane = 0; lane < 32; ++lane) o[lane] = (lane >= d) ? A[lane - d] : A[lane];
        for (int lane = 0; lane < 32; ++lane) if (lane >= d) A[lane] = mat_acc(c.Mp[s], o[lane], A[lane]);
      }
      for (int lane = 0; lane < 32; ++lane) E[lane] = (lane == 0) ? Sl : A[lane - 1];
      for (int lane = 0; lane < 32; ++lane)
        for (int k = 0; k < 32; ++k) {
          const float cb = fmaf(c.h1[31 - k], Eb[lane].x, c.h2[31 - k] * Eb[lane].y);
          const float cf = fmaf(c.h1[k], E[lane].x, c.h2[k] * E[lane].y);
          tile[(W + lane) * TS + k] = z[lane][k] + (cb + cf);
        }
      for (int qq = 0; qq < 32; ++qq)
        for (int lane = 0; lane < 32; ++lane) {
          long long j = j0 + 32 * qq + lane;
          if (j < n1) h[i * n1 + j] = tile[(W + qq) * TS + lane];
        }
    }
  }
}

// ---------------- emulation of col_kernel ----------------
static void emu_col(const float* in, float* out, long long n0, long long n1, const Coef& c, int W) {
  const long long stripes = (n1 + 31) / 32, chunks = (n0 + 511) / 512;
  std::vector<float> xv(16 * 32 * 32), zbuf(16 * 32 * 32);
  float2 pbuf[16][32], ebuf[16][32], swin[2][32];
  float xpv[16][32], xn1v[16][32], xn2v[16][32];
  for (long long s = 0; s < stripes; ++s)
    for (long long chunk = 0; chunk < chunks; ++chunk) {
      const long long i0 = chunk * 512;
      auto ld = [&](long long r, int lane) -> float {
        long long col = s * 32 + lane;
        return (col < n1 && r >= 0 && r < n0) ? in[r * n1 + col] : 0.f;
      };
      for (int warp = 0; warp < W; ++warp)
        for (int lane = 0; lane < 32; ++lane) {
          long long r0 = i0 - 32LL * (W - warp);
          float xw[32];
          for (int k = 0; k < 32; ++k) xw[k] = ld(r0 + k, lane);
          float xp = (warp > 0) ? ld(r0 - 1, lane) : 0.f;
          pbuf[warp][lane] = local_fwd(xw, xp, 0.f, 0.f, c);
        }
      for (int wi = 0; wi < W; ++wi)
        for (int lane = 0; lane < 32; ++lane) {
          long long r0 = i0 + 512 + 32LL * wi;
          float xw[32];
          for (int k = 0; k < 32; ++k) xw[k] = ld(r0 + k, lane);
          bool nxt = wi + 1 < W;
          pbuf[8 + wi][lane] = local_bwd(xw, nxt ? ld(r0 + 32, lane) : 0.f, nxt ? ld(r0 + 33, lane) : 0.f, 0.f, 0.f, c);
        }
      for (int lane = 0; lane < 32; ++lane) {
        float2 S = make_float2(0.f, 0.f);
        for (int q = 0; q < W; ++q) S = mat_acc(c.Mp[0], S, pbuf[q][lane]);
        swin[0][lane] = S;
        S = make_float2(0.f, 0.f);
        for (int q = W - 1; q >= 0; --q) S = mat_acc(c.Mp[0], S, pbuf[8 + q][lane]);
        swin[1][lane] = S;
      }
      for (int warp = 0; warp < 16; ++warp)
        for (int lane = 0; lane < 32; ++lane) {
          long long r0 = i0 + 32LL * warp;
          float* xw = &xv[(warp * 32 + lane) * 32];
          for (int k = 0; k < 32; ++k) xw[k] = ld(r0 + k, lane);
          xpv[warp][lane] = ld(r0 - 1, lane);
          xn1v[warp][lane] = ld(r0 + 32, lane);
          xn2v[warp][lane] = ld(r0 + 33, lane);
          pbuf[warp][lane] = local_bwd(xw, xn1v[warp][lane], xn2v[warp][lane], 0.f, 0.f, c);
        }
      for (int lane = 0; lane < 32; ++lane) {
        float2 S = swin[1][lane];
        for (int p = 15; p >= 0; --p) { ebuf[p][lane] = S; S = mat_acc(c.Mp[0], S, pbuf[p][lane]); }
      }
      for (int warp = 0; warp < 16; ++warp)
        for (int lane = 0; lane < 32; ++lane) {
          const float* xw = &xv[(warp * 32 + lane) * 32];
          float t4 = ebuf[warp][lane].x, t5 = ebuf[warp][lane].y, n1v = xn1v[warp][lane], n2v = xn2v[warp][lane];
          for (int k = 31; k >= 0; --k) {
            float v = step_bwd(n1v, n2v, t4, t5, c);
            zbuf[(warp * 32 + k) * 32 + lane] = v;
            t5 = t4; t4 = v; n2v = n1v; n1v = xw[k];
          }
          pbuf[warp][lane] = local_fwd(xw, xpv[warp][lane], 0.f, 0.f, c);
        }
      for (int lane = 0; lane < 32; ++lane) {
        float2 S = swin[0][lane];
        for (int p = 0; p < 16; ++p) { ebuf[p][lane] = S; S = mat_acc(c.Mp[0], S, pbuf[p][lane]); }
      }
      for (int warp = 0; warp < 16; ++warp)
        for (int lane = 0; lane < 32; ++lane) {
          long long r0 = i0 + 32LL * warp, col = s * 32 + lane;
          const float* xw = &xv[(warp * 32 + lane) * 32];
          float t0 = ebuf[warp][lane].x, t1 = ebuf[warp][lane].y, t2 = xpv[warp][lane];
          for (int k = 0; k < 32; ++k) {
            float v = step_fwd(xw[k], t2, t0, t1, c);
            t2 = xw[k]; t1 = t0; t0 = v;
            float hv = v + zbuf[(warp * 32 + k) * 32 + lane];
            long long r = r0 + k;
            if (col < n1 && r < n0) out[r * n1 + col] = hv;
          }
        }
    }
}

