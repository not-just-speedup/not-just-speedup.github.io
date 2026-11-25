// SIMT host simulation of the actual kernel source versus the validated emulation.
// Usage: sim_test n0 n1 s0 [seed]
#include "sim_shim.h"
#define PB_HOST_SIM 1
#include "../candidate/solution.cu"
#include "emu.h"
#include <random>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <cmath>

template <int W>
static void sim_pair(const float* x, float* h, float* out, long long n0, long long n1, const Coef& c) {
  const long long G = (n1 + ROW_CHUNK - 1) / ROW_CHUNK;
  const long long items = n0 * G;
  const long long rblocks = (items + ROW_WARPS - 1) / ROW_WARPS;
  const size_t rsmem = (size_t)ROW_WARPS * (PIECE + 2 * W) * TS * sizeof(float);
  sim_launch((unsigned)rblocks, ROW_WARPS * 32, rsmem, [&] { row_kernel<W>(x, h, n1, G, items, c); });
  const long long stripes = (n1 + 31) / 32;
  const long long chunks = (n0 + COL_CHUNK - 1) / COL_CHUNK;
  sim_launch((unsigned)(stripes * chunks), COL_WARPS * 32, COL_SMEM_BYTES, [&] { col_kernel<W>(h, out, n0, n1, stripes, c); });
}
static void sim_all(int W, const float* x, float* h, float* out, long long n0, long long n1, const Coef& c) {
  switch (W) {
    case 1: sim_pair<1>(x, h, out, n0, n1, c); break;
    case 2: sim_pair<2>(x, h, out, n0, n1, c); break;
    case 3: sim_pair<3>(x, h, out, n0, n1, c); break;
    case 4: sim_pair<4>(x, h, out, n0, n1, c); break;
    default: sim_pair<5>(x, h, out, n0, n1, c); break;
  }
}

int main(int argc, char** argv) {
  if (argc < 4) { printf("usage: sim_test n0 n1 s0 [seed]\n"); return 2; }
  const long long n0 = atoll(argv[1]), n1 = atoll(argv[2]);
  const float s0 = (float)atof(argv[3]);
  const unsigned seed = argc > 4 ? (unsigned)atoi(argv[4]) : 1u;
  const long long N = n0 * n1;
  std::vector<float> x(N);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  for (long long e = 0; e < N; ++e) x[e] = U(rng);
  Coef c; int W;
  make_coef(s0, c, W);

  // emulation
  std::vector<float> h_emu(N), out_emu(N);
  emu_row(x.data(), h_emu.data(), n0, n1, c, W);
  emu_col(h_emu.data(), out_emu.data(), n0, n1, c, W);

  // simulated kernels on guarded arrays: NaN pads around x (stray reads -> NaN results) and around
  // the outputs (stray writes -> corrupted guard); outputs pre-filled with 12345 to catch unwritten elements
  const long long GP = 4096;
  const float QNAN = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> xg(N + 2 * GP, QNAN), hg(N + 2 * GP, QNAN), og(N + 2 * GP, QNAN);
  for (long long e = 0; e < N; ++e) { xg[GP + e] = x[e]; hg[GP + e] = 12345.f; og[GP + e] = 12345.f; }
  sim_all(W, xg.data() + GP, hg.data() + GP, og.data() + GP, n0, n1, c);
  long long mh = 0, mo = 0, gbad = 0;
  for (long long e = 0; e < N; ++e) { if (hg[GP + e] != h_emu[e]) ++mh; if (og[GP + e] != out_emu[e]) ++mo; }
  for (long long g = 0; g < GP; ++g) {
    if (!std::isnan(xg[g]) || !std::isnan(xg[N + 2 * GP - 1 - g])) ++gbad;
    if (!std::isnan(hg[g]) || !std::isnan(hg[N + 2 * GP - 1 - g])) ++gbad;
    if (!std::isnan(og[g]) || !std::isnan(og[N + 2 * GP - 1 - g])) ++gbad;
  }
  if (gbad) printf("  GLOBAL GUARD CORRUPTED: %lld guard words\n", gbad);
  std::vector<float> h_sim(hg.begin() + GP, hg.begin() + GP + N), out_sim(og.begin() + GP, og.begin() + GP + N);
  printf("n0=%lld n1=%lld s0=%g W=%d: simulated fast kernels vs emulation: H mismatches %lld, out mismatches %lld\n",
         n0, n1, s0, W, mh, mo);
  if (mo) {
    int shown = 0;
    for (long long e = 0; e < N && shown < 5; ++e)
      if (out_sim[e] != out_emu[e]) { printf("  e=%lld (i=%lld j=%lld) sim=%.9g emu=%.9g\n", e, e / n1, e % n1, out_sim[e], out_emu[e]); ++shown; }
  }

  // simulated fallback kernels vs their emulation (baseline loops with step_fwd/step_bwd)
  std::vector<float> hh(N), nv(N);
  for (long long i = 0; i < n0; ++i) {
    float t0 = 0, t1 = 0, t2 = 0;
    for (long long j = 0; j < n1; ++j) { float xj = x[i*n1+j]; float y = step_fwd(xj, t2, t0, t1, c); hh[i*n1+j] = y; t2 = xj; t1 = t0; t0 = y; }
    float t4 = 0, t5 = 0, t6 = 0, t7 = 0;
    for (long long j = n1 - 1; j >= 0; --j) { float z = step_bwd(t6, t7, t4, t5, c); hh[i*n1+j] += z; t7 = t6; t6 = x[i*n1+j]; t5 = t4; t4 = z; }
  }
  for (long long j = 0; j < n1; ++j) {
    float t0 = 0, t1 = 0, t3 = 0;
    for (long long i = 0; i < n0; ++i) { float xi = hh[i*n1+j]; float y = step_fwd(xi, t3, t0, t1, c); nv[i*n1+j] = y; t3 = xi; t1 = t0; t0 = y; }
    float t4 = 0, t5 = 0, t8 = 0, t9 = 0;
    for (long long i = n0 - 1; i >= 0; --i) { float z = step_bwd(t8, t9, t4, t5, c); nv[i*n1+j] += z; t9 = t8; t8 = hh[i*n1+j]; t5 = t4; t4 = z; }
  }
  std::vector<float> h2(N, 12345.f), o2(N, 12345.f);
  sim_launch((unsigned)((n0 + 255) / 256), 256, 0, [&] { naive_rows(x.data(), h2.data(), n0, n1, c); });
  sim_launch((unsigned)((n1 + 255) / 256), 256, 0, [&] { naive_cols(h2.data(), o2.data(), n0, n1, c); });
  long long mn = 0;
  for (long long e = 0; e < N; ++e) if (o2[e] != nv[e]) ++mn;
  printf("  simulated fallback kernels vs their emulation: out mismatches %lld\n", mn);
  return (mo || mn || gbad) ? 1 : 0;
}
