// Floyd-Warshall min-plus closure on an n x n int32 matrix (entries in [0,256], exact integer result) -- GPU translation.
//
// Solvers (all exact for every n and every input; everything runs on packed int16 pairs with the native VIADDMNMX.S16x2):
//   n <= 64            k_small_fw   one block, register-resident closure
//   64 < n <= 512      k_sq         persistent cooperative min-plus squaring closure (P <- min(P, P (x) P) until nothing changes)
//   n <= 1024          k_fw_engine  persistent dependency-driven blocked FW (fallback of k_sq, dense fallback up to 1024)
//   otherwise          v1           multi-kernel blocked FW (last-resort fallback)
// Fast path (n >= NFAST): the zero-weight edges of a random matrix contain a giant strongly connected component; all its vertices
// are at distance 0 from each other and can be contracted into one vertex exactly. F1 zero bitmaps + degrees -> F2 forward/backward
// BFS (2-block cluster) -> F3 reduced matrix -> F4 closure of the reduced matrix (k_small_fw for m <= 64, k_sq for m <= MMAX)
// -> F5 expansion to the full matrix. When the contraction does not pay off (no big component, m > MMAX) the kernels decline and a
// dense solver runs on the untouched input.
// Every persistent kernel has a watchdog / timeout; the host falls back to a safer path if one fires.
#include "interface.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#define INF16 1000
#define TB 64

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) return (int)(__LINE__); } while (0)
// ---------------------------------------------------------------- single-block closure
// Single-block register-resident Floyd-Warshall closure for m <= 64, 16-bit packed pairs.
// Input/output are int32 matrices with given pitches. m is read from device control memory (ctl[0]=ok, ctl[1]=m)
// so the kernel can be launched before m is known to the host.


constexpr int SW_THREADS = 512;
constexpr int SW_MAXS = 64;
constexpr int SW_PS = SW_MAXS + 2;                                 // staging pitch (elements)
constexpr size_t SW_SMEM = (size_t)SW_MAXS * SW_PS * 2 + (4 * (SW_MAXS / 2) + 2 * SW_MAXS) * 4;  // dynamic smem bytes

// S-class body: 8 word-groups x 64 row-groups; each thread owns RPT rows x WPT words in registers.
// Pivots are processed in pairs (k, k+1), k even, with one barrier per pair. Applying steps k then k+1 to row i gives
//   d2[i][j] = min(d[i][j], a' + R_k[j], b1 + R_{k+1}[j]),   b1 = min(d[i][k+1], d[i][k] + R_k[k+1]),  a' = min(d[i][k], b1 + R_{k+1}[k])
// with all quantities taken before the pair (R_k = pivot row k); the two column entries d[i][k], d[i][k+1] are the two halves
// of one packed word, so the pair costs two VIADDMNMX per word, exactly like two separate steps.
template <int S>
__device__ __forceinline__ void small_fw_body(const int* __restrict__ in, int ld_in, int* __restrict__ out, int ld_out, int m,
                                              u16* stage, u32* smem) {
  constexpr int W = S / 2, WPT = S / 16, RPT = S / 64;
  const int t = threadIdx.x;
  const int wg = t & 7, rg = t >> 3;
  u32* stage32 = (u32*)stage;
  static_assert(S == 64, "single-block closure handles m <= 64 only");
  // ---- coalesced load into staging (elements as u16): all 8 loads of a thread are issued before the first store
  {
    const int c0 = t & 63, r0 = t >> 6;
    int x[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int r = r0 + 8 * i;
      x[i] = (r < m && c0 < m) ? __ldg(&in[(size_t)r * ld_in + c0]) : 0;
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int r = r0 + 8 * i;
      if (r < m && c0 < m) stage[r * SW_PS + c0] = (u16)x[i];
    }
  }
  __syncthreads();
  u32 acc[RPT][WPT];
#pragma unroll
  for (int i = 0; i < RPT; i++) {
    const int row = rg + 64 * i;
#pragma unroll
    for (int w = 0; w < WPT; w++) {
      const int wd = wg * WPT + w;
      const int col = 2 * wd;
      u32 v = (INF16 | (INF16 << 16));
      if (row < m && col < m) {
        const u32 x = stage32[row * (SW_PS / 2) + wd];
        const u32 lo = x & 0xFFFFu;
        const u32 hi = (col + 1 < m) ? (x >> 16) : (u32)INF16;
        v = lo | (hi << 16);
      }
      acc[i][w] = v;
    }
  }
  u32* rowbuf = smem;          // [2 buffers][2 rows][W]
  u32* colbuf = smem + 4 * W;  // [2 buffers][S]
  for (int kk = 0; kk < 8; kk++) {
    if (kk * 2 * WPT >= m) break;  // uniform
#pragma unroll
    for (int pp = 0; pp < WPT; pp++) {
      const int p = kk * WPT + pp;          // pair index: pivots 2p, 2p+1 (word p of every row)
      if (2 * p < m) {                       // uniform
        const int buf = pp & 1;
        // publish: word p of every row (owner: word-group kk) and pivot rows 2p, 2p+1 (owners: row-groups (2p)&63 and +1)
        if (wg == kk) {
#pragma unroll
          for (int i = 0; i < RPT; i++) colbuf[buf * S + rg + 64 * i] = acc[i][pp];
        }
        {
          const int rg0 = (2 * p) & 63, i0 = (2 * p) >> 6;
          if (rg == rg0 || rg == rg0 + 1) {
            const int which = rg - rg0;
#pragma unroll
            for (int i = 0; i < RPT; i++) {
              if (i == i0) {
#pragma unroll
                for (int w = 0; w < WPT; w++) rowbuf[(buf * 2 + which) * W + wg * WPT + w] = acc[i][w];
              }
            }
          }
        }
        __syncthreads();
        const u32 r0w = rowbuf[(buf * 2 + 0) * W + p];   // row 2p, word p:   {R_k[k], R_k[k+1]}
        const u32 r1w = rowbuf[(buf * 2 + 1) * W + p];   // row 2p+1, word p: {R_{k+1}[k], R_{k+1}[k+1]}
        const u32 rk_k1 = r0w >> 16;                      // R_k[k+1]
        const u32 c_k1k = r1w & 0xFFFFu;                  // R_{k+1}[k]
        u32 A[RPT], B[RPT];
#pragma unroll
        for (int i = 0; i < RPT; i++) {
          const u32 cw = colbuf[buf * S + rg + 64 * i];
          const u32 a = cw & 0xFFFFu, b0 = cw >> 16;
          const u32 b1 = min(b0, a + rk_k1);
          const u32 a2 = min(a, b1 + c_k1k);
          A[i] = a2 * 0x10001u;
          B[i] = b1 * 0x10001u;
        }
#pragma unroll
        for (int w = 0; w < WPT; w++) {
          const u32 x0 = rowbuf[(buf * 2 + 0) * W + wg * WPT + w];
          const u32 x1 = rowbuf[(buf * 2 + 1) * W + wg * WPT + w];
#pragma unroll
          for (int i = 0; i < RPT; i++) {
            acc[i][w] = __viaddmin_s16x2(A[i], x0, acc[i][w]);
            acc[i][w] = __viaddmin_s16x2(B[i], x1, acc[i][w]);
          }
        }
      }
    }
  }
  __syncthreads();  // everyone done reading stage/rowbuf before reuse
#pragma unroll
  for (int i = 0; i < RPT; i++) {
    const int row = rg + 64 * i;
#pragma unroll
    for (int w = 0; w < WPT; w++) stage32[row * (SW_PS / 2) + wg * WPT + w] = acc[i][w];
  }
  __syncthreads();
  {
    const int c0 = t & 63, r0 = t >> 6;
    for (int r = r0; r < m; r += 8) {
      int* dst = out + (size_t)r * ld_out;
      const u16* src = stage + r * SW_PS;
#pragma unroll 4
      for (int c = c0; c < m; c += 64) dst[c] = (int)src[c];
    }
  }
}

// ctl[0] = run flag (nonzero => run), ctl[1] = m.  Dynamic shared memory: SW_SMEM bytes.
__global__ void __launch_bounds__(SW_THREADS) k_small_fw(const int* __restrict__ in, int ld_in, int* __restrict__ out, int ld_out,
                                                         const int* __restrict__ ctl, int mmax_small) {
  extern __shared__ __align__(16) unsigned char sw_raw[];
  if (ctl[0] == 0 || ctl[1] > min(mmax_small, 64)) return;
  u16* stage = (u16*)sw_raw;
  u32* smem = (u32*)(sw_raw + (size_t)SW_MAXS * SW_PS * 2);
  const int m = ctl[1];
  small_fw_body<64>(in, ld_in, out, ld_out, m, stage, smem);
}
// ======================================================================================================
// Dense engine: blocked Floyd-Warshall on a padded 16-bit copy (fallback and small/medium n)
// ======================================================================================================
__device__ __forceinline__ u32 rep16(u32 x) { return (x & 0xFFFFu) * 0x10001u; }

__global__ void conv_in(const int* __restrict__ in, u16* __restrict__ W, int n, int np) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t wpr = np / 2;
  size_t row = idx / wpr;
  if (row >= (size_t)np) return;
  int c = (int)(idx % wpr) * 2;
  int v0 = INF16, v1 = INF16;
  if (row < (size_t)n) {
    if (c < n) v0 = in[row * n + c];
    if (c + 1 < n) v1 = in[row * n + c + 1];
  }
  ((u32*)W)[idx] = ((u32)v0 & 0xFFFFu) | ((u32)v1 << 16);
}

__global__ void conv_out(const u16* __restrict__ W, int* __restrict__ out, int n, int np) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)n * n;
  if (idx >= total) return;
  size_t row = idx / n; int col = (int)(idx % n);
  out[idx] = (int)W[row * np + col];
}

__global__ void __launch_bounds__(256) k1_diag(u16* W, int ld, int kb) {
  __shared__ u32 D[TB * (TB / 2)];
  int t = threadIdx.x;
  int row = t >> 2, cq = t & 3;
  u16* tile = W + (size_t)kb * TB * ld + (size_t)kb * TB;
  {
    const uint4* src = (const uint4*)(tile + (size_t)row * ld + cq * 16);
    uint4 a = src[0], b = src[1];
    u32* dst = &D[row * 32 + cq * 8];
    dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w;
    dst[4] = b.x; dst[5] = b.y; dst[6] = b.z; dst[7] = b.w;
  }
  __syncthreads();
  for (int k = 0; k < TB; k++) {
    u32 aw = D[row * 32 + (k >> 1)];
    u32 a = rep16((k & 1) ? (aw >> 16) : aw);
    u32 rk[8], cur[8];
#pragma unroll
    for (int w = 0; w < 8; w++) { rk[w] = D[k * 32 + cq * 8 + w]; cur[w] = D[row * 32 + cq * 8 + w]; }
#pragma unroll
    for (int w = 0; w < 8; w++) cur[w] = __viaddmin_s16x2(a, rk[w], cur[w]);
    __syncthreads();
#pragma unroll
    for (int w = 0; w < 8; w++) D[row * 32 + cq * 8 + w] = cur[w];
    __syncthreads();
  }
  {
    uint4* dst = (uint4*)(tile + (size_t)row * ld + cq * 16);
    const u32* s = &D[row * 32 + cq * 8];
    dst[0] = make_uint4(s[0], s[1], s[2], s[3]);
    dst[1] = make_uint4(s[4], s[5], s[6], s[7]);
  }
}

template <int MODE>
__global__ void __launch_bounds__(64) k_tile(u16* W, int ld, int kb) {
  __shared__ __align__(16) u32 As[TB * TB];
  __shared__ __align__(16) u32 Bs[TB * (TB / 2)];
  int t = threadIdx.x;
  int ti, tj;
  if (MODE == 0) { ti = blockIdx.y; ti += (ti >= kb); tj = blockIdx.x; tj += (tj >= kb); }
  else if (MODE == 1) { ti = kb; tj = blockIdx.x; tj += (tj >= kb); }
  else { tj = kb; ti = blockIdx.x; ti += (ti >= kb); }
  int ai = (MODE == 1) ? kb : ti;
  int bj = (MODE == 2) ? kb : tj;
  const u16* Atile = W + (size_t)ai * TB * ld + (size_t)kb * TB;
  const u16* Btile = W + (size_t)kb * TB * ld + (size_t)bj * TB;
  u16* Ctile = W + (size_t)ti * TB * ld + (size_t)tj * TB;
  {
    const uint4* src = (const uint4*)(Atile + (size_t)t * ld);
#pragma unroll
    for (int q = 0; q < 8; q++) {
      uint4 v = src[q];
      u32 w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int e = 0; e < 4; e++) {
        As[(q * 8 + 2 * e) * TB + t] = rep16(w4[e]);
        As[(q * 8 + 2 * e + 1) * TB + t] = rep16(w4[e] >> 16);
      }
    }
  }
  {
#pragma unroll
    for (int it = 0; it < 8; it++) {
      int r = it * 8 + (t >> 3), ch = t & 7;
      uint4 v = *(const uint4*)(Btile + (size_t)r * ld + ch * 8);
      *(uint4*)&Bs[r * 32 + ch * 4] = v;
    }
  }
  int rg = t >> 3, cg = t & 7;
  u32 acc[8][4];
#pragma unroll
  for (int r = 0; r < 8; r++) {
    uint4 v = *(const uint4*)(Ctile + (size_t)(rg * 8 + r) * ld + cg * 8);
    acc[r][0] = v.x; acc[r][1] = v.y; acc[r][2] = v.z; acc[r][3] = v.w;
  }
  __syncthreads();
#pragma unroll 4
  for (int k = 0; k < TB; k++) {
    uint4 a0 = *(const uint4*)&As[k * TB + rg * 8];
    uint4 a1 = *(const uint4*)&As[k * TB + rg * 8 + 4];
    uint4 b = *(const uint4*)&Bs[k * 32 + cg * 4];
    u32 a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    u32 bb[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int r = 0; r < 8; r++)
#pragma unroll
      for (int c = 0; c < 4; c++) acc[r][c] = __viaddmin_s16x2(a[r], bb[c], acc[r][c]);
  }
#pragma unroll
  for (int r = 0; r < 8; r++)
    *(uint4*)(Ctile + (size_t)(rg * 8 + r) * ld + cg * 8) = make_uint4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
}
// ======================================================================================================
// Persistent dependency-driven blocked Floyd-Warshall engine (tile edge EB = 32).
//   * int32 source -> tile-major int16 work matrix (round 0), int16 -> int32 destination (last round): no separate conversions
//   * tasks (round k, tile) are dequeued in wavefront order from a global counter; every task waits on per-tile version /
//     reader counters, so the critical chain per round is ~3 flag hops and everything else overlaps.
// ======================================================================================================
constexpr int EB = 32;
constexpr int ETILE = EB * EB;
constexpr int ETHREADS = 128;
constexpr int EMAX_N = 1024;   // largest n the engine handles (N <= 32 tiles per side)

struct EngState {   // all zero between launches (the watchdog block clears it at the end of a run)
  int head;          // task queue head
  int pad0[31];      // (the flag arrays start 64 ints after the queue head)
  int pad1[32];
};
constexpr unsigned long long ENG_WATCHDOG_NS = 2000000000ull;   // no queue progress for 2 s => hung (a healthy run advances every few microseconds)

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void decode_task(int q, int N, int& k, int& type, int& ti, int& tj) {
  const int NN = N * N;
  k = q / NN;
  const int p = q - k * NN;
  if (p == 0) { type = 0; ti = tj = k; return; }
  if (p <= 2 * (N - 1)) {
    const int s = (p - 1) >> 1;
    int idx = k + 1 + s; if (idx >= N) idx -= N;
    if ((p - 1) & 1) { type = 2; ti = idx; tj = k; }   // column panel (i, k)
    else { type = 1; ti = k; tj = idx; }               // row panel (k, j)
    return;
  }
  const int t3 = p - (1 + 2 * (N - 1));
  int m = (int)sqrtf((float)t3);
  while (m * m > t3) m--;
  while ((m + 1) * (m + 1) <= t3) m++;
  const int o = t3 - m * m;
  int ri, rj;
  if (o <= m) { ri = m; rj = o; } else { ri = o - m - 1; rj = m; }
  type = 3;
  ti = k + 1 + ri; if (ti >= N) ti -= N;
  tj = k + 1 + rj; if (tj >= N) tj -= N;
}

// Diagonal tile closure by one warp: lane = row, 16 packed words per lane in registers, pivot row via shuffles.
__device__ __forceinline__ void tile_closure(u32 (&acc)[16]) {
#pragma unroll
  for (int k = 0; k < EB; k++) {
    u32 rk[16];
#pragma unroll
    for (int w = 0; w < 16; w++) rk[w] = __shfl_sync(0xFFFFFFFFu, acc[w], k);
    const u32 a = rep16((k & 1) ? (acc[k >> 1] >> 16) : acc[k >> 1]);
#pragma unroll
    for (int w = 0; w < 16; w++) acc[w] = __viaddmin_s16x2(a, rk[w], acc[w]);
  }
}

// Read one 32x32 tile of the int32 source (rows/cols beyond n -> INF16): thread (r = t>>2, cq = t&3) gets 4 packed words.
__device__ __forceinline__ uint4 src_chunk(const int* __restrict__ src, int ld, int n, int ti, int tj, int t) {
  const int r = t >> 2, cq = t & 3;
  const int row = ti * EB + r;
  u32 w[4];
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const int col = tj * EB + cq * 8 + 2 * e;
    u32 v0 = INF16, v1 = INF16;
    if (row < n) {
      if (col < n) v0 = (u32)__ldg(&src[(size_t)row * ld + col]);
      if (col + 1 < n) v1 = (u32)__ldg(&src[(size_t)row * ld + col + 1]);
    }
    w[e] = (v0 & 0xFFFFu) | (v1 << 16);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ void dst_chunk(int* __restrict__ dst, int ld, int n, int ti, int tj, int t, uint4 c) {
  const int r = t >> 2, cq = t & 3;
  const int row = ti * EB + r;
  if (row >= n) return;
  const u32 w[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const int col = tj * EB + cq * 8 + 2 * e;
    if (col < n) dst[(size_t)row * ld + col] = (int)(w[e] & 0xFFFFu);
    if (col + 1 < n) dst[(size_t)row * ld + col + 1] = (int)(w[e] >> 16);
  }
}

__global__ void __launch_bounds__(ETHREADS) k_fw_engine(const int* __restrict__ src, int ld_src, int* __restrict__ dst, int ld_dst,
                                                        int n, u16* __restrict__ work,
                                                        int* __restrict__ flags /* EngState | ver | rd */, volatile int* __restrict__ wd_flag) {
  __shared__ __align__(16) u32 At[EB * EB];        // A^T replicated: [k][row]
  __shared__ __align__(16) u32 Bs[EB * (EB / 2)];  // B: [k][word]
  __shared__ int s_task;
  const int N = (n + EB - 1) / EB;
  EngState* st = (EngState*)flags;
  int* ver = flags + 64;
  int* rd = ver + EMAX_N / EB * EMAX_N / EB;
  const int t = threadIdx.x;
  const int total = N * N * N;

  // Block 0 is a watchdog, the other blocks are workers (each performs exactly one failing fetch at the end, so a healthy run
  // ends with head == total + workers). If the queue head makes no progress for ENG_WATCHDOG_NS the run is declared hung: the host
  // is told through wd_flag and every flag is set to a huge value so that all waiting tasks are released and the kernel ends.
  const int workers = gridDim.x - 1;
  if (blockIdx.x == 0) {
    if (t == 0) {
      int last = -1;
      bool fired = false;
      unsigned long long t_last = gtime();
      while (true) {
        const int h = ld_acquire(&st->head);
        if (h >= total + workers) break;
        const unsigned long long now = gtime();
        if (h != last) { last = h; t_last = now; }
        else if (!fired && now - t_last > ENG_WATCHDOG_NS) {
          fired = true;
          *wd_flag = 1;
          for (int i = 0; i < 2 * (EMAX_N / EB) * (EMAX_N / EB); i++) ((volatile int*)ver)[i] = 0x40000000;
          __threadfence();
        }
        __nanosleep(h >= total ? 100 : 500);   // poll faster near the end so the kernel does not wait for the watchdog
      }
    }
    __syncthreads();
    // every worker has made its final fetch: reset the flags for the next launch (replaces a memset node in the graph)
    for (int i = t; i < 2 * (EMAX_N / EB) * (EMAX_N / EB); i += ETHREADS) ver[i] = 0;
    __syncthreads();
    if (t == 0) { __threadfence(); st->head = 0; }
    return;
  }
  const int r = t >> 2, cq = t & 3;

  while (true) {
    if (t == 0) s_task = atomicAdd(&st->head, 1);
    __syncthreads();
    const int q = s_task;
    if (q >= total) break;
    int k, type, ti, tj;
    decode_task(q, N, k, type, ti, tj);
    const bool first = (k == 0), last = (k == N - 1);
    u16* tileC = work + ((size_t)ti * N + tj) * ETILE;

    // ---------------- wait for dependencies (thread 0 spins, barrier releases the block) ----------------
    if (t == 0) {
      const int need_own = k;
      int need_rd;
      if (ti == tj) need_rd = (ti < k) ? 2 * (N - 1) : 0;
      else need_rd = ((ti < k) ? (N - 1) : 0) + ((tj < k) ? (N - 1) : 0);
      while (ld_acquire(&ver[ti * N + tj]) < need_own) __nanosleep(20);
      if (type == 1 || type == 2) { while (ld_acquire(&ver[k * N + k]) < k + 1) __nanosleep(20); }
      if (type == 3) {
        while (ld_acquire(&ver[ti * N + k]) < k + 1) __nanosleep(20);
        while (ld_acquire(&ver[k * N + tj]) < k + 1) __nanosleep(20);
      }
      if (need_rd) { while (ld_acquire(&rd[ti * N + tj]) < need_rd) __nanosleep(20); }
    }
    __syncthreads();

    if (type == 0) {
      // ---------------- T1: diagonal tile closure (warp 0) ----------------
      if (t < 32) {
        u32 acc[16];
        if (first) {
          const int row = ti * EB + t;
#pragma unroll
          for (int w = 0; w < 16; w++) {
            const int col = tj * EB + 2 * w;
            u32 v0 = INF16, v1 = INF16;
            if (row < n) {
              if (col < n) v0 = (u32)__ldg(&src[(size_t)row * ld_src + col]);
              if (col + 1 < n) v1 = (u32)__ldg(&src[(size_t)row * ld_src + col + 1]);
            }
            acc[w] = (v0 & 0xFFFFu) | (v1 << 16);
          }
        } else {
          const uint4* p = (const uint4*)(tileC + t * EB);
#pragma unroll
          for (int c = 0; c < 4; c++) { const uint4 v = __ldcg(p + c); acc[4 * c] = v.x; acc[4 * c + 1] = v.y; acc[4 * c + 2] = v.z; acc[4 * c + 3] = v.w; }
        }
        tile_closure(acc);
        uint4* po = (uint4*)(tileC + t * EB);
#pragma unroll
        for (int c = 0; c < 4; c++) __stcg(po + c, make_uint4(acc[4 * c], acc[4 * c + 1], acc[4 * c + 2], acc[4 * c + 3]));
        if (last) {
          const int row = ti * EB + t;
          if (row < n) {
#pragma unroll
            for (int w = 0; w < 16; w++) {
              const int col = tj * EB + 2 * w;
              if (col < n) dst[(size_t)row * ld_dst + col] = (int)(acc[w] & 0xFFFFu);
              if (col + 1 < n) dst[(size_t)row * ld_dst + col + 1] = (int)(acc[w] >> 16);
            }
          }
        }
      }
    } else {
      // ---------------- T2 / T3: C = min(C, A (x) B) ----------------
      // operands: type 1 (row panel): A = D'(k,k), B = C = tile(k,j); type 2 (col panel): A = C = tile(i,k), B = D'(k,k);
      //           type 3: A = tile(i,k), B = tile(k,j), C = tile(i,j)
      const u16* tA = (type == 1) ? (work + ((size_t)k * N + k) * ETILE) : (work + ((size_t)ti * N + k) * ETILE);
      const u16* tB = (type == 2) ? (work + ((size_t)k * N + k) * ETILE) : (work + ((size_t)k * N + tj) * ETILE);
      uint4 ca, cb, cc;
      // A chunk, B chunk (row = r, words 4cq..4cq+3); C chunk
      ca = __ldcg((const uint4*)(tA + r * EB + cq * 8));
      cb = __ldcg((const uint4*)(tB + r * EB + cq * 8));
      if (type == 1) cc = cb;
      else if (type == 2) cc = ca;
      else cc = first ? src_chunk(src, ld_src, n, ti, tj, t) : __ldcg((const uint4*)(tileC + r * EB + cq * 8));
      if (first && type != 3) {
        // own tile of round 0 comes from the int32 source (A or B operand that is the tile itself)
        const uint4 s = src_chunk(src, ld_src, n, ti, tj, t);
        if (type == 1) { cb = s; cc = s; }
        else { ca = s; cc = s; }
      }
      // stage: At[k][r] = rep(A[r][k]); Bs[k][word] = B row k
      {
        const u32 aw[4] = {ca.x, ca.y, ca.z, ca.w};
#pragma unroll
        for (int e = 0; e < 4; e++) {
          At[(cq * 8 + 2 * e) * EB + r] = rep16(aw[e]);
          At[(cq * 8 + 2 * e + 1) * EB + r] = rep16(aw[e] >> 16);
        }
        *(uint4*)&Bs[r * 16 + cq * 4] = cb;
      }
      __syncthreads();
      // release the operand tiles for writers of later rounds
      if (t == 0) {
        __threadfence();
        if (type == 3) { atomicAdd(&rd[ti * N + k], 1); atomicAdd(&rd[k * N + tj], 1); }
        else atomicAdd(&rd[k * N + k], 1);
      }
      u32 acc[4] = {cc.x, cc.y, cc.z, cc.w};
#pragma unroll 8
      for (int kk = 0; kk < EB; kk++) {
        const u32 a = At[kk * EB + r];
        const uint4 b = *(const uint4*)&Bs[kk * 16 + cq * 4];
        acc[0] = __viaddmin_s16x2(a, b.x, acc[0]);
        acc[1] = __viaddmin_s16x2(a, b.y, acc[1]);
        acc[2] = __viaddmin_s16x2(a, b.z, acc[2]);
        acc[3] = __viaddmin_s16x2(a, b.w, acc[3]);
      }
      const uint4 out = make_uint4(acc[0], acc[1], acc[2], acc[3]);
      __stcg((uint4*)(tileC + r * EB + cq * 8), out);
      if (last) dst_chunk(dst, ld_dst, n, ti, tj, t, out);
    }
    // ---------------- publish ----------------
    __syncthreads();
    if (t == 0) {
      __threadfence();
      atomicAdd(&ver[ti * N + tj], 1);
    }
    __syncthreads();
  }
}
// ======================================================================================================
// Fast path: contraction of the zero-weight strongly connected component ("core").
// All d_ij for i,j in the core are 0; the rest is solved on the reduced graph (super vertex for the core).
// ======================================================================================================
constexpr int MMAX = 320;        // max reduced size (incl. super vertex) of the fast path (squaring kernel for m > MSMALL)
constexpr int MPITCH = 320;
constexpr int MSMALL = 64;       // reduced sizes up to this are solved by the single-block closure

struct FastCtl {
  int ok;        // 1 => contraction is valid and the remaining kernels run
  int m;         // reduced size (m' + 1)
  int mprime;    // number of non-core vertices
  int z;         // core size
  int pad[4];
};

// ---- F1: zero bitmaps (row bitmap zb[u][jt], column bitmap zt[v][band]) and zero-graph degrees ---------------
// block: 256 threads (8 warps); blockIdx.x = row band (32 rows), blockIdx.y = column slice.
// outdeg / indeg (n ints, zero on entry; k_build_m clears them again) receive the row / column popcounts by atomics.
__global__ void __launch_bounds__(256) k_bitmap(const int* __restrict__ w, u32* __restrict__ zb, u32* __restrict__ zt, int* __restrict__ outdeg,
                                                int* __restrict__ indeg, int n, int nw, int nwp, int tiles_per_slice) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int r0 = blockIdx.x * 32;
  const int jt_begin = blockIdx.y * tiles_per_slice;
  const int jt_end = min(nw, jt_begin + tiles_per_slice);
  for (int jt = jt_begin + warp; jt < jt_end; jt += 8) {
    const int c = jt * 32 + lane;
    int v[32];
#pragma unroll
    for (int i = 0; i < 32; i++) {
      const int r = r0 + i;
      v[i] = (r < n && c < n) ? __ldg(&w[(size_t)r * n + c]) : 1;
    }
    u32 ztw = 0;   // bit i = (row r0+i, column c) is zero  -> column bitmap word of column c for this band
#pragma unroll
    for (int i = 0; i < 32; i++) ztw |= (u32)(v[i] == 0) << i;
    // 32x32 bit-matrix transpose across the warp (lane = row of the bit matrix): afterwards lane l holds the row word of
    // row r0+l (bit j = column jt*32+j is zero). Five shuffle stages are much cheaper than 32 ballots.
    u32 mine = ztw;
#pragma unroll
    for (int j = 16, m = 0x0000FFFF; j; j >>= 1, m ^= m << j) {
      const u32 y = __shfl_xor_sync(0xFFFFFFFFu, mine, j);
      mine = (lane & j) ? ((mine & ~(u32)m) | ((y >> j) & (u32)m)) : ((mine & (u32)m) | ((y & (u32)m) << j));
    }
    if (c < n) zt[(size_t)c * nwp + blockIdx.x] = ztw;
    if (r0 + lane < n) zb[(size_t)(r0 + lane) * nwp + jt] = mine;
    if (c < n && ztw) atomicAdd(&indeg[c], __popc(ztw));
    if (r0 + lane < n && mine) atomicAdd(&outdeg[r0 + lane], __popc(mine));
  }
}

// ---- F1 (wide variant, n % 4 == 0): warp task = (32-row band, 128-column chunk), 16-byte loads of 512-byte row stretches.
// Produces exactly the same bitmaps and degrees as k_bitmap but reads the matrix with a friendlier access pattern (about 1.5x faster
// for n >= 2000). Lane L holds columns 4L..4L+3 of the chunk; after 32 rows it owns the 4 column words of its columns, and the 8 lanes of
// a tile (32 columns) transpose their 32x32 bit matrix with 3 cross-lane and 2 in-register stages.
__global__ void __launch_bounds__(256) k_bitmap2(const int* __restrict__ w, u32* __restrict__ zb, u32* __restrict__ zt, int* __restrict__ outdeg,
                                                 int* __restrict__ indeg, int n, int nw, int nwp, int strips, int tasks) {
  const int lane = threadIdx.x & 31;
  const int gw = (blockIdx.x * 256 + threadIdx.x) >> 5;
  const int nwarps = (gridDim.x * 256) >> 5;
  const int l8 = lane & 7, jj = lane >> 3;   // lane within its 8-lane group, tile within the chunk
  for (int task = gw; task < tasks; task += nwarps) {
    const int band = task / strips, strip = task - band * strips;
    const int r0 = band * 32;
    const int c = strip * 128 + 4 * lane;    // first of this lane's 4 columns
    u32 z[4] = {0u, 0u, 0u, 0u};
    for (int i0 = 0; i0 < 32; i0 += 8) {
      int4 a[8];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int r = r0 + i0 + i;
        a[i] = (r < n && c < n) ? __ldg((const int4*)(w + (size_t)r * n + c)) : make_int4(1, 1, 1, 1);
      }
#pragma unroll
      for (int i = 0; i < 8; i++) {
        z[0] |= (u32)(a[i].x == 0) << (i0 + i);
        z[1] |= (u32)(a[i].y == 0) << (i0 + i);
        z[2] |= (u32)(a[i].z == 0) << (i0 + i);
        z[3] |= (u32)(a[i].w == 0) << (i0 + i);
      }
    }
    // column words: bit i = (row r0+i, column c+k) is zero
#pragma unroll
    for (int k = 0; k < 4; k++) {
      if (c + k < n) {
        zt[(size_t)(c + k) * nwp + band] = z[k];
        if (z[k]) atomicAdd(&indeg[c + k], __popc(z[k]));
      }
    }
    // transpose: word index within the tile is a = 4*l8 + k; stage j exchanges the off-diagonal j x j blocks of words a and a^j
    u32 x[4] = {z[0], z[1], z[2], z[3]};
#pragma unroll
    for (int j = 16, m = 0x0000FFFF; j; j >>= 1, m ^= m << j) {
      if (j >= 4) {
        const int sh = j >> 2;
        const bool upper = (l8 & sh) != 0;
#pragma unroll
        for (int k = 0; k < 4; k++) {
          const u32 y = __shfl_xor_sync(0xFFFFFFFFu, x[k], sh);
          x[k] = upper ? ((x[k] & ~(u32)m) | ((y >> j) & (u32)m)) : ((x[k] & (u32)m) | ((y & (u32)m) << j));
        }
      } else {
#pragma unroll
        for (int p = 0; p < 4; p++) {
          if ((p & j) == 0) {
            const u32 lo = x[p], hi = x[p | j];
            x[p] = (lo & (u32)m) | ((hi & (u32)m) << j);
            x[p | j] = (hi & ~(u32)m) | ((lo >> j) & (u32)m);
          }
        }
      }
    }
    // x[k] = row word of row r0 + 4*l8 + k, tile strip*4 + jj (bit b = column 32*tile + b is zero)
    const int jt = strip * 4 + jj;
    if (jt < nw) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const int r = r0 + 4 * l8 + k;
        if (r < n) {
          zb[(size_t)r * nwp + jt] = x[k];
          if (x[k]) atomicAdd(&outdeg[r], __popc(x[k]));
        }
      }
    }
  }
}

// ---- F2: forward/backward BFS from one pivot in a 2-block cluster (block 0 = forward, block 1 = backward) ----------
// Any partial BFS is valid: every vertex in F_partial & B_partial has a genuine zero-weight walk from and to the pivot,
// so contracting that set is exact (needs >= 2 vertices so that all d_ii = 0 inside).
__device__ __forceinline__ u32 valid_mask(int j, int n, int nw) {
  return (j == nw - 1 && (n & 31)) ? ((1u << (n & 31)) - 1u) : 0xFFFFFFFFu;
}
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src) {
  const unsigned sa = (unsigned)__cvta_generic_to_shared(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem_src));
}
constexpr int SCC_SMEM_N = 1280;
// bitmap kept in shared memory when n <= SCC_SMEM_N
constexpr int SCC_T = 1024;   // threads per block of the SCC kernel
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(SCC_T)
k_scc2(const u32* __restrict__ zb, const u32* __restrict__ zt, const int* __restrict__ outdeg, const int* __restrict__ indeg, int n, int nw, int nwp,
       int smem_mode, u32* __restrict__ corebits, int* __restrict__ wprefix, int* __restrict__ rvert, u16* __restrict__ mapv,
       FastCtl* __restrict__ ctl, int mok) {
  extern __shared__ __align__(16) u32 sm[];
  cg::cluster_group cluster = cg::this_cluster();
  const int dir = (int)cluster.block_rank();
  u32* vis = sm;
  u32* fr = sm + nwp;
  u32* nx = sm + 2 * nwp;
  u32* core_s = sm + 3 * nwp;            // merge (block 0): core bits
  int* wp_s = (int*)(sm + 4 * nwp);      // merge (block 0): word prefix of non-core counts
  u32* skip_s = sm + 5 * nwp;            // vertices this direction's BFS can never reach (no zero edge into / out of them)
  u32* rows_s = sm + 6 * nwp;            // smem mode: this block's direction bitmap [n][nwp]
  int* flist = (int*)(rows_s + (smem_mode ? (size_t)n * nwp : 0));   // smem mode: frontier vertex list of the single-warp BFS
  __shared__ u64 red[32];
  __shared__ int cntArr[64];
  __shared__ int s_pivot;
  __shared__ int s_nskip;
  __shared__ int scan_w[33];
  __shared__ int s_carry;
  __shared__ int s_z;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const u32* my_rows = dir ? zt : zb;   // top-down rows: out-neighbours (fwd) / in-neighbours (bwd)
  const u32* alt_rows = dir ? zb : zt;  // bottom-up rows (L2 mode only)
  const int nq = nwp >> 2;

  for (int j = t; j < nwp; j += SCC_T) { vis[j] = 0; fr[j] = 0; nx[j] = 0; skip_s[j] = 0; }
  if (t < 64) cntArr[t] = 0;
  if (t == 0) s_nskip = 0;
  if (smem_mode) {
    // asynchronous copy of this direction's bitmap into shared memory (waited for before the BFS)
    const int total4 = n * (nwp >> 2);
    const uint4* src = (const uint4*)my_rows;
    uint4* dst = (uint4*)rows_s;
    for (int i = t; i < total4; i += SCC_T) cp_async16(dst + i, src + i);
    asm volatile("cp.async.commit_group;\n" ::);
  }
  __syncthreads();

  // ---- pivot: the vertex with the largest min(out-degree, in-degree); also the vertices this direction can never reach
  {
    u64 best = 0;  // key = (score+1) << 32 | (0xFFFFFFFF - v)
    int nskip = 0;
    for (int wi = warp; wi < nw; wi += SCC_T / 32) {
      const int v = wi * 32 + lane;
      const bool in_range = v < n;
      const int o = in_range ? outdeg[v] : 0, d_in = in_range ? indeg[v] : 0;
      if (in_range) {
        const u64 key = ((u64)(min(o, d_in) + 1) << 32) | (u64)(0xFFFFFFFFu - (u32)v);
        if (key > best) best = key;
      }
      const u32 m = __ballot_sync(0xFFFFFFFFu, in_range && (dir ? (o == 0) : (d_in == 0)));
      if (lane == 0) { skip_s[wi] = m; nskip += __popc(m); }
    }
    if (lane == 0 && nskip) atomicAdd(&s_nskip, nskip);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const u64 other = __shfl_xor_sync(0xFFFFFFFFu, best, o);
      if (other > best) best = other;
    }
    if (lane == 0) red[warp] = best;
    __syncthreads();
    if (warp == 0) {
      u64 x = red[lane];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const u64 other = __shfl_xor_sync(0xFFFFFFFFu, x, o);
        if (other > x) x = other;
      }
      if (lane == 0) {
        const int sc = (int)(x >> 32) - 1;
        s_pivot = (x == 0 || sc <= 0) ? -1 : (int)(0xFFFFFFFFu - (u32)(x & 0xFFFFFFFFu));
      }
    }
    __syncthreads();
  }
  const int p = s_pivot;
  if (smem_mode) {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
  }

  // ---- BFS
  if (p >= 0 && smem_mode) {
    // The whole graph is in shared memory and the frontiers are thin, so one warp runs the entire BFS without any block-wide
    // synchronization: lane l owns words l and l+32 of the visited set and of the frontier (n <= SCC_SMEM_N => nw <= 40).
    if (warp == 0) {
      const u32 vm0 = (lane < nw) ? valid_mask(lane, n, nw) : 0u;
      const u32 vm1 = (lane + 32 < nw) ? valid_mask(lane + 32, n, nw) : 0u;
      const bool has0 = lane < nwp, has1 = lane + 32 < nwp;
      u32 vis0 = 0, vis1 = 0, fr0 = 0, fr1 = 0;
      if ((p >> 5) == lane) { vis0 = 1u << (p & 31); fr0 = vis0; }
      if ((p >> 5) == lane + 32) { vis1 = 1u << (p & 31); fr1 = vis1; }
      const bool two_words = nwp > 32;   // n > 1024: rows have more than 32 words
      const u32* rp0 = rows_s + (has0 ? lane : 0);   // lanes without a row word read a dummy word; vm0/vm1 mask them out below
      const u32* rp1 = rows_s + (has1 ? 32 + lane : 0);
      for (int level = 0; level < 40; level++) {
        u32 acc0 = 0, acc1 = 0;
        // compact the frontier into a vertex list (lane l owns words l and l+32)
        int incl = __popc(fr0) + __popc(fr1);
        const int mine = incl;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const int y = __shfl_up_sync(0xFFFFFFFFu, incl, o);
          if (lane >= o) incl += y;
        }
        const int total = __shfl_sync(0xFFFFFFFFu, incl, 31);
        int pos = incl - mine;
        for (u32 x = fr0; x; x &= x - 1) flist[pos++] = lane * 32 + (__ffs(x) - 1);
        if (two_words) for (u32 x = fr1; x; x &= x - 1) flist[pos++] = (lane + 32) * 32 + (__ffs(x) - 1);
        __syncwarp();
        for (int k = 0; k < total; k += 16) {
          int off[16];
#pragma unroll
          for (int i = 0; i < 16; i++) off[i] = flist[min(k + i, total - 1)] * nwp;   // padding entries repeat the last vertex (OR is idempotent)
          u32 a0[16];
#pragma unroll
          for (int i = 0; i < 16; i++) a0[i] = rp0[off[i]];
#pragma unroll
          for (int i = 0; i < 16; i++) acc0 |= a0[i];
          if (two_words) {
            u32 a1[16];
#pragma unroll
            for (int i = 0; i < 16; i++) a1[i] = rp1[off[i]];
#pragma unroll
            for (int i = 0; i < 16; i++) acc1 |= a1[i];
          }
        }
        __syncwarp();
        const u32 new0 = acc0 & ~vis0 & vm0, new1 = acc1 & ~vis1 & vm1;
        vis0 |= new0; vis1 |= new1; fr0 = new0; fr1 = new1;
        if (!__any_sync(0xFFFFFFFFu, (new0 | new1) != 0u)) break;
      }
      if (lane < nw) vis[lane] = vis0;
      if (lane + 32 < nw) vis[lane + 32] = vis1;
    }
  } else if (p >= 0) {
    // ---- multi-warp BFS for large n (level-synchronous, 2 barriers per level)
    if (t == 0) { vis[p >> 5] = 1u << (p & 31); fr[p >> 5] = 1u << (p & 31); }
    __syncthreads();
    int cntFr = 1, unvis = n - 1 - s_nskip, visited = 1;
    const int cap = smem_mode ? 30 : 14;
    for (int level = 0; level < cap; level++) {
      const bool bottom_up = (!smem_mode) && (unvis < cntFr);
      if (!bottom_up) {
        // warp-cooperative top-down expansion: lane covers 16-byte chunks (cg + lane, cg + 32 + lane) of each frontier row;
        // rows of up to 4 frontier vertices are OR-ed in registers; one atomic per chunk word per warp at the end.
        for (int cgi = 0; cgi < nq; cgi += 64) {
          uint4 acc0 = make_uint4(0, 0, 0, 0), acc1 = make_uint4(0, 0, 0, 0);
          const int q0 = cgi + lane, q1 = cgi + 32 + lane;
          for (int wi = warp; wi < nw; wi += SCC_T / 32) {
            u32 x = fr[wi];
            while (x) {
              uint4 y0[4], y1[4];
#pragma unroll
              for (int i = 0; i < 4; i++) {
                y0[i] = make_uint4(0, 0, 0, 0);
                y1[i] = make_uint4(0, 0, 0, 0);
                if (x) {
                  const int u = wi * 32 + (__ffs(x) - 1);
                  x &= x - 1;
                  if (smem_mode) {
                    const uint4* row = (const uint4*)(rows_s + (size_t)u * nwp);
                    if (q0 < nq) y0[i] = row[q0];
                    if (q1 < nq) y1[i] = row[q1];
                  } else {
                    const uint4* row = (const uint4*)(my_rows + (size_t)u * nwp);
                    if (q0 < nq) y0[i] = __ldg(&row[q0]);
                    if (q1 < nq) y1[i] = __ldg(&row[q1]);
                  }
                }
              }
#pragma unroll
              for (int i = 0; i < 4; i++) {
                acc0.x |= y0[i].x; acc0.y |= y0[i].y; acc0.z |= y0[i].z; acc0.w |= y0[i].w;
                acc1.x |= y1[i].x; acc1.y |= y1[i].y; acc1.z |= y1[i].z; acc1.w |= y1[i].w;
              }
            }
          }
          if (q0 < nq) {
            if (acc0.x) atomicOr(&nx[q0 * 4], acc0.x);
            if (acc0.y) atomicOr(&nx[q0 * 4 + 1], acc0.y);
            if (acc0.z) atomicOr(&nx[q0 * 4 + 2], acc0.z);
            if (acc0.w) atomicOr(&nx[q0 * 4 + 3], acc0.w);
          }
          if (q1 < nq) {
            if (acc1.x) atomicOr(&nx[q1 * 4], acc1.x);
            if (acc1.y) atomicOr(&nx[q1 * 4 + 1], acc1.y);
            if (acc1.z) atomicOr(&nx[q1 * 4 + 2], acc1.z);
            if (acc1.w) atomicOr(&nx[q1 * 4 + 3], acc1.w);
          }
        }
      } else {
        for (int wi = warp; wi < nw; wi += SCC_T / 32) {
          const u32 x = ~vis[wi] & ~skip_s[wi] & valid_mask(wi, n, nw);
          if (!x) continue;
          const int bit = (int)__fns(x, 0, lane + 1);
          if (bit >= 0) {
            const uint4* row = (const uint4*)(alt_rows + (size_t)(wi * 32 + bit) * nwp);
            bool hit = false;
            for (int j4 = 0; j4 < nq && !hit; j4 += 4) {
              uint4 y[4];
#pragma unroll
              for (int q = 0; q < 4; q++) y[q] = (j4 + q < nq) ? __ldg(&row[j4 + q]) : make_uint4(0, 0, 0, 0);
#pragma unroll
              for (int q = 0; q < 4; q++) {
                const uint4 f = *(const uint4*)&fr[(j4 + q) * 4];
                hit |= ((y[q].x & f.x) | (y[q].y & f.y) | (y[q].z & f.z) | (y[q].w & f.w)) != 0u;
              }
            }
            if (hit) atomicOr(&nx[wi], 1u << bit);
          }
        }
      }
      __syncthreads();
      int c = 0;
      for (int j = t; j < nw; j += SCC_T) {
        const u32 nv = nx[j] & ~vis[j];
        nx[j] = 0;
        vis[j] |= nv;
        fr[j] = nv;
        c += __popc(nv);
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xFFFFFFFFu, c, o);
      if (lane == 0 && c) atomicAdd(&cntArr[level], c);
      __syncthreads();
      const int total = cntArr[level];
      if (total == 0) break;
      cntFr = total;
      unvis -= total;
      visited += total;
      // tail levels add little: stop once a level yields < n/256 new vertices after the bulk was reached (partial sets stay valid)
      if (total < (n >> 8) && visited * 2 >= n) break;
    }
  }
  cluster.sync();   // both BFS results ready

  if (dir == 0) {
    // ---------------- merge (block 0 only) ----------------
    const u32* visB = cluster.map_shared_rank(vis, 1);
    int cnt = 0;
    for (int j = t; j < nw; j += SCC_T) {
      const u32 c = (p >= 0) ? (vis[j] & visB[j] & valid_mask(j, n, nw)) : 0u;
      core_s[j] = c;
      cnt += __popc(c);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xFFFFFFFFu, cnt, o);
    if (t == 0) s_z = 0;
    __syncthreads();
    if (lane == 0 && cnt) atomicAdd(&s_z, cnt);
    __syncthreads();
    const int z = s_z;
    const int mprime = n - z;
    const bool ok = (z >= 2) && (mprime + 1 <= mok);
    if (t == 0) {
      ctl->ok = ok ? 1 : 0;
      ctl->z = z;
      ctl->mprime = mprime;
      ctl->m = mprime + 1;
    }
    if (ok) {
      if (t == 0) s_carry = 0;
      __syncthreads();
      for (int base = 0; base < nw; base += SCC_T) {
        const int j = base + t;
        const u32 cw = (j < nw) ? core_s[j] : 0u;
        const int ncnt = (j < nw) ? __popc(~cw & valid_mask(j, n, nw)) : 0;
        int incl = ncnt;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const int y = __shfl_up_sync(0xFFFFFFFFu, incl, o);
          if (lane >= o) incl += y;
        }
        if (lane == 31) scan_w[warp] = incl;
        __syncthreads();
        if (warp == 0) {
          int w = scan_w[lane];
          int wi_ = w;
#pragma unroll
          for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xFFFFFFFFu, wi_, o);
            if (lane >= o) wi_ += y;
          }
          scan_w[lane] = wi_ - w;                 // exclusive offsets of warps
          if (lane == 31) scan_w[32] = wi_;       // chunk total
        }
        __syncthreads();
        if (j < nw) {
          const int excl = s_carry + scan_w[warp] + incl - ncnt;
          wp_s[j] = excl;
          wprefix[j] = excl;
          corebits[j] = cw;
        }
        __syncthreads();
        if (t == 0) s_carry += scan_w[32];
        __syncthreads();
      }
      for (int v = t; v < n; v += SCC_T) {
        const int wd = v >> 5, bit = v & 31;
        const u32 cw = core_s[wd];
        if ((cw >> bit) & 1u) {
          mapv[v] = (u16)mprime;
        } else {
          const int idx = wp_s[wd] + __popc(~cw & ((1u << bit) - 1u));
          rvert[idx] = v;
          mapv[v] = (u16)idx;
        }
      }
    }
  }
  cluster.sync();   // block 1 must stay alive while block 0 reads its shared memory
}

// ---- F3: build the reduced matrix M (int32, pitch MPITCH) ------------------------------------------
// grid = 2*MMAX blocks: blocks [0,MMAX): row a of M (M[a][*]); blocks [MMAX,2*MMAX): entry M[m'][b]
__global__ void __launch_bounds__(256) k_build_m(const int* __restrict__ w, const u32* __restrict__ corebits, const int* __restrict__ wprefix,
                                                 const int* __restrict__ rvert, const FastCtl* __restrict__ ctl, int* __restrict__ M, int n,
                                                 int* __restrict__ outdeg, int* __restrict__ indeg) {
  // the degree arrays (filled by k_bitmap, read by k_scc2) must be zero again for the next call
  for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) { outdeg[i] = 0; indeg[i] = 0; }
  if (!ctl->ok) return;
  const int mp = ctl->mprime;
  const int t = threadIdx.x;
  __shared__ int red[8];
  int bidx = blockIdx.x;
  if (bidx < MMAX) {
    const int a = bidx;
    if (a >= mp) return;
    const int r = rvert[a];
    int mn = 1 << 20;
    const int* row = w + (size_t)r * n;
    for (int j0 = t; j0 < n; j0 += 256 * 8) {   // 8 independent elements per thread in flight
      int v[8], wpv[8];
      u32 cw[8];
#pragma unroll
      for (int k = 0; k < 8; k++) {
        const int j = j0 + k * 256;
        v[k] = 0; wpv[k] = 0; cw[k] = 0;
        if (j < n) { v[k] = __ldg(&row[j]); cw[k] = corebits[j >> 5]; wpv[k] = wprefix[j >> 5]; }
      }
#pragma unroll
      for (int k = 0; k < 8; k++) {
        const int j = j0 + k * 256;
        if (j < n) {
          if ((cw[k] >> (j & 31)) & 1u) {
            mn = min(mn, v[k]);
          } else {
            const int idx = wpv[k] + __popc(~cw[k] & ((1u << (j & 31)) - 1u) & valid_mask(j >> 5, n, (n + 31) >> 5));
            M[(size_t)a * MPITCH + idx] = v[k];
          }
        }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mn = min(mn, __shfl_xor_sync(0xFFFFFFFFu, mn, o));
    if ((t & 31) == 0) red[t >> 5] = mn;
    __syncthreads();
    if (t == 0) {
      int x = red[0];
      for (int i = 1; i < 8; i++) x = min(x, red[i]);
      M[(size_t)a * MPITCH + mp] = x;
    }
  } else {
    const int b = bidx - MMAX;
    if (b > mp) return;
    if (b == mp) { if (t == 0) M[(size_t)mp * MPITCH + mp] = 0; return; }
    const int r = rvert[b];
    int mn = 1 << 20;
    for (int u0 = t; u0 < n; u0 += 256 * 8) {   // 8 independent loads per thread in flight
      int x[8];
      u32 cw[8];
#pragma unroll
      for (int k = 0; k < 8; k++) {
        const int u = u0 + k * 256;
        x[k] = 1 << 20; cw[k] = 0;
        if (u < n) { cw[k] = corebits[u >> 5]; x[k] = __ldg(&w[(size_t)u * n + r]); }
      }
#pragma unroll
      for (int k = 0; k < 8; k++) {
        const int u = u0 + k * 256;
        if (u < n && ((cw[k] >> (u & 31)) & 1u)) mn = min(mn, x[k]);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mn = min(mn, __shfl_xor_sync(0xFFFFFFFFu, mn, o));
    if ((t & 31) == 0) red[t >> 5] = mn;
    __syncthreads();
    if (t == 0) {
      int x = red[0];
      for (int i = 1; i < 8; i++) x = min(x, red[i]);
      M[(size_t)mp * MPITCH + b] = x;
    }
  }
}

// ---- F5: expand the reduced closure to the full n x n output ---------------------------------------
// out[i][j] = Dm[map(i)][map(j)]; also publishes the fast-path status word (2 = declined, 3 = done) for the host.
__global__ void __launch_bounds__(256) k_expand(const int* __restrict__ Dm, const u16* __restrict__ mapv, const FastCtl* __restrict__ ctl,
                                                int* __restrict__ out, int n, volatile int* __restrict__ host_flag) {
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) *host_flag = ctl->ok + 2;
  if (!ctl->ok) return;
  const int chunks = (n + 3) >> 2;
  for (int i = blockIdx.y; i < n; i += gridDim.y) {
    const int mi = mapv[i];
    const int* drow = Dm + (size_t)mi * MPITCH;
    int* orow = out + (size_t)i * n;
    for (int ch = blockIdx.x * 256 + threadIdx.x; ch < chunks; ch += gridDim.x * 256) {
      const int j0 = ch * 4;
      if (j0 + 3 < n && ((((size_t)i * n + j0) & 3) == 0)) {
        const uint2 mj = *(const uint2*)(mapv + j0);
        int4 o;
        o.x = __ldg(&drow[mj.x & 0xFFFFu]);
        o.y = __ldg(&drow[mj.x >> 16]);
        o.z = __ldg(&drow[mj.y & 0xFFFFu]);
        o.w = __ldg(&drow[mj.y >> 16]);
        *(int4*)(orow + j0) = o;
      } else {
        for (int j = j0; j < min(n, j0 + 4); j++) orow[j] = __ldg(&drow[mapv[j]]);
      }
    }
  }
}
// ======================================================================================================
// Min-plus squaring closure: P <- min(P, P (x) P), repeated until nothing changes (at most ceil(log2 n) times; every shortest walk
// has at most n edges). Persistent kernel: one 32x32 output tile per block and iteration, the K range is split over the 16 warps,
// a grid barrier per iteration also carries the number of blocks that changed something.
// ======================================================================================================
constexpr int SQ_THREADS = 512;
constexpr int SQ_NMAX = 512;                     // largest n (np <= 512: shared memory and the 1024-bit arrival field)
constexpr int SQ_TMAX = 10;                      // barrier lines for the iterations
constexpr int SQ_LINE = 32;                      // ints per sync line (one 128-byte line per counter)
constexpr int SQ_SYNC_INTS = (SQ_TMAX + 3) * SQ_LINE;   // iteration barriers | prologue barrier | done | abort
constexpr unsigned long long SQ_TIMEOUT_NS = 2000000000ull;

__device__ __forceinline__ u32 min16x2(u32 a, u32 b) { return __viaddmin_s16x2(a, 0u, b); }

// Grid barrier over the G active blocks; `add` = 1 + 1024 * changed. Returns the number of blocks that reported a change, or -1
// when the wait was abandoned (timeout or abort by another block).
__device__ int sq_barrier(int* sync, int line, int G, int add) {
  __shared__ int s_res;
  __syncthreads();
  if (threadIdx.x == 0) {
    int* cnt = sync + line * SQ_LINE;
    __threadfence();
    atomicAdd(cnt, add);
    unsigned long long t0 = 0;
    int spins = 0, res;
    while (true) {
      const int v = ld_acquire(cnt);
      if ((v & 1023) >= G) { res = v >> 10; break; }
      if ((++spins & 63) == 0) {
        const unsigned long long now = gtime();
        if (t0 == 0) t0 = now;
        else if (now - t0 > SQ_TIMEOUT_NS || ld_acquire(sync + (SQ_TMAX + 2) * SQ_LINE)) { res = -1; break; }
      }
      __nanosleep(20);
    }
    s_res = res;
  }
  __syncthreads();
  return s_res;
}

// 8 consecutive source columns of one row as 4 packed 16-bit pairs (padding = INF16)
__device__ __forceinline__ uint4 load8_src(const int* __restrict__ src, int ld, int n, int row, int col0) {
  u32 wd[4];
  if (row < n) {
    const size_t off = (size_t)row * ld + col0;
    if (col0 + 8 <= n && (off & 3) == 0) {
      const int4 x0 = __ldg((const int4*)(src + off)), x1 = __ldg((const int4*)(src + off + 4));
      wd[0] = ((u32)x0.x & 0xFFFFu) | ((u32)x0.y << 16);
      wd[1] = ((u32)x0.z & 0xFFFFu) | ((u32)x0.w << 16);
      wd[2] = ((u32)x1.x & 0xFFFFu) | ((u32)x1.y << 16);
      wd[3] = ((u32)x1.z & 0xFFFFu) | ((u32)x1.w << 16);
    } else {
#pragma unroll
      for (int e = 0; e < 4; e++) {
        u32 v0 = INF16, v1 = INF16;
        if (col0 + 2 * e < n) v0 = (u32)__ldg(src + off + 2 * e);
        if (col0 + 2 * e + 1 < n) v1 = (u32)__ldg(src + off + 2 * e + 1);
        wd[e] = (v0 & 0xFFFFu) | (v1 << 16);
      }
    }
  } else {
    wd[0] = wd[1] = wd[2] = wd[3] = (u32)INF16 * 0x10001u;
  }
  return make_uint4(wd[0], wd[1], wd[2], wd[3]);
}

// src/dst: int32 matrices (row pitches ld_src/ld_dst); n = n_static, or ctl[1] when ctl != nullptr (the reduced size of the fast
// path; the kernel then only runs when ctl[0] != 0 and n_lo < ctl[1] <= n_hi). P: two u16 buffers of psz elements with row pitch ldp
// (>= np, multiple of 8). sync: SQ_SYNC_INTS ints, all zero on entry and again on exit (the last block to finish clears them).
__global__ void __launch_bounds__(SQ_THREADS, 1)
k_sq(const int* __restrict__ src, int ld_src, int* __restrict__ dst, int ld_dst, int n_static, int* __restrict__ ctl, int n_lo, int n_hi,
     u16* __restrict__ P, int ldp, int psz, int* __restrict__ sync, volatile int* wd_flag) {
  if (ctl && (ctl[0] == 0 || ctl[1] <= n_lo || ctl[1] > n_hi)) return;
  const int n = ctl ? ctl[1] : n_static;
  const int N = (n + 31) >> 5, np = N << 5, NN = N * N;
  const int RA = ((np >> 1) + 31) & ~31;   // words per A row in shared memory: a multiple of 32 (whole swizzle groups)
  const int W = gridDim.x, G = min(W, NN), w = blockIdx.x;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  extern __shared__ __align__(16) unsigned char sq_raw[];
  u32* As = (u32*)sq_raw;      // A rows I*32..+31, natural [row][k pair] with a 16-byte-unit swizzle
  u32* Bs = As + 32 * RA;      // B columns J*32..+31: [k][16 words]
  u32* red = Bs + np * 16;     // per-warp partial tiles [16][32 rows][16 words]
  int res = 0;

  if (w < G) {
    const int T = 32 - __clz(max(n - 1, 1));   // 2^T >= n
    for (int it = 0; it < T && res >= 0; it++) {
      const u16* Pin = P + (size_t)((it - 1) & 1) * psz;   // iteration 0 reads the int32 source instead
      u16* Pout = P + (size_t)(it & 1) * psz;
      int chg = 0;
      for (int q = w; q < NN; q += G) {
        const int I = q / N, J = q - I * N;
        // ---- load the A rows and B columns of this tile (iteration 0: from the source; later: L2 only, other SMs wrote them)
        const int U = np >> 3;   // 16-byte units per A row
        uint4 old = make_uint4(0, 0, 0, 0);
        if (it == 0) {
          for (int c = t; c < 32 * U; c += SQ_THREADS) {
            const int r = c / U, u = c - r * U;
            *(uint4*)(As + r * RA + ((u ^ ((r >> 2) & 7)) << 2)) = load8_src(src, ld_src, n, I * 32 + r, u * 8);
          }
          for (int c = t; c < np * 4; c += SQ_THREADS) {
            const int kk = c >> 2, ch = c & 3;
            *(uint4*)(Bs + kk * 16 + ch * 4) = load8_src(src, ld_src, n, kk, J * 32 + ch * 8);
          }
          if (t < 128) old = load8_src(src, ld_src, n, I * 32 + (t >> 2), J * 32 + (t & 3) * 8);
        } else {
          for (int c = t; c < 32 * U; c += SQ_THREADS) {
            const int r = c / U, u = c - r * U;
            cp_async16(As + r * RA + ((u ^ ((r >> 2) & 7)) << 2), Pin + (size_t)(I * 32 + r) * ldp + u * 8);
          }
          for (int c = t; c < np * 4; c += SQ_THREADS) {
            const int kk = c >> 2, ch = c & 3;
            cp_async16(Bs + kk * 16 + ch * 4, Pin + (size_t)kk * ldp + J * 32 + ch * 8);
          }
          asm volatile("cp.async.commit_group;\n" ::);
          if (t < 128) old = __ldcg((const uint4*)(Pin + (size_t)(I * 32 + (t >> 2)) * ldp + J * 32 + (t & 3) * 8));
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        __syncthreads();
        // ---- partial product over this warp's K slice (N pairs of k): lane = 4 rows x 4 words
        {
          const int rg = lane >> 2, wq = lane & 3;
          const int mask = (rg & 7) << 2;
          const u32* a0 = As + (4 * rg) * RA;
          const u32* bq = Bs + 4 * wq;
          u32 acc[4][4];
#pragma unroll
          for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][j] = 0x7FFF7FFFu;
          const int kp0 = warp * N;
#pragma unroll 2
          for (int kp = kp0; kp < kp0 + N; kp++) {
            const int ws = kp ^ mask;
            const u32 xs[4] = {a0[ws], a0[RA + ws], a0[2 * RA + ws], a0[3 * RA + ws]};
            const uint4 b0 = *(const uint4*)(bq + (2 * kp) * 16);
            const uint4 b1 = *(const uint4*)(bq + (2 * kp + 1) * 16);
#pragma unroll
            for (int i = 0; i < 4; i++) {
              const u32 lo = __byte_perm(xs[i], 0, 0x1010), hi = __byte_perm(xs[i], 0, 0x3232);
              acc[i][0] = __viaddmin_s16x2(lo, b0.x, acc[i][0]);
              acc[i][1] = __viaddmin_s16x2(lo, b0.y, acc[i][1]);
              acc[i][2] = __viaddmin_s16x2(lo, b0.z, acc[i][2]);
              acc[i][3] = __viaddmin_s16x2(lo, b0.w, acc[i][3]);
              acc[i][0] = __viaddmin_s16x2(hi, b1.x, acc[i][0]);
              acc[i][1] = __viaddmin_s16x2(hi, b1.y, acc[i][1]);
              acc[i][2] = __viaddmin_s16x2(hi, b1.z, acc[i][2]);
              acc[i][3] = __viaddmin_s16x2(hi, b1.w, acc[i][3]);
            }
          }
#pragma unroll
          for (int i = 0; i < 4; i++)
            *(uint4*)&red[warp * 512 + (4 * rg + i) * 16 + 4 * wq] = make_uint4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        }
        __syncthreads();
        // ---- combine the partial tiles, keep the old value when it is smaller, publish
        int chg_t = 0;
        if (t < 128) {
          const int r = t >> 2, cq = t & 3;
          uint4 m = *(const uint4*)&red[r * 16 + cq * 4];
#pragma unroll
          for (int w16 = 1; w16 < 16; w16++) {
            const uint4 v = *(const uint4*)&red[w16 * 512 + r * 16 + cq * 4];
            m.x = min16x2(m.x, v.x); m.y = min16x2(m.y, v.y); m.z = min16x2(m.z, v.z); m.w = min16x2(m.w, v.w);
          }
          m.x = min16x2(m.x, old.x); m.y = min16x2(m.y, old.y); m.z = min16x2(m.z, old.z); m.w = min16x2(m.w, old.w);
          chg_t = (m.x != old.x) | (m.y != old.y) | (m.z != old.z) | (m.w != old.w);
          const int row = I * 32 + r, col0 = J * 32 + cq * 8;
          __stcg((uint4*)(Pout + (size_t)row * ldp + col0), m);
          if (row < n) {   // the final iteration leaves the closure in dst (every iteration writes, the last one wins)
            const u32 wd[4] = {m.x, m.y, m.z, m.w};
            int* d = dst + (size_t)row * ld_dst + col0;
            if (col0 + 8 <= n && ((((size_t)row * ld_dst + col0) & 3) == 0)) {
              *(int4*)d = make_int4((int)(wd[0] & 0xFFFFu), (int)(wd[0] >> 16), (int)(wd[1] & 0xFFFFu), (int)(wd[1] >> 16));
              *(int4*)(d + 4) = make_int4((int)(wd[2] & 0xFFFFu), (int)(wd[2] >> 16), (int)(wd[3] & 0xFFFFu), (int)(wd[3] >> 16));
            } else {
#pragma unroll
              for (int e = 0; e < 4; e++) {
                if (col0 + 2 * e < n) d[2 * e] = (int)(wd[e] & 0xFFFFu);
                if (col0 + 2 * e + 1 < n) d[2 * e + 1] = (int)(wd[e] >> 16);
              }
            }
          }
        }
        chg |= __syncthreads_or(chg_t);   // also separates this tile's shared-memory use from the next one
      }
      res = sq_barrier(sync, it, G, 1 + (chg ? 1024 : 0));
      if (res == 0) break;   // nothing changed: P is closed
    }
  }
  __syncthreads();
  if (t == 0) {
    if (res < 0) {   // abandoned: tell the host, stop the others, and make sure the fast path does not use the garbage
      *wd_flag = 1;
      atomicExch(sync + (SQ_TMAX + 2) * SQ_LINE, 1);
      if (ctl) ctl[0] = 0;
    }
    __threadfence();
    if (atomicAdd(sync + (SQ_TMAX + 1) * SQ_LINE, 1) == W - 1) {   // last block out: reset the counters for the next launch
      for (int l = 0; l < SQ_TMAX + 3; l++) sync[l * SQ_LINE] = 0;
    }
  }
}

// ---- F5 (split variant): the output buffer is zero-filled by k_zero in a parallel graph branch (the core x core block is all zeros);
// k_fixup then writes the rows and columns of the non-core vertices: out[r][j] = Dm[a][map j], out[i][r] = Dm[map i][a] with r = rvert[a].
__global__ void __launch_bounds__(256) k_zero(int4* __restrict__ p, size_t n4, int* __restrict__ tail, int ntail) {
  const int4 z = make_int4(0, 0, 0, 0);
  const size_t stride = (size_t)gridDim.x * 256;
  size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  for (; i + 3 * stride < n4; i += 4 * stride) {
    __stcs(p + i, z); __stcs(p + i + stride, z); __stcs(p + i + 2 * stride, z); __stcs(p + i + 3 * stride, z);
  }
  for (; i < n4; i += stride) __stcs(p + i, z);
  if (blockIdx.x == 0 && (int)threadIdx.x < ntail) tail[threadIdx.x] = 0;
}

// grid = (gx, MMAX): blockIdx.y = a (non-core index, blocks with a >= m' return); also publishes the status word (2 declined, 3 done)
__global__ void __launch_bounds__(256) k_fixup(const int* __restrict__ Dm, const u16* __restrict__ mapv, const int* __restrict__ rvert,
                                               const FastCtl* __restrict__ ctl, int* __restrict__ out, int n, volatile int* __restrict__ host_flag) {
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) *host_flag = ctl->ok + 2;
  if (!ctl->ok) return;
  const int a = blockIdx.y;
  if (a >= ctl->mprime) return;
  const int r = rvert[a];
  const int* drow = Dm + (size_t)a * MPITCH;
  int* orow = out + (size_t)r * n;
  for (int j = blockIdx.x * 256 + threadIdx.x; j < n; j += gridDim.x * 256) orow[j] = __ldg(&drow[mapv[j]]);
  for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) out[(size_t)i * n + r] = __ldg(&Dm[(size_t)mapv[i] * MPITCH + a]);
}
// ======================================================================================================
// Host side
// ======================================================================================================
enum { DENSE_SMALL = 0, DENSE_SQ = 1, DENSE_ENGINE = 2, DENSE_V1 = 3 };
constexpr int NFAST = 385;             // try the contraction fast path when n >= NFAST (P(reduced size <= MMAX) >= 0.95 there)
constexpr int SMALL_DENSE_MAX = 64;    // dense n <= this: single-block register closure; up to SQ_NMAX: squaring; up to EMAX_N: persistent engine

static int g_n = 0, g_np = 0, g_N = 0, g_np32 = 0, g_N32 = 0, g_nw = 0, g_nwp = 0;
static int* g_d = nullptr;      // int32 n x n: uploaded input (the in-place fast path, n < NSPLIT, overwrites it with the result)
static int* g_out = nullptr;    // int32 n x n: result of the dense paths and of the split fast path (n >= NSPLIT)
static int* g_res = nullptr;    // buffer holding the result of the last pb_compute
static u16* g_w = nullptr;      // v1 dense work matrix (np x np, 16-bit); allocated on demand
static cudaStream_t g_stream = nullptr;
static int* g_dctl = nullptr;   // {1, n} control word pair for the single-block closure
static int g_dense_kind = DENSE_V1;
// persistent engine
static u16* g_ework = nullptr;
static int* g_eflags = nullptr;
static int g_engine_grid = 0;
constexpr int EFLAGS_INTS = 64 + 2 * (EMAX_N / EB) * (EMAX_N / EB);
// squaring kernel
static u16* g_sqP = nullptr;
static int* g_sqsync = nullptr;
static bool g_sq_ok = false;            // the cooperative squaring kernel ran once at setup
static int g_sq_cap = 0;                // co-resident blocks
// fast path scratch
static char* g_scratch = nullptr;
static u32 *g_zb, *g_zt, *g_corebits;
static int *g_outdeg, *g_indeg, *g_wprefix, *g_rvert, *g_M, *g_Dm;
static u16* g_mapv;
static FastCtl* g_ctl;
static int g_nslices = 1, g_tps = 1;
static size_t g_scc_smem = 0;
static int g_use_fast = 0;
// status words the kernels write: [0] fast-path result (2 declined, 3 done), [1] watchdog of the engine / squaring kernel. Mapped pinned
// host memory when available (read for free after the final synchronization), otherwise a device word read back with a copy.
static volatile int* g_hst = nullptr;
static int* g_hst_dev = nullptr;
static bool g_hst_mapped = false;
static cudaGraphExec_t g_gfast = nullptr, g_gdense = nullptr, g_gdense_fb = nullptr;
static bool g_f1wide = false;   // fast path: use the wide F1 kernel (n % 4 == 0, n >= NWIDE)
static bool g_split = false;    // fast path: zero-fill of the output runs in a parallel graph branch, k_fixup writes the rest into g_out
static cudaStream_t g_stream2 = nullptr;
static cudaEvent_t g_evf = nullptr, g_evj = nullptr;
constexpr int NSPLIT = 1200;   // split topology (zero-fill overlapped with F2..F4) for n >= NSPLIT: faster once F5 costs more than the fork/join
constexpr int NWIDE = 2048;    // wide F1 for n >= NWIDE
constexpr int ZBLOCKS = 64;    // blocks of the zero-fill: more of them slow down the latency-bound kernels it overlaps with, fewer make it the critical path

static size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
static size_t sq_smem_bytes(int np) { return (size_t)(32 * (((np >> 1) + 31) & ~31) + np * 16 + 16 * 512) * 4; }

static cudaError_t status_reset() {
  if (g_hst_mapped) { g_hst[0] = 0; g_hst[1] = 0; return cudaSuccess; }
  return cudaMemset(g_hst_dev, 0, 2 * sizeof(int));
}
static cudaError_t status_read(int* v) {
  if (g_hst_mapped) { v[0] = g_hst[0]; v[1] = g_hst[1]; return cudaSuccess; }
  return cudaMemcpy(v, g_hst_dev, 2 * sizeof(int), cudaMemcpyDeviceToHost);
}

// cooperative launch: all blocks are co-resident (or the launch fails), which the squaring kernel's grid barrier needs
static cudaError_t launch_sq(cudaStream_t s, int grid, size_t smem, const int* src, int ld_src, int* dst, int ld_dst, int n_static, int* ctl,
                             int n_lo, int n_hi, int ldp, int psz) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(SQ_THREADS);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute at;
  at.id = cudaLaunchAttributeCooperative;
  at.val.cooperative = 1;
  cfg.attrs = &at;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, k_sq, src, ld_src, dst, ld_dst, n_static, ctl, n_lo, n_hi, g_sqP, ldp, psz, g_sqsync, (volatile int*)(g_hst_dev + 1));
}
static cudaError_t sq_rezero() { return cudaMemset(g_sqsync, 0, SQ_SYNC_INTS * sizeof(int)); }

static cudaError_t launch_fast(cudaStream_t s) {
  const int n = g_n, nw = g_nw, nwp = g_nwp;
  if (g_f1wide) {
    const int bands = nw, strips = (nw + 3) / 4, tasks = bands * strips;
    k_bitmap2<<<min((tasks + 7) / 8, 148 * 8), 256, 0, s>>>(g_d, g_zb, g_zt, g_outdeg, g_indeg, n, nw, nwp, strips, tasks);
  } else {
    dim3 g1((n + 31) / 32, g_nslices);
    k_bitmap<<<g1, 256, 0, s>>>(g_d, g_zb, g_zt, g_outdeg, g_indeg, n, nw, nwp, g_tps);
  }
  if (g_split) {   // fork after F1: zero-fill the output buffer while the latency-bound part of the pipeline runs
    cudaError_t e = cudaEventRecord(g_evf, s);
    if (e != cudaSuccess) return e;
    e = cudaStreamWaitEvent(g_stream2, g_evf, 0);
    if (e != cudaSuccess) return e;
    const size_t n4 = ((size_t)n * n) / 4;
    const int zb = (int)min((n4 + 1023) / 1024, (size_t)ZBLOCKS);
    k_zero<<<zb, 256, 0, g_stream2>>>((int4*)g_out, n4, g_out + n4 * 4, (int)((size_t)n * n - n4 * 4));
    e = cudaEventRecord(g_evj, g_stream2);
    if (e != cudaSuccess) return e;
  }
  k_scc2<<<2, SCC_T, g_scc_smem, s>>>(g_zb, g_zt, g_outdeg, g_indeg, n, nw, nwp, (n <= SCC_SMEM_N) ? 1 : 0, g_corebits, g_wprefix, g_rvert, g_mapv, g_ctl, MMAX);
  k_build_m<<<2 * MMAX, 256, 0, s>>>(g_d, g_corebits, g_wprefix, g_rvert, g_ctl, g_M, n, g_outdeg, g_indeg);
  k_small_fw<<<1, SW_THREADS, SW_SMEM, s>>>(g_M, MPITCH, g_Dm, MPITCH, (const int*)g_ctl, MSMALL);
  cudaError_t e = launch_sq(s, min(g_sq_cap, (MMAX / 32) * (MMAX / 32)), sq_smem_bytes(MMAX), g_M, MPITCH, g_Dm, MPITCH, 0, (int*)g_ctl, MSMALL, MMAX, MPITCH, MPITCH * MPITCH);
  if (e != cudaSuccess) return e;
  if (g_split) {   // join
    cudaError_t e = cudaStreamWaitEvent(s, g_evj, 0);
    if (e != cudaSuccess) return e;
    k_fixup<<<dim3(8, MMAX), 256, 0, s>>>(g_Dm, g_mapv, g_rvert, g_ctl, g_out, n, g_hst_dev + 0);
    return cudaGetLastError();
  }
  const int chunks = (n + 3) / 4;
  dim3 g5(min((chunks + 255) / 256, 16), min(n, 65535));
  k_expand<<<g5, 256, 0, s>>>(g_Dm, g_mapv, g_ctl, g_d, n, g_hst_dev + 0);   // result in place (g_res = g_d); does nothing when declined
  return cudaGetLastError();
}

// multi-kernel blocked FW on the 16-bit padded copy: fallback for n > EMAX_N and safety net for the engine
static cudaError_t launch_v1(cudaStream_t s) {
  const int n = g_n;
  size_t words = (size_t)g_np * g_np / 2;
  conv_in<<<(unsigned)((words + 255) / 256), 256, 0, s>>>(g_d, g_w, n, g_np);
  for (int kb = 0; kb < g_N; kb++) {
    k1_diag<<<1, 256, 0, s>>>(g_w, g_np, kb);
    if (g_N > 1) {
      k_tile<1><<<dim3(g_N - 1, 1), 64, 0, s>>>(g_w, g_np, kb);
      k_tile<2><<<dim3(g_N - 1, 1), 64, 0, s>>>(g_w, g_np, kb);
      k_tile<0><<<dim3(g_N - 1, g_N - 1), 64, 0, s>>>(g_w, g_np, kb);
    }
  }
  size_t total = (size_t)n * n;
  conv_out<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(g_w, g_out, n, g_np);
  return cudaGetLastError();
}

static cudaError_t launch_engine(cudaStream_t s) {
  const int n = g_n;
  k_fw_engine<<<g_engine_grid, ETHREADS, 0, s>>>(g_d, n, g_out, n, n, g_ework, g_eflags, g_hst_dev + 1);
  return cudaGetLastError();
}

static cudaError_t launch_dense(cudaStream_t s) {
  const int n = g_n;
  if (g_dense_kind == DENSE_SMALL) {
    k_small_fw<<<1, SW_THREADS, SW_SMEM, s>>>(g_d, n, g_out, n, g_dctl, 1 << 20);
    return cudaGetLastError();
  }
  if (g_dense_kind == DENSE_SQ) {
    return launch_sq(s, min(g_sq_cap, g_N32 * g_N32), sq_smem_bytes(g_np32), g_d, n, g_out, n, n, nullptr, 0, 0, g_np32, g_np32 * g_np32);
  }
  if (g_dense_kind == DENSE_ENGINE) return launch_engine(s);
  return launch_v1(s);
}

static cudaError_t ensure_v1_workspace() {
  if (g_w) return cudaSuccess;
  return cudaMalloc(&g_w, (size_t)g_np * g_np * 2);
}

static cudaError_t capture(cudaGraphExec_t* out, cudaError_t (*fn)(cudaStream_t)) {
  cudaGraph_t g = nullptr;
  cudaError_t e = cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal);
  if (e != cudaSuccess) return e;
  cudaError_t e1 = fn(g_stream);
  cudaError_t e2 = cudaStreamEndCapture(g_stream, &g);
  if (e1 != cudaSuccess) return e1;
  if (e2 != cudaSuccess) return e2;
  e = cudaGraphInstantiate(out, g, 0);
  cudaGraphDestroy(g);
  return e;
}

// run a graph once and wait for it; false on any failure (non-sticky errors are cleared)
static bool probe_graph(cudaGraphExec_t g) {
  cudaError_t e = cudaGraphLaunch(g, g_stream);
  cudaError_t e2 = cudaStreamSynchronize(g_stream);
  if (e == cudaSuccess && e2 == cudaSuccess) return true;
  cudaGetLastError();
  return false;
}

// launch a graph and wait for it; false on any launch / execution error (non-sticky errors are cleared so that the caller can fall back)
static bool run_graph(cudaGraphExec_t g) {
  if (cudaGraphLaunch(g, g_stream) != cudaSuccess) { cudaGetLastError(); return false; }
  if (cudaStreamSynchronize(g_stream) != cudaSuccess) { cudaGetLastError(); return false; }
  return true;
}

// the dense graph failed (launch error, or the squaring / engine watchdog fired): redo the computation with the next safer path.
// The input is untouched (the dense paths write g_out only).
static int dense_fallback(void) {
  int st[2] = {0, 0};
  if (g_dense_kind == DENSE_SQ && g_gdense_fb) {
    CK(sq_rezero());
    CK(status_reset());
    if (run_graph(g_gdense_fb)) {
      CK(status_read(st));
      if (!st[1]) return 0;
    }
  }
  if (g_dense_kind == DENSE_SMALL) return 6;   // nothing simpler to fall back to
  if (g_eflags) CK(cudaMemset(g_eflags, 0, EFLAGS_INTS * sizeof(int)));
  CK(ensure_v1_workspace());
  CK(launch_v1(g_stream));
  CK(cudaStreamSynchronize(g_stream));
  return 0;
}

extern "C" {
int pb_setup(const pb_problem_shape* shape) {
  if (!shape || shape->abi_version != PB_SOLVE_ABI_VERSION || shape->dimension_count != 1 || !shape->dimensions) return 1;
  if (shape->dimensions[0] <= 0 || shape->dimensions[0] > 2147483647LL) return 1;
  g_n = (int)shape->dimensions[0];
  const int n = g_n;
  g_N = (n + TB - 1) / TB;
  g_np = g_N * TB;
  g_N32 = (n + 31) / 32;
  g_np32 = g_N32 * 32;
  g_nw = (n + 31) / 32;
  g_nwp = (g_nw + 3) & ~3;
  CK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  CK(cudaMalloc(&g_d, (size_t)n * n * 4));
  CK(cudaMalloc(&g_out, (size_t)n * n * 4));
  g_res = g_d;
  CK(cudaMalloc(&g_dctl, 2 * sizeof(int)));
  int h[2] = {1, n};
  CK(cudaMemcpy(g_dctl, h, sizeof h, cudaMemcpyHostToDevice));
  CK(cudaFuncSetAttribute(k_small_fw, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SW_SMEM));
  // status words
  {
    void* p = nullptr;
    if (cudaHostAlloc(&p, 64, cudaHostAllocMapped) == cudaSuccess) {
      g_hst = (volatile int*)p;
      if (cudaHostGetDevicePointer((void**)&g_hst_dev, p, 0) == cudaSuccess) g_hst_mapped = true;
      else { cudaGetLastError(); cudaFreeHost(p); g_hst = nullptr; }
    } else {
      cudaGetLastError();
    }
    if (!g_hst_mapped) CK(cudaMalloc(&g_hst_dev, 64));
    CK(status_reset());
  }
  // fast path scratch
  const bool want_fast = (n >= NFAST);
  const bool want_sq_dense = (n > SMALL_DENSE_MAX && n <= SQ_NMAX);
  g_use_fast = want_fast;
  g_f1wide = want_fast && (n % 4 == 0) && n >= NWIDE;
  g_split = want_fast && n >= NSPLIT;
  if (g_split) {
    if (cudaStreamCreateWithFlags(&g_stream2, cudaStreamNonBlocking) != cudaSuccess || cudaEventCreateWithFlags(&g_evf, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&g_evj, cudaEventDisableTiming) != cudaSuccess) {
      cudaGetLastError();
      g_split = false;
    }
  }
  if (g_use_fast) {
    const int nw = g_nw, nwp = g_nwp;
    const int bands = (n + 31) / 32;
    g_nslices = (int)min((long long)nw, max(1LL, (4LL * 148 + bands - 1) / bands));
    g_tps = (nw + g_nslices - 1) / g_nslices;
    size_t off = 0;
    auto take = [&](size_t bytes) { size_t o = off; off = align_up(off + bytes, 256); return o; };
    size_t o_zb = take((size_t)n * nwp * 4), o_zt = take((size_t)n * nwp * 4), o_out = take((size_t)n * 4), o_in = take((size_t)n * 4);
    size_t o_core = take((size_t)nw * 4), o_wp = take((size_t)nw * 4);
    size_t o_rv = take(MMAX * 4), o_map = take((size_t)(n + 8) * 2), o_ctl = take(sizeof(FastCtl));
    size_t o_M = take((size_t)MMAX * MPITCH * 4), o_Dm = take((size_t)MMAX * MPITCH * 4);
    if (cudaMalloc(&g_scratch, off) != cudaSuccess) { cudaGetLastError(); g_scratch = nullptr; g_use_fast = 0; }
    else {
      CK(cudaMemset(g_scratch, 0, off));
      g_zb = (u32*)(g_scratch + o_zb); g_zt = (u32*)(g_scratch + o_zt); g_outdeg = (int*)(g_scratch + o_out); g_indeg = (int*)(g_scratch + o_in);
      g_corebits = (u32*)(g_scratch + o_core); g_wprefix = (int*)(g_scratch + o_wp); g_rvert = (int*)(g_scratch + o_rv);
      g_mapv = (u16*)(g_scratch + o_map); g_ctl = (FastCtl*)(g_scratch + o_ctl); g_M = (int*)(g_scratch + o_M); g_Dm = (int*)(g_scratch + o_Dm);
      g_scc_smem = (size_t)6 * nwp * 4 + ((n <= SCC_SMEM_N) ? ((size_t)n * nwp * 4 + (size_t)n * 4) : 0);
      CK(cudaFuncSetAttribute(k_scc2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g_scc_smem));
    }
  }
  // squaring kernel: dense 64 < n <= SQ_NMAX and the reduced problem of the fast path
  if (want_sq_dense || g_use_fast) {
    int nsm = 0, bpsm = 0;
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    size_t smem = 0, pelems = 0;
    if (want_sq_dense) { smem = max(smem, sq_smem_bytes(g_np32)); pelems = max(pelems, (size_t)2 * g_np32 * g_np32); }
    if (g_use_fast) { smem = max(smem, sq_smem_bytes(MMAX)); pelems = max(pelems, (size_t)2 * MPITCH * MPITCH); }
    bool ok = (cudaFuncSetAttribute(k_sq, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess) &&
              (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, k_sq, SQ_THREADS, smem) == cudaSuccess) && bpsm >= 1 && nsm >= 1 &&
              (cudaMalloc(&g_sqP, pelems * 2) == cudaSuccess) && (cudaMalloc(&g_sqsync, SQ_SYNC_INTS * sizeof(int)) == cudaSuccess);
    if (ok) {
      g_sq_cap = bpsm * nsm;
      ok = (cudaMemset(g_sqsync, 0, SQ_SYNC_INTS * sizeof(int)) == cudaSuccess) && (cudaMemset(g_sqP, 0, pelems * 2) == cudaSuccess);
    }
    if (ok && g_use_fast) {   // run the kernel once for real (zeros): loads the module and warms the caches
      ok = (launch_sq(g_stream, min(g_sq_cap, 9), sq_smem_bytes(MMAX), g_M, MPITCH, g_Dm, MPITCH, 96, nullptr, 0, 0, MPITCH, MPITCH * MPITCH) == cudaSuccess) &&
           (cudaStreamSynchronize(g_stream) == cudaSuccess);
    }
    if (ok && want_sq_dense) {
      ok = (cudaMemset(g_d, 0, (size_t)n * n * 4) == cudaSuccess) &&
           (launch_sq(g_stream, min(g_sq_cap, g_N32 * g_N32), sq_smem_bytes(g_np32), g_d, n, g_out, n, n, nullptr, 0, 0, g_np32, g_np32 * g_np32) == cudaSuccess) &&
           (cudaStreamSynchronize(g_stream) == cudaSuccess);
    }
    cudaGetLastError();
    g_sq_ok = ok;
    if (!ok) {
      if (g_sqP) { cudaFree(g_sqP); g_sqP = nullptr; }
      if (g_sqsync) { cudaFree(g_sqsync); g_sqsync = nullptr; }
      g_use_fast = 0;
    }
  }
  // dense path selection
  if (n <= SMALL_DENSE_MAX) g_dense_kind = DENSE_SMALL;
  else if (n <= SQ_NMAX && g_sq_ok) g_dense_kind = DENSE_SQ;
  else if (n <= EMAX_N) g_dense_kind = DENSE_ENGINE;
  else g_dense_kind = DENSE_V1;
  const bool engine_needed = (g_dense_kind == DENSE_ENGINE) || (g_dense_kind == DENSE_SQ);
  if (engine_needed) {
    int bpsm = 0, nsm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, k_fw_engine, ETHREADS, 0));
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    const int N = (n + EB - 1) / EB;
    CK(cudaMalloc(&g_ework, (size_t)N * N * ETILE * 2));
    CK(cudaMalloc(&g_eflags, EFLAGS_INTS * sizeof(int)));
    CK(cudaMemset(g_eflags, 0, EFLAGS_INTS * sizeof(int)));   // the watchdog block re-zeroes the flags after every run
    g_engine_grid = 1 + max(1, min(max(1, bpsm) * nsm - 1, 2 * N * N));   // + the watchdog block
  }
  if (g_dense_kind == DENSE_V1) CK(ensure_v1_workspace());
  if (g_use_fast) {
    cudaError_t ce = capture(&g_gfast, launch_fast);
    if (ce != cudaSuccess && g_split) {   // the fork/join capture failed: use the single-stream topology
      cudaGetLastError();
      g_split = false;
      ce = capture(&g_gfast, launch_fast);
    }
    if (ce != cudaSuccess) { cudaGetLastError(); g_use_fast = 0; g_gfast = nullptr; }
  }
  {
    cudaError_t ce = capture(&g_gdense, launch_dense);
    if (ce != cudaSuccess && g_dense_kind == DENSE_SQ && n <= EMAX_N) {   // the cooperative launch could not be captured: engine instead
      cudaGetLastError();
      g_dense_kind = DENSE_ENGINE;
      ce = capture(&g_gdense, launch_dense);
    }
    CK(ce);
  }
  if (g_dense_kind == DENSE_SQ) CK(capture(&g_gdense_fb, launch_engine));

  // ---- warm-up on zeros: loads every module and touches all scratch so that the timed calls are steady-state.
  CK(cudaMemset(g_d, 0, (size_t)n * n * 4));
  if (g_use_fast) {
    CK(status_reset());
    if (!probe_graph(g_gfast)) g_use_fast = 0;   // e.g. cluster launch unavailable: fall back to the dense paths
  }
  if (g_dense_kind == DENSE_V1 && n > 4096) {
    // a full dense warm-up would take long at this size: touch each kernel once on a tiny grid instead
    conv_in<<<1, 256, 0, g_stream>>>(g_d, g_w, n, g_np);
    k1_diag<<<1, 256, 0, g_stream>>>(g_w, g_np, 0);
    if (g_N > 1) {
      k_tile<1><<<dim3(1, 1), 64, 0, g_stream>>>(g_w, g_np, 0);
      k_tile<2><<<dim3(1, 1), 64, 0, g_stream>>>(g_w, g_np, 0);
      k_tile<0><<<dim3(1, 1), 64, 0, g_stream>>>(g_w, g_np, 0);
    }
    conv_out<<<1, 256, 0, g_stream>>>(g_w, g_out, n, g_np);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(g_stream));
  } else {
    CK(status_reset());
    int st[2] = {0, 0};
    bool ok = probe_graph(g_gdense);
    if (ok && (g_dense_kind == DENSE_ENGINE || g_dense_kind == DENSE_SQ)) { CK(status_read(st)); if (st[1]) ok = false; }   // watchdog on zeros: broken kernel
    while (!ok) {
      // the squaring kernel / engine could not run: use the next safer path
      if (g_dense_kind == DENSE_SMALL || g_dense_kind == DENSE_V1) return 3;
      g_dense_kind = (g_dense_kind == DENSE_SQ && n <= EMAX_N) ? DENSE_ENGINE : DENSE_V1;
      if (g_dense_kind == DENSE_V1) CK(ensure_v1_workspace());
      cudaGraphExecDestroy(g_gdense); g_gdense = nullptr;
      if (g_gdense_fb) { cudaGraphExecDestroy(g_gdense_fb); g_gdense_fb = nullptr; }
      if (g_sqsync) CK(sq_rezero());
      if (g_eflags) CK(cudaMemset(g_eflags, 0, EFLAGS_INTS * sizeof(int)));
      CK(capture(&g_gdense, launch_dense));
      CK(status_reset());
      ok = probe_graph(g_gdense);
      if (ok && g_dense_kind == DENSE_ENGINE) { CK(status_read(st)); if (st[1]) ok = false; }
    }
  }
  return 0;
}

int pb_upload(const pb_problem* p, pb_buffer* b) {
  if (!p || !b || !b[0].data || p->dimension_count != 1 || !p->dimensions || p->dimensions[0] != g_n) return 1;
  if (b[0].nbytes != (uint64_t)g_n * (uint64_t)g_n * 4) return 1;
  CK(cudaMemcpy(g_d, b[0].data, (size_t)g_n * g_n * 4, cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());   // pageable copies may return before the DMA lands; compute runs in another stream
  return 0;
}

int pb_compute(void) {
  int st[2] = {0, 0};
  CK(status_reset());
  if (g_use_fast) {
    if (run_graph(g_gfast)) {
      CK(status_read(st));
      if (st[0] == 3) { g_res = g_split ? g_out : g_d; return 0; }   // contraction succeeded and wrote the result
      if (st[1]) CK(sq_rezero());                                     // the squaring kernel gave up: its counters may be dirty
    }
    CK(status_reset());   // declined, or the reduced-size solver gave up, or the graph could not run: the input is untouched
  }
  g_res = g_out;
  bool ok = run_graph(g_gdense);
  if (ok && (g_dense_kind == DENSE_SQ || g_dense_kind == DENSE_ENGINE)) {
    CK(status_read(st));
    if (st[1]) ok = false;   // watchdog fired
  }
  if (!ok) return dense_fallback();
  return 0;
}

int pb_download(pb_buffer* b) {
  if (!b || !b[0].data || b[0].nbytes != (uint64_t)g_n * (uint64_t)g_n * 4) return 1;
  CK(cudaMemcpy(b[0].data, g_res, (size_t)g_n * g_n * 4, cudaMemcpyDeviceToHost));
  return 0;
}

int pb_teardown(void) {
  if (g_gfast) cudaGraphExecDestroy(g_gfast);
  if (g_gdense) cudaGraphExecDestroy(g_gdense);
  if (g_gdense_fb) cudaGraphExecDestroy(g_gdense_fb);
  cudaFree(g_d); cudaFree(g_out); cudaFree(g_w); cudaFree(g_scratch); cudaFree(g_dctl);
  cudaFree(g_ework); cudaFree(g_eflags); cudaFree(g_sqP); cudaFree(g_sqsync);
  if (g_hst_mapped) cudaFreeHost((void*)g_hst); else cudaFree(g_hst_dev);
  if (g_stream2) cudaStreamDestroy(g_stream2);
  if (g_evf) cudaEventDestroy(g_evf);
  if (g_evj) cudaEventDestroy(g_evj);
  g_stream2 = nullptr; g_evf = g_evj = nullptr; g_split = false;
  if (g_stream) cudaStreamDestroy(g_stream);
  g_gfast = g_gdense = g_gdense_fb = nullptr; g_d = g_out = g_res = nullptr; g_w = nullptr; g_scratch = nullptr; g_dctl = nullptr;
  g_ework = nullptr; g_eflags = nullptr; g_sqP = nullptr; g_sqsync = nullptr; g_sq_ok = false; g_hst = nullptr; g_hst_dev = nullptr; g_hst_mapped = false;
  g_stream = nullptr; g_use_fast = 0; g_dense_kind = DENSE_V1;
  return 0;
}
}
