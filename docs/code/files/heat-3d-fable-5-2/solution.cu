// heat-3d style 7-point stencil, FP32, bit-identical to the CPU baseline.
//
// The computation is 2*n0 Jacobi sweeps (A->B, B->A, ...) over the interior of an n1^3 grid whose six faces
// stay fixed (A's faces keep A's initial values, B's faces B's).  It is run here as "passes" of S sweeps
// (S = 4, or 2).  A pass reads the field of sweep level s (complete, in one array) and writes the field of
// level s+S into another array; the S-1 levels in between only ever live in registers.  Every sweep level
// of every cell is still evaluated with the baseline's operation order and rounding points, so the result
// is the baseline's, bit for bit (see stencil() for the one exception: subnormal intermediate products).
// The last pass also writes level 2*n0-1, which is the final content of buf1.
//
// Layout of one pass (pass_kernel):
//   * a thread block is one tile: one warp wide in k (32 lanes of W columns each), up to 16 thread rows of
//     H rows each in j, and a chunk of planes in i; tiles overlap by S cells per side (overlapped tiling)
//     and only the core of a tile is written;
//   * i is streamed: a thread walks the planes of its chunk and keeps, for every level, the three planes
//     that level needs in registers (the sliding window is rotated by unrolling three steps);
//   * a thread owns H rows x W columns of cells, so most j and k neighbours are its own registers; the k
//     neighbours at the ends of its W columns are the neighbouring lanes (warp shuffles); the j neighbours
//     at the ends of its H rows go through a small double-buffered shared-memory ring, one barrier per plane;
//   * fixed (face) cells take the value they had two levels earlier (it alternates between A's and B's face
//     value); their level-1 value is B's face value: it is fetched from the B field one plane ahead of its
//     use, or (k faces) streamed through level 0 by a "ghost lane" next to the face.
// pb_setup times a handful of tile configurations (and the plain one-sweep kernel) on scratch data and keeps
// the fastest plan for this shape.
#include "interface.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <chrono>
#include <vector>

namespace {

constexpr int MAXBY = 16;                // max thread rows per block (16 * 32 = 512 threads)
constexpr size_t FRONT = 64;             // unused cells in front of every device array

struct Pass {
  const float *in;    // level-0 field (complete, including its boundary)
  const float *bnd;   // field whose boundary cells hold the other-parity boundary values
  float *out;         // level-S field (complete)
  float *out2;        // (DUAL) level S-1 field: the same cells as out
  int N;
  int ntk, ntj, nti;  // tiles per dimension
};

// One sweep of one cell, with the baseline's rounding sequence:
//   0.125f*(ip - 2c + im) + 0.125f*(jp - 2c + jm) + 0.125f*(kp - 2c + km) + c
// 2c is exact, so fma(-2,c,x) rounds exactly where x - 2c does. Scaling by 0.125 is exact too unless the
// product is subnormal, so 0.125*x1 + 0.125*x2 + 0.125*x3 + c rounds exactly where fma(0.125, x1+x2+x3, c)
// does. Only when a 0.125*x product is subnormal (inputs of magnitude around 1e-38 and below) can the last
// bit differ from the baseline; the single rounding here is then the more accurate one.
__device__ __forceinline__ float stencil(float im, float c, float ip, float jm, float jp, float km, float kp) {
  float x1 = fmaf(-2.0f, c, ip) + im;
  float x2 = fmaf(-2.0f, c, jp) + jm;
  float x3 = fmaf(-2.0f, c, kp) + km;
  return fmaf(0.125f, (x1 + x2) + x3, c);
}

__device__ __forceinline__ int tsplit(int m, int nt, int N) { return (int)(((unsigned)m * (unsigned)N) / (unsigned)nt); }

template <int S, int C> struct Regs { float w[3][S][C]; };

// One plane step. A thread owns H rows (j) x W columns (k) of cells; cell (h, w) has index h*W + w.
// Roles for each level t: A = w[R] (plane p-1), B = w[R+1] (plane p), C = w[R+2] (plane p+1), where
// p = n-t-1 is the plane whose level t+1 values are produced now.
// GEN == false: the only fixed (boundary) cells this warp may own are the row j == 0 as its cells h == 0
//               (flag ft) and the row j == N-1 as its cells h == H-1 (flag fb) -- only if HOOKS -- and no
//               plane touched by this step is plane 0 or N-1.
// GEN == true : general: fixed cells by mask (bm), and planes 0 / N-1 (all cells fixed).
template <int S, int H, int W, int R, bool GEN, bool DUAL, bool HOOKS>
__device__ __forceinline__ void plane_step(Regs<S, H * W> &g, float *smem, const int sidx, const int n, const int N,
                                           const float *pin, const ptrdiff_t nn, const ptrdiff_t dBnd, const ptrdiff_t dOut,
                                           const ptrdiff_t dOut2, const unsigned bm, const unsigned sm,
                                           const bool ft, const bool fb,
                                           const bool fl, const bool fr, const unsigned cfm, const bool ghost,
                                           const int iv0, const int iv1, const int if1) {
  constexpr int RA = R % 3, RB = (R + 1) % 3, RC = (R + 2) % 3;
  constexpr int NC = H * W;
  constexpr unsigned ALLM = (NC >= 32) ? 0xffffffffu : ((1u << NC) - 1u);
  constexpr int L1 = 1 % S;                       // level-1 register row
  constexpr bool HK = HOOKS && !GEN;
  constexpr int KP = 4 * S * W + 1;               // floats per (thread row, lane) in the shared ring (odd: no bank conflicts)
  // shared ring: [thread row + 1][lane][set][level][top|bot][w]
  float *const wrs = smem + sidx + ((n & 1) ? 2 * S * W : 0);
  const float *const rds = smem + sidx + ((n & 1) ? 0 : 2 * S * W);
  if (HK) {
    // A fixed row's level S-1 oldest-plane registers are never needed (its stencil results are discarded and
    // nothing takes its boundary value from that level): fetch the BND values of plane n, which is level 1's
    // newest plane in the next step, into them now. They are handed to the level-1 registers at the end of
    // the step.
#pragma unroll
    for (int w = 0; w < W; ++w) {
      if (ft) g.w[RA][S - 1][w] = (pin + dBnd)[w];
      if (fb) g.w[RA][S - 1][(H - 1) * W + w] = (pin + dBnd)[(H - 1) * N + w];
    }
  }
#pragma unroll
  for (int t = 0; t < S; ++t) {
    const int p = n - t - 1;
    float *const wr = wrs + 2 * W * t;
    const float *const rd = rds + 2 * W * t;
    float up[W], dn[W];
#pragma unroll
    for (int w = 0; w < W; ++w) {
      wr[w] = g.w[RC][t][w];                         // top row of this thread, plane p+1
      wr[W + w] = g.w[RC][t][(H - 1) * W + w];       // bottom row
      up[w] = rd[W + w - 32 * KP];                   // bottom row of the thread row above, plane p
      dn[w] = rd[w + 32 * KP];                       // top row of the thread row below, plane p
    }
    float nw[NC];
#pragma unroll
    for (int h = 0; h < H; ++h) {
      const float lft = __shfl_up_sync(0xffffffffu, g.w[RB][t][h * W + W - 1], 1);   // k-1 neighbour of cell (h, 0)
      const float rgt = __shfl_down_sync(0xffffffffu, g.w[RB][t][h * W], 1);         // k+1 neighbour of cell (h, W-1)
#pragma unroll
      for (int w = 0; w < W; ++w) {
        const int c = h * W + w;
        const float km = w > 0 ? g.w[RB][t][c - 1] : lft;
        const float kp = w < W - 1 ? g.w[RB][t][c + 1] : rgt;
        const float jm = h > 0 ? g.w[RB][t][c - W] : up[w];
        const float jp = h < H - 1 ? g.w[RB][t][c + W] : dn[w];
        nw[c] = stencil(g.w[RA][t][c], g.w[RB][t][c], g.w[RC][t][c], jm, jp, km, kp);
        if (GEN && t == 0) {
          // A fixed column next to a ghost lane: the ghost lane streams the BND column through level 0, so
          // its plane-p value is this cell's level-1 value.
          if (w == 0 && fl) nw[c] = km;
          if (w == W - 1 && fr) nw[c] = kp;
        }
      }
    }
    // Fixed boundary cells. Level t+1 carries the boundary values of level t-1. For t >= 1 this thread still
    // holds them as the oldest plane of level t-1 (plane p). For t == 0 they come from BND and were fetched
    // during the previous step straight into this level-1 register, so no load is waited for here.
    if (HK) {
#pragma unroll
      for (int w = 0; w < W; ++w) {
        if (ft) nw[w] = (t == 0) ? g.w[RC][L1][w] : g.w[RA][t - 1][w];
        if (fb) nw[(H - 1) * W + w] = (t == 0) ? g.w[RC][L1][(H - 1) * W + w] : g.w[RA][t - 1][(H - 1) * W + w];
      }
    }
    if (GEN) {
      const unsigned m = ((p == 0) | (p == N - 1)) ? ALLM : bm;
      if (m != 0) {
        if (t == 0) {
          const unsigned m0 = m & ~cfm;               // cells in cfm took the ghost lane's value above
#pragma unroll
          for (int c = 0; c < NC; ++c) if (m0 & (1u << c)) nw[c] = g.w[RC][L1][c];
        } else {
#pragma unroll
          for (int c = 0; c < NC; ++c) if (m & (1u << c)) nw[c] = g.w[RA][t - 1][c];
        }
      }
      if (t == (S > 2 ? 2 : S - 1)) {
        // Level 1's oldest plane is dead now: fetch the BND values of plane n for the cells that are fixed
        // there into it; it is level 1's newest plane in the next step.
        const unsigned mp = (((n == 0) | (n == N - 1)) ? ALLM : bm) & ~cfm;
        if (mp != 0 && n <= N - 1) {
          const float *src = ghost ? pin : pin + dBnd;   // a ghost lane's pointer is already in BND
#pragma unroll
          for (int h = 0; h < H; ++h)
#pragma unroll
            for (int w = 0; w < W; ++w) if (mp & (1u << (h * W + w))) g.w[RA][L1][h * W + w] = src[h * N + w];
        }
      }
    }
    if (t == 1 && n + 1 < if1) {
      // level 0 (and the fix-up of level 1) no longer need plane n-2 of level 0: fetch plane n+1 into its
      // registers now, so that the global-load latency is hidden behind the remaining levels
#pragma unroll
      for (int h = 0; h < H; ++h)
#pragma unroll
        for (int w = 0; w < W; ++w) g.w[RA][0][h * W + w] = pin[nn + h * N + w];
    }
    if (t == S - 1) {
      // Outputs. The level-S values of plane p go to OUT; in the last pass (DUAL) the level S-1 values of
      // plane p+1 (the newest plane of level S-1) also go to OUT2, for the same cells: the fixed cells of
      // OUT2 are thereby rewritten with the values they already hold (that is where they came from).
      if (DUAL) {
        if (p + 1 >= iv0 && p + 1 < iv1) {
          float *o2 = const_cast<float *>(pin) + dOut2 - (ptrdiff_t)(S - 1) * nn;
#pragma unroll
          for (int h = 0; h < H; ++h)
#pragma unroll
            for (int w = 0; w < W; ++w) if (sm & (1u << (h * W + w))) o2[h * N + w] = g.w[RC][t][h * W + w];
        }
      }
      if (p >= iv0 && p < iv1) {
        float *o = const_cast<float *>(pin) + dOut - (ptrdiff_t)S * nn;
#pragma unroll
        for (int h = 0; h < H; ++h)
#pragma unroll
          for (int w = 0; w < W; ++w) if (sm & (1u << (h * W + w))) o[h * N + w] = nw[h * W + w];
      }
    } else {
#pragma unroll
      for (int c = 0; c < NC; ++c) g.w[RC][t + 1][c] = nw[c];
    }
  }
  if (HK && S > 2) {
#pragma unroll
    for (int w = 0; w < W; ++w) {
      if (ft) g.w[RA][L1][w] = g.w[RA][S - 1][w];
      if (fb) g.w[RA][L1][(H - 1) * W + w] = g.w[RA][S - 1][(H - 1) * W + w];
    }
  }
}

template <int S, int H, int W, bool DUAL, bool HOOKS>
__global__ void __launch_bounds__(MAXBY * 32, 1) pass_kernel(Pass P) {
  extern __shared__ float smem[];
  constexpr int KP = 4 * S * W + 1;
  const int N = P.N;
  const int bx = blockIdx.x, bj = blockIdx.y, bi = blockIdx.z;
  const int kv0 = tsplit(bx, P.ntk, N), kv1 = tsplit(bx + 1, P.ntk, N);
  const int jv0 = tsplit(bj, P.ntj, N), jv1 = tsplit(bj + 1, P.ntj, N);
  const int iv0 = tsplit(bi, P.nti, N), iv1 = tsplit(bi + 1, P.nti, N);
  const int kf0 = max(0, kv0 - S), kf1 = min(N, kv1 + S);
  const int jf0 = max(0, jv0 - S), jf1 = min(N, jv1 + S);
  const int if0 = max(0, iv0 - S), if1 = min(N, iv1 + S);
  const int lane = threadIdx.x, ty = threadIdx.y, by = blockDim.y;
  const bool kleft = kf0 == 0, kright = kf1 == N;
  const bool jtop = jf0 == 0, jbot = jf1 == N;
  // Rows of a tile start at jf0, except (HOOKS) in a tile that ends at the bottom face and not at the top one:
  // its rows are aligned to the end, so that the fixed row j == N-1 is row H-1 of the last thread row.
  const bool balign = HOOKS & jbot & !jtop & (N - H * by >= 0);
  const int jbase = balign ? N - H * by : jf0;
  // Columns. A tile that touches a k face gets a "ghost lane" outside that face: a lane that streams the
  // BND values of the face column through level 0 (its pointer is in BND), so the fixed column next to it
  // gets its level-1 values by shuffle instead of from memory. Left face: lane 0 is the ghost (its last
  // cell is column 0 of BND) and the real columns start at lane 1. Right face: the columns are aligned so
  // that column N-1 is the last cell of a lane, and the next lane is the ghost (its first cell is column
  // N-1 of BND). If that does not fit into the warp, the tile has no ghost lanes and its fixed columns are
  // treated like any other fixed cell.
  const int kal = (kright & !kleft) ? ((N - kf0) % W == 0 ? 0 : W - (N - kf0) % W) : 0;   // extra halo columns
  const bool gok = (kleft | kright) & !(kleft & kright & (N % W != 0)) &
                   ((kf1 - kf0) + kal + (kleft ? W : 0) + (kright ? W : 0) <= 32 * W);
  const int kfs = gok ? kf0 - kal - (kleft ? W : 0) : kf0;      // (virtual) column of lane 0, cell 0
  const int kr0 = kfs + W * lane;
  const bool ghostl = gok & kleft & (lane == 0), ghostr = gok & kright & (kr0 == N);
  const bool ghost = ghostl | ghostr;
  const bool fl = gok & kleft & (lane == 1), fr = gok & kright & (kr0 + W == N);
  const int jr0 = jbase + H * ty;
  unsigned bm = 0, sm = 0, cfm = 0;
#pragma unroll
  for (int h = 0; h < H; ++h) {
#pragma unroll
    for (int w = 0; w < W; ++w) {
      const int jr = jr0 + h, kr = kr0 + w;
      const bool act = (kr >= kf0) & (kr < kf1) & (jr >= jf0) & (jr < jf1);
      const bool bc = act & ((kr == 0) | (kr == N - 1) | (jr == 0) | (jr == N - 1));
      const bool sc = act & (kr >= kv0) & (kr < kv1) & (jr >= jv0) & (jr < jv1);
      bm |= (unsigned)bc << (h * W + w);
      sm |= (unsigned)sc << (h * W + w);
      cfm |= (unsigned)(bc & ((fl & (w == 0)) | (fr & (w == W - 1)))) << (h * W + w);
    }
  }
  // Warp-uniform treatment of fixed cells (every lane of a warp has the same thread row): a warp whose only
  // fixed cells are a whole row at h == 0 or h == H-1 stays on the fast path (HOOKS, tiles that touch no k
  // face and whose rows line up with the faces); any other warp with fixed cells takes the general path.
  const bool simple = HOOKS & !(kleft | kright) & (!jbot | balign | (jtop & (N % H == 0)));
  const bool ft = simple & jtop & (ty == 0);                 // row j == 0 is this thread row's row 0
  const bool fb = simple & jbot & (jr0 + H == N);            // row j == N-1 is this thread row's row H-1
  const bool wgen = !simple && __any_sync(0xffffffffu, bm != 0);
  const ptrdiff_t nn = (ptrdiff_t)N * N;
  // (plane n, row jr0, column kr0); a ghost lane points at the face column of BND instead
  const float *pin = ghostl ? P.bnd + ((ptrdiff_t)if0 * N + jr0) * N - (W - 1)
                   : ghostr ? P.bnd + ((ptrdiff_t)if0 * N + jr0) * N + (N - 1)
                            : P.in + ((ptrdiff_t)if0 * N + jr0) * N + kr0;
  const ptrdiff_t dBnd = P.bnd - P.in, dOut = P.out - P.in, dOut2 = DUAL ? P.out2 - P.in : 0;
  const int sidx = ((ty + 1) * 32 + lane) * KP;
  Regs<S, H * W> g;
#pragma unroll
  for (int r = 0; r < 3; ++r)
#pragma unroll
    for (int t = 0; t < S; ++t)
#pragma unroll
      for (int c = 0; c < H * W; ++c) g.w[r][t][c] = 0.0f;
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int w = 0; w < W; ++w) g.w[2][0][h * W + w] = pin[h * N + w];
  const int nEnd = iv1 - 1 + S;
  int n = if0;
#define PB_STEP(R)                                                                                                     \
  {                                                                                                                    \
    if ((n <= S) | (n >= N - 1) | wgen)                                                                                \
      plane_step<S, H, W, R, true, DUAL, HOOKS>(g, smem, sidx, n, N, pin, nn, dBnd, dOut, dOut2, bm, sm, ft, fb,       \
                                                fl, fr, cfm, ghost, iv0, iv1, if1);                                    \
    else                                                                                                               \
      plane_step<S, H, W, R, false, DUAL, HOOKS>(g, smem, sidx, n, N, pin, nn, dBnd, dOut, dOut2, bm, sm, ft, fb,      \
                                                 fl, fr, cfm, ghost, iv0, iv1, if1);                                   \
    ++n;                                                                                                               \
    if (n > nEnd) break;                                                                                               \
    pin += nn;                                                                                                         \
    __syncthreads();                                                                                                   \
  }
  while (true) {
    PB_STEP(0)
    PB_STEP(1)
    PB_STEP(2)
  }
#undef PB_STEP
}

// Plain one-sweep kernel (one thread per cell); used when it is the fastest choice (very small grids).
__global__ void sweep_kernel(const float *__restrict__ in, float *__restrict__ out, int n) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int i = blockIdx.z + 1;
  if (k >= n - 1 || j >= n - 1) return;
  const size_t nn = (size_t)n * n;
  const size_t idx = (size_t)i * nn + (size_t)j * n + k;
  out[idx] = stencil(in[idx - nn], in[idx], in[idx + nn], in[idx - n], in[idx + n], in[idx - 1], in[idx + 1]);
}

// ------------------------------------------------------------------------------------------------- host side

// Kernel families: fam 0: a thread owns 8 rows x 1 column, fam 1: 4 rows x 2 columns.
constexpr int NFAM = 2;
constexpr int FAM_H[NFAM] = {8, 4};
constexpr int FAM_W[NFAM] = {1, 2};

struct Cfg { int fam, ntk, ntj, nti, by, hooks; };

struct State {
  bool ready = false;
  int n0 = 0, N = 0;
  size_t cells = 0, bytes = 0;
  float *base[3] = {nullptr, nullptr, nullptr};   // allocations
  float *x[2] = {nullptr, nullptr};               // A-parity fields (ping-pong); x[cur] holds buf0
  float *b = nullptr;                             // B field (buf1)
  int cur = 0;
  int nsm = 1;
  int mode = 0;                                   // 0: plain sweeps, 1: S=2 passes, 2: S=4 passes (+ one S=2)
  Cfg cfg2{}, cfg4{};
  // plain-sweep plan: the launches of gsteps (grem) time steps recorded as CUDA graphs, replayed by pb_compute
  cudaGraphExec_t gexec = nullptr, grexec = nullptr;
  int gsteps = 0, grem = 0;
} G;

int max_footprint(int N, int nt, int S) {
  int m = 0;
  for (int t = 0; t < nt; ++t) {
    const int v0 = (int)(((long long)t * N) / nt), v1 = (int)(((long long)(t + 1) * N) / nt);
    m = std::max(m, std::min(N, v1 + S) - std::max(0, v0 - S));
  }
  return m;
}

// Smallest number of tiles whose footprints (core + halos) fit in `limit` cells; 0 if there is none.
int min_tiles(int N, int S, int limit) {
  for (int nt = 1; nt <= N; ++nt)
    if (max_footprint(N, nt, S) <= limit) return nt;
  return 0;
}

template <int S, int H, int W, bool DUAL, bool HOOKS>
cudaError_t launch_t(const Cfg &c, const float *in, const float *bnd, float *out, float *out2, int N, bool one) {
  Pass P;
  P.in = in; P.bnd = bnd; P.out = out; P.out2 = out2; P.N = N; P.ntk = c.ntk; P.ntj = c.ntj; P.nti = c.nti;
  dim3 grid(c.ntk, c.ntj, c.nti);
  if (one) grid = dim3(1, 1, 1);                 // (kernel warm-up) only the first tile
  const dim3 block(32, c.by, 1);
  const size_t shared = (size_t)(c.by + 2) * 32 * (4 * S * W + 1) * sizeof(float);
  pass_kernel<S, H, W, DUAL, HOOKS><<<grid, block, shared>>>(P);
  return cudaGetLastError();
}

template <int S, int H, int W>
cudaError_t launch_f(bool dual, const Cfg &c, const float *in, const float *bnd, float *out, float *out2, int N, bool one) {
  if (c.hooks) return dual ? launch_t<S, H, W, true, true>(c, in, bnd, out, out2, N, one) : launch_t<S, H, W, false, true>(c, in, bnd, out, out2, N, one);
  return dual ? launch_t<S, H, W, true, false>(c, in, bnd, out, out2, N, one) : launch_t<S, H, W, false, false>(c, in, bnd, out, out2, N, one);
}

// One pass of S sweeps: in (level s) -> out (level s+S); dual: also level s+S-1 -> out2.
cudaError_t launch_pass(int S, bool dual, const Cfg &c, const float *in, const float *bnd, float *out, float *out2, int N,
                        bool one = false) {
  if (S == 4) return c.fam == 0 ? launch_f<4, 8, 1>(dual, c, in, bnd, out, out2, N, one) : launch_f<4, 4, 2>(dual, c, in, bnd, out, out2, N, one);
  return c.fam == 0 ? launch_f<2, 8, 1>(dual, c, in, bnd, out, out2, N, one) : launch_f<2, 4, 2>(dual, c, in, bnd, out, out2, N, one);
}

cudaError_t launch_sweep(const float *in, float *out, int N) {
  const dim3 block(32, 8, 1);
  const dim3 grid((N - 2 + 31) / 32, (N - 2 + 7) / 8, N - 2);
  sweep_kernel<<<grid, block>>>(in, out, N);
  return cudaGetLastError();
}

// A CUDA graph of `steps` time steps of plain sweeps (a -> b, b -> a, in order); nullptr if it cannot be built.
// Replaying it costs less per kernel than launching the kernels one by one.
cudaGraphExec_t build_sweep_graph(float *a, float *b, int N, int steps) {
  cudaGraph_t graph;
  if (cudaGraphCreate(&graph, 0) != cudaSuccess) { cudaGetLastError(); return nullptr; }
  const dim3 block(32, 8, 1);
  const dim3 grid((N - 2 + 31) / 32, (N - 2 + 7) / 8, N - 2);
  cudaGraphNode_t prev{}, node{};
  bool ok = true;
  for (int s = 0; s < 2 * steps && ok; ++s) {
    const float *in = (s & 1) ? b : a;
    float *out = (s & 1) ? a : b;
    int n = N;
    void *args[3] = {(void *)&in, (void *)&out, (void *)&n};
    cudaKernelNodeParams kp = {};
    kp.func = (void *)sweep_kernel;
    kp.gridDim = grid;
    kp.blockDim = block;
    kp.sharedMemBytes = 0;
    kp.kernelParams = args;
    kp.extra = nullptr;
    ok = cudaGraphAddKernelNode(&node, graph, s ? &prev : nullptr, s ? 1 : 0, &kp) == cudaSuccess;
    prev = node;
  }
  cudaGraphExec_t exec = nullptr;
  if (ok) ok = cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
  cudaGraphDestroy(graph);
  if (!ok) { cudaGetLastError(); return nullptr; }
  return exec;
}

// n0 time steps of plain sweeps: x[cur] -> b -> x[cur] ..., exactly the baseline's two sweeps per time step.
cudaError_t run_plain(void) {
  cudaError_t e = cudaSuccess;
  long long steps = G.n0;
  if (G.gexec && G.cur == 0) {                 // (the graphs were recorded for x[0]; this plan never moves buf0)
    for (; steps >= G.gsteps && e == cudaSuccess; steps -= G.gsteps) e = cudaGraphLaunch(G.gexec, 0);
    if (steps > 0 && steps == G.grem && G.grexec && e == cudaSuccess) { e = cudaGraphLaunch(G.grexec, 0); steps = 0; }
  }
  for (; steps > 0 && e == cudaSuccess; --steps) {
    e = launch_sweep(G.x[G.cur], G.b, G.N);
    if (e == cudaSuccess) e = launch_sweep(G.b, G.x[G.cur], G.N);
  }
  return e;
}

template <int S, int H, int W> cudaError_t set_shared() {
  const int bytes = (int)((MAXBY + 2) * 32 * (4 * S * W + 1) * sizeof(float));
  cudaError_t e = cudaFuncSetAttribute(pass_kernel<S, H, W, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(pass_kernel<S, H, W, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(pass_kernel<S, H, W, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(pass_kernel<S, H, W, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  return e;
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Time one pass of S sweeps (S == 1: one plain sweep) with configuration c on the scratch contents of the
// device arrays. Short passes are launched in batches so that the interval is long enough to be measured.
// Returns seconds per pass, < 0 on failure.
double time_pass(int S, const Cfg &c, int reps) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    const double t0 = now_s();
    int launches = 0;
    double t = 0.0;
    do {
      const int batch = launches == 0 ? 1 : launches;
      for (int q = 0; q < batch; ++q) {
        const int a = (launches + q) & 1;
        const cudaError_t e = (S == 1) ? launch_sweep(G.x[a], G.x[a ^ 1], G.N)
                                       : launch_pass(S, false, c, G.x[a], G.b, G.x[a ^ 1], nullptr, G.N);
        if (e != cudaSuccess) return -1.0;
      }
      launches += batch;
      if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
      t = now_s() - t0;
    } while (t < 400e-6 && launches < 128);
    best = std::min(best, t / launches);
  }
  return best;
}

// The configuration of family fam with tiles of at most bymax thread rows; ntj == 0 if such tiles cannot
// hold their halos.
Cfg base_cfg(int N, int S, int fam, int bymax, int nti, int hooks) {
  Cfg c;
  c.fam = fam;
  c.ntk = min_tiles(N, S, 32 * FAM_W[fam]);
  c.ntj = min_tiles(N, S, FAM_H[fam] * bymax);
  c.by = c.ntj ? (max_footprint(N, c.ntj, S) + FAM_H[fam] - 1) / FAM_H[fam] : 0;
  c.nti = std::max(1, std::min(nti, N));
  c.hooks = hooks;
  return c;
}

// Choose the tile configuration for passes of S sweeps by timing candidates on scratch data: kernel family,
// tile height (thread rows per block), number of chunks along i, row hooks on/off. Returns the time of one
// pass with the chosen configuration (< 0 on failure).
double tune(int S, Cfg &best) {
  const int N = G.N;
  const double tstart = now_s();
  double budget = 2.0, tbest = 1e30;
  bool any = false;
  for (int fam = 0; fam < NFAM; ++fam) {
    std::vector<Cfg> shapes;                       // distinct (ntj, by)
    for (int bymax : {16, 12, 8, 6, 4, 3, 2}) {
      const Cfg c = base_cfg(N, S, fam, bymax, 1, 1);
      if (c.ntj == 0) continue;
      bool dup = false;
      for (const Cfg &s : shapes) dup |= (s.by == c.by && s.ntj == c.ntj);
      if (!dup) shapes.push_back(c);
    }
    double tfam = 1e30;
    for (size_t si = 0; si < shapes.size(); ++si) {
      Cfg c = shapes[si];
      const int cols = c.ntk * c.ntj;
      // candidate chunk counts: a few small ones, and block counts near multiples of what the device holds
      std::vector<int> ntis = {1, 2, 3, 4, 6, 8};
      const int per_sm = std::max(1, MAXBY / c.by);                 // blocks of this size that fit on one SM
      for (int cmul = 1; cmul <= 6; ++cmul) {
        const long long target = (long long)G.nsm * per_sm * cmul;
        ntis.push_back((int)(target / cols));
        ntis.push_back((int)((target + cols - 1) / cols));
      }
      std::sort(ntis.begin(), ntis.end());
      ntis.erase(std::unique(ntis.begin(), ntis.end()), ntis.end());
      double tshape = 1e30;
      bool out_of_time = false;
      for (int nti : ntis) {
        if (nti < 1 || nti > std::max(1, N / 4)) continue;
        c.nti = nti;
        const double t = time_pass(S, c, 2);
        if (t < 0) return -1.0;
        if (!any) budget = std::min(2.0, std::max(0.25, 80.0 * t));   // spend about 80 passes per S, within limits
        any = true;
        tshape = std::min(tshape, t);
        tfam = std::min(tfam, t);
        if (t < tbest) { tbest = t; best = c; }
        if (now_s() - tstart > budget) { out_of_time = true; break; }
      }
      if (out_of_time) break;
      if (si > 0 && tshape > 1.15 * tfam) break;     // smaller blocks are getting worse: stop
    }
  }
  if (!any) return -1.0;
  // row hooks on/off for the chosen shape, and a steadier estimate of the pass time
  Cfg alt = best;
  alt.hooks ^= 1;
  const int reps = tbest < 2e-3 ? 5 : 2;
  const double t0 = time_pass(S, best, reps), t1 = time_pass(S, alt, reps);
  if (t0 < 0 || t1 < 0) return -1.0;
  if (t1 < t0) { best = alt; return t1; }
  return t0;
}

} // namespace

extern "C" int pb_setup(const pb_problem_shape *shape) {
  if (!shape || shape->abi_version != PB_SOLVE_ABI_VERSION || shape->dimension_count != 2 || shape->scalar_count != 0 ||
      shape->buffer_count != 2 || !shape->dimensions || !shape->buffers) return 1;
  const int64_t n0 = shape->dimensions[0], n1 = shape->dimensions[1];
  if (n0 <= 0 || n0 > INT32_MAX || n1 <= 0 || n1 > 2048) return 1;   // 2 * n1^3 * 4 bytes <= 64 GiB
  for (int q = 0; q < 2; ++q) {
    const pb_buffer_shape &bs = shape->buffers[q];
    if (bs.dtype != PB_F32 || bs.rank != 3 || bs.shape[0] != n1 || bs.shape[1] != n1 || bs.shape[2] != n1 ||
        bs.nbytes != (uint64_t)n1 * n1 * n1 * sizeof(float)) return 1;
  }
  if (G.ready) return 1;
  G = State();
  G.n0 = (int)n0; G.N = (int)n1;
  G.cells = (size_t)n1 * n1 * n1;
  G.bytes = G.cells * sizeof(float);
  int dev = 0;
  cudaDeviceProp prop;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 1;
  G.nsm = std::max(1, prop.multiProcessorCount);
  // Tiles are processed with whole warps and whole threads, so loads may run past the last cell of the last
  // plane (never stores): give every array that much slack.
  // (a ghost lane of two-column threads also loads the cell before its column: FRONT cells in front)
  const size_t slack = (size_t)(8 * MAXBY + 2) * G.N + 512;
  const size_t alloc = (FRONT + G.cells + slack) * sizeof(float);
  for (int q = 0; q < 3; ++q) {
    if (cudaMalloc(&G.base[q], alloc) != cudaSuccess) { pb_teardown(); return 1; }
    if (cudaMemset(G.base[q], 0, alloc) != cudaSuccess) { pb_teardown(); return 1; }
  }
  G.x[0] = G.base[0] + FRONT; G.x[1] = G.base[1] + FRONT; G.b = G.base[2] + FRONT;
  G.cur = 0;
  if (set_shared<2, 8, 1>() != cudaSuccess || set_shared<4, 8, 1>() != cudaSuccess ||
      set_shared<2, 4, 2>() != cudaSuccess || set_shared<4, 4, 2>() != cudaSuccess) { pb_teardown(); return 1; }
  G.mode = 0;
  if (G.N >= 3) {
    const int N = G.N;
    // load every kernel once (one block each), on scratch data
    for (int fam = 0; fam < NFAM; ++fam)
      for (int hooks = 0; hooks < 2; ++hooks)
        for (int S = 2; S <= 4; S += 2)
          for (int dual = 0; dual < 2; ++dual)
            if (launch_pass(S, dual != 0, base_cfg(N, S, fam, MAXBY, 1, hooks), G.x[0], G.b, G.x[1], dual ? G.b : nullptr, N, true) != cudaSuccess) {
              pb_teardown();
              return 1;
            }
    if (launch_sweep(G.x[0], G.x[1], N) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) { pb_teardown(); return 1; }
    G.cfg2 = base_cfg(N, 2, 0, MAXBY, 1, 1);
    G.cfg4 = base_cfg(N, 4, 0, MAXBY, 1, 1);
    // Choose the plan: time one pass of each kind and compare the totals for 2*n0 sweeps.
    double t1 = time_pass(1, G.cfg2, 5);                    // seconds per sweep, launched one by one
    // When a sweep is so short that launching it is a visible part of its cost, the plain-sweep plan replays
    // CUDA graphs of up to 256 time steps (plus one graph for the remainder of n0) instead.
    if (t1 < 25e-6) {
      G.gsteps = (int)std::min<long long>(G.n0, 256);
      G.grem = (int)(G.n0 % G.gsteps);
      G.gexec = build_sweep_graph(G.x[0], G.b, N, G.gsteps);
      G.grexec = (G.gexec && G.grem) ? build_sweep_graph(G.x[0], G.b, N, G.grem) : nullptr;
    }
    if (G.gexec) {
      double tg = 1e30;
      for (int r = 0; r < 3; ++r) {
        if (cudaDeviceSynchronize() != cudaSuccess) { pb_teardown(); return 1; }
        const double t0 = now_s();
        int launches = 0;
        double t = 0.0;
        do {
          if (cudaGraphLaunch(G.gexec, 0) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) { pb_teardown(); return 1; }
          ++launches;
          t = now_s() - t0;
        } while (t < 400e-6 && launches < 64);
        tg = std::min(tg, t / ((double)launches * 2.0 * G.gsteps));
      }
      t1 = std::min(t1, tg);
    }
    const double t2 = tune(2, G.cfg2);
    const double t4 = tune(4, G.cfg4);
    if (t1 < 0 || t2 < 0 || t4 < 0) { pb_teardown(); return 1; }
    const long long M = 2LL * G.n0;
    const double total1 = (double)M * t1;
    const double total2 = (double)(M / 2) * t2;
    const double total4 = (double)(M / 4) * t4 + (double)((M % 4) / 2) * t2;
    G.mode = 2;
    double tb = total4;
    if (total2 < tb) { tb = total2; G.mode = 1; }
    if (total1 < tb) { tb = total1; G.mode = 0; }
    // the tuning runs wrote into the arrays; leave them zeroed (pb_upload overwrites the fields anyway)
    for (int q = 0; q < 3; ++q)
      if (cudaMemset(G.base[q], 0, alloc) != cudaSuccess) { pb_teardown(); return 1; }
  }
  if (cudaDeviceSynchronize() != cudaSuccess) { pb_teardown(); return 1; }
  G.ready = true;
  return 0;
}

extern "C" int pb_upload(const pb_problem *problem, pb_buffer *buffers) {
  if (!G.ready || !problem || !buffers || problem->abi_version != PB_SOLVE_ABI_VERSION || problem->dimension_count != 2 ||
      problem->scalar_count != 0 || problem->buffer_count != 2 || !problem->dimensions) return 1;
  if (problem->dimensions[0] != G.n0 || problem->dimensions[1] != G.N) return 1;
  for (int q = 0; q < 2; ++q)
    if (!buffers[q].data || buffers[q].dtype != PB_F32 || buffers[q].rank != 3 || buffers[q].nbytes != G.bytes) return 1;
  if (cudaMemcpy(G.x[G.cur], buffers[0].data, G.bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
  if (cudaMemcpy(G.b, buffers[1].data, G.bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
  return 0;
}

extern "C" int pb_compute(void) {
  if (!G.ready) return 1;
  const int N = G.N;
  if (N < 3) return 0;                       // no interior cell: nothing is ever written
  long long M = 2LL * G.n0;                  // sweeps
  cudaError_t e = cudaSuccess;
  if (G.mode == 0) {
    e = run_plain();
  } else {
    // Passes of S sweeps: x[cur] (level s) -> x[cur^1] (level s+S); the B field supplies B's face values.
    // The last pass also writes level 2*n0-1 into the B field (its faces get the values they already hold).
    if (G.mode == 2 && M % 4 == 2) {
      M -= 2;
      e = launch_pass(2, M == 0, G.cfg2, G.x[G.cur], G.b, G.x[G.cur ^ 1], M == 0 ? G.b : nullptr, N);
      G.cur ^= 1;
    }
    const int S = (G.mode == 2) ? 4 : 2;
    const Cfg &c = (G.mode == 2) ? G.cfg4 : G.cfg2;
    while (M > 0 && e == cudaSuccess) {
      M -= S;
      e = launch_pass(S, M == 0, c, G.x[G.cur], G.b, G.x[G.cur ^ 1], M == 0 ? G.b : nullptr, N);
      G.cur ^= 1;
    }
  }
  const cudaError_t es = cudaDeviceSynchronize();
  return (e != cudaSuccess || es != cudaSuccess) ? 1 : 0;
}

extern "C" int pb_download(pb_buffer *buffers) {
  if (!G.ready || !buffers) return 1;
  for (int q = 0; q < 2; ++q)
    if (!buffers[q].data || buffers[q].nbytes != G.bytes) return 1;
  if (cudaMemcpy(buffers[0].data, G.x[G.cur], G.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
  if (cudaMemcpy(buffers[1].data, G.b, G.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
  return 0;
}

extern "C" int pb_teardown(void) {
  int rc = 0;
  if (G.gexec && cudaGraphExecDestroy(G.gexec) != cudaSuccess) rc = 1;
  if (G.grexec && cudaGraphExecDestroy(G.grexec) != cudaSuccess) rc = 1;
  G.gexec = G.grexec = nullptr;
  for (int q = 0; q < 3; ++q) {
    if (G.base[q] && cudaFree(G.base[q]) != cudaSuccess) rc = 1;
    G.base[q] = nullptr;
  }
  G.x[0] = G.x[1] = G.b = nullptr;
  G.ready = false;
  return rc;
}
