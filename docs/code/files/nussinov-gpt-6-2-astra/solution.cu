#include "interface.h"

#include <cuda_runtime.h>
#include <climits>
#include <cstddef>

namespace {
constexpr int tile = 32;
constexpr int lowest = INT_MIN / 2;
static_assert(sizeof(int) == 4);

__device__ __forceinline__ size_t index_of(int row, int col, int n) {
    return size_t(row) * n + col;
}

// One warp per row. All split operands for a given gap are already in shared
// memory, including the original subdiagonal used by the adjacent-cell case.
__global__ void diagonal(int n, const char *sequence, int *table,
                         int *counts, int *append_weights) {
    __shared__ int d[tile][tile + 1], seq[tile];
    int x = threadIdx.x, y = threadIdx.y;
    int base = blockIdx.x * tile;
    int width = min(tile, n - base);
    if (y < width && x < width)
        d[y][x] = table[index_of(base + y, base + x, n)];
    if (y == 0 && x < width) {
        seq[x] = sequence[base + x];
        counts[base + x] = 0;
    }
    __syncthreads();
    for (int gap = 1; gap < width; ++gap) {
        int j = y + gap;
        if (j < width) {
            int a = y + 2 + x;
            int split = a <= j ? d[y][a - 1] + d[a][j] : lowest;
            for (int step = 16; step > 0; step /= 2)
                split = max(split, __shfl_down_sync(0xffffffffu, split, step));
            if (x == 0) {
                int wrap = d[y + 1][j - 1] + (gap > 1 && seq[y] + seq[j] == 3);
                int skip = max(d[y][j - 1], d[y + 1][j]);
                d[y][j] = max(max(d[y][j], wrap), max(split, skip));
            }
        }
        __syncthreads();
    }

    // Weight of appending [y,x] to an already legal left split operand.
    // Leading singletons can be skipped or appended individually. As soon
    // as a piece has length >= 2, d itself can absorb all subsequent pieces.
    int append = 0;
    if (x < width) {
        for (int a = x; a >= y; --a)
            append = max(d[a][x], max(0, d[a][a]) + append);
    }
    append_weights[(size_t(blockIdx.x) * tile + y) * tile + x] = append;
    if (y < x && x < width)
        table[index_of(base + y, base + x, n)] = d[y][x];
}

__global__ void rectangle(int n, const char *sequence, int *table,
                          const int *append_weights, int *counts,
                          int2 *all_records, int tile_gap) {
    // d has a halo below and to the left. Cell (y,x) is stored at d[y][x+1].
    __shared__ int d[tile + 1][tile + 1];
    __shared__ int prepend[tile][tile + 1], append[tile + 1][tile + 1];
    __shared__ int change_row[tile], change_col[tile], change_value[tile];
    __shared__ int change_count;
    __shared__ unsigned keep[tile];

    int x = threadIdx.x, y = threadIdx.y;
    int col_tile = blockIdx.x + tile_gap;
    int r0 = blockIdx.x * tile, c0 = col_tile * tile;
    int i = r0 + y, j = c0 + x;
    int width = min(tile, n - c0);
    // One list per endpoint, with room for all c0 earlier start positions.
    // Adjacent columns' records of the same rank are contiguous.
    int2 *records = all_records + size_t(col_tile) * (col_tile - 1) / 2 * tile * tile;
    int old_count = 0;
    int exterior = lowest, input = lowest;
    int diag_i = table[index_of(i, i, n)], diag_j = 0;
    bool pair = false;
    if (x == 0) d[y][0] = table[index_of(i, c0 - 1, n)];
    if (y == 0) {
        keep[x] = 0;
        if (x == 0) {
            change_count = 0;
            d[tile][0] = table[index_of(r0 + tile, c0 - 1, n)];
        }
    }
    if (x < width) {
        if (y == 0) d[tile][x + 1] = table[index_of(r0 + tile, j, n)];
        old_count = counts[j];
        input = table[index_of(i, j, n)];
        diag_j = table[index_of(j, j, n)];
        pair = i + 1 < j && sequence[i] + sequence[j] == 3;
        for (int slot = 0; slot < old_count; ++slot) {
            int2 atom = records[size_t(slot) * tile + x];
            if (atom.x >= i + 2)
                exterior = max(exterior, table[index_of(i, atom.x - 1, n)] + atom.y);
        }
    }

    // Adding a prefix of length one on the left can only skip that element.
    prepend[y][x] = x - y >= 2 ? max(0, table[index_of(i, r0 + x - 1, n)]) : 0;
    if (x == 0)
        prepend[y][tile] = y + 1 == tile ? 0 : max(0, table[index_of(i, r0 + tile - 1, n)]);
    append[y][x] = append_weights[(size_t(col_tile) * tile + y) * tile + x];
    if (y == 0) append[tile][x] = 0;
    __syncthreads();

    // Best score obtainable by extending either boundary into this tile.
    int best = lowest;
    if (x < width) {
        int bottom = d[tile][x + 1];
        int west_extension = append[0][x];
        if (tile_gap == 1) {
            // At the corner both boundaries end in singletons. Once the first
            // interior column has been reached, its interval has length >= 2.
            bottom = max(bottom, d[tile][1] + append[1][x]);
            if (y == tile - 1) west_extension = append[1][x];
        }
        best = max(prepend[y][tile] + bottom, d[y][0] + west_extension);
    }
    d[y][x + 1] = best;
    int direct = max(input, exterior);

    for (;;) {
        __syncthreads();
        int proposal = x < width ? max(direct, d[y + 1][x] + pair) : lowest;
        if (proposal > best) {
            int slot = atomicAdd(&change_count, 1);
            if (slot < tile) {
                change_row[slot] = y;
                change_col[slot] = x;
                change_value[slot] = proposal;
            }
        }
        __syncthreads();
        int changes = change_count;
        if (changes == 0) break;

        // All local skips and splits around a new score consist of one
        // independent prefix and one suffix. Propagate both in a single round.
        if (changes <= tile) {
            for (int k = 0; k < changes; ++k) {
                int r = change_row[k], c = change_col[k];
                if (r >= y && c <= x)
                    best = max(best, prepend[y][r] + change_value[k] + append[c + 1][x]);
            }
        } else {
            // The same maximum, factored into row and column reductions.
            best = max(best, proposal);
            d[y][x + 1] = best;
            __syncthreads();
            for (int c = 0; c < x; ++c)
                best = max(best, d[y][c + 1] + append[c + 1][x]);
            __syncthreads();
            d[y][x + 1] = best;
            __syncthreads();
            for (int r = y + 1; r < tile; ++r)
                best = max(best, prepend[y][r] + d[r][x + 1]);
        }
        __syncthreads();
        d[y][x + 1] = best;
        if (x == 0 && y == 0) change_count = 0;
    }

    // Only a raw or wrapped score can produce a new irreducible right
    // operand. Discard ties with a split or an endpoint that an outer prefix
    // can consume independently. All remaining starts are kept, with no cap.
    if (x < width && (best == input || best == d[y + 1][x] + pair) &&
        best > exterior && best > d[y + 1][x + 1] + max(0, diag_i) &&
        best > d[y][x] + max(0, diag_j)) {
        bool needed = true;
        for (int a = y + 2; a <= tile && needed; ++a)
            needed = best > prepend[y][a] + d[a][x + 1];
        for (int a = max(0, i + 2 - c0); a < x && needed; ++a)
            needed = best > d[y][a] + append[a][x];
        if (needed) atomicOr(&keep[x], 1u << y);
    }
    __syncthreads();
    if (x < width) {
        table[index_of(i, j, n)] = best;
        if (y == 0) {
            unsigned bits = keep[x];
            while (bits) {
                int r = __ffs(bits) - 1;
                records[size_t(old_count++) * tile + x] = make_int2(r0 + r, d[r][x + 1]);
                bits &= bits - 1;
            }
            counts[j] = old_count;
        }
    }
}

int n;
size_t table_bytes;
char *device_sequence;
int *device_table, *counts, *append_weights;
int2 *records;
cudaStream_t stream;
cudaGraph_t graph;
cudaGraphExec_t executable;

cudaError_t schedule(int tiles) {
    dim3 block(tile, tile);
    diagonal<<<tiles, dim3(tile, n < tile ? n : tile), 0, stream>>>(
        n, device_sequence, device_table, counts, append_weights);
    cudaError_t status = cudaGetLastError();
    for (int gap = 1; gap < tiles && status == cudaSuccess; ++gap) {
        rectangle<<<tiles - gap, block, 0, stream>>>(
            n, device_sequence, device_table, append_weights, counts, records, gap);
        status = cudaGetLastError();
    }
    return status;
}

void record_error(cudaError_t &status, cudaError_t next) {
    if (status == cudaSuccess) status = next;
}
} // namespace

extern "C" int pb_teardown() {
    cudaError_t status = cudaSuccess;
    if (executable) record_error(status, cudaGraphExecDestroy(executable));
    if (graph) record_error(status, cudaGraphDestroy(graph));
    if (stream) record_error(status, cudaStreamDestroy(stream));
    if (records) record_error(status, cudaFree(records));
    if (append_weights) record_error(status, cudaFree(append_weights));
    if (counts) record_error(status, cudaFree(counts));
    if (device_table) record_error(status, cudaFree(device_table));
    if (device_sequence) record_error(status, cudaFree(device_sequence));
    executable = nullptr;
    graph = nullptr;
    stream = nullptr;
    records = nullptr;
    append_weights = nullptr;
    counts = nullptr;
    device_table = nullptr;
    device_sequence = nullptr;
    return int(status);
}

extern "C" int pb_setup(const pb_problem_shape *shape) {
    if (!shape || !shape->dimensions || !shape->buffers ||
        shape->abi_version != PB_SOLVE_ABI_VERSION || shape->dimension_count != 1 ||
        shape->scalar_count != 0 || shape->buffer_count != 2 ||
        shape->dimensions[0] < 1 || shape->dimensions[0] > INT_MAX) return 1;
    n = int(shape->dimensions[0]);
    table_bytes = size_t(n) * n * sizeof(int);
    if (shape->buffers[0].nbytes != size_t(n) || shape->buffers[1].nbytes != table_bytes)
        return 1;
    size_t tiles = (size_t(n) + tile - 1) / tile;
    size_t record_slots = tiles * (tiles - 1) / 2 * tile * tile;
    cudaError_t status = cudaMalloc(&device_sequence, n);
    if (status == cudaSuccess) status = cudaMalloc(&device_table, table_bytes);
    if (status == cudaSuccess) status = cudaMalloc(&counts, size_t(n) * sizeof(int));
    if (status == cudaSuccess)
        status = cudaMalloc(&append_weights, tiles * tile * tile * sizeof(int));
    if (status == cudaSuccess && record_slots)
        status = cudaMalloc(&records, record_slots * sizeof(int2));
    if (status == cudaSuccess) status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (status == cudaSuccess) status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (status == cudaSuccess) {
        status = schedule(int(tiles));
        record_error(status, cudaStreamEndCapture(stream, &graph));
    }
    if (status == cudaSuccess) status = cudaGraphInstantiate(&executable, graph, 0);
    if (status == cudaSuccess) status = cudaGraphUpload(executable, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) pb_teardown();
    return int(status);
}

extern "C" int pb_upload(const pb_problem *, pb_buffer *buffers) {
    if (!buffers) return 1;
    cudaError_t status = cudaMemcpyAsync(device_sequence, buffers[0].data, n,
                                         cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess)
        status = cudaMemcpyAsync(device_table, buffers[1].data, table_bytes,
                                 cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    return int(status);
}

extern "C" int pb_compute() {
    cudaError_t status = cudaGraphLaunch(executable, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    return int(status);
}

extern "C" int pb_download(pb_buffer *buffers) {
    if (!buffers) return 1;
    cudaError_t status = cudaMemcpyAsync(buffers[1].data, device_table, table_bytes,
                                         cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    return int(status);
}
