<div class="njs-table" markdown="1">

| Kernel | Category | Scored instance | Ops | Bytes | Intensity (Op/B) | Peak | Attainable (Top/s) | Bound by |
| :--- | :--- | :--- | :--- | :--- | ---: | :---: | ---: | :---: |
| `gemver` **✱** | BLAS | n = 16,381 | `4*n*n` | `4*(n*n + n + n + n + n + n + n + n + n + n)` | 1.00 | FP32 | 7.7 | memory |
| `trmm` | BLAS | m = 12,211, n = 9,767 | `n*m*m` | `4*(m*(m - 1)/2 + 2*m*n)` | 1,163 | FP32 | 75.0 | compute |
| `gemm` | BLAS | ni = 9,461, nj = 9,463, nk = 9,439 | `2*ni*nj*nk` | `4*(ni*nj + ni*nj + ni*nk + nk*nj)` | 1,181 | FP32 | 75.0 | compute |
| `gesummv` | BLAS | n = 11,587 | `4*n*n` | `4*(n*n + n*n + n + n)` | 0.50 | FP32 | 3.8 | memory |
| `symm` | BLAS | m = 9,067, n = 10,267 | `2*m*m*n` | `4*(m*(m + 1)/2 + 3*m*n)` | 1,317 | FP32 | 75.0 | compute |
| `syr2k` | BLAS | n = 10,223, m = 8,017 | `2*m*n*(n + 1)` | `4*(n*(n + 1) + 2*n*m)` | 1,561 | FP32 | 75.0 | compute |
| `syrk` | BLAS | n = 12,301, m = 9,461 | `m*n*(n + 1)` | `4*(n*(n + 1) + n*m)` | 1,337 | FP32 | 75.0 | compute |
| `bicg` | Kernels | m = 16,381, n = 16,369 | `4*n*m` | `4*(n*m + m + n + m + n)` | 1.00 | FP32 | 7.7 | memory |
| `3mm` **✱** | Kernels | ni = 6,521, nj = 6,073, nk = 6,229, nl = 5,897, nm = 6,359 | `2*min(ni*nk*nj + nj*nm*nl + ni*nj*nl, ni*nk*nj + ni*nj*nm + ni*nm*nl, nk*nj*nm + ni*nk*nm + ni*nm*nl, nk*nj*nm + nk*nm*nl + ni*nk*nl, nj*nm*nl + nk*nj*nl + ni*nk*nl)` | `4*(ni*nk + nk*nj + nj*nm + nm*nl + ni*nl)` | 1,788 | FP32 | 75.0 | compute |
| `atax` | Kernels | m = 20,011, n = 13,411 | `4*m*n` | `4*(m*n + n + n)` | 1.00 | FP32 | 7.7 | memory |
| `2mm` **✱** | Kernels | ni = 7,841, nj = 7,127, nk = 7,487, nl = 6,709 | `2*min(ni*nk*nj + ni*nj*nl, nk*nj*nl + ni*nk*nl)` | `4*(ni*nk + nk*nj + nj*nl + ni*nl + ni*nl)` | 1,418 | FP32 | 75.0 | compute |
| `doitgen` | Kernels | nr = 487, nq = 521, np = 1,051 | `2*nr*nq*np*np` | `4*(nr*nq*np + nr*nq*np + np*np)` | 262 | FP32 | 75.0 | compute |
| `mvt` | Kernels | n = 16,381 | `4*n*n` | `4*(n + n + n + n + n + n + n*n)` | 1.00 | FP32 | 7.7 | memory |
| `ludcmp` | Solvers | n = 16,381 | `(2*n*n*n)/3 + 2*n*n` | `4*(n*n + 2*n)` | 2,730 | FP32 | 75.0 | compute |
| `gramschmidt` | Solvers | m = 12,457, n = 8,123 | `2*m*n*n + 3*m*n` | `4*(2*m*n + n*(n + 1)/2)` | 1,746 | FP32 | 75.0 | compute |
| `durbin` | Solvers | n = 100,003 | `2*n*n` | `4*(n + n)` | 25,001 | FP32 | 75.0 | compute |
| `trisolv` | Solvers | n = 12,163 | `n*n` | `4*(n*(n + 1)/2 + 2*n)` | 0.50 | FP32 | 3.8 | memory |
| `cholesky` | Solvers | n = 16,381 | `(n*n*n)/3` | `4*(n*(n + 1))` | 1,365 | FP32 | 75.0 | compute |
| `lu` | Solvers | n = 16,381 | `(2*n*n*n)/3` | `4*(n*n + n*n)` | 1,365 | FP32 | 75.0 | compute |
| `covariance` | Data mining | m = 11,317, n = 12,391 | `n*m*(m + 1) + 2*n*m` | `4*(n*m + m*m)` | 1,479 | FP32 | 75.0 | compute |
| `correlation` | Data mining | m = 11,321, n = 12,373 | `n*m*(m - 1) + 6*n*m` | `4*(n*m + m*m)` | 1,479 | FP32 | 75.0 | compute |
| `floyd-warshall` | Medley | n = 16,000 | `2*n*n*n` | `4*(n*n + n*n)` | 4,000 | INT32 | 37.5 | compute |
| `nussinov` | Medley | n = 16,381 | `(n*n*n)/3` | `4*(n*(n - 1) + 3*n - 1)` | 1,365 | INT32 | 37.5 | compute |
| `deriche` | Medley | w = 10,009, h = 6,703 | `32*w*h` | `4*(w*h + w*h)` | 4.0 | FP32 | 30.8 | memory |
| `heat-3d` **✱** | Stencils | tsteps = 523, n = 509 | `16*tsteps*(n - 2)*(n - 2)*(n - 2)` | `tsteps*4*(4*n*n*n)` | 0.99 | FP32 | 7.6 | memory |
| `fdtd-2d` **✱** | Stencils | tmax = 887, nx = 10,007, ny = 8,941 | `tmax*(3*(nx - 1)*ny + 3*nx*(ny - 1) + 5*(nx - 1)*(ny - 1))` | `tmax*4*(6*nx*ny)` | 0.46 | FP32 | 3.5 | memory |
| `jacobi-2d` **✱** | Stencils | tsteps = 829, n = 11,587 | `10*tsteps*(n - 2)*(n - 2)` | `tsteps*4*(4*n*n)` | 0.62 | FP32 | 4.8 | memory |
| `seidel-2d` **✱** | Stencils | tsteps = 211, n = 16,381 | `9*tsteps*(n - 2)*(n - 2)` | `tsteps*4*(2*n*n)` | 1.1 | FP32 | 8.7 | memory |
| `adi` **✱** | Stencils | tsteps = 239, n = 5,791 | `18*tsteps*(n - 2)*(n - 2)` | `tsteps*8*(4*n*n)` | 0.56 | FP64 | 1.2 | compute |
| `jacobi-1d` **✱** | Stencils | tsteps = 331, n = 134,217,689 | `6*tsteps*(n - 2)` | `tsteps*4*(4*n)` | 0.37 | FP32 | 2.9 | memory |

</div>
