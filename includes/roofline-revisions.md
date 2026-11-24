<div class="njs-table" markdown="1">

| Kernel | Rule | Ops: AI model → revised | Bytes: AI model → revised | Intensity (Op/B) | Attainable (Top/s) | Bound by |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| `jacobi-1d` | D1 | unchanged | $12 n$ → $16 T n$ | 165 → **0.37** | 75.0 → **2.9** | compute → **memory** |
| `fdtd-2d` | D1 | unchanged | $4 (T + 6 n_x n_y)$ → $24 T n_x n_y$ | 406 → **0.46** | 75.0 → **3.5** | compute → **memory** |
| `jacobi-2d` | D1 | unchanged | $12 n^2$ → $16 T n^2$ | 691 → **0.62** | 75.0 → **4.8** | compute → **memory** |
| `seidel-2d` | D1 | unchanged | $8 n^2$ → $8 T n^2$ | 237 → **1.1** | 75.0 → **8.7** | compute → **memory** |
| `heat-3d` | D1, D2 | $30 T (n-2)^3$ → $16 T (n-2)^3$ | $12 n^3$ → $16 T n^3$ | 1,292 → **0.99** | 75.0 → **7.6** | compute → **memory** |
| `adi` | D1, D2 | $30 T (n-2)^2$ → $18 T (n-2)^2$ | $16 n^2$ → $32 T n^2$ | 448 → **0.56** | 1.2 → **1.2** | compute |
| `2mm` | D2 | $2 n_i n_j (n_k + n_l)$ → $2\min$ over the 2 evaluation orders | unchanged | 1,496 → **1,418** | 75.0 → **75.0** | compute |
| `3mm` | D2 | $2 n_j (n_i n_k + n_i n_l + n_l n_m)$ → $2\min$ over the 5 evaluation orders | unchanged | 1,834 → **1,788** | 75.0 → **75.0** | compute |
| `gemver` | D2 | $8 n^2$ → $4 n^2$ | unchanged | 2.0 → **1.00** | 15.4 → **7.7** | memory |

</div>
