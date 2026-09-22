### α extended-grid addendum (alloc 0; predeclared NEXT_STEPS §Step 0.7)

Grid `[1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]`, selection on validation only (strict max, smallest α wins ties, as in precalibration); test at α* reported for completeness (exploratory).

| family | frozen α | val @ frozen | α* | val @ α* | gain | test @ frozen | test @ α* |
|---|---|---|---|---|---|---|---|
| Summary_linear | 1000 | 0.7161 | 300 | 0.7176 | +0.0015 | 0.7107 | 0.7118 |
| Profile24 | 1000 | 0.7459 | 1000 | 0.7459 | +0.0000 | 0.7404 | 0.7404 |
| Profile288 | 1000 | 0.7204 | 10000 | 0.7343 | +0.0138 | 0.7339 | 0.7178 |
| MultiRocket | 1000 | 0.8234 | 3000 | 0.8348 | +0.0114 | 0.8507 | 0.8626 |
| HYDRA | 1000 | 0.8194 | 10000 | 0.8468 | +0.0274 | 0.8401 | 0.8674 |
| Combined | 1000 | 0.8446 | 10000 | 0.8598 | +0.0152 | 0.8625 | 0.8784 |

**Verdict: USER DECISION: v3 full rerun (Combined val gain +0.0152 >= 0.01)**

Rebuild sanity vs v2 metrics csv (alloc 0, frozen α): max |Δval| = 0.0e+00, max |Δtest| = 0.0e+00 (≤1e-6 gate; expected 1–2 ULP from threaded-BLAS ridge).
Wall 428.9 s; peak RSS 6308.6 MiB (6.16 GiB; darwin ru_maxrss is bytes — unit corrected, see json note).
