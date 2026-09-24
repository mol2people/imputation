# Days-scaling check — AUROC vs k adequate days
MultiRocket / Profile24 / Summary_RF, 10 allocations. k=40 anchor cited from the frozen v2 run (`results/temporal_metrics.csv`).
**Design** (`day_scaling.py`): MR kernels fit once per allocation on 4 seeded days drawn from the first K_MAX=30 adequate days of training users (same seed stream as v2); kernels held fixed across all k. α frozen at the v2 value (1e3) for Profile24 and MultiRocket_primary; MultiRocket_sens3e3 = sensitivity at α=3e3 (extended-grid optimum at k=40). No HYDRA, no Shuffled_MR. Fill: train-only clock-bin medians from the first 30 days. B40_k / P24_k recomputed on the first k adequate days per user.
**Caveat:** the k=4 → k=30 trajectory isolates the pooling-window effect (kernels held fixed). For a fully deployment-realistic run (kernels refit per k), kernel-sampling noise would add to the curve.

## Validation AUROC (mean ± SD, 10 allocs)
| k | arm | val AUROC | test AUROC | n cols | α |
|---|---|---|---|---|---|
| 4 | `MultiRocket_primary` | 0.6834 ± 0.0184 | 0.6932 ± 0.0112 | 19,103 | 1000 |
| 4 | `MultiRocket_sens3e3` | 0.7050 ± 0.0192 | 0.7168 ± 0.0121 | 19,103 | 3000 |
| 4 | `Profile24` | 0.6852 ± 0.0218 | 0.6930 ± 0.0199 | 353 | 1000 |
| 4 | `Summary_RF` | 0.6719 ± 0.0154 | 0.6713 ± 0.0256 | 305 | — |
| 7 | `MultiRocket_primary` | 0.7188 ± 0.0197 | 0.7322 ± 0.0123 | 19,103 | 1000 |
| 7 | `MultiRocket_sens3e3` | 0.7400 ± 0.0201 | 0.7535 ± 0.0133 | 19,103 | 3000 |
| 7 | `Profile24` | 0.6936 ± 0.0212 | 0.6942 ± 0.0172 | 353 | 1000 |
| 7 | `Summary_RF` | 0.6656 ± 0.0228 | 0.6681 ± 0.0205 | 305 | — |
| 14 | `MultiRocket_primary` | 0.7770 ± 0.0207 | 0.7750 ± 0.0149 | 19,103 | 1000 |
| 14 | `MultiRocket_sens3e3` | 0.7953 ± 0.0192 | 0.7946 ± 0.0148 | 19,103 | 3000 |
| 14 | `Profile24` | 0.7069 ± 0.0231 | 0.7123 ± 0.0138 | 353 | 1000 |
| 14 | `Summary_RF` | 0.6860 ± 0.0190 | 0.6944 ± 0.0193 | 305 | — |
| 21 | `MultiRocket_primary` | 0.7962 ± 0.0162 | 0.7979 ± 0.0131 | 19,109 | 1000 |
| 21 | `MultiRocket_sens3e3` | 0.8118 ± 0.0178 | 0.8151 ± 0.0115 | 19,109 | 3000 |
| 21 | `Profile24` | 0.7123 ± 0.0221 | 0.7210 ± 0.0167 | 353 | 1000 |
| 21 | `Summary_RF` | 0.6852 ± 0.0130 | 0.6967 ± 0.0154 | 305 | — |
| 30 | `MultiRocket_primary` | 0.8138 ± 0.0169 | 0.8117 ± 0.0195 | 19,115 | 1000 |
| 30 | `MultiRocket_sens3e3` | 0.8261 ± 0.0180 | 0.8281 ± 0.0153 | 19,115 | 3000 |
| 30 | `Profile24` | 0.7171 ± 0.0213 | 0.7292 ± 0.0181 | 353 | 1000 |
| 30 | `Summary_RF` | 0.6924 ± 0.0170 | 0.7026 ± 0.0201 | 305 | — |

## v2 k=40 anchor (10 allocs, frozen α=1e3)
| k | arm | val AUROC | test AUROC | α |
|---|---|---|---|---|
| 40 | `Summary_RF` | 0.6962 ± 0.0132 | 0.7031 ± 0.0192 | — |
| 40 | `Profile24` | 0.7248 ± 0.0233 | 0.7370 ± 0.0157 | 1000 |
| 40 | `MultiRocket` | 0.8284 ± 0.0184 | 0.8361 ± 0.0149 | 1000 |

**Note:** the MultiRocket v2 anchor at α=1e3 corresponds to `MultiRocket_primary` in this check (same α, same protocol minus the kernel-sharing/k-window design noted above).
