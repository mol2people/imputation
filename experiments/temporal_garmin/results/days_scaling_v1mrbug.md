# Days-scaling check — AUROC vs k adequate days
MultiRocket / Profile24 / Summary_RF, 10 allocations. k=40 anchor cited from the frozen v2 run (`results/temporal_metrics.csv`).
**Design** (`day_scaling.py`): MR kernels fit once per allocation on 4 seeded days drawn from the first K_MAX=30 adequate days of training users (same seed stream as v2); kernels held fixed across all k. α frozen at the v2 value (1e3) for Profile24 and MultiRocket_primary; MultiRocket_sens3e3 = sensitivity at α=3e3 (extended-grid optimum at k=40). No HYDRA, no Shuffled_MR. Fill: train-only clock-bin medians from the first 30 days. B40_k / P24_k recomputed on the first k adequate days per user.
**Caveat:** the k=4 → k=30 trajectory isolates the pooling-window effect (kernels held fixed). For a fully deployment-realistic run (kernels refit per k), kernel-sampling noise would add to the curve.

## Validation AUROC (mean ± SD, 10 allocs)
| k | arm | val AUROC | test AUROC | n cols | α |
|---|---|---|---|---|---|
| 4 | `MultiRocket_primary` | 0.6817 ± 0.0184 | 0.6900 ± 0.0113 | 18,798 | 1000 |
| 4 | `MultiRocket_sens3e3` | 0.7037 ± 0.0193 | 0.7143 ± 0.0122 | 18,798 | 3000 |
| 4 | `Profile24` | 0.6852 ± 0.0218 | 0.6930 ± 0.0199 | 353 | 1000 |
| 4 | `Summary_RF` | 0.6719 ± 0.0154 | 0.6713 ± 0.0256 | 305 | — |
| 7 | `MultiRocket_primary` | 0.7187 ± 0.0201 | 0.7306 ± 0.0130 | 18,798 | 1000 |
| 7 | `MultiRocket_sens3e3` | 0.7400 ± 0.0203 | 0.7520 ± 0.0134 | 18,798 | 3000 |
| 7 | `Profile24` | 0.6936 ± 0.0212 | 0.6942 ± 0.0172 | 353 | 1000 |
| 7 | `Summary_RF` | 0.6656 ± 0.0228 | 0.6681 ± 0.0205 | 305 | — |
| 14 | `MultiRocket_primary` | 0.7774 ± 0.0211 | 0.7735 ± 0.0151 | 18,798 | 1000 |
| 14 | `MultiRocket_sens3e3` | 0.7958 ± 0.0195 | 0.7938 ± 0.0155 | 18,798 | 3000 |
| 14 | `Profile24` | 0.7069 ± 0.0231 | 0.7123 ± 0.0138 | 353 | 1000 |
| 14 | `Summary_RF` | 0.6860 ± 0.0190 | 0.6944 ± 0.0193 | 305 | — |
| 21 | `MultiRocket_primary` | 0.7972 ± 0.0173 | 0.7975 ± 0.0138 | 18,804 | 1000 |
| 21 | `MultiRocket_sens3e3` | 0.8128 ± 0.0184 | 0.8147 ± 0.0126 | 18,804 | 3000 |
| 21 | `Profile24` | 0.7123 ± 0.0221 | 0.7210 ± 0.0167 | 353 | 1000 |
| 21 | `Summary_RF` | 0.6852 ± 0.0130 | 0.6967 ± 0.0154 | 305 | — |
| 30 | `MultiRocket_primary` | 0.8138 ± 0.0171 | 0.8111 ± 0.0208 | 18,810 | 1000 |
| 30 | `MultiRocket_sens3e3` | 0.8267 ± 0.0175 | 0.8278 ± 0.0169 | 18,810 | 3000 |
| 30 | `Profile24` | 0.7171 ± 0.0213 | 0.7292 ± 0.0181 | 353 | 1000 |
| 30 | `Summary_RF` | 0.6924 ± 0.0170 | 0.7026 ± 0.0201 | 305 | — |

## v2 k=40 anchor (10 allocs, frozen α=1e3)
| k | arm | val AUROC | test AUROC | α |
|---|---|---|---|---|
| 40 | `Summary_RF` | 0.6962 ± 0.0132 | 0.7031 ± 0.0192 | — |
| 40 | `Profile24` | 0.7248 ± 0.0233 | 0.7370 ± 0.0157 | 1000 |
| 40 | `MultiRocket` | 0.8284 ± 0.0184 | 0.8361 ± 0.0149 | 1000 |

**Note:** the MultiRocket v2 anchor at α=1e3 corresponds to `MultiRocket_primary` in this check (same α, same protocol minus the kernel-sharing/k-window design noted above).
