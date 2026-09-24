# Days-scaling check — RANDOM k adequate days (without replacement)
Per participant per allocation, k distinct day indices drawn uniformly from the 40 adequate days (seeded, new component ID 5; independent draws across k). MR kernels/fill v2-identical (all 40 days); transform applied to all 40 days once, pooled per k over sampled indices. α frozen (MR 1e3 + 3e3 sensitivity, P24 1e3). **k=40 = identity = v2 replay** (validity gate).

## Validation AUROC (mean ± SD, 10 allocs)
| k | arm | val AUROC | test AUROC | n cols | α |
|---|---|---|---|---|---|
| 4 | `MultiRocket_primary` | 0.7043 ± 0.0310 | 0.7144 ± 0.0199 | 19,095 | 1000 |
| 4 | `MultiRocket_sens3e3` | 0.7242 ± 0.0293 | 0.7352 ± 0.0197 | 19,095 | 3000 |
| 4 | `Profile24` | 0.6859 ± 0.0199 | 0.6906 ± 0.0177 | 353 | 1000 |
| 4 | `Summary_RF` | 0.6659 ± 0.0120 | 0.6762 ± 0.0181 | 305 | — |
| 7 | `MultiRocket_primary` | 0.7422 ± 0.0170 | 0.7441 ± 0.0197 | 19,115 | 1000 |
| 7 | `MultiRocket_sens3e3` | 0.7630 ± 0.0158 | 0.7659 ± 0.0152 | 19,115 | 3000 |
| 7 | `Profile24` | 0.6948 ± 0.0169 | 0.7050 ± 0.0194 | 353 | 1000 |
| 7 | `Summary_RF` | 0.6724 ± 0.0146 | 0.6873 ± 0.0248 | 305 | — |
| 14 | `MultiRocket_primary` | 0.7811 ± 0.0221 | 0.7896 ± 0.0177 | 19,121 | 1000 |
| 14 | `MultiRocket_sens3e3` | 0.7989 ± 0.0233 | 0.8070 ± 0.0171 | 19,121 | 3000 |
| 14 | `Profile24` | 0.7051 ± 0.0231 | 0.7143 ± 0.0166 | 353 | 1000 |
| 14 | `Summary_RF` | 0.6838 ± 0.0199 | 0.6843 ± 0.0264 | 305 | — |
| 21 | `MultiRocket_primary` | 0.8034 ± 0.0165 | 0.8043 ± 0.0133 | 19,121 | 1000 |
| 21 | `MultiRocket_sens3e3` | 0.8185 ± 0.0174 | 0.8229 ± 0.0133 | 19,121 | 3000 |
| 21 | `Profile24` | 0.7107 ± 0.0272 | 0.7297 ± 0.0141 | 353 | 1000 |
| 21 | `Summary_RF` | 0.6925 ± 0.0197 | 0.6949 ± 0.0158 | 305 | — |
| 30 | `MultiRocket_primary` | 0.8139 ± 0.0195 | 0.8253 ± 0.0116 | 19,121 | 1000 |
| 30 | `MultiRocket_sens3e3` | 0.8286 ± 0.0194 | 0.8382 ± 0.0091 | 19,121 | 3000 |
| 30 | `Profile24` | 0.7215 ± 0.0188 | 0.7294 ± 0.0168 | 353 | 1000 |
| 30 | `Summary_RF` | 0.6930 ± 0.0157 | 0.7009 ± 0.0171 | 305 | — |
| 40 | `MultiRocket_primary` | 0.8284 ± 0.0184 | 0.8361 ± 0.0149 | 19,121 | 1000 |
| 40 | `MultiRocket_sens3e3` | 0.8387 ± 0.0184 | 0.8476 ± 0.0123 | 19,121 | 3000 |
| 40 | `Profile24` | 0.7248 ± 0.0233 | 0.7370 ± 0.0157 | 353 | 1000 |
| 40 | `Summary_RF` | 0.6962 ± 0.0132 | 0.7031 ± 0.0192 | 305 | — |

## k=40 v2-replay gate — **PASS** (max |Δval| = 0.00e+00, max |Δtest| = 0.00e+00, tol 0.002; recorded 1–2 ULP threaded-BLAS ridge nondeterminism)

## Random-k − first-k (paired over 10 allocs; val)
| k | arm | mean Δ | 95% t-CI | share > 0 |
|---|---|---|---|---|
| 4 | `MultiRocket_primary` | +0.0208 | [+0.0010, +0.0406] | 0.90 |
| 7 | `MultiRocket_primary` | +0.0234 | [+0.0029, +0.0439] | 0.70 |
| 14 | `MultiRocket_primary` | +0.0041 | [-0.0114, +0.0196] | 0.60 |
| 21 | `MultiRocket_primary` | +0.0072 | [+0.0006, +0.0138] | 0.80 |
| 30 | `MultiRocket_primary` | +0.0001 | [-0.0107, +0.0109] | 0.50 |
| 4 | `MultiRocket_sens3e3` | +0.0193 | [-0.0002, +0.0388] | 0.90 |
| 7 | `MultiRocket_sens3e3` | +0.0230 | [+0.0044, +0.0416] | 0.70 |
| 14 | `MultiRocket_sens3e3` | +0.0037 | [-0.0097, +0.0170] | 0.60 |
| 21 | `MultiRocket_sens3e3` | +0.0067 | [-0.0010, +0.0145] | 0.60 |
| 30 | `MultiRocket_sens3e3` | +0.0024 | [-0.0077, +0.0126] | 0.50 |
| 4 | `Profile24` | +0.0007 | [-0.0107, +0.0121] | 0.50 |
| 7 | `Profile24` | +0.0012 | [-0.0056, +0.0080] | 0.50 |
| 14 | `Profile24` | -0.0018 | [-0.0106, +0.0070] | 0.40 |
| 21 | `Profile24` | -0.0015 | [-0.0101, +0.0071] | 0.40 |
| 30 | `Profile24` | +0.0045 | [+0.0012, +0.0077] | 0.80 |
| 4 | `Summary_RF` | -0.0060 | [-0.0163, +0.0043] | 0.50 |
| 7 | `Summary_RF` | +0.0067 | [-0.0048, +0.0182] | 0.80 |
| 14 | `Summary_RF` | -0.0022 | [-0.0126, +0.0082] | 0.40 |
| 21 | `Summary_RF` | +0.0074 | [+0.0005, +0.0143] | 0.70 |
| 30 | `Summary_RF` | +0.0007 | [-0.0078, +0.0092] | 0.60 |

Positive Δ = random days beat the *first* k days at the same k (early-history specialness); Δ ≈ 0 = only the budget matters (signal stationary across the 40-day window).
