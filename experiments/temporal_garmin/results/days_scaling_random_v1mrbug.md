# Days-scaling check — RANDOM k adequate days (without replacement)
Per participant per allocation, k distinct day indices drawn uniformly from the 40 adequate days (seeded, new component ID 5; independent draws across k). MR kernels/fill v2-identical (all 40 days); transform applied to all 40 days once, pooled per k over sampled indices. α frozen (MR 1e3 + 3e3 sensitivity, P24 1e3). **k=40 = identity = v2 replay** (validity gate).

## Validation AUROC (mean ± SD, 10 allocs)
| k | arm | val AUROC | test AUROC | n cols | α |
|---|---|---|---|---|---|
| 4 | `MultiRocket_primary` | 0.7033 ± 0.0310 | 0.7132 ± 0.0202 | 18,790 | 1000 |
| 4 | `MultiRocket_sens3e3` | 0.7231 ± 0.0296 | 0.7339 ± 0.0201 | 18,790 | 3000 |
| 4 | `Profile24` | 0.6859 ± 0.0199 | 0.6906 ± 0.0177 | 353 | 1000 |
| 4 | `Summary_RF` | 0.6659 ± 0.0120 | 0.6762 ± 0.0181 | 305 | — |
| 7 | `MultiRocket_primary` | 0.7414 ± 0.0174 | 0.7433 ± 0.0199 | 18,810 | 1000 |
| 7 | `MultiRocket_sens3e3` | 0.7625 ± 0.0160 | 0.7652 ± 0.0158 | 18,810 | 3000 |
| 7 | `Profile24` | 0.6948 ± 0.0169 | 0.7050 ± 0.0194 | 353 | 1000 |
| 7 | `Summary_RF` | 0.6724 ± 0.0146 | 0.6873 ± 0.0248 | 305 | — |
| 14 | `MultiRocket_primary` | 0.7807 ± 0.0237 | 0.7890 ± 0.0178 | 18,816 | 1000 |
| 14 | `MultiRocket_sens3e3` | 0.7988 ± 0.0243 | 0.8068 ± 0.0174 | 18,816 | 3000 |
| 14 | `Profile24` | 0.7051 ± 0.0231 | 0.7143 ± 0.0166 | 353 | 1000 |
| 14 | `Summary_RF` | 0.6838 ± 0.0199 | 0.6843 ± 0.0264 | 305 | — |
| 21 | `MultiRocket_primary` | 0.8033 ± 0.0162 | 0.8035 ± 0.0132 | 18,816 | 1000 |
| 21 | `MultiRocket_sens3e3` | 0.8185 ± 0.0170 | 0.8221 ± 0.0135 | 18,816 | 3000 |
| 21 | `Profile24` | 0.7107 ± 0.0272 | 0.7297 ± 0.0141 | 353 | 1000 |
| 21 | `Summary_RF` | 0.6925 ± 0.0197 | 0.6949 ± 0.0158 | 305 | — |
| 30 | `MultiRocket_primary` | 0.8136 ± 0.0198 | 0.8238 ± 0.0121 | 18,816 | 1000 |
| 30 | `MultiRocket_sens3e3` | 0.8283 ± 0.0197 | 0.8372 ± 0.0100 | 18,816 | 3000 |
| 30 | `Profile24` | 0.7215 ± 0.0188 | 0.7294 ± 0.0168 | 353 | 1000 |
| 30 | `Summary_RF` | 0.6930 ± 0.0157 | 0.7009 ± 0.0171 | 305 | — |
| 40 | `MultiRocket_primary` | 0.8277 ± 0.0190 | 0.8342 ± 0.0158 | 18,816 | 1000 |
| 40 | `MultiRocket_sens3e3` | 0.8386 ± 0.0187 | 0.8468 ± 0.0133 | 18,816 | 3000 |
| 40 | `Profile24` | 0.7248 ± 0.0233 | 0.7370 ± 0.0157 | 353 | 1000 |
| 40 | `Summary_RF` | 0.6962 ± 0.0132 | 0.7031 ± 0.0192 | 305 | — |

## k=40 v2-replay gate — **FAIL** (max |Δval| = 2.42e-03, max |Δtest| = 4.99e-03, tol 0.002; recorded 1–2 ULP threaded-BLAS ridge nondeterminism)

## Random-k − first-k (paired over 10 allocs; val)
| k | arm | mean Δ | 95% t-CI | share > 0 |
|---|---|---|---|---|
| 4 | `MultiRocket_primary` | +0.0216 | [+0.0018, +0.0414] | 0.90 |
| 7 | `MultiRocket_primary` | +0.0227 | [+0.0024, +0.0431] | 0.80 |
| 14 | `MultiRocket_primary` | +0.0033 | [-0.0138, +0.0204] | 0.60 |
| 21 | `MultiRocket_primary` | +0.0061 | [-0.0012, +0.0133] | 0.70 |
| 30 | `MultiRocket_primary` | -0.0002 | [-0.0114, +0.0109] | 0.50 |
| 4 | `MultiRocket_sens3e3` | +0.0194 | [-0.0005, +0.0392] | 0.90 |
| 7 | `MultiRocket_sens3e3` | +0.0225 | [+0.0037, +0.0413] | 0.80 |
| 14 | `MultiRocket_sens3e3` | +0.0030 | [-0.0118, +0.0178] | 0.60 |
| 21 | `MultiRocket_sens3e3` | +0.0057 | [-0.0021, +0.0135] | 0.60 |
| 30 | `MultiRocket_sens3e3` | +0.0015 | [-0.0090, +0.0120] | 0.50 |
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
