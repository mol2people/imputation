### Coverage at per-class target precision (mean ± SD across 10 allocs; alloc = repeat unit)

Variant codes: `raw` = empirical threshold; `cpc` = Clopper-Pearson LCB-corrected (δ=0.1, confidence 0.90). `BASE_x_P40` = R1b half-sample cross-fit (test-only preds; n_test per direction ≈ 286). `Random` = seeded uniform scores using real val/test users. Coverage reported as fraction of cohort labeled with that class; recall = true positives / class size. Wilson 95% CI on precision.


#### `Combined`

| target | variant | cov₁ (%) | recall₁ (%) | prec₁ (mean [Wilson]) | cov₀ (%) | recall₀ (%) | prec₀ (mean [Wilson]) | cov_total (%) | abst (%) |
|---|---|---|---|---|---|---|---|---|---|
| 95% | raw | 21.4 ± 9.8 | 31.6 ± 13.7 | 0.956 (0.897–0.977) | 3.8 ± 3.4 | 9.6 ± 8.5 | 0.937 (0.664–0.974) | 25.2 ± 10.5 | 74.8 ± 10.5 |
| 95% | cpc | 8.1 ± 9.2 | 12.1 ± 13.6 | 0.959 (0.896–0.983) | — | — | — | 8.1 ± 9.2 | 91.9 ± 9.2 |
| 98% | raw | 10.0 ± 7.7 | 15.0 ± 11.3 | 0.971 (0.845–0.989) | 2.3 ± 2.6 | 5.8 ± 6.2 | 0.945 (0.611–0.981) | 12.3 ± 8.9 | 87.7 ± 8.9 |
| 98% | cpc | — | — | — | — | — | — | 0.0 ± 0.0 | 100.0 ± 0.0 |

#### `Summary_RF`

| target | variant | cov₁ (%) | recall₁ (%) | prec₁ (mean [Wilson]) | cov₀ (%) | recall₀ (%) | prec₀ (mean [Wilson]) | cov_total (%) | abst (%) |
|---|---|---|---|---|---|---|---|---|---|
| 95% | raw | 1.4 ± 1.7 | 2.0 ± 2.5 | 0.873 (0.455–0.972) | 0.6 ± 0.7 | 1.1 ± 1.4 | 0.660 (0.299–0.907) | 2.0 ± 2.1 | 98.0 ± 2.1 |
| 95% | cpc | — | — | — | — | — | — | 0.0 ± 0.0 | 100.0 ± 0.0 |
| 98% | raw | 1.3 ± 1.3 | 1.7 ± 1.8 | 0.867 (0.442–0.970) | 0.6 ± 0.7 | 1.1 ± 1.4 | 0.660 (0.299–0.907) | 1.8 ± 1.6 | 98.2 ± 1.6 |
| 98% | cpc | — | — | — | — | — | — | 0.0 ± 0.0 | 100.0 ± 0.0 |

#### `BASE_x_P40`

| target | variant | cov₁ (%) | recall₁ (%) | prec₁ (mean [Wilson]) | cov₀ (%) | recall₀ (%) | prec₀ (mean [Wilson]) | cov_total (%) | abst (%) |
|---|---|---|---|---|---|---|---|---|---|
| 95% | raw | 10.3 ± 8.3 | 14.5 ± 11.3 | 0.909 (0.737–0.965) | 2.5 ± 2.4 | 5.7 ± 5.4 | 0.874 (0.457–0.958) | 12.8 ± 9.1 | 87.2 ± 9.1 |
| 95% | cpc | — | — | — | — | — | — | 0.0 ± 0.0 | 100.0 ± 0.0 |
| 98% | raw | 5.8 ± 5.5 | 8.4 ± 7.8 | 0.936 (0.683–0.982) | 1.9 ± 2.0 | 4.5 ± 4.3 | 0.900 (0.435–0.969) | 7.8 ± 6.1 | 92.2 ± 6.1 |
| 98% | cpc | — | — | — | — | — | — | 0.0 ± 0.0 | 100.0 ± 0.0 |

#### `Random`

| target | variant | cov₁ (%) | recall₁ (%) | prec₁ (mean [Wilson]) | cov₀ (%) | recall₀ (%) | prec₀ (mean [Wilson]) | cov_total (%) | abst (%) |
|---|---|---|---|---|---|---|---|---|---|
| 95% | raw | 0.2 ± 0.3 | 0.2 ± 0.2 | 0.644 (0.144–0.893) | 0.0 ± 0.1 | 0.0 ± 0.2 | 0.500 (0.103–0.897) | 0.2 ± 0.3 | 99.8 ± 0.3 |
| 95% | cpc | — | — | — | — | — | — | 0.0 ± 0.0 | 100.0 ± 0.0 |
| 98% | raw | 0.2 ± 0.3 | 0.2 ± 0.2 | 0.644 (0.144–0.893) | 0.0 ± 0.1 | 0.0 ± 0.2 | 0.500 (0.103–0.897) | 0.2 ± 0.3 | 99.8 ± 0.3 |
| 98% | cpc | — | — | — | — | — | — | 0.0 ± 0.0 | 100.0 ± 0.0 |

#### Combined − BASE⊕P40 (applied coverage delta)

| target | variant | cov₁ Combined | cov₁ BASE⊕P40 | Δ cov₁ | cov₀ Combined | cov₀ BASE⊕P40 | Δ cov₀ | Δ cov_total |
|---|---|---|---|---|---|---|---|---|
| 95% | raw | 21.4 | 10.3 | +11.1 | 3.8 | 2.5 | +1.3 | +12.4 |
| 95% | cpc | 8.1 | 0.0 | +8.1 | 0.0 | 0.0 | +0.0 | +8.1 |
| 98% | raw | 10.0 | 5.8 | +4.2 | 2.3 | 1.9 | +0.4 | +4.5 |
| 98% | cpc | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | +0.0 |
