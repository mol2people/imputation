# Temporal representations for recorded-salutation prediction — results

Implements [`../PLAN.md`](../PLAN.md). 8 arms × 10 allocations; R1b cohort/folds/first-40-day window. Frozen plan at commit `dabab35`. CPU torch 2.14.0 + numba 0.67.0 (`workqueue` threading layer).

**This is the v2 run (post-erratum).** v1 (commit `168b9d7`, as-run) carried the zero-sentinel contamination defect (erratum 1, quantified per arm in section 9). Defects 2-3 (verify dispatch, section 4 pairing) have no numeric effect on arm AUROCs.


## 1. Absolute AUROC (mean ± SD over 10 allocations)

| arm | val AUROC | test AUROC | n cols (post-hygiene) | α (frozen) |
|---|---|---|---|---|
| `Summary_linear` | 0.7077 ± 0.0193 | 0.7157 ± 0.0204 | 305 | 1000 |
| `Summary_RF` | 0.6962 ± 0.0132 | 0.7031 ± 0.0192 | 305 | — |
| `Profile24` | 0.7248 ± 0.0233 | 0.7370 ± 0.0157 | 353 | 1000 |
| `Profile288` | 0.7140 ± 0.0181 | 0.7327 ± 0.0159 | 881 | 1000 |
| `MultiRocket` | 0.8284 ± 0.0184 | 0.8361 ± 0.0149 | 19,118 | 1000 |
| `HYDRA` | 0.8374 ± 0.0149 | 0.8442 ± 0.0113 | 12,593 | 1000 |
| `Combined` | 0.8489 ± 0.0141 | 0.8565 ± 0.0146 | 31,406 | 1000 |
| `Shuffled_MR` | 0.6619 ± 0.0139 | 0.6643 ± 0.0183 | 19,121 | 1000 |

## 2. Primary family — 4 paired deltas vs `Summary_RF` on VALIDATION (Bonferroni two-sided 95%-simult, `t.ppf(1 − 0.05/8, 9)`, df = 9)

| comparison | mean Δ | 95% t-CI | Bonferroni 95%-simult | SD | share > 0 |
|---|---|---|---|---|---|
| Profile288 − Summary_RF | +0.0177 | [+0.0081, +0.0274] | [+0.0045, +0.0310] | 0.0135 | 0.90 |
| MultiRocket − Summary_RF | +0.1322 | [+0.1150, +0.1494] | [+0.1085, +0.1558] | 0.0241 | 1.00 |
| HYDRA − Summary_RF | +0.1412 | [+0.1263, +0.1561] | [+0.1207, +0.1617] | 0.0208 | 1.00 |
| Combined − Summary_RF | +0.1527 | [+0.1379, +0.1675] | [+0.1323, +0.1730] | 0.0207 | 1.00 |

## 3. Secondaries (uncorrected 95% CIs, no family claim)

| comparison | mean Δ | 95% t-CI | share > 0 |
|---|---|---|---|
| Profile24 − Summary_RF | +0.0285 | [+0.0171, +0.0400] | 0.90 |
| Profile288 − Profile24 | -0.0108 | [-0.0172, -0.0044] | 0.10 |
| MultiRocket − Shuffled_MR | +0.1664 | [+0.1503, +0.1826] | 1.00 |
| Shuffled_MR − Summary_RF | -0.0343 | [-0.0456, -0.0230] | 0.10 |
| MultiRocket − Summary_RF | +0.1322 | [+0.1150, +0.1494] | 1.00 |
| Combined − Summary_RF | +0.1527 | [+0.1379, +0.1675] | 1.00 |
| Summary_linear − Summary_RF | +0.0115 | [+0.0031, +0.0199] | 0.80 |

## 4. Combined − BASE⊕P40 (paired TEST AUROCs, R1b test-only participant preds)

| alloc | n | AUROC Combined | AUROC BASE⊕P40 | Δ |
|---|---|---|---|---|
| 0 | 573 | 0.8625 | 0.7567 | +0.1058 |
| 1 | 573 | 0.8607 | 0.7053 | +0.1554 |
| 2 | 573 | 0.8575 | 0.7193 | +0.1382 |
| 3 | 573 | 0.8515 | 0.7578 | +0.0937 |
| 4 | 573 | 0.8847 | 0.7788 | +0.1059 |
| 5 | 573 | 0.8720 | 0.7734 | +0.0986 |
| 6 | 573 | 0.8521 | 0.7501 | +0.1020 |
| 7 | 573 | 0.8481 | 0.7566 | +0.0915 |
| 8 | 573 | 0.8369 | 0.7514 | +0.0856 |
| 9 | 573 | 0.8386 | 0.7639 | +0.0748 |
| **pooled (10 allocs)** | 573 | — | — | +0.1051  95% t-CI [+0.0878, +0.1225]  (share > 0: 1.00, df=9) |


## 5. §5 benchmark (100 train participants × 40 days, warmup excluded)

| metric | bench | full-projection (10 allocs) |
|---|---|---|
| MultiRocket transform wall | 0.88s | ~11.2 min (×2 incl. shuffled) |
| HYDRA transform wall | 8.51s | ~54.6 min |
| peak RSS | 4256 MiB | — |
| fill train-clock-bin medians: min=54.8 max=77.2 used_nan_fallback=0 | — |


## 6. Epoch-span audit (predeclared rule)

- Weighted frac of retained epochs with span > 5 min: **0.0004** (threshold > 0.50 → fractional Profile288 sensitivity).

- Decision (recorded pre-model): **no fractional sensitivity (start-bin assignment kept)**.

- Per-bin observed mask coverage (288-bin days): min=20, p50=282.0, lt8=0, zero-mask days=0.

- NaN-timezoneOffset events rejected: 0 (PLAN §3 divergence from `src/sidequest/diurnal.py`'s UTC-substitution convention — zero effect in practice for this cohort).


## 7. Interpretation

**Within-day value placement carries the signal.** Permuting observed HR values among a participant's observed clock-bins within each day (Shuffled_MR: identical value multiset, identical wear mask per day) collapses MultiRocket from 0.828 to 0.662 val - the +0.17 AUROC is destroyed by breaking the value-to-clock-position assignment alone.

**Bin resolution is not the bottleneck; representation is.** Profile288 already operates at 5-minute resolution yet reaches only 0.714 (Profile24 hourly: 0.725) - per-bin means across days discard the local dilation/position patterns that convolutional kernels (MultiRocket, HYDRA) exploit on the same resolution.

**The ladder is monotone in representation complexity** - summaries (RF/linear) -> per-bin profiles -> convolutional kernels - and Combined adds a further increment over the best single representation (sec 1), positive vs Summary_RF in 10/10 allocations (sec 2 Bonferroni family significant; test corroborates, exploratory).

**Erratum robustness:** the v1->v2 zero-sentinel fix moved every arm by at most 0.004 AUROC (sec 9) and left the B40-only arms bit-identical - the headline is an artifact of neither the contamination nor its correction.

**Open question:** whether the placement signal is physiological (circadian phase/shape) or device-behavioral (wear-time routines correlated with the recorded salutation). The predeclared probes (night-only arm, activity-window exclusion, importance-by-dilation - NEXT_STEPS sec 2.1) remain the next step. Recorded salutation != biological sex/gender.

**Headroom note:** alloc-0 precalibration hit the alpha-grid boundary (1e3) for the wide blocks (MultiRocket +0.041, HYDRA +0.047 last-decade val gains) - wide arms are likely undershrunk, so the placement gap is if anything understated (extended-grid addendum, NEXT_STEPS).


## 8. Caveats and recorded errata

- **Erratum (mechanism correction):** numba's `np.random.randint` inside `_fit_biases` uses numba's **internal** RNG state, not numpy's global RNG. The plan's premise ("global RNG inside njit") is incorrect; an `@njit _nb_seed(s)` shim seeds the correct state before each `MultiRocket.fit()`. Reproducibility gates pass.

- **Erratum (environment):** the default numba threading layer `omp` segfaults when torch is loaded in the same process (duplicate libomp — pip torch + conda llvmlite). `NUMBA_THREADING_LAYER=workqueue` is the documented mitigation; prange over per-row independent writes is byte-identical at 1/4/8 threads under workqueue.

- **Plan divergence (recorded):** NaN-timezoneOffset events rejected (PLAN §3) rather than substituted as UTC (`src/sidequest/diurnal.py` convention). Zero events rejected in practice for this cohort.

- **Erratum 1 (v1 defect, fixed in v2):** unobserved bins are stored as 0.0 sentinels in `day_bins.npz` (scan_bins); v1's Profile288 `nanmean/nanstd` and per-bin fill `nanmedian` included those zeros → P288 features mixed coverage with HR (193/288 bins |mean shift| > 2 bpm; 37.7% of user-bin pairs > 5 bpm) and fill medians biased low ~2 bpm (68.4 vs 70.4 mask-aware). Fixed: mask-aware per-bin aggregation and fill (train-only medians, unchanged hygiene). Quantified per arm in §9 (v1 ↔ v2).

- **Erratum 2 (v1 defect, fixed):** `verify` CLI mode was accepted but never dispatched in `main()`; `run_verify` also could not fail. v1 determinism was established by calling `run_verify()` directly (record: `cache/verify_v1.json`); v2 dispatches `verify` and hard-fails beyond tolerance.

- **Erratum 3 (v1 defect, fixed):** v1's report §4 computed mean raw-score differences (ridge score − RF probability) — not an AUROC increment. Fixed to per-model AUROCs on matched test users (§4 of this report). No effect on arm AUROCs.

- **Determinism gate (relaxed standard, recorded):** alloc-0 reruns are not byte-identical: all arms differ at 1–2 ULP (max 2.2e-16), attributed to threaded-BLAS reduction order in the ridge solve (even pure-B40 Summary_linear shows it; MR/HYDRA transforms and all seed streams are bit-stable). Gate relaxed to max diff ≤ 1e-6 per arm — AUROC-equivalent by construction; `cache/verify.json`.

- **alpha-grid boundary (recorded):** alloc-0 precalibration selected the grid maximum (1e3) for every family in v1 and v2; the wide blocks (MultiRocket +0.041, HYDRA +0.047 val AUROC gain over the last grid decade) were still climbing - the frozen grid truncates their optima. Same protocol across arms keeps the ladder comparison fair; absolute AUROCs of wide arms likely have headroom (extended-grid addendum: NEXT_STEPS).

- **Pairing scope (recorded):** Combined − BASE⊕P40 paired on TEST predictions only — R1b saved test-only participant-level predictions in `r1b_predictions_part.csv`.

- R = 10 under-powers the primary family for true deltas ≲ 0.005; Bonferroni 95%-simult CIs are wide (R1b/R1a-AF caveat).

- Test split reported as exploratory — these participants have already supported R1a-AF and R1b model development.

- Allocation variability is not population uncertainty.

- Recorded salutation ≠ biological sex/gender.


## 9. v1 ↔ v2 — erratum 1 (zero-sentinel contamination) quantification

v1 = as-run commit `168b9d7` (defect present); v2 = this run (mask-aware fill + mask-aware Profile288). Defects 2–3 (verify dispatch, §4 pairing) have no numeric effect.

| arm | val v1 | val v2 | Δ val | test v1 | test v2 | Δ test |
|---|---|---|---|---|---|---|
| `Summary_linear` | 0.7077 | 0.7077 | +0.0000 | 0.7157 | 0.7157 | +0.0000 |
| `Summary_RF` | 0.6962 | 0.6962 | +0.0000 | 0.7031 | 0.7031 | +0.0000 |
| `Profile24` | 0.7248 | 0.7248 | +0.0000 | 0.7370 | 0.7370 | +0.0000 |
| `Profile288` | 0.7165 | 0.7140 | -0.0026 | 0.7291 | 0.7327 | +0.0036 |
| `MultiRocket` | 0.8276 | 0.8284 | +0.0008 | 0.8354 | 0.8361 | +0.0007 |
| `HYDRA` | 0.8392 | 0.8374 | -0.0018 | 0.8431 | 0.8442 | +0.0010 |
| `Combined` | 0.8496 | 0.8489 | -0.0007 | 0.8539 | 0.8565 | +0.0026 |
| `Shuffled_MR` | 0.6656 | 0.6619 | -0.0036 | 0.6657 | 0.6643 | -0.0014 |

## 10. Selective classification — coverage at per-class target precision (predeclared procedure; `selective.py`)

Thresholds on val, evaluation on test; `raw` = empirical, `cpc` = Clopper-Pearson LCB-corrected (δ=0.10). See NEXT_STEPS.md §2.6 / Step 0.5 for the predeclared procedure and the recorded CP-vs-CRC deviation.

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


### Abstention composition (Combined, 95% target)

| variant | group | mean mask coverage | mean obs bins/day | mean d_ch3000_mean | mean d_ch3000_cov_h | n |
|---|---|---|---|---|---|---|
| raw | labeled | 0.898 | 258.7 | 71.1 | 21.48 | 144 |
| raw | abstained | 0.881 | 253.6 | 74.0 | 21.06 | 428 |
| cpc | labeled | 0.903 | 259.9 | 69.7 | 21.59 | 93 |
| cpc | abstained | 0.883 | 254.4 | 73.6 | 21.13 | 526 |

Wear-coverage interaction: if abstained users are systematically lower-wear, a minimum-wear-time gate is an upstream engineering lever (NEXT_STEPS §2.5).
