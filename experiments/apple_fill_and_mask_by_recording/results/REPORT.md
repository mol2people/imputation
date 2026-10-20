# Apple channel 3000: filling, difference masking, and observation-mask representations

**Reconciled 2026-10-19 from saved predictions and metrics.** The available benchmark contains **allocation 0 only**, with eight arms, two resolutions, two history budgets, and three penalty rules: 96 configurations and 160,416 participant predictions. Earlier statements of three completed allocations, cross-allocation means/SDs, and sign agreement were unsupported by these artifacts. No model was refitted for this reconciliation.

## Scope and methods

The target is **recorded administrative salutation** (`10 → 0/F`, `20 → 1/M`), not verified biological sex or gender identity. The source-6 Apple cohort contains 11,139 participants. Allocation 0 assigns 7,797 to training, 1,671 to validation, and 1,671 to test; test contains 579 class-0 and 1,092 class-1 participants. The existing OR-Tools allocation balances salutation and demographic strata. Participants qualify with at least 14 adequate days. An adequate local calendar day contains at least 60 valid channel-3000 HR readings in at least eight distinct clock hours; valid HR is 25–230 bpm.

The same cached participant-days supply the 5-minute grid (288 bins/day) and 15-minute grid (96 bins/day). Each bin is the mean of its available readings. The binary observation mask is `M = isfinite(hr)` before filling. These masks describe observed bins, not verified off-wrist or sensor states; sub-bin timestamps and counts are absent from the caches.

`all_days` uses all adequate days (2,967,608 participant-days across the cohort). `up_to_40d` uses the earliest `min(D, 40)` adequate days per participant (432,345 participant-days), retaining the same cohort. Fourteen days is the inclusion threshold, not a history-budget arm. Full-history recording descriptors define subgroups for both budgets.

### Feature blocks and eight arms

The static reference block `B_L` has `17 + L` columns: across-day means and sample SDs of eight daily statistics, weekday-minus-weekend mean HR, and per-clock-bin observation frequencies. The eight statistics are `mean`, `median`, `sd`, `vmin`, `vmax` (bpm), `n` (valid event count), `cov_h` (union of recorded event durations, hours), and `hours` (number of occupied clock hours). Thus the reference already combines HR and recording information. It is not a physiology-only baseline.

`T(X, ΔX)` denotes MultiRocket features from a daily base sequence and its first difference: 4,704 features per branch, 9,408 per day, pooled by across-day mean and sample SD into 18,816 participant features. Daily ordering within each sequence is represented; chronological order between days is discarded. No separate `n_days` predictor is appended. History can nevertheless affect estimated distributions, recording summaries, and the finite-sample SD correction.

`S(M)` contains per-clock-bin across-day mask means and sample SDs (2L columns). For binary masks and D > 1, `SD² = D/(D−1) × mean × (1−mean)`. Conditional on D, this SD block supplies a nonlinear occupancy basis rather than independent information. `S_SD(M)` contains only the SD half.

| Arm | Definition |
|---|---|
| `H_clock` | `B_L + T(X_clock, ΔX_clock)`: observed HR retained; gaps filled with training clock-bin medians. |
| `H_global` | Same blocks with a single training global median filling all gaps. |
| `H_clean_diff` | Same clock-filled base and static features; difference is observed HR difference when both endpoints are observed and zero otherwise. |
| `H_short_interp` | Same feature blocks after linear interpolation across interior gaps whose bounding bin centers are at most 30 minutes apart; training clock medians elsewhere. |
| `M_static` | `S(M)` alone, without HR values. |
| `M_temporal` | `S(M) + T(M, ΔM)`, without HR values. |
| `H_plus_M_concat` | One Ridge fit on `B_L + S_SD(M) + T_HR + T_M`; duplicate static mask means are excluded. |
| `H_plus_M_stack` | Convex blend of separately trained `H_clock` and `M_temporal` scores. |

Interpolation bridges 1–5 missing bins at 5m and one missing bin at 15m. It uses original observed endpoints within the same day, preserves observed values, and does not bridge midnight. Matching the maximum physical span does not make intervention exposure equal across resolutions.

### Fitting and selection

Fill templates and MultiRocket bias samples use training participants only, separately for each resolution and budget. At most 4,096 sampled training days are shared across arms within a configuration. The sequence seeds follow the reference benchmark. Each feature arm fits median imputation, zero-variance filtering, and standardization on training participants, then fits Ridge on training labels. Validation and test are transformed with these fitted quantities; there is no train-plus-validation refit.

- **Primary:** `all_days`, `ridge_selected_ext`, α ∈ {100, 500, 1000, 5000, 10000, 100000}.
- **Secondary:** `ridge_selected`, α ∈ {100, 500, 1000, 5000, 10000}; `ridge_fixed`, α = 1000; capped histories.

Alpha maximizes validation AUROC; near ties under the reference `np.isclose` rule favor larger alpha. Stacking uses component scores standardized by their validation means and sample SDs, then selects `s = (1−w)s_HR + w s_mask`, with w ∈ {0, .05, …, 1}, by validation AUROC; near ties favor smaller w. Saved extended-grid weights are .20 for both all-days resolutions, .25 for capped 5m, and .20 for capped 15m. Neither alpha nor blending is selected by balanced accuracy. This benchmark saves continuous scores and does not define a new classification-threshold protocol.

The [plan](../PLAN.md) specifies a three-allocation pilot and advancement heuristic, but only allocation 0 is available here. Its appended review comments describe earlier proposals; the amended protocol and implemented definitions above govern this readout.

## Saved test discrimination

All values below are allocation-0 test AUROCs under `ridge_selected_ext`. Differences use the same test participants. They are not cross-allocation estimates.

| Arm | All days 5m | All days 15m | 5m − 15m | Up to 40d 5m | Up to 40d 15m |
|---|---:|---:|---:|---:|---:|
| `H_clock` | 0.8678 | 0.8448 | +0.0229 | 0.8168 | 0.7972 |
| `H_global` | 0.8586 | 0.8438 | +0.0148 | 0.8140 | 0.8059 |
| `H_clean_diff` | 0.8718 | 0.8538 | +0.0180 | 0.8247 | 0.7992 |
| `H_short_interp` | 0.8727 | 0.8504 | +0.0223 | 0.8246 | 0.8026 |
| `M_static` | 0.6466 | 0.6282 | +0.0185 | 0.6342 | 0.6131 |
| `M_temporal` | 0.7459 | 0.7280 | +0.0180 | 0.7107 | 0.6843 |
| `H_plus_M_concat` | 0.8723 | 0.8517 | +0.0206 | 0.8284 | 0.8122 |
| `H_plus_M_stack` | 0.8749 | 0.8517 | +0.0232 | 0.8255 | 0.8097 |

At 5m with all days, global filling reduces AUROC by 0.0092 relative to clock filling. Withholding gap-related differences increases it by 0.0040; short interpolation increases it by 0.0049. At 15m, the corresponding changes are −0.0011, +0.0090, and +0.0055. The 5m advantage remains under each of these changes; clean differences narrow it from 0.0229 to 0.0180. These are sensitivities of refitted pipelines, not an additive decomposition of predictive signal.

The mask-only representation increment is substantial: `M_temporal − M_static` is +0.0993 at 5m and +0.0998 at 15m. Observation patterns therefore discriminate recorded salutation in this cohort without HR values. This comparison adds nonlinear occupancy/gap features as well as sequence arrangement; it does not isolate temporal order.

Combining HR and mask representations yields modest increments over the mixed reference: concatenation +0.0045 and stacking +0.0071 at 5m; +0.0069 and +0.0068 at 15m. The largest all-days 5m AUROC is stacking (0.8749), but these point estimates do not establish a reliably superior arm. Including w = 0 protects validation performance under the selection criterion, not unseen test performance.

All-days performance exceeds capped-history performance for every arm and resolution here. The 5m reference difference is +0.0509. This may reflect more precise estimates and broader sampled dates/routines as well as additional physiological observations. Pooling does not identify which explanation accounts for the gain. Under capped 15m histories, global filling improves AUROC by +0.0088, reversing its all-days direction; a universal clock-fill benefit is therefore unsupported.

## Recording subgroups and uncertainty

Subgroups use training-derived terciles of full-history adequate-day count (history), 5m occupancy, adequate-day density (continuity), and daily occupancy SD. Continuity is adequate-day count divided by the inclusive calendar span between first and last adequate dates. It is distinct from within-day occupancy and history length. The crossed cells are continuity × occupancy and history × occupancy; small cells may yield unstable AUROCs.

The primary occupancy readout illustrates heterogeneity. The last two columns are 5m arm-minus-`H_clock` AUROC differences within the same subgroup, with exploratory conditional 95% bootstrap intervals in brackets.

| Occupancy tercile | Test N | Clock 5m | Clock 15m | Global − clock | Stack − clock |
|---|---:|---:|---:|---:|---:|
| low | 575 | 0.8581 | 0.8530 | +0.0092 [-0.0115, +0.0287] | +0.0108 [+0.0033, +0.0194] |
| middle | 580 | 0.8721 | 0.8316 | -0.0198 [-0.0413, +0.0016] | +0.0063 [-0.0028, +0.0151] |
| high | 516 | 0.8616 | 0.8365 | -0.0184 [-0.0427, +0.0071] | +0.0038 [-0.0073, +0.0170] |

Global filling has a positive point estimate in low occupancy and negative estimates in middle/high occupancy. All three subgroup intervals include zero; this point-estimate crossover does not establish an occupancy interaction or an individualized filling rule. Sampling uncertainty and the many inspected comparisons remain material. History capping and resolution change overall discrimination more than most filling interventions here, while filling can have larger local subgroup effects.

The focal cells were selected after inspection of the earlier ten-allocation binning study. Current subgroup comparisons are exploratory. See [subgroup_summary.csv](subgroup_summary.csv) and [paired_contrasts.csv](paired_contrasts.csv) for the reconciled allocation-0 estimates. They must not be read as three independent replications. The subgroup and contrast CSVs contain point estimates; cross-allocation SDs and unavailable allocation columns are blank. [conditional_bootstrap_contrasts.csv](conditional_bootstrap_contrasts.csv) supplies 1,000 class-stratified paired bootstrap resamples (seed 20261019), using the same sampled test participants across scores. Percentile 95% intervals remain conditional on these fitted predictors and this held-out sample; they omit model-fitting, hyperparameter-selection, and subgroup-discovery uncertainty and have no multiplicity adjustment. For context, the overall 5m stacking increment is +0.0071 [0.0015, 0.0125], the mask representation increment is +0.0993 [0.0719, 0.1285], and the reference 5m−15m difference is +0.0229 [0.0076, 0.0372]. These intervals support prioritization rather than confirmatory claims.

## Interpretation

`H_global` changes imputed levels, boundary jumps, and template slopes simultaneously. `H_clean_diff` removes gap-boundary and gap-interior values from the difference branch together, while retaining its zero pattern and the clock-filled base branch. Biases and Ridge weights are refitted under these changes. None of these contrasts identifies the physiological or behavioral cause of gaps, partitions their contributions, or establishes that sparse observations arise from exercise, charging, or off-wrist periods.

## Companion reports and provenance

[HARRELL_PROPER_SCORING_AUDIT.md](HARRELL_PROPER_SCORING_AUDIT.md) requires a separate interpretation from the ranking benchmark. Its script fits a logistic recalibration using test labels and evaluates probabilities on those same participants, then fits another logistic calibration to the resulting probabilities. Brier/log-loss are apparent performance after test-label fitting; slopes near 1 and intercepts near 0 do not validate calibration or Ridge shrinkage. This posthoc operation does not affect the benchmark AUROCs or validation-based alpha/weight selection. Held-out probability assessment needs a calibrator fitted outside the test set.

The reconciled companion exploration notes separate computed allocation-0 training descriptors from fixed narrative text and older `apple_cohort_scaling` resolution/error analyses. They provide provenance and limitations, not independent verification of current arm effects. Correlations between occupancy and observed HR cannot distinguish MAR from MNAR without assumptions about unobserved HR. Clinical dipping categories, autonomic mechanisms, and device-specific wear explanations are not established by these artifacts.

## Verification boundary

The authoritative quantitative inputs are [test_predictions.parquet](test_predictions.parquet), [benchmark_metrics.csv](benchmark_metrics.csv), and [alpha_and_weight_selection.csv](alpha_and_weight_selection.csv). The reconciliation checks saved-score coherence and derives companion summaries without fitting the prediction models. The upstream reference permits direct `H_clock` score parity checks under original-grid and fixed penalties; those checks do not validate the other arms' mechanisms. No allocation-1/2 performance, independent cohort generalization, or valid held-out probability calibration is established here.

Numeric summaries can be regenerated with [09_reconcile_saved_summaries.py](../scripts/09_reconcile_saved_summaries.py); see [RECONCILIATION_PROVENANCE.json](RECONCILIATION_PROVENANCE.json). Original script `03_summarize_subgroup_contrasts.py` expects three allocations and currently fails on the available one-allocation input; after future runs it can overwrite this prose. Original exploration/calibration scripts `04`–`08` also retain report-writing sections with withdrawn claims. Do not use those report generators to recreate the reconciled documents.

The conditional intervals can be regenerated with [10_bootstrap_saved_contrasts.py](../scripts/10_bootstrap_saved_contrasts.py); see [BOOTSTRAP_PROVENANCE.json](BOOTSTRAP_PROVENANCE.json). This resamples saved scores and does not refit models.
