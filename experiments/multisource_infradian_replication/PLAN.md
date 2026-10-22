# Multisource infradian age-dependency expansion

Date: 2026-10-22. Status: implementation complete in `multisource/` and the staged CLI; bounded validation runs passed, and the multisource default now uses analytical-gradient L-BFGS-B with adaptive screening. The historical Nelder–Mead path remains selectable with `MULTISOURCE_REML_OPTIMIZER=nelder`. Computational architecture and `.npz` curve storage optimizations are implemented and numerically verified (see the implementation status under "Computational architecture and storage efficiency" and the 2026-10-22 update); the full `B=199` endpoint run with `n_boot=1,000` completed on 2026-10-22 (65,449 rows; 65,443 converged; 28m23s), and six degenerate rows remain flagged for user inspection. The prespecified `2,000`-replicate bootstrap was exercised in bounded validation, not the full run. This plan extends the historical source-specific analysis while keeping its results and protocols intact, with the one documented optimizer-induced endpoint change described below.

## Objective and source scope

Assess age-patterned 25–35-day periodic excess in daily heart-rate endpoints, separately by source and endpoint. The planned sources are Fitbit (2), Garmin (3), Apple (6), Samsung (7), Nokia (9), Google Fit (13), and Oura (19). Polar (4) is explicitly excluded. Unknown source IDs 38, 46, and 48 remain inventory-only until their source mapping, endpoint semantics, and provenance are resolved; any inclusion requires a dated plan addendum before endpoint analysis.

There are three primary analyses. They have different ascertainment and estimands and must not be pooled or described as interchangeable replications. Other exported daily-like fields such as channels 3001/3002 are not primary endpoints; they may be proposed in a later addendum only after their source-specific value definition and aggregation are prespecified and supported by provenance.

| Analysis | Endpoint and cohort gate | Estimand and role |
|---|---|---|
| A. Dense observed-bin HR | Source-specific channel 3000; retain a local date only with at least 120 originally observed five-minute bins; daily HRp10 is the 10th percentile of the finite bin means. HRp5 and HRp1 are optional matched sensitivities. | Source-specific age association in the dense-recording cohort. This continues the existing observed-bin endpoint. |
| B. Daily type 65 RHR | One resolved, finite daily value in [25, 230] bpm per literal exported calendar date. No channel-3000 or within-day density prerequisite. | Source-specific age association among participants qualifying on this endpoint's own date coverage. This is a separately ascertained supplier endpoint, not a denser version of Analysis A. |
| C. Daily type 66 sleep HR | Same independent daily-value rule as B, analyzed separately wherever type 66 is available. | Source-specific sleep-HR age association, conditional on endpoint availability and support. It is not RHR and is not combined with type 65. |

Do not concatenate device trajectories or combine sources into one HR series. A participant can contribute separate source-specific records. The analysis key is `(user_id, source, endpoint, protocol, window_number)`; source-specific inference treats each record once, and any explicitly reported cross-source summary must cluster uncertainty by global `user_id`. Multiple endpoints from one source remain separate analyses.

## Inventory and provenance gate

The current raw inventory is a planning inventory, not an eligibility result. These counts must be refreshed from the raw inputs and mapping before implementation. Counts refer to source-specific daily observations and are not additive across sources or evidence of a qualifying 180-day window.

| Source | Daily endpoint inventory | Participants with any endpoint | At least 90 dates | At least 180 dates | Median dates | Planned status |
|---|---|---:|---:|---:|---:|---|
| Fitbit (2) | Type 65 | 4,956 | 4,468 | 3,967 | 494 | Include, subject to provenance and eligibility checks |
| Garmin (3) | Type 65 | 6,136 | 5,704 | 5,334 | 607 | Include as a new daily endpoint arm; preserve existing analysis |
| Apple (6) | Type 65 | 14,499 | 11,852 | 10,282 | 350 | Include as a new daily endpoint arm; preserve existing analysis |
| Samsung (7) | Type 65 | 2,145 | 1,364 | 824 | 135 | Include, subject to provenance and eligibility checks |
| Nokia (9) | Type 65 | 1,527 | 1,043 | 806 | 196 | Include, subject to provenance and eligibility checks |
| Google Fit (13) | Type 65 | 1,189 | 390 | 157 | 49 | Include; likely limited window eligibility, determine from actual dates |
| Fitbit (2) | Type 66 | 2,269 | 0 | 0 | 17 | Include in inventory; current dates span only 2020-04-12 to 2020-05-10, so no `expanded90` window is currently supported |
| Oura (19) | Type 66 only | 134 | 110 | 94 | 284.5 | Include type 66 as a separate endpoint; no type 65 arm is currently inventoried |
| Polar (4) | None in current inventory | — | — | — | — | Excluded by scope |
| Unknown (38, 46, 48) | Unresolved | — | — | — | — | Inventory and provenance review only |

Use `mapping/pg_daily_sources.csv` and `mapping/pg_daily_value_types.csv` in the upstream `datenspende_epoch2daily` project for source and value-type mappings, respectively. Keep its Thryve source namespace distinct from other source namespaces; numeric IDs are not interchangeable without an explicit mapping. Verify the exported field, units, daily-value definition, source identity, date semantics, and participant linkage for each endpoint. A code label is not evidence of vendor filtering or physiological equivalence.

For the daily arms, preserve the exported calendar date as the date key under the existing convention; do not apply a timezone offset to an already-calendar-date value or shift it twice. Existing use of timezone offsets 0, 60, 120, or null with a Berlin assumption is a convention, not universal validation of all new sources. Audit rare or unexpected offsets separately and document any source-specific date policy before selecting windows. For channel-3000 extraction, retain the existing per-record timezone-offset local-date construction and source-specific binning rule. Do not silently apply the dense-bin date policy to daily records. The [25, 230] bpm daily-value range is a fixed provisional QC convention borrowed from dense HR filtering; it is not a validated vendor rule. Verify units and endpoint provenance before applying it. If that cannot be established, stop that endpoint arm. Report exclusions under this gate; do not tune the range after inspecting spectra.

Collapse duplicate rows only when all endpoint-relevant values are exactly identical, and report raw duplicate counts. If a participant has conflicting values for the same source and date, flag the conflict and resolve it only from documented source chronology/provenance; otherwise exclude that date with a reason. Do not average conflicting daily values. No endpoint is imputed. Record source, endpoint, raw file identity, mapping/version, date convention, unit checks, duplicate/conflict counts, missingness, and input hashes.

## Eligibility and protocol rules

Retain the existing primary `expanded90` protocol and `balanced90` reference. Apply them independently to each source and endpoint using that arm's own observed qualifying dates. Do not require an old cohort label, single-source membership, BMI, or availability of another endpoint. Counts of 90 or 180 recorded dates are only inventory screens; eligibility requires constructing the local/exported calendar dates and applying the full window rule.

| Rule | Dense observed-bin HR (A) | Daily type 65/66 (B/C) | Status and rationale |
|---|---|---|---|
| Endpoint date qualification | At least 120 of 288 original five-minute bins finite after the existing channel-3000 value and timestamp filters. | Exactly one resolved finite value in [25, 230] bpm for a source/date. No bin-density or wear-duration rule is inferred. | **Amended by endpoint.** The existing 120-bin gate remains unchanged for A. B/C use independent daily ascertainment because the inputs are daily summaries. A daily value cannot be said to satisfy the within-day gate. |
| Window | 180 calendar days, six consecutive 30-day blocks. | Same calendar window and blocks, based only on that daily endpoint's qualifying dates. | Existing window length preserved; qualifying dates are endpoint-specific. |
| `expanded90` | At least 10 qualifying dates in each block and at least 90 in total; deterministically select exactly 90 using the existing allocation/seed convention. | Same 10-per-block, 90-total rule and deterministic selection, applied to daily endpoint dates. | Existing primary protocol retained. |
| `balanced90` | At least 15 qualifying dates in every block; select 15/block. | Same requirement and selection on the daily endpoint's own dates. | Existing reference retained. |
| Window choice | Earliest qualifying window for each source and protocol. | Earliest qualifying window independently for each source, endpoint, and protocol. | Required; choose before inspecting spectra or age associations. |
| Repeat window | Earliest qualifying start at least 180 days after the first window start, same source and endpoint. | Same rule, re-evaluated on that endpoint's dates. | Secondary longitudinal availability only; actual calendar time is not compressed. |
| Daily coverage | Original date-level bin mask and selected dates retained. HRp5/p1 use the same people, dates, and bins as HRp10. | One value per selected calendar date; no within-day coverage fraction is fabricated. | No imputation or invented density. |
| Nuisance adjustment | Intercept, linear elapsed calendar time, weekday indicators, and four six-hour clock-coverage fractions. | Intercept, linear elapsed calendar time, and six weekday indicators; no clock-coverage covariates. | **Amended by endpoint.** Daily summaries have no within-day mask from which to derive the fractions. The adjusted estimands differ. |

For each source/endpoint, enumerate candidate starts from sorted unique qualifying dates in ascending order. Windows are half-open `[s, s+180)` calendar days, with blocks `[s+30b, s+30(b+1))`, `b=0,...,5`; choose the earliest qualifying start independently for each source, endpoint, and protocol. For `expanded90`, use the existing six-block allocation rule: initialize each block quota to 10 and allocate the remaining 30 one at a time to the currently smallest quota among blocks with remaining capacity, breaking ties by block order. Sample uniformly without replacement from sorted eligible dates within each block. Use `numpy.random.Generator(PCG64(seed))`, with `seed` the unsigned little-endian integer from the first eight SHA-256 bytes of `20261020|infradian_age|{source}|{user_id}|{window_start_day}|dates_block_{b}`; generate one permutation and take its first `q_b` entries, then sort selected dates. Persist selected dates, quotas, capacities, window starts, and hashes. For `balanced90`, select 15 eligible dates per block with the same deterministic sampling convention. Do not reuse an Analysis A window or date list as the independent daily-arm selection.

For female curves, include only unambiguous salutation 10 and valid birth-year records. For sex comparisons, use only unambiguous salutation 10 and 20; exclude missing, conflicting, or other salutation values and report counts. These are administrative analytical labels and do not establish reproductive anatomy or clinical cycle status. Do not infer salutation from other fields. Age is derived from the birth year using the existing December-1 convention at the selected window start: `age = start.year - birth_year - 1 + I(start.month == 12)`. This is a year-resolution convention, not exact age. Exclude missing, nonintegral, future birth years and derived ages outside the plausibility range [0, 120]; do not impute age or use category midpoints. The primary age-curve evaluation grid is 18–80 years, distinct from the [0, 120] data plausibility check. Preserve the RCS knots [25, 42, 52, 65] and birthday-convention sensitivities from the [historical protocol](../infradian_age_dependency/PLAN.md). Age support and fitted endpoint counts must be reported separately from date-window eligibility.

For female age curves, fit on all valid finite endpoint rows and display only the observed age intervals in five-year bands with at least 30 women. Fit the shared-sex interaction model on shared-support bands with at least 30 women and 30 men, requiring at least three such bands and a full-rank design. These are support rules, not reasons to relax date coverage. A nonlinear inflection is estimable only if its fitted location is inside supported data and not a boundary artifact; otherwise report it as undefined or unsupported. If a source/endpoint cannot support the prespecified model, report it as non-estimable for that contrast and show eligibility/support counts; do not weaken thresholds after examining spectra. Do not invent a single adult-age cutoff.

## Estimands and matched comparisons

For each source and endpoint, calculate the primary spectrum on the selected actual calendar dates. The primary participant endpoint remains the normalized band-integrated **signed null-adjusted excess partial R-squared** over periods 25–35 days:

`D_i(f) = P_i(f) - mean_b(P_i,null,b(f))`

`Q_i = integral_[1/35,1/25] D_i(f) df / (1/25 - 1/35)`

Use trapezoidal quadrature on the fixed frequency grid including exact band boundaries. `Q_i` is not the maximum peak. The discrete peak excess `E_i2 = max_T P_i(1/T) - mean_b max_T P_i,null,b(1/T)`, for `T = 25, 25.5, ..., 35` days, is secondary and must be labeled separately. For dense observed-bin HR, retain the existing nuisance design (intercept, linear elapsed calendar time, weekday indicators, and four six-hour clock-coverage fractions). Daily type 65/66 values do not contain within-day masks, so prespecify a separate daily-arm nuisance design before computing endpoints: intercept, linear elapsed calendar time, and weekday indicators; omit clock-coverage fractions and document this endpoint-specific difference. Do not claim exact method equivalence across these endpoint arms. Preserve the existing RCS age models, four chunk tests, HC3 chunk-test covariance, and participant bootstrap where support permits. Use a source- and endpoint-specific two-component continuous-time REML null fitted on actual dates; refit it for every endpoint arm. Retain the existing covariance parameterization, convergence reporting, `B=199` null draws, and 2,000 participant bootstrap resamples. Do not change to a multi-harmonic model or infer biological cycle prevalence from null rejection.

The independently selected cohorts in A, B, and C answer separate source/endpoint age-association questions. Their effect estimates may be compared descriptively, but their differences do not estimate a measurement effect because ascertainment, calendar coverage, and endpoint definition differ. In particular, a daily type 65 result cannot be called a replication of dense-bin HRp10 without qualification.

Matched endpoint comparisons are secondary and conditional. First restrict to participants with the same source and both endpoint definitions available on the same 90 dates selected by the dense channel-3000 `expanded90` (or `balanced90`) window. Report this exact-date matched cohort separately, including retention and age-by-salutation support. In the primary paired contrast, recompute both endpoint series under the same reduced nuisance design (intercept, elapsed-time slope, weekday indicators) and fit a separate endpoint-specific null for each. Save new matched outputs; do not reuse saved coverage-adjusted HRp10 as if it had the same adjustment. The coverage-adjusted HRp10 estimate on this matched subset may be shown separately as a named adjustment sensitivity, not as the primary paired contrast.

Paired comparisons must match participants and dates before fitting, and must explicitly require:

1. **A common participant set with finite endpoints and converged null fits in every compared arm.** Participants whose null fit fails or whose endpoint is non-finite in any arm are excluded from the paired comparison and reported as fit failures.
2. **Age support recalculated on that common set.** Age-by-salutation support for the paired analysis is the intersection of the per-arm supports, not the union of either arm's age range.
3. **Identical bootstrap participant draws across arms.** Resampling must use the same participant IDs (bootstrap clusters) in every arm; a different draw in each arm violates pairing.
4. **Arm-specific failure/retention counts and paired-difference confidence intervals.** Each arm's fit failures, retention, and age support must be reported separately, along with the paired-difference confidence intervals computed on the common set.

These four requirements together ensure that differential fit failures, differential retention, or independent resampling do not silently compromise the paired comparison. The paired comparison is valid only when all four are satisfied; otherwise the comparison is conditional on the common set and must be labeled as such.

If exact-date matching is too sparse for the planned contrast, a separate common-date reselection may be considered: determine common dates using only source-specific endpoint availability and fixed date-qualification rules, before examining HR values or spectra; select the same 90 dates for both arms under the stated block quotas; and apply the same participant set and dates to both arms. Label this as a distinct conditional analysis. Do not compare independently selected dates as if paired, or silently change eligibility to improve overlap. Report endpoint agreement separately from spectral-estimand contrasts; correlation alone does not establish agreement or accuracy.

## Computational architecture and storage efficiency

To allow the full \(B=199\) null and 2,000-replicate bootstrap pipeline to scale efficiently across all arms while preserving numerical fidelity and historical reproducibility:

1. **Bootstrap curve array storage (`.npz`)**:
   Individual participant-resampled age curve draws are not expanded into long-format relational tables (which generate multi-gigabyte DataFrames and extreme disk/memory footprints over 249 evaluation ages). Instead, resampled curve predictions for each arm, metric, and scope are persisted in compressed NumPy array archives (`bootstrap_curves.npz`) as float32 matrices of shape \((N_{\text{boot}}, N_{\text{ages}})\), accompanied by the `age_grid` coordinate vector. Per-key metadata is recorded in `bootstrap_curves_manifest.csv` rather than duplicated as arrays inside the archive. Derived tabular summaries—pointwise and simultaneous confidence bands (`confidence_bands.parquet`), point curves (`curves_point.parquet`), transition metrics (`transition_metrics.csv`), and chunk Wald distributions (`bootstrap_chunk_summary.csv`)—remain tabular artifacts.

2. **Effort-optimal computational acceleration**:
   Optimize execution by targeting high-leverage bottlenecks without altering the mathematical specifications or historical convergence paths:
   - **Core REML engine**: The historical continuous-time two-component REML Nelder-Mead optimizer remains the default for upstream experiments and is selectable in the multisource pipeline with `MULTISOURCE_REML_OPTIMIZER=nelder`. The multisource default now uses the analytical-gradient L-BFGS-B path described in [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md), with the same variance-share and \(\phi_2\) grid, adaptive screening, two low-\(\phi_1\) anchors, a boundary-rescue start, and a historical Nelder–Mead continuity fallback when multi-start L-BFGS-B exposes alternative basins or selects a high-\(\phi_1\)/large-\(v_2\) boundary fit. Numerical continuity is measured rather than assumed; see the 2026-10-22 update below.
   - **Invariant design precomputation in bootstrap**: Precompute the complete restricted cubic spline design matrix \(X \in \mathbb{R}^{N \times p}\) once per arm; obtain bootstrap resample matrices via index slicing \(X_b = X[\pi_b, :]\) rather than re-evaluating Stone-Koo spline basis functions in Python across 2,000 draws.
   - **Omission of unneeded per-draw standard errors**: Discontinue pointwise HC3 standard error calculations on the 249 grid points inside individual bootstrap iterations, as simultaneous confidence bands derive their studentizing scale directly from the empirical across-draw bootstrap variance. Per-draw HC3 covariances are retained because the four chunk-Wald bootstrap distributions require them.
   - **Vectorized spectral and quadrature evaluation**: In local daily-arm processing, replace frequency-by-frequency Python loops with vectorized 2D array expressions for FWL operator construction, partial \(R^2\) evaluation, and multi-replicate trapezoidal quadrature.
   - **Orthogonal projection via QR (evaluated, not adopted)**: A pivoted-QR nuisance basis and a `null_space` residual basis were implemented and measured. They save only ~0.14% of a REML-dominated task but rotate the basis enough to shift \(Q_i\)/\(E_{i2}\) by ~\(10^{-8}\) and flip the discrete `phi2` label when \(v_2\approx0\). The original Gram-Schmidt + `eigh` path reproduces historical endpoints to machine precision and was retained.
   - **Parallelized matched cohort reading**: Parallelize raw 5-minute cache decompression and percentile evaluations across worker processes rather than executing serially on the main thread.

### Implementation status (2026-10-21)

The architecture above is implemented in `multisource/`:

- **Invariant design and draws.** `bootstrap.bootstrap_frame` builds the female-only (\(n \times 4\)) and sex-interaction (\(n \times 8\)) designs and the evaluation-grid designs once per arm. A draw is the row slice \(X[\pi_b, :]\); `draws[b, j]` is the replacement row for participant row \(j\) within its age-band × sex stratum. Paired arms reuse the same `draws` matrix.
- **Storage schema.** `bootstrap/bootstrap_curves.npz` contains `age_grid` and one float32 \((N_{\text{boot}}, N_{\text{ages}})\) array per key `<source>|<endpoint>|<protocol>|<window_index>|<metric>|<scope>`. `bootstrap_curves_manifest.csv` records key metadata (`n_boot`, `n_ages`). Derived tables are `curves_point.parquet`, `confidence_bands.parquet` (`point`, `lo`, `hi`, `boot_se`, `hc3_se`, `sim_lo`, `sim_hi`, `sim_critical_value`), `bootstrap_chunk_summary.csv`, and `transition_metrics.csv`.
- **Scope labels.** The female-only model writes `female`. The sex-interaction model writes `sex_female` and `sex_male`; these exact labels are what `matched._paired_difference` selects. The historical `models/curves.parquet` retains `female`/`male` for the sex model; the two artifacts require an explicit scope mapping before row-wise comparison.
- **Draw-level SEs.** Pointwise HC3 SEs are computed only for the observed fit. HC3 covariances are still computed per draw because the chunk-Wald bootstrap distributions require them.
- **Endpoint algebra.** `endpoints.py` uses a vectorized FWL operator build, vectorized spectra, and batched quadrature. The daily nuisance basis and residual basis retain the original Gram–Schmidt + `eigh` construction: a pivoted-QR / `null_space` replacement was measured, found to perturb the REML optimizer path (~\(10^{-8}\) in \(Q_i\), discrete `phi2` label flips when \(v_2\approx0\)), and reverted. The upstream `profile_ar1_mle` and null-simulation functions are imported unchanged; `fit_two_component_reml` remains the historical fitter, while the multisource pipeline injects `multisource/reml.py` through `task["reml_fitter"]`.
- **Matched I/O.** `exact_date_cohort` decompresses participant caches in the worker pool; `workers=1` and `workers>1` produce identical outputs.
- **Verification.** All 25 tests pass. A \(B=8\) stage-5 smoke run over 64 arm–metric jobs reproduced stage-4 point predictions and SEs to float32 precision and stage-4 observed Wald statistics exactly. On 120 historical daily tasks at \(B=30\), the retained endpoint path reproduced \(Q_i\), \(E_{i2}\), \(A_{\text{hat}}\), `sigma_resid`, `p_value_discrete`, `peak_period`, and the fitted `phi2` label to machine precision (max \(|\Delta Q_i| = 9.7\times10^{-17}\)). Profiling attributed 99.4% of the historical daily task to REML; the new default L-BFGS-B path is 2.3× faster on the same 120 tasks, with all 120 tasks within `|ΔQ_i| < 1e-5` (max `2.5e-6`) and 13/120 tasks using the historical-solver continuity fallback. The full \(B=50\), `limit=2000` comparison is less exact: 18,623/18,696 endpoints satisfy `|ΔQ_i| < 1e-5` (99.61%); 73 change above `1e-5`, 43 p-values change, and the new fitter finds a lower NLL on 72 tasks while being worse by more than `1e-6` on 16 (max `3.94e-5`). See [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md) §7.2 for the level-by-level results and the strict-parity caveat. The final second-wave bounded run, which adds Stage-7 fitter injection and the Cholesky/precomputed-basis changes while retaining historical tolerances and low-`phi1` anchors, completed in 6m39s with Stage 3 at 210.33s (2.63×). Relative to the first-wave run, it changes `Q_i` by at most `1.89e-6` (0/18,696 above `1e-5`), changes no p-values, and leaves the 407 continuity fallbacks unchanged; see §7.3. The full \(B=199\) endpoint run is reported under "Full B=199 primary-window run (2026-10-22)" below.

### REML optimizer and Stage 3 queue update (2026-10-22)

The multisource default now uses `multisource/reml.py`:

- **Optimizer.** Analytical-gradient L-BFGS-B, bounded `theta=(log s1, logit(phi1/0.95), log s2)`, six canonical starts, adaptive central screening, and a low-`phi1` boundary-rescue start. The historical fitter remains available with `MULTISOURCE_REML_OPTIMIZER=nelder`; `MULTISOURCE_REML_SCREENING=0` disables adaptive screening.
- **Gradient check.** Central finite differences on a synthetic 90-day task agree to max absolute `9.2e-8` and relative `5e-10`.
- **Measured speed.** On 120 source-2 RHR `balanced90` tasks at \(B=30\), wall time falls from 24.6 s to 10.7 s (2.3×) in a paired run under concurrent load. The new fit is never worse in NLL by more than `4.4e-7`; all 120 tasks satisfy `|ΔQ_i| < 1e-5` (max `2.50e-6`). Twenty-nine `phi2` labels change, all where `v2` is at the numerical floor; p-values are unchanged. The low-`phi1` anchors and the 13/120 historical-solver continuity fallbacks are what maintain endpoint continuity. On the final full `B=50` run, Stage 3 is **2.63×** faster, but 73/18,696 endpoints change \(Q_i\) above `1e-5`; strict parity needs the historical fitter. The second-wave changes are limited to 8.1–8.4; 8.5 and 8.6 were rejected after full-run tests, so the unconditional low-\(\phi_1\) anchors and historical tolerances remain.
- **Persistent queue.** `multisource/stage3.py` runs all daily and channel arms on one `ProcessPoolExecutor` and consumes ordered results, writing each arm with the historical row order. `scripts/03_compute_endpoints.py` builds the unified job list.
- **Cache fix.** `build_caches` now uses `np.nanpercentile` over 288-bin rows; the first implementation used `np.percentile` and produced NaN `hrp10` whenever a day had fewer than 288 observed bins. The fallback path remains for historical caches without `hrp10`. A regression test covers a 120-bin day with NaNs.
- **Thread cap.** `multisource/runtime.py` is imported before NumPy/Pandas in every entrypoint. This environment's NumPy is Accelerate-linked, so macOS uses `VECLIB_MAXIMUM_THREADS=1` and `OMP_NUM_THREADS=1`; OpenBLAS/MKL variables are not set on Darwin.
- **Full-run check.** The final clean \(B=50\), `limit=2000` shadow run in `artifacts/runs/b50_shadow_secondwave_final_20261022` completed in 6m39s, with Stage 3 at `210.33 s` versus `553.65 s` for `artifacts/runs/b50_limit2000_20261021` (2.63×); the first-wave reference is `b50_shadow_lowphi_20261022` at 6m47s. All 18,696 endpoints converged; keys, columns, per-arm order, and global row order match. Endpoint parity is not exact: 73/18,696 rows (0.39%) change \(Q_i\) by more than `1e-5` (max `1.55e-2`), 43 p-values change, and 2,919 `phi2` labels flip, almost all at the `v2` floor. The new fitter finds a lower NLL on 72 tasks and is worse by >`1e-6` on 16 (max `3.94e-5`); 407 tasks use the historical continuity fallback. Models, bootstrap curves, and matched daily values remain close (details in [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md) §7.3). Strict historical parity requires `MULTISOURCE_REML_OPTIMIZER=nelder`.

### Full B=199 primary-window run (2026-10-22)

The unbounded primary-window run used:

```bash
python scripts/run_all.py \
  --run-dir artifacts/runs/full_b199_boot1000_20261022 \
  --sources 2 3 6 7 9 13 19 \
  --protocols expanded90 balanced90 \
  --workers 9 \
  --null-replicates 199 \
  --n-boot 1000 \
  --seed 20261020
```

No `--limit` was applied. The run completed in **1703.04 s (28m23s)** with
**65,449** Stage-3 endpoint rows: **43,855** daily and **21,594** channel
tasks. **65,443** rows converged. Stage 1b built 22,345 channel caches
(about 10m48s), Stage 3 took about 11m32s, Stage 4 fitted 104 models, Stage 5
completed 96 bootstrap jobs with `status=ok`, Stage 6 produced 16 matched
comparisons, and `REPORT.md` was written. The run directory is 727 MB.

The 18,696 endpoint rows shared with the bounded
`b199_limit2000_boot5000_20261022` run reproduce exactly for `Q_i`,
`Q_i_mc_se`, `E_i2`, `A_hat`, `sigma_resid`, and `p_value_discrete` (zero
difference). This verifies that the unbounded run did not change endpoint
computation for the overlapping tasks.

On the shared bounded 18,696-task cohort, `B=199` versus `B=50` gives median
\(|\Delta Q_i| = 1.88\times10^{-3}\), median `Q_i_mc_se` ratio
`B=50/B=199 = 1.98` (approximately \(\sqrt{199/50}\)), and endpoint-level
\(p<0.05\) counts of 1,445 versus 1,268. The 256 bounded group chunk tests are
stable: 150 versus 151 significant at \(p<0.05\), with 3 disagreements. The
full cohort is larger, so its 104 models and 304 chunk tests are not directly
comparable to the bounded cohort; those differences are not evidence about the
B-level comparison.

Six full-run rows are `rss0_degenerate` with `null_status=failed`: source 2
user `1124280` (RHR, both protocols) and source 6 user `1222449` (RHR and
HRp10, both protocols). They are excluded from the converged count and must
not be treated as converged. The user will inspect the source data; no
remediation, exclusion, or rerun is planned for this run. See
`artifacts/runs/full_b199_boot1000_20261022/RUN_NOTES.md`.

The full run used \(N_{\text{boot}}=1{,}000\), not the prespecified 2,000
resamples; bootstrap precision is correspondingly lower than the 2,000-resample
protocol. Exact-date matched comparisons are non-empty only for
sources 3 and 6; source 9 has zero paired rows, and sources 2, 7, 13, and 19
have no exact-date complete cases. The default L-BFGS-B optimizer remains in
use; strict historical parity still requires `MULTISOURCE_REML_OPTIMIZER=nelder`.

## Staged work and reporting

1. **Inventory and provenance:** Refresh source × endpoint counts and verify mappings, units, date semantics, duplicate policy, valid ranges, and participant linkage. Resolve unknown source IDs only through documented provenance. Stop any unresolved arm at this stage.
2. **Eligibility manifests:** Construct source- and endpoint-specific qualifying dates and deterministic `expanded90`/`balanced90` windows. Report date counts, qualifying windows, age metadata, age-by-salutation support, and exclusions by reason. Confirm that daily arms did not inherit the channel-3000 density gate.
3. **Endpoint and null computation:** Compute daily endpoints without imputation; retain masks and selected-date hashes. Fit endpoint-specific nuisance models and two-component nulls, recording numerical convergence and failures. Reuse historical Apple/Garmin artifacts only when endpoint, dates, hashes, method, and null match exactly; otherwise leave historical results untouched and compute in a separate run namespace.
4. **Primary age analyses:** Estimate source × endpoint age curves for each estimable arm. Keep daily65, daily66, and channel-3000 HRp10 results separate. Report `Q_i` as primary, `E_i2` as secondary, sample/support counts, endpoint distributions, null failures, and uncertainty.
5. **Matched sensitivities:** Run exact-date endpoint contrasts where feasible, followed by the separately labeled common-date fallback only if necessary. Analyze HRp5/p1 only as matched sensitivities within Analysis A. Keep all matched analyses distinct from independent-cohort analyses.
6. **Reporting:** Save manifests, source/endpoint provenance, date hashes, model/null diagnostics, result tables, and figures in a new run-specific namespace. Do not overwrite historical reports, caches, or artifacts. State which arms were estimable and which stopped at provenance, eligibility, or support gates.

Interpret age associations as observational and source-specific. Do not infer menopause onset, individual cycle status, vendor algorithm accuracy, endpoint equivalence, or causal within-person decline from these analyses. Any single-period decline versus noise analysis remains secondary and noncausal. Existing Apple/Garmin results and their interpretation remain historical context; this plan does not revise or replace them.
