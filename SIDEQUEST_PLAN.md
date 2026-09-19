# Side-quest: strict-coverage per-source cohorts — any-core eligibility and source-specific RF modelling (sq)

> **Status:** exploration protocol only. Updating this document does not construct cohorts, extract features, allocate splits, fit models, or run bootstrap inference.

## 1. Frozen estimand and eligibility rules

- **Outcome:** recorded `salutation` 10 vs 20 only (`y = 0` for 10, `y = 1` for 20); code 30 remains excluded.
- **Allowed epoch sources:** `{3, 6, 7, 9, 13}` (v2 exclusions: 2, 4, 19, 38, 46, 48).
- **Cohort base:** v2-eligible participants with exactly one allowed source among retained valid core-epoch events (`events > 0`). This is an epoch-source definition, not a claim that every raw or daily record has one source.
- **Core channels:** 3000, 3001, and 3002. Days are never pooled across channels. A participant qualifies if **any one channel** passes the complete longitudinal rule below.
- **Strict day:** for a given participant and channel, `strict_day = 1[cov_s >= 43,200]` using the already-materialized `artifacts_v2/epoch_days.parquet` statistic.
- **Frozen `cov_s` semantics:** `epoch_days` assigns an interval to the local date derived from its start and unions its full UTC interval without clipping at local midnight. Consequently, some `cov_s` values exceed 24 h. This side-quest freezes that artifact semantics rather than silently redefining coverage; counts and examples of `cov_s > 24 h` must be reported by source and channel.
- **Weekly grid:** for each `(user, channel)`, derive the first and last observed ISO-week Monday from **all** valid `epoch_days` rows, then create a dense sequence of Mondays over that observed weekly span. Internally absent weeks are explicit zeroes. Do not manufacture candidate windows before the first or after the last observed week.
- **Adequate week:** an ISO Monday–Sunday week is adequate iff it contains at least 5 strict days for that channel.
- **Passing window:** a candidate contains 13 consecutive week starts and is wholly within the observed weekly span. It passes iff it has at least 11 adequate weeks and at least 65 strict days in total. Strict days in all 13 weeks, including inadequate weeks, contribute to the total.
- **Calendar endpoints:** if the first Monday is `m`, the inclusive window is `[m, m + 90]`; it always contains exactly 91 local calendar dates. The thirteenth Monday is `m + 84`.
- **Selected window:** rank all passing `(channel, window)` candidates by (1) descending total strict days, (2) ascending window start, and (3) fixed channel priority `3000 < 3001 < 3002`. Store all channel-level pass flags as well as the selected channel and window statistics.
- **Event validity:** retain the v2 HR range 25–230 bpm and its timezone-derived local date/hour rules.

Formally, for dense week grid $w$,

\[
d_{u,c,w} = \sum_{t \in w} \mathbf{1}(\mathrm{cov}_{u,c,t} \ge 43200),
\qquad
a_{u,c,w} = \mathbf{1}(d_{u,c,w} \ge 5).
\]

A start Monday $m$ passes for channel $c$ when

\[
\sum_{j=0}^{12} a_{u,c,m+7j} \ge 11
\quad\text{and}\quad
\sum_{j=0}^{12} d_{u,c,m+7j} \ge 65.
\]

## 2. Cohort construction and frozen-count checks

- **Inputs:** `artifacts_v2/cohort_manifest.parquet`, `artifacts_v2/epoch_sources.parquet`, and `artifacts_v2/epoch_days.parquet`.
- **Configuration guard:** side-quest entry points must explicitly use the v2 artifact directory and v2 exclusions; the repository default is v1.
- **No full raw epoch re-scan for eligibility:** the single-source test and `epoch_days` were built from the same filtered event universe. A targeted raw mini-scan is still needed for window-restricted diurnal features (§3.B).
- **Procedure:**
  1. Restrict to v2-eligible participants and determine the sole retained epoch source from `epoch_sources(events > 0)`.
  2. For every `(user, channel)`, build the dense bounded weekly grid described in §1 from all observed day rows.
  3. Enumerate 13-week candidates and retain those meeting both passing criteria.
  4. For every user/channel, record observed-week bounds, candidate count, pass flag, best-window statistics, and one deterministic reason (`no_channel_rows`, `span_lt_13_weeks`, `max_adequate_weeks_lt_11`, `max_strict_days_among_week_qualifying_lt_65`, or `pass`, in that order).
  5. Apply the frozen ranking across windows and channels; record `qualified_channels`, `selected_channel`, `window_start`, inclusive `window_end`, adequate-week count, and strict-day count.
  6. Export all passing source cohorts for audit and apply the source-level modelling-eligibility rule below.

With the current frozen v2 artifacts, the implementation must reproduce the following any-core union counts exactly; otherwise stop and report input or semantic drift:

| sole epoch source | single-source v2-eligible | any-core strict pass | sal10 / sal20 | modelling disposition |
|---|---:|---:|---:|---|
| 3 (Garmin) | 4,410 | **3,848** | **1,384 / 2,464** | model |
| 6 (Apple) | 11,143 | **4,442** | **1,372 / 3,070** | model |
| 7 (Samsung) | 1,498 | **540** | **208 / 332** | model |
| 9 (Nokia) | 396 | **9** | **4 / 5** | audit only |
| 13 (GoogleFit) | 543 | **41** | **5 / 36** | audit only |

The corresponding channel-level pass counts are diagnostic and are not additive because participants may pass multiple channels:

| sole epoch source | ch3000 | ch3001 | ch3002 | any core |
|---|---:|---:|---:|---:|
| 3 | 3,847 | 2,148 | 3,181 | 3,848 |
| 6 | 28 | 105 | 4,363 | 4,442 |
| 7 | 1 | 0 | 539 | 540 |
| 9 | 0 | 0 | 9 | 9 |
| 13 | 19 | 1 | 26 | 41 |

Coverage eligibility and modelling eligibility are distinct. A coverage-eligible source cohort is model-eligible only if it contains at least 200 participants in **each** outcome class after the any-core gate. This count-only rule is applied before feature inspection, splitting, or performance estimation. Under an 80/10/10 allocation it guarantees at least 20 participants from each class in both validation and test; it is a pragmatic common-protocol threshold, not a power or precision guarantee. Garmin, Apple, and Samsung pass; Nokia and GoogleFit remain audit-only because their class counts are too small for the common evaluation protocol. The all-source pass manifest has 8,880 participants, of whom exactly 8,830 are in the three modelling cohorts.

Fit models **separately** within the three model-eligible sole-source cohorts; do not pool sources. The source code is cohort provenance, not a predictor. The `all` model name below means all feature blocks within one source cohort, not all sources combined.

## 3. Feature contract

All blocks are participant-level and are constructed without using the outcome. Build them for the union of model-eligible participants and retain `source_id` only as provenance for filtering. `user_id` and `source_id` are keys, never predictors. All block columns receive explicit prefixes (`demo__`, `rec__`, `win__`, `roll_rec__`, `roll_win__`) so combinations cannot silently collide.

### D. Common demographic block

- `demo__age_group` and `demo__bmi_grp`, added exactly once to every model variant.
- Missing values are handled by training-fitted most-frequent imputation, matching the v2 convention.

### A. Recording-level block (`rec__`)

- Filter `artifacts_v2/epoch_features.parquet` to the union of the Garmin, Apple, and Samsung modelling cohorts.
- Exclude `epoch_primary_source`, `epoch_multisource`, all `epoch_src_*`, and `tz_fully_usable`. These are source/provenance fields or structural constants within a source-specific cohort and must not enter a predictor matrix.
- Exclude `ch*_first_date`, `ch*_last_date`, `overall_first_date`, and `overall_last_date` from predictors. Retain those dates only as cohort provenance. Keep duration/count features such as `*_span_days` and `*_observed_days`.
- Rename the existing `overall_adequate_days` field to `overall_adequate_channel_days`: it counts adequate channel-day rows and can count the same local date more than once. State this definition in the feature dictionary.
- Under the current schema, these exclusions leave **78 numeric recording features**, plus the separate two-column demographic block.

### B. Selected-window block (`win__`)

- Recompute the same 78 numeric families as A within each participant's selected inclusive 91-day window. The selected channel determines the window; features still describe all three core channels over those dates.
- Value, variability, and weekday/weekend summaries use only days satisfying the v2 feature-day rule: at least 8 distinct clock hours and (`cov_s >= 8 h` or `n >= 60`). This rule is intentionally distinct from the 12 h strict eligibility rule.
- Acquisition summaries use all valid observed channel-days within the selected window.
- Absolute window dates and eligibility-selection statistics are provenance, not predictors.
- Window-restricted diurnal means require a targeted raw mini-scan because `epoch_hours.parquet` has no date dimension:
  - Read only the union of Garmin, Apple, and Samsung cohort files from the same raw snapshot used for v2; scan each participant file once.
  - Apply that participant's sole source (`3`, `6`, or `7`), core-channel, HR, timezone, and selected-window filters identical to the frozen upstream rules.
  - Compute event-weighted mean HR for local buckets `night 0–5`, `morning 6–11`, `afternoon 12–17`, and `evening 18–23`, plus per-user/channel/hour counts for diagnostics.
  - Write `artifacts_sq/window_diurnal.parquet`.

### C. Rolling blocks (`roll_rec__`, `roll_win__`)

- Base channel-day series: `daily_mean`, `daily_median`, `daily_sd`, and `daily_hours`.
- Construct a full calendar-day grid per `(user, channel)`. Set all four series to missing on days that fail the v2 feature-day rule; do not use observed-row rolling across gaps.
- Compute right-aligned trailing windows `[t-6, t]` and `[t-29, t]`. A rolling value is defined only when day `t` itself is adequate and at least 4 of 7 or 15 of 30 calendar days, respectively, have non-missing values.
- For each raw series, window width, and channel, summarize defined rolling endpoints with `mean`, sample `sd` (`ddof=1`), `min`, `max`, and OLS `slope` against actual calendar-day offsets. A slope requires at least two defined endpoints.
- `C_rec`: summarize endpoints over the participant's observed recording span.
- `C_win`: summarize endpoints whose dates lie in the selected window. Their trailing inputs may reach up to 29 days before `window_start`; describe this block as **window-endpoint summaries with pre-window lookback**, not as data wholly contained in the strict window.
- Each scope has `4 series × 2 widths × 5 summaries × 3 channels = 120` numeric columns; the two scopes together have 240.

### Combined feature matrix and prespecified variants

| variant | feature blocks | interpretation |
|---|---|---|
| `demo` | D | demographic reference |
| `rec` | A + D | recording-level epoch information beyond the common demographics |
| `win` | B + D | selected-window restriction |
| `roll_rec` | C_rec + D | recording-span multi-scale summaries |
| `roll_win` | C_win + D | selected-window endpoint summaries with pre-window lookback |
| `all` | A + B + C_rec + C_win + D | all prespecified blocks, with D included once |

Before training-fitted transformations and excluding the `user_id`/`source_id` keys, the frozen raw widths are: D = 2 categorical; A = 78 numeric; B = 78 numeric; C_rec = 120 numeric; C_win = 120 numeric. Thus `demo` has 2 inputs, `rec` and `win` have 80 each, `roll_rec` and `roll_win` have 122 each, and `all` has **398**. Any deviation must be explained by an updated feature dictionary rather than an approximate count.

## 4. Participant-level train / validation / test allocation

- Allocate participants independently within each model-eligible source. Keep every participant's records, channels, days, and selected window in one split.
- Use deterministic 80/10/10 largest-remainder totals coupled to the exact outcome counts. The implementation must reproduce this frozen table:

  | source | split | total | sal10 | sal20 |
  |---|---|---:|---:|---:|
  | 3 Garmin | train | **3,078** | **1,107** | **1,971** |
  | 3 Garmin | validation | **385** | **138** | **247** |
  | 3 Garmin | test | **385** | **139** | **246** |
  | 6 Apple | train | **3,554** | **1,098** | **2,456** |
  | 6 Apple | validation | **444** | **137** | **307** |
  | 6 Apple | test | **444** | **137** | **307** |
  | 7 Samsung | train | **432** | **166** | **266** |
  | 7 Samsung | validation | **54** | **21** | **33** |
  | 7 Samsung | test | **54** | **21** | **33** |

- **Seed:** `SEED = 20260918`.
- Use a side-quest-specific allocation by adapting the participant-profile aggregation and SciPy/HiGHS MILP structure in `src/allocate.py`; do not use that file unchanged because it hard-codes v2 paths and source-profile fields.
- Within each source, keep split totals and outcome counts hard. Attempt floor/ceiling proportional allocation for age-group, BMI-group, and qualifying-channel pass-pattern marginals. Optimize residual imbalance in occupied `y × age_group × bmi_grp` cells and selected-channel marginals.
- Implement smallest-feasible relaxation as a two-stage MILP: first minimize the sum of nonnegative integer violations outside all requested floor/ceiling marginal bounds while keeping totals/outcome counts hard; then fix that minimum total violation and minimize the secondary absolute-imbalance objective. Save every nonzero bound violation with its variable, category, split, requested bounds, and achieved count. This is especially relevant to Samsung's smaller occupied strata.
- Reuse the identical source-specific split for every model variant; allocation is feature- and performance-independent.

## 5. Models, evaluation, and exactly 100 bootstrap resamples

- For each of sources 3, 6, and 7, fit `RandomForestClassifier(random_state=20260918)` with every other hyperparameter at the installed scikit-learn default. Save the resolved `get_params()` output and package version.
- Use a side-quest-specific model pipeline; `src/model.py` cannot be reused unchanged because it hard-codes source categorical columns that this protocol excludes and assumes one feature matrix.
- No tuning, cross-validation, calibration, performance-driven feature selection, or repeat-seed search.
- Fit preprocessing independently on each source's training participants and apply it unchanged to that source's validation/test participants:
  - remove all-missing raw columns using training data only and save their names;
  - median-impute remaining numeric columns with missingness indicators;
  - impute categorical demographics and one-hot encode with unknown-level handling;
  - remove zero-variance transformed terms using training data only and save their names;
  - do not scale.
- Fit all six prespecified variants within each model-eligible source: **18 RF fits total**. Report validation and test AUROC, accuracy, balanced accuracy, per-class precision/recall/F1, confusion matrices, and the source-specific training-majority accuracy reference. All model comparisons are exploratory; do not select and declare a test-set winner.
- Samsung's validation and test sets each contain only 54 participants (`21/33` by class); report their metrics and bootstrap summaries as especially imprecise and do not use them for a stable vendor ranking.
- **Bootstrap is fixed at exactly 100 resamples:**
  - `SQ_BOOTSTRAP_N = 100`
  - `SQ_BOOTSTRAP_SEED = 20260918`
  - within each source, resample test participants with replacement separately within salutation class;
  - derive a deterministic source-specific random stream with `numpy.random.SeedSequence([SQ_BOOTSTRAP_SEED, source_id])`;
  - reuse the same 100 source-specific resample-index sets for every variant within that source; resample indices are not shared across sources because cohort sizes differ;
  - recompute metrics from fixed saved test predictions without refitting.
- Save all draw-level metrics. Report the point estimate, bootstrap mean, sample SD, and an explicitly exploratory 2.5/97.5 percentile interval using `quantile(method="linear")`; with 100 draws, the interval is coarse.
- Within each source, use aligned draws for paired differences, at minimum: `rec - demo`, `win - rec`, `roll_win - roll_rec`, and `all - rec`. Report paired point differences and their draw-level distributions; do not construct paired cross-source differences and do not infer superiority from overlap/non-overlap of separate marginal intervals.
- The side-quest constant is independent of the shared v2 `BOOTSTRAP_N = 10`; do not mutate v2 configuration or artifacts.

## 6. Reports and artifacts

Create `artifacts_sq/`; leave v1/v2 artifacts untouched. Add `artifacts_sq/` to `.gitignore` before execution.

```text
artifacts_sq/
  source_cohorts.csv                    # base, per-channel, and any-core counts by source/class
  eligibility_by_channel.parquet        # one row per user/channel with reason/pass/window stats
  cohort_manifest_all_sources.parquet   # every any-core passer; selected-window provenance
  cohort_manifest_model_sources.parquet # 8,830 users; union of source 3/6/7 cohorts
  coverage_semantics_audit.csv          # cov_s > 24 h counts/examples by source/channel
  window_diurnal.parquet                # union of model cohorts; source_id is provenance
  features_demographic.parquet          # D; union of model cohorts
  features_recording.parquet            # A; union of model cohorts
  features_window.parquet               # B; union of model cohorts
  features_rolling.parquet              # C_rec + C_win; union of model cohorts
  features_all.parquet                  # A + B + C_rec + C_win + D once
  feature_dictionary_sq.csv
  sidequest_report.md
  reproducibility_sq.json               # config, versions, input/script hashes
  source_<source_id>/                    # source_id is 3, 6, or 7
    cohort_manifest.parquet
    split_manifest.csv
    split_allocation_report.json
    split_balance_report.md
    feature_schema_<variant>.json
    metrics_sq.json
    predictions_{val,test}_<variant>.csv
    bootstrap_test_draws.csv             # 100 aligned draws per variant
    bootstrap_paired_deltas.csv
    model_pipeline_<variant>.joblib
    model_params_<variant>.json
    transformed_feature_names_<variant>.json
```

`<source_id>` ranges over `3`, `6`, and `7`. `<variant>` ranges over `demo`, `rec`, `win`, `roll_rec`, `roll_win`, and `all`. Every tabular artifact containing participants must carry `user_id`; union artifacts must also carry provenance-only `source_id`.

## 7. Implementation order and loop contract

Keep the implementation source-agnostic by driving it from the frozen ordered tuple `MODEL_SOURCE_IDS = (3, 6, 7)` rather than copying Garmin code three times. A suggested execution order is:

1. **Cohorts:** build all coverage-eligible cohorts, assign the count-based modelling disposition, and write both global cohort manifests.
2. **Features:** build the five feature blocks once for the union of model-eligible participants. Validate one-to-one joins on `user_id` and retain `source_id` only for filtering and provenance.
3. **Per-source loop:** for each `source_id`, filter the cohort and feature rows, allocate the frozen split, then loop over the six variants. Construct a new preprocessing pipeline and RF for every `(source_id, variant)`; no fitted object or learned preprocessing state may cross either boundary.
4. **Predictions and bootstrap:** save validation/test predictions first, then compute all bootstrap metrics from the fixed test predictions without refitting.
5. **Verification and report:** run the checks in §8 from saved artifacts. Generate the report only after every hard check passes.

Use side-quest-specific entry points (for example, a `src/sidequest/` package plus one driver) and leave the existing v1/v2 modules and artifacts unchanged. Each stage must be restartable from its declared inputs, fail on duplicate/missing participant IDs, and log the source and variant currently being processed.

## 8. Executable verification contract

The eventual verifier must recompute checks from machine-readable artifacts; a Markdown assertion is not evidence.

- **Eligibility:** reproduce every base, per-channel, and any-core count in §2; verify any-core membership is the set union of independent channel passers.
- **Weekly semantics:** confirm dense internal zero weeks, bounded candidate starts, at least 11 adequate weeks, at least 65 strict days, Monday starts, Sunday ends, and exactly 91 inclusive dates.
- **Selection:** independently reproduce the descending strict-day, ascending start-date, then channel-priority tie-break; verify stored selected-channel/window statistics.
- **Coverage audit:** record `cov_s > 24 h` prevalence and examples without treating the frozen artifact values as physical local-day duration.
- **Cohort integrity:** recompute sole-source status from retained `epoch_sources(events > 0)`; require 8,880 rows in the all-source pass manifest and 8,830 rows in its exact source-3/6/7 model subset. Confirm that sources 9/13 have no model directory.
- **Split integrity:** within each modeled source, require unique participant IDs, legal labels only, pairwise-disjoint/exhaustive splits, the exact total/class table in §4, and recomputed marginal/joint imbalance with every relaxation explicit. The three source cohorts must also be mutually disjoint.
- **Feature integrity:** unique IDs and column names; exact union-of-model-cohorts row set in every block; disjoint block prefixes; exact raw widths D/A/B/C_rec/C_win/all = 2/78/78/120/120/398; demographics present only in D and exactly once in each design matrix.
- **Forbidden predictors:** reject `salutation`, `y`, gender/sex fields, `who_average`, user/file identifiers, raw timestamps, absolute date bounds, source fields, selected-channel/window identifiers, and eligibility statistics. `age_group` and `bmi_grp` are allowed only through D.
- **Rolling implementation:** test calendar-gap, endpoint, minimum-count, pre-window-lookback, and calendar-slope behavior on small deterministic fixtures. Report empirical non-missing proportions; do not invent an arbitrary completeness threshold. All-missing raw columns and zero-variance transformed terms must appear in the recorded training-derived drop lists.
- **Preprocessing/model schema:** for every source/variant, transformed features must include expected available numeric and demographic terms; save exact input/transformed schemas, source-training-derived dropped columns, RF parameters, and package versions. Assert that every pipeline object is newly fitted and that `source_id` is absent from transformed feature names.
- **Predictions:** within each source, validation/test prediction IDs must exactly equal their split IDs, with one row per participant and no cross-split or cross-source IDs.
- **Bootstrap:** exactly 100 unique resample IDs per source/variant, common resample IDs across variants within a source, exact source-specific class sizes per draw, and within-source paired deltas reproducible from the saved draw table.
- **Provenance:** save hashes for the input parquets, raw-file manifest, side-quest scripts, and frozen configuration.

## 9. Contextual comparison anchors

- v2 pooled test: AUROC **0.68333**. Its saved 10-draw conditional bootstrap AUROC SD is **0.00873**; this is not a confidence interval.
- v1 model, primary-source-3 subgroup on the v1 test (`n = 533`): AUROC **0.74038**. It is neither a strict-singleton cohort nor generated under this side-quest protocol; use the point estimate as context only.
- Do not prestate an expected side-quest AUROC or interval. Results are source-specific descriptive discrimination and within-source paired differences among prespecified feature blocks. The existing anchors do not validate or predict Apple or Samsung performance.

## 10. Runtime and I/O expectation

The targeted Garmin raw files alone are approximately 146 GiB in the current snapshot. Before stating a total runtime, build a raw-file manifest with bytes by source for all model-eligible participants. Benchmark the exact parser separately on a representative subset from sources 3, 6, and 7, record sustained throughput, and estimate each scan as `source_bytes / measured_source_throughput`. Do not reuse the old Garmin-only 12–25 minute total after expanding the scan.

| step | planning range |
|---|---:|
| Cohort construction and audits | ~1–2 min |
| Window-diurnal raw scan | derive from measured bytes/throughput for sources 3/6/7 |
| Feature blocks | benchmark union build; report elapsed time by block |
| Three allocations | expected < 1 min each; measure |
| 18 RF fits, predictions, and saved-prediction bootstrap | benchmark one source/variant, then update estimate |
| **Total** | **not frozen until the expanded raw manifest and benchmark exist** |

## 11. Interpretation and out of scope

- **Target populations/estimands:** separately for each source in `{3, 6, 7}`, predictive discrimination of recorded salutation among v2-eligible, sole-epoch-source participants who satisfy the any-core 13-week gate under the frozen export semantics. These analyses do not estimate performance in all users or a causal/biological effect.
- Cross-source differences in metrics are descriptive, not effects of device/source: cohort composition, available channels, missingness, and vendor processing differ. Do not rank vendors from these results.
- Any-core eligibility materially changes the non-Garmin cohorts, chiefly through channel 3002. Do not describe Apple/Samsung as structurally excluded or channel 3000 as generally dominant.
- Channels 3001/3002 may be vendor-derived quantities that incorporate proprietary processing and potentially demographic inputs. Results concern predictability from the export, not pure physiology.
- The selected-window rule is outcome-blind but conditions the analysis on sustained recording; describe this selection explicitly.
- v1/v2 outputs remain untouched. No side-quest artifact is treated as produced until execution and verification complete.
- Nokia and GoogleFit are audit-only because they fail the frozen class-count rule; this is not evidence that prediction is impossible in those populations.
- **Out of scope unless separately requested:** a pooled all-source model, fitting sources 9/13 under a different evaluation design, tuning, calibration, repeated random splits, cross-validation, non-RF models, causal interpretation, formal cross-source performance tests, and feature-importance claims.
