# Proposed plan: structured recording-missingness (\(R_X\)) as a predictor of recorded salutation

Draft protocol, 2026-09-19. **Exploration protocol only.** Updating this document does not construct cohorts, extract features, allocate splits, fit models, or run bootstrap inference. It does not amend `EXPERIMENT_PLAN.md` or `SIDEQUEST_PLAN.md`.

v1 remains the frozen primary experiment. This plan is a subsequent, predeclared analysis of the **recording/missingness process** already measured in the v1 epoch aggregates. It treats feature missingness and non-wear as potentially MNAR: \(R_X\) may depend on unobserved wear, on recorded salutation \(Y\), or on both. That is an analysis assumption, not a test of the missingness mechanism.

## 0. Relationship to existing work

- **Inputs:** frozen v1 artifacts under `artifacts/` (`cohort_manifest.parquet`, `split_manifest.csv`, `epoch_days.parquet`, `epoch_hours.parquet`, `epoch_meta.parquet`, `epoch_sources.parquet`, `predictions_test.csv`). Do not reread the ~300 GiB raw epoch export.
- **Do not mutate** v1, v2, or side-quest modules or artifacts. Write new outputs under `artifacts_mnar/` (add that directory to `.gitignore` before execution). Use new entry points (for example `src/mnar/`), not `src/model.py` unchanged.
- **Do not** replace the v1 default random forest, retune it, or select a “winner” against v1 from test AUROC.
- Side-quest sole-source / 13-week gates are out of scope here. Source enters only as v1 `epoch_primary_source` / `epoch_multisource` (device choice as part of \(R_X\)).

## 1. Question, mechanisms, and estimands

Let \(Y\) be recorded salutation (10 \(\to\) 0, 20 \(\to\) 1). Let \(X\) be epoch-derived HR summaries. Let \(R_X\) be the recording/missingness process (wear, adequate-day pattern, channel presence, diurnal event placement, device choice). Let \(R_Y\) be missingness of salutation.

**Label missingness \(R_Y\).** The v1 flow records **11** participants with `salutation_missing`. They remain inventory-only. This plan does **not** impute unlabeled salutation and does not identify \(P(Y \mid X, R_Y=1)\). MNAR of \(Y\) is not identified from labeled data without extra assumptions (exclusion restriction or a sensitivity model). Those assumptions are not imposed here.

**Feature missingness \(R_X\).** Among participants with observed binary salutation, v1 already shows a large association between failing the 14-day coverage gate and class 0:

| cohort (v1 flow) | n | salutation 10 | salutation 20 | P(class 0) |
|---|---:|---:|---:|---:|
| eligible (`gate_pass`) | 20,485 | 6,788 | 13,697 | 0.331 |
| below_coverage_gate | 5,299 | 2,874 | 2,425 | 0.542 |

So \(P(\text{fail gate}\mid 10)\approx 0.297\) and \(P(\text{fail gate}\mid 20)\approx 0.150\). Conditioning on the gate changes the target population and truncates \(R_X\). The v1 median-imputation plus per-column `__missing` bits further coarsens what remains.

Two **participant-level** predictive estimands, both among people with observed \(Y \in \{10,20\}\) and at least one valid allowed-source epoch measurement:

| code | target population | split | interpretation |
|---|---|---|---|
| **G** (gated) | v1 eligible (\(n=20{,}485\)) | **reuse frozen v1** `split_manifest.csv` | discrimination of recorded salutation **conditional on** \(\ge 14\) adequate days in at least one core channel; paired with v1 on the same test IDs |
| **U** (ungated) | v1 `eligible` \(\cup\) `below_coverage_gate` (\(n=25{,}784\)) | **new** 80/10/10 allocation (§4) | same prediction task **without** selecting on the 14-day gate; `gate_pass` is a predictor, not an eligibility rule |

Both are retrospective classification of **recorded** salutation. They do not estimate biological sex, gender identity, causal effects of physiology, vendor quality, or performance among participants with missing salutation.

The fitted object for prediction of observed \(Y\) is the pattern-mixture

\[
P(Y \mid X_{\mathrm{obs}}, R_X).
\]

Do not impute \(X_{\mathrm{mis}}\) under MAR (no MICE, no median fill, no impute-then-classify). Native missing-value splits in the classifier plus an explicit process block for \(R_X\) are the intended encoding.

## 2. Cohort definitions (recomputed from v1 artifacts)

Reuse v1 source exclusions (38/46/48), HR validity (25–230 bpm), local-day rule, and adequate-day rule (`hours >= 8` and (`cov_s >= 8 h` or `n >= 60`)). Days are never pooled across channels.

- **G:** `cohort_category == "eligible"` in `artifacts/cohort_manifest.parquet`. Must reproduce \(n=20{,}485\) (6,788 / 13,697). Join `split_manifest.csv`; train/val/test = 16,388 / 2,049 / 2,048.
- **U:** `cohort_category` in `{"eligible","below_coverage_gate"}`. Must reproduce \(n=25{,}784\) (6,788+2,874=9,662 class 0; 13,697+2,425=16,122 class 1).
- All other v1 flow categories stay excluded, including `salutation_missing` (11) and `salutation_not_10_20` (8).
- Assert every U participant has `y` in \(\{0,1\}\) and at least one valid allowed-source event in `epoch_days`.

If any frozen count fails, stop and report drift. Do not relax definitions to recover a count.

## 3. Feature blocks

Construct features from `epoch_days`, `epoch_hours`, `epoch_meta`, and the cohort manifest. No raw re-scan. Prefixes: `r__` (process / \(R_X\)), `v__` (values / \(X_{\mathrm{obs}}\)). Keys `user_id` only; `y` and salutation never enter a matrix.

Forbidden in every matrix: `salutation`, `y`, linked `gender`/sex, `who_average`, user/file identifiers, absolute first/last dates, eligibility-selection statistics other than the explicitly declared `r__gate_pass` in **U only**, age group, and BMI group. Age/BMI remain allocation and reporting variables for U, not predictors (v1 convention).

Timezone: if `tz_fully_usable` is false, all local-hour fractions and local time-of-day means are missing, not zero.

### 3.1 Process block \(R\) (`r__`)

Channel set \(C=\{3000,3001,3002\}\). Adequate-day indicator as in v1.

**Per channel** (15 numeric columns \(\times\) 3 = 45):

| stem | definition |
|---|---|
| `observed_days` | row count in `epoch_days` |
| `adequate_days` | count of adequate days |
| `events` | sum of `n` |
| `span_days` | last observed local date − first + 1; missing if no rows |
| `duty_cycle` | `adequate_days / span_days`; missing if span missing or 0 |
| `events_per_observed_day` | `events / observed_days`; missing if no rows |
| `mean_hours` | mean of `hours` over **all observed** days (not only adequate) |
| `mean_cov_h` | mean of `cov_s/3600` over all observed days |
| `longest_inadequate_run` | longest run of consecutive local dates in `[first, last]` that fail the adequate-day rule; a date with no row counts as a fail; missing if no rows |
| `todfrac_night` | events in local hours 0–5 over channel events, from `epoch_hours.n` |
| `todfrac_morning` | hours 6–11 |
| `todfrac_afternoon` | hours 12–17 |
| `todfrac_evening` | hours 18–23 |
| `n_hours_with_events` | distinct local hours 0–23 with `n > 0` |
| `present` | 1 if `observed_days > 0`, else 0 |

The four `todfrac_*` columns sum to 1 when events exist; keep all four (no leave-one-out). They are **event-placement** features, not mean HR.

**Overall** (6 numeric):

- `overall_events`, `overall_unique_days` (nunique dates with any channel row), `overall_span_days`
- `any_channel_adequate_days`: nunique dates that are adequate on **at least one** core channel
- `any_channel_duty`: `any_channel_adequate_days / overall_span_days`
- `n_core_channels_present`: sum of the three `present` flags

**Categorical (device choice as part of \(R_X\)):** `epoch_primary_source`, `epoch_multisource` from the v1 manifest.

**U only:** `r__gate_pass` \(\in \{0,1\}\), the v1 14-day gate. In G this column is identically 1; **do not include it** (zero variance). Do not use `adequate_days_max` as a second copy of the gate.

Frozen raw width of \(R\): **51 numeric + 2 categorical** in G; **52 numeric + 2 categorical** in U. Deviations require an updated dictionary, not an approximate count.

### 3.2 Value block \(V\) (`v__`)

Value, variability, weekday/weekend, and diurnal **means** use **adequate days only**, same as v1. If a participant has no adequate day on that channel, the corresponding value columns are **missing, not imputed**.

**Per channel** (11 summaries + 4 diurnal means = 15 \(\times\) 3 = 45):

- `mean_of_daily_mean`, `sd_of_daily_mean`, `mean_of_daily_sd`, `median_of_daily_median`, `iqr_of_daily_median`, `min_of_daily_min`, `max_of_daily_max`, `range_of_daily_extremes`, `weekday_mean`, `weekend_mean`, `weekend_minus_weekday`
- `tod_night`, `tod_morning`, `tod_afternoon`, `tod_evening` (event-weighted mean HR in the same hour buckets as v1)

**Contrasts** (2): `ch3000_minus_ch3001_mean`, `ch3000_minus_ch3002_mean` (missing if either side is missing).

Frozen raw width of \(V\): **47 numeric**. No acquisition columns, no source fields, no dates.

Diurnal **means** (`v__*tod_*`) and diurnal **fractions** (`r__*todfrac_*`) are distinct: the former are bpm given events in the bucket; the latter are where events fall. A participant can have a defined night fraction with a missing night mean if night events exist but the day fails the adequate-day rule; that is acceptable and must be documented.

### 3.3 Prespecified variants

| variant | columns | role |
|---|---|---|
| `R` | process block only | \(R_X\)-only baseline: wear, coverage, channel presence, diurnal placement, device choice |
| `V` | value block only, NaNs retained | \(X_{\mathrm{obs}}\)-only: HR summaries without explicit process features |
| `RV` | \(R\) + \(V\) | pattern-mixture encoding \(P(Y \mid X_{\mathrm{obs}}, R_X)\) |
| `V0` | \(V\) minus all `ch3001_*` and `ch3002_*` value/diurnal-mean columns and both resting contrasts | drop vendor resting-HR **values** |
| `P` | `V0` + `r__ch3001_present` + `r__ch3002_present` | circularity probe: resting-HR **existence** plus 3000 values |

Fit every variant under **both** estimands (10 forests). Do not add per-column `__missing` indicators: they duplicate native NaN splits and the `present` / duty-cycle columns.

If `R` captures most of the discrimination of `RV`, the v1 signal is largely recording/device, not HR summaries. If `P` approaches `RV` while `V0` does not, resting-channel **presence** (possibly vendor-derived) is doing the work. These are descriptive paired contrasts, not causal claims.

## 4. Allocation

**G:** do not reallocate. Reuse `artifacts/split_manifest.csv`. Assert exact v1 totals and class counts.

**U:** new participant-level 80/10/10 with `SEED = 20260918`. Adapt the v1 profile-aggregation + SciPy/HiGHS MILP; do not call `src/allocate.py` unchanged (it restricts to `eligible`).

Hard constraints: split totals by largest-remainder 80/10/10 (ties to validation before test); joint salutation counts within floor/ceiling of proportional targets. Then floor/ceiling, where feasible, for: `gate_pass`, `epoch_primary_source`, `epoch_multisource`, age group, BMI group, and salutation-by-`gate_pass`. Residual objective: salutation-by-source, salutation-by-age, salutation-by-BMI, then occupied joint cells with size \(\ge 10\). Same relaxation discipline as v1 (record every bound violation). Balancing `gate_pass` is required so the ungated test set is not an accidental re-gating.

Allocation is feature- and performance-blind. Freeze the U manifest before fitting.

## 5. Preprocessing and model

Installed stack is the v1 environment (Python 3.14, scikit-learn 1.9.0). sklearn 1.9 random forests support missing values; **pass NaNs through**.

For each `(estimand, variant)` independently, fit on that estimand’s training participants only:

1. Drop columns that are all-missing in training; save names.
2. **No median imputation. No missingness-indicator expansion.**
3. One-hot encode categoricals present in the variant (`epoch_primary_source`; `epoch_multisource` may stay binary) with unknown levels ignored.
4. Drop zero-variance transformed columns using training only; save names.
5. Fit `RandomForestClassifier(random_state=20260918)` with every other hyperparameter at the installed default (`n_jobs=-1` is allowed and does not change results given `random_state`). Save `get_params()` and the sklearn version.

No tuning, cross-validation, calibration, threshold search, class-weight search, or performance-driven feature selection. Validation is descriptive of the same frozen fit; test is evaluated once per `(estimand, variant)`. Do not refit on train+val.

`predict_proba` for class 1 via `classes_`. Default `predict` for hard labels.

## 6. Evaluation and bootstrap

Report AUROC, accuracy, balanced accuracy, per-class precision/recall/F1, confusion matrices, and the training-majority accuracy reference, separately for validation and test, for every `(estimand, variant)`.

**G bootstrap (paired with v1).** Exactly **10** within-class resamples of the frozen v1 test participants (`n=2048`).

- `MNAR_BOOTSTRAP_N = 10`
- `MNAR_BOOTSTRAP_SEED = 20260918`
- `SeedSequence([MNAR_BOOTSTRAP_SEED, 0])` for G
- resample test IDs with replacement separately within salutation class, preserving class counts
- **one** index set shared across all G variants and the joined v1 `predictions_test.csv`
- recompute metrics from saved predictions; do not refit

Report point estimate, bootstrap mean, sample SD, min/max. Ten draws are a rough variability check, not a 95% CI.

Predeclared paired deltas on aligned G draws (point difference and draw-level distribution):

- `R - V`, `RV - R`, `RV - V`
- `V0 - V`, `P - V0`, `P - RV`
- `RV - v1` using v1’s saved test probabilities joined on `user_id`

Do not infer superiority from overlap of separate intervals. Do not pick a test-set winner.

**U bootstrap.** The same 10-draw within-class recipe on the **new** U test IDs, `SeedSequence([MNAR_BOOTSTRAP_SEED, 1])`, aligned across U variants. No pairing with v1 (different population and split). Paired deltas: `R - V`, `RV - R`, `RV - V`, `V0 - V`, `P - V0`.

**Descriptive slices (not separate models, not ranking):** on each test set, also tabulate AUROC and class counts by `epoch_primary_source` (sources with \(\ge 50\) test participants). On U test only, tabulate by `gate_pass`. These slices are exploratory descriptions of heterogeneity in \(P(Y \mid X_{\mathrm{obs}}, R_X)\). Do not test vendors or claim an effect of the gate.

## 7. Artifacts

```text
artifacts_mnar/
  cohort_g.parquet                      # 20,485; v1 eligible
  cohort_u.parquet                      # 25,784; eligible ∪ below_gate
  split_manifest_u.csv
  split_allocation_report_u.json
  split_balance_report_u.md
  features_r.parquet                    # union of U; G is a row subset
  features_v.parquet
  feature_dictionary_mnar.csv
  predictions_v1_test_joined.csv        # v1 test IDs/probas for paired G deltas
  mnar_report.md
  reproducibility_mnar.json
  G/
    feature_schema_<variant>.json
    model_pipeline_<variant>.joblib
    model_params_<variant>.json
    transformed_feature_names_<variant>.json
    predictions_{val,test}_<variant>.csv
    bootstrap_resample_indices.csv      # 10 draws; shared across variants
    bootstrap_test_draws.csv
    bootstrap_paired_deltas.csv
    metrics.json
  U/
    ...                                 # same layout; own indices; no v1 delta
```

Every participant table carries `user_id`. Union feature tables may carry provenance `cohort_category` / `gate_pass` as **non-predictor** columns for filtering; they must not appear in transformed feature names except `r__gate_pass` under U variants that include \(R\).

## 8. Implementation order

1. Rebuild G/U cohort tables from v1 `cohort_manifest` and assert §2 counts.
2. Build \(R\) and \(V\) once on the U user set (G is a subset). Write the dictionary with units and missingness semantics.
3. G: attach frozen v1 splits; fit five variants; save predictions; G bootstrap including v1 join.
4. U: allocate; fit five variants; U bootstrap.
5. Verify (§9); write `mnar_report.md` only after hard checks pass.

Restartable stages; fail on duplicate/missing IDs.

## 9. Verification contract

Recompute from machine-readable artifacts.

- **Cohorts:** exact G/U sizes and class counts; U is the disjoint union of v1 eligible and below-gate; no `salutation_missing` rows; every U user appears in `epoch_days`.
- **G split:** identical user sets to v1 train/val/test.
- **U split:** disjoint/exhaustive, frozen totals/class counts, `gate_pass` margins reported, every relaxation explicit.
- **Features:** unique IDs; exact raw widths \(R\)/\(V\); G rows \(\subset\) U rows with identical feature values; no forbidden predictors; `r__gate_pass` absent from G matrices; diurnal fractions sum to 1 (\(\pm 10^{-8}\)) whenever events exist and tz is usable; value columns missing (not zero) when `adequate_days==0` for that channel.
- **Longest-run fixture:** a tiny synthetic `[first,last]` calendar with known holes must reproduce `longest_inadequate_run`.
- **Preprocessing:** all-missing and zero-variance drop lists are training-derived; no `__missing` columns; NaNs remain in numeric inputs to the forest for columns that are partially observed.
- **Predictions:** val/test IDs equal the split IDs; one row per participant.
- **Bootstrap:** 10 unique draw IDs; class sizes preserved; G indices shared across variants and usable on v1 test IDs; paired deltas reproducible from the draw table.
- **Provenance:** hashes of v1 input parquets, this plan, mnar scripts, sklearn version, seeds.

## 10. Interpretation and out of scope

- **G** answers: given the v1 coverage-selected population, how much of recorded-salutation discrimination is in the recording process versus HR summaries, and whether 3001/3002 **values** or **presence** move the metric.
- **U** answers the same question without discarding the 5,299 below-gate participants, among whom class 0 is more prevalent. U is a different target population; do not treat \(\mathrm{AUROC}_U - \mathrm{AUROC}_G\) as an effect of “using MNAR.”
- Native missing-value splits plus \(R\) are a predictive encoding, not an identification strategy for MNAR \(X\) or MNAR \(Y\).
- Channels 3001/3002 may be vendor-derived and may incorporate proprietary processing or demographic inputs. Variant `P` vs `V0` is a circularity/provenance probe, not a physiological contrast.
- Device choice is part of \(R_X\). Source-sliced metrics are descriptive.
- v1 test AUROC 0.713 (10-draw SD 0.016) is a pairing anchor for G, not a performance target.

**Out of scope unless separately requested:** imputing the 11 unlabeled salutations; δ-adjusted pattern mixture or selection models for \(R_Y\); MICE/MAR fill of \(X\); Heckman corrections; XGBoost or HistGradientBoosting; hyperparameter tuning; pooling with v2 demographic predictors; sole-source 13-week cohorts; causal claims; ranking of variants or vendors from test AUROC.

## 11. Runtime (planning)

All inputs are already-aggregated v1 tables. Expected: cohort asserts and feature builds in minutes; ten default RF fits small relative to v1; bootstrap from saved predictions in seconds. Measure and record elapsed times; do not freeze a wall-clock claim before execution.
