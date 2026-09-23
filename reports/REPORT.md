# Experiment report: predicting recorded SALUTATION from epoch data

Implementation of `plans/EXPERIMENT_PLAN.md` (protocol frozen 2026-09-18). Status: **complete**.
All artifacts are under `experiments/artifacts/`; code under `src/`. Environment:
`~/micromamba/envs/datenspende/bin/python` (Python 3.14.5, pandas 3.0.3,
numpy 2.4.6, scikit-learn 1.9.0, pyarrow 25.0.1).

## 1. What was run

A full streaming pass over the epoch export (~300 GiB, 25,806 candidate files) reduced each
participant to compact daily/channel aggregates; eligibility and a predeclared coverage gate
were applied; a seeded constrained 80/10/10 allocation was solved to optimality; one default
random forest was fit on training only; validation and test were evaluated once; ten
within-class bootstrap resamples of the fixed test predictions were computed.

Run order (`python src/run_all.py --skip-scan` to reuse the scan):
`epoch_scan.py → build_manifest.py → allocate.py → features.py → balance_report.py →
cohort_report.py → model.py → verify.py`.

## 2. Data and provenance

| Input | Resolved target | Role |
| --- | --- | --- |
| `data/15Sep_2230_PG_type65-66.csv` | `datenspende_epoch2daily/…` | candidate set, daily source profile |
| `data/13Aug_1222.csv` | `Documents/DATA/…` | recorded `salutation` (10/20/30) |
| `data/df_whoOneAverage.csv` | `datenspende_epoch2daily/…` | `age_group`, `bmi_grp` (allocation/reporting only) |
| `out/<user_id>.csv` | `datenspende_epoch2daily/out` | epoch channels 3000/3001/3002 |

Epoch facts established from the data: `startTimestamp`/`endTimestamp` are epoch **ms**;
`timezoneOffset` is **minutes** to add to UTC to obtain local time (range −480…+540); sources
present are 2,3,4,6,7,9,13,19 (plus excluded 46,48; 38 absent). Channel codes map to
HeartRate (3000), HeartRateResting (3001), HeartRateRestingHourly (3002) per
`mapping/epoch_value_types.csv`. **Apple (source 6) emits point-event HR samples while
Garmin (source 3) emits 1-minute interval buckets**; per-source semantics are not
interchangeable, so source availability is kept as an explicit predictor and balance
variable. Whether processed channels (3001/3002) use user-entered sex/salutation in the
vendor calculation is **unverified from these exports**; this limits causal interpretation.
See `experiments/artifacts/cohort_coverage_report.md`.

## 3. Frozen eligibility and coverage gate

- Source exclusions 38/46/48 at the record level, before any count or feature.
- Valid HR: `25 ≤ longValue ≤ 230` bpm, `startTimestamp > 0`; `endTimestamp < start` or
  missing is treated as a point event.
- Local day = floor((start + tz_min·60 000)/86 400 000).
- **Adequate day** per channel: ≥8 distinct local clock hours **and** (union coverage ≥8 h
  **or** ≥60 valid measurements). The disjunction covers interval and point-event devices
  without conflating them; overlapping intervals are merged (union), and span is never used
  as duration.
- **Gate:** ≥14 adequate days in at least one core channel; days are never pooled across
  channels.

## 4. Cohort flow

| Category | Participants |
| --- | ---: |
| Daily type-65/66 users (before source exclusion) | 27,256 |
| No allowed-source daily record | 317 |
| Epoch file zero-byte | 1,133 |
| No valid allowed-source measurement | 3 |
| Missing salutation | 11 |
| Salutation not 10/20 (code 30) | 8 |
| Below coverage gate | 5,299 |
| **Eligible** | **20,485** (sal 10 = 6,788; sal 20 = 13,697) |

Coverage by channel among the gate-passing cohort: 3000 → 19,315 participants with ≥14
adequate days (median 273); 3001 → 18,436 (median 194); 3002 → 19,965 (median 213).
Per-group exclusion tabulation: `experiments/artifacts/exclusion_by_group.csv`.

## 5. Allocation and balance

Participants were aggregated into **1,016 identical allocation profiles** (salutation ×
daily/epoch primary source × every source-membership indicator × multisource flags ×
age group × BMI group). A MILP (scipy/HiGHS) fixed split totals by largest-remainder
rounding of 80/10/10 (ties to validation) and salutation counts jointly, constrained
one-variable category margins within floor/ceiling, and minimised weighted deviations of
salutation×covariate margins, joint strata (size ≥10) and pooled rarity groups. The
hard-margin problem was **feasible and solved to optimality** (status 0, gap 0).

Targets: train 16,388 / val 2,049 / test 2,048; class-1 (salutation 20) 10,957 / 1,370 / 1,370.

Balance outcome: **max one-variable pairwise difference 0.392 pp; zero categories exceed
1 pp**; all joint strata ≥10 have members in every split; all continuous SMDs vs train
≤0.027 (birth year, epoch span, adequate days, valid events). Full details:
`experiments/artifacts/split_balance_report.md`, `split_balance.csv`, `split_balance_continuous.csv`,
`split_manifest.csv`.

## 6. Features

One row per eligible participant, 89 columns (167 after imputation/one-hot), built from
daily-channel summaries. Acquisition counts use every observed day; **value, variability
and weekday/weekend summaries use only adequately observed days**, per the frozen coverage
rules. Features are: value summaries (mean/median/IQR/min/max of daily summaries),
between-day and within-day variability, weekday/weekend contrast, local time-of-day profile
(night/morning/afternoon/evening over all valid local-time events; timezone fully usable for
all 20,485), acquisition/coverage summaries, and epoch channel/source availability.
Categorical predictors: `epoch_primary_source`, `epoch_multisource`. All-missing-in-training
columns are dropped; median imputation adds missingness indicators; one-hot uses unknown
levels ignored. No IDs, salutation, linked gender, age, BMI, WHO or survey metadata enter
the predictor matrix. Dictionary: `experiments/artifacts/feature_dictionary.csv`.

## 7. Model and results

`RandomForestClassifier(random_state=20260918)` with all other scikit-learn defaults
(100 trees, unrestricted depth, `min_samples_leaf=1`, no class weighting), fit once on the
training 80% with equally weighted participants. `predict_proba` column for outcome 1 chosen
via `classes_`.

| Metric | Validation (n=2,049) | Test (n=2,048) |
| --- | ---: | ---: |
| AUROC | 0.668 | **0.713** |
| Accuracy | 0.688 | 0.703 |
| Balanced accuracy | 0.574 | 0.590 |
| Class 0 P/R/F1 | 0.569 / 0.236 / 0.333 | 0.625 / 0.255 / 0.362 |
| Class 1 P/R/F1 | 0.706 / 0.912 / 0.796 | 0.715 / 0.924 / 0.806 |

Majority-class reference accuracy (training majority = class 1): **0.669**; test accuracy
(0.703) exceeds it but per-class recall is very asymmetric. Ten within-class bootstrap
resamples of test predictions: AUROC mean 0.713 (SD 0.016, min/max 0.688/0.733), accuracy
mean 0.704 (SD 0.005), balanced accuracy mean 0.592 (SD 0.007). These condition on the
fitted model, fixed split and observed class counts; the range is not a 95% CI.
Predictions and per-resample rows: `predictions_val.csv`, `predictions_test.csv`,
`bootstrap_test.csv`, `metrics.json`, `model_report.md`.

**Interpretation.** Epoch HR features carry modest, above-chance information about recorded
salutation. High class-1 recall with low class-0 recall is consistent with a 66.9% prevalent
positive class and weak separation. This is retrospective classification of *recorded*
salutation, not prediction of biological sex, gender identity, future participants, or
participants with missing salutation; source–salutation associations may contribute and a
physiological contribution is not isolated.

## 8. Verification and reproducibility

`src/verify.py` asserts and `experiments/artifacts/reproducibility.json` records: labels exclusively
10/20; no excluded-source eligibility records or predictors; all included pass the frozen
gate; each eligible participant appears exactly once; splits disjoint and complete; feature
rows match eligible participants; no prohibited metadata or salutation in features;
numerical features reach the estimator; gate recomputation matches. Input paths, symlink
targets, file inventory, package versions, seeds (`20260918`) and git commit are saved.
**All checks passed.**

## 9. Artifact index

`cohort_manifest.parquet`, `cohort_flow.csv`, `epoch_days.parquet`, `epoch_sources.parquet`,
`epoch_hours.parquet`, `epoch_meta.parquet`, `epoch_channel_audit.csv`,
`coverage_by_channel.csv`, `exclusion_by_group.csv`, `split_manifest.csv`,
`split_allocation_report.{md,json}`, `split_balance.{csv}`, `split_balance_continuous.csv`,
`split_balance_report.md`, `epoch_features.parquet`, `feature_dictionary.csv`,
`model_pipeline.joblib`, `model_params.json`, `feature_names.json`, `predictions_val.csv`,
`predictions_test.csv`, `bootstrap_test.csv`, `metrics.json`, `model_report.md`,
`cohort_coverage_report.md`, `reproducibility.json`.
