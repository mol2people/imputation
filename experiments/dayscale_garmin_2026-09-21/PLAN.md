# Experiment plan: day-level salutation classification (Garmin) — dayscale

Frozen 2026-09-21, before any day-level results were inspected. Status:
**exploratory follow-up** to the FS phase and the age experiment
(`../age_vs_agegroup_garmin_2026-09-20/`), reusing the same participant splits
→ paired across arms and across experiments.

Scope contract (user, 2026-09-21): everything lives in
`experiments/dayscale_garmin_2026-09-21/`; no repo script outside this folder
is created or modified (existing modules imported read-only); compute capped
at 5 workers. Raw time-series exports are never read or committed; the sole
epoch input is the cached per-channel-per-day table `artifacts_v2/epoch_days.parquet`.

## 1. Question

How much recorded-salutation (M/F) information lives in a **single adequate
day** of heart-rate data, and how does it integrate when k days are pooled?
Participant-level models (AUROC ≈ 0.744–0.747 on this cohort) aggregate daily
stats and cannot answer this; the day-level unit can. (Cycle-phase-aware
features for women are a separate future experiment; this one is plain days.)

## 2. Cohort, label, unit

- Source 3 (Garmin) strict-coverage model union: n = 3,848 participants
  (sal10 = 1,384 / sal20 = 2,464). One participant has no ch3000 days at all
  → day pool covers 3,847.
- y = recorded salutation (0 = code 10, 1 = code 20).
- **Day unit** = one (user, local date) with adequate ch3000 data
  (`hours ≥ 8 ∧ (cov_s ≥ 8·3600 ∨ n ≥ 60)` — the v2 per-channel-day rule,
  computed on the fly; 98.6% of s3 ch3000 day rows pass).
- Adequate ch3000-days/user: min 80, p25 343, median 476, p75 501, max 979
  (total ≈ 1.64M). By sex (medians): M 467 / F 482. Day-pool class prior
  (F-share, day-weighted) ≈ 0.647 vs participant prior 0.640.

### 2a. Baseline-day allocation (amendment, frozen before any results)

The discussion draft said arm-B baselines are computed "from that user's
training days" — impossible for val/test users (their days are all val/test).
Amended, symmetric rule: each user reserves a **fixed** seeded random subset
of `min(60, ⌊n_u/4⌋)` adequate days (drawn once from
`SeedSequence([20260921])`, constant across repeats) as **baseline days**,
used only to compute that user's arm-B baseline means. All arms evaluate on
the remaining **model days** (median ≈ 416/user, total ≈ 1.43M), so A, B and
A+covs share an identical, paired day pool. This mirrors deployment (some
history establishes the person's baseline; new days get classified) and is
label-blind.

## 3. Features (per model day)

From `epoch_days.parquet` (per user/channel/date: `mean, median, sd, vmin,
vmax, n, cov_s, hours`), pivoted to one row per (user, date):

- **Arm A (primary, ≈ 28 cols)**: 24 per-channel day stats
  `d_ch{3000,3001,3002}_{mean, median, sd, vmin, vmax, n, cov_h, hours}`
  + 4 within-day contrasts `d_gap_3000_3001_{mean,sd}`,
  `d_gap_3000_3002_{mean,sd}`. Channels absent on a day → NaN → train-median
  imputation (SQPreprocessor), missingness indicators dropped by A1 hygiene.
- **Arm B (secondary, ≈ 52 cols)**: A + 24 per-user baseline deviations
  `dev_ch{c}_{stat} = day_stat − user_baseline_mean(stat)`, baseline means
  from that user's ≤60 designated baseline days (§2a).
- **Arm A+covs (secondary, ≈ 33 cols)**: A + `age_at_day` (birth-year counter,
  Dec-1 rollover, evaluated at the day itself) + `demo__bmi_grp` one-hots.

**Known limitation**: per-day time-of-day (tod) bins are NOT cached anywhere
(only their participant-level means exist in `epoch_features.parquet`), so
within-day circadian shape is absent from all arms. Recovering it needs a
targeted epoch rescan — deferred.

## 4. Protocol

- 30 repeats r = 0..29. **Participant splits reused verbatim from the FS
  phase**: `SeedSequence([20260919, 3, r])`, stratified 70/15/15 per class,
  test first (tr/va/te = 2692/578/578 users) → day-level results paired with
  participant-level numbers on identical test participants.
- Train rows: model days of train users, **capped at K = 300 per participant**
  (seeded subsample per repeat, `SeedSequence([20260919, 3, r, 10])`;
  ≈ 808k rows; p75 model-days ≈ 441 so ~⅓ of users are actually capped).
- Val days: uncapped model days of val users (diagnostics: calibration, PI).
- Test days: uncapped model days of test participants.
- Preprocessing: SQPreprocessor fit on train days + A1 hygiene (drop
  `__missing` indicators; caret nzv: dominant >95% ∧ unique/n <10%).
- RF: G1 @ 100 trees (`max_features=0.4`, `min_samples_leaf=10`,
  `max_depth=None`, `class_weight="balanced_subsample"`),
  `random_state = SeedSequence([20260919, 3, r, 6])` (slot 6, after "all" = 5).
- PI (arm A only) on a seeded 30k-day subsample of val model days
  (`SeedSequence([20260919, 3, r, 30])`), `roc_auc`, 3 repeats.
- Compute: 5 joblib workers, `n_jobs=1` per fit.
- **SA_K sensitivity** (repeat 0 only): arm A at K = 100 and uncapped.

## 5. Estimands

1. **Day-level test AUROC** per arm × repeat; mean ± SD over 30 repeats.
   Within-repeat uncertainty via **participant-cluster bootstrap**: 500
   resamples of test participants (with replacement,
   `SeedSequence([20260919, 3, r, 40])`), pooled-day AUROC per draw → 95% CI.
2. **k-day scaling curve** (primary product): for k ∈ {1, 2, 4, 8, 16, 32, 64,
   128, all}, per test participant with ≥ k model days, mean logit of k seeded
   days (200 trials per (k, user), `SeedSequence([20260919, 3, r, 20])`, same
   draws shared across arms); AUROC over participants per trial; mean ± SD per
   (arm, k). References: participant-level 0.7445 (frozen A3_k50) and 0.7470
   (AGE_all ≈ hygiene fit, age experiment).
3. **Top day-level features** (arm A): Gini + PI (val subsample), averaged
   over repeats — tests whether `mean_of_daily_sd`-type signals are
   within-day informative or cross-day artifacts.
4. **Covariates decomposition**: paired Δ AUROC (A+covs − A) per k.
5. Day-level calibration: 10-bin ECE on val days per arm × repeat
   (class-weighted forest ⇒ probabilities shifted toward 50/50; diagnostic
   only). Day counts and class prior by split reported.

## 6. Caveats

- Clustered units: days within participants; MC SD over repeats is split
  noise only — the cluster bootstrap is the day-pooling uncertainty.
- Adequate-day selection may interact with sex (compliance); near-null here
  (M 467 / F 482 medians) but recorded.
- No within-day tod decomposition (§3 limitation).
- Vendor-processing caveat for ch3001/ch3002 (REPORT.md §2) unchanged —
  especially relevant since ch3001 is literally a daily vendor value.
- Label is recorded salutation, not biological sex/gender.
- Exploratory; same-partition reuse is not independent validation; arm-B
  baselines are transductive per user (label-blind, own-days only).

## 7. Outputs (this folder only)

- `cache/day_table.parquet` — derived per-day table (git-ignored; rebuildable).
- `results/dayscale_metrics.csv` — per (arm, repeat): val/test day AUROC,
  day counts, ECE, fit time; SA_K rows.
- `results/scaling_curve.csv` — per (arm, repeat, k): AUROC mean/SD, n users.
- `results/feature_importance.csv` — per (repeat, feature): Gini + PI (arm A).
- `results/cluster_bootstrap.csv` — per (arm, repeat): bootstrap mean, 2.5/97.5%.
- `results/user_scores_all_repeats.parquet` — per (arm, repeat, user): y,
  n model days, mean logit (all days).
- `results/day_predictions_r0.parquet` — test-day predictions, repeat 0, all arms.
- `results/scaling_curve.png` — the figure (if matplotlib present).
- `results/run_log.txt`, `results/dayscale_repro.json`, `results/REPORT.md`.
