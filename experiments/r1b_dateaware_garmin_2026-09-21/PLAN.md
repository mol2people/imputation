# R1b — date-aware circadian features (Garmin)

Status: exploratory follow-up to R1a-AF (commit `ea67e5a`). **Frozen plan — do not modify; deviations require a new plan.** Light by design: one rescan, R = 10 allocations, ~10–12 min wall on 8 workers.

## 1. Question

Does a per-day (date-aware) circadian decomposition buy anything beyond the
participant-level R1a curve and the day-level `dayscale` arm A on this cohort?
Concretely, for a **fixed 40-day prospective window** of adequate ch3000 data:

- Does adding 35 per-day curve features (same formulas as R1a §3) to the
  dayscale arm A day stats improve day-level salutation classification?
- Does coverage-partialling of those 35 features (transductive OLS, same shape
  as `run_r1a.residualise`) change the answer? (Wear-vs-biology mediation, R1a
  precedent.)
- Do **participant-level** aggregates of the per-day curves (mean, SD,
  weekday/weekend contrast) add to the BASE matrix on the **same test users**
  as R1a-AF? (Pairing with R1a-AF BASE predictions is the primary rigour
  transfer — see §5.)

Light, in the user's sense: one targeted rescan (ch3000 hour-of-day on the
designated 40 days only); no within-allocation cluster bootstrap (dayscale
RAM-pressure precedent — cut); no minute-scale dynamics, episodes, or other
deferred-to-phase-2 ideas.

## 2. User rulings (carry-overs and new)

- All experiment code in its own folder (`experiments/r1b_dateaware_garmin_2026-09-21/`).
  No raw time-series (`out/`) committed; rescan output gitignored.
- **8-worker grant** (user 2026-09-21). `N_JOBS_OUTER = 8` for the day-level and
  participant-level `Parallel(loky)`; n_jobs=1 per RF fit.
- ch3000-only convention for **new** curve features (standing; vendor caveat
  ch3001/ch3002). The dayscale arm A ch3001/ch3002 stat/gap cols are reused
  unchanged from `epoch_days.parquet`; the new curve features are ch3000 only.
- Recorded salutation label (not biological sex/gender).
- Do not modify the `cohort_allocator` package.
- Coarser age bands; 70+ merged into a single band.
- Vendor-circularity audit declined (2026-09-21).
- **R = 10** (matches R1a-AF precedent; user ruling).
- **Use first 40 days only** of adequate ch3000 coverage (chronological),
  applied to **training and evaluation** — prospective/deployment estimand.
  Motivated by dayscale k-curve saturation (k=32 → 0.7269, k=all → 0.7302 for
  arm A; gap ≈ 0.003 between k=32 and k=all is below the R1a-AF BASE
  per-alloc noise floor).
- Frozen plan; per-alloc seeds predeclared in §10.

## 3. Cohort & inputs

- **s3 strict cohort, n = 3,848** (2,464 y=1 / 1,384 y=0).
  - _Erratum to dayscale PLAN §2:_ the dayscale PLAN predicted 3,847 from
    `FROZEN_CHANNEL_PASS[3][3000] = 3847` (record-level), but the actual
    `artifacts_v2/epoch_days.parquet` contains ch3000 day-rows for **all 3,848**
    s3 users; the dayscale `cache/day_table.parquet` likewise has 3,848 users
    (verified read-only). Every s3 user has ≥ 80 adequate ch3000 days
    (min 80 / p25 343 / median 476 / p75 501 / max 979). The R1b day pool
    covers all 3,848 users; no exclusion.
- **Designated 40-day window per user** = first 40 adequate ch3000 days,
  chronological (dayscale adequacy rule — see §4). Every user has ≥ 40
  adequate days, so the cap is exactly 40 days × 3,848 = **153,920 day-rows**
  in the day pool.
- Demographics cached, reused verbatim: `experiments/r1a_allocfolds_garmin_2026-09-21/
  cache/demographics.parquet` (3,848 rows; sha-verified into repro). Built by
  `run_r1a_allocfolds.build_demographics` — same age Dec-1 cutoff convention.
- Frozen FS artefacts (READ-ONLY): `artifacts_sq/{split_manifest_sq,
  features_all}.parquet`, `13Aug_1222.csv`, `cohort_manifest_model_sources.
  parquet`.
- Day-stat cache (READ-ONLY): `artifacts_v2/epoch_days.parquet` — used to
  build arm A's 28 day stats (dayscale bit-exact construction, restricted to
  the designated 40 days per user) and to identify the adequate-day sequence.
- R1a feature cache (READ-ONLY): `experiments/r1a_circadian_garmin_2026-09-21/
  cache/r1a_features_epoch_hours.parquet` — **not used by R1b** (R1b has its own
  per-day curve features); included in sha manifest for reproducibility
  bookkeeping only.
- **R1a-AF predictions (READ-ONLY, reused for pairing):**
  `experiments/r1a_allocfolds_garmin_2026-09-21/results/r1aaf_predictions.csv`
  — BASE test predictions per alloc (573 users × 10 allocs) are reused for the
  `BASE⊕P40 − BASE` paired delta. **No BASE refits in R1b.** Same allocator
  folds ⇒ identical test users per alloc (determinism gate, §13).

## 4. Adequacy rule + window selection

- **Adequate ch3000 day** (dayscale, frozen): `h3.hours ≥ 8 AND ((h3.cov_s ≥
  8·3600) OR (h3.n ≥ 60))`. `h3.cov_s` is seconds-of-union-coverage (epoch_days
  schema); `h3.hours` is distinct event-hour count (epoch_days schema).
- **Designated 40-day window** = first 40 adequate ch3000 days per user,
  sorted ascending by local date. Computation: per user, filter
  `epoch_days[channel==3000 & adequate]` → sort by `date` ascending → take
  first 40. Deterministic; no RNG.
- **Weekend flag** (`d_is_weekend`): `local_date.dt.dayofweek ≥ 5` (Sat/Sun).
  Computed once per day row; used by arm A40 and by P40's weekend-contrast
  block.
- **Why 40:** dayscale k-curve saturation shows test-side gain from k=32 to
  k=all is ≈ 0.003 (0.7269 → 0.7302); participant-level gap (k=all → 0.7445) is
  closed by participant-feature aggregation (R1a, dayscale §6). 40 covers
  the saturation plateau without committing the full ~476-day median pool.
  Caveat: dayscale's k-curve used with-replacement within-user sampling; the
  R1b first-k sensitivity (§12) directly tests this on the actual R1b model.

## 5. Allocator spec (verbatim from R1a-AF §4)

```
partition_columns  = ('age_band', 'bmi_group')   # 23 partitions, ≤ 601 users
stratify_columns   = ('salutation',)             # 2 joint cells per partition
fold_sizes         = {'train': 70, 'val': 15, 'test': 15}
seed               = alloc_seed(r) = int(SeedSequence([20260921, 3, r])
                                         .generate_state(1, uint32)[0])
time_limit_seconds = 300
```

Fold totals per alloc on this cohort (R1a-AF precedent): 2,696 / 579 / 573
(train/val/test); test user set per alloc is **identical to R1a-AF** (same
seed stream, same demographics, single-threaded CP-SAT — verified by §13
determinism gate against `r1aaf_predictions.csv` BASE test user lists).

## 6. Rescan schema

`epoch_hours.parquet` has hour-of-day but no date dimension;
`epoch_days.parquet` has date but no hour-of-day. The R1b rescan is the only
way to get `(user, date, hour, n, vsum)` for ch3000 on adequate days.

- **Scope:** s3 strict cohort (3,848 files in `out/`); ch3000 only (`type == 3000`);
  filter to `(user, local_date) ∈ adequate_ch3000_set[user]` — **ALL adequate
  days, not only the designated 40**: the random-40 sensitivity (§12.2)
  samples training days from the full adequate pool, so curves must exist for
  every adequate day. Parse cost is unchanged; only the emitted aggregate
  grows. Aggregate per `(user, date, hour)` to `n, vsum`.
- **`vsumsq` is NOT emitted:** the 35 R1a curve formulas consume only `n` and
  `vsum` (`run_r1a.compute_r1a`); dropping it halves the emission width.
- **Parsing rules (frozen, from `src/sidequest/diurnal.py` and `sq_config.py`):**
  - usecols = `['startTimestamp','endTimestamp','type','longValue','source','timezoneOffset']`
  - drop `source` NaN, exclude `EXCLUDED_SOURCES` = (38, 46, 48, 2, 4, 19)
  - keep `type ∈ CORE_CHANNELS` = (3000, 3001, 3002)  → **further restrict to type == 3000**
  - drop `longValue`/`startTimestamp` NaN; `startTimestamp > 0`;
    `HR_MIN ≤ longValue ≤ HR_MAX` = (25.0, 230.0)
  - `source == 3` (sole source for s3)
  - `local_ms = start + tz · TZ_MS` (TZ_MS = 60 000)
  - `local_date = floor(local_ms / LOCAL_DAY_MS)` (LOCAL_DAY_MS = 86 400 000)
  - `local_hour = floor(local_ms / 3 600 000) % 24`
- **Worker pool:** `ProcessPoolExecutor(max_workers=8)`; jobs sorted by raw
  file size descending for load balancing (sidequest precedent); workers
  return compact per-user DataFrames (downcast dtypes), concatenated once —
  no 25M-tuple accumulation.
- **Output:** `cache/per_day_hour_ch3000.parquet` (columns: `user, date,
  hour, n, vsum`; ~1.64 M user-days × ~13–16 distinct hours ≈ 21–26 M rows;
  ~0.4–0.6 GB in-memory, snappy parquet on disk). Plus a scan log
  `cache/scan_log.json` (files processed, failures, elapsed s, workers,
  designated-day coverage stats).
- **Anchors:** 145.84 GiB total raw, 3,848 files. Sidequest mini-scan (186.2
  GiB, 8 workers) took 177 s; R1b scope ≈ 78% of that → ~140 s ≈ 2.5 min.

## 7. Feature definitions

All day-level columns are pre-A1 (raw, no imputation); A1 hygiene
(`SQPreprocessor` + drop `__missing` + nzv at 95%/10%) is applied per arm by
`apply_arm_transform` (reused from `run_r1a`, identical config).

### 7.1 Day-level arm base columns (28 — dayscale bit-exact)

From `epoch_days.parquet[user, date, channel]` pivoted to wide on `(mean,
median, sd, vmin, vmax, n, cov_h, hours)` for `channel ∈ {3000, 3001, 3002}`,
restricted to the 40 designated dates per user. `cov_h = cov_s / 3600`. Plus
the four gap columns:

```
STAT_COLS = [d_ch{c}_{s}  for c in (3000,3001,3002) for s in (mean,median,sd,vmin,vmax,n,cov_h,hours)]   # 24
GAP_COLS  = [d_gap_3000_3001_mean, d_gap_3000_3002_mean,
             d_gap_3000_3001_sd,   d_gap_3000_3002_sd]                                                   # 4
A_BASE_COLS = STAT_COLS + GAP_COLS                                                                       # 28
```

### 7.2 Per-day curve features (35 — verbatim port of R1a §3 formulas)

Computed per `(user, date)` from the rescan's `(user, date, hour, n, vsum)`
rows:

```
COSINOR = [cosinor_M, cosinor_A1, cosinor_A2, cosinor_A3, cosinor_acro_h, cosinor_resid_sd]   # 6
CURVE   = [curve_night_mean, curve_morning_slope, curve_day_night_contrast,
           curve_range, curve_entropy]                                                       # 5
HOURS   = [hour_h0..hour_h23]                                                                # 24
CURVE35 = COSINOR + CURVE + HOURS                                                            # 35
```

Math is a **verbatim port** of `run_r1a.compute_r1a` (committed at
`4868482`) with the groupby key extended from `['user','hour']` to
`['user','date','hour']`. Same weighted cosinor lstsq (design matrix
`[1, cos 2πh/24, sin 2πh/24, cos 4πh/24, sin 4πh/24, cos 6πh/24, sin 6πh/24]`),
same night/day/morning hour masks, same morning-slope weighted regression,
same entropy formula. Per-day features are NaN where the day's coverage is
insufficient (same rule as R1a: missing hours → NaN; missing all 24 → all 35
NaN). Computed for **all adequate days** (§6); stored float64.

Per-day **coverage columns** for residualisation (stored float64):
`n_total = sum of hourly n`, `log_n = log(n_total)`, `w_h0..w_h23 =
hourly n`.

### 7.3 Per-day residualisation (35 — transductive OLS)

Verbatim port of `run_r1a.residualise` with the row index extended by `date`.
Per feature `c ∈ CURVE35`: OLS `c ~ [1, share_h0..share_h22, log_n]` fit on
**train days of the current allocation only** (`is_train` mask); residuals
returned for all days. Shares use `n_total`-of-the-day denominator
(in-transductive fold). NaN-safe (same convention: fill missing shares/log_n
with 0/nanmean on train). Fail-soft: if `ok.sum() < X.shape[1] + 5` for a
feature, return the raw feature unchanged (preserves R1a behaviour).

### 7.4 Participant arm P40 features (85)

Aggregated per user over the 40 designated days' CURVE35 columns:
- 35 × **mean** (cross-day mean of each per-day curve feature)
- 35 × **sd** (cross-day SD of each per-day curve feature — cross-day
  variability; impossible from R1a's `epoch_hours` cache which has no date)
- 15 × **weekday/weekend contrast** on the 5 CURVE features (curve_night_mean,
  curve_morning_slope, curve_day_night_contrast, curve_range, curve_entropy):
  `[wd_mean, we_mean, wd − we]` × 5 = 15. `wd` = local weekday (Mon–Fri),
  `we` = Sat/Sun; both use pandas mean (NaN-skipping).

NaN handling: pandas `groupby().mean()` and `.std()` skip NaN. A user with
< 40 days (none in this cohort) would have NaN for the days they lack; with
all users ≥ 40, no NaN from missing days. Per-day NaN values (e.g., a day
with too few hours for cosinor) propagate to NaN in the mean/SD/contrast.

### 7.5 Arm composition

| arm       | level        | features (pre-A1)                                                                                | n cols |
|-----------|--------------|--------------------------------------------------------------------------------------------------|--------|
| `A40`     | day          | `A_BASE_COLS` + `d_is_weekend`                                                                   | 29     |
| `C40`     | day          | `A40` ∪ `CURVE35`                                                                                 | 64     |
| `C40_resid` | day        | `A40` ∪ `CURVE35_resid`                                                                           | 64     |
| `P40`     | participant  | 35·mean + 35·sd + 15·weekend-contrast                                                             | 85     |
| `BASE`    | participant  | `features_all` minus id cols (s3 only)                                                            | (varies) |
| `BASE⊕P40`| participant  | `BASE` ∪ `P40`                                                                                    | (BASE+85) |

`_resid` is omitted on the COSINOR/CURVE block; `A_BASE_COLS` and
`d_is_weekend` are kept raw.

**Precision convention:** STAT/GAP columns float32 (dayscale-matching);
CURVE35 / per-day weights / `n_total` / `log_n` float64 (fidelity to the
verbatim port). Day-stat pivot built with `pivot` on unique
`(user, channel, date)` keys (defensively asserted unique) — mathematically
identical to dayscale's `pivot_table(aggfunc="first")`, much faster at
1.64 M rows.

**BASE block integrity:** the participant arm's BASE block is `features_all`
s3 minus `user_id, source_id, y` **exactly as R1a-AF** (including
`demo__bmi_grp` and every demo/rec/win column) — no column drops — so the
`BASE⊕P40 − BASE` pairing compares against the same BASE the R1a-AF
predictions were made with.

## 8. Primary family

Three paired deltas within allocation r ∈ 0..R-1, evaluated on test:

1. **`C40 − A40`** — day-level pooled test AUROC.
2. **`C40_resid − A40`** — day-level pooled test AUROC.
3. **`BASE⊕P40 − BASE`** — participant-level pooled test AUROC (BASE reused
   from R1a-AF saved predictions).

Family-wise inference: Bonferroni-simultaneous **two-sided 95%** over the 3
comparisons (per-comparison quantile `t.ppf(1 − 0.05/6, 9) = t.ppf(0.991667,
9) ≈ 3.250`). Reported side-by-side with per-comparison 95% t-CI
(`t.ppf(0.975, 9) ≈ 2.262`), matching R1a-AF convention. Detection-floor
note (informative, not a power calc): at SD ≈ 0.01 the Bonferroni half-width
is ≈ 0.010; the test family is **underpowered** to confirm deltas ≲ 0.005,
consistent with the R1a-AF "R=10 sanity-check" caveat.

## 9. Protocol per allocation

Per allocation r:

1. **Allocate.** `seed = alloc_seed(r)`;
   `cfg = make_config(id_column="user_id", stratify_columns=("salutation",),
   partition_columns=("age_band","bmi_group"), fold_sizes={"train":70,"val":
   15,"test":15}, seed=seed, time_limit_seconds=300)`; `res =
   allocate_cohort(demo, cfg)`. **Gate:** `all(all_phases_optimal for p in
   res.summary["partitions"])` (fail-loud).
2. **Determinism gate (cross-experiment).** Re-derived `test_user_set[r]` must
   equal saved `r1aaf_predictions.csv[arm=="BASE", alloc==r]["user_id"]`
   sorted, set-wise (§13).
3. **Fold mapping.** Build `tr, va, te` integer index arrays aligned to the
   cohort-sorted user order (assert complete coverage).
4. **Build day pool.** From `epoch_days` (s3 ∩ adequate ch3000) → first 40
   dates per user → build day table with `A_BASE_COLS + d_is_weekend + CURVE35`
   for every `(user, date)` in the pool.
5. **Day-level split.** `tr_days` = day-pool rows whose user is in `tr`,
   `va_days`/`te_days` analogously.
6. **Fit day-level arms.** For each of `A40, C40, C40_resid`:
   - `C40_resid`: compute `CURVE35_resid` transductively — **one OLS fit on
     the full day40 table with the allocation's train-day mask**, residuals
     returned for all rows (R1a `residualise` convention; same fold-leakage
     discipline).
   - Build `X_tr, X_va, X_te` per arm's column list (§7.5).
   - `Z_tr, Z_va, Z_te, names = apply_arm_transform(X_tr, X_va, X_te)`.
   - `rf = RandomForestClassifier(n_estimators=100, random_state=day_seed(r),
     n_jobs=1, **G1)`; fit; record val/test AUROC, n_features, day counts.
   - Save per-day test predictions **for all three day arms** (A40, C40,
     C40_resid; ~23 k rows × 3 × 10 allocs ≈ 700 k rows → parquet; first-k
     uses A40/C40).
7. **Build P40.** From the day table, aggregate per user over their 40 days
   (§7.4).
8. **Fit BASE⊕P40.** Concatenate BASE (`features_all` s3, minus id cols) with
   P40; A1 hygiene on `tr` only; RF with `part_seed(r) = r1a.rf_seed(r)`
   (slot 5 — R1a-AF participant stream). BASE test predictions **reused**
   from `r1aaf_predictions.csv[arm=="BASE", alloc==r]`.
9. **Record.** Metrics, predictions, Gini (C40 + BASE⊕P40), fold sizes, day
   counts, alloc_seed, wall time.

## 10. RF config + seeds

| use                              | seed formula                                  |
|----------------------------------|-----------------------------------------------|
| RF (day-level, R1b)              | `SeedSequence([20260919, 3, r, 7])`           |
| RF (participant-level, P40)      | `SeedSequence([20260919, 3, r, 5])` (R1a slot)|
| Allocator                        | `SeedSequence([20260921, 3, r])`              |
| Adequacy (deterministic)         | none (chronological)                          |
| Random-40 day sampling, trial t  | `SeedSequence([20260919, 3, 0, 50+t])` (r=0)  |
| Random-40 RF (all trials)        | fixed `day_seed(0)` — isolates sampling variation |
| Test-train cap                   | first-40 (deterministic; no within-user RNG)  |

Slot assignments avoid collisions: **5** = R1a / R1a-AF participant (FS-phase
& allocator); **6** = dayscale day-level; **7** = R1b day-level (new). Same
seed stream with a new slot means R1b day-level fits are independent of both
R1a-AF and dayscale.

G1 (frozen, all RF fits): `n_estimators=100, max_features=0.4,
min_samples_leaf=10, max_depth=None, class_weight="balanced_subsample"`.
`n_jobs=1` per fit; `N_JOBS_OUTER = 8` for `Parallel(loky)` over
allocations/trials.

## 11. Evaluation + inference

- **Day-level test AUROC:** pooled over test days (dayscale convention), one
  score per (alloc, arm). Day-level arms are paired within allocation on the
  same test days.
- **Participant-level test AUROC:** pooled over test users. BASE⊕P40 vs BASE
  are paired per allocation on the same test users.
- **CIs:** per-comparison 95% t-CI `t.ppf(0.975, 9)·sd/√10`; Bonferroni
  95%-simultaneous `t.ppf(0.991667, 9)·sd/√10`. Share > 0 reported per
  comparison.
- **Significance bar:** declare "informally positive" only if the Bonferroni
  CI excludes 0; report every comparison's two-sided 95% + Bonferroni + share
  honestly regardless.

## 12. Sensitivities (secondary; pre-declared)

1. **First-k curve (arm C40, r=0..9).** For each k ∈ {1, 2, 4, 8, 16, 24,
   32, 40}, per test user compute mean-logit across their **first k** of the
   designated 40 days (chronological), then pooled participant AUROC across
   allocs. Free from saved per-day C40 test predictions.
2. **Random-40 (arm C40, r=0 only).** 50 trials; per trial, per training user
   sample 40 days without replacement from **all** their adequate ch3000 days
   (seed `SeedSequence([20260919,3,0,50+t])`, users iterated in sorted order);
   RF seed fixed at `day_seed(0)` across trials (isolates sampling variation);
   refit C40; evaluate on the **same** first-40 test days as main r=0 (val =
   the same first-40 val-day view, for hygiene application only). Reports
   mean ± SD of (Δ random vs first-40) — quantifies chronological-window
   bias. Wall ≈ 3–4 min on 8 workers (batched: 8 worker processes, each
   reads `cache/sens_train_table.parquet` once).
3. **A40_alldays (arm A40, r=0 only).** Fit A40 on **all** adequate ch3000
   days of training users (no first-40 cap; no baseline split), evaluate on
   the first-40 test days (same rows/order as main r=0's A40 test set).
   Reports A40_alldays vs A40 — tests whether the first-40 TRAINING
   truncation loses information. Wall ≈ 2–4 min single-threaded (large
   training set).

## 13. Gates

1. **Allocator internal:** `all_phases_optimal == True` for every partition,
   every allocation (fail-loud). `assignments` covers 3,848 users exactly.
2. **Cross-experiment determinism (R1b ↔ R1a-AF):** for every r ∈ 0..9,
   re-derived `test_user_set[r]` equals saved `r1aaf_predictions.csv[arm==
   "BASE", alloc==r]["user_id"]` sorted, set-wise. **Primary gate** for
   pairing BASE⊕P40 − BASE with R1a-AF BASE predictions.
3. **Balance-CSV equality:** re-derived `cache/balance_run{r}.csv` must be
   byte-equal to R1a-AF's `cache/balance_run{r}.csv` for every r
   (secondary; follows from (2) but catches silent fold drift).
4. **Rescan coverage:** every `(user, date)` in the designated first-40 set
   must appear in `cache/per_day_hour_ch3000.parquet` with ≥ 1 hour-row.
   Report distribution of distinct-hour counts and a flag if any designated
   day has 0 hour-rows (fail-loud) or < 8 distinct hours (warn; flag count).
5. **Cohort gate:** 3,848 users × 40 days = 153,920 rows in day pool
   (asserted; min adequate days ≥ 40 verified empirically for all users).
6. **No BASE refits in R1b** (asserted at runner level: BASE participant
   predictions come exclusively from `r1aaf_predictions.csv[arm=="BASE"]`).

R1a's frozen-split `BASE` bit-exact gate does **not** apply here (allocator
folds change test users); rigour transfer is allocator determinism (single
CP-SAT worker + fixed seeds + `ortools 9.15.6755`) + cross-experiment test-
user-set equality.

## 14. Compute (8 workers)

| step                                       | wall                                |
|--------------------------------------------|-------------------------------------|
| Allocator (10 × CP-SAT, serial)            | ≈ 60 s                              |
| Day-table build (epoch_days pivot + 1.64 M-day curve port, 8-way chunked) | ≈ 2–3 min |
| Rescan (145.84 GiB, 8 workers)             | ≈ 140–180 s (≈ 2.5–3 min)           |
| Day-level fits (3 × 10 = 30 fits)          | ≈ 75–95 s                           |
| Participant fits (10 × BASE⊕P40)           | ≈ 20–30 s                           |
| Sensitivity: A40_alldays (r=0)             | ≈ 120–240 s                         |
| Sensitivity: random-40 (r=0, 50 trials)    | ≈ 180–240 s                         |
| Total wall                                 | **≈ 12–15 min**                     |

Scan anchors: 145.84 GiB Garmin-only (3,848 files), sidequest mini-scan
(186.2 GiB, 8 workers) = 177 s → R1b ≈ 78% of that. Curve port on 1.64 M
days ≈ 250–400 s single-core → ~40–60 s at 8 workers (chunked by user).

## 15. Outputs

- `results/r1b_metrics.csv` — per arm × per alloc: val/test AUROC, n cols,
  fold sizes, day counts, alloc_seed, day_seed, part_seed.
- `results/r1b_predictions_day.parquet` — day-level test predictions for
  A40, C40, C40_resid (test days only; ~700 k rows; parquet, not CSV).
- `results/r1b_predictions_part.csv` — BASE⊕P40 test predictions per alloc
  (5 730 rows).
- `results/r1b_predictions_base.csv` — BASE test predictions per alloc
  (copied verbatim from R1a-AF; 5 730 rows). Documents the reuse.
- `results/r1b_top_features.csv` — Gini for C40, C40_resid, BASE⊕P40 per
  alloc.
- `results/r1b_firstk.csv` — first-k participant AUROC for arms A40, C40 per
  k per alloc.
- `results/r1b_random40.csv` — 50-trial random-40 sensitivity (arm C40, r=0).
- `results/r1b_alldays.csv` — A40_alldays vs A40 at r=0.
- `results/r1b_repro.json` — versions, seeds, allocator spec, ortools
  version, R1a-AF prediction-file sha, day-table sha, rescan-output sha, all
  per-allocation wall times.
- `results/run_log.txt` — human log.
- `results/REPORT.md` — §1 absolute AUROC; §2 paired deltas (95% + Bonferroni
  side-by-side); §3 first-k curve; §4 random-40 + A40_alldays sensitivities;
  §5 top features; §6 caveats (incl. 3,847→3,848 erratum); §7 interpretation.
- `cache/demographics.parquet` (link to R1a-AF's, gitignored).
- `cache/balance_run{0..9}.csv` (re-derived; gitignored).
- `cache/alloc_summary_run{0..9}.json` (re-derived; gitignored).
- `cache/alloc_verify.json` — determinism-gate results (test-user-set equality
  per alloc, balance-CSV equality, BASE AUROC recompute, wall times).
- `cache/per_day_hour_ch3000.parquet` (rescan output; gitignored).
- `cache/scan_log.json` (rescan stats; gitignored).
- `cache/dayall_table.parquet` (all adequate days + CURVE35; gitignored).
- `cache/day40_table.parquet` (first-40 view; gitignored).
- `cache/p40.parquet` (85 participant features; gitignored).
- `cache/folds.parquet` (10 allocations × user → fold; gitignored).
- `cache/sens_train_table.parquet` (C40 cols for r=0 train users, all
  adequate days; gitignored).

## 16. Caveats

- **R = 10 under-powers** the 3-comparison primary family for any true delta
  ≲ 0.005; Bonferroni CIs are wide. Direction/magnitude check, not
  confirmatory (same caveat as R1a-AF §10).
- **3,847 → 3,848 cohort correction** (vs dayscale PLAN §2). The dayscale
  PLAN predicted 3,847 from `FROZEN_CHANNEL_PASS[3][3000]` (record-level);
  the actual `epoch_days` ch3000 day-rows cover all 3,848 s3 users, and the
  built `day_table.parquet` has 3,848 users. R1b uses 3,848 throughout.
  Allocator fold totals in R1a-AF (2 696 / 579 / 573) reconcile to 3 848.
- **First-40 is a prospective-window estimator, not a random sample.**
  Random-40 sensitivity (arm C40, r=0, 50 trials) probes how special the
  chronological-first window is.
- **Within-day clustered units** (days within users) inflate day-level test
  AUROC apparent precision. No within-allocation cluster bootstrap (RAM-
  pressure precedent — dayscale). Across-alloc t-CIs only; honest framing.
- **Same-participant reuse** across R1a-AF and R1b is intentional (paired
  BASE predictions); not external validation.
- **ch3000-only** for new curve features (standing; vendor caveat ch3001/
  ch3002). Arm A40 keeps the dayscale ch3001/ch3002 stat/gap cols unchanged.
- **Recorded salutation ≠ biological sex/gender.**
- **No vendor-circularity audit** (user decision 2026-09-21).
- **Rescan portability:** identical to the sidequest `window_diurnal` scan
  (same filters, same `EPOCH_DIR`, same `sq_config` constants); raw file
  layout is unchanged from the original export.
- **R1a formula identity:** the per-day curve computation is a verbatim port
  of `run_r1a.compute_r1a` (committed `4868482`); per-day residualisation a
  verbatim port of `run_r1a.residualise`. No formula drift.
