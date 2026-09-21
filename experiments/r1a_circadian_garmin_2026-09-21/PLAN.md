# R1a — participant-level circadian curve features from cached hour-of-day data (Garmin)

Frozen 2026-09-21 before any result inspection. Status: **exploratory
follow-up** to the FS phase and the age experiment, reusing the same
participant splits → paired across arms and across experiments. Read-only
w.r.t. the rest of the repo (imports `src/sidequest` modules); writes only
into this folder. Cores: 2 joblib workers during the dayscale run, n_jobs=1
per fit. Raw time-series exports are never read or committed; the sole
input is the cached `(user, channel, hour-of-day)` table.

## User rulings (locked 2026-09-21)

1. **R1a now**, **R1b deferred entirely** (no rescan this round).
2. **Participant-level only** — no day-level R1b in this experiment.
3. **Wear-partialling sub-arm added** — explicit linear partial-out of
   hour-of-day coverage shares; tests whether curve shape carries info
   *beyond* wear pattern, not just whether curve features add anything
   net of a feature-blind RF.

## 1. Question

Does **participant-level circadian curve shape** (circadian harmonics,
night/morning/day-night contrast, per-hour HR profile) carry salutation
information beyond the cached 398 daily/channel summaries? And — sharper
— does it carry information **beyond wear pattern** (the confound the
RF would otherwise absorb via its existing coverage features)?

## 2. Cohort, label, inputs

- Source 3 (Garmin) strict-coverage model union: n = 3,848 participants
  (`artifacts_sq/cohort_manifest_model_sources.parquet`,
  `source_id == 3`). Label = recorded salutation (0/1) from
  `artifacts_sq/split_manifest_sq.parquet`.
- BASE matrix: `artifacts_sq/features_all.parquet`, source 3, variant
  `all` (398 columns), A1 hygiene.
- R1a input: hour-of-day aggregated cache. We try both:
  - `artifacts_v2/epoch_hours.parquet` (preferred; matches the
    sidequest/v2 cohort and feature definitions); fallback
  - `artifacts/epoch_hours.parquet` (main v2 export).
  Each has schema `(user, channel, hour, n, vsum, vsumsq)`. The cohort
  filter (`channel == 3000 ∧ user ∈ s3`) is applied after load;
  coverage statistics are printed and recorded.
- **ch3000 only** (vendor caveat). **Standing decision 2026-09-21:
  the vendor-circularity audit is declined** — ch3000-only is the
  convention for feature-side arms, not a deferral.

## 3. Features (35 R1a columns per user)

Per `(user, channel=3000)`:

- **Weighted cosinor** on 24 hourly means, weights = hour coverage `n_h`:
  `HR(h) = M + Σ A_j cos(2π j h / 24 + φ_j), j = 1,2,3`. Solved by
  weighted least squares (`np.linalg.lstsq`, handles rank deficiency
  via min-norm solution; user-level all-zero hours get weight 0).
- **Acrophase** = argmax of the fitted curve over a 15-min grid in [0, 24),
  reported in local hours. Phase-convention-free (no atan2 detangle).
- **Curve entropy**: `p_h ∝ max(h̄_h − min h̄, 0)` (nonneg-normalised);
  `H = −Σ p_h ln p_h`; `H = 0` guard when range = 0.

| # | feature | meaning |
|---|---|---|
| 1–6 | `cosinor_M, A1, A2, A3, acro_h, resid_sd` | cosinor fit (amplitude, phase, fit quality) |
| 7–11 | `curve_night_mean, curve_morning_slope, curve_day_night_contrast, curve_range, curve_entropy` | curve summaries |
| 12–35 | `hour_h{0..23}` | raw per-hour HR means |

NaN handling: hour cells with `n_h = 0` produce `h̄_h = NaN` (worn-hour
mask); structural NaN propagates through `night_mean` and
`day_night_contrast` if the user has zero worn hours in those windows
— SQPreprocessor median-imputes; the resulting `__missing` indicators
are dropped by A1 hygiene.

## 4. Protocol

- 30 repeats r = 0..29. Participant splits reused verbatim from the FS
  phase: `np.random.SeedSequence([20260919, 3, r])`, stratified 70/15/15
  per class.
- Train rows = participant train users; val = val users; test = test
  users (same participants as every prior phase).
- Preprocessing: SQPreprocessor fit on **train** of each arm's matrix;
  A1 hygiene (drop `__missing` indicators + caret nzv on train).
- RF: G1 @ 100 trees (`max_features=0.4`, `min_samples_leaf=10`,
  `max_depth=None`, `class_weight="balanced_subsample"`),
  `random_state = SeedSequence([20260919, 3, r, 5])` — slot 5 = VI_ALL,
  identical to the FS phase and the age experiment so BASE replicates
  bit-exactly. **Same seed across all arms within a repeat** (paired).
- Compute: 2 joblib workers, `n_jobs=1` per fit (does not disturb the
  5 dayscale workers).

### Arms (5)

| arm | matrix | role |
|---|---|---|
| `BASE` | features_all s3 `all` + A1 hygiene | validation gate + reference |
| `R1a_only` | 35 R1a cols + A1 hygiene | does curve shape *alone* classify? |
| `BASE⊕R1a` | BASE ∪ R1a cols + A1 hygiene | **primary 1**: Δ vs BASE |
| `R1a_resid_only` | OLS-residualised R1a cols + A1 hygiene | wear-partialled curve only |
| `BASE⊕R1a_resid` | BASE ∪ OLS-residualised R1a cols + A1 hygiene | **primary 2**: Δ vs BASE |

OLS residualisation (per repeat, fit on train users only):
- For each of the 35 R1a features `f`, regress
  `f ~ 1 + share_h0 + ... + share_h22 + log_total_n`
  (24 predictors; the last share is dropped to avoid the shares-sum-to-1
  singularity with intercept; log total night-minutes added for scale).
- Use `np.linalg.lstsq` (min-norm on rank deficiency); residuals used
  for **all users** (train and test). Test residuals use train-fit
  coefficients — transductive, label-blind, consistent with the
  participant-level feature convention used everywhere in this program.

### Primary family

- Source 3, variant `all`. Two comparisons: `BASE⊕R1a − BASE` and
  `BASE⊕R1a_resid − BASE`. 97.5% t-CIs (two-sided), Bonferroni-simultaneous
  95% family. **Detection floor ≈ 0.003 AUROC at R=30.**
  *(Erratum 2026-09-21: the run implemented the program's 95% t-CI formula
  (`t.ppf(0.975, 29)`, see `feat_sel._mean_ci`); the reported CIs in
  `results/REPORT.md` are 95% t-CIs. Bonferroni-simultaneous 97.5% CIs
  added in REPORT §6 erratum. R1a-AF follow-up commit `a87027c` reports
  both side-by-side.)*
- Secondary cells: source 3 `rec`, `win` (FS precedent); the standalone
  `R1a_only` / `R1a_resid_only` arms (how far curve-alone gets without
  the cached matrix); top-Gini R1a features inside `BASE⊕R1a`;
  univariate AUROCs of the 35 R1a features pooled across the cohort
  (descriptive only; label-blind).

## 5. Validation gates

1. **BASE bit-exactness** vs FS-phase `A1` (s3 `all`):
   max |ΔAUROC| ≤ 1e-12 across all 30 repeats against
   `artifacts_sq/fsplit/fsplit_metrics.csv` `A1` rows. If exceeded, fail
   fast before any arm is interpreted.
2. **Cohort + cache coverage report**: fraction of s3 strict users
   present in the chosen epoch_hours cache; per-hour coverage
   distribution; rejection rate of users with degenerate (constant)
   hourly curve. Printed at start, recorded in `repro.json`.

## 6. Caveats

- **Same-partition reuse** — not independent validation.
- **Curve features correlate with wear pattern by construction**;
  `BASE⊕R1a_resid` exists to test the deconfounded version.
- **ch3001/ch3002 excluded** — vendor-processing caveat. The
  vendor-circularity audit is **declined** (standing decision,
  2026-09-21); ch3000-only stands.
- **Weekday/weekend contrast not derivable** from the hour-of-day cache
  (dates are lost); that family lives in R1b.
- **Caveats carried from the FS phase**: vendor-processing for ch3001/3002,
  label = recorded salutation not biological sex/gender, single-epoch
  cohort selection by `≥14 adequate days in a core channel`.

## 7. Outputs

- `cache/r1a_features.parquet` (derived, gitignored) — 3,848 × 35 + coverage cols.
- `results/r1a_metrics.csv` — per `(arm, repeat)`: test/val AUROC, n cols.
- `results/r1a_top_features.csv` — per `(repeat, feature)`: Gini in `BASE⊕R1a`.
- `results/r1a_univariate.csv` — per R1a feature: pooled cohort AUROC.
- `results/r1a_predictions.csv` — per `(arm, repeat, split, user_id)`:
  y_true and p_class1.
- `results/r1a_repro.json` — versions, seeds, gate line, input SHA-256,
  cache coverage.
- `results/run_log.txt`, `results/REPORT.md` (mechanical tables;
  §Interpretation filled after inspection).
