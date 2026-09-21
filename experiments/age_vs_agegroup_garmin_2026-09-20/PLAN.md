# Experiment plan: DOB-derived age counter vs categorical age_group — Garmin only

Frozen 2026-09-20, before any 30-repeat results were inspected (a single-repeat
timing/determinism benchmark on repeat 0 was run; its numbers are subsumed by the
deterministic full run). Status: **exploratory follow-up to the FS phase**
(`artifacts_sq/fsplit/`), same split seeds → paired across arms.

Scope contract (user, 2026-09-20): everything for this experiment lives in
`experiments/age_vs_agegroup_garmin_2026-09-20/`; no repo script outside this
folder is created or modified (existing modules are imported read-only);
compute capped at 5 workers.

## 1. Question

Does replacing the categorical `demo__age_group` (WHO brackets: Young Adults /
Middle Ager / Elderly) with a continuous DOB-derived age counter change the
top-50 permutation-importance selection arm, when the selection pool is the
full all-blocks matrix ("permute over all blocks")?

## 2. Cohort and label

- Source 3 (Garmin) strict-coverage model union: n = 3,848 (sal10 = 1,384,
  sal20 = 2,464; class 1 prevalence 64.0%).
- y = recorded salutation (0 = code 10, 1 = code 20). **Not** biological sex or
  gender identity (REPORT.md §7 interpretation limits carry over).

## 3. Age feature (new)

- Birth year B from `13Aug_1222.csv` (`birth_date` holds the birth **year**
  only; complete for all 3,848 cohort members; range 1935–2000, median 1970).
- Reference date: each participant's `window_start` — the start of their own
  selected 91-day analysis window (2020-04-13 → 2022-09-19, median 2021-10-25).
  This is "DOB at the start of the study [window]", per-participant.
- Age-change convention (user-specified): age increments every 1 Dec, i.e.
  birthday = Dec 1 of the birth year:
  `age = ws.year − B − 1 + 1[ws.month == 12]`.
- Sensitivity definition: age at the fixed data cutoff 2022-12-18 (last
  `window_end` in the data), which reduces to `2022 − B`. Report Spearman ρ
  between the two definitions; a fixed anchor makes age a pure monotone
  transform of birth year (identical model information), the per-participant
  anchor additionally encodes recording timing.
- Age enters as a plain numeric column (SQPreprocessor median-imputes if
  missing; none are missing in this cohort).

## 4. Protocol (identical to the FS phase; machinery imported read-only from `src/sidequest/`)

- 30 repeats r = 0..29; stratified 70/15/15 split drawn from
  `SeedSequence([20260919, 3, r])`; per-class permutation (test first, then val).
- `SQPreprocessor` fit on train_r (drop all-missing; numeric median impute +
  `__missing` indicators; categorical mode impute + one-hot; drop zero-variance).
- A1 hygiene: drop all `__missing` indicator columns, then the caret
  near-zero-variance rule (dominant value > 95% of train rows AND unique-value
  fraction < 10%); train-fitted, target-blind.
- RF: G1 @ 100 trees (`max_features=0.4`, `min_samples_leaf=10`,
  `max_depth=None`, `class_weight="balanced_subsample"`),
  `random_state = SeedSequence([20260919, 3, r, 5])` — the "all" variant index,
  shared across arms within a repeat (paired).
- Permutation importance on **val_r** (`roc_auc`, 3 repeats,
  `random_state = SeedSequence([20260919, 3, r, 5, 99])`); stable descending
  argsort; top-50 of the A1-hygiene columns; refit on train_r; test_r used once.
- Compute: 5 joblib workers, `n_jobs=1` per fit/PI (phase convention).
  Determinism pre-check: repeat 0 reproduced the frozen A3_k50 val/test AUROC
  bit-exactly both with `n_jobs=1` and `n_jobs=-1` (RF is deterministic given
  the seed).

## 5. Arms (selection pool = all blocks, per user instruction)

| arm | matrix | selection |
|---|---|---|
| `REF50` | phase `all` matrix (`demo__age_group` + `demo__bmi_grp` one-hots + 78 `rec__` + 78 `win__` + 120 `roll_rec__` + 120 `roll_win__`) | top-50 by PI — exact replica of FS arm A3_k50, validated bit-exact against `fsplit_metrics.csv` |
| `AGE50` | `age` replaces `demo__age_group`; `demo__bmi_grp` kept | top-50 by PI (age competes in the ranking) |
| `AGE50_force` | same matrix as `AGE50` | `age` forced in + top-49 of the rest by the same PI ranking (secondary; literal "age feature + top-50" reading) |
| `AGE50_nodemo` | `age` replaces the whole demo block (bmi dropped) | top-50 by PI |
| `AGE_all` | same matrix as `AGE50` | none (A1 hygiene only; context for the selection effect) |

## 6. Estimand and inference

- Primary: paired test-AUROC delta (arm − REF50) over the same 30 repeats;
  mean, SD, 95% t-CI (df = 29), share > 0.
- Absolute AUROC (mean ± SD) per arm; val AUROC as diagnostic only.
- Secondary: age top-50 membership (x/30), overall PI rank distribution, mean
  PI value; top-50 composition frequency per arm; mean pairwise cross-repeat
  Jaccard within arm; same-repeat Jaccard AGE50 vs REF50.
- Univariate context: AUROC of age alone, of the fixed-anchor age, and of the
  `Elderly` one-hot indicator.

## 7. Caveats

- Exploratory; reuses the FS-phase participants and split seeds —
  same-partition reuse is never independent validation.
- Age is a linked profile covariate (same family as the demo block it
  replaces), not epoch-derived; its association with salutation is cohort
  structure, not physiology.
- Label is recorded salutation; the vendor-processing caveat for ch3001/ch3002
  (REPORT.md §2) is unchanged.
- The 95% t-CIs quantify split/model randomization conditional on this cohort
  (MC error), not participant sampling, transportability, or analyst
  adaptivity.

## 8. Outputs (this folder only)

- `results/age_metrics.csv` — one row per (arm, repeat): val/test AUROC of the
  top-50 fit and of the hygiene fit, column counts, age diagnostics.
- `results/age_predictions.csv` — per (arm, repeat, split): user_id, y_true,
  p_class1 for the evaluated fit (top-50 fits; hygiene fit for `AGE_all`).
- `results/age_top50_sets.json` — per (arm, repeat) selected column lists.
- `results/age_repro.json` — config echo, package versions, script SHA-256,
  input hashes.
- `results/REPORT.md` — tables, diagnostics, reading.
