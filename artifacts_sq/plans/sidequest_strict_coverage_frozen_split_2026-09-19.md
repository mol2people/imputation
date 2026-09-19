# Side Quest — Strict-Coverage Per-Source Modeling (Plan As Executed)

**Label:** `sidequest_strict_coverage_v1_frozen_split`
**Archive date:** 2026-09-19
**Status:** Baseline pipeline complete and verified (`verify.py` 135/135); tuned-500 extension complete (18/18 cells) but not covered by that baseline verifier
**Superseded by:** Feature-selection phase under repeated (non-frozen) splits — separate plan (see §14)

---

## 1. Objective
Per-source (Garmin=3, Apple=6, Samsung=7) strict-coverage cohorts → 398-feature blocks → frozen stratified splits → RF variant comparison (demo / rec / win / roll_rec / roll_win / all) predicting recorded salutation class (sal10 vs sal20), plus a 48-config × 5-fold tuned-500 extension.

## 2. Scope guard
- Writes confined to `artifacts_sq/` and `src/sidequest/`. v1 / v2 modules and artifacts untouched.
- Env: micromamba `datenspende` (`~/micromamba/envs/datenspende/bin/python`, Python 3.14.5, pandas 3.0.3, sklearn 1.9.0, scipy/HiGHS, joblib). Seeds anchored in `src/sidequest/sq_config.py`.
- Spec: `SIDEQUEST_PLAN.md` §1 semantics authoritative. §2/§4 count rows **superseded** (8,880 / 8,830 → **8,885 / 8,835**).

## 3. Pipeline stages (as executed)

| # | Stage | Module | Key numbers / artifacts |
|---|---|---|---|
| 1 | Cohorts | `cohorts.py` | any-core **8,885**; model-union **8,835**; per-source s3 / s6 / s7 = **3,848 / 4,446 / 541**; strict-coverage 13-week windows, ≥65 strict days (D2 rule) |
| 2 | Raw manifest + benchmark | `raw_manifest.py` | **186.22 GiB**, **8,835** files |
| 3 | Window mini-scan | `diurnal.py` | **625,743** hour rows, **0** failures |
| 4 | Features | `features.py` | D2 / A78 / B78 / C120 / C120 = **398** per user + feature dictionary; blocks `demo`, `rec`, `win`, `roll_rec`, `roll_win` |
| 5 | Allocation | `allocate.py` | `T = largest_remainder(FRACS, N)`; class-1 (sal20) apportioned by the same rule, ties → lowest split index; `sal10 = T − sal20` |
| 6 | Preprocessing | `preprocess.py` (`SQPreprocessor`) | median impute + per-column `__missing` indicators (≥1 train-missing → indicator); one-hot demographics, mode impute; drop all-missing-in-train + zero-variance-post-transform (train-only); no scaling |
| 7 | Model — default-100 | `model.py` | RF, default sklearn config, `n_estimators=100`; test = frozen fold (385 / 444 / 54); 100-draw within-class bootstrap, `SeedSequence([20260918, source_id])` → **identical draws across runs / variants**; paired deltas |
| 8 | Baseline verification | `verify.py` | **135/135 checks pass** for the default-100 sidequest and its upstream artifacts |
| 9 | Baseline report + reproducibility | `report.py` | `artifacts_sq/sidequest_report.md`, `reproducibility_sq.json` (**22 SHA-256**), `verification_sq.json` |
| 10 | Later tuned-500 extension | `model_tune.py` | 18/18 source × variant cells complete; see §7; not included in the baseline verifier or baseline SHA-256 manifest |
| 11 | Default-vs-tuned comparison | `compare_models.py` | `artifacts_sq/tuned_500_comparison.md` generated |

## 4. Frozen-split allocation (FRACS ≈ 80 / 10 / 10)

| source | N | train | val | test | pool (train+val) |
|---|---:|---:|---:|---:|---:|
| 3 (Garmin) | 3,848 | 3,078 | 385 | 385 | 3,463 |
| 6 (Apple) | 4,446 | 3,557 | 445 | 444 | 4,002 |
| 7 (Samsung) | 541 | 433 (167 sal10 / 266 sal20) | 54 | 54 | 487 |

- Allocation rule: `T = largest_remainder(FRACS, N)`, class-1 (sal20) by same rule with ties → lowest split index; sal10 = T − sal20.
- 0 bound violations. Frozen Garmin + Apple split tables reproduced exactly. Samsung train allocation **corrected** (rule applied cleanly post-`FROZEN_SPLITS` definition; the user ruling) to (433 total; 167 sal10; 266 sal20).

## 5. Preprocessing contract (`SQPreprocessor`)
- Fit on training fold **only** (no leakage).
- Drops: columns all-missing in train; columns zero-variance post-transform in train.
- Imputation: median for numerics, mode for one-hot demographics.
- Indicators: every numeric column with ≥1 training missing → `{col}__missing` (binary). **Empirically dead weight** under strict coverage — see §10.
- Block-C rolling value = trailing-window mean of non-missing series values (**vetoable interpretation**; documented in report + dictionary).

## 6. Model runs

### 6.1 default-100 (RF default sklearn config, n=100 trees)
Test AUROC (frozen fold):

| source | demo | rec | win | roll_rec | roll_win | all |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 0.5743 | 0.6568 | 0.6918 | 0.5928 | 0.6445 | 0.6318 |
| 6 | 0.5618 | 0.7224 | 0.6814 | 0.7012 | 0.6503 | 0.7184 |
| 7 | 0.5750 | 0.6638 | 0.6609 | 0.6392 | 0.6061 | 0.5880 |

### 6.2 tuned-500 (best of 48 configs × 500 trees, see §7)
Test AUROC (frozen fold):

| source | demo | rec | win | roll_rec | roll_win | all |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 0.5743 | 0.6981 | 0.7279 | 0.6412 | 0.6719 | 0.7068 |
| 6 | 0.5619 | 0.7551 | 0.7472 | 0.7203 | 0.6809 | **0.7714** |
| 7 | 0.5996 | 0.6696 | 0.6342 | 0.6328 | 0.6551 | 0.6797 |

## 7. Tuning grid and protocol
- Grid = **48 configs**: `max_features` ∈ {`sqrt`, `0.2`, `0.4`} × `min_samples_leaf` ∈ {1, 2, 5, 10} × `max_depth` ∈ {None, 12} × `class_weight` ∈ {None, `balanced_subsample`}. `min_samples_split` was omitted to keep the grid bounded; it was not separately evaluated.
- CV: **5-fold StratifiedKFold** (`shuffle=True`, seed `20260918`) on the **train+val pool** (test fold untouched).
- Selection: **pooled OOF AUROC** across the 5 folds; ties → first grid order.
- Preprocessing **refit per fold** (no leakage).
- Final fit: best config × `n_estimators=500`, full train+val pool; one test prediction; **identical bootstrap seed stream** to default-100 → draw-level comparisons valid across runs.
- Artifacts present for every completed cell: `best_params_<variant>.json`, `predictions_{test,val}_<variant>.csv`, `model_pipeline_<variant>.joblib`, `model_params_<variant>.json`, `feature_schema_<variant>.json`, and `transformed_feature_names_<variant>.json`; consolidated results are in `metrics_sq.json`. Per-cell `metrics_<variant>.json` and `tuning_results_<variant>.csv` exist only for cells completed after resume support was added; see §12 for the surviving full grid tables.
- Resume: the implemented cell check uses `best_params_<variant>.json` + `predictions_test_<variant>.csv` + `model_pipeline_<variant>.joblib`.

## 8. Hyperparameter landscape (post-hoc analysis, 2026-09-19)
- Across the **eight retained full grid tables**, `min_samples_leaf` is the clearest and nearly monotone main effect: mean CV AUROC generally rises from 1 → 10 (+0.005 to +0.014), with a near-flat exception for s7 `rec`.
- **`max_features`**: 0.4 (or 0.2) beats sqrt by ~+0.005 on wide blocks (398 features), wash on narrow; second-tier effect.
- **`class_weight = balanced_subsample`**: +0.002 to +0.006 mean gain.
- **`max_depth` showed little marginal association** in the retained tables (≤0.002 mean difference, with sign changes); it rarely appeared influential once `min_samples_leaf ≥ 5`.
- The available two-cell 100-vs-500 diagnostic found no systematic AUROC advantage from 500 trees. This supports 100 trees as a compute-conscious choice; it does not establish that tree count can only affect variance.
- **Global config G1** (`mf=0.4`, `leaf=10`, `depth=None`, `cw=balanced_subsample`) was within 0.006 CV AUROC of the searched optimum in each retained full table (mean gap ≈ 0.003); selected examples are:

| cell | G1 CV | Δ to searched best |
|---|---:|---:|
| s3 `all` | 0.7489 | +0.0002 |
| s6 `all` | 0.7239 | +0.0000 |
| s6 `roll_win` | 0.6531 | +0.0027 |
| s7 `rec` | 0.6190 | +0.0036 |
| s7 `win` | 0.6327 | +0.0055 |

- **Samsung estimates are especially unstable.** Pool n=487 gives noisy per-configuration CV results; best−median differences are only 0.013–0.018. The s7 `demo` CV surface spans 0.459–0.474, and selecting the maximum of 48 configurations adds winner's-curse risk. For s7 `win`, the CV-selected default configuration was 0.027 lower than the default-100 test AUROC; this is consistent with substantial selection noise but does not identify its magnitude uniquely.

## 9. Interpretation framing (agreed)
- The selected configuration's apparent CV AUROC gain (roughly +0.015 to +0.020 in the inspected tables) is **post-selection**: the same five folds were used to choose the maximum among 48 configurations and to describe its gain. It is useful tuning evidence, not an unbiased performance estimate; nested CV would be required for that interpretation.
- Test deltas (+0.027 to +0.075 on Garmin/Apple; mixed signs at n=54 on Samsung) are descriptive. The test fold was not used for tuning, but it had already been examined in the default run, making the tuned analysis a second look with multiplicity and adaptivity concerns.
- The default configuration ranks 46–48/48 in the retained full tables. Together with the two-cell tree-count diagnostic, this is **consistent with** regularization—especially larger leaves—being the main driver of the observed gains; the design does not isolate a causal contribution from each hyperparameter.
- For Apple, the observed tuned test AUROC for `all` (0.7714) exceeds `rec` (0.7551), but their paired bootstrap difference is small relative to its conditional variability. This does not establish that the C blocks add stable signal.

## 10. Feature-space facts (empirical, read-only diagnostics on s3 `all` pool)
- **Indicators** (per-column `__missing`): **131 of 534** transformed terms in the inspected s3 `all` pool; summed Gini importance was 0.000 and the best indicator ranked around 400/534 in that fitted forest. Strict coverage and `min_samples_leaf=10` plausibly make rare indicators unhelpful. This motivates a prespecified indicator-removal arm; it does not establish irrelevance in every source, split, or variant.
- **No block exceeds 50% missing** in the train pool; median 0% across all blocks. A "drop high-missing columns" filter would be a no-op here.
- A custom near-zero-variance diagnostic (dominant value >95% and unique fraction <10%; not exactly caret's frequency-ratio rule) flagged **137/534** transformed columns in the s3 pool, including `win__ch3000_observed_days`, `win__overall_span_days`, near-constant indicators, and `roll_rec__ch3000_daily_hours_w7_max`. Many are plausibly induced by strict-coverage selection, but predictive irrelevance must be tested rather than assumed.
- **Top Gini importances** in the tuned s3 `all` RF were `rec__ch3000_weekday_mean` 0.045, `rec__ch3000_tod_morning` 0.035, `rec__ch3000_minus_ch3002_mean` 0.0176, `rec__ch3000_weekend_minus_weekday` 0.0153, `roll_rec__ch3001_daily_sd_w7_max` 0.0146, and `win__ch3000_minus_ch3002_mean` 0.0137. These mix physiological summaries, temporal patterns, and acquisition effects; correlated-feature Gini importance cannot separate their mechanisms.
- Rolling features appear strongly redundant across window widths (w7/w15/w30 of the same series), motivating a prespecified correlation-deduplication arm.

## 11. Incidents and fixes
- **Crash 1** (`ValueError: cannot convert float NaN to integer`): pandas coerced `max_depth=None` → NaN in the grid lookup. Fix: `best_params = dict(GRID[int(best["config_id"])])` (`model_tune.py`).
- **Crash 2** (silent kill mid s6 `all` CV, no traceback, process gone — suspected SIGKILL / OOM / sleep): Fix — added **resume logic** (per-cell skip when artifacts present) and **per-variant `tuning_results_{variant}.csv`** output (replaces the previous shared `tuning_results.csv` that overwrote itself per source).
- **Stale log cleanup**: `window_diurnal_errors.log` (own failed-first-scan artifact) deleted.

## 12. Provenance notes
- **Full 48-config CV tables surviving on disk**:
  - s3 `all` — old shared `tuning_results.csv` (last variant written per source in v0 run).
  - s6 `roll_win` — same shared file.
  - s6 `all`, s7 `demo` / `rec` / `win` / `roll_rec` / `roll_win` / `all` — per-variant `tuning_results_{variant}.csv` from the resumed run.
  - Net: **8 full tables available** for any hyperparameter analysis beyond `best_params` JSONs.
- Identical 100-draw bootstrap seed stream across default-100 and tuned-500 → draw-level paired deltas are valid for cross-run comparisons.
- The baseline `verification_sq.json` and `reproducibility_sq.json` were generated before tuning. They do not verify or hash `model_tune.py`, `compare_models.py`, or the tuned artifacts; tuned completion is supported by the saved per-cell outputs, aggregate metrics, and run log rather than the 135-check baseline verifier.

## 13. Deliverables inventory
```
artifacts_sq/
  sidequest_report.md
  reproducibility_sq.json              (22 SHA-256)
  verification_sq.json
  split_manifest_sq.parquet
  features_all.parquet                 (398 features + user_id + source_id)
  tune_run.log
  source_3/{metrics_sq.json, tuned_500/...}
  source_6/{metrics_sq.json, tuned_500/...}
  source_7/{metrics_sq.json, tuned_500/...}
src/sidequest/
  sq_config.py cohorts.py features.py allocate.py preprocess.py
  model.py model_tune.py verify.py report.py
  diurnal.py raw_manifest.py test_rolling_fixture.py compare_models.py
```

## 14. Open / superseded items at archive time
- `compare_models.py` has been executed; its output is `artifacts_sq/tuned_500_comparison.md`.
- **v2 pooled rerun under tuned / G1 config** to contextualize the 0.68333 anchor: **deferred / optional**, now superseded by the FS phase under repeated splits (the new protocol makes a single-anchor context unnecessary).
- **Feature-selection phase** under repeated (non-frozen) splits, uniform G1 config — the existing implementation plan remains in `sidequest_feat_sel_repeated_splits_2026-09-19.md`; the independent alternative `CODEX_feature_selection_repeated_holdouts_2026-09-19.md` is documented separately without replacing it.
- **Feature-engineering phase 2** remains deferred. If its specification is adapted after inspecting feature-selection evaluation results, subsequent results on the same participants must be described as exploratory; reusing the same evaluation partitions would not constitute independent confirmation.

---

*Archive of the frozen-split side-quest protocol as executed on 2026-09-19. Frozen artifacts (`artifacts_sq/`, `src/sidequest/`) unchanged by the next phase.*
