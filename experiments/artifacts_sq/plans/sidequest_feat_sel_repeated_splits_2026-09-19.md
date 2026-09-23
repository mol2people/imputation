# Side Quest Phase FS — Feature Selection under Repeated Splits (Plan)

**Label:** `sidequest_feat_sel_repeated_splits_v1`
**Date:** 2026-09-19
**Status:** Implemented + executed 2026-09-19 (90/90 cells, 11.5 min); results in `experiments/artifacts_sq/fsplit/`, comparison in `experiments/artifacts_sq/fsplit/feat_sel_report.md`, summary in `experiments/artifacts_sq/sidequest_report.md` §9.1
**Companion to:** `sidequest_strict_coverage_frozen_split_2026-09-19.md` (frozen-split phase, archived)

---

## 1. Objective
Quantify whether feature selection improves per-source salinity-class prediction, replacing the frozen-single-test protocol with repeated stratified resampling. Uniform regularized config, compute-conscious (100 trees). Frozen-split artifacts (`tuned_500/`, `metrics_sq.json`) remain untouched as the single-look record.

## 2. User rulings (2026-09-19)
1. Uniform config everywhere — no per-cell tuning.
2. Iterate train/val/test splits — test **not** frozen; **no allocation matching** (no largest-remainder reproduction).
3. Compute-conscious: **100 trees** — verified empirically (§4). **No 500-tree confirmation tier** (dropped by user decision).
4. Feature selection now; feature engineering as a later phase (§10).

## 3. Protocol

| element | spec |
|---|---|
| repeats | **R = 30** per source |
| split | per repeat: stratified (on `y`, within source) **train/val/test = 70/15/15**, drawn from `SeedSequence([20260919, source_id, r])`; splits disjoint and exhaustive per repeat (asserted) |
| model | **G1 @ `n_estimators=100`**: `max_features=0.4`, `min_samples_leaf=10`, `max_depth=None`, `class_weight=balanced_subsample` |
| seeds | RF `random_state` derived from `(FS_SEED, source, r, variant)` — **identical across arms within a cell** → row-bootstrap streams common-mode; paired deltas deterministic given split |
| variants | demo, rec, win, roll_rec, roll_win, all (demo = null anchor) |
| sources | 3 (Garmin), 6 (Apple), 7 (Samsung; val ≈ 81 rows → A3 PI noisy, include with caveat) |
| labels/cohort | same model-union per-source users (`features_all.parquet` + `y` from `split_manifest_sq.parquet`); no re-derivation of eligibility |

## 4. Empirical basis for 100 trees (measured 2026-09-19, G1 config, same seed, stratified pool holdout)
| cell | 100-tree AUROC | 500-tree AUROC | importance ρ | top-20 overlap | fit time ratio |
|---|---:|---:|---:|---:|---:|
| s3 `all` (n=693 holdout) | 0.7339 | 0.7386 | 0.975 | 17/20 | 1.1 s vs 5.3 s |
| s6 `all` (n=801 holdout) | 0.7164 | 0.7132 | 0.987 | 17/20 | 1.4 s vs 7.1 s |

AUROC delta flips sign across sources (±0.005 ≈ single-fit noise); importances rank-identical (ρ≈0.98); boundary swaps immaterial for top-k. Paired design makes tree-count noise common-mode; across-repeat averaging supplies the variance reduction 500 trees would buy.

## 5. Arms (selection rules fit on that repeat's train only)

| arm | rule | leakage |
|---|---|---|
| **A0** baseline | `SQPreprocessor` as-is (per-column `__missing` indicators included) + G1@100 | none |
| **A1** hygiene | drop **all** `{col}__missing` indicator columns + **caret-nzv** on remaining transformed train matrix (dominant value >95% of train rows **and** unique-value fraction <10%) | none (target-blind) |
| **A2** = A1 + dedup | greedy pairwise **\|ρ\|>0.95** pruning on transformed train matrix; keep first in canonical sorted order, drop later | none (target-blind) |
| **A3** top-k | permutation importance (`scoring=roc_auc`, `n_repeats=3`, on that repeat's **val**) computed from the **A1 fit** (reused); keep top **k ∈ {50, 100}** (ties → canonical order); refit on train with selected columns; eval on test. If A1 matrix ≤ k columns → A3k ≡ A1 (record as no-op) | none — selection consumes val_r only; test_r untouched per repeat |

Motivating facts (from frozen-phase diagnostics, s3 `all`): indicators 131/534 transformed terms with summed Gini importance **0.000**; nzv flags **137/534** (strict-coverage design artifacts, e.g. `win__ch3000_observed_days`); no block >50% missing; roll features redundant across w7/w15/w30.

## 6. Evaluation
- Primary: **paired Δ(arm − A0) on identical splits** → mean, SD, 95% CI over R=30; also Δ(A2 − A1). All arms pre-registered, every arm sees every repeat → no multiplicity tangle.
- Secondary: mean ± SD test AUROC per (source, variant, arm); val AUROC (diagnostic); per-repeat pruned-column counts; A3 top-k set overlap across repeats (selection stability).
- **No per-fit bootstrap** — across-repeat SD replaces it.

## 7. Artifacts
```
experiments/artifacts_sq/fsplit/
  fsplit_metrics.csv        # long: source, variant, arm, repeat, auroc_val, auroc_test,
                           # n_features, n_ind_dropped, n_nzv_dropped, n_corr_dropped, seconds
  pruned_{source}_{r}.json # per repeat: indicator / nzv / corr-dedup column lists
  topk_{source}_{variant}_{r}.json   # A3 selected sets (k=50, k=100)
  fs_repro.json             # seeds, G1 config, versions, R, fractions
  feat_sel_report.md        # comparison write-up
src/sidequest/feat_sel.py   # driver; resumable per (source, repeat) cell
                           # (skip cells whose rows already exist in fsplit_metrics.csv)
```

## 8. Compute
5 fits per (variant, repeat) × 6 variants × 3 sources × 30 repeats = **2,700 fits @ ~0.3–1.5 s** ≈ 45–90 min sequential; parallelize over (source, repeat) cells with `joblib` (RF `n_jobs=1` inside) → ~10–20 min wall. PI: 540 runs (A1-fitted models, val ≈ 580/580/81 rows).

## 9. Verification checklist (driver asserts)
- Splits disjoint + exhaustive per (source, repeat); stratification preserves class counts within ±1.
- Selection rules touch train_r only; A3 importance computed on val_r only; test_r used once per (arm, repeat) for AUROC.
- RF `random_state` equal across arms within each (source, r, variant) cell.
- A1 ⊇ A2 columns; A3k ⊆ A1 columns; demo arm rules may no-op (record).
- Resume: completed (source, repeat) cells skipped; metric rows idempotent.

## 10. Feature engineering phase 2 (deferred; spec after FS results)
Same split seeds → paired across phases. Candidates from the wear-pattern structure: `ch3000` weekday × time-of-day interactions, `ch3000/ch3002` ratios, w30−w7 trend deltas, weekend−weekday × tod. Retention decisions on val_r; test_r final. Informed by FS survivors + importance stability.

---

*Plan approved 2026-09-19. Implementation blocked until agent switch; frozen artifacts untouched by this phase.*
