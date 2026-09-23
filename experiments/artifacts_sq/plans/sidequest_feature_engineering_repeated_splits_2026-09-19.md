# Side Quest Phase FE — Feature Engineering under Repeated Splits (Plan)

**Label:** `sidequest_feature_engineering_v1`
**Date:** 2026-09-19
**Status:** Pre-registered; implementation follows in `src/sidequest/feat_eng.py`
**Guards:** This phase is **adaptive relative to the cohort's analysis history** and **pre-registered relative to the FS phase's test summaries** (candidate families were listed in `sidequest_feat_sel_repeated_splits_2026-09-19.md` §10 before the FS run completed; the exact column-level formulas below are frozen before any FE evaluation). Per the Codex §12 guard: reusing the FS test partitions supports descriptive paired comparisons only and never independent validation; confirmatory evidence requires new participants or an external dataset.

---

## 1. Objective
Estimate, per source, whether adding pre-specified engineered feature families to the adopted baseline (FS arm A1: indicators + nzv dropped) changes participant-level test AUROC, under the same split-and-fit randomization as the FS phase. Primary estimand: paired AUROC difference over the 30 repeated 70/15/15 splits, conditional on cohort, feature tables, and fixed config.

## 2. Frozen inputs, config, and splits (identical to FS phase)
- Cohorts/features/config: unchanged (`features_*.parquet`, G1 @ `n_estimators=100`, `min_samples_leaf=10`, `max_features=0.4`, `max_depth=None`, `class_weight=balanced_subsample`).
- Splits: the **same 30 repeats per source** — `SeedSequence([20260919, source, r])`, stratified 70/15/15, shared across variants and arms; no re-draw.
- RF seeds: same derivation `SeedSequence([20260919, source, r, variant_index])`, shared across arms (common random numbers).
- **Verification anchor**: arm `B0` (hygiene on the original matrix) must reproduce FS arm `A1` exactly (same split, same prep, same rule, same seed) — asserted against `fsplit/fsplit_metrics.csv` to < 1e-9 for every (source, repeat, variant). A mismatch halts the run.

## 3. Engineered candidate families (frozen; 54 columns)

Parents are raw (pre-imputation) feature values; any engineered value undefined (missing parent, or denominator ≤ 0) becomes NaN and goes through the standard preprocessor (median impute; its `__missing` indicator is then dropped by the hygiene rule like every other indicator).

**E1 — wear-pattern interactions (24).** Product of weekday level and time-of-day share:
`fe1__{p}_ch{c}_weekday_x_tod_{b} = {p}__ch{c}_weekday_mean × {p}__ch{c}_tod_{b}`
for p ∈ {rec, win}, c ∈ {3000, 3001, 3002}, b ∈ {night, morning, afternoon, evening}.

**E2 — channel ratios (12).** Scale-normalized channel contrast (complements the existing minus-contrasts):
`fe2__{p}_ch{a}_div_ch{b}_{s} = {p}__ch{a}_{num} / {p}__ch{b}_{den}` (NaN unless denominator > 0)
for p ∈ {rec, win}, (a, b) ∈ {(3000,3002), (3000,3001), (3001,3002)}, s ∈ {daily_mean → `mean_of_daily_mean`, weekday → `weekday_mean`}.

**E3 — recent-vs-longer drift deltas (18).** Longer-window mean minus recent-week mean:
`fe3__{blk}_ch{c}_{ser}_w30_minus_w7 = {blk}__ch{c}_{ser}_w30_mean − {blk}__ch{c}_{ser}_w7_mean`
for blk ∈ {roll_rec, roll_win}, c ∈ {3000, 3001, 3002}, ser ∈ {daily_mean, daily_hours, daily_sd}.

**Per-variant applicability** (parent closure, asserted at build time): demo → none (all F-arms no-ops); rec → E1+E2 (18); win → E1+E2 (18); roll_rec → E3 (9); roll_win → E3 (9); all → E1+E2+E3 (54).

*Count correction (2026-09-19, before any FE evaluation; smoke run only):* the initial draft mis-multiplied E3 as 36 total (18 per block); the correct block-level count is 9 (3 channels × 3 series), 18 total, grand total **54**. No test summary was inspected before this correction.

## 4. Arms

| arm | rule |
|---|---|
| `B0` | hygiene (FS `A1` rule) on the original matrix — must equal FS `A1` |
| `F1` | hygiene on original + **all applicable families** |
| `F2` | `F1` + greedy \|ρ\|>0.95 dedup, **originals-first keep order** |
| `F1_E1` / `F1_E2` / `F1_E3` | hygiene on original + single family (applicable families only) |

Hygiene = drop all `__missing` indicators (including engineered ones) + caret nzv on transformed train, as in FS. **Documented deviation from FS `A2`:** dedup scans original columns first, then engineered columns, so an engineered term may be evicted by a correlated original but can never evict its parent — preserves the additive semantics of the arm. The nzv variant used is the one implemented in FS (dominant value > 95% AND unique fraction < 10%).

## 5. Primary family and inference discipline (as adopted in report §9.2)
Primary = source 3 `all` and source 6 `all`: `F1 − B0` and `F2 − B0`, two-sided **97.5% t intervals** (Bonferroni-simultaneous 95% over the 4-comparison family). Everything else — single-family arms, `rec`/`win`/`roll_*` variants, all of source 7 — is **secondary/exploratory**; no winner declarations, no pooling, no vendor ranking. All conclusions adaptive/exploratory per the guard above.

## 6. Artifacts, compute, verification
```
experiments/artifacts_sq/feng/
  cells/{source}_{repeat}.json   # rows + engineered survival lists (resume unit)
  feng_metrics.csv               # + n_eng_input / n_eng_after_hygiene / n_eng_after_dedup
  feng_repro.json                # formulas echo, seeds, versions, script hashes
  feat_eng_report.md
```
~2,250 fits @ 100 trees ≈ 12–18 min wall (9 workers). Driver asserts: split disjointness/exhaustive stratified counts identical to FS; B0 ≡ FS A1 (< 1e-9); A2-style containment (`F2 ⊆ F1` columns); parent closure per variant; idempotent resume.
