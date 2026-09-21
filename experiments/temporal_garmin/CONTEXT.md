# TEMPORAL — execution context checkpoint (written pre-run, 2026-09-21)

Durable recovery point for the temporal-representation experiment, written before the run per
user instruction ("compact the context before the run"). If the session compacts or dies:
reload state from THIS FILE plus `experiments/temporal_garmin/PLAN.md`. No other conversation
context is required.

## Authorization

User, 2026-09-21: "i want to go ahead and make me proud. I expect to see results / report /
updated readme. But before the start of the run compact the context."
→ Full authorization: implement, run all stages, deliver results, REPORT.md, README.md update,
commit. Context checkpointed to disk (this file); user asked to run /compact before "go".

## Frozen documents

- Plan: `experiments/temporal_garmin/PLAN.md` (= root `TEMPORAL_REPRESENTATION_PLAN.md`,
  amended version). Commit `dabab35`, pushed to origin/main. Deviations require a new plan.
- Vendor pins (GPL-3.0, clone into gitignored `cache/vendor/`, SHAs recorded in repro JSON):
  MultiRocket `ChangWeiTan/MultiRocket` @ `3ccaa4f8ed11769b904af885e17018c39c89a2ac`;
  HYDRA `angus924/hydra` @ `144bb7aa3186654042f00f36b078370c620acb06`.

## Deliverables

1. `experiments/temporal_garmin/results/` — per-alloc per-arm metrics; participant predictions
   (val+test: user IDs, labels, scores, allocation) for all 8 arms; alpha trace; span audit;
   repro JSON (template: R1b `results/r1b_repro.json`); resource measurements.
2. `experiments/temporal_garmin/results/REPORT.md` — house format (model: R1b REPORT.md).
3. Root `README.md` updated — results row + interpretation. NOTE: user added a 14-line
   "research direction and evaluation principles" section (commit `94e1bd9`) — read and
   respect it when editing.
4. Commit (predictions committed, `cache/` gitignored — R1b precedent), push.

## Read-only inputs

- `experiments/r1b_dateaware_garmin_2026-09-21/cache/day40_table.parquet` — 3,848×40 adequate-day
  table (per-day ch3000 stats, `hour_h0..h23`, CURVE35). Supplies B40 stats, Profile24 (bit-exact
  gate) and the frozen user/date pairs.
- `experiments/r1b_dateaware_garmin_2026-09-21/cache/folds.parquet` — 10 allocs,
  train/val/test = 2,696/579/573. Gate: per-alloc test-user set == R1b prediction record.
- `experiments/r1b_dateaware_garmin_2026-09-21/results/r1b_predictions_part.csv` — BASE⊕P40
  predictions (val+test) for the Combined−BASE⊕P40 pairing secondary; zero refits.
- `artifacts_sq/cohort_manifest_model_sources.parquet` + `artifacts_sq/split_manifest_sq.parquet`
  — cohort (3,848 Garmin/source-3; labels 10→0, 20→1) and membership.
- Raw epochs via `src/sidequest/sq_config.py:EPOCH_DIR`; parse rules frozen from
  `src/sidequest/diurnal.py` + `sq_config.py`: source==3, type 3000, HR 25–230, positive starts,
  `local_ms = startTimestamp + 60000*timezoneOffset`, reject unresolved tz.
- Anchors: R1b first-k C40 plateau 0.7286 (day level, same window); BASE⊕P40 0.7513
  (full-history BASE + demographics, same folds).

## Execution stages (order)

0. `vendor_setup.py` — clone pinned commits into `cache/vendor/`, verify file SHA256s, sys.path
   shim; import + fixture on a 288-length series: MultiRocket module-level fit/transform,
   HYDRA `Hydra(k=8,g=64)` + `SparseScaler`, expected widths (~10k/day MR after base+diff1;
   6,144/day HYDRA), byte-identical repeat, batch-vs-unbatched agreement. aeon fallback ONLY
   on fixture failure, recorded as plan amendment.
1. `scan_bins.py` — ≤2 procs, cohort files size-desc → `cache/day_bins.parquet`: per (user, day)
   288 × (event-mean HR float32, count, observed mask) + span audit (end−start distribution).
   Predeclared rule: >50% of retained epochs spanning >1 bin → fractional-allocation Profile288
   sensitivity; decide BEFORE any model results exist.
2. §5 benchmark — 100 train participants × 40 days: timing (exclude first-use JIT compile), peak
   memory, full-run projection (target <8 GiB). Adjust batching only if needed.
3. `run_temporal.py` — sequential allocs; write per-alloc results INCREMENTALLY (interrupt-safe).
   Seeds `SeedSequence([20260921, 3, alloc, component])` with stable component IDs
   (day-selection, mr_fit, hydra, shuffle, rf). MultiRocket fit on 4 seeded train days per
   participant, transform all 40; HYDRA needs no fit data; shuffle arm permutes observed HR
   values within day (masks unchanged, refit transform on shuffled train days). Alpha
   precalibration on alloc-0 val only: log grid 1e-3…1e3 (7 pts) per family, frozen for allocs
   1–9; Shuffled-MultiRocket inherits MultiRocket's alpha. RF = house G1 (100 trees,
   max_features=0.4, min_samples_leaf=10, class_weight="balanced_subsample"). StandardScaler for
   B40/MR blocks, SparseScaler for HYDRA block; all fits train-only.
4. Gates + stats — Bonferroni family (PRIMARY, validation): 4 paired deltas vs Summary-RF
   (Profile288, MultiRocket, HYDRA, Combined), two-sided, per-comparison quantile
   `t.ppf(1 − 0.05/8, 9)`, df=9. Secondaries uncorrected 95% CIs: Profile24−Summary-RF,
   Profile288−Profile24, shuffle decomposition, Combined−BASE⊕P40. Test split exploratory.
   Then REPORT.md, README.md, commit.

## Environment (verified 2026-09-21)

`/Users/bulat/micromamba/envs/datenspende/bin/python` — Python 3.14.5, numpy 2.4.6, pandas 3.0.3,
sklearn 1.9.0, numba 0.67.0, torch 2.14.0 CPU (MPS available but unused by predeclaration).
Determinism verified: conv1d→maxpool→scatter_add chain byte-identical and thread-count-invariant
(1/4/8 threads) on CPU. Host: Apple M5 (4P+6E, 24 GB). Caps: numba/torch 8 threads, scan ≤2
procs, sequential allocs, initial transform batch 32 days. Pre-benchmark wall projection
~1–1.5 h (measure; do not claim before measuring).

## Conventions / gotchas

- MultiRocket: module-level `fit()`/`transform()` ONLY — the wrapper class imports torch and its
  ridge path passes `normalize=False` (removed sklearn ≥1.2). Verify exact module signatures at
  vendoring. Bias quantiles use the global RNG inside njit → `np.random.seed(<comp seed>)`
  immediately before each `fit()`. `num_features=1250` per transformation (base + first-diff)
  → (1250+1250)×4 = 10,000 features/day. Input float64; `np.nan_to_num` after transform.
- HYDRA: W is pure seeded random — no fit data; input_length 288 → 6 dilations → 6,144/day.
  torch CPU float32; use `.batch()` for chunking.
- Reference author code over aeon/sktime: reimplementations have different RNG streams (would
  change the experiment).
- Never materialize per-day transform slabs for all users (6.2 GB risk): chunk by participant
  (~256 users ≈ 10,240 days ≈ 0.4 GB slab), pool mean/SD across each participant's 40 days
  immediately (float64 accumulation), discard slabs. Peak target <8 GiB.
- B40 = mean+sample-SD over 40 days of (ch3000 mean, median, SD, min, max, count, coverage
  hours, observed hours = 16 stats) + weekend−weekday mean HR + 288 mask means = 305 columns.
- Profile24/Profile288: pandas NaN-skipping mean + sample SD across days (P40 convention);
  Profile24 must be BIT-EXACT vs `day40_table` hour aggregation.
- Event counts: combining/span-audit inputs only, never model features.
- Missing-bin fill per alloc: training-only clock-bin medians, overall train median fallback,
  masks unchanged. No interpolation, no per-day normalization.
- Ridge on 0/1 labels → continuous scores → AUROC; intercept, no class weighting.
- Do not print participant data in logs. Report caveats honestly; allocation variability is not
  population uncertainty; never treat repeated participant predictions as independent.

## Repo state at checkpoint (2026-09-21, post-push)

main @ `dabab35` (TEMPORAL frozen plan) ← `94e1bd9` (user: README research-direction section)
← `b695d76` (session log) ← `5387afc` (R1b results) — all pushed to origin/main.
Working tree clean except this untracked CONTEXT.md. `experiments/temporal_garmin/` contains
PLAN.md, .gitignore, CONTEXT.md only — no code yet.

NEXT ACTION on user "go": scaffold `vendor_setup.py`, `scan_bins.py`, `run_temporal.py`; run
stage 0 (vendor clone + fixture), then 1–4.
