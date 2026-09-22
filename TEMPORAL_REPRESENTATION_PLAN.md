# Temporal representations for recorded-salutation prediction

> **Status (2026-09-22):** executed 2026-09-21/22 in [`experiments/temporal_garmin/`](experiments/temporal_garmin/) — results in [`results/REPORT.md`](experiments/temporal_garmin/results/REPORT.md) (v2 primary; v1 retained as the erratum record), roadmap and multichannel draft plan in [`NEXT_STEPS.md`](experiments/temporal_garmin/NEXT_STEPS.md). This root copy is the freeze-time authorization record, byte-identical to the experiment's `PLAN.md` (commit `dabab35`). Body below unchanged.

Status: **Frozen plan — do not modify; deviations require a new plan.** Amended 2026-09-21 after review (profile ladder arms, Ridge alpha precalibration, locked validation-split primary family, epoch-span audit, BASE⊕P40 pairing); execution authorized 2026-09-21 in `experiments/temporal_garmin/`. No results exist at freeze.

## 1. Objective and scope

Test whether **within-day** temporal patterns discarded by existing summaries improve participant-level AUROC. Days are transformed independently and pooled by mean/SD; cross-day dynamics (trend, weekday cycle beyond the weekend contrast, autocorrelation across days) are out of scope (§7.4). Start with CPU-only MultiRocket and HYDRA against cheap profile baselines; defer learned embeddings, clustering, calibration and abstention thresholds. AUROC >0.80 is a research target, not an expected result; no performance ceiling is established. Matched-window anchors: R1b first-k C40 plateau **0.7286** (day level, same window); BASE⊕P40 **0.7513** (full-history BASE with demographics, same folds).

Implement in `experiments/temporal_garmin/`. Preserve existing code, artifacts and allocator. Commit participant-level predictions (scores, labels, IDs, allocation) as results; gitignore raw bin and transform caches only (R1b precedent). Target remains recorded salutation: 10→0, 20→1.

## 2. Frozen cohort and inputs

- Garmin/source 3 strict cohort: **3,848 participants**, from `artifacts_sq/cohort_manifest_model_sources.parquet`; labels from `artifacts_sq/split_manifest_sq.parquet`.
- Reuse the first **40 adequate ch3000 days** per participant from `experiments/r1b_dateaware_garmin_2026-09-21/cache/day40_table.parquet`. These are adequate days, not 40 consecutive calendar days; the cohort was selected using longer histories. The table's per-day `hour_h0..h23` columns supply the Profile24 arm without a new scan.
- Reuse all ten participant allocations from that experiment's `cache/folds.parquet` (train/validation/test = 2,696/579/573). Do not regenerate allocations or split days independently. This is the **last experiment on these matched folds**; further work on this line uses held-out participants or export data.
- Obtain sequences from raw files resolved through `src/sidequest/sq_config.py:EPOCH_DIR`. Existing hourly caches cannot recover subhourly sequences.

## 3. Sequence construction

Stream only cohort files; retain source 3, channel 3000, positive start timestamps and HR 25–230 bpm. Use event-local time: `local_ms = startTimestamp + 60000 * timezoneOffset` (timestamps in milliseconds, offsets in minutes). Reject unresolved timezone metadata rather than substitute UTC. Select only the frozen user/date pairs.

Build **288 five-minute clock bins per day**, storing event-mean HR, event count and an observed-bin mask. Assign by event start time; do not expand interval values across their duration. Combine repeated clock bins by event count, including repeated hours. Counts serve combining and the span audit only; they are not model inputs (mask means carry placement-of-observation). Audit timezone changes and missing bins. Cache this label-independent representation once.

**Epoch-span audit (predeclared):** on the benchmark scan, record the `endTimestamp − startTimestamp` distribution. If more than half of retained epochs span more than one 5-minute bin, add a fractional-allocation sensitivity for Profile288 (epoch value distributed across overlapped bins in proportion to duration overlap); otherwise record the distribution and keep start-bin assignment. Decide before any model results exist.

For each allocation, fill missing HR bins with training-only clock-bin medians; use the overall training median if a clock bin is entirely missing. Keep masks unchanged. Do not interpolate gaps or normalize away each day's HR level. Missing-bin filling is an explicit modelling convention, not reconstructed physiology.

## 4. Representations and models

Build a common participant summary `B40`: mean and sample SD across days of ch3000 mean, median, SD, minimum, maximum, count, coverage hours and observed hours; weekend-minus-weekday mean HR; mean mask at each of the 288 clock bins. Use cached daily statistics and the new masks. This retains HR level and recording information in every arm.

| Arm | Participant inputs | Classifier |
|---|---|---|
| Summary-linear | B40 | Ridge |
| Summary-RF | B40 | RF |
| Profile24 | B40 + per-hour cross-day mean/SD of `hour_h0..h23` (48) | Ridge |
| Profile288 | B40 + per-bin cross-day mean/SD of bin HR (576) | Ridge |
| MultiRocket | B40 + pooled MultiRocket features | Ridge |
| HYDRA | B40 + pooled HYDRA features | Ridge |
| Combined | B40 + both pooled transforms | Ridge |
| Shuffled-MultiRocket | B40 + MultiRocket of shuffled observed HR bins | Ridge |

Profile aggregation: pandas NaN-skipping mean and sample SD across the 40 days (P40 convention). Profile24 replicates R1b's P40 hour block on the same window and folds — sanity anchor, expected positive. Profile24 → Profile288 → transforms is a resolution ladder: flat gains from 24 to 288 to ~20k kill 5-minute resolution and the transforms alike.

- MultiRocket: approximately **10,000 output features per day**, recording actual width. HYDRA: official univariate configuration, `k=8`, `g=64`; record actual width.
- Apply each transform to individual days; aggregate each feature across the 40 days by **mean and sample SD**. Fit the classifier on participants, not days.
- Shuffle control: independently permute observed HR values within each day, leaving observed positions and missing bins unchanged; then apply the same filling procedure. Refit its transform on shuffled training days. Interpretation is a **decomposition, not a null**: `MultiRocket − Shuffled` = order/placement information beyond the daily value distribution; `Shuffled − Summary` = distribution + mask information beyond B40; `MultiRocket − Summary` = both. Shuffling destroys the known hour-of-day signal (R1b morning block), so the control is a lower bound on that signal.
- Ridge: intercept enabled, no class weighting; evaluate continuous decision scores. **Alpha precalibration** — the one exception to deferred tuning: on allocation 0 validation only, fit log-grid `alpha` over 1e-3 … 1e3 (7 points; grid matches the official convention in both repos, `RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))`) per representation family (Summary-linear, Profile24, Profile288, MultiRocket, HYDRA, Combined); freeze each arm's validation-argmax alpha for all ten allocations. Shuffled-MultiRocket **inherits MultiRocket's alpha**, so its delta isolates the transform change. Report the full alpha trace. Allocation 0 validation scores are mildly optimistic by selection; noted in the report — paired deltas over ten allocations remain the inference unit. Fit median imputation, zero-variance removal and scaling on training participants only. Use standard scaling for B40/MultiRocket and the official HYDRA `SparseScaler` for its pooled block; document fitted columns.
- RF: 100 trees, `max_features=0.4`, `min_samples_leaf=10`, `max_depth=None`, `class_weight="balanced_subsample"` (house G1, matches `run_dayscale.py`/R1a/R1b).
- Any data-dependent transform parameters use training days only. Fit MultiRocket on four seeded days per training participant; transform all 40. Share each fitted transform across its standalone/combined arms within an allocation. Never reuse a training-fitted transform across allocations.
- Seed streams: `SeedSequence([20260921, 3, allocation, component])`; assign and save stable component IDs for day selection, transforms, shuffling and RF.
- Implementations (reference author code, pinned): MultiRocket — `ChangWeiTan/MultiRocket`, `multirocket/multirocket.py`, **module-level `fit()`/`transform()` only** (the wrapper class's ridge path passes `normalize=False`, removed in sklearn ≥ 1.2, and its import chain pulls torch via `logistic_regression.py`); call `np.random.seed(<component seed>)` immediately before each `fit()` — bias quantiles draw from the global RNG inside njit. HYDRA — `angus924/hydra`, `code/hydra.py`, official univariate `Hydra` (`k=8`, `g=64` defaults) + `SparseScaler`, torch CPU, `seed=<component seed>`; expected width 6,144 at input length 288 (record actual). Pins frozen 2026-09-21: MultiRocket commit `3ccaa4f8ed11769b904af885e17018c39c89a2ac`, HYDRA commit `144bb7aa3186654042f00f36b078370c620acb06`. Both GPL-3.0: clone at a pinned commit into gitignored `cache/vendor/`, never commit third-party source into this repo; record commit SHA + file SHA256 in the repro record. Fallback on fixture failure only: aeon ports, recorded as an amendment.

## 5. CPU and memory budget

Environment: datenspende micromamba env — Python 3.14.5, numpy 2.4.6, numba 0.67.0 (JIT import verified 2026-09-21), `torch==2.14.0` (macOS wheel; MPS available but **unused by predeclaration** — scatter_add atomics would break byte-identical reruns, M5 gain marginal). Before the benchmark, import the pinned MultiRocket and HYDRA implementations and run a fixture transform on a 288-length series to confirm kernel compilation.

CPU only; cap total active threads at eight (host: Apple M5, 4P+6E; `numba.set_num_threads(8)`, `torch.set_num_threads(8)`). Run allocations sequentially, with no parallel model workers. Scan with at most two processes; transform in batches initially limited to 32 days. Accumulate participant means/variances immediately and discard daily transform matrices. Use float32 caches and float64 accumulation.

First benchmark 100 training participants × 40 days, excluding first-use compilation from steady-state timing. Record scan/transform/classifier time and peak memory separately; project the full ten-allocation cost. Target peak memory below 8 GiB. If exceeded, reduce batching/concurrency; if still infeasible, stop and report the bottleneck rather than silently changing the experiment. Do not claim a runtime before measuring it.

## 6. Evaluation and checks

Primary, locked to the **validation** split (test reported as exploratory — these folds have already supported R1a-AF and R1b): participant AUROC and four paired deltas vs **Summary-RF** — Profile288, MultiRocket, HYDRA, Combined. Bonferroni-simultaneous two-sided 95% over the 4 comparisons: per-comparison quantile `t.ppf(1 − 0.05/8, 9)`, df = 9, paired over the 10 allocations. Report each allocation, mean, SD and fraction of positive deltas. Summary-linear is reported but outside the family.

Secondary (uncorrected 95% CIs, no family claim): Profile24 − Summary-RF (R1b hour-block replication); Profile288 − Profile24 (5-minute resolution increment); the shuffle decomposition (§4); Combined − BASE⊕P40.

Production pairing: per allocation, pair Combined's validation and test scores with the saved R1b `r1b_predictions_part.csv` BASE⊕P40 predictions on the same users (identical folds, zero refits, R1b pairing template). BASE⊕P40 is full-history BASE with demographics — context, not a matched-window baseline; likewise R1a-AF/P40.

Use mean validation AUROC for advancement decisions; test results remain exploratory because these participants have already supported model development. Allocation variability is not population uncertainty. Never treat repeated predictions of one participant as independent observations.

Before scaling up, verify exact cohort/day/fold membership; fold test-user equality against the R1b prediction record; label isolation; finite model inputs; training-only fitting; unchanged masks under shuffling; batch-versus-unbatched transform/pooling agreement on a small fixture; transform fixture import and compile on the pinned env; Profile24 aggregation bit-exact against R1b `day40_table`; and seed reproducibility. Save validation **and** test scores, labels, IDs and allocation for every arm, plus feature schemas, configuration, versions, checks and resource measurements. Write a repro record (JSON) with pinned MultiRocket/HYDRA commit SHAs, env versions, seed streams, span-audit outcome and gates (`r1b_repro.json` template). Produce one concise results report. Do not print participant data in logs.

## 7. Subsequent research stages

After this comparison, freeze a separate follow-up plan chosen from:

1. **One-minute resolution:** test whether five-minute aggregation discarded useful dynamics; a negative five-minute result does not rule this out.
2. **TS2Vec embeddings:** compare continuous participant pooling against training-fitted motif clusters and participant occupancy/transition features. Do not assume two label-aligned clusters.
3. **Supervised participant pooling:** small shared window encoder plus attention, trained with one loss per participant.
4. **Latent-state dynamics:** shared states with participant-specific dwell-time and transition summaries.

Keep the observation budget and participant splits matched. Only after selecting a representation, tune the learner and test complementarity with the existing model using development-only fitting. Reserve new participants/export data for independent confirmation; then assess high-precision labeling with abstention.

## References

- [MultiRocket implementation](https://github.com/ChangWeiTan/MultiRocket)
- [HYDRA implementation](https://github.com/angus924/hydra)
- [TS2Vec](https://arxiv.org/abs/2106.10466)
- [Attention-based multiple-instance learning](https://proceedings.mlr.press/v80/ilse18a.html)
