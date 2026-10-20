# Implementer handoff: monthly descriptors and downstream prediction

2026-10-20. Read [PLAN.md](PLAN.md) as the method authority and [READINESS.md](READINESS.md) as the artifact inventory. This handoff is operational guidance, not a replacement protocol. Only inspection and documentation are complete. No monthly features, transforms, models, eligibility manifests or inference for this experiment have been executed. Implementation/execution requires the subsequent task authorization; do not launch an older pipeline automatically.

## Start here: scope and resolved specifications

Implement an exploratory **allocation-0**, separate-source comparison of monthly-feature augmentation. Primary inputs are Apple 11,139 participants (7,797/1,671/1,671 train/validation/test) and the full-history Garmin pilot 1,000 (700/150/150). These counts precede monthly eligibility. Administrative salutation 10→0 and 20→1 is the target. Exact age and clinical cycle status are unavailable.

The primary contrast adds original-observation monthly HR features to a controlled HR+recording+categorical-age baseline. Compare clock filling with short linear filling as specified below. Period-aware filling, completed-grid monthly endpoints, rhythm removal, extra allocations, strict-Garmin expansion and strongest-HR/mask-stack follow-up are deferred. Do not claim biological cycles or improvement over the strongest historical architecture.

**Resolved handoff clarification:** HR R uses intercept + linear elapsed time + weekday effects + four six-hour original clock-coverage fractions. Q monthly occupancy/nocturnal-mask descriptors use intercept + linear elapsed time + weekday effects **only**, on the same 90 dates and 21-period grid. Occupancy is the average of those fractions and nocturnal mask is the first fraction; including coverage as predictors would deterministically remove their residual variation. This clarification is now explicit in PLAN.md.

The reproducibility defaults below are implementation decisions, not new scientific arms. They resolve routine choices; record them in the run configuration before execution. No scientific launch blocker remains in the documented specification.

## Inputs and code to inspect/reuse

| Input | Exact location / schema |
|---|---|
| Apple splits | [apple_10_allocations.parquet](../apple_cohort_scaling/artifacts/apple_10_allocations.parquet): `customer_id,user_id,salutation,y,age_group,bmi_group,n_days,alloc_0,…`; retain `alloc_0`. |
| Garmin pilot splits | [garmin_binning_5_allocations.parquet](../garmin_temporal_binning/artifacts/garmin_binning_5_allocations.parquet): participant demographics and `alloc_0`; inspect actual schema before joining. |
| Apple per-user inputs | [cache_5min](../apple_cohort_scaling/artifacts/cache_5min/), indexed by `customers.csv` (`customer_id,file,n_days` plus occupancy summaries). |
| Garmin per-user inputs | [cache_5min](../garmin_temporal_binning/artifacts/cache_5min/), same indexed per-user schema. |
| Each NPZ | `date[D]` integer epoch days, `hr[D,288]` float32/NaN, `stats[D,8]` float32. Dates are already local calendar dates; do not normalize timezones again. |

Daily statistics are ordered `mean,median,sd,vmin,vmax,n,cov_h,hours`, with native event-statistic semantics. Do not replace them with quantile/IQR columns or assume every column is bpm.

Reuse narrow helpers from [Apple transforms](../apple_fill_and_mask_by_recording/scripts/transforms.py): `fit_fill_templates` and `interpolate_short_gaps`. In the [Apple benchmark runner](../apple_fill_and_mask_by_recording/scripts/02_run_fill_and_mask_benchmark.py), inspect `compute_static_features_b_l`, `transform_sequence_batched`, and `evaluate_feature_arm`; adapt their interfaces rather than copying the full runner. Its existing main launches a larger benchmark and its selection helper emits three penalty rules, not this plan's single common rule. [heartbeat_lab/model.py](../../HEARTBEAT_LAB/src/heartbeat_lab/model.py) provides `reference_functions`/`seed_numba` for the pinned MultiRocket implementation.

The old [cosinor helper](../female_infradian_cycles/scripts/00_verify_infradian_methods.py) illustrates harmonic fitting but is **not drop-in**: it uses a simpler nuisance model, total-variance ΔR² and nominal selected-period p-values. Implement the new joint nuisance/partial-R² specification explicitly. Existing [v1 metrics](../female_infradian_cycles/results/infradian_metrics_by_participant.csv) use different dates/exposure and cannot substitute for new R features.

No ready train/validation MultiRocket matrices were found. Apple saved test scores and Garmin aggregate metrics do not support augmentation without rebuilding features. Strict Garmin [r1b folds](../r1b_dateaware_garmin_2026-09-21/cache/folds.parquet) are a separate 3,848-person protocol; all-cohort k5 production bundles are not holdout-safe allocation-0 models.

## Ordered implementation checkpoints

1. **Freeze configuration and inspect inputs.** Record source/code hashes, package versions, thread limits, seed streams and optional diagnostic decision. Establish ID uniqueness, salutation consistency, source cohort disjointness, split parity and ordered-statistic agreement. Preserve all upstream artifacts.
2. **Build the eligibility/date manifest without outcomes.** For each person, qualify original dates with ≥120 finite bins. Find the earliest `[start,start+180)` window with ≥15 qualified dates in each of six 30-day blocks; sample exactly 15/block using the frozen seed20261020 derivation. Use sorted dates, original masks and no HR magnitudes/labels for selection. Retain original allocation assignments. Report age×salutation×split retention before fitting. Apple global feasibility4631 is a check, not audited split-specific counts; Garmin window eligibility remains unknown.
3. **Small training-only smoke test.** Where available, use 2–3 training participants per age×salutation cell to verify extraction, masks, feature order, missing-status handling, pooling and determinism. This is an engineering check, not tuning or an estimable salutation benchmark. It does not authorize scanning every raw file or calling legacy full runners. Then run the full source feature build only under execution authorization.
4. **Extract shared observations/controls and fill-specific H.** Persist the manifests, unprocessed features and fitted source-training quantities before head fitting. Check resumability/provenance; stale caches must not match merely by filename.
5. **Fit/select/freeze heads.** Source validation selects alpha only. Finish schema/arm decisions first. Save training/validation scores and selected alphas before evaluating test. Preserve test IDs across arms.
6. **Evaluate and generate the report from tables.** Save all test scores, paired draws/metrics and age diagnostics. Do not insert fixed narrative numbers into a report generator.

## Feature and fitting invariants

Every arm uses the identical 90 dates/person. Raw observed HR values and original masks are immutable. H consists of 305 static+18,816 pooled daily MultiRocket features. Compute static summaries from native unfilled statistics/masks. Fit source-training clock medians over eligible training participants' selected dates; global observed-cell median is fallback. Short linear interpolation uses original observed endpoints within a day, at most 5 missing bins (≤30-minute endpoint separation), clock fallback otherwise. Never interpolate across midnight, use filled anchors or requalify days.

Fit two H transforms with matched sampled training dates/seeds, one per filling rule. Existing sampling helpers cap the training sample at 4096 days using participant-wise sampling followed by truncation; do not silently replace that with a uniform pooled-day sampler. The new 90-observation, 180-calendar-day budget uses the explicit seed specification below. Between fills, biases/heads are refitted. Within a fill, persist and reuse bit-identical H matrices and preprocessing between B and B+R. Source-training fitting applies to templates, MultiRocket biases, imputation, zero-variance filtering and scaling; validation/test never fit them. Adapt the old whole-matrix preprocessing helper into frozen H and separate Q/R/age blocks; refitting a combined preprocessor separately per head would violate the requested H reuse.

R has exactly A, S and T from original observed-bin HRp10 on these 90 actual dates. Fit intercept+linear elapsed time+weekday+four six-hour original coverage fractions jointly with each of21 harmonic candidates25:.5:35 days. S uses baseline residual RSS, not total variance. Exclude phase and nominal significance flags. Undefined metrics remain missing until source-training median imputation. Shared Q carries HR fit-availability/zero-variance flags in **both** baseline and augmented arms; retain noncyclic/degenerate participants.

Q also contains the frozen nonperiodic HRp10 controls, original clock means/date-span/longest-gap controls and monthly recording descriptors. Age is a separate three-level one-hot block. Preserve Q/R identically across fills, avoid identical duplicate columns and save the final ordered definitions. Follow the plan's linear-only residual-SD denominator90−2.

Eight core heads are H_clock; B_clock; B_clock+R; B_linear; B_linear+R; B_clock_without_age; that baseline+R; and age-only. Optional `B_clock+R_control` uses21 periods38–56 days and must be included/excluded before validation/test results, never triggered by a test gain.

Use `sklearn.linear_model.Ridge` with `.predict` continuous scores, not `RidgeClassifier`, and alpha `{100,500,1000,5000,10000,100000}` with source-validation AUROC; Apple `np.isclose(atol=1e−12, default rtol)` near-ties favor larger alpha. No train+validation refit, test-selected alpha, probability-calibration/Brier claim or test-median threshold. Historical grids/scores are context; changed exposure/cohort requires a newly fitted matched baseline, not numerical equality with old full-cohort scores.

## Frozen implementation defaults

- Sort participant IDs by their stable string representation and all date arrays chronologically. Keep source plus ID as the unique participant key; never use Python's randomized `hash()`.
- Derive each RNG seed from SHA-256 of the UTF-8 string `20261020|source|participant_id|component`; use the first eight digest bytes as an unsigned little-endian integer for NumPy `default_rng`. For source-wide streams use participant ID `ALL`. Component names separate `dates_block_0` through `dates_block_5`, `mr_day_sample`, `mr_base`, `mr_diff` and `evaluation_bootstrap`. No fill/arm token enters shared date/day-sample or paired evaluation streams. Map the MR branch seed modulo `2**31−1` for `seed_numba`; retain distinct branch components.
- Select 15 dates per block with that block's stream and `choice(..., replace=False)`, then sort. For MR fitting reuse the existing at-most-4096 scheme: sorted eligible training IDs, `max(1,4096//n_train)` sampled dates per user capped by availability, concatenate, and randomly truncate to4096 if necessary. Sample once and reuse selected day identities and branch seeds for both fills.
- Monthly nuisance-column priority is intercept, linear elapsed time, weekday indicators Tuesday–Sunday (Monday reference), then the four clock fractions for HR only. Retain columns in that order only when they increase numerical rank using NumPy's documented default matrix-rank tolerance; preserve the full design's fitted column-space projection and verify fitted-value equivalence, logging versions and retained columns. Encode nonfinite RSS/coefficients as failed-fit status with missing A/S/T; encode zero/undefined residual status when `RSS0 <= 1e−12 * max(1,total_SS)`. Do not pass nonfinite values to Ridge; use only the training-derived handling below. Exact equal-score period ties choose the smallest candidate T. Shared Q contains only the specified HR fit-availability/zero-variance and recording-control failure statuses. Rank diagnostics and selected-period boundary flags are audit outputs only, not predictors; R remains exactly A, S and T.
- Define longest gap as the maximum number of unselected calendar dates, including both window boundaries: `first−start`, interior `next−previous−1`, and `start+179−last`. Save actual last-minus-first selected span separately.
- Use all three fixed age one-hot columns with an intercept-enabled Ridge; remove only training-zero-variance columns. For a feature wholly missing in training, omit that numeric column from every split and record it explicitly; do not learn a fill from validation/test. Preserve its shared status information where nonconstant. Rank-failed/zero-variance recording-control descriptors also receive shared-Q status flags.
- Default to the eight core heads without the optional out-of-band diagnostic. Changing that choice requires a recorded pre-run amendment before validation/test results, not a response to performance. Preserve numerical feature order and save the library/BLAS versions; runtime remains unmeasured.

## Evaluation, artifacts and completion

Primary per source: `δ_clock=AUC(B_clock+R)−AUC(B_clock)`. Secondary: δ_linear and η=δ_linear−δ_clock. Use 2000 participant resamples stratified by age×salutation, identical draws across arms; joint four-score draws estimate η. Report 95% pointwise and 97.5% per-source Bonferroni intervals for the two primary contrasts, conditional on fitted models. Include overall/within-age AUROCs and fixed-training-age-weighted within-age AUROC. Freeze weights from pooled eligible training ages across sources. Missing label support makes an age AUROC/full weighted summary not estimable; never silently reweight. Garmin elderly y0 test has only 4 people before eligibility.

Suggested checkpoint names are implementation choices: eligibility/selected-dates tables, feature schema/config, source/fill raw and processed H caches, shared Q/R/age tables, fitted transforms/preprocessors, alpha traces, train/validation/test predictions, paired/age metrics and report. Prediction keys must include source, allocation0, split, arm and participant. Record cache/input hashes, dates/mask hashes, exclusions, rank/status flags, retained columns and runtime. Persist sufficient fitted artifacts for reproduction; coefficients alone are insufficient.

Completion requires 90 unique dates/person and 15/block; mask/observed-value invariance; fold/source fitting isolation; paired-H bit equality; shared status flags; saved-score AUROC verification; explicit sparse/undefined age results; all scores/configurations saved; allocation0 only; and report numbers traceable to tables. Runtime is unmeasured. No existing monthly CLI exists: create/document one only during the implementation task.

Readiness-only commands from the repository, without running models:

```sh
/Users/bulat/micromamba/envs/datenspende/bin/python -c 'import numpy,pandas,scipy,sklearn,numba; print(numpy.__version__,pandas.__version__,scipy.__version__,sklearn.__version__,numba.__version__)'
/Users/bulat/micromamba/envs/datenspende/bin/python -c 'import pyarrow.parquet as p; print(p.read_schema("experiments/apple_cohort_scaling/artifacts/apple_10_allocations.parquet")); print(p.read_schema("experiments/garmin_temporal_binning/artifacts/garmin_binning_5_allocations.parquet"))'
```

These commands are supplied for the implementer; they were not executed as experiment checks in this documentation task.
