# Codex proposal — feature selection under repeated outer holdouts

**Label:** `codex_sidequest_feature_selection_v1`

**Author:** Codex

**Date:** 2026-09-19

**Status:** Alternative proposal only; not implemented

**Relationship to active work:** This document does **not** replace or amend
`sidequest_feat_sel_repeated_splits_2026-09-19.md`, which is being implemented.
If this proposal is ever executed, it must use a separate code entry point and
artifact directory.

## 1. Objective and estimand

Estimate, separately within sources 3 (Garmin), 6 (Apple), and 7 (Samsung), how
prespecified feature-selection pipelines change participant-level test AUROC
relative to the same regularized random forest without feature selection.

The estimand is the expected paired AUROC difference over the specified random
70/15/15 split-and-fit procedure, conditional on:

- the observed strict-coverage cohort;
- the already-constructed, outcome-blind feature tables;
- the fixed model configuration and software environment.

It is **not** an external-population performance estimand, a biological effect,
or independent confirmation. This dataset and its original frozen test split
have already informed analysis choices; all results from this phase are
exploratory algorithm-development evidence.

## 2. Frozen inputs and scope

- Cohorts: the existing mutually disjoint source-3, source-6, and source-7 model
  cohorts in `artifacts_sq/split_manifest_sq.parquet`.
- Features: the existing `features_demographic.parquet`,
  `features_recording.parquet`, `features_window.parquet`,
  `features_rolling.parquet`, and `features_all.parquet`.
- Outcome: recorded salutation 10 versus 20 (`y=0` versus `y=1`).
- Variants: `demo`, `rec`, `win`, `roll_rec`, `roll_win`, and `all` under the
  existing block definitions.
- No eligibility, selected-window, or feature-value recomputation.
- No modification of the frozen split, baseline, tuned-500, or active
  feature-selection artifacts.
- No hyperparameter search.

## 3. Repeated outer partitions

Run **30 independent repeats per source**. Within each repeat, partition
participants into approximately 70% analysis-training, 15% selection-validation,
and 15% evaluation-test sets, stratified only on `y`.

Deterministic construction:

1. Sort participants by `user_id` before randomization.
2. Construct a `SeedSequence([20260919, source_id, repeat_id])` and spawn two
   recorded integer seeds.
3. Use the first seed for a stratified 15% test draw.
4. Use the second seed to draw validation from the remaining participants with
   fraction `0.15 / 0.85`; the remainder is training.
5. Require unique participant IDs, pairwise-disjoint splits, exhaustive source
   coverage, and both classes in every split.

No age/BMI/channel allocation matching is performed. Approximate per-repeat
sizes are 2,694/577/577 for source 3, 3,112/667/667 for source 6, and 379/81/81
for source 7; exact class-preserving integer counts must be saved.

All records, channels, days, and windows belonging to a participant inherit that
participant's split.

## 4. Fixed model and stochastic coupling

Use the same configuration for every source, repeat, variant, and arm:

```text
RandomForestClassifier(
    n_estimators=100,
    max_features=0.4,
    min_samples_leaf=10,
    max_depth=None,
    class_weight="balanced_subsample",
)
```

Derive a stable integer RF seed from
`(20260919, source_id, repeat_id, variant_index)`. Use that same RF seed for all
arms within the cell as a common-random-number device. This can reduce Monte
Carlo noise in paired contrasts, but it does not make fitted trees identical
when feature matrices differ.

## 5. Train/validation/test discipline

For each `(source, repeat, variant)`:

1. Fit one `SQPreprocessor` on the 70% training partition only.
2. Transform training, validation, and test with that unchanged preprocessor.
3. Derive every feature-removal rule from transformed training only, except the
   explicitly supervised A3 permutation-importance ranking, which may use the
   validation partition.
4. Freeze each arm's transformed feature-name list.
5. Fit the arm's final RF on transformed **training + validation** rows using
   the frozen training-derived preprocessor and feature list.
6. Evaluate once on that repeat's test rows.

The test partition must not be accessed for preprocessing, pruning, importance,
choice of `k`, failure recovery, or model selection. A3 validation performance is
a selection diagnostic and must not be reported as an unbiased validation
estimate.

## 6. Prespecified arms

| Arm | Rule |
|---|---|
| `A0` | Existing `SQPreprocessor` output; no additional selection. |
| `A1` | Remove all generated `__missing` terms, then remove near-zero-variance terms using caret's default-style rule on transformed training data: percent unique ≤10% and most-common/second-most-common frequency ratio >19. Constants have already been removed by `SQPreprocessor`. |
| `A2` | A1 plus deterministic correlation pruning. Compute Pearson correlation on transformed training data; scan feature names in sorted order and drop a later term when its absolute correlation with any retained term exceeds 0.95. Record every retained/dropped pair. |
| `A3_50` | Starting from A1, fit an A1 selection-stage RF on training; compute validation permutation importance with AUROC scoring and three deterministic repeats; retain the 50 highest-ranked transformed terms and refit as specified in §5. |
| `A3_100` | Same as `A3_50`, retaining 100 terms. |

For A3 ties, use transformed feature name as the final deterministic key. If A1
contains no more than `k` terms, record A3 as a no-op rather than fitting a
nominally distinct arm.

Permutation importance is a prediction perturbation diagnostic, not a causal or
physiological importance measure. Correlated predictors can divide or mask
importance; A3 is therefore secondary rather than the primary selection rule.

## 7. Primary and secondary comparisons

### Primary family

Restrict the primary family to the wide `all` variant in the two adequately
sized cohorts:

- source 3: `A2 − A0` test AUROC;
- source 6: `A2 − A0` test AUROC.

For each source, report the 30 paired deltas, their mean, SD, median, IQR,
Monte Carlo SE (`SD / sqrt(30)`), and proportion above zero. If interval language
is desired, use two-sided **97.5% t intervals** for these two means, giving a
Bonferroni simultaneous 95% family under the split-randomization interpretation.

These are intervals over the prespecified split/model randomization conditional
on the observed cohort. They do not quantify participant-sampling,
transportability, analyst-adaptivity, or external-population uncertainty.

### Secondary, explicitly exploratory summaries

- `A1 − A0` and `A2 − A1` to separate indicator/NZV hygiene from correlation
  pruning;
- `A3_50 − A0` and `A3_100 − A0`;
- all arms for `demo`, `rec`, `win`, `roll_rec`, and `roll_win`;
- all source-7 comparisons;
- accuracy, balanced accuracy, class-specific recall, selected feature count,
  runtime, and selection stability.

Do not use unadjusted secondary intervals to declare winners. Preserve
source-specific conclusions; do not pool source metrics or rank vendors.

## 8. Stability and diagnostics

For every arm and repeat, record:

- raw, transformed, and retained feature counts;
- all-missing and zero-variance terms removed by `SQPreprocessor`;
- indicator, near-zero-variance, and correlation-pruned terms;
- A3 importance seed, ranking, selected terms, and no-op status;
- pairwise Jaccard overlap of A3 selected sets across repeats;
- prediction prevalence, AUROC, balanced accuracy, accuracy, class recalls, and
  confusion matrix;
- elapsed preprocessing, selection, fit, and prediction times.

Inspect failures and feature counts without using test AUROC to modify the
protocol. Any post-result change creates a new, explicitly exploratory version.

## 9. Artifact and concurrency contract

Use a namespace that cannot collide with the active implementation:

```text
artifacts_sq/fsplit_codex/
  split_manifest.parquet
  cells/source_<s>/repeat_<r>/
    metrics.parquet
    predictions.parquet
    selections.json
    completion.json
  aggregate_metrics.parquet
  aggregate_predictions.parquet
  codex_fs_repro.json
  codex_feature_selection_report.md
```

Parallel workers must never append to a shared CSV. Each `(source, repeat)` worker
writes to a temporary cell directory, validates row counts, and atomically
renames it to the final location. A single parent process then performs a sorted,
deterministic aggregation. Resume only from a valid `completion.json` whose
input/config hashes match the current run.

Saving thousands of fitted joblib objects is unnecessary. Saved predictions,
split membership, exact feature lists, seeds, configuration, and code/input
hashes are sufficient for metric reconstruction; retain models only for a small
predeclared diagnostic subset if needed.

## 10. Executable verification contract

An independent verifier must establish:

- exactly 30 valid repeats for every source;
- participant-level disjointness and exhaustiveness within each repeat;
- exact agreement between prediction IDs and test IDs;
- no test IDs used in preprocessing or selection provenance;
- one common split per `(source, repeat)` across all variants and arms;
- one stable RF seed per `(source, repeat, variant)` across arms;
- `A2 ⊆ A1 ⊆ A0` and `A3_k ⊆ A1` feature sets;
- exact recomputation of every saved metric from predictions;
- expected row counts, uniqueness keys, no missing cells, and idempotent
  aggregation;
- hashes of this plan, the eventual driver/verifier, input feature tables,
  cohort manifest, and package versions.

## 11. Compute plan

Upper bound: one A1 selection-stage fit plus five final arm fits per
`(source, repeat, variant)`, or 3,240 RF fits before no-op reductions. There are
up to 540 permutation-importance jobs, and these may dominate wall time.

Before the full run, benchmark one `all` cell for sources 3, 6, and 7, including
permutation importance and correlation pruning. Parallelize across
`(source, repeat)` cells with RF and permutation-importance internals restricted
to one thread. Select worker count from measured peak memory and throughput; do
not promise a wall time from RF fit timing alone.

## 12. Guard for later feature engineering

No internally untouched confirmation set remains: the cohort and original test
results have already informed development. Therefore:

- freeze any feature-engineering definitions before inspecting this phase's
  evaluation-test summaries if paired comparison on the same repeats is desired;
- if feature-engineering choices are instead informed by these results, label
  the next phase adaptive and exploratory;
- reusing the same test partitions can support descriptive paired comparisons,
  but cannot be presented as independent validation;
- genuinely confirmatory evidence requires new participants or a genuinely
  external dataset.

## 13. Interpretation limits

The target is recorded salutation, not biological sex or gender identity.
Strict coverage selects sustained recorders, and source-specific channels may
contain vendor processing or acquisition-pattern information. Feature selection
can improve predictive efficiency within this selected export; it cannot isolate
physiology, establish causality, or support vendor comparisons.
