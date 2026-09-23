# Codex proposal: informative recording patterns as predictive features

- **Label:** `codex_informative_recording_features_v1`
- **Date:** 2026-09-20
- **Status:** proposed; not implemented or run
- **Authorship:** Codex proposal, prepared as a separate scientific protocol

This document does **not** replace or amend `MNAR_PLAN.md`, `EXPERIMENT_PLAN.md`,
`SIDEQUEST_PLAN.md`, or either feature-selection plan. It does not authorize changes to
existing analysis functions or artifacts.

## 1. Decision and scientific question

Do not continue searching for a generally superior subset or deterministic re-expression
of the existing strict-cohort features. Repeated-split experiments found no reliable
general AUROC improvement from correlation pruning, permutation top-k selection,
interactions, channel ratios, or short-versus-long-window differences.

The next question is instead:

> Among participants with recorded salutation 10 or 20, how much discrimination is carried
> by the recording process and device context, how much is added by measured heart-rate
> values, and how much apparently value-based performance depends on implicit missingness?

This is an **informative recording-pattern analysis**. It can show that availability and
recording patterns predict recorded salutation. It cannot establish an MNAR mechanism,
recover missing outcomes, identify biological sex or gender identity, or isolate a causal
physiological effect.

## 2. Why this is the next useful analysis

The frozen v1 counts show:

| cohort | salutation 10 | salutation 20 | total |
|---|---:|---:|---:|
| passes 14-day gate | 6,788 | 13,697 | 20,485 |
| below gate | 2,874 | 2,425 | 5,299 |

Thus the gate-pass rate is 0.703 for class 0 and 0.850 for class 1. Treating gate pass as a
binary score gives an AUROC of approximately

\[
\tfrac{1}{2}\{P(\mathrm{pass}\mid Y=1)+P(\mathrm{fail}\mid Y=0)\}
= \tfrac{1}{2}(0.850+0.297)=0.574.
\]

The observation process therefore contains nontrivial discrimination even before detailed
feature modelling. Conditioning on the gate removes part of that signal and changes the
target population.

## 3. Populations and estimands

Use only the frozen v1 aggregate artifacts. Do not rescan the raw epoch export.

| code | population | expected n | role |
|---|---|---:|---|
| **G** | v1 `eligible` participants | 20,485 | sustained recorders; bridge to v1, but exploratory because its test set has already informed later work |
| **U** | v1 `eligible` plus `below_coverage_gate` | 25,784 | primary population: labelled participants with at least one valid allowed-source epoch event |

Expected class counts are G = 6,788/13,697 and U = 9,662/16,122 for classes 0/1.
Map salutation 10 to class 0 and salutation 20 to class 1. Exclude missing salutation and
code 30. Preserve v1 record-level source exclusions, heart-rate limits, local-day rules,
and channel-specific adequate-day definition.

The predictive estimand is discrimination of recorded salutation under a fixed learning
procedure, conditional on each observed cohort. G and U are different populations;
`AUROC_U - AUROC_G` is not an effect of missingness-aware modelling.

## 4. Feature representations

Keys and outcomes are never predictors. Exclude user/file identifiers, absolute dates,
linked gender/sex, WHO variables, age, and BMI. Age and BMI may be used only for descriptive
balance reporting.

### 4.1 Coarse anchor block `A`

- `epoch_primary_source`
- `epoch_multisource`
- `gate_pass` in U only; omit it from G because it is constant

Also fit gate-only (U) and source-only anchors descriptively. These establish how much of
the signal is available from the two coarsest contextual variables.

### 4.2 Detailed recording-process block `R`

Use the process features specified in `MNAR_PLAN.md`, but keep source and gate in `A` so
their contribution is explicit:

- per channel: observed and adequate days, event count, span, duty cycle, events per
  observed day, mean observed hours, mean interval coverage, longest inadequate run,
  four time-of-day event fractions, number of clock hours represented, and presence;
- overall: events, unique days, span, adequate days in any channel, overall duty cycle,
  and number of core channels present.

Time-of-day event fractions use all valid events. They describe **when recording occurs**,
not heart-rate level. If timezone is not fully usable for a participant, all local-hour
fractions and the number of represented clock hours are missing rather than calculated
after substituting UTC. Raw event counts and spans may remain untransformed: monotone log
transforms do not add information to tree splits.

For interpretation, fit `R0 = gate + R` without source fields in U (`R0 = R` in G), as
well as `A + R`. The difference is descriptive evidence about explicit source information;
it is not a vendor effect.

### 4.3 Heart-rate value block `V`

Use adequate-day summaries recoverable from `epoch_days.parquet`:

- per channel: mean of daily means, SD of daily means, mean daily SD, median of daily
  medians, IQR of daily medians, minimum daily minimum, maximum daily maximum, range,
  weekday mean, weekend mean, and weekend-minus-weekday;
- channel-3000 minus channel-3001 and channel-3002 mean contrasts.

This gives 35 raw numeric features. If a channel has no adequate day, its value features
are missing rather than zero.

Do **not** claim adequate-day-only diurnal means from `epoch_hours.parquet`: that table is
aggregated by participant, channel, and clock hour and contains no date. To keep the primary
comparison clean, omit diurnal HR means. An explicitly labelled all-event diurnal
sensitivity analysis could be added later, but is not part of this protocol.

### 4.4 Resting-channel probe

- `V0`: `V` restricted to channel-3000 values.
- `P`: `V0` plus presence of channels 3001 and 3002.

`P - V0` asks whether the existence of vendor-processed resting channels contributes
without using their values. This remains a provenance/circularity probe, not a
physiological contrast.

## 5. Prespecified model arms

All arms use the same participants and split within a cohort/repeat.

| arm | predictors | preprocessing | interpretation |
|---|---|---|---|
| `Agate` | gate only; U only | none | coverage-selection anchor |
| `Asource` | source and multisource | categorical one-hot | device-context anchor |
| `A` | coarse anchor block | categorical one-hot; no value imputation | source/gate baseline |
| `R0` | gate plus detailed process in U; detailed process only in G | native missing values | recording pattern without explicit source |
| `R` | `A + R` | native missing values | detailed process beyond coarse context |
| `Vm` | `A + V` | training-median fill, **no missing indicators** | median-imputed value encoding |
| `Vn` | `A + V` | native missing values | values plus implicit missingness |
| `RV` | `A + R + V` | native missing values | full recording-aware representation |
| `V0` | `A + V0` | native missing values | channel-3000 value reference |
| `P` | `A + V0 + resting-channel presence` | native missing values | resting-channel presence probe |

In G, `A` and `Asource` are the same arm and are fit only once.

For every arm, drop all-missing and zero-variance columns using training participants only.
Fit categorical levels on training only and ignore unseen levels. Do not generate
per-column missingness indicators.

## 6. Fixed learning procedure

Use one prespecified regularized random forest for all arms:

```text
RandomForestClassifier(
    n_estimators=100,
    max_features=0.4,
    min_samples_leaf=10,
    max_depth=None,
    class_weight="balanced_subsample",
    random_state=<shared cell seed>,
)
```

This is the previously identified global G1 configuration. It avoids giving wider blocks
an unfair disadvantage from the demonstrably weak sklearn defaults. Do not tune by cohort,
source, arm, or repeat. Do not apply correlation pruning, top-k selection, or the previously
tested engineered families.

Use the same forest seed across arms within a cohort/repeat. This does not make the fitted
forests identical, but it reduces avoidable Monte Carlo variation in paired contrasts.

## 7. Repeated participant-level evaluation

The primary analysis uses **30 repeated stratified 80/20 train/test splits** in each
cohort. There is no validation partition because every model, feature block, preprocessing
rule, metric, and contrast is fixed before fitting.

- G: stratify on recorded salutation.
- U: stratify on recorded salutation by gate status.
- Train and test sets are participant-disjoint and exhaustive within every repeat.
- Every arm receives exactly the same split in a cohort/repeat.
- No feature, threshold, or model choice may use test performance.
- Report source composition in every split. Large n should make source imbalance small;
  do not redraw a split based on model performance.

Use `STUDY_SEED = 20260920`, cohort codes G = 0 and U = 1, and repeat numbers 0--29.
Derive split seeds from `SeedSequence([STUDY_SEED, 0, cohort_code, repeat])` and shared
within-repeat forest seeds from `SeedSequence([STUDY_SEED, 1, cohort_code, repeat])`.
Save the realized participant memberships and integer seeds.

The repeated-split estimand is the mean paired performance difference over the prescribed
split/model randomization, conditional on this finite cohort. It is not a confidence
interval for transport to new participants or devices.

### Primary U comparisons

1. `R - A`: detailed recording pattern beyond source and gate.
2. `RV - R`: heart-rate values beyond the recording process.
3. `RV - Vn`: recording-process information beyond values with implicit missingness.
4. `Vn - Vm`: native-NaN versus training-median value encoding under the fixed forest.

For each, report the mean paired test-AUROC delta, SD, Monte Carlo SE, median, IQR, and
share of repeats above zero. Use two-sided **98.75% t intervals**, calculated explicitly as
`t.ppf(1 - 0.05 / (2 * 4), df=29)`, to give a Bonferroni-simultaneous 95% family across the
four primary U comparisons.

G comparisons and all remaining contrasts are exploratory and use clearly labelled 95%
intervals. Important secondary contrasts are `R - R0`, `P - V0`, and `Vn - V0`.

Report absolute AUROC and balanced accuracy as mean ± SD over repeats. Accuracy,
class-specific precision/recall/F1, and confusion matrices are descriptive because the
0.5 threshold is not optimized.

## 8. Frozen-v1 bridge

Separately from the repeated-split primary analysis, fit the G arms once on the frozen v1
train/validation/test membership. Join predictions to the saved v1 test predictions and
report paired deltas as descriptive continuity with the original AUROC 0.713 result.

Because this bridge changes both the representation and the forest configuration relative
to v1, a delta against v1 cannot be attributed to features alone.

If uncertainty is reported for this bridge, use 1,000 aligned within-class participant
bootstrap draws shared across arms and v1. These draws condition on the fitted models and
frozen split; they do not represent training-set or split uncertainty. Do not select a
winner from this already-used test set.

## 9. Interpretation rules

- `R - A > 0`: detailed wear/coverage/timing patterns add information beyond source and
  gate alone.
- `RV - R > 0`: measured HR summaries add information after recording context is known.
- `RV - Vn > 0`: explicit recording-process features add information beyond implicit NaN
  patterns in the value block.
- `Vn - Vm > 0`: native-NaN encoding predicts better than training-median encoding under
  the fixed forest. This is not a pure missingness effect: median-imputed values can still
  reveal missingness through the point mass created at the training median, and the two
  encodings induce different candidate splits.
- `P - V0 > 0`: resting-channel availability adds information; it does not show that the
  resting-HR values are physiological or independently derived from salutation.
- `Vn - V0 > 0`: the full resting-channel block adds information, but this contrast mixes
  recorded values with their implicit availability pattern.

Interpret effect sizes and their consistency, not a winning arm. Source-specific metrics
may be shown descriptively where both classes are adequately represented, but sources must
not be ranked.

## 10. Hard verification checks

- Reproduce G/U sizes and class counts exactly before modelling.
- Confirm U is the disjoint union of v1 gate passers and below-gate labelled participants.
- Confirm no missing or code-30 outcomes and at least one valid allowed-source event per U
  participant.
- Assert participant-disjoint, exhaustive splits and identical membership across arms.
- Fit every preprocessing operation on training participants only.
- Confirm forbidden variables and identifiers never enter transformed matrices.
- Confirm `gate_pass` is absent from G predictors.
- Confirm adequate-day value features are NaN, not zero, when no adequate day exists.
- Confirm time-of-day recording fractions sum to one when defined.
- Confirm `Vm` has no missing indicators and `Vn` passes partial NaNs to the forest.
- Recompute every paired delta from saved participant-level predictions.
- Record seeds, package/model parameters, feature dictionaries, input hashes, and all
  exclusions.

## 11. Proposed outputs

Write only new artifacts, under a new directory such as
`artifacts_informative_recording/`. Do not mutate v1, v2, side-quest, feature-selection, or
feature-engineering artifacts.

Minimum scientific outputs:

- cohort and split summaries without participant identifiers in committed tables;
- participant-level manifests and predictions retained locally and excluded from Git;
- a feature dictionary separating `A`, `R`, `V`, and resting-channel probes;
- per-repeat absolute metrics and paired deltas;
- primary simultaneous-interval table;
- frozen-v1 bridge table;
- source and gate descriptive slices;
- a concise report distinguishing observed evidence, interpretation, and limitations.

## 12. Stopping rule

This protocol is complete after the prespecified block comparisons. Do not initiate another
round of feature selection or hand-built transformations based on these test results.
Meaningful continuation would require new measurements, new participants, an external
export, or a genuinely different scientific question.
