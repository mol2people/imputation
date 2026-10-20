# Computational experiments after the Apple–Garmin transfer pilot

Date: 2026-10-20. Status: research agenda; the experiments below have **not** been run. The transfer pilot and its limitations are documented in [REPORT.md](REPORT.md). This agenda concerns existing data only. Its target remains recorded administrative salutation (`10 → 0`, `20 → 1`), not independently verified biological sex or gender identity.

## Completed feasibility checks

These checks inspected saved arrays and manifest columns; they did not train models or create perturbed datasets.

| Existing input | Verified contents or result |
|---|---|
| [Garmin first-40 cache](../temporal_garmin/cache/day_bins.npz) | Arrays `hr`, `cnt`, `mask`, `user`, `date`; 153,920 day rows × 288 five-minute bins; 3,848 participants. |
| Same cache, fully observed days | 47,898 days have all 288 mask entries observed, distributed across 3,223 participants. |
| Same cache, participants with ≥14 fully observed days | 1,753 participants: 613 class 0 and 1,140 class 1. No participant has all 40 days fully observed. |
| Same cache, ≥95% observed bins/day | 94,133 days; 3,131 participants have at least 14 such days. This larger subset has residual gaps. |
| [Garmin source-specific manifest](../artifacts_sq/cohort_manifest_model_sources.parquet) | Source-3 cohort has 3,848 participants. Columns `age_group` and `bmi_grp` are nonmissing for all 1,753 participants in the ≥14-complete-day subset. |
| [Apple manifest](../apple_cohort_scaling/artifacts/apple_full_manifest.csv) joined to [cached customers](../apple_cohort_scaling/artifacts/cache_5min/customers.csv) | All 11,139 cached participants match manifest rows and have nonmissing `age_group` and `bmi_group`. The manifest contains 11,143 rows before cache qualification. |
| Same Apple manifest, cached cohort | Age categories are `Young Adults`, `Middle Ager`, `Elderly`; BMI categories are `underweight`, `normal`, `overweight`, `obese`. These are grouped covariates, not continuous age/BMI measurements. |

The Garmin cache contains the first 40 **adequate**, rather than consecutive, dates from a cohort selected using longer histories. Full-bin coverage does not establish continuous wear or complete underlying physiology. These feasibility counts describe this particular cache and selection.

Provenance of the counts: compute `mask.mean(axis=1)` or `mask.all(axis=1)` from the cache, group qualifying rows by `user`, and count participants with at least 14 qualifying days. Join those participant IDs to source 3 in the Garmin manifest for class counts and covariate completeness. Apple completeness uses the customer-ID intersection with `cache_5min/customers.csv`. The checks were performed during the 2026-10-19/20 analysis session; no separate feasibility output table was saved. The source files above permit reproduction.

## 1. Hold HR fixed and transplant Apple observation patterns

**Question.** Does the placement of Apple-like observations change apparent salutation signal beyond the effect of simply observing fewer bins?

Use the 1,753 Garmin participants with at least 14 complete days. Select a fixed, label-independent 14-day rule, such as the earliest 14 complete dates within the cached first 40 adequate days. Freeze participants, selected dates, and labels before any artificial masking. Do not reapply adequacy gates after masking: that would compare different days and participants. The estimand is restricted to this high-coverage Garmin population; it is not the effect in all Garmin users.

Use observed/missing patterns from [Apple five-minute caches](../apple_cohort_scaling/artifacts/cache_5min/). For each complete Garmin day, retain exactly the bins indicated by an independently sampled Apple donor-day mask. Compare:

1. The original complete observed record.
2. The same record under the Apple mask.
3. The same record under a uniform mask with exactly the same retained-bin count as that Apple mask.

Repeat mask assignments to measure sensitivity to donor selection and random thinning. Sample donor participants before donor days so participants with long histories do not dominate by accident. Donors must come from an explicitly separated training/development pool. Choose any additional matching rules using development data, not evaluation scores.

A factorial diagnostic separates the Garmin HR-record label from the independently assigned Apple mask-donor label. Give each Garmin record masks from both donor-label groups, matched on retained-bin count, with repeated assignments. The HR label is an observed participant attribute; the assigned mask is the computational intervention. Test whether scores follow HR-record label, assigned mask group, or their interaction. A difference between donor groups describes the influence of their sampled observation patterns; it does not mean donor salutation causes a physiological change. Matching counts alone does not match the donors' clock profiles or other recording characteristics.

**Outcomes and interpretation.** Estimate paired score shifts and discrimination differences, keeping each underlying HR record identical across masking arms. Also compare observed-only HR summaries, such as lower quantiles and day/night means, with their values on the original complete-bin record. This quantifies how observation placement changes the measurements themselves. The original observed record is the reference; no imputation is treated as ground truth.

- If uniform thinning reproduces the Apple-mask effect, reduced observation quantity is sufficient to explain that effect under the tested conditions.
- If Apple masks have an additional effect, placement or bout structure matters beyond quantity. This does not separate clock placement from local arrangement; experiment 2 addresses that ambiguity.
- If assigned donor-label groups shift scores after matching counts, the frozen model is sensitive to label-associated observation arrangements. This is evidence about model susceptibility, not a causal explanation of the original population association.

**Incomplete-record extension.** If using ≥95%-coverage Garmin days, define the realized mask as `original_mask & donor_mask`. The uniform comparator must retain the same number of bins from the same original observed support. Never manufacture observations where the Garmin record was missing. This extension no longer reproduces the original Apple mask exactly; its estimand is thinning on the available Garmin support. The complete-day experiment avoids this ambiguity.

## 2. Separate clock alignment from local arrangement

**Question.** Does discrimination depend on when patterns occur, their local ordering, or their distributions?

The existing arbitrary within-day shuffle changes both local ordering and clock alignment; it cannot isolate these effects. See the [Garmin recording-structure interpretation](../recording_structure_garmin/results/REPORT.md).

Compare the original sequence with two prespecified transformations:

- Circularly rotate HR and mask together by a randomly assigned clock offset. This preserves the day-level value multiset and cyclic bout structure while disrupting absolute clock alignment. Daily boundaries and the artificial wraparound require explicit handling; linear sequence kernels may still change at the boundary.
- Shuffle observed HR values within fixed clock windows, such as one-hour windows, leaving the observation mask unchanged. This preserves each window's observed-value multiset and exact mask while disrupting arrangement within the window. It preserves clock information only at the chosen window resolution and may induce artificial jumps.

Include a per-day distribution reference and repeat random transformations. Specify what remains fixed in each feature block: unmodified clock-profile summaries can otherwise carry the clock signal that a sequence-branch perturbation is intended to remove.

A large rotation effect with little within-window-shuffle effect would prioritize clock alignment. The converse would prioritize local arrangement within those windows. Changes under both would support neither explanation alone. These are controlled representation comparisons, not estimates of biological circadian mechanisms.

## Two distinct modelling estimands for experiments 1 and 2

**Frozen-model perturbation sensitivity:** freeze the source-trained feature transform, preprocessing, filling templates, coefficients, and selection choices; modify only the evaluation record. Measure how the existing model responds. Perturbed inputs can be outside its training distribution. A performance loss therefore establishes sensitivity, not that the removed information is indispensable to a model trained for that regime.

**Matched retraining:** apply the same masking or transformation regime to training, validation, and test records, with independent random draws; refit learned preprocessing and predictors on training participants only. Select settings on validation participants. This estimates achievable discrimination under that observation/representation regime and answers a different question. Hold participants, underlying days, observation budgets, modelling family, and selection rules matched between arms. An HR-value baseline must be explicitly defined: deleting counts and masks does not eliminate sampling information from observed HR values.

For either estimand, static event counts, duration coverage, occupied-hour counts, and clock-mask features must not silently retain unmasked information. Recompute quantities supported by the transformed data or exclude them consistently. Bin thinning cannot reconstruct raw-event duration coverage or within-bin event counts from the Apple caches. A purpose-built common-bin feature set is cleaner than presenting the original mixed reference block as fully transformed physiology.

## 3. Test within-person longitudinal stability

**Question.** Are HR profiles and model scores stable within people when history size and recording exposure are held comparable?

Use dated full-history arrays: [Apple caches](../apple_cohort_scaling/artifacts/cache_5min/) and, after inspecting its schema and eligible dates, [Garmin all-day cache directory](../temporal_garmin/cache/bins_all/). The first-40 cache alone restricts the available follow-up. Compare equal-sized chronological early/late windows within participant. Define the minimum separation and eligibility without using outcomes or model errors. Match day budget first, then weekday composition and clock occupancy; examine calendar season and elapsed separation explicitly. Season matching is possible only where the available dates support it and changes the eligible population.

Evaluate paired feature changes, score changes, rank stability, and associations between score drift and changes in recording coverage. Compare with repeated random partitions of the same participant's available days at the same window sizes. If chronological separation produces no more instability than random partitioning, evidence for temporal drift is weak. If drift attenuates after matching recording exposure, that supports a recording-related explanation, although unmeasured changes remain possible.

Use models fitted on other participants for the primary stability analysis. If a model was trained on an individual's early window and evaluated on their later window, label that as a separate known-person deployment estimand. Participant-level pooling currently discards date order; stable aggregate classification is not evidence that the underlying day-level phenotype is stable.

## 4. Standardize cohort composition before interpreting transport

**Question.** Does cross-device discrimination persist within comparable measured demographic and recording strata?

Use the actual manifest schemas: Apple `age_group`/`bmi_group`; Garmin `age_group`/`bmi_grp`. Inspect category definitions and overlap before mapping columns. Compare within-stratum AUROC where both labels are represented, and composition-standardized contrasts using a prespecified common-support population. In addition to demographic groups, match or stratify history length, within-day occupancy, and adequate-day continuity separately.

Compare explicitly defined HR-value, recording-only, and mixed representations under the same observation budget. If recording-only transfer performs similarly, or an apparent gain disappears after exposure standardization, the evidence for a shared HR phenotype weakens. Persistence supports transport of an association beyond the measured composition differences; it does not identify a biological mechanism or remove residual confounding within broad age/BMI categories. Device choice, cohort eligibility, and unmeasured activity or health characteristics remain possible explanations.

## Leakage and uncertainty boundaries

- Keep evaluation HR participants outside all predictor fitting and validation selection. Check cross-device participant-ID overlap explicitly rather than assuming source-specific cohorts imply independence.
- Keep mask donors in a training/development pool. Their labels may define the factorial manipulation, but evaluation labels must not determine mask assignment or matching choices.
- Preserve participant clustering: repeated days, donor assignments, and perturbations are not independent participants. Resample HR participants for paired uncertainty; account for donor reuse separately if generalizing beyond the finite donor pool.
- Distinguish finite-donor sensitivity, sampling uncertainty, fitting uncertainty, and selection uncertainty. Repeated random masks alone quantify only part of this uncertainty.
- Existing folds and results have been inspected repeatedly. Treat follow-ups as exploratory unless a genuinely untouched participant set or export is reserved. Do not present random resplits as independent replication.
- Prespecify a small primary contrast set. No power calculation, new split, masking dataset, stability model, matched retraining, or confirmatory analysis has been completed for this agenda.

The proposed order is controlled masking first, clock/local-arrangement controls second, then longitudinal stability and composition-standardized interpretation of transfer. These address distinct scientific uncertainties rather than merely tuning predictive performance.
