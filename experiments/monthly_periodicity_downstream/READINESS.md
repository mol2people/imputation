# Monthly periodicity downstream: artifact readiness

Date: 2026-10-20. Planning only: no monthly extraction, feature transformation, model fitting or inference was run for this document. Counts below describe existing cohorts/splits, not eligibility for the proposed monthly windows. Existing test sets and their performance have already been inspected; allocation-0 comparisons will be exploratory.

## Cohorts and evaluation protocols

| Existing protocol | Cohort | Train / validation / test | Reusable artifacts and limits |
|---|---:|---:|---|
| Apple allocation 0 | 11,139 | 7,797 / 1,671 / 1,671 | [Allocation table](../apple_cohort_scaling/artifacts/apple_10_allocations.parquet), [reference test scores](../apple_cohort_scaling/artifacts/scaling_predictions_test.parquet), [fill/mask test scores](../apple_fill_and_mask_by_recording/results/test_predictions.parquet). Full-history daily caches exist. |
| Garmin full-history binning pilot allocation 0 | 1,000 | 700 / 150 / 150 | [Allocation table](../garmin_temporal_binning/artifacts/garmin_binning_5_allocations.parquet), [cache index](../garmin_temporal_binning/artifacts/cache_5min/customers.csv), [benchmark metrics](../garmin_temporal_binning/artifacts/garmin_binning_benchmark_results.csv). No saved participant benchmark scores were found. |
| Garmin strict r1b / recording / filling allocation 0 | 3,848 | 2,696 / 579 / 573 | [Frozen r1b splits](../r1b_dateaware_garmin_2026-09-21/cache/folds.parquet); this is the strict-cohort holdout protocol reused by the recording/filling studies. Its 40-day HR cache does not supply full-history monthly exposure. |
| Garmin strict k5 interpolation | 3,848 | Five-fold OOF, then full-cohort production fit | [Five-fold assignments](../k5_strict_garmin_interpolation/folds_5fold_disjoint.parquet), [median OOF predictions](../k5_strict_garmin_interpolation/track_40days/median/oof_predictions.csv), [three-tier OOF predictions](../k5_strict_garmin_interpolation/track_40days/three_tier/oof_predictions.csv). Production bundles are fitted on the full cohort; they are not holdout-safe allocation-0 models. |

The same strict Garmin participant cohort can have different evaluation protocols. It is incorrect either to call its five-fold OOF results allocation-0 holdout results or to say that no strict Garmin holdout split exists. Keep pilot and strict participant sets distinct; 881 pilot participants overlap the strict cohort. The single-source Apple and Garmin cache cohorts have no shared participant IDs.

## Monthly history and phenotype readiness

The [Apple full-history cache index](../apple_cohort_scaling/artifacts/cache_5min/customers.csv) covers 11,139 participants. The Garmin pilot contains 395,486 adequate participant-days, with 14–979 dates per person. These are the available full-history 5-minute caches. Apply any frozen monthly eligibility rules to actual dates and original masks; history length alone does not establish adequate cycle coverage.

The strict Garmin [day-bin cache](../temporal_garmin/cache/day_bins.npz) contains exactly 40 selected days per person. Its [day40 table](../r1b_dateaware_garmin_2026-09-21/cache/day40_table.parquet) and longer-history summary features cannot reconstruct missing full-history 5-minute sequences. Forty selected dates are insufficient for the [infradian v2 plan's](../female_infradian_cycles_v2/PLAN.md) 180-day/90-observation representation. A downstream plan may choose another estimand, but must name its history budget and identifiability limits. Adding full-history monthly information to a 40-day baseline also adds observation history.

The [existing Apple infradian participant table](../female_infradian_cycles/results/infradian_metrics_by_participant.csv) matches all 11,139 Apple IDs. Its **v1** screen retains 5,900 training, 1,265 validation and 1,260 test participants; one screened participant lacks a finite p10 fit. Those counts are not v2-window eligibility. V1 features use all available history and a selected 25–35-day band, so they cannot be relabeled as newly defined matched-window monthly features. No existing Garmin infradian feature table was found.

Apple test labels are y0/y1=579/1,092; Garmin pilot test labels are 55/95. The pilot has only four elderly y0 test participants before monthly screening. Age-stratified performance can therefore become unsupported. Apple/pilot use broad `age_group`; k5 strict artifacts use finer `age_band`. Verify metadata lineage and freeze a mapping rather than silently joining these fields. Salutation 10/20 remains the target, not verified biological sex or reproductive status.

## Feature reuse and matched comparisons

No persisted MultiRocket participant feature matrix was found in the inspected Apple fill/scaling or Garmin pilot benchmark folders. Apple runners save metrics, selections and test scores, not fitted model bundles or train/validation features; the Garmin pilot runner saves aggregate metrics. Test scores alone cannot fit a feature-augmented Ridge model. A later implementation needs controlled feature extraction and baseline parity checks, with train-only transforms and saved feature provenance. Full-cohort k5 bundles do not solve this leakage boundary.

The [transfer-pilot feature table](../apple_garmin_transfer_pilot/participant_features.csv) is reusable for a separate static-summary baseline: Apple 10,214 plus Garmin 979 participants, exact earliest 40 dates, HR/recording/combined blocks. It is not the original MultiRocket representation or full-cohort baseline.

Compare baseline and monthly-augmented models on identical eligible participants, frozen splits, histories and preprocessing. Report full-cohort baseline performance separately from the monthly-eligible subset; otherwise retention can masquerade as incremental value. Relevant existing protocols are the [Apple fill/mask plan](../apple_fill_and_mask_by_recording/PLAN.md), [Garmin filling plan](../filling_sensitivity_garmin/PLAN.md), [Garmin recording plan](../recording_structure_garmin/PLAN.md), and [strict k5 plan](../k5_strict_garmin_interpolation/PLAN.md). This inventory does not authorize or report execution of their extensions.
