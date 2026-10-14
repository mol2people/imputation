# Saved results reference

This reference lists the main saved results checked on 2026-10-14. Each comparison includes its cohort, input history, and evaluation design. Implementation and output paths are in the [experiment index](../learnings/experiment_lineage_worktree.md). Verification details are in the [results audit](results_audit_2026-10-14.md).

## Terms and conventions

| Term | Meaning here |
|---|---|
| Target | Recorded salutation: `10 → 0/F`, `20 → 1/M` |
| Participant | The unit assigned to training, validation, or test |
| Allocation | One saved assignment of participants to training, validation, and test |
| Arm | One model or preprocessing condition evaluated within an experiment |
| Repeated holdouts | Several allocations of the same cohort; their test participants can overlap |
| Adequate HR day | At least eight observed local clock-hours, plus either eight hours of interval-union coverage or 60 valid readings |
| First 40 days | The first 40 adequate days in date order; not necessarily consecutive calendar days |
| Sampled days | A fixed number of adequate days drawn from the history specified by that experiment |
| All days / full history | Every adequate day available for a participant; history lengths vary between participants |
| Frozen model | A previously fitted model applied without retraining |
| Clock-bin median fill | Missing HR bins filled with training-derived medians for the corresponding clock positions |
| `±` | Sample SD across allocations, unless explicitly stated otherwise |

AUROC uses participant scores unless a row says **day-level**. SD across repeated holdouts is not a confidence interval or variation across independent cohorts. Compare models within the same experiment. Absolute scores across different experiments need not estimate the same quantity.

## Model and experiment names

| Name in reports or code | Meaning |
|---|---|
| RF | Random forest |
| MR / MultiRocket | Sequence transform; the reported temporal arms also include summary features |
| B40 | The historical 305-feature participant summary and observation-mask block; later models apply the same recipe to other history lengths |
| Profile24 / P24 | Hourly HR profile summaries appended to B40 |
| Profile288 | Five-minute HR profile summaries appended to B40 |
| R1a | Circadian-feature experiment; its feature block is evaluated alongside that study’s BASE model |
| R1b / P40 | Date-aware experiment and its participant-level feature block; P40 is distinct from P24 and B40 |
| BASE | The baseline defined by the named experiment; do not assume that BASE and B40 are interchangeable |
| Combined | B40 + MultiRocket + HYDRA in the first-40-day temporal experiment |
| `win` | The named default RF arm quoted in the source-specific summary |
| `mr_clock` / `mr_global` | Saved Heartbeat Lab bundles using clock-bin / global median filling |

For feature construction, see the [MultiRocket model notes](../learnings/multirocket_garmin_synthesis.md). For each study’s original naming, see its report in the [experiment index](../learnings/experiment_lineage_worktree.md).

## Cohorts

Device provenance is recorded as Garmin **source 3**, Apple **source 6**, and Samsung **source 7**. Core channels are HR **3000**, resting HR **3001**, and hourly resting HR **3002**. The source identifies the recording origin; the channel identifies the exported quantity.

Pooled v1 includes **20,485** participants; pooled v2 includes **20,423**. Base eligibility requires a retained daily source, valid epoch measurements, and recorded salutation 10/20. It also requires at least 14 adequate days in one core channel. Source-specific analyses then select participants with exactly one retained epoch source.

| Source | Base v2 single-source cohort | Strict cohort |
|---|---:|---:|
| Garmin / 3 | 4,410 | 3,848 |
| Apple / 6 | 11,143 | 4,446 |
| Samsung / 7 | 1,498 | 541 |

Source: [source_cohorts.csv](../experiments/artifacts_sq/source_cohorts.csv).

The strict gate searches 13-week windows. A qualifying window contains at least 11 adequate weeks and 65 strict days. An adequate week contains at least five strict days. A strict day has at least 12 hours of interval-union coverage in the cached channel data.

Each core channel is assessed separately. The final strict cohort is the union of participants who pass on any core channel. Intervals are assigned to their start date without midnight clipping, so the cached coverage quantity can exceed 24 hours.

| Garmin label in code | Cohort |
|---|---|
| Tier 0 | Strict cohort, N=3,848 |
| Delta | Added participants who failed the strict gate, N=562 |
| Tier 1c | Strict cohort plus added participants, N=4,410 |

The adequate-day rule differs from the strict longitudinal gate. Forty adequate days need not be consecutive calendar days. The strict Garmin full-history cache contains 80–979 adequate days per participant. Eighty is the observed minimum, not another eligibility cutoff.

## Garmin first 40 adequate days

**Cohort:** 3,848 strict Garmin participants. **History:** first 40 adequate days. **Evaluation:** ten repeated holdouts, each with 2,696/579/573 training/validation/test participants.

| Model | Mean test AUROC across allocations |
|---|---:|
| Summary ridge | 0.7157 |
| Summary RF | 0.7031 |
| Hourly profile ridge | 0.7370 |
| MultiRocket | 0.8361 |
| HYDRA | 0.8442 |
| B40 + MultiRocket + HYDRA | 0.8565 |
| MultiRocket with within-day HR values shuffled | 0.6643 |

B40 contains 305 summary and observation-mask features. MultiRocket and HYDRA add sequence features to B40. The combined model does not include the separate profile blocks.

These are the corrected v2 results. Preprocessing is fitted within each allocation; regularization was selected on allocation-0 validation and reused. Some later test participants belonged to that earlier validation set.

Sources: [report](../experiments/temporal_garmin/results/REPORT.md), [metrics](../experiments/temporal_garmin/results/temporal_metrics.csv).

## Garmin day budgets

**Cohort:** the same 3,848 participants. **History:** days sampled across available adequate history. **Evaluation:** the same ten allocations. **Model:** MultiRocket plus B40, ridge alpha 1000.

| Days per participant | Test AUROC, mean ± SD |
|---|---:|
| 40 | 0.8583 ± 0.0115 |
| 80 | 0.8854 ± 0.0115 |
| All, mean 425 | 0.9264 ± 0.0055 |

The all-days endpoint reuses allocation-0 fill values and MultiRocket biases across all allocations. Later test participants therefore contributed to fitted preprocessing. Its effect on AUROC is unknown. This endpoint also uses a variable history budget and one kernel draw.

Sources: [day-budget report](../experiments/temporal_garmin/results/DAYS_SCALING_REPORT.md), [metrics](../experiments/temporal_garmin/results/days_scaling_all_metrics.csv).

## Saved Heartbeat Lab models

**Cohort:** 3,848 strict Garmin participants. **History:** all adequate days. **Evaluation:** one 2,692/578/578 split.

| Saved bundle | Fill | Validation AUROC | Test AUROC |
|---|---|---:|---:|
| `mr_clock` | Clock-bin median | 0.8962 | 0.9177 |
| `mr_global` | Global median | 0.9034 | 0.9225 |

Both models fit preprocessing on training participants. They use different fitting choices from the historical endpoint and cannot quantify its evaluation bias.

Sources: local `HEARTBEAT_LAB/artifacts/models/`, [Lab audit](../HEARTBEAT_LAB/AUDIT_REPORT.md), [model notes](../learnings/multirocket_garmin_synthesis.md).

## Expanded Garmin cohort

**Cohort:** 4,410 single-source Garmin participants. **History:** all adequate days. **Evaluation:** ten repeated holdouts, each with 662 test participants.

| Model and test population | Test AUROC, mean ± SD |
|---|---:|
| MultiRocket, all participants | 0.9094 ± 0.0058 |
| MultiRocket, strict cohort within test | 0.9235 ± 0.0069 |
| MultiRocket, added participants within test | 0.8156 ± 0.0448 |
| Summary ridge, all participants | 0.7196 ± 0.0179 |

The mean paired MultiRocket minus summary difference is **+0.1898**. Source: [Tier1c metrics](../HEARTBEAT_LAB/artifacts/tier1c_cv_results/tier1c_10_alloc_results.csv).

A separate evaluation applies the frozen strict-cohort `mr_clock` model to all 562 added participants. Its AUROC is **0.7850**. That evaluation and expanded-cohort retraining use different models and test samples. Their difference does not isolate the effect of retraining. Source: [frozen-model results](../HEARTBEAT_LAB/artifacts/eval_delta_562/delta_evaluation_report.json).

## Garmin and Apple bin-size pilots

Each pilot has **1,000 participants** and five repeated 700/150/150 holdouts. Both use all adequate days. Garmin histories range from 14–979 days, mean 395.5; Apple histories range from 14–887 days, mean 264.9.

| Model | Garmin test AUROC, mean ± SD | Apple test AUROC, mean ± SD |
|---|---:|---:|
| MultiRocket, 1 minute | 0.8350 ± 0.0369 | Not evaluated |
| MultiRocket, 5 minutes | 0.8410 ± 0.0380 | 0.7554 ± 0.0594 |
| MultiRocket, 15 minutes | 0.8228 ± 0.0411 | 0.7431 ± 0.0437 |
| Summary ridge | 0.7143 ± 0.0360 | 0.6029 ± 0.0631 |

The populations are not matched across devices. Their source cohorts, recording histories, and sampling procedures differ.

The Garmin added-participant test groups contain only 15–24 participants per allocation. Their mean AUROCs are 0.6022, 0.6753, and 0.7687 at 1, 5, and 15 minutes, respectively. The paired 15-minute minus one-minute difference is **+0.1665 ± 0.1743 SD**.

The Apple five-minute minus summary difference is **+0.1526 ± 0.0423 SD**. All 1,000 sampled participants produced adequate-day caches. This retention result applies to the selected pilot, not the full source population.

Tier1c and both pilots save allocation metrics without participant predictions. Their reported means and SDs can be checked, but their AUROCs cannot be reconstructed independently from saved scores.

Sources: [Garmin report](../experiments/garmin_temporal_binning/README.md), [Apple report](../experiments/apple_pilot_multirocket/README.md). See [recording-pattern notes](../learnings/wear_phenotypes_and_missingness_synthesis.md) for interpretation.

## Earlier summary and feature experiments

| Experiment | Cohort and evaluation | Saved result |
|---|---|---|
| [Pooled v1](../experiments/artifacts/model_report.md) | N=20,485; one 80/10/10 split | Test AUROC 0.7132 |
| [Pooled v2](../experiments/artifacts_v2/model_report.md) | N=20,423; one 80/10/10 split | Test AUROC 0.6833 |
| [Source-specific default RF](../experiments/artifacts_sq/sidequest_report.md) | Strict cohorts; one split per source | `win` arm: Garmin 0.6918; Apple 0.6814; Samsung 0.6609 |
| [Informative recording](../experiments/2026-09-20_informative_recording_features/README.md) | Ungated N=25,784; 30 repeated splits | Recording + values 0.7239; paired gain over native-NaN values +0.0167 |
| [Balanced-allocation R1a](../experiments/r1a_allocfolds_garmin_2026-09-21/results/REPORT.md) | Strict Garmin; ten allocations | BASE 0.7415; BASE + R1a 0.7509 |
| [Date-aware R1b](../experiments/r1b_dateaware_garmin_2026-09-21/results/REPORT.md) | Strict Garmin; ten allocations | Day-level A40 0.6589; C40 0.6782; participant BASE + P40 0.7513 |
| [Recording controls](../experiments/recording_structure_garmin/results/REPORT.md) | Strict Garmin; first 40 adequate days; three allocations | Mask-only temporal model 0.6166; count-only temporal model 0.6332 |
| [Filling controls](../experiments/filling_sensitivity_garmin/results/REPORT.md) | Strict Garmin; first 40 adequate days; three allocations | Four fitted pipelines: mean test AUROC 0.8464–0.8485 |

Pooled v1 and v2 change both cohort and predictors. The source-specific `win` scores describe one named arm, not the best arm for each source. R1b BASE contains longer-history features; BASE + P40 is not a 40-day-only model.

The [experiment index](../learnings/experiment_lineage_worktree.md) also links RF tuning, feature selection, feature engineering, age, day-level aggregation, and the broader 50/50 source experiment.
