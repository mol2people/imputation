# Future experiments backlog

The next experiments should distinguish recording coverage, temporal resolution and filling effects while preserving participant-level evaluation. The [results audit](../reports/results_audit_2026-10-14.md) identifies the limits of the saved benchmarks. The opening section records completed benchmarks. The remaining sections are proposals, not scheduled or running jobs. Each states a question and the comparison needed to answer it.

## Completed temporal binning benchmarks

The [Garmin pilot](../experiments/garmin_temporal_binning/README.md) is complete across five allocations. Overall test AUROC was 0.8350 at one minute, 0.8410 at five minutes and 0.8228 at 15 minutes. Delta-stratum AUROC was 0.6022, 0.6753 and 0.7687, respectively. These delta estimates use only 15–24 test participants per allocation and overlapping test sets.

The [Apple pilot](../experiments/apple_pilot_multirocket/README.md) is also complete: five-minute MultiRocket plus summaries scored 0.7554, versus 0.7431 at 15 minutes and 0.6029 for summary ridge. Allocation SDs and full results appear in the experiment README. These results motivate the proposed comparisons below; binning alone does not isolate a filling mechanism.

The [full Apple cohort benchmark](../experiments/apple_cohort_scaling/README.md) and its [recording-pattern follow-up](../experiments/apple_cohort_scaling/BINNING_BY_RECORDING.md) are also complete. Full-history five-minute MR + summaries reaches **0.8561 ± 0.0111** in 11,139 participants. See [completed Apple findings](#full-apple-cohort-scaling) below.

## Apple recording patterns and filling comparisons

Does the effect of the filling rule depend on observed recording density?

History, bin occupancy, adequate-day continuity, gap length, and daily occupancy variability are now summarized from the full Apple cohort's saved caches. Five-minute bins lead across broad groups. Under selected alpha, the all-day grids are near a tie in low-continuity/low-occupancy participants (five-minute advantage +0.0058), but five minutes retains +0.0246 among high-continuity/low-occupancy participants. Short-history/high-occupancy participants are also near a tie (+0.0056 all days; +0.0001 capped). These contrasting groups are useful targets for the next comparison.

Compare clock-bin medians and HR-plus-mask representations at five and fifteen minutes on the same participants, dates and allocations. Examine the continuity × occupancy groups already defined in the saved-score analysis. The existing five-minute summary baseline already includes 288 observation-frequency features; mask channels would preserve the daily arrangement of missingness. Additional grids and interpolation rules remain possible follow-ups; no new model fits have been run for the recording-pattern analysis.

## Cross-device transfer

How does a frozen Garmin model perform on the same Apple participants used to evaluate an Apple-trained model?

Evaluate a frozen Garmin bundle on the Apple pilot with compatible five-minute inputs and explicit device/channel parsing. Compare it with the Apple-trained benchmark on identical evaluation participants. The saved Garmin `mr_clock` model was fitted on 2,692 training participants within a 3,848-participant cohort, rather than on all 3,848.

Any transfer difference combines cohort, recording and model differences. It does not isolate hardware effects. Keep the transferred calibration separate from discrimination: validation calibration from Garmin has no established probability guarantee for Apple.

## Apple channel 3002

What comparison with channel 3000 is supported by the exported channel definition and coverage?

Verify the exported channel definition, timestamp semantics, coverage and eligible participant pool before designing a daily representation. The earlier proposed count of 11,056 users requires reconciliation with the exact source/eligibility table and target restriction. A 24-bin representation and a comparison with channel 3000 remain proposals, without assuming that an exported resting-HR summary is physiologically cleaner or more predictive.

## Full Apple cohort scaling
Completed on October 14, 2026 across ten balanced holdouts (N=11,139). See the [experiment overview](../experiments/apple_cohort_scaling/README.md) and [detailed report](../experiments/apple_cohort_scaling/REPORT.md).

Full-history five-minute AUROC is **0.8561 ± 0.0111**, compared with the pilot mean of **0.7554**. The earlier 0.8022 figure is pilot allocation 0, not a separate pilot benchmark. The summary baseline reaches 0.6474; adding MR gains **+0.2087 ± 0.0193**. Full history exceeds first-up-to-40-day inputs by **+0.0643 ± 0.0144** at fixed alpha and **+0.0442 ± 0.0122** when both select alpha. Capped inputs benefit from stronger shrinkage: +0.0193 at five minutes and +0.0350 at fifteen minutes.

The October 15 saved-score follow-up separates history length from occupancy and continuity. It finds no Garmin-sized fifteen-minute gain in Apple; coarser bins become competitive in some combined profiles. The full-history five-minute fixed-alpha model scores 0.8043 in the lowest history tercile and 0.8942 in the highest. The [complete subgroup results](../experiments/apple_cohort_scaling/BINNING_BY_RECORDING.md) also compare resolutions and alpha rules. A controlled sample-size learning curve remains a possible extension.

## Garmin Tier 2 expansion

How does the model perform in a broader Garmin population after the cohort definitions are reconciled?

The upstream source-specific allocation manifest contains **5,636** primary-Garmin participants. The current relaxed Garmin manifest marks **5,616** as Tier 2 because 20 upstream participants are absent from its `artifacts_v2` input universe. Reconcile these populations before preparing an expanded cohort; neither count is an established completed Tier 2 benchmark.

Within the current manifest, Tier 2 contains all 4,410 Tier 1c participants and adds 1,206 others. Multi-source membership does not itself establish chronological device switching. Inspect source-specific dates before attributing effects to transitions, and preserve the target restriction, source filtering and participant-level split boundaries.

## Evaluation requirements for future runs

Use saved participant assignments for matched comparisons, fit every learned transformation on training participants, and distinguish overlapping repeated holdouts from disjoint folds. Save per-participant scores and model/configuration provenance so AUROCs, paired uncertainty and subgroup denominators can be checked independently. Larger delta test groups and an evaluation cohort separate from repeated model development would resolve more uncertainty than further interpretation of the existing five small pilot splits.
