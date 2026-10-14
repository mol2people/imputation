# Future experiments backlog

The next experiments should distinguish recording coverage, temporal resolution and filling effects while preserving participant-level evaluation. The [results audit](../reports/results_audit_2026-10-14.md) identifies the limits of the saved benchmarks. The opening section records completed benchmarks. The remaining sections are proposals, not scheduled or running jobs. Each states a question and the comparison needed to answer it.

## Completed temporal binning benchmarks

The [Garmin pilot](../experiments/garmin_temporal_binning/README.md) is complete across five allocations. Overall test AUROC was 0.8350 at one minute, 0.8410 at five minutes and 0.8228 at 15 minutes. Delta-stratum AUROC was 0.6022, 0.6753 and 0.7687, respectively. These delta estimates use only 15–24 test participants per allocation and overlapping test sets.

The [Apple pilot](../experiments/apple_pilot_multirocket/README.md) is also complete: five-minute MultiRocket plus summaries scored 0.7554, versus 0.7431 at 15 minutes and 0.6029 for summary ridge. Allocation SDs and full results appear in the experiment README. These results motivate the proposed comparisons below; binning alone does not isolate a filling mechanism.

## Apple recording patterns and filling comparisons

Does the effect of the filling rule depend on observed recording density?

Describe observed nighttime reading frequency, bin occupancy, adequate-day density and gap lengths with reproducible saved outputs. Use neutral recording-pattern labels: absent exported readings do not establish charging or off-wrist behavior.

Compare clock-bin medians, justified interpolation rules and HR-plus-mask representations on the same participants, dates and allocations. Keep long gaps explicit and avoid interpreting interpolated values as recovered measurements. Fifteen-minute binning is already benchmarked; coarser grids and alternative fills remain proposed. Assess their interaction with recording density rather than assuming a performance benefit. The existing five-minute summary baseline already includes 288 observation-frequency features; adding mask channels tests a richer representation of missingness.

## Cross-device transfer

How does a frozen Garmin model perform on the same Apple participants used to evaluate an Apple-trained model?

Evaluate a frozen Garmin bundle on the Apple pilot with compatible five-minute inputs and explicit device/channel parsing. Compare it with the Apple-trained benchmark on identical evaluation participants. The saved Garmin `mr_clock` model was fitted on 2,692 training participants within a 3,848-participant cohort, rather than on all 3,848.

Any transfer difference combines cohort, recording and model differences. It does not isolate hardware effects. Keep the transferred calibration separate from discrimination: validation calibration from Garmin has no established probability guarantee for Apple.

## Apple channel 3002

What comparison with channel 3000 is supported by the exported channel definition and coverage?

Verify the exported channel definition, timestamp semantics, coverage and eligible participant pool before designing a daily representation. The earlier proposed count of 11,056 users requires reconciliation with the exact source/eligibility table and target restriction. A 24-bin representation and a comparison with channel 3000 remain proposals, without assuming that an exported resting-HR summary is physiologically cleaner or more predictive.

## Full Apple cohort scaling

How does performance change with training sample size at a fixed evaluation population and history budget?

The sampling script identifies an upstream eligible single-source candidate pool of 11,143. Establish preparation retention for the full pool before treating that number as a final modeling cohort; 100% retention in the already eligible pilot does not prove full-pool retention.

A learning curve with fixed evaluation participants and matched history budgets would address sample-size effects more directly than one larger fit. Fit all preprocessing within each allocation and save participant predictions, settings and input provenance. Keep development comparisons on validation data, with a prespecified final evaluation where possible.

## Garmin Tier 2 expansion

How does the model perform in a broader Garmin population after the cohort definitions are reconciled?

The upstream source-specific allocation manifest contains **5,636** primary-Garmin participants. The current relaxed Garmin manifest marks **5,616** as Tier 2 because 20 upstream participants are absent from its `artifacts_v2` input universe. Reconcile these populations before preparing an expanded cohort; neither count is an established completed Tier 2 benchmark.

Within the current manifest, Tier 2 contains all 4,410 Tier 1c participants and adds 1,206 others. Multi-source membership does not itself establish chronological device switching. Inspect source-specific dates before attributing effects to transitions, and preserve the target restriction, source filtering and participant-level split boundaries.

## Evaluation requirements for future runs

Use saved participant assignments for matched comparisons, fit every learned transformation on training participants, and distinguish overlapping repeated holdouts from disjoint folds. Save per-participant scores and model/configuration provenance so AUROCs, paired uncertainty and subgroup denominators can be checked independently. Larger delta test groups and an evaluation cohort separate from repeated model development would resolve more uncertainty than further interpretation of the existing five small pilot splits.
