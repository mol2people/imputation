# Recording coverage and temporal binning across Garmin and Apple

The useful question is whether the preferred bin size changes with the recording pattern. Garmin's added-participant group benefits from fifteen-minute bins; in the full Apple cohort, five minutes leads across broad history and coverage groups, with near ties in some combined profiles. History length, within-day occupancy, and continuity across dates provide separate views of missingness. Target codes retain their original meaning: `10 → 0/F`, `20 → 1/M`.

See the [results audit](../reports/results_audit_2026-10-14.md) for the wider evaluation and provenance review.

## Garmin cohort expansion

Tier 1c contains 4,410 selected single-source Garmin participants: 3,848 from the earlier strict cohort and 562 who failed that strict longitudinal gate. The strict gate concerns an eligible 13-week window; it does not establish continuous wear throughout each participant's full history. The later preparation uses a different adequate-day rule: at least eight observed clock hours and either 60 events or eight hours of interval-union coverage.

The following summaries use **adequate dates in the saved delta cache**, rather than enrollment or all observed dates. Span includes both endpoint dates. Maximum separation is the difference between consecutive adequate dates, so it is one day larger than the number of intervening dates. Active clock hours describe adequate days only.

| Earlier strict-gate failure reason | Participants | Median adequate days | Median adequate-date span | Median adequate-day density | Median maximum date separation | Median participant mean active hours |
|---|---:|---:|---:|---:|---:|---:|
| Fewer than 11 adequate weeks in a qualifying 13-week window | 432 | 182.0 | 460.5 days | 0.413 | 82.5 days | 14.3 h |
| Recorded span shorter than 13 weeks | 128 | 38.5 | 46.5 days | 1.000 | 1.0 day | 22.9 h |
| Fewer than 65 strict days among week-qualifying windows | 2 | 202.0 | 402.0 days | 0.556 | 120.5 days | 14.8 h |

The two participants in the last row have 140 and 264 adequate days; describing them as empty non-wearers is incorrect. Longer gaps between adequate dates can reflect absent or inadequate recordings. They do not establish off-wrist duration, abandonment or why a participant stopped contributing.

## Frozen model evaluation on delta

All 562 delta participants are absent from the training, validation and test assignments of the saved strict-cohort Heartbeat Lab bundles. Frozen-model results from [`delta_predictions.csv`](../HEARTBEAT_LAB/artifacts/eval_delta_562/delta_predictions.csv) are:

| Model | Delta AUROC | Stratified participant bootstrap 95% interval |
|---|---:|---:|
| Summary ridge | 0.6272 | 0.5751–0.6755 |
| MultiRocket clock-bin fill | **0.7850** | 0.7464–0.8218 |
| MultiRocket global fill | 0.7757 | 0.7398–0.8138 |

The intervals condition on these fitted models and sampled participants; they do not include training uncertainty. The clock-fill model scores 0.8559 among the 128 short-span cases and 0.7599 among the 432 week-count failures. Different history lengths and recording distributions prevent attributing that contrast to a single wear mechanism. The two remaining cases cannot support useful subgroup inference despite their empirical bootstrap interval of 1.0–1.0.

The subsequent Tier 1c experiment uses all cached adequate days and refits all preprocessing and models within ten overlapping repeated holdouts: mean test AUROC **0.9094 ± 0.0058** overall, **0.9235 ± 0.0069** in strict participants and **0.8156 ± 0.0448** in delta. These are allocation means ± sample SD, not confidence intervals. The difference between delta 0.8156 and frozen-model 0.7850 is not a controlled estimate of the benefit of adding delta training participants: training sets, models and evaluation memberships differ, and new per-participant predictions were not saved.

## Garmin temporal binning

The [completed 1,000-participant pilot](../experiments/garmin_temporal_binning/README.md) uses 881 strict and 119 delta participants, with five overlapping 700/150/150 allocations.

| Resolution | Overall AUROC | Strict AUROC | Delta AUROC |
|---|---:|---:|---:|
| 1 minute | 0.8350 ± 0.0369 | 0.8645 ± 0.0335 | 0.6022 ± 0.2150 |
| 5 minutes | **0.8410 ± 0.0380** | **0.8647 ± 0.0319** | 0.6753 ± 0.1245 |
| 15 minutes | 0.8228 ± 0.0411 | 0.8320 ± 0.0344 | **0.7687 ± 0.0753** |

Fifteen minutes exceeds five minutes in four of five delta allocations, with paired gain **+0.0934 ± 0.0761**, and exceeds one minute in all five, with gain **+0.1665 ± 0.1743**. In strict participants, five minutes exceeds fifteen minutes in all five allocations: **+0.0326 ± 0.0081**. Delta test groups contain 15–24 participants. These saved scores combine the short dense histories and intermittent records described above; a separate bin-size advantage for either pattern was not saved.

Raw events are binned before filling. Coarser bins combine nearby observations but leave bins without readings missing. Mean participant-level delta occupancy changes from **64.7%** at one minute to **65.3%** at five minutes and **66.5%** at 15 minutes; multi-hour gaps therefore remain. Strict participants average **89.6%** occupancy at five minutes in this pilot.

Resolution changes the physical time spanned by kernels, smoothing, mask-feature width and missingness together. Clock-bin median filling produces a time-varying diurnal template; it is not generally a flat line with zero differences throughout gaps. The benchmarks do not isolate which of these changes explains discrimination differences.

## Full Apple cohort: bin size within recording patterns

The [full Apple benchmark](../experiments/apple_cohort_scaling/README.md) includes 11,139 participants. Five-minute MultiRocket plus summaries scores **0.8561 ± 0.0111** at fixed alpha, versus **0.8348 ± 0.0079** at fifteen minutes. Mean occupancy rises from **54.0% to 68.3%** with coarsening, but discrimination decreases. More populated bins alone therefore do not explain the preferred resolution.

The [saved-prediction exploration](../experiments/apple_cohort_scaling/BINNING_BY_RECORDING.md) defines low/middle/high groups using training-derived terciles of full-history adequate-day count, five-minute occupancy, and adequate-day continuity. Continuity is the fraction of dates between the first and last adequate day that are themselves adequate. Approximate tercile boundaries are 154/357 adequate days, 45%/63% occupancy, and 54%/96% continuity. The same groups are used for all-days and capped-history comparisons.

Five minutes has higher mean AUROC in all nine broad groups under both fixed and validation-selected alpha. The combinations are more informative:

| Full-history recording profile | Mean test N | 5-min AUROC | 15-min AUROC | Paired 5 − 15 min, mean ± SD |
|---|---:|---:|---:|---:|
| Low continuity + low occupancy | 192.8 | 0.8228 | 0.8170 | +0.0058 ± 0.0211 |
| High continuity + low occupancy | 127.9 | 0.8700 | 0.8454 | +0.0246 ± 0.0291 |
| Low history + high occupancy | 193.1 | 0.8130 | 0.8073 | +0.0056 ± 0.0209 |
| Middle history + high occupancy | 173.8 | 0.8950 | 0.8596 | +0.0354 ± 0.0161 |

These are all-days models with validation-selected alpha. The low-history/high-occupancy group has about 65 adequate days at its median and 98% median continuity, so it captures the short, consistently populated pattern of interest. Yet its two resolutions are close, and essentially tied at 0.8000 when inputs are capped at 40 adequate days. The larger five-minute advantage occurs in the middle-history/high-occupancy group.

For records sparse both within and across days, fifteen minutes is competitive: the capped models slightly favor fifteen minutes (+0.0042), whereas all-days models slightly favor five minutes (+0.0058). Sparse within-day recording with high continuity retains a larger five-minute advantage. The current working hypothesis is that the combination of continuity and within-day information changes the value of temporal detail; sparse bins alone are insufficient to choose a grid.

One targeted follow-up is to compare filling and explicit mask channels within these same groups, retaining the saved participants, dates, and allocations. This would test whether the near ties reflect missing-data handling or loss of temporal detail through coarsening. The new analysis reads existing caches and predictions only; complete cross-tables are saved alongside the linked report.

## Apple pilot

The [Apple pilot](../experiments/apple_pilot_multirocket/README.md) sampled 1,000 participants from an already eligible single-source pool. All passed its preparation gate of at least 14 days with eight observed clock hours and 60 events per day. Mean adequate history was 264.9 days; mean participant-level occupancy was 54.1% at five minutes and 68.3% at 15 minutes.

Across five overlapping repeated holdouts, MultiRocket plus summaries scored **0.7554 ± 0.0594** at five minutes and **0.7431 ± 0.0437** at 15 minutes. Summary ridge scored **0.6029 ± 0.0631**. The paired five-minute increment over summary ridge was **+0.1526 ± 0.0423**. The summary includes observation-frequency features, so the increment is over a combined HR-and-recording baseline.

Nighttime absence of exported readings cannot identify charging, sleep tracking or actual wear. Earlier night-coverage percentages and behavioral group labels lack a reproducible saved analysis in this experiment and are not retained as cohort estimates. Likewise, structured missingness alone does not establish its statistical missingness mechanism relative to unobserved HR or salutation.

## Implications for follow-up work

Describe recording patterns using observed coverage, gap duration and night-reading frequency, with explicit denominators and adequate-day definitions. Compare filling rules and mask representations on the same participants, dates and allocations, fitting every learned step within training. Reserve causal claims about imputation and physiological or behavioral interpretations for designs that isolate those hypotheses. Cross-device comparisons also require attention to cohort selection, history lengths and training sample size.
