# Apple bin size by recording pattern

Five-minute bins retain the higher mean AUROC across broad Apple recording-pattern groups. Their advantage becomes small when recordings are sparse both within days and across dates. The large fifteen-minute gain seen in Garmin's added-participant group does not recur here.

This exploration was added on **2026-10-15**, using existing test predictions and adequate-day caches from the [full Apple benchmark](REPORT.md). It fits no models and leaves the original outputs unchanged. The target remains recorded salutation: `10 → 0/F`, `20 → 1/M`.

## What defines a recording pattern?

The 11,139 participants have a mean of **266.4 adequate days** (median 261). Mean participant-level observed-bin occupancy is **54.0% at five minutes** and **68.3% at fifteen minutes**. Keep these three dimensions separate:

| Dimension | Definition | Approximate lower / upper tercile boundaries |
|---|---|---|
| History | Number of cached adequate days | 154 / 357 days |
| Within-day occupancy | Fraction of the 288 five-minute bins with an observed value, averaged across adequate days | 45% / 63% |
| Continuity | Adequate-day count divided by the inclusive calendar span from first to last adequate date | 54% / 96% |

Each allocation derives its boundaries from training participants and applies them to test participants. Actual lower/upper boundaries range from 151–157 / 353–361 days for history, 45.00–45.31% / 62.73–63.20% for occupancy, and 53.51–54.53% / 96.00–96.35% for continuity. Low includes values at the lower boundary; middle includes values at the upper boundary.

All descriptors use full cached history, including when evaluating the capped-history model. Thus, the same participants form each subgroup across resolutions and history budgets within an allocation. This describes existing recording profiles; it is not a rule for choosing a model from only a new user's first 40 days. Dates without an adequate record include both absent and insufficient recordings. Coverage before the first or after the last adequate date is not measured.

## Full-history comparison

Alpha is selected using each model's whole validation allocation, without subgroup-specific tuning. The executed candidate grid is `{100, 500, 1000, 5000, 10000}`. Positive differences favor five-minute bins. AUROCs are means across ten allocations; **± describes the sample SD of the paired differences**. Mean test N is per allocation.

| Recording group | Mean test N | 5-min AUROC, selected α | 15-min AUROC, selected α | Paired 5 − 15 min, selected α | Paired 5 − 15 min, fixed α = 1000 |
|---|---:|---:|---:|---:|---:|
| All participants | 1671.0 | 0.8553 | 0.8378 | +0.0175 ± 0.0082 | +0.0213 ± 0.0068 |
| History: low | 556.7 | 0.8053 | 0.7979 | +0.0074 ± 0.0179 | +0.0220 ± 0.0198 |
| History: middle | 559.1 | 0.8756 | 0.8491 | +0.0266 ± 0.0126 | +0.0263 ± 0.0105 |
| History: high | 555.2 | 0.8910 | 0.8692 | +0.0218 ± 0.0183 | +0.0165 ± 0.0124 |
| Occupancy: low | 555.2 | 0.8404 | 0.8242 | +0.0162 ± 0.0149 | +0.0210 ± 0.0136 |
| Occupancy: middle | 560.9 | 0.8571 | 0.8405 | +0.0166 ± 0.0163 | +0.0183 ± 0.0129 |
| Occupancy: high | 554.9 | 0.8609 | 0.8415 | +0.0194 ± 0.0157 | +0.0246 ± 0.0166 |
| Continuity: low | 546.4 | 0.8443 | 0.8298 | +0.0145 ± 0.0114 | +0.0165 ± 0.0109 |
| Continuity: middle | 563.7 | 0.8627 | 0.8408 | +0.0218 ± 0.0130 | +0.0264 ± 0.0114 |
| Continuity: high | 560.9 | 0.8587 | 0.8428 | +0.0159 ± 0.0103 | +0.0209 ± 0.0121 |

Five minutes leads on average in all nine one-dimensional groups under both alpha rules. Low occupancy alone does not select a fifteen-minute winner. Low continuity reduces the advantage somewhat, whereas the low-history difference shrinks substantially after alpha selection: from +0.0220 to +0.0074.

## Combinations reveal more than occupancy alone

These groups cross the same training-derived terciles. Values below use validation-selected alpha. All-day and capped models use identical test subgroup memberships.

| Recording group | Mean test N | All-day 5-min AUROC | All-day 15-min AUROC | All-day paired 5 − 15 min | First up to 40 days: paired 5 − 15 min |
|---|---:|---:|---:|---:|---:|
| Low continuity + low occupancy | 192.8 | 0.8228 | 0.8170 | +0.0058 ± 0.0211 | -0.0042 ± 0.0206 |
| High continuity + low occupancy | 127.9 | 0.8700 | 0.8454 | +0.0246 ± 0.0291 | +0.0168 ± 0.0260 |
| Low history + high occupancy | 193.1 | 0.8130 | 0.8073 | +0.0056 ± 0.0209 | +0.0001 ± 0.0276 |
| Middle history + high occupancy | 173.8 | 0.8950 | 0.8596 | +0.0354 ± 0.0161 | +0.0114 ± 0.0276 |

- **Sparse within and across days:** the all-day difference is only +0.0058, with five minutes ahead in 6/10 allocations. When capped at 40 adequate days, fifteen minutes leads by 0.0042 on average (6/10 allocations). This is a small, variable difference, not the approximately +0.09 fifteen-minute gain seen in Garmin's mixed added group.
- **Sparse within days but consistent across dates:** five minutes retains a +0.0246 all-day advantage. This group averages 40.7% five-minute occupancy, but its median adequate-day continuity is about 99%. Intermittent point readings and long gaps across dates need different descriptions.
- **Shorter history with high occupancy:** the all-day advantage is +0.0056, with five minutes ahead in only 5/10 allocations. With capped inputs the AUROCs are 0.8001 and 0.8000. This group has roughly 65 adequate days at its median, 71.2% mean five-minute occupancy, and 97.7% median continuity. Short, dense histories therefore do not show a strong five-minute preference after regularization selection.
- **Middle-length history with high occupancy:** the larger all-day five-minute advantage, +0.0354, appears in all ten allocations. The corresponding capped-history difference is +0.0114.

The profile descriptions above average the allocation-specific subgroup means or medians. Full cross-tables, including less prominent groups, are retained in the CSVs. Under selected alpha, the only negative all-day mean among the nine history × occupancy cells is low-history/middle-occupancy (−0.0018); all nine continuity × occupancy means favor five minutes. Day-to-day occupancy variability is also included in the saved tables; each of its three groups favors five minutes on average.

## Connection to Garmin and the next experiment

In the [Garmin pilot](../garmin_temporal_binning/README.md), fifteen-minute bins improve the added-participant group by **+0.0934 ± 0.0761 over five minutes** and **+0.1665 ± 0.1743 over one minute**. In the strict group, five minutes instead improves on fifteen minutes by **+0.0326 ± 0.0081**. Garmin's saved binning scores combine short dense histories with records interrupted across weeks; they do not give a separate bin-size effect for each of those patterns.

For Apple, five minutes remains the useful default among the two tested grids. Coarsening substantially increases occupied-bin fraction without improving overall discrimination. One working explanation is that five-minute inputs retain useful temporal detail, while coarsening helps most when both within-day information and continuity are limited. The current result is a pattern to pursue, not a demonstrated filling mechanism: a resolution change also changes smoothing, kernel duration in minutes, and the observation-frequency features.

A targeted next experiment would keep the saved allocations and histories, compare clock-bin filling with an explicit missingness-mask representation at both resolutions, and examine the same continuity × occupancy groups. The near-tie group and the high-continuity/low-occupancy group provide a useful contrast.

## Reproducible outputs

The [summary script](scripts/06_summarize_binning_by_recording.py) reads `artifacts/scaling_predictions_test.parquet`, `artifacts/apple_10_allocations.parquet`, `artifacts/cache_5min/customers.csv`, and the existing five-minute `.npz` caches. These are local participant-level inputs. No raw extraction, feature fitting, tuning, or model inference is repeated.

- [Summary CSV](artifacts/binning_by_recording/summary.csv): 124 comparisons across two budgets, two alpha rules, and 31 groups, with paired differences, sample SDs, and allocation win counts.
- [Per-allocation metrics](artifacts/binning_by_recording/metrics_by_allocation.csv): 1,240 rows, including subgroup test and class counts. All cells contain both classes; minimum class-0/class-1 counts are 32/71.
- [Training cutpoints](artifacts/binning_by_recording/training_cutpoints.csv): exact subgroup boundaries for each allocation.
- [Subgroup descriptors](artifacts/binning_by_recording/subgroup_descriptors.csv): history, occupancy, continuity, and gap summaries for each test subgroup.
- [Cohort descriptors](artifacts/binning_by_recording/cohort_descriptors.csv): full-cohort descriptive distribution.

The CSV outputs contain aggregates, not participant IDs. The comparisons use within-allocation AUROCs on matched participants; scores from different fitted models are not pooled.
