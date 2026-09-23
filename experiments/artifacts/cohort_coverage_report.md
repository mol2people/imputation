# Cohort coverage, exclusions and epoch channel provenance

## Discipline

- Source exclusions 38/46/48 applied at the record level before any count, coverage measurement or feature.
- Valid HR measurement: 25 <= longValue <= 230 bpm; startTimestamp > 0; endTimestamp < startTimestamp or missing treated as a point event.
- Adequate day (per channel): >= 8 distinct local clock hours AND (union coverage >= 8 h OR >= 60 measurements). Disjunction covers interval and point-event device representations without conflating them.
- Primary gate: >= 14 adequate days in at least one core channel; days are never pooled across channels.

## Coverage by core channel

| channel | name | participants_any_day | participants_ge14_adequate | median_adequate_days |
|---|---|---|---|---|
| 3000 | HeartRate | 25745 | 19315 | 273.0 |
| 3001 | HeartRateResting | 24752 | 18436 | 194.0 |
| 3002 | HeartRateRestingHourly | 24610 | 19965 | 213.0 |

## Epoch channel representations (sampled audit)

| channel | source | files | events | zero_duration_frac | median_duration_s | median_events_per_day | representation |
|---|---|---|---|---|---|---|---|
| 3000 | 2 | 29 | 8316 | 0.0 | 1550.0 | 1.0 | interval |
| 3000 | 3 | 59 | 29237326 | 0.0 | 60.0 | 1410.5 | interval |
| 3000 | 4 | 5 | 54398 | 0.0 | 60.0 | 50.0 | interval |
| 3000 | 6 | 137 | 16798683 | 0.95 | 0.0 | 262.0 | point-event |
| 3000 | 7 | 24 | 166319 | 0.0 | 60.0 | 17.0 | interval |
| 3000 | 9 | 18 | 1016850 | 0.0 | 1.0 | 57.75 | interval |
| 3000 | 13 | 3 | 171221 | 0.0 | 60.0 | 847.0 | interval |
| 3000 | 19 | 2 | 83177 | 0.0 | 300.0 | 94.5 | interval |
| 3001 | 2 | 20 | 322 | 0.0 | 1526.5 | 1.0 | interval |
| 3001 | 3 | 57 | 14728460 | 0.0 | 60.0 | 797.0 | interval |
| 3001 | 4 | 4 | 1145 | 0.0 | 60.0 | 24.75 | interval |
| 3001 | 6 | 137 | 5406349 | 0.934 | 46352.5 | 109.0 | point-event |
| 3001 | 7 | 23 | 103465 | 0.0 | 60.0 | 14.0 | interval |
| 3001 | 9 | 15 | 210556 | 0.0 | 1.0 | 52.5 | interval |
| 3001 | 13 | 3 | 118370 | 0.0 | 60.0 | 978.0 | interval |
| 3002 | 2 | 20 | 316 | 0.0 | 3600.0 | 1.0 | interval |
| 3002 | 3 | 57 | 332615 | 0.0 | 3600.0 | 18.0 | interval |
| 3002 | 4 | 4 | 54 | 0.0 | 3600.0 | 1.0 | interval |
| 3002 | 6 | 136 | 422157 | 0.0 | 3600.0 | 14.0 | interval |
| 3002 | 7 | 23 | 58456 | 0.0 | 3600.0 | 13.0 | interval |
| 3002 | 9 | 15 | 28905 | 0.0 | 3600.0 | 11.0 | interval |
| 3002 | 13 | 3 | 2298 | 0.0 | 3600.0 | 23.0 | interval |

## Timestamp / timezone units

- sampled_files: 240
- startTimestamp_min: 1592172169000
- startTimestamp_max: 1671306626000
- startTimestamp_span_days: 915.9080671296297
- timestamp_encoding: epoch milliseconds (16-digit, /1000 -> seconds; plausible 2020-2024 range)
- timezoneOffset_min: -420.0
- timezoneOffset_max: 300.0
- timezoneOffset_units: minutes to add to UTC to obtain local time

## Channel / source inventory among gate-passing participants

| channel | source | participants | events | days |
|---|---|---|---|---|
| 3000 | 2 | 689 | 135641 | 71121 |
| 3000 | 3 | 5679 | 2727872724 | 2228289 |
| 3000 | 4 | 162 | 1464895 | 16156 |
| 3000 | 6 | 13174 | 1652989086 | 3611582 |
| 3000 | 7 | 1769 | 16458052 | 451019 |
| 3000 | 9 | 1434 | 76849695 | 327120 |
| 3000 | 13 | 921 | 102267489 | 136494 |
| 3000 | 19 | 124 | 3382393 | 37755 |
| 3001 | 2 | 351 | 6709 | 4233 |
| 3001 | 3 | 5614 | 1393954621 | 1936156 |
| 3001 | 4 | 111 | 162631 | 3927 |
| 3001 | 6 | 13160 | 497748964 | 3436186 |
| 3001 | 7 | 1762 | 10059575 | 342131 |
| 3001 | 9 | 1374 | 18574190 | 219217 |
| 3001 | 13 | 878 | 45059058 | 94250 |
| 3002 | 2 | 351 | 6578 | 4240 |
| 3002 | 3 | 5614 | 31451780 | 1936230 |
| 3002 | 4 | 111 | 5582 | 3927 |
| 3002 | 6 | 13044 | 43263067 | 3089936 |
| 3002 | 7 | 1762 | 4625394 | 342128 |
| 3002 | 9 | 1374 | 2417804 | 219221 |
| 3002 | 13 | 878 | 1131312 | 94055 |

## Provenance caveats

- Types 3000/3001/3002 map to HeartRate / HeartRateResting / HeartRateRestingHourly in `mapping/epoch_value_types.csv`. The export establishes codes, not physiological interpretation or vendor derivation.
- Apple (source 6) emits point-event HR samples; Garmin (source 3) emits 1-minute interval buckets. Per-source semantics are not interchangeable; source availability is therefore kept as explicit predictors and as balance variables.
- Whether any processed channel (3001/3002) uses a user-entered sex/salutation in its vendor calculation is unverified from these exports. This limits causal interpretation: the experiment measures overall predictability of recorded salutation, not an isolated physiological contribution.

