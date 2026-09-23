# Split balance report

Eligible participants: 20485; train/val/test = [16388, 2049, 2048].

Target: <= 1 percentage point (pp) pairwise difference in every main category proportion. Continuous SMD < 0.05 is a supplementary target.

## Largest pairwise differences (pp)

| variable | category | max pairwise pp |
|---|---|---:|
| y_x__rarity | (1.0, '3-9') | 0.392 |
| y_x__rarity | (0.0, '3-9') | 0.390 |
| y_x__rarity | (1.0, '1-2') | 0.292 |
| y_x__rarity | (0.0, '1-2') | 0.245 |
| y_x__rarity | (0.0, '>=10') | 0.113 |
| y_x_daily_src_19 | (1.0, 0) | 0.081 |
| y_x_epoch_src_19 | (1.0, 0) | 0.081 |
| y_x_epoch_src_9 | (1.0, 0) | 0.079 |
| y_x_daily_src_3 | (1.0, 0) | 0.072 |
| y_x_age_group | (1.0, 'Middle Ager') | 0.070 |
| y_x_epoch_src_6 | (1.0, 1) | 0.070 |
| y_x_daily_src_6 | (1.0, 1) | 0.070 |
| y_x__rarity | (1.0, '>=10') | 0.067 |
| y_x_epoch_primary_source | (0.0, 6.0) | 0.058 |
| y_x_epoch_src_3 | (1.0, 1) | 0.058 |
| y_x_bmi_grp | (1.0, 'obese') | 0.055 |
| y_x_daily_src_7 | (1.0, 1) | 0.051 |
| y_x_daily_primary_source | (1.0, 7.0) | 0.051 |
| y_x_epoch_primary_source | (0.0, 7.0) | 0.050 |
| y_x_epoch_primary_source | (1.0, 9.0) | 0.050 |

Main/interaction categories exceeding 1 pp: 0. Joint strata (size>=10) with no validation member: 0; no test member: 0. All joint strata (including sparse source-set profiles) with no validation member: 474; no test member: 473.

## Salutation-conditional margins (max pairwise pp over categories)

| variable | max pairwise pp |
|---|---:|
| daily_primary_source | 0.051 |
| epoch_primary_source | 0.058 |
| age_group | 0.070 |
| bmi_grp | 0.055 |
| daily_multisource | 0.044 |
| epoch_multisource | 0.036 |
| _rarity | 0.392 |

## Rare-profile representation (joint strata size buckets)

| bucket | train | val | test |
|---|---:|---:|---:|
| 1-2 | 403 | 51 | 50 |
| 3-9 | 657 | 82 | 82 |
| >=10 | 15328 | 1916 | 1916 |

## Continuous diagnostics

| variable | split | n | mean | SD | median | IQR | SMD vs train |
|---|---|---:|---:|---:|---:|---|---:|
| birth_year | train | 16388 | 1972.28 | 12.33 | 1970.00 | [1965.00, 1980.00] |  |
| birth_year | val | 2049 | 1972.35 | 12.25 | 1975.00 | [1965.00, 1980.00] | -0.006 |
| birth_year | test | 2048 | 1972.25 | 12.37 | 1975.00 | [1965.00, 1980.00] | 0.002 |
| epoch_span_days | train | 16388 | 487.31 | 276.23 | 474.00 | [265.00, 749.00] |  |
| epoch_span_days | val | 2049 | 487.33 | 272.81 | 476.00 | [270.00, 746.00] | -0.000 |
| epoch_span_days | test | 2048 | 481.11 | 276.40 | 472.50 | [259.75, 736.00] | 0.022 |
| adequate_days_max | train | 16388 | 289.52 | 184.48 | 284.00 | [124.00, 423.25] |  |
| adequate_days_max | val | 2049 | 284.66 | 178.29 | 277.00 | [124.00, 419.00] | 0.027 |
| adequate_days_max | test | 2048 | 289.76 | 185.70 | 286.00 | [121.00, 429.00] | -0.001 |
| valid_events | train | 16388 | 324726.93 | 380730.83 | 174423.00 | [57624.00, 426852.25] |  |
| valid_events | val | 2049 | 314754.92 | 367405.73 | 165352.00 | [54388.00, 415159.00] | 0.027 |
| valid_events | test | 2048 | 323891.32 | 372120.09 | 178736.00 | [58423.00, 434142.50] | 0.002 |

Non-significant balance tests would not establish equality; these are descriptive design diagnostics. Covariate-based allocation never used labels through epoch values, classifier results or test performance.
