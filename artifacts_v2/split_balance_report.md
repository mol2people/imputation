# Split balance report

Eligible participants: 20423; train/val/test = [16339, 2042, 2042].

Target: <= 1 percentage point (pp) pairwise difference in every main category proportion. Continuous SMD < 0.05 is a supplementary target.

## Largest pairwise differences (pp)

| variable | category | max pairwise pp |
|---|---|---:|
| y_x__rarity | (0.0, '3-9') | 0.196 |
| y_x__rarity | (1.0, '3-9') | 0.190 |
| y_x__rarity | (0.0, '1-2') | 0.153 |
| y_x__rarity | (1.0, '1-2') | 0.153 |
| y_x__rarity | (0.0, '>=10') | 0.061 |
| y_x__rarity | (1.0, '>=10') | 0.055 |
| y_x_epoch_multisource | (0.0, True) | 0.055 |
| y_x_epoch_src_6 | (1.0, 0) | 0.053 |
| y_x_daily_multisource | (1.0, True) | 0.051 |
| y_x_epoch_src_3 | (0.0, 0) | 0.051 |
| y_x_daily_primary_source | (1.0, 9.0) | 0.049 |
| daily_primary_source | 6.0 | 0.049 |
| y_x_daily_primary_source | (0.0, 6.0) | 0.049 |
| y_x_epoch_primary_source | (1.0, 3.0) | 0.049 |
| y_x_daily_multisource | (0.0, False) | 0.049 |
| y_x_daily_multisource | (1.0, False) | 0.049 |
| y_x_epoch_multisource | (0.0, False) | 0.049 |
| y_x_epoch_multisource | (1.0, False) | 0.049 |
| daily_src_6 | 0 | 0.049 |
| daily_src_6 | 1 | 0.049 |

Main/interaction categories exceeding 1 pp: 0. Joint strata (size>=10) with no validation member: 0; no test member: 0. All joint strata (including sparse source-set profiles) with no validation member: 226; no test member: 224.

## Salutation-conditional margins (max pairwise pp over categories)

| variable | max pairwise pp |
|---|---:|
| daily_primary_source | 0.049 |
| epoch_primary_source | 0.049 |
| age_group | 0.036 |
| bmi_grp | 0.021 |
| daily_multisource | 0.051 |
| epoch_multisource | 0.055 |
| _rarity | 0.196 |

## Rare-profile representation (joint strata size buckets)

| bucket | train | val | test |
|---|---:|---:|---:|
| 1-2 | 192 | 24 | 24 |
| 3-9 | 385 | 48 | 48 |
| >=10 | 15762 | 1970 | 1970 |

## Continuous diagnostics

| variable | split | n | mean | SD | median | IQR | SMD vs train |
|---|---|---:|---:|---:|---:|---|---:|
| birth_year | train | 16339 | 1972.31 | 12.31 | 1970.00 | [1965.00, 1980.00] |  |
| birth_year | val | 2042 | 1972.31 | 12.44 | 1970.00 | [1965.00, 1980.00] | -0.000 |
| birth_year | test | 2042 | 1972.13 | 12.33 | 1975.00 | [1965.00, 1980.00] | 0.015 |
| epoch_span_days | train | 16339 | 476.53 | 274.53 | 453.00 | [256.00, 730.00] |  |
| epoch_span_days | val | 2042 | 470.34 | 270.41 | 432.00 | [255.25, 709.00] | 0.023 |
| epoch_span_days | test | 2042 | 476.56 | 272.45 | 460.00 | [254.25, 724.75] | -0.000 |
| adequate_days_max | train | 16339 | 289.18 | 184.39 | 284.00 | [123.00, 424.00] |  |
| adequate_days_max | val | 2042 | 287.75 | 179.67 | 286.00 | [128.00, 422.00] | 0.008 |
| adequate_days_max | test | 2042 | 288.37 | 184.20 | 280.00 | [125.00, 425.00] | 0.004 |
| valid_events | train | 16339 | 324816.97 | 378463.94 | 174729.00 | [57936.50, 427826.50] |  |
| valid_events | val | 2042 | 324495.20 | 391973.09 | 170895.00 | [56378.75, 427157.00] | 0.001 |
| valid_events | test | 2042 | 317954.61 | 367501.47 | 173175.50 | [57245.50, 427699.25] | 0.018 |

Non-significant balance tests would not establish equality; these are descriptive design diagnostics. Covariate-based allocation never used labels through epoch values, classifier results or test performance.
