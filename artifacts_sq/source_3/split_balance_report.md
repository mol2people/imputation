# Source 3 split balance

N = 3,848 (2,464 sal20); 102 allocation profiles; stage-1 minimum total bound violation = 0.

| split | total | target | sal10 | sal20 |
|---|---:|---:|---:|---:|
| train | 3078 | 3078 | 1107 | 1971 |
| val | 385 | 385 | 138 | 247 |
| test | 385 | 385 | 139 | 246 |

## Marginal proportions by split (pp spread)

| variable | level | train % | val % | test % | max-min pp |
|---|---|---:|---:|---:|---:|
| age_group | Elderly | 11.47 | 11.43 | 11.43 | 0.04 |
| age_group | Middle Ager | 68.39 | 68.31 | 68.31 | 0.08 |
| age_group | Young Adults | 20.14 | 20.26 | 20.26 | 0.12 |
| bmi_grp | normal | 54.00 | 54.03 | 54.03 | 0.03 |
| bmi_grp | obese | 13.19 | 13.25 | 12.99 | 0.26 |
| bmi_grp | overweight | 31.64 | 31.69 | 31.69 | 0.04 |
| bmi_grp | underweight | 1.17 | 1.04 | 1.30 | 0.26 |
| pass_pattern | 3000 | 17.32 | 17.40 | 17.40 | 0.09 |
| pass_pattern | 3000+3001+3002 | 55.85 | 55.84 | 55.58 | 0.26 |
| pass_pattern | 3000+3002 | 26.80 | 26.75 | 27.01 | 0.26 |
| pass_pattern | 3002 | 0.03 | 0.00 | 0.00 | 0.03 |
| selected_channel | 3000.0 | 93.11 | 92.99 | 93.25 | 0.26 |
| selected_channel | 3001.0 | 0.36 | 0.52 | 0.26 | 0.26 |
| selected_channel | 3002.0 | 6.53 | 6.49 | 6.49 | 0.04 |

Occupied y x age x bmi cells: 22; max within-cell spread 0.26 pp (cell 1|Elderly|normal).

