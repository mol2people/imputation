# Source 6 split balance

N = 4,446 (3,073 sal20); 79 allocation profiles; stage-1 minimum total bound violation = 0.

| split | total | target | sal10 | sal20 |
|---|---:|---:|---:|---:|
| train | 3557 | 3557 | 1098 | 2459 |
| val | 445 | 445 | 138 | 307 |
| test | 444 | 444 | 137 | 307 |

## Marginal proportions by split (pp spread)

| variable | level | train % | val % | test % | max-min pp |
|---|---|---:|---:|---:|---:|
| age_group | Elderly | 13.69 | 13.71 | 13.51 | 0.19 |
| age_group | Middle Ager | 67.84 | 67.87 | 68.02 | 0.18 |
| age_group | Young Adults | 18.47 | 18.43 | 18.47 | 0.04 |
| bmi_grp | normal | 41.92 | 42.02 | 41.89 | 0.13 |
| bmi_grp | obese | 22.24 | 22.25 | 22.30 | 0.06 |
| bmi_grp | overweight | 34.58 | 34.61 | 34.46 | 0.15 |
| bmi_grp | underweight | 1.27 | 1.12 | 1.35 | 0.23 |
| pass_pattern | 3000 | 0.25 | 0.22 | 0.23 | 0.03 |
| pass_pattern | 3000+3001 | 0.03 | 0.00 | 0.00 | 0.03 |
| pass_pattern | 3000+3001+3002 | 0.06 | 0.00 | 0.00 | 0.06 |
| pass_pattern | 3000+3002 | 0.31 | 0.22 | 0.45 | 0.23 |
| pass_pattern | 3001 | 1.49 | 1.57 | 1.58 | 0.09 |
| pass_pattern | 3001+3002 | 0.79 | 0.90 | 0.68 | 0.22 |
| pass_pattern | 3002 | 97.08 | 97.08 | 97.07 | 0.01 |
| selected_channel | 3000.0 | 0.48 | 0.45 | 0.45 | 0.03 |
| selected_channel | 3001.0 | 1.88 | 2.02 | 1.80 | 0.22 |
| selected_channel | 3002.0 | 97.64 | 97.53 | 97.75 | 0.22 |

Occupied y x age x bmi cells: 23; max within-cell spread 0.26 pp (cell 1|Middle Ager|normal).

