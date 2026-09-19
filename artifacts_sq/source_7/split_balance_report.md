# Source 7 split balance

N = 541 (332 sal20); 24 allocation profiles; stage-1 minimum total bound violation = 0.

| split | total | target | sal10 | sal20 |
|---|---:|---:|---:|---:|
| train | 433 | 433 | 167 | 266 |
| val | 54 | 54 | 21 | 33 |
| test | 54 | 54 | 21 | 33 |

## Marginal proportions by split (pp spread)

| variable | level | train % | val % | test % | max-min pp |
|---|---|---:|---:|---:|---:|
| age_group | Elderly | 16.17 | 14.81 | 16.67 | 1.85 |
| age_group | Middle Ager | 67.21 | 66.67 | 66.67 | 0.54 |
| age_group | Young Adults | 16.63 | 18.52 | 16.67 | 1.89 |
| bmi_grp | normal | 38.34 | 38.89 | 38.89 | 0.55 |
| bmi_grp | obese | 24.94 | 25.93 | 25.93 | 0.98 |
| bmi_grp | overweight | 34.64 | 35.19 | 33.33 | 1.85 |
| bmi_grp | underweight | 2.08 | 0.00 | 1.85 | 2.08 |
| pass_pattern | 3000 | 0.23 | 0.00 | 0.00 | 0.23 |
| pass_pattern | 3002 | 99.77 | 100.00 | 100.00 | 0.23 |
| selected_channel | 3000.0 | 0.23 | 0.00 | 0.00 | 0.23 |
| selected_channel | 3002.0 | 99.77 | 100.00 | 100.00 | 0.23 |

Occupied y x age x bmi cells: 23; max within-cell spread 1.85 pp (cell 1|Young Adults|underweight).

