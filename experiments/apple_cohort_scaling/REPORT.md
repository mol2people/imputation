# Full Apple Watch Cohort Benchmark Report

**Study date:** 2026-10-14

**Target:** Recorded salutation (`10 → 0/F`, `20 → 1/M`)

**Cohort:** Qualified Single-Source Apple Watch (Source 6) participants with $\ge 14$ adequate days ($N = 11139$ qualified from $11,143$ upstream candidates).

**Evaluation:** 10 repeated OR-Tools balanced 70/15/15 holdouts ($N_{\text{train}} \approx 7797$, $N_{\text{val}} \approx 1671$, $N_{\text{test}} \approx 1671$).

**History:** Mean 266.4 adequate days, median 261, range 14–902. The 925 participants (8.30%) with 14–39 adequate days have identical input histories in both budget arms.

**Executed alpha grid:** `{100, 500, 1000, 5000, 10000}`; this differs from the originally proposed grid in `PLAN.md`.

**Follow-up:** [Bin size by recording pattern](BINNING_BY_RECORDING.md), added on 2026-10-15 from existing scores and caches.

---

## 1. Executive Summary

This study extends the within-day sequence classification of wearable heart-rate records from the $N=1,000$ pilot to the full qualified single-source Apple Watch population. All preprocessing, clock-bin median fills, MultiRocket bias quantiles, and Ridge regression models were fitted strictly on training participants within each allocation.

### Primary Benchmark Results (10 Allocations, Mean ± Sample SD)

| Representation | History Budget | Resolution | Fixed $\alpha=1000$ Test AUROC | Validation-Selected $\alpha$ Test AUROC | Fixed-α Val AUROC | Fixed-α Test Balanced Accuracy |
|---|---|---|---:|---:|---:|---:|
| MultiRocket + summaries | all_days | 5m | 0.8561 ± 0.0111 | 0.8553 ± 0.0123 | 0.8554 | 0.7829 ± 0.0115 |
| MultiRocket + summaries | all_days | 15m | 0.8348 ± 0.0079 | 0.8378 ± 0.0106 | 0.8324 | 0.7588 ± 0.0071 |
| MultiRocket + summaries | up_to_40d | 5m | 0.7918 ± 0.0148 | 0.8110 ± 0.0131 | 0.7869 | 0.7186 ± 0.0126 |
| MultiRocket + summaries | up_to_40d | 15m | 0.7636 ± 0.0113 | 0.7985 ± 0.0097 | 0.7611 | 0.6913 ± 0.0104 |
| Summary | all_days | 5m | 0.6474 ± 0.0180 | 0.6468 ± 0.0151 | 0.6449 | 0.6110 ± 0.0165 |
| Summary | up_to_40d | 5m | 0.6298 ± 0.0164 | 0.6340 ± 0.0143 | 0.6280 | 0.5923 ± 0.0138 |

---

## 2. Paired Differences (Within-Allocation Deltas)

- **MultiRocket increment over summary baseline (5m, All days, Fixed $\alpha$):**
  $\Delta = +0.2087 \pm 0.0193$
- **Resolution effect (5m vs 15m, MultiRocket All days, Fixed $\alpha$):**
  $\Delta = +0.0213 \pm 0.0068$
- **History budget effect (All days vs First up to 40 days, MultiRocket 5m, Fixed $\alpha$):**
  $\Delta = +0.0643 \pm 0.0144$
- **Regularization selection effect on full history (Selected vs Fixed $\alpha$, MultiRocket 5m All days):**
  $\Delta = -0.0008 \pm 0.0018$
- **Regularization selection effect on short history (Selected vs Fixed $\alpha$, MultiRocket 5m $\le$ 40d):**
  $\Delta = +0.0193 \pm 0.0055$
- **Regularization selection effect on coarse short history (Selected vs Fixed $\alpha$, MultiRocket 15m $\le$ 40d):**
  $\Delta = +0.0350 \pm 0.0072$
- **History budget effect after selecting alpha for both arms (All days vs first up to 40 adequate days, MultiRocket 5m):**
  $\Delta = +0.0442 \pm 0.0122$
- **Resolution effect after selecting alpha for both grids (5m vs 15m, MultiRocket all days):**
  $\Delta = +0.0175 \pm 0.0082$

Full-history five-minute models selected alpha 1,000 in eight allocations and 5,000 in two. Capped five-minute models selected 5,000 in seven and 10,000 in three; capped fifteen-minute models selected 10,000 in all ten. The paired SDs describe allocation variation.

---

## 3. Subgroup Stratification (MultiRocket 5-min All Days, Fixed $\alpha$)

| Dimension | Subgroup | Mean $N$ | Test AUROC (Mean ± SD) | Class 0 (F) Recall | Class 1 (M) Recall |
|---|---|---:|---:|---:|---:|
| age_group | Elderly | 203.0 | 0.8163 ± 0.0315 | 0.7573 | 0.7463 |
| age_group | Middle Ager | 1120.7 | 0.8581 ± 0.0128 | 0.7774 | 0.7904 |
| age_group | Young Adults | 347.3 | 0.8713 ± 0.0155 | 0.7986 | 0.7996 |
| history_tercile | T1_Low | 556.7 | 0.8043 ± 0.0186 | 0.7343 | 0.7412 |
| history_tercile | T2_Mid | 559.1 | 0.8769 ± 0.0101 | 0.7925 | 0.8099 |
| history_tercile | T3_High | 555.2 | 0.8942 ± 0.0148 | 0.8092 | 0.8103 |

---

## 4. Verification and Audit

- **Score reconstruction:** 100% of test predictions in `scaling_predictions_test.parquet` independently reproduce `apple_benchmark_metrics.csv` within $10^{-10}$.
- **Leakage verification:** MultiRocket biases and fill vectors strictly derived from training customer IDs. Test participants did not influence preprocessing, scaling, alpha selection, or threshold calibration.

## 5. Bin Size by Recording Pattern

The [saved-score follow-up](BINNING_BY_RECORDING.md) compares both grids within groups defined by history length, five-minute occupancy, and adequate-day continuity. Five minutes leads on average in all broad terciles under both alpha rules. With validation-selected alpha, the low-continuity/low-occupancy group is nearly tied (0.8228 versus 0.8170), whereas high-continuity/low-occupancy participants retain a larger five-minute advantage (0.8700 versus 0.8454).

Shorter-history/high-occupancy participants score 0.8130 versus 0.8073 with full history, and approximately 0.8000 at both resolutions with capped inputs. Thus the Apple pattern is more specific than a general fifteen-minute benefit for sparse recording. The follow-up includes subgroup counts, paired SDs, complete aggregate CSVs, and a script that reads existing caches and predictions without refitting.
