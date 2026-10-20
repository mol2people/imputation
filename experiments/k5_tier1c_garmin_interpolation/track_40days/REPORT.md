# Expanded Garmin Cohort (Tier 1c, N=4,410) Benchmark Report

**Date**: 2026-10-15  
**Cohort**: Expanded Garmin Single-Source Cohort ($N = 4,410$, Tier 1c)  
**Composition**:
- **Strict Wear Stratum**: $N = 3,848$ participants (Tier 0)
- **Added Delta Stratum**: $N = 562$ participants who failed the strict longitudinal gate:
  * **Intermittent Wear Failure**: $N = 432$ participants (`max_adequate_weeks_lt_11`)
  * **Short History Span**: $N = 128$ participants (`span_lt_13_weeks`)
  * **Low Strict Days Total**: $N = 2$ participants
**Input History**: First up to 40 adequate days (`min(n_adequate, 40)`; $174,990$ total participant-days)  
**Evaluation**: $K=5$ disjoint stratified cross-validation (OR-Tools CP-SAT) with exact stratum balancing ($N_{\text{untested}} = 0$).

---

## 1. Executive Summary & Paired Comparison

| Metric / Stratum | `median` (Clock Medians) | `three_tier` (Linear + PCHIP + Medians) | Paired Difference (`three_tier − median`) |
|---|---:|---:|---:|
| **Optimal Regularization $\alpha^*$** | `5000` | `5000` | — |
| **Overall Pooled OOF AUROC ($N = 4,410$)** | **0.8375** | **0.8382** | **+0.0007** |
| **DeLong 95% Confidence Interval** | `[0.8253, 0.8498]` | `[0.8260, 0.8504]` | — |
| **Stratified Bootstrap 95% CI (2,000 draws)** | `[0.8249, 0.8488]` | `[0.8258, 0.8495]` | **`[-0.0023, +0.0037]`** |
| **Strict Wear Stratum AUROC ($N = 3,848$)** | **0.8464** | **0.8469** | **+0.0004** |
| **Delta Stratum AUROC ($N = 562$)** | **0.7694** | **0.7734** | **+0.0041** |
| **Intermittent Wear Subgroup ($N = 432$)** | **0.7462** | **0.7420** | **-0.0042** |
| **Short Span Subgroup ($N = 128$)** | **0.8304** | **0.8582** | **+0.0278** |
| **Average Precision (AUPRC)** | `0.8904` | `0.8914` | `+0.0009` |
| **Brier Score** | `0.1537` | `0.1534` | `-0.0003` |
| **Balanced Accuracy (at $\tau^*$)** | `0.7680` | `0.7696` | `+0.0016` |
| **Fold AUROC (Mean $\pm$ SD)** | `0.8382 ± 0.0178` | `0.8385 ± 0.0159` | `+0.0003 ± 0.0041` |

---

## 2. Imputation Exposure Audit on Expanded Cohort

Of the **7,178,421** total missing bins across all 174,990 participant-days (235,622 maximal missing runs):
* **Tier 1 (Linear $\le 5\text{ min}$)**: 30,838 bins (**0.43%**).
* **Tier 2 (PCHIP $10\text{--}60\text{ min}$)**: 335,051 bins (**4.67%**).
* **Tier 2 (Linear Fallback $10\text{--}60\text{ min}$)**: 65,107 bins (**0.91%**).
* **Tier 3 (Diurnal Clock Median $>60\text{ min}$ / Boundary)**: 6,747,425 bins (**94.00%**).

---

## 3. Fold-by-Fold AUROC Breakdown

| Split | $N_{\text{val}}$ | `median` ($\alpha^*=5000$) | `three_tier` ($\alpha^*=5000$) | Delta (`three_tier − median`) |
|---|---:|---:|---:|---:|
| **Fold 0** | 882 | 0.8260 | 0.8247 | -0.0013 |
| **Fold 1** | 882 | 0.8401 | 0.8448 | +0.0046 |
| **Fold 2** | 882 | 0.8673 | 0.8632 | -0.0042 |
| **Fold 3** | 882 | 0.8355 | 0.8331 | -0.0025 |
| **Fold 4** | 882 | 0.8220 | 0.8267 | +0.0047 |
| **Mean $\pm$ SD** | — | **0.8382 ± 0.0178** | **0.8385 ± 0.0159** | **+0.0003 ± 0.0041** |
| **Overall OOF** | **4,410** | **0.8375** | **0.8382** | **+0.0007** |

---

## 4. Key Scientific Insights

1. **Impact of Adding Intermittent & Short Wearers**:
   - In the strict cohort ($N = 3,848$), OOF AUROC was **0.8484**.
   - In the expanded cohort ($N = 4,410$), overall OOF AUROC is **0.8375**.
   - Within the test sets, the strict stratum scores **0.8464**, while the added delta stratum scores **0.7694**.
   - Specifically, participants with intermittent wear (`max_adequate_weeks_lt_11`) score **0.7462**, and those with short histories (`span_lt_13_weeks`) score **0.8304**.
2. **Imputation Comparison in Intermittent Wear**:
   - In the intermittent wear subgroup ($N = 432$), the paired `three_tier − median` difference is **-0.0042**.
   - In the short history subgroup ($N = 128$), the difference is **+0.0278**.
3. **Reproducibility & Production Models**:
   - Production bundles retrained on 100% of the cohort ($N = 4,410$) are saved to:
     * `experiments/k5_tier1c_garmin_interpolation/track_40days/median/production_bundle.joblib`
     * `experiments/k5_tier1c_garmin_interpolation/track_40days/three_tier/production_bundle.joblib`
