# Strict Garmin Track A (40-Day Budget) Benchmark Report

**Date**: 2026-10-15  
**Cohort**: Strict Garmin Single-Source Cohort ($N = 3,848$, Tier 0)  
**Input History**: First 40 adequate days ($153,920$ total participant-days)  
**Evaluation**: $K=5$ disjoint stratified cross-validation (OR-Tools CP-SAT) with zero untested participants.

---

## 1. Executive Summary & Paired Comparison

| Metric | `median` (Clock Medians) | `three_tier` (Linear + PCHIP + Medians) | Paired Difference (`three_tier − median`) |
|---|---:|---:|---:|
| **Optimal Regularization $\alpha^*$** | `5000` | `5000` | — |
| **Pooled OOF AUROC** | **0.8484** | **0.8477** | **-0.0007** |
| **DeLong 95% Confidence Interval** | `[0.8358, 0.8611]` | `[0.8351, 0.8604]` | — |
| **Stratified Bootstrap 95% CI (2,000 draws)** | `[0.8356, 0.8611]` | `[0.8349, 0.8601]` | **`[-0.0040, +0.0025]`** |
| **Average Precision (AUPRC)** | `0.9001` | `0.8993` | `-0.0008` |
| **Brier Score** | `0.1483` | `0.1484` | `+0.0001` |
| **Balanced Accuracy (at $\tau^*$)** | `0.7737` | `0.7736` | `-0.0001` |
| **Sensitivity / Specificity** | `0.7662 / 0.7811` | `0.8109 / 0.7363` | — |
| **Fold AUROC (Mean $\pm$ SD)** | `0.8485 ± 0.0183` | `0.8480 ± 0.0209` | `-0.0005 ± 0.0030` |

---

## 2. Imputation Intervention Exposure Audit

Of the **5,089,133** total missing bins in the 40-day dataset (189,241 maximal missing runs):
* **Tier 1 (Linear $\le 5\text{ min}$)**: 27,008 bins (**0.53%** of missing data).
* **Tier 2 (PCHIP $10\text{--}60\text{ min}$)**: 301,479 bins (**5.92%**).
* **Tier 2 (Linear Fallback $10\text{--}60\text{ min}$)**: 43,144 bins (**0.85%**).
* **Tier 3 (Diurnal Clock Median $>60\text{ min}$ / Boundary)**: 4,717,502 bins (**92.70%**).

---

## 3. Fold-by-Fold AUROC Breakdown

| Split | $N_{\text{val}}$ | `median` ($\alpha^*=5000$) | `three_tier` ($\alpha^*=5000$) | Delta (`three_tier − median`) |
|---|---:|---:|---:|---:|
| **Fold 0** | 770 | 0.8590 | 0.8593 | +0.0003 |
| **Fold 1** | 770 | 0.8323 | 0.8273 | -0.0050 |
| **Fold 2** | 770 | 0.8565 | 0.8576 | +0.0012 |
| **Fold 3** | 769 | 0.8261 | 0.8244 | -0.0017 |
| **Fold 4** | 769 | 0.8685 | 0.8713 | +0.0028 |
| **Mean $\pm$ SD** | — | **0.8485 ± 0.0183** | **0.8480 ± 0.0209** | **-0.0005 ± 0.0030** |
| **Pooled OOF** | **3,848** | **0.8484** | **0.8477** | **-0.0007** |

---

## 4. Subgroup Audits on Complete Out-of-Fold Cohort ($N = 3,848$)

### Age Groups

| Age Group | $N$ | Males | `median` AUROC | `three_tier` AUROC | Delta |
|---|---:|---:|---:|---:|---:|
| **20–29** | 128 | 77 | 0.8513 | 0.8533 | +0.0020 |
| **30–39** | 648 | 390 | 0.8538 | 0.8539 | +0.0001 |
| **40–49** | 1120 | 693 | 0.8649 | 0.8626 | -0.0023 |
| **50–59** | 1128 | 698 | 0.8362 | 0.8358 | -0.0005 |
| **60–69** | 635 | 461 | 0.8449 | 0.8463 | +0.0015 |
| **70+** | 189 | 145 | 0.8541 | 0.8531 | -0.0009 |

### BMI Groups

| BMI Category | $N$ | Males | `median` AUROC | `three_tier` AUROC | Delta |
|---|---:|---:|---:|---:|---:|
| **Normal ($<25$)** | 2078 | 1315 | 0.8402 | 0.8397 | -0.0004 |
| **Overweight ($25\text{--}30$)** | 1218 | 831 | 0.8539 | 0.8559 | +0.0020 |
| **Obese ($\ge 30$)** | 507 | 300 | 0.8787 | 0.8750 | -0.0037 |
| **Underweight ($<18.5$)** | 45 | 18 | 0.7325 | 0.7058 | -0.0267 |

---

## 5. Artifact Verification & Invariants

1. **Partition Integrity**: Every participant in the $N = 3,848$ strict Garmin cohort was evaluated in exactly one validation fold. Zero missing entries, zero duplicate evaluations.
2. **Leakage Prevention**: All imputation medians, MultiRocket transform biases, variance thresholding, and standard scaling were fitted strictly on training participants within each fold.
3. **Matched Experimental Control**: `median` and `three_tier` arms used bit-identical partitions, identical B40 static summaries, and matched initial MultiRocket random seeds.
4. **Production Bundles**: Retrained on 100% of the cohort ($N = 3,848$) under optimal $\alpha^*$ and saved to:
   - `experiments/k5_strict_garmin_interpolation/track_40days/median/production_bundle.joblib`
   - `experiments/k5_strict_garmin_interpolation/track_40days/three_tier/production_bundle.joblib`
