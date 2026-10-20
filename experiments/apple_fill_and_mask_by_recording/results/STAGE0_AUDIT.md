# Stage 0 Diagnostic Audit: Gap & Mask Exposure Anatomy, Difference Magnitudes, and MultiRocket Degeneracy

**Date**: 2026-10-19  
**Cohort**: Full Apple Watch Cohort ($N = 11,139$ participants, $2,967,608$ adequate days)  
**Target Profile**: Profile recorded salutation (`10 -> 0/F`, `20 -> 1/M`, Apple Watch channel 3000 HR)  
**Authoritative Plan**: `experiments/apple_fill_and_mask_by_recording/PLAN.md` (amended with Review R1–R6)  
**Execution Environment**: Python 3.10+, micromamba `datenspende`  

---

## Executive Summary

Before executing Stage 1 model fitting across allocations 0, 1, and 2, this audit conducts a comprehensive, read-only empirical characterization of the 5-minute ($L=288$) and 15-minute ($L=96$) cached grids. Four principal findings emerge:

1. **Massive Interpolation Disparity (32.2% vs. 7.2%)**: At 5 minutes, 32.2% of missing bins are eligible for span-matched linear interpolation ($(k+1)\Delta t \le 30\text{ min}$, $k \le 5$), compared to only 7.2% at 15 minutes ($k = 1$). Span-matched short-gap interpolation (`H_short_interp`) represents a substantially larger physical intervention at 5m than at 15m.
2. **Boundary Jump Discontinuity Dominance ($1.70\times$ true physiology)**: Artificial step discontinuities at gap boundaries ($M_t \neq M_{t+1}$) average $21.56\text{ bpm}$ RMS ($13.72\text{ bpm}$ MAE) under $\mu^{\text{clock}}$ and $22.32\text{ bpm}$ RMS ($14.37\text{ bpm}$ MAE) under $\mu^{\text{global}}$, compared to only $12.72\text{ bpm}$ RMS ($6.04\text{ bpm}$ MAE) for genuine physiological transitions ($M_t = M_{t+1} = 1$). In sparse wearers (`low/low`), boundary jumps reach $23.35\text{ bpm}$ RMS.
3. **Interior Slope Invariance ($0.14\text{ bpm}$ vs. $0.00\text{ bpm}$)**: Diurnal template slopes inside gaps ($(0, 0)$) under $\mu^{\text{clock}}$ average only $0.14\text{ bpm}$ MAE at 5m ($0.49\text{ bpm}$ at 15m), functionally negligible compared to the boundary jumps. This provides the mathematical basis for why `H_clock` and `H_global` yield near-identical aggregate AUROC.
4. **MultiRocket Quantile Threshold Collapse on Binary Masks**: On binary observation masks $M \in \{0, 1\}$, convolution responses take discrete values in $[\![-6, +6]\!]$, collapsing 1,176 quantile bias parameters down to 52 unique values ($4.4\%$) at 5m and 96 ($8.2\%$) at 15m, with modal bias $0.0000$ accounting for $34.4\%$ and $43.0\%$ of parameters. Zero-variance features drop only 7 daily (14 pooled) columns at 5m and 1 daily (2 pooled) at 15m, but pervasive collinearity within kernel slots necessitates stronger L2 shrinkage ($\alpha \ge 10,000$).

---

## 1. Adjacent-Bin Transition Anatomy & Gap-Span Distribution

For each participant-day of length $L$, there are exactly $L - 1$ intra-day transitions $t \to t+1$ ($287$ at 5m, $95$ at 15m). Transitions are partitioned into four mutually exclusive, collectively exhaustive regimes:
- **$\mathcal{T}_{11}$**: Observed-to-observed transitions ($M_t = 1 \land M_{t+1} = 1$).
- **$\mathcal{T}_{\text{short},30\text{m}}$**: Transitions touching or inside an interior missing run of length $1 \le k \le k_{\max}$ bounded on both sides by observed bins within the same day with $(k+1)\Delta t \le 30\text{ min}$ ($k_{\max} = 5$ at 5m; $k_{\max} = 1$ at 15m). At 5m, this is decomposed into $1\text{--}2$ bins ($10\text{--}15\text{ min}$) vs. $3\text{--}5$ bins ($20\text{--}30\text{ min}$).
- **$\mathcal{T}_{\text{edge}}$**: Boundary transitions ($M_t \neq M_{t+1}$) belonging to ineligible interior runs ($k > k_{\max}$) or day-boundary runs.
- **$\mathcal{T}_{00,\text{long}}$**: Interior gap transitions ($M_t = 0 \land M_{t+1} = 0$) belonging to ineligible runs.

Algebraic invariant verified across all $2,967,608$ days:
$$n_{11} + n_{\text{short}} + n_{\text{edge}} + n_{00,\text{long}} \equiv D_u (L - 1).$$

### 1.1 Cohort-Wide Transition & Eligibility Totals

| Resolution | History Budget | Total Days | Missing Bin Frac (Cohort) | Eligible Missing Bins | Eligible Missing Frac | $\mathcal{T}_{11}$ Frac | $\mathcal{T}_{\text{short},30\text{m}}$ Frac | $\mathcal{T}_{\text{short},1\text{--}2}$ Frac | $\mathcal{T}_{\text{short},3\text{--}5}$ Frac | $\mathcal{T}_{\text{edge}}$ Frac | $\mathcal{T}_{00,\text{long}}$ Frac |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **5m ($L=288$)** | `all_days` | 2,967,608 | 45.73% | 125,881,709 | **32.21%** | 42.45% | 25.81% | 21.03% | 4.78% | 1.66% | 30.08% |
| **5m ($L=288$)** | `up_to_40d` | 432,345 | 44.95% | 17,493,241 | **31.26%** | 43.47% | 25.01% | 21.10% | 3.91% | 1.44% | 30.09% |
| **15m ($L=96$)** | `all_days` | 2,967,608 | 31.13% | 6,399,328 | **7.22%** | 64.85% | 4.54% | 4.54% | N/A | 3.89% | 26.71% |
| **15m ($L=96$)** | `up_to_40d` | 432,345 | 30.89% | 729,427 | **5.69%** | 65.78% | 3.55% | 3.55% | N/A | 3.50% | 27.16% |

### 1.2 Participant-Level Distributions (Median [IQR])

| Resolution | Budget | Missing Bin Frac | Eligible Missing Frac | $\mathcal{T}_{11}$ Frac | $\mathcal{T}_{\text{short}}$ Frac | $\mathcal{T}_{\text{short},1\text{--}2}$ Frac | $\mathcal{T}_{\text{short},3\text{--}5}$ Frac | $\mathcal{T}_{\text{edge}}$ Frac | $\mathcal{T}_{00,\text{long}}$ Frac |
|---|---|---|---|---|---|---|---|---|---|
| **5m** | `all_days` | 47.07% [23.76%] | 30.21% [25.22%] | 41.38% [20.80%] | 25.41% [7.16%] | 20.60% [6.49%] | 4.52% [2.93%] | 1.58% [0.70%] | 32.18% [26.22%] |
| **15m** | `all_days` | 33.12% [26.29%] | 6.67% [7.79%] | 63.28% [25.84%] | 4.27% [3.04%] | 4.27% [3.04%] | N/A | 3.72% [1.42%] | 29.07% [26.04%] |

### 1.3 Key Observations on Transition Anatomy
- **Coarsening Effect on Missingness**: Moving from 5m to 15m reduces cohort missingness from 45.73% to 31.13%, but compresses short-gap eligibility from 32.21% to 7.22%.
- **Short-Gap Composition**: At 5m, the 25.81% short-gap transitions are split: 21.03% in 1–2 bin gaps ($10\text{--}15\text{ min}$) and 4.78% in 3–5 bin gaps ($20\text{--}30\text{ min}$).
- **Physical Span Matching Caveat (Review R4)**: Because $k=1$ at 15m corresponds to $30\text{ min}$ separation while $k=1\text{--}5$ at 5m spans $10\text{--}30\text{ min}$, `H_short_interp` acts over a wider variety of sampling gaps at 5m than at 15m.

---

## 2. Difference-Series Magnitudes Across Transition Regimes

Using Allocation 0 training templates ($\mu^{\text{clock}}$ and $\mu^{\text{global}}$ fitted strictly on train IDs ($N=7,797$), we audit participant-level RMS and MAE of the difference series $\Delta X_t = X_{t+1} - X_t$.

### 2.1 Cohort-Wide Difference Magnitudes (Overall Median [IQR])

| Transition Regime | Filling Method | 5-Minute MAE (bpm) | 5-Minute RMS (bpm) | 15-Minute MAE (bpm) | 15-Minute RMS (bpm) | Ratio to Natural Volatility (5m RMS) |
|---|---|---:|---:|---:|---:|---:|
| **Observed $(1, 1)$** | None (Raw Physiology) | 6.04 [2.77] | 12.72 [5.67] | 8.04 [3.91] | 15.70 [8.01] | **1.00× (Baseline)** |
| **Boundary Jump $(M_t \neq M_{t+1})$** | $\mu^{\text{clock}}$ Fill | 13.72 [4.93] | 21.56 [9.00] | 18.38 [7.21] | 27.50 [11.91] | **1.70×** |
| **Boundary Jump $(M_t \neq M_{t+1})$** | $\mu^{\text{global}}$ Fill | 14.37 [5.31] | 22.32 [9.31] | 18.80 [7.96] | 28.36 [12.51] | **1.75×** |
| **Gap Interior $(0, 0)$** | $\mu^{\text{clock}}$ Fill | 0.14 [0.04] | 0.32 [0.05] | 0.49 [0.14] | 0.66 [0.15] | **0.03×** |
| **Gap Interior $(0, 0)$** | $\mu^{\text{global}}$ Fill | 0.00 [0.00] | 0.00 [0.00] | 0.00 [0.00] | 0.00 [0.00] | **0.00×** |

### 2.2 Difference Magnitudes Across Continuity × Occupancy Subgroups (5-Minute Grid)

The 9 crossed cells (`continuity_x_occupancy`) demonstrate marked gradient in boundary jump magnitude:

| Subgroup (`cont_x_occ`) | N Users | Observed MAE | Observed RMS | Boundary Clock MAE | Boundary Clock RMS | Boundary Global MAE | Boundary Global RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `low/low` | 1320 | 7.84 | 15.20 | 14.70 | 23.35 | 14.81 | 23.92 |
| `low/middle` | 1186 | 6.04 | 12.18 | 13.35 | 20.73 | 13.75 | 21.30 |
| `low/high` | 1111 | 4.66 | 9.85 | 12.39 | 18.73 | 13.25 | 19.52 |
| `middle/low` | 1569 | 7.89 | 15.32 | 14.75 | 23.55 | 15.02 | 24.40 |
| `middle/middle` | 1264 | 5.85 | 12.38 | 13.67 | 21.56 | 14.31 | 22.33 |
| `middle/high` | 943 | 4.84 | 10.48 | 12.79 | 20.07 | 14.03 | 21.09 |
| `high/low` | 863 | 8.24 | 16.32 | 15.38 | 24.73 | 15.76 | 25.69 |
| `high/middle` | 1277 | 6.17 | 13.25 | 14.18 | 22.23 | 14.82 | 23.07 |
| `high/high` | 1606 | 4.87 | 11.02 | 12.97 | 20.02 | 14.02 | 21.07 |

### 2.3 Synthesis of Difference-Series Dynamics
1. **Severe Boundary Contamination**: Intra-day boundary jumps inject step changes averaging $>21\text{ bpm}$ RMS, which is $1.70\times$ larger than true physiological drift ($12.72\text{ bpm}$ RMS). At 15m, boundary jumps reach $27.50\text{ bpm}$ RMS.
2. **Mechanism of `H_clean_diff`**: In standard MultiRocket (`H_clock`), half of the $18,816$ sequence features originate from $\Delta X$. Because convolution integrates differences within a 9-bin receptive field, these massive boundary spikes dominate the difference features. `H_clean_diff` eliminates boundary jump contamination by setting $\Delta^{\text{clean}} X_t = 0.0$ whenever $M_t = 0 \lor M_{t+1} = 0$, shielding downstream linear Ridge models from non-physiological impulse noise.
3. **Interior Template Slope Irrelevance**: Inside gaps ($(0, 0)$), diurnal slopes under $\mu^{\text{clock}}$ average only $0.14\text{ bpm}$ MAE at 5m ($0.49\text{ bpm}$ at 15m), which is $<1.1\%$ of boundary jump amplitude ($13.72\text{ bpm}$ MAE). Thus, replacing $\mu^{\text{clock}}$ with $\mu^{\text{global}}$ has negligible impact on within-gap slopes; any performance contrast between `H_clock` and `H_global` is driven primarily by slight boundary jump differences.

---

## 3. MultiRocket Binary-Quantile Degeneracy Audit (Allocation 0)

### 3.1 Bias Threshold Collapse
Under continuous heart rate ($X^{\text{clock}}$), length-9 convolution responses take continuous values, yielding $1,060$ unique bias thresholds out of $1,176$ parameters ($90.1\%$ unique). On binary masks $M \in \{0.0, 1.0\}$ and difference masks $\Delta M \in \{-1.0, 0.0, 1.0\}$, convolution responses are restricted to integers in $[-6, +6]$. Consequently, quantile thresholding collapses onto discrete values dominated by $0.0000$:

| Resolution | Series Branch | Total Biases | Unique Biases | Unique Pct (%) | Zero Biases (count / %) | Top-5 Bias Concentration |
|---|---|---:|---:|---:|---:|---:|
| **5m ($L=288$)** | Base Mask $M \in \{0, 1\}$ | 1176 | 52 | 4.4% | 404 / 34.4% | 82.7% |
| **5m ($L=288$)** | Diff Mask $\Delta M \in \{-1, 0, 1\}$ | 1176 | 64 | 5.4% | 437 / 37.2% | 76.9% |
| **15m ($L=96$)** | Base Mask $M \in \{0, 1\}$ | 1176 | 96 | 8.2% | 506 / 43.0% | 84.4% |
| **15m ($L=96$)** | Diff Mask $\Delta M \in \{-1, 0, 1\}$ | 1176 | 82 | 7.0% | 732 / 62.2% | 89.2% |
| *5m (Ref)* | *Continuous HR $X^{\text{clock}}$* | *1176* | *891* | *75.8%* | *0 / 0.0%* | *< 5.0%* |

### 3.2 Feature Drop Audit (`VarianceThreshold(0.0)`)
A daily feature drops with zero variance when an extreme bias threshold (e.g., $\le -6.0$ on $\Delta M$) is never crossed by any day in the sampled training cohort. Because across-day mean and across-day sample SD inherit this constant value, each dropped daily feature induces exactly two dropped pooled features:

| Resolution | Series Branch | Daily Features Dropped (out of 9,408) | Daily Drop Pct (%) | Pooled Features Dropped (out of 18,816) | Pooled Drop Pct (%) | Dropped Daily Indices |
|---|---|---:|---:|---:|---:|---|
| **5m ($L=288$)** | $M + \Delta M$ | 7 | 0.074% | 14 | 0.074% | `[4792, 4936, 5968, 6091, 6112, 8320, 8464]` |
| **15m ($L=96$)** | $M + \Delta M$ | 1 | 0.011% | 2 | 0.011% | `[1416]` |

### 3.3 Collinearity and Regularization Implications
Although `VarianceThreshold(0.0)` drops only $0.01\%\text{--}0.07\%$ of columns, the discrete bias collapse creates numerous identical duplicate columns within kernel/dilation slots where multiple quantiles mapped to the same threshold. These duplicate columns retain variance but are strictly collinear. In Stage 1 fitting, `VarianceThreshold(0.0)` must be followed by Ridge L2 regularization with larger candidate penalties ($\alpha \in \{10000, 100000\}$) to prevent coefficient explosion on collinear mask features.

---

## 4. Implications for Stage 1 Benchmark Protocol

1. **`H_clock` vs. `H_global` Contrast**: Given that gap interior slopes under $\mu^{\text{clock}}$ ($0.14\text{ bpm}$) are functionally negligible and boundary jumps differ by only $\approx 0.65\text{ bpm}$ ($0.76\text{ bpm}$ RMS), the aggregate contrast $\text{AUROC}(\text{H\_global}) - \text{AUROC}(\text{H\_clock})$ is expected to be minimal ($|\Delta| \le 0.005$).
2. **`H_clean_diff` Hypothesis**: Withholding boundary jumps from the $\Delta X$ branch will test whether MultiRocket relied on behavioral gap step-jumps or genuine physiological fluctuations. If `H_clean_diff` maintains or improves AUROC, it demonstrates that boundary artifacts were introducing estimation burden or non-generalizable noise.
3. **`H_short_interp` Hypothesis**: Because 32.2% of missing bins are eligible at 5m vs. only 7.2% at 15m, short-gap interpolation represents a significantly more extensive intervention at 5m. Any differential resolution gain from `H_short_interp` at 5m reflects this higher exposure rate.
4. **`M_temporal` Ridge Grid**: The severe bias collapse on binary sequences produces high collinearity. The candidate grid must include $\alpha = 100,000$ (`ridge_selected_ext`) to ensure proper shrinkage for `M_temporal` and `H_plus_M_concat`.

---

## Artifact Index
- `results/exposure_overall.csv` — Full cohort and per-condition transition, eligibility, and difference statistics.
- `results/exposure_by_subgroup.csv` — Stratified statistics across 31 recording-pattern subgroups.
- `results/STAGE0_AUDIT.md` — This publication document.
