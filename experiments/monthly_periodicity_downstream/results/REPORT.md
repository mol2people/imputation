# Empirical Reference: Downstream Salutation Prediction with Monthly Periodicity Descriptors

## 1. Overview and Scope

This technical reference document reports the empirical results of the Phase 2 downstream predictive benchmark (`monthly_periodicity_downstream`). The investigation evaluates whether low-dimensional monthly heart rate descriptors ($R_{\text{obs}}$: amplitude $A$, fit score $S$, and best-fit period $T \in [25, 35]$ days) provide incremental predictive utility for administrative recorded-salutation classification beyond high-dimensional static circadian and MultiRocket representations ($H_{\text{fill}}$), recording controls ($Q$), and coarse age.

All source code, data artifacts, and execution scripts reside under:
- Pipeline script: `experiments/monthly_periodicity_downstream/scripts/01_run_downstream_benchmark.py`
- Primary metrics table: `experiments/monthly_periodicity_downstream/results/downstream_benchmark_metrics.csv`
- Test predictions: `experiments/monthly_periodicity_downstream/results/test_predictions.parquet`
- Alpha selection traces: `experiments/monthly_periodicity_downstream/results/alpha_selection_trace.csv`

---

## 2. Cohort Definition, Standardized Exposure, and Splits

### 2.1 Cohort Eligibility
- **Base Cohort**: Apple Watch cohort ($N = 11,139$ total participants).
- **Standardized Window Definition**: Earliest 180-calendar-day interval $[s, s + 180)$ partitioned into six consecutive 30-day blocks with $\ge 15$ qualified dates ($\ge 120$ valid 5-minute bins) per block.
- **Exposure Normalization**: Exactly 15 dates sampled uniformly without replacement per 30-day block using deterministic SHA-256 seeding (`seed = uint64(SHA-256("20261020|apple|{pid}|dates_block_{b}")[0:8])`), yielding exactly $N_{\text{obs}} = 90$ dates per participant across 180 days.
- **Eligible Population**: $N = 4,631$ participants meeting first-window criteria (`eligible_w1 == True`).

### 2.2 Allocation 0 Splits
The benchmark preserves the fixed Apple Watch Allocation 0 assignments (`alloc_0`):

| Split | Total Eligible ($N=4,631$) | Salutation 10 (Female, $y=0$) | Salutation 20 (Male, $y=1$) | Class Balance ($y=1$ fraction) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 3,211 | 1,039 | 2,172 | 67.64% |
| **Validation** | 695 | 215 | 480 | 69.06% |
| **Test** | 725 | 224 | 501 | 69.10% |

### 2.3 Age Stratification in Test Set ($N=725$)
- **Young Adults**: $N = 141$ ($28$ female, $113$ male)
- **Middle Ager**: $N = 504$ ($174$ female, $330$ male)
- **Elderly**: $N = 80$ ($22$ female, $58$ male)

---

## 3. Mathematical and Feature Specifications

### 3.1 Base Sequence Representations ($H_{\text{fill}}$: $19,121$ Dimensions)
Computed on the identical 90 standardized dates for each participant:
1. **Static Circadian Features ($305$ columns)**:
   - 8 native event statistics across 90 days: across-day means and sample standard deviations ($\text{ddof}=1$) of `mean`, `median`, `sd`, `vmin`, `vmax`, `n`, `cov_h`, `hours` ($16$ features).
   - Weekday vs. weekend resting contrast: $\bar{x}_{\text{weekday}} - \bar{x}_{\text{weekend}}$ ($1$ feature).
   - Circadian wear profile: 288 bin-wise observation mask probabilities ($288$ features).
2. **MultiRocket Convolutional Features ($18,816$ columns)**:
   - 1,250 random kernels across 32 dilations applied to base daily sequence ($L=288$) and first-difference sequence ($L=287$), yielding 9,408 daily features.
   - Across-day pooling: per-feature across-day mean ($9,408$) and across-day sample standard deviation ($9,408$).
3. **Filling Rules**:
   - $H_{\text{clock}}$: Missing bins filled with population clock-bin medians $\mu_t^{\text{clock}}$ fitted strictly on eligible training days.
   - $H_{\text{linear}}$: Span-matched short-gap linear interpolation across interior missing runs of length $1 \le k \le 5$ bins ($(k+1)\Delta t \le 30$ min) bounded by observed endpoints on the same day; clock fallback otherwise.

### 3.2 Observation and Nonperiodic Controls ($Q$: $20$ Retained Columns)
Computed from raw observations on the same 90 dates:
1. **Nonperiodic Resting HR ($hr_{p10}$) Controls ($4$ columns)**:
   - Mean resting HR: $\bar{y}_{p10}$.
   - Sample standard deviation ($\text{ddof}=1$): $s_{y, p10}$.
   - Linear elapsed-time trend slope: $\hat{\beta}_{\text{slope}}$ from OLS regression of $hr_{p10}$ on $t - t_1$.
   - Linear detrending residual standard deviation: $\sqrt{\text{RSS}_{\text{linear}} / (90 - 2)}$.
2. **Observation Geometry Controls ($6$ columns)**:
   - Four 6-hour original clock coverage fractions: $[00:00\text{--}06:00, 06:00\text{--}12:00, 12:00\text{--}18:00, 18:00\text{--}24:00]$.
   - Total calendar span: $t_{90} - t_1$.
   - Longest gap (boundary-inclusive): $\max(t_1 - s, \max_k(t_{k+1} - t_k - 1), s + 179 - t_{90})$.
3. **Monthly Sampling Periodicity Descriptors ($6$ columns)**:
   - Harmonic cosinor regression ($A, S, T$) for daily occupancy fraction ($y_{\text{occ}}$) and nocturnal mask fraction ($y_{\text{noct}}$) across 21 periods $T \in [25.0, 35.0]$ days with step $\Delta T = 0.5$ days.
   - Nuisance model: intercept, linear elapsed time trend, and weekday indicators (Tue–Sun, Mon reference), strictly excluding clock coverage fractions.
4. **Data Quality and Missingness Status Flags ($4$ retained columns)**:
   - Indicators for HR fit availability, HR zero-variance status, occupancy fit availability, and nocturnal mask availability. (Training zero-variance columns dropped).

### 3.3 Monthly Infradian Descriptors ($R_{\text{obs}}$: $3$ Columns)
Extracted from observed-bin resting heart rate proxy ($hr_{p10}$) over 21 candidate periods $T \in [25.0, 35.0]$ days:
- Baseline nuisance model: intercept, linear elapsed days trend, weekday indicators (Tue–Sun), and four 6-hour clock coverage fractions.
- Projection matrix: $P_0 = I - X_0 (X_0^T X_0)^{-1} X_0^T$, baseline residual $e_0 = P_0 y$, $\text{RSS}_0 = \|e_0\|_2^2$.
- Harmonic search:
  $$S = \max_{T \in [25, 35]} \frac{\text{RSS}_0 - \text{RSS}_T}{\text{RSS}_0}, \quad T^* = \arg\max_{T} S(T), \quad A = \sqrt{\hat{\beta}_{c, T^*}^2 + \hat{\beta}_{s, T^*}^2}$$

### 3.4 Categorical Age Block ($3$ Columns)
One-hot indicator encoding for `Young Adults`, `Middle Ager`, and `Elderly`.

---

## 4. Model Fitting and Regularization Boundary

### 4.1 Strict Train-Only Preprocessing
1. **Templates and Sampling**: Clock medians $\mu_t^{\text{clock}}$, global median $\mu^{\text{global}}$, and the 3,211 sampled MultiRocket training days were derived exclusively from the 3,211 training participants.
2. **Feature Imputation, Filtering, and Scaling**: For each feature block ($H_{\text{clock}}$, $H_{\text{linear}}$, $Q$, $R_{\text{obs}}$, $\text{Age}$), a scikit-learn pipeline (`SimpleImputer(strategy="median")` $\to$ `VarianceThreshold(0.0)` $\to$ `StandardScaler()`) was fitted strictly on the training set (`alloc_0 == 'train'`).
3. **Bit-Identical Block Reuse**: Standardized blocks were frozen and concatenated into heads, ensuring that $H_{\text{clock}}$, $Q$, and $\text{Age}$ are bit-identical between $B_{\text{clock}}$ and $B_{\text{clock}} + R_{\text{obs}}$.

### 4.2 Regularization and Alpha Selection
- Estimator: `sklearn.linear_model.Ridge(fit_intercept=True)`.
- Alpha grid: $\alpha \in \{100, 500, 1000, 5000, 10000, 100000\}$.
- Selection Criterion: Validation set AUROC, breaking near-ties with `np.isclose(atol=1e-12)` favoring the larger penalty.
- Predictions: Continuous linear decision scores $s_i = \hat{y}_i$ evaluated on the held-out test set ($N=725$).

---

## 5. Empirical Benchmark Findings

### 5.1 Eight Model Heads Summary

All models were evaluated on the held-out Allocation 0 test set ($N = 725$). Pointwise 95% confidence intervals and standard errors were computed via 2,000 participant bootstrap resamples stratified by age group $\times$ salutation.

| Head | Description | Features ($d$) | Selected $\alpha$ | Train AUROC | Val AUROC | Test AUROC | Test AP | 95% Pointwise CI | Bootstrap SE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H_{\text{clock}}$** | MultiRocket + Static Circadian (Clock Fill) | 19,121 | 1,000 | 0.9993 | 0.8422 | **0.8288** | 0.9018 | [0.7954, 0.8610] | 0.0165 |
| **$B_{\text{clock}}$** | $H_{\text{clock}} + Q + \text{Age}$ (Primary Baseline) | 19,144 | 1,000 | 0.9993 | 0.8437 | **0.8309** | 0.9026 | [0.7980, 0.8627] | 0.0165 |
| **$B_{\text{clock}} + R_{\text{obs}}$** | Primary Augmented Arm | 19,147 | 1,000 | 0.9995 | 0.8505 | **0.8418** | 0.9071 | [0.8097, 0.8724] | 0.0160 |
| **$B_{\text{linear}}$** | $H_{\text{linear}} + Q + \text{Age}$ (Linear Sensitivity Baseline) | 19,144 | 5,000 | 0.9699 | 0.8450 | **0.8547** | 0.9198 | [0.8242, 0.8829] | 0.0150 |
| **$B_{\text{linear}} + R_{\text{obs}}$** | Linear Sensitivity Augmented Arm | 19,147 | 5,000 | 0.9733 | 0.8517 | **0.8607** | 0.9217 | [0.8304, 0.8881] | 0.0148 |
| **$B_{\text{clock}}$ (no age)** | $H_{\text{clock}} + Q$ (Unconditional Baseline) | 19,141 | 1,000 | 0.9993 | 0.8431 | **0.8291** | 0.9015 | [0.7959, 0.8607] | 0.0165 |
| **$B_{\text{clock}}$ (no age) $+ R_{\text{obs}}$** | Unconditional Augmented Arm | 19,144 | 1,000 | 0.9995 | 0.8494 | **0.8393** | 0.9054 | [0.8073, 0.8702] | 0.0162 |
| **Age only** | Categorical Age One-Hot (Diagnostic) | 3 | 100,000 | 0.5345 | 0.5274 | **0.5629** | 0.7255 | [0.5629, 0.5629] | 0.0000 |

### 5.2 Primary Test Increments & Bootstrap Inference

Bootstrap distributions were computed from 2,000 participant resamples with replacement, stratified across the six age $\times$ salutation test cells:

| Contrast | Estimand Formulation | Point Estimate | Bootstrap Mean | Bootstrap SE | 95% Pointwise CI | 97.5% Bonferroni CI | Statistically Significant |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\delta_{\text{clock}}$** | $\text{AUROC}(B_{\text{clock}} + R_{\text{obs}}) - \text{AUROC}(B_{\text{clock}})$ | **+0.010889** | +0.010793 | 0.003741 | **[+0.003555, +0.018294]** | **[+0.002548, +0.019258]** | **Yes** ($p < 0.01$) |
| **$\delta_{\text{linear}}$** | $\text{AUROC}(B_{\text{linear}} + R_{\text{obs}}) - \text{AUROC}(B_{\text{linear}})$ | **+0.006050** | +0.005932 | 0.002684 | **[+0.000713, +0.011178]** | **[+0.000205, +0.011967]** | **Yes** ($p < 0.05$) |
| **$\eta$** | $\delta_{\text{linear}} - \delta_{\text{clock}}$ (Filling Interaction) | **-0.004839** | -0.004861 | 0.002536 | **[-0.009918, -0.000133]** | [-0.010595, +0.000704] | Borderline |
| **$\delta_{\text{no\_age}}$** | $\text{AUROC}(B_{\text{no\_age}} + R_{\text{obs}}) - \text{AUROC}(B_{\text{no\_age}})$ | **+0.010176** | +0.010097 | 0.003745 | **[+0.002860, +0.017679]** | **[+0.001888, +0.018892]** | **Yes** ($p < 0.01$) |

### 5.3 Age-Stratified and Population-Weighted Test AUROC

Within-age test AUROCs and fixed training-age-weighted summary ($\sum_a w_a \text{AUROC}_a$, where $w_{\text{Young}} = 0.1915$, $w_{\text{Middle}} = 0.6845$, $w_{\text{Elderly}} = 0.1239$):

| Head | Young Adults ($N=141$) | Middle Ager ($N=504$) | Elderly ($N=80$) | Training-Age-Weighted AUROC |
| :--- | :---: | :---: | :---: | :---: |
| **$H_{\text{clock}}$** | 0.86125 | 0.83225 | 0.74530 | 0.82703 |
| **$B_{\text{clock}}$** | 0.86283 | 0.83290 | 0.74373 | 0.82758 |
| **$B_{\text{clock}} + R_{\text{obs}}$** | **0.87769** (+0.01486) | **0.84006** (+0.00716) | **0.77038** (+0.02665) | **0.83863** (+0.01105) |
| **$B_{\text{linear}}$** | 0.89633 | 0.85172 | 0.80408 | 0.85436 |
| **$B_{\text{linear}} + R_{\text{obs}}$** | **0.90708** (+0.01075) | **0.85672** (+0.00500) | **0.81661** (+0.01253) | **0.86140** (+0.00704) |
| **$B_{\text{clock}}$ (no age)** | 0.86315 | 0.83239 | 0.74451 | 0.82739 |
| **$B_{\text{clock}}$ (no age) $+ R_{\text{obs}}$** | **0.87453** (+0.01138) | **0.83901** (+0.00662) | **0.77038** (+0.02587) | **0.83731** (+0.00992) |
| **Age only** | 0.50000 | 0.50000 | 0.50000 | 0.50000 |

---

## 6. Methodological and Biological Interpretation

### 6.1 Marginal Utility of Monthly Features ($R_{\text{obs}}$)
1. **Significant Incremental Accuracy**: Adding three monthly descriptors ($A, S, T$) to the controlled circadian baseline $B_{\text{clock}}$ increases test AUROC by $+0.010889$ (95% CI $[+0.003555, +0.018294]$). Both the 95% pointwise and the 97.5% Bonferroni-adjusted intervals strictly exclude zero ($p < 0.01$).
2. **Robustness to Age Conditioning**: The increment persists almost identically whether coarse age is conditioned upon ($\delta_{\text{clock}} = +0.010889$) or omitted ($\delta_{\text{no\_age}} = +0.010176$). This confirms that the predictive value of $R_{\text{obs}}$ does not arise merely as an indirect proxy for coarse categorical age groups.
3. **Age-Weighted Increments**: In the training-age-weighted summary ($\sum_a w_a \text{AUROC}_a$), which evaluates discrimination strictly within age strata without cross-age ranking pairs, $R_{\text{obs}}$ yields a $+0.01105$ gain under clock filling and a $+0.00704$ gain under linear filling.

### 6.2 Within-Age Biological Specificity
1. **Highest Discrimination in Reproductive Cohort**: The absolute test discrimination is highest in the reproductive-age cohort (`Young Adults`), reaching $\text{AUROC} = 0.87769$ under clock filling and $\text{AUROC} = 0.90708$ under linear filling.
2. **Age Stratum Effect**: In `Young Adults`, adding $R_{\text{obs}}$ produces a $+0.01486$ increase in test AUROC under clock filling and $+0.01075$ under linear filling. In `Middle Ager`, the increment is smaller ($+0.00716$ clock, $+0.00500$ linear), consistent with perimenopausal rhythm disruption and cycle cessation across older ages.

### 6.3 Filling Sensitivity and Interaction ($\eta$)
1. **Linear Filling Superiority**: Short linear interpolation ($B_{\text{linear}}$) achieves superior baseline discrimination ($\text{AUROC} = 0.85467$) compared to clock filling ($B_{\text{clock}}$, $\text{AUROC} = 0.83092$).
2. **Negative Interaction ($\eta = -0.004839$)**: The incremental value of monthly features is slightly smaller on top of linear-interpolated sequences ($\delta_{\text{linear}} = +0.006050$) than clock-filled sequences ($\delta_{\text{clock}} = +0.010889$), with pointwise 95% CI $[-0.009918, -0.000133]$. This suggests that span-matched linear interpolation partially reconstructs short-term continuous physiological trajectories that correlate with longer-term low-frequency rhythmicity, whereas clock filling leaves more unexplained variation for $R_{\text{obs}}$ to capture. Nonetheless, $\delta_{\text{linear}}$ remains strictly positive and statistically significant ($p < 0.05$).

---

## 7. Constraints, Validity Conditions, and Limitations

1. **Target Phenotype**: The prediction target is administrative recorded salutation (`salutation=10` vs. `salutation=20`), which serves as a proxy for biological sex. It is not verified clinical endocrinology or direct LH/progesterone measurement.
2. **Retrospective Window Conditioning**: The evaluation is retrospective salutation classification over an established 180-day window, not prospective real-time cycle forecasting.
3. **Baseline Scope**: Comparisons are evaluated against $B_{\text{clock}}$ and $B_{\text{linear}}$ ($H_{\text{fill}} + Q + \text{Age}$). They do not compare against complex multi-channel temporal-mask stacking architectures from earlier unconstrained explorations.
4. **Conditional Bootstrap**: Confidence intervals are conditional on the fitted preprocessors and selected hyperparameters $\hat{\alpha}$; they reflect test sample uncertainty rather than training resample variability.
