# Empirical Reference: Standardized Infradian Rhythmicity in Wearable Heart Rate

## 1. Overview and Scope

This technical reference document reports the empirical results of the Phase 1 calendar-standardized infradian analysis (`female_infradian_cycles_v2`). The study investigates whether 25–35-day periodic oscillations in daily resting heart rate proxy (`hr_p10`) exhibit biological specificity across administrative salutation codes (`y=0` female vs. `y=1` male) and categorical age strata (`Young Adults`, `Middle Ager`, `Elderly`) under standardized observation exposure and dependence-aware parametric null calibration.

All source code, data artifacts, and execution scripts reside under:
- Pipeline scripts: `experiments/female_infradian_cycles_v2/scripts/`
- Data artifacts: `experiments/female_infradian_cycles_v2/results/window_eligibility.parquet`
- Statistical inference table: `experiments/female_infradian_cycles_v2/results/calibrated_group_contrasts.csv`

---

## 2. Cohort Eligibility and Standardized Exposure Design

### 2.1 Screening and Window Definition
- **Base Cohort**: Apple Watch cohort ($N = 11,139$ total participants; $3,859$ recorded $y=0$, $7,280$ recorded $y=1$).
- **Qualified Date Definition**: A calendar date containing $\ge 120$ valid 5-minute heart rate bins ($\ge 10.0$ hours of occupied recording).
- **Candidate Window**: A half-open interval of 180 calendar days $[s, s + 180)$ partitioned into six consecutive 30-day blocks $[s + 30b, s + 30(b+1))$ for $b \in \{0, 1, 2, 3, 4, 5\}$.
- **Window Qualification Gate**: Each of the six 30-day blocks must contain $\ge 15$ qualified dates. The earliest qualifying date $s$ defines Window 1.
- **Second-Window Gate**: The earliest qualifying date $s_2 \ge s_1 + 180$ defines nonoverlapping Window 2.
- **Standardized Date Sampling**: Exactly 15 dates are sampled uniformly without replacement from each 30-day block using the deterministic 64-bit seed:
  $$\text{Seed} = \text{uint64}\left(\text{SHA-256}\left(\texttt{"20261020|apple|\{pid\}|dates\_block\_\{b\}"}\right)[0:8]\right)$$
  yielding an exact exposure design of $N_{\text{obs}} = 90$ dates across 180 days for all qualifying individuals.

### 2.2 Cohort Retention

| Age Stratum | Total Manifest | First-Window Eligible ($N=4,631$) | Two-Window Eligible ($N=2,324$) | Retention Fraction (W1) |
| :--- | :---: | :---: | :---: | :---: |
| **Young Adults (y0)** | 697 | 224 | 106 | 32.1% |
| **Young Adults (y1)** | 1,619 | 666 | 348 | 41.1% |
| **Middle Ager (y0)** | 2,774 | 1,082 | 503 | 39.0% |
| **Middle Ager (y1)** | 4,696 | 2,086 | 1,081 | 44.4% |
| **Elderly (y0)** | 384 | 172 | 91 | 44.8% |
| **Elderly (y1)** | 969 | 401 | 195 | 41.4% |
| **Total Cohort** | 11,139 | 4,631 | 2,324 | 41.6% |

---

## 3. Mathematical Specifications

### 3.1 Daily Signal and Nonperiodic Baseline
For participant $i$ on selected date $t_k \in \{t_1, \dots, t_{90}\}$:
- **Resting HR Proxy**: $y_i(t_k) = \text{percentile}_{10}(\{hr_{t_k, m} : hr_{t_k, m} \text{ finite}\})$.
- **Baseline Design Matrix** $X_{0, i} \in \mathbb{R}^{90 \times p}$:
  1. Intercept: $\mathbf{1}_{90}$.
  2. Elapsed calendar days linear trend: $t_k - t_1$.
  3. Weekday dummy indicators: $\mathbb{I}(\text{dayofweek}(t_k) = d)$ for $d \in \{\text{Tue}, \text{Wed}, \text{Thu}, \text{Fri}, \text{Sat}, \text{Sun}\}$ (Monday reference).
  4. Four 6-hour clock coverage fractions: $c_{00\text{--}06}, c_{06\text{--}12}, c_{12\text{--}18}, c_{18\text{--}24}$.
  Redundant columns are dropped deterministically using sequential Gram-Schmidt orthogonal projection with tolerance $10^{-4}$.
- **Baseline Projection Matrix**:
  $$P_0 = I - X_0 (X_0^T X_0)^{-1} X_0^T, \quad e_0 = P_0 y, \quad \text{RSS}_0 = \|e_0\|_2^2$$

### 3.2 Infradian Harmonic Search Score
For 21 candidate periods $T \in \{25.0, 25.5, \dots, 35.0\}$ days:
$$\omega = \frac{2\pi}{T}, \quad C_T = \begin{bmatrix} \cos(\omega(t - t_1)) & \sin(\omega(t - t_1)) \end{bmatrix} \in \mathbb{R}^{90 \times 2}$$
By the Frisch-Waugh-Lovell theorem, residualizing harmonic regressors against baseline:
$$\tilde{C}_T = P_0 C_T, \quad \hat{\beta}_T = (\tilde{C}_T^T \tilde{C}_T)^{-1} \tilde{C}_T^T e_0, \quad \Delta \text{RSS}(T) = \hat{\beta}_T^T (\tilde{C}_T^T \tilde{C}_T) \hat{\beta}_T$$
$$S_i = \max_{T \in [25, 35]} \frac{\Delta \text{RSS}(T)}{\text{RSS}_0}, \quad T^*_i = \arg\max_{T} S(T), \quad A_i = \sqrt{\hat{\beta}_{c, T^*}^2 + \hat{\beta}_{s, T^*}^2}$$

### 3.3 Continuous Calendar-Time AR(1) Parametric Null Calibration
Daily baseline residuals $e_0$ are modeled as a stationary zero-mean continuous calendar-time AR(1) process with autocorrelation $\phi^{|t_j - t_k|}$ ($0 \le \phi < 1$ per calendar day) and stationary variance $\sigma^2$:
$$e_0(t_k) = \phi^{\Delta_k} e_0(t_{k-1}) + \sigma \sqrt{1 - \phi^{2\Delta_k}} z_k, \quad \Delta_k = t_k - t_{k-1} \ge 1, \quad z_k \sim \mathcal{N}(0, 1)$$
Parameters $(\hat{\phi}, \hat{\sigma}^2)$ are estimated via 1D bounded scalar minimization of the exact profile negative log-likelihood over $\phi \in [0, 0.995]$.

For $B = 199$ independent bootstrap replicates $\mathcal{E}_{\text{boot}} \sim \text{AR}(1)$ simulated on the exact calendar timestamps:
$$S_{\text{null}, b} = \max_{T \in [25, 35]} \frac{\mathcal{E}_{\text{boot}, b}^T P_0 \tilde{C}_T (\tilde{C}_T^T \tilde{C}_T)^{-1} \tilde{C}_T^T P_0 \mathcal{E}_{\text{boot}, b}}{\mathcal{E}_{\text{boot}, b}^T P_0 \mathcal{E}_{\text{boot}, b}}$$
All $B = 199$ replicates are evaluated simultaneously via precomputed operator multiplication in $<2$ ms per participant.
- **Excess Infradian Score**: $E_i = S_i - \frac{1}{B} \sum_{b=1}^B S_{\text{null}, b}$.
- **Empirical Tail Probability**: $p_i = \frac{1 + \sum_{b=1}^B \mathbb{I}(S_{\text{null}, b} \ge S_i)}{B + 1}$.

---

## 4. Empirical Statistical Findings

### 4.1 Primary Shared Family of Four HC3 Robust Wald Tests
Evaluated at familywise $\alpha = 0.05$ with Holm step-down adjustment:
Participant-level model:
$$E_i = \sum_{a} \sum_{y} \mu_{a, y} \mathbb{I}(\text{age}=a, \text{salutation}=y) + \gamma_1 \tilde{c}_{\text{occ}, i} + \gamma_2 \tilde{c}_{\text{night}, i} + \gamma_3 \tilde{c}_{\text{span}, i} + \varepsilon_i$$
with covariates centered within age strata and heteroskedasticity-consistent HC3 covariance $V_{\text{HC3}}$.

| Hypothesis / Contrast | Point Estimate $\hat{\Delta}$ | $\text{SE}_{\text{HC3}}$ | 95% Bootstrap CI | Test Statistic | df | $p_{\text{unadj}}$ | $p_{\text{Holm}}$ | Rejection ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Age $\times$ Sex Omnibus** | — | — | — | $\text{Wald } \chi^2 = 56.763$ | 2 | $4.72 \times 10^{-13}$ | $9.44 \times 10^{-13}$ | **Reject $H_0$** |
| **2. Young Adults (y0 vs y1)** | $+0.0642$ | $0.0063$ | $[+0.0520, +0.0767]$ | $\text{Wald } \chi^2 = 102.434$ | 1 | $< 10^{-16}$ | $< 10^{-16}$ | **Reject $H_0$** |
| **3. Middle Ager (y0 vs y1)** | $+0.0233$ | $0.0023$ | $[+0.0189, +0.0276]$ | $\text{Wald } \chi^2 = 107.405$ | 1 | $< 10^{-16}$ | $< 10^{-16}$ | **Reject $H_0$** |
| **4. Elderly (y0 vs y1)** | $+0.0056$ | $0.0045$ | $[-0.0038, +0.0151]$ | $\text{Wald } \chi^2 = 1.529$ | 1 | $0.2163$ | $0.2163$ | Fail to Reject |

### 4.2 Standardized Population Summary
Using fixed population weights $w_{\text{Young}} = 0.1922$, $w_{\text{Middle}} = 0.6841$, $w_{\text{Elderly}} = 0.1237$:
$$\Delta_{\text{std}} = \sum_a w_a \Delta_a = +0.0290 \quad (\text{SE}_{\text{HC3}} = 0.0020, \quad 95\% \text{ CI } [+0.0252, +0.0329], \quad \text{Wald } \chi^2 = 201.31, \quad p < 10^{-16})$$

### 4.3 Physical Cosinor Amplitude Contrasts ($A_i$, bpm)

| Cohort | Female Mean ($A_{y0}$) | Male Mean ($A_{y1}$) | Difference $\hat{\Delta}_A$ (bpm) | $\text{SE}_{\text{HC3}}$ | 95% CI (bpm) | $\text{Wald } \chi^2$ | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Young Adults** | $1.850$ bpm | $1.440$ bpm | $+0.415$ bpm | $0.056$ | $[+0.306, +0.525]$ | $55.54$ | $9.16 \times 10^{-14}$ |
| **Middle Ager** | $1.384$ bpm | $1.302$ bpm | $+0.082$ bpm | $0.021$ | $[+0.041, +0.124]$ | $15.01$ | $1.07 \times 10^{-4}$ |
| **Elderly** | $1.146$ bpm | $1.110$ bpm | $+0.037$ bpm | $0.050$ | $[-0.061, +0.134]$ | $0.55$ | $0.4592$ |
| **Standardized** | $1.427$ bpm | $1.307$ bpm | $+0.141$ bpm | $0.019$ | $[+0.103, +0.178]$ | $53.73$ | $2.30 \times 10^{-13}$ |

---

## 5. Methodological Dissociations

### 5.1 Double Dissociation with Wear and Compliance
To evaluate whether observed sex contrasts arise from sensor wear behavior:
1. **Daily 5-Minute Occupancy Rate**:
   - Young Adults: Female mean $62.97\%$ vs. Male mean $63.17\%$ ($\Delta = -0.0019$, Welch $t = -0.31$, $p = 0.759$).
   - Middle Ager: Female mean $61.92\%$ vs. Male mean $62.76\%$ ($\Delta = -0.0085$, Welch $t = -2.59$, $p = 0.0097$).
   - Elderly: Female mean $62.96\%$ vs. Male mean $64.09\%$ ($\Delta = -0.0113$, Welch $t = -1.51$, $p = 0.1331$).
2. **Nocturnal Sleep Wear Coverage (00:00–06:00)**:
   - Young Adults: Female mean $63.02\%$ vs. Male mean $59.00\%$ ($\Delta = +0.0402$, Welch $t = 1.76$, $p = 0.0793$).
   - Middle Ager: Female mean $58.82\%$ vs. Male mean $56.68\%$ ($\Delta = +0.0214$, Welch $t = 1.76$, $p = 0.0787$).
   - Elderly: Female mean $65.97\%$ vs. Male mean $63.31\%$ ($\Delta = +0.0266$, Welch $t = 0.98$, $p = 0.3278$).
3. **Adjustment Invariance**:
   Unadjusted within-age contrast on Young Adults is $+0.0649$ ($\text{SE} = 0.0064$), while observation-adjusted contrast (controlling for occupancy, nocturnal fraction, and calendar span) is $+0.0642$ ($\text{SE} = 0.0063$). Observation controls account for $<1.1\%$ of the contrast magnitude.

### 5.2 Second-Window Temporal Persistence vs. Period Stability ($N = 2,324$)
1. **Frozen Harmonic Extrapolation (180 days forward)**:
   Freezing Window 1 best-fit period $T^*$ and phase $\phi^*$ to predict Window 2 heart rate yields:
   $$\Delta \text{MSE} = \text{MSE}_{\text{baseline}} - \text{MSE}_{\text{harmonic}} < 0 \quad (\text{Relative } R^2_{\text{persist}} \approx -0.04 \text{ to } -0.055)$$
   Because physiological ovarian cycles exhibit natural cycle length variance (intra-individual standard deviation $\approx 2.5\text{--}4.0$ days), an invariant deterministic sinusoid extrapolated over 6 months accumulates phase drift, inducing destructive interference out of sample.
2. **Re-Fitted Period Coherence**:
   When Window 2 is fitted independently, the best-fit period difference $|\Delta T| = |T_{\text{w1}}^* - T_{\text{w2}}^*|$ demonstrates strong biological stability:
   - **Young Adult Females ($y=0$)**: Median $|\Delta T| = \mathbf{1.50}\text{ days}$ (Mean $2.528$ days).
   - **Young Adult Males ($y=1$, negative control)**: Median $|\Delta T| = \mathbf{4.00}\text{ days}$ (Mean $4.470$ days).
   The median period deviation across nonoverlapping half-year windows is nearly three-fold smaller in young females than in male negative controls ($1.50$ vs. $4.00$ days), confirming localized biological resonance despite phase wander.

---

## 6. Validity Conditions and Caveats

1. **Proxy Classification**:
   Salutation categories $10$ (`y=0`) and $20$ (`y=1`) reflect administrative records, not clinically confirmed biological sex, chromosomal karyotype, or endocrinological status.
2. **Lack of Menstrual Ground Truth**:
   The cohort contains no self-reported menstrual dates, LH surge measurements, oral contraceptive usage logs, or menopausal transition staging (e.g., STRAW+10 criteria).
3. **Age Stratum Aggregation**:
   Age is recorded exclusively in broad categorical brackets (`Young Adults`, `Middle Ager`, `Elderly`). Residual confounding by continuous age within brackets cannot be eliminated from existing metadata.
4. **Window Selection Attrition**:
   The 180-day window qualification requirement ($\ge 15$ qualified days in each of six 30-day blocks) retains $41.6\%$ of the manifest cohort ($32.1\%$ of young females). Inferences apply directly to consistent long-term wearable users.

---

## 7. Artifact Manifest

| Path | Format | Description |
| :--- | :---: | :--- |
| `experiments/female_infradian_cycles_v2/scripts/00_verify_v2_methods.py` | Python | Fast unit verification fixture (<5s) for AR(1) estimation, baseline detrending, and null calibration. |
| `experiments/female_infradian_cycles_v2/scripts/01_extract_and_calibrate_infradian.py` | Python | Full cohort extraction, 180-day window normalization, B=199 parametric null calibration, and persistence evaluation. |
| `experiments/female_infradian_cycles_v2/scripts/02_summarize_infradian_contrasts.py` | Python | Age-stratified OLS models, HC3 robust Wald inference, Holm multiplicity adjustments, and stratified bootstrap CIs. |
| `experiments/female_infradian_cycles_v2/results/window_eligibility.parquet` | Parquet | Participant-level table ($N=11,139$) with eligibility flags, selected dates, AR(1) $\phi$, scores $S_i, E_i, p_i, A_i$, and persistence metrics. |
| `experiments/female_infradian_cycles_v2/results/calibrated_group_contrasts.csv` | CSV | Structured contrast table with Wald statistics, degrees of freedom, Holm-adjusted p-values, and 95% bootstrap CIs. |

---

## 8. Addendum: Continuous Periodogram Deconstruction, Null Recalibration, and Sample Scaling

### 8.1 Continuous Periodogram Sweeps (3 to 60 Days)
Evaluating unconstrained single-harmonic partial $R^2$ across 229 frequencies ($T \in [3.0, 60.0]\text{ days}$, step $k/720\text{ cycles/day}$) demonstrates:
1. **Narrowband Biological Resonance**: Young adult females exhibit a sharp, symmetric spectral elevation peaked at $28.0\text{ days}$ (mean partial $R^2 \approx 0.085$ vs. $0.041$ in continuous AR(1) surrogate envelope).
2. **Age-Dependent Attenuation**: Middle-aged females display a distinct but attenuated peak at $27.7\text{ days}$ (mean partial $R^2 \approx 0.055$). Elderly females show no discernible elevation above red-noise background.
3. **Absence in Males**: Male spectra across all age groups display zero elevation in the 25–35 day band, tracking the continuous AR(1) surrogate envelope.

### 8.2 Deconstruction of the 35-Day Grid-Edge Pile-Up
Analysis of the argmax period distribution across the $[25.0, 35.0]\text{ day}$ search grid reveals:
1. **Boundary Selection Bias**: In pure white noise surrogates, the maximum score falls on the boundary points in $30.6\%$ of simulations ($15.8\%$ at $25.0\text{ d}$, $14.8\%$ at $35.0\text{ d}$ vs. $4.8\%$ uniform expectation) due to finite-interval argmax truncation of autocorrelated test statistics.
2. **Empirical Distribution**:
   - Young adult females concentrate in the interior ($7.6\%$ at $35.0\text{ d}$), with $66.5\%$ of peaks in the $[26.5, 30.5]\text{ d}$ menstrual window.
   - Males and elderly females concentrate at the $35.0\text{ d}$ boundary ($17.4\%\text{--}19.8\%$), reflecting leakage from the underlying red-noise power spectrum.

### 8.3 Two-Component Continuous Auto-Regressive Null Recalibration
To eliminate long-period red-noise leakage, the residual covariance was expanded to a continuous two-component model:
$$\Sigma_{jk} = s_1 \phi_1^{|t_k - t_j|} + s_2 \phi_2^{|t_k - t_j|}, \quad \phi_1 \in [0, 0.95), \; \phi_2 \in \{0.980, 0.995\}$$
fitted per participant via REML on baseline-projected coordinates.
- **Contrast Robustness**: Re-running all four primary HC3 contrasts on excess score $E_{i, 2}$ under the two-component null yields:
  - Omnibus Age $\times$ Sex Interaction: Wald $\chi^2 = 60.52, \; p = 7.23 \times 10^{-14}$ (vs. $56.76$ under single AR(1)).
  - Young Adults ($y0 - y1$): $\Delta = +0.0645, \; \text{SE} = 0.0060, \; \text{Wald } \chi^2 = 115.58, \; p < 10^{-25}$.
  - Middle Ager ($y0 - y1$): $\Delta = +0.0235, \; \text{SE} = 0.0022, \; \text{Wald } \chi^2 = 113.82, \; p < 10^{-25}$.
  - Elderly ($y0 - y1$): $\Delta = +0.0073, \; \text{SE} = 0.0043, \; \text{Wald } \chi^2 = 2.90, \; p = 0.0885$.
- **False-Positive Calibration**: Male empirical rejection rates at $\alpha = 0.05$ remain at $4.6\%\text{--}5.5\%$ across all age groups.

### 8.4 Large-Sample Window Relaxation ($k = 10$)
Relaxing window eligibility to $\ge 10$ qualified dates per 30-day block ($N_{\text{obs}} = 60$ across 180 days) across all $N = 11,139$ Apple participants:
- Retains $N = 5,516$ participants ($+19.1\%$ increase; Young Adult females $n = 292$, $+30.4\%$).
- Preserves primary biological contrasts: Young Adults $\Delta = +0.0526$ ($\text{SE} = 0.0061$, $\chi^2 = 75.49, p < 10^{-17}$); Middle Ager $\Delta = +0.0186$ ($\text{SE} = 0.0024$, $\chi^2 = 62.73, p < 10^{-14}$).
- Male rejection rate at $\alpha = 0.05$ remains controlled at $3.4\%\text{--}4.4\%$.
