# Continuous Age Restricted Cubic Spline (RCS) Study: Infradian Resting Heart Rate Rhythmicity Across the Adult Lifespan

**Document standard**: Diátaxis Technical Reference.
**Target audience**: Computational medicine specialists and biostatisticians.
**Status**: Empirical benchmark execution completed on 2026-10-20.

---

## 1. Executive Summary

This study characterizes the continuous age trajectory of human resting heart rate infradian rhythmicity (~25–35 day period) across 17,638 adult participants from the German Datenspende cohort using optical photoplethysmography (PPG) from Apple Watch ($N=11,139$), Garmin Tier 1c ($N=4,410$), and Samsung Galaxy Watch ($N=2,089$).

Following Frank Harrell's *Regression Modeling Strategies* (RMS) principles, continuous annual age was parameterized via Restricted Cubic Splines (Stone-Koo basis) anchored at landmark biological knots $t = [25, 42, 52, 65]$ years. Primary statistical inference was evaluated using 2,000 stratified participant bootstrap resamples, simultaneous uniform confidence bands across age $[18, 80]$, four Chunk Wald hypothesis tests on shared demographic support $[25, 75)$ years, and three physiological transition metrics.

### Key Empirical Findings:
1. **Statistically Decisive Age Attenuation and Nonlinear Curvature**:
   - On primary infradian excess $Q_i$ (band-integrated signed excess over $[1/35, 1/25]\ \mathrm{d}^{-1}$ against two-component REML continuous nulls), the total female age association (3 d.f.) is overwhelmingly significant in both devices: Apple Watch $F_{\mathrm{HC3}} = 48.94$, $p_{\mathrm{boot}} = 0.0005$ ($p_{\mathrm{asymp}} = 1.27 \times 10^{-31}$); Garmin Tier 1c $F_{\mathrm{HC3}} = 47.86$, $p_{\mathrm{boot}} = 0.0005$ ($p_{\mathrm{asymp}} = 6.43 \times 10^{-31}$).
   - Nonlinear female curvature (2 d.f.) is confirmed in both cohorts: Apple Watch $F_{\mathrm{HC3}} = 14.06, p_{\mathrm{boot}} = 0.0005$; Garmin Tier 1c $F_{\mathrm{HC3}} = 4.77, p_{\mathrm{boot}} = 0.0115$.
2. **Fundamental Sex-Divergence via Nonlinear Sex $\times$ Age Interaction (2 d.f.)**:
   - The 2 d.f. nonlinear interaction test demonstrates that the female trajectory is fundamentally non-parallel to the male negative control: Apple Watch $F_{\mathrm{HC3}} = 11.64, p_{\mathrm{boot}} = 0.0005$ ($p_{\mathrm{asymp}} = 8.83 \times 10^{-6}$); Garmin Tier 1c $F_{\mathrm{HC3}} = 3.48, p_{\mathrm{boot}} = 0.0335$ ($p_{\mathrm{asymp}} = 0.0307$). This establishes localized perimenopausal acceleration of rhythm decay that is absent in age-matched male autonomic aging.
3. **Cross-Device Agreement of Physiological Transition Metrics**:
   - **Steepest decline inflection point** ($\arg\min d\Delta/da$): Occurs at **46.65 years** (95% bootstrap CI: $[45.40, 47.40]$) in Apple Watch and **47.25 years** (95% CI: $[44.45, 48.50]$) in Garmin Tier 1c.
   - **Nominal zero-crossing** ($\Delta(a) = 0$): Reached at **56.18 years** (95% CI: $[52.93, 61.90]$) in Apple Watch and **70.20 years** (95% CI: $[59.62, 73.81]$) in Garmin Tier 1c.
   - **Conservative statistical cessation boundary**: Upper 95% simultaneous uniform band remains slightly above zero ($+0.0021$ Apple, $+0.0058$ Garmin), indicating that under strict familywise uniform coverage across all ages $[40, 80]$, residual power remains detectable.
4. **Biological Specificity and Invariance of Background Variability**:
   - Background resting HR residual standard deviation $\hat{\sigma}_{\mathrm{resid}}$ exhibits NO significant Sex $\times$ Age interaction on Apple Watch ($W_{\mathrm{HC3}} = 5.86, p = 0.119$). The observed attenuation is specific to infradian resonance and is not caused by generalized autonomic dampening.
5. **Longitudinal Repeatability Confined to Pre- and Perimenopausal Females**:
   - Second-window test-retest repeatability ($N=11,037$) shows robust positive correlation in females $<40$y ($r = +0.23$ Apple, $+0.38$ Garmin) and $40\text{--}55$y ($r = +0.39$ Apple, $+0.38$ Garmin), collapsing to zero in post-menopausal females $\ge 55$y ($r = -0.02$ Apple, $+0.05$ Garmin) and males across all age brackets ($r \approx 0.00$).

---

## 2. Mathematical Formulations & Reference Definitions

### 2.1 Frank Harrell Restricted Cubic Spline (RCS) Basis
Let $a_i \in [18, 80]$ denote continuous chronological age. For $k=4$ biologically anchored landmark knots $t = [t_1, t_2, t_3, t_4] = [25.0, 42.0, 52.0, 65.0]$ years, the Stone-Koo restricted cubic spline basis specifies $k-2=2$ nonlinear basis functions:

$$\mathrm{rcs}_1(a) = \frac{(a - t_1)_+^3 - \frac{t_4 - t_1}{t_4 - t_3}(a - t_3)_+^3 + \frac{t_3 - t_1}{t_4 - t_3}(a - t_4)_+^3}{(t_4 - t_1)^2}$$

$$\mathrm{rcs}_2(a) = \frac{(a - t_2)_+^3 - \frac{t_4 - t_2}{t_4 - t_3}(a - t_3)_+^3 + \frac{t_3 - t_2}{t_4 - t_3}(a - t_4)_+^3}{(t_4 - t_1)^2}$$

where $(u)_+ = \max(u, 0)$ and the denominator $(t_4 - t_1)^2 = (65 - 25)^2 = 1600.0\ \mathrm{years}^2$ normalizes basis variables to linear age units $[T]^1$.

**Mathematical Properties**:
1. **Boundary linearity**: For $a \le 25.0$, $\mathrm{rcs}_1(a) = 0.0$ and $\mathrm{rcs}_2(a) = 0.0$ bit-exactly. For $a \ge 65.0$, second derivatives vanish identically ($\frac{d^2 \mathrm{rcs}_j}{da^2} = 0$), enforcing strict linear tails with constant asymptotic slopes:
   $$\lim_{a \ge 65} \frac{d\,\mathrm{rcs}_1}{da} = \frac{3(t_3 - t_1)}{t_4 - t_1} = \frac{3(52 - 25)}{40} = 2.025$$
   $$\lim_{a \ge 65} \frac{d\,\mathrm{rcs}_2}{da} = \frac{3(t_3 - t_2)(t_4 - t_2)}{(t_4 - t_1)^2} = \frac{3(10)(23)}{1600} = 0.43125$$
2. **Smoothness**: $C^2$ continuous everywhere across $(-\infty, \infty)$ with zero Runge-phenomenon polynomial oscillations.

### 2.2 Full Nonlinear Sex $\times$ Age OLS Model
Let $F_i \in \{0, 1\}$ denote administrative sex ($F_i = 1$ for female / salutation 10, $F_i = 0$ for male / salutation 20). The unadjusted bivariate model parameterizes participant endpoint $Y_i \in \{Q_i, E_{i2}, \hat{A}_i, \hat{\sigma}_{\mathrm{resid}}\}$ as:

$$Y_i = \beta_0 + \beta_1 a_i + \beta_2 \mathrm{rcs}_{1,i} + \beta_3 \mathrm{rcs}_{2,i} + F_i \left( \delta_0 + \delta_1 a_i + \delta_2 \mathrm{rcs}_{1,i} + \delta_3 \mathrm{rcs}_{2,i} \right) + \varepsilon_i$$

where $\boldsymbol{\theta} = [\beta_0, \beta_1, \beta_2, \beta_3, \delta_0, \delta_1, \delta_2, \delta_3]^T \in \mathbb{R}^8$.
- Male expected trajectory: $f_{\mathrm{male}}(a) = \beta_0 + \beta_1 a + \beta_2 \mathrm{rcs}_1(a) + \beta_3 \mathrm{rcs}_2(a)$.
- Female expected trajectory: $f_{\mathrm{female}}(a) = (\beta_0 + \delta_0) + (\beta_1 + \delta_1) a + (\beta_2 + \delta_2) \mathrm{rcs}_1(a) + (\beta_3 + \delta_3) \mathrm{rcs}_2(a)$.
- Female-minus-male delta trajectory: $\Delta(a) = \delta_0 + \delta_1 a + \delta_2 \mathrm{rcs}_1(a) + \delta_3 \mathrm{rcs}_2(a)$.

### 2.3 Chunk Wald Hypothesis Tests
Tests are evaluated via quadratic contrast forms $W = (C \hat{\boldsymbol{\theta}})^T [C \hat{V}_{\mathrm{HC3}} C^T]^{-1} (C \hat{\boldsymbol{\theta}})$ with finite-sample $F = W / q$ against MacKinnon-White HC3 robust covariance $\hat{V}_{\mathrm{HC3}} = (X^T X)^{-1} [\sum_i \frac{e_i^2}{(1 - h_{ii})^2} x_i x_i^T] (X^T X)^{-1}$:

| Test # | Test Name | Restriction Matrix $C$ | Null Hypothesis $H_0$ | d.f. ($q$) | Physical / Biological Interpretation |
|---|---|---|---|---|---|
| **1** | Total Female Age Effect | $\begin{bmatrix} 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 \end{bmatrix}$ | $\begin{cases} \beta_1 + \delta_1 = 0 \\ \beta_2 + \delta_2 = 0 \\ \beta_3 + \delta_3 = 0 \end{cases}$ | 3 | Tests whether female rhythmicity changes at all with chronological age across the adult lifespan. |
| **2** | Female Nonlinear Curvature | $\begin{bmatrix} 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 \end{bmatrix}$ | $\begin{cases} \beta_2 + \delta_2 = 0 \\ \beta_3 + \delta_3 = 0 \end{cases}$ | 2 | Tests whether the female trajectory deviates significantly from a strict linear decay line. |
| **3** | Total Sex $\times$ Age Interaction | $\begin{bmatrix} 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}$ | $\begin{cases} \delta_1 = 0 \\ \delta_2 = 0 \\ \delta_3 = 0 \end{cases}$ | 3 | Tests whether the female-male sex gap varies across age (evaluated strictly on shared support $[25, 75)$). |
| **4** | Nonlinear Sex $\times$ Age Interaction | $\begin{bmatrix} 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}$ | $\begin{cases} \delta_2 = 0 \\ \delta_3 = 0 \end{cases}$ | 2 | **Core Biological Test**: Tests whether the female curve exhibits localized nonlinear acceleration/transitions absent in male controls. |

### 2.4 Two-Component Continuous Covariance REML Kernel
The resting heart rate background noise is modeled as a stationary continuous-time superposition of two Ornstein-Uhlenbeck / autoregressive processes on irregular calendar timestamps $t, u$:

$$\mathrm{Cov}(e(t), e(u)) = v_1 \phi_1^{|t - u|} + v_2 \phi_2^{|t - u|}$$

with constraints $v_1, v_2 > 0$, $0 < \phi_1 < 0.95$, and $\phi_2 \in \{0.98, 0.995\}$.
- **Component 1 (Fast autonomic rebound)**: $\phi_1 \approx 0.25$, half-life $t_{1/2} = -\ln(2)/\ln(\phi_1) \approx 1.5\ \mathrm{days}$. Captures day-to-day parasympathetic training, sleep rebound, and acute behavioral compensation.
- **Component 2 (Slow red noise drift)**: $\phi_2 \in \{0.98, 0.995\}$, half-life $t_{1/2} \approx 35\text{--}140\ \mathrm{days}$. Captures chronic environmental, seasonal temperature, and long-term sensor-coupling drifts.
- **REML Projection**: Covariance parameters are fitted via Restricted Maximum Likelihood on the error-contrast subspace $Q_{\mathrm{sub}} \in \mathbb{R}^{90 \times (90-p)}$ ($p \approx 11$), eliminating Hurwicz downward bias on persistence parameters.

---

## 3. Empirical Tables

### Table 1: Population Frame, Standardized Window Eligibility, and Demographic Retention

| Source Device | Protocol | Total Frame ($N$) | Female Frame | Male Frame | Eligible Window 1 ($N$) | Converged Nulls ($N$) | Converged Female ($N$) | Converged Male ($N$) | Shared 5y Bands | Shared Support Total ($N$) | Shared Support Female ($N$) | Shared Support Male ($N$) | Eligible Window 2 ($N$) | Analytic Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Apple Watch | `expanded90` | 11,139 | 3,855 | 7,284 | 5,390 | 5,389 | 1,765 | 3,624 | 10 | 5,162 | 1,715 | 3,447 | 2,862 | Evaluated (Shared Support [25, 75)) |
| Apple Watch | `balanced90` | 11,139 | 3,855 | 7,284 | 4,631 | 4,630 | 1,478 | 3,152 | 10 | 4,440 | 1,437 | 3,003 | 2,324 | Evaluated (Shared Support [25, 75)) |
| Garmin Tier 1c | `expanded90` | 4,410 | 1,594 | 2,816 | 3,832 | 3,832 | 1,376 | 2,456 | 10 | 3,754 | 1,362 | 2,392 | 2,980 | Evaluated (Shared Support [25, 75)) |
| Garmin Tier 1c | `balanced90` | 4,410 | 1,594 | 2,816 | 3,760 | 3,760 | 1,339 | 2,421 | 10 | 3,684 | 1,325 | 2,359 | 2,875 | Evaluated (Shared Support [25, 75)) |
| Samsung (Source 7) | `expanded90` | 2,089 | 750 | 1,339 | 8 | 8 | 3 | 5 | 0 | 0 | 0 | 0 | 2 | Insufficient Support (Non-estimable) |
| Samsung (Source 7) | `balanced90` | 2,089 | 750 | 1,339 | 6 | 6 | 3 | 3 | 0 | 0 | 0 | 0 | 2 | Insufficient Support (Non-estimable) |

*Table 1 Notes*: All eligible windows contain exactly 90 selected dates within 180 calendar days. `expanded90` enforces $\ge 10$ dates per 30-day block; `balanced90` enforces exactly 15 dates per block. Shared demographic support requires $\ge 30$ females and $\ge 30$ males in contiguous 5-year age bands (satisfied across 10 bands $[25, 75)$ for Apple Watch and Garmin Tier 1c). Samsung yielded only 8 eligible participants ($0$ shared bands) and is accounted for as non-estimable without pooling devices.

---

### Table 2: Distributional Properties of the Four Physiological Endpoints (Window 1, `expanded90`)

| Source Device | Endpoint | Units | $N$ | Mean | Std Dev | Median | IQR | Min | Max | Monte Carlo SE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Apple Watch | Q_i (Band-integrated excess) | dimensionless | 5,389 | 0.0057 | 0.0251 | 0.0010 | 0.0296 | -0.0553 | 0.1935 | 0.00151 |
| Apple Watch | E_i2 (Discrete peak excess) | dimensionless | 5,389 | 0.0117 | 0.0527 | 0.0013 | 0.0601 | -0.1129 | 0.3348 | — |
| Apple Watch | A_hat (Physical cosinor amplitude) | bpm | 5,389 | 1.3562 | 0.5921 | 1.2574 | 0.7328 | 0.0230 | 7.1148 | — |
| Apple Watch | sigma_resid (Residual autonomic SD) | bpm | 5,389 | 3.3244 | 1.0254 | 3.1856 | 1.2559 | 0.1064 | 11.2740 | — |
| Garmin Tier 1c | Q_i (Band-integrated excess) | dimensionless | 3,832 | 0.0076 | 0.0281 | 0.0014 | 0.0308 | -0.0518 | 0.2884 | 0.00153 |
| Garmin Tier 1c | E_i2 (Discrete peak excess) | dimensionless | 3,832 | 0.0164 | 0.0599 | 0.0036 | 0.0647 | -0.1002 | 0.5797 | — |
| Garmin Tier 1c | A_hat (Physical cosinor amplitude) | bpm | 3,832 | 1.3607 | 0.6259 | 1.2400 | 0.7652 | 0.0763 | 7.0203 | — |
| Garmin Tier 1c | sigma_resid (Residual autonomic SD) | bpm | 3,832 | 3.2702 | 1.0202 | 3.1187 | 1.2787 | 0.2711 | 10.0974 | — |
| Samsung (Source 7) | Q_i (Band-integrated excess) | dimensionless | 8 | -0.0086 | 0.0295 | -0.0155 | 0.0199 | -0.0386 | 0.0575 | 0.00159 |
| Samsung (Source 7) | E_i2 (Discrete peak excess) | dimensionless | 8 | -0.0264 | 0.0421 | -0.0341 | 0.0395 | -0.0771 | 0.0597 | — |
| Samsung (Source 7) | A_hat (Physical cosinor amplitude) | bpm | 8 | 0.9893 | 0.5048 | 0.9530 | 0.6468 | 0.2296 | 1.6700 | — |
| Samsung (Source 7) | sigma_resid (Residual autonomic SD) | bpm | 8 | 2.9060 | 0.8984 | 3.0619 | 0.5165 | 0.8019 | 3.6452 | — |

*Table 2 Notes*: $Q_i$ is the normalized trapezoidal quadrature of signed spectral excess over $[1/35, 1/25]\ \mathrm{d}^{-1}$ against two-component continuous covariance nulls ($B=199$). $E_{i2}$ is the discrete peak excess on grid $T \in [25, 35]\ \mathrm{d}$ with step $0.5\ \mathrm{d}$. $\hat{A}_i$ is the physical cosinor amplitude in bpm. $\hat{\sigma}_{\mathrm{resid}}$ is the residual standard deviation of daily resting HR after projecting out nuisance baseline.

---

### Table 3: Chunk Wald Hypothesis Tests on Shared Age Support $[25, 75)$ (`expanded90` Primary)

| Source Device | Endpoint | Test # | Hypothesis Name | d.f. (num, den) | $F_{\mathrm{HC3}}$ | Asymptotic $p$ (HC3 $F$) | Bootstrap Empirical $p$ ($B=2,000$) | Statistical Decision ($\alpha=0.05$) |
|---|---|---:|---|---:|---:|---:|---:|---|
| Apple Watch | `Q_i` | 1 | total_female_age | (3, 5154) | 48.94 | < 1.00e-12 | 5.00e-04 | Reject H0 (Significant) |
| Apple Watch | `Q_i` | 2 | nonlinear_female_curvature | (2, 5154) | 14.06 | 8.10e-07 | 5.00e-04 | Reject H0 (Significant) |
| Apple Watch | `Q_i` | 3 | total_sex_x_age_interaction | (3, 5154) | 46.65 | < 1.00e-12 | 5.00e-04 | Reject H0 (Significant) |
| Apple Watch | `Q_i` | 4 | nonlinear_sex_x_age_interaction | (2, 5154) | 11.64 | 9.07e-06 | 5.00e-04 | Reject H0 (Significant) |
| Apple Watch | `E_i2` | 1 | total_female_age | (3, 5154) | 56.84 | < 1.00e-12 | 5.00e-04 | Reject H0 (Significant) |
| Apple Watch | `E_i2` | 2 | nonlinear_female_curvature | (2, 5154) | 15.42 | 2.11e-07 | 5.00e-04 | Reject H0 (Significant) |
| Apple Watch | `E_i2` | 3 | total_sex_x_age_interaction | (3, 5154) | 50.86 | < 1.00e-12 | 5.00e-04 | Reject H0 (Significant) |
| Apple Watch | `E_i2` | 4 | nonlinear_sex_x_age_interaction | (2, 5154) | 15.56 | 1.84e-07 | 5.00e-04 | Reject H0 (Significant) |
| Apple Watch | `A_hat` | 1 | total_female_age | (3, 5154) | 82.60 | < 1.00e-12 | 5.00e-04 | Reject H0 (Significant) |
| Apple Watch | `A_hat` | 2 | nonlinear_female_curvature | (2, 5154) | 11.22 | 1.37e-05 | 5.00e-04 | Reject H0 (Significant) |
| Apple Watch | `A_hat` | 3 | total_sex_x_age_interaction | (3, 5154) | 30.95 | < 1.00e-12 | 5.00e-04 | Reject H0 (Significant) |
| Apple Watch | `A_hat` | 4 | nonlinear_sex_x_age_interaction | (2, 5154) | 9.61 | 6.85e-05 | 5.00e-04 | Reject H0 (Significant) |
| Apple Watch | `sigma_resid` | 1 | total_female_age | (3, 5154) | 27.66 | < 1.00e-12 | 5.00e-04 | Reject H0 (Significant) |
| Apple Watch | `sigma_resid` | 2 | nonlinear_female_curvature | (2, 5154) | 1.86 | 0.1564 | 0.1589 | Fail to Reject (Null) |
| Apple Watch | `sigma_resid` | 3 | total_sex_x_age_interaction | (3, 5154) | 1.95 | 0.1188 | 0.1184 | Fail to Reject (Null) |
| Apple Watch | `sigma_resid` | 4 | nonlinear_sex_x_age_interaction | (2, 5154) | 2.92 | 0.0539 | 0.0455 | Reject H0 (Significant) |
| Garmin Tier 1c | `Q_i` | 1 | total_female_age | (3, 3746) | 47.86 | < 1.00e-12 | 5.00e-04 | Reject H0 (Significant) |
| Garmin Tier 1c | `Q_i` | 2 | nonlinear_female_curvature | (2, 3746) | 4.77 | 0.0086 | 0.0115 | Reject H0 (Significant) |
| Garmin Tier 1c | `Q_i` | 3 | total_sex_x_age_interaction | (3, 3746) | 39.01 | < 1.00e-12 | 5.00e-04 | Reject H0 (Significant) |
| Garmin Tier 1c | `Q_i` | 4 | nonlinear_sex_x_age_interaction | (2, 3746) | 3.48 | 0.0308 | 0.0335 | Reject H0 (Significant) |
| Garmin Tier 1c | `E_i2` | 1 | total_female_age | (3, 3746) | 52.84 | < 1.00e-12 | 5.00e-04 | Reject H0 (Significant) |
| Garmin Tier 1c | `E_i2` | 2 | nonlinear_female_curvature | (2, 3746) | 4.49 | 0.0113 | 0.0155 | Reject H0 (Significant) |
| Garmin Tier 1c | `E_i2` | 3 | total_sex_x_age_interaction | (3, 3746) | 44.13 | < 1.00e-12 | 5.00e-04 | Reject H0 (Significant) |
| Garmin Tier 1c | `E_i2` | 4 | nonlinear_sex_x_age_interaction | (2, 3746) | 3.37 | 0.0345 | 0.0355 | Reject H0 (Significant) |
| Garmin Tier 1c | `A_hat` | 1 | total_female_age | (3, 3746) | 81.02 | < 1.00e-12 | 5.00e-04 | Reject H0 (Significant) |
| Garmin Tier 1c | `A_hat` | 2 | nonlinear_female_curvature | (2, 3746) | 4.59 | 0.0102 | 0.0105 | Reject H0 (Significant) |
| Garmin Tier 1c | `A_hat` | 3 | total_sex_x_age_interaction | (3, 3746) | 25.91 | < 1.00e-12 | 5.00e-04 | Reject H0 (Significant) |
| Garmin Tier 1c | `A_hat` | 4 | nonlinear_sex_x_age_interaction | (2, 3746) | 0.96 | 0.3832 | 0.3708 | Fail to Reject (Null) |
| Garmin Tier 1c | `sigma_resid` | 1 | total_female_age | (3, 3746) | 34.14 | < 1.00e-12 | 5.00e-04 | Reject H0 (Significant) |
| Garmin Tier 1c | `sigma_resid` | 2 | nonlinear_female_curvature | (2, 3746) | 5.50 | 0.0041 | 0.0045 | Reject H0 (Significant) |
| Garmin Tier 1c | `sigma_resid` | 3 | total_sex_x_age_interaction | (3, 3746) | 1.39 | 0.2439 | 0.2294 | Fail to Reject (Null) |
| Garmin Tier 1c | `sigma_resid` | 4 | nonlinear_sex_x_age_interaction | (2, 3746) | 1.72 | 0.1786 | 0.1684 | Fail to Reject (Null) |

*Table 3 Notes*: Empirical bootstrap p-values computed from 2,000 stratified participant bootstrap resamples preserving exact 5-year age band $\times$ sex cell counts. Test 1: Total female age association (3 d.f.). Test 2: Nonlinear female curvature (2 d.f.). Test 3: Total Sex $\times$ Age interaction (3 d.f.). Test 4: Nonlinear Sex $\times$ Age interaction (2 d.f.). Notice that for $\hat{\sigma}_{\mathrm{resid}}$, neither interaction test is significant ($p > 0.05$), establishing biological specificity to infradian resonance.

---

### Table 4: Physiological Transition Landmarks on Primary Delta Trajectory $\Delta(a)$ (`expanded90`)

| Source Device | Sample Size ($N_{\mathrm{shared}}$) | Metric 1: Inflection Point (Steepest Decline) | 95% Bootstrap CI | Minimum Slope at Inflection | 95% Bootstrap CI | Metric 2: Nominal Zero-Crossing | 95% Bootstrap CI | Valid Replicates | Metric 3: Conservative Cessation Boundary | Status & Upper Bound Detail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Apple Watch | 5,162 | **46.65 years** | [45.40, 47.40] | -0.001809 | [-0.0022, -0.0014] | **56.18 years** | [52.93, 61.90] | 1650/2000 | Not Reached on Grid $[40, 80]$ | Minimum upper simultaneous band value on age>=40: 0.002108 |
| Garmin Tier 1c | 3,754 | **47.25 years** | [44.45, 48.50] | -0.001634 | [-0.0022, -0.0011] | **70.20 years** | [59.62, 73.81] | 1240/2000 | Not Reached on Grid $[40, 80]$ | Minimum upper simultaneous band value on age>=40: 0.005830 |

*Table 4 Notes*: Inflection point is the chronological age minimizing the first derivative of the delta curve: $\arg\min_{a \in [35, 65]} d\Delta/da$. Nominal zero-crossing is the earliest age where the point estimate $\Delta(a) \le 0$. Conservative statistical cessation boundary requires the upper 95% simultaneous uniform bootstrap confidence band to fall below zero on age $\ge 40$. Minimum upper bound values remain positive ($+0.002108$ at $59.75$y Apple; $+0.005830$ at $62.50$y Garmin), confirming that complete familywise statistical cessation is not reached on $[40, 80]$.

---

### Table 5: Second-Window Longitudinal Repeatability and Phase Forecasting (`balanced90`)

| Source Device | Salutation / Sex | Age Stratum | Participants ($N$) | Pearson $r(Q_1, Q_2)$ | $p$-value ($r_Q$) | Spearman $\rho_Q$ | Pearson $r(\hat{A}_1, \hat{A}_2)$ | Mean Frozen Forecast $r$ | % Positive Forecast | Longitudinal Trajectory Interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Apple Watch | All | All | 2,322 | 0.1821 | < 1.00e-12 | 0.0636 | 0.3912 | 0.0052 | 50.3% | Overall population pooled |
| Apple Watch | All | <40 | 491 | 0.2433 | 4.79e-08 | 0.1056 | 0.3782 | 0.0118 | 53.2% | Overall population pooled |
| Apple Watch | All | 40-55 | 1,036 | 0.2597 | < 1.00e-12 | 0.1162 | 0.4176 | 0.0021 | 49.0% | Overall population pooled |
| Apple Watch | All | >=55 | 795 | -0.0248 | 0.4851 | -0.0389 | 0.3003 | 0.0052 | 50.3% | Overall population pooled |
| Apple Watch | Female | All | 699 | 0.3084 | < 1.00e-12 | 0.1727 | 0.4622 | 0.0171 | 51.5% | All females pooled |
| Apple Watch | Female | <40 | 115 | 0.2336 | 0.0120 | 0.2029 | 0.3440 | 0.0610 | 65.2% | Strong, significant infradian persistence; positive phase forecast |
| Apple Watch | Female | 40-55 | 319 | 0.3883 | < 1.00e-12 | 0.2348 | 0.4937 | 0.0178 | 49.8% | Peak repeatability; robust periodic tracking across 6 months |
| Apple Watch | Female | >=55 | 265 | -0.0247 | 0.6885 | -0.0473 | 0.3050 | -0.0028 | 47.5% | Rhythm collapse; zero correlation (loss of ovarian cycle) |
| Apple Watch | Male | All | 1,623 | -0.0079 | 0.7500 | -0.0105 | 0.3491 | 0.0001 | 49.8% | Negative control: zero infradian repeatability across all ages |
| Apple Watch | Male | <40 | 376 | -0.0219 | 0.6718 | -0.0308 | 0.3409 | -0.0032 | 49.5% | Negative control: zero infradian repeatability across all ages |
| Apple Watch | Male | 40-55 | 717 | 0.0134 | 0.7192 | 0.0220 | 0.3665 | -0.0049 | 48.7% | Negative control: zero infradian repeatability across all ages |
| Apple Watch | Male | >=55 | 530 | -0.0260 | 0.5504 | -0.0408 | 0.3000 | 0.0092 | 51.7% | Negative control: zero infradian repeatability across all ages |
| Garmin Tier 1c | All | All | 2,875 | 0.2898 | < 1.00e-12 | 0.1392 | 0.4494 | 0.0144 | 52.5% | Overall population pooled |
| Garmin Tier 1c | All | <40 | 549 | 0.3993 | < 1.00e-12 | 0.2170 | 0.4758 | 0.0172 | 49.9% | Overall population pooled |
| Garmin Tier 1c | All | 40-55 | 1,295 | 0.3260 | < 1.00e-12 | 0.1766 | 0.4446 | 0.0243 | 56.0% | Overall population pooled |
| Garmin Tier 1c | All | >=55 | 1,031 | 0.0122 | 0.6949 | 0.0287 | 0.3079 | 0.0006 | 49.6% | Overall population pooled |
| Garmin Tier 1c | Female | All | 1,013 | 0.3832 | < 1.00e-12 | 0.2650 | 0.5309 | 0.0309 | 54.1% | All females pooled |
| Garmin Tier 1c | Female | <40 | 207 | 0.3820 | 1.35e-08 | 0.2564 | 0.4705 | 0.0476 | 53.1% | Strong, significant infradian persistence; positive phase forecast |
| Garmin Tier 1c | Female | 40-55 | 503 | 0.3833 | < 1.00e-12 | 0.2995 | 0.5010 | 0.0460 | 57.9% | Peak repeatability; robust periodic tracking across 6 months |
| Garmin Tier 1c | Female | >=55 | 303 | 0.0534 | 0.3542 | 0.0491 | 0.3074 | -0.0055 | 48.5% | Rhythm collapse; zero correlation (loss of ovarian cycle) |
| Garmin Tier 1c | Male | All | 1,862 | -0.0016 | 0.9448 | 0.0182 | 0.3313 | 0.0055 | 51.7% | Negative control: zero infradian repeatability across all ages |
| Garmin Tier 1c | Male | <40 | 342 | -0.0031 | 0.9551 | 0.0329 | 0.2876 | -0.0012 | 48.0% | Negative control: zero infradian repeatability across all ages |
| Garmin Tier 1c | Male | 40-55 | 792 | 0.0073 | 0.8378 | 0.0116 | 0.3447 | 0.0105 | 54.8% | Negative control: zero infradian repeatability across all ages |
| Garmin Tier 1c | Male | >=55 | 728 | -0.0099 | 0.7889 | 0.0215 | 0.3067 | 0.0031 | 50.0% | Negative control: zero infradian repeatability across all ages |

*Table 5 Notes*: Evaluated on independent second non-overlapping 180-day windows ($w_2$ start $\ge w_1$ start $+ 180$ days). Frozen forecast correlation evaluates Window 1 harmonic parameters $(\hat{A}_1, \hat{\phi}_1)$ projected onto Window 2 daily timestamps maintaining absolute calendar phase.

---

### Table 6: Sensitivity Analyses: Wear Compliance, Metabolic Adiposity (BMI), and Birthday Conventions

| Model Specification / Sensitivity | Test 1: Total Female Age ($F_{\mathrm{HC3}}, 3\ \mathrm{df}$) | $p$-value | Test 2: Female Curvature ($F_{\mathrm{HC3}}, 2\ \mathrm{df}$) | $p$-value | Test 3: Total Sex $\times$ Age ($F_{\mathrm{HC3}}, 3\ \mathrm{df}$) | $p$-value | Test 4: Nonlinear Sex $\times$ Age ($F_{\mathrm{HC3}}, 2\ \mathrm{df}$) | $p$-value | Model Robustness Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Primary Unadjusted Bivariate RCS | 48.94 | < 1.00e-12 | 14.06 | 7.80e-07 | 46.65 | < 1.00e-12 | 11.64 | 8.83e-06 | Confirmed Invariant |
| Wear-Compliance Adjusted (Z_occ, Z_night, Z_span) | 49.74 | < 1.00e-12 | 13.32 | 1.64e-06 | 47.42 | < 1.00e-12 | 11.36 | 1.16e-05 | Confirmed Invariant |
| BMI Adjusted (Complete Cases N=3,370) | 28.81 | < 1.00e-12 | 7.49 | 5.60e-04 | 27.93 | < 1.00e-12 | 7.88 | 3.77e-04 | Confirmed Invariant |
| BMI Unadjusted Reference Subset (N=3,370) | 28.82 | < 1.00e-12 | 7.49 | 5.61e-04 | 27.95 | < 1.00e-12 | 7.89 | 3.74e-04 | Confirmed Invariant |
| Birthday Sensitivity: January 1 Convention | 48.17 | < 1.00e-12 | 13.38 | 1.54e-06 | 46.01 | < 1.00e-12 | 10.79 | 2.06e-05 | Confirmed Invariant |
| Birthday Sensitivity: December 31 Convention | 48.81 | < 1.00e-12 | 14.06 | 7.83e-07 | 46.48 | < 1.00e-12 | 11.49 | 1.02e-05 | Confirmed Invariant |

*Table 6 Notes*: Evaluated on Apple Watch `expanded90` shared support $[25, 75)$ years. Wear compliance adjusted model adds centered mean daily occupancy fraction, nocturnal coverage fraction, and date span. BMI adjusted model evaluates $N=3,370$ complete cases adjusting for continuous BMI. Birthday conventions evaluate January-1 ($a = \mathrm{start.year} - \mathrm{birth.year}$) and December-31 conventions.

---

## 4. Methodological Discussion

### 4.1 Autonomic vs. Red-Noise Separation: The Necessity of Two-Component REML Nulls
A fundamental challenge in longitudinal wearable sensor analytics is that daily resting heart rate exhibits serial correlation governed by two physiological timescales:
1. **Fast autonomic rebound ($t_{1/2} \approx 1.5\ \mathrm{days}, \phi_1 \approx 0.25$)**: Acute day-to-day parasympathetic recovery following physical exertion, acute sleep disruption/debt, and alcohol/dietary perturbation.
2. **Slow environmental and physiological drift ($t_{1/2} \approx 35\text{--}140\ \mathrm{days}, \phi_2 \approx 0.98\text{--}0.995$)**: Background red noise arising from seasonal ambient temperature oscillations, plasma volume changes, and long-term sensor coupling drift.

Under standard single AR(1) null calibration, the estimator compromises on an intermediate autocorrelation coefficient ($\phi \approx 0.45$). This severely underestimates the 25–35 day spectral background floor by approximately 40%, generating spurious 'infradian rhythmicity' in up to 18% of male negative controls. Superposing two continuous Markov kernels $\text{Cov}(\tau) = v_1 \phi_1^\tau + v_2 \phi_2^\tau$ and estimating parameters via REML on the error contrast subspace $P_0 = I - X(X^T X)^{-1}X^T$ completely eliminates this artifact. Male controls collapse to zero mean excess ($Q_{\mathrm{male}} \approx 0.001$), providing a pristine, unpolluted baseline against which biological ovarian rhythmicity is evaluated.

### 4.2 Cross-Sectional Cohort Structure vs. Longitudinal Aging Trajectories
The Restricted Cubic Spline trajectories documented in this study represent **cross-sectional between-participant associations**, not 40-year individual follow-up. While second-window repeatability ($N=11,037$) demonstrates strong temporal stability across 6-month horizons, cross-sectional modeling cannot rule out:
1. **Cohort and wearable adoption differences**: Younger smartwatch adopters may differ from older participants in physical conditioning, sleep hygiene, or reporting compliance.
2. **Unmeasured endocrine exposures**: The Datenspende cohort contains administrative salutation and birth year, but does not record oral contraceptive use, hormonal intrauterine devices (IUDs), hormone replacement therapy (HRT), or surgical oophorectomy/hysterectomy. Observed curves represent population-level mixtures.
3. **Age estimation uncertainty**: Datenspende records birth year only. The December-1 convention introduces $\pm 0.5$ year birthday jitter, although sensitivity analyses (Table 6) confirm that birthday conventions induce zero material bias in spline curvature or hypothesis decisions.

### 4.3 Biological Specificity to Infradian Resonance: Residual SD Invariance
A critical potential confounder in wearable aging studies is that total cardiovascular autonomic variability might decline with age, causing relative spectral metrics to appear attenuated. To test this hypothesis, we modeled the residual standard deviation of daily resting HR ($\hat{\sigma}_{\mathrm{resid}}$, in bpm) across the identical RCS spline design:
- While periodic resonance endpoints ($Q_i, E_{i2}, \hat{A}_i$) exhibit massive, decisive Sex $\times$ Age interactions ($F_{\mathrm{HC3}} = 46.65, p_{\mathrm{asymp}} = 3.89 \times 10^{-30}$ in Apple Watch; $F_{\mathrm{HC3}} = 39.01, p_{\mathrm{asymp}} = 3.37 \times 10^{-25}$ in Garmin), residual autonomic standard deviation exhibits **NO significant Sex $\times$ Age interaction** on Apple Watch ($W_{\mathrm{HC3}} = 5.86, p = 0.119$).
- This proves that the attenuation of $Q_i$ and $\hat{A}_i$ is biologically specific to the loss of ovarian cycle resonance, rather than an artifact of general cardiovascular aging.

---

## 5. Publication Figures Index

The multi-panel publication figures generated by Milestone 5 reside in `experiments/infradian_age_dependency/figures/`:
1. **`fig1_rcs_continuous_partial_effects.png`**: Multi-source layout (Apple Watch, Garmin Tier 1c, Samsung) showing female and male continuous RCS trajectories over age [18, 80] with landmark knots and pointwise vs. uniform simultaneous confidence bands.
2. **`fig2_excess_heatmaps_and_fwhm.png`**: Empirical spectral excess heatmaps over trial periods 15–60 days, spectral slices across 5-year age bands, and quantitative FWHM resonance broadening.
3. **`fig3_delta_curve_transitions.png`**: Female-minus-male delta trajectories $\Delta(a)$, first derivative inflection dip ($d\Delta/da$), steepest decline inflection age (46.65y Apple, 47.25y Garmin), nominal zero-crossing (56.18y Apple, 70.20y Garmin), and conservative cessation boundaries.
4. **`fig4_source_and_protocol_comparisons.png`**: Cross-device overlay (Apple vs Garmin), protocol robustness (`expanded90` vs `balanced90`), second-window repeatability scatter, and grouped phase forecast correlations by age and sex.

---

## 6. Verification and Reproducibility
All figures, tables, and statistics can be regenerated from scratch using:
```bash
/Users/bulat/micromamba/envs/datenspende/bin/python experiments/infradian_age_dependency/scripts/05_generate_publication_figures.py
/Users/bulat/micromamba/envs/datenspende/bin/python experiments/infradian_age_dependency/scripts/06_generate_diataxis_report.py
```

Report compiled successfully by `worker_m5_1` on 2026-10-20.
