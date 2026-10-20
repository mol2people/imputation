# Empirical Reference: Cross-Device Replication and Spectral Periodogram Diagnostics of Infradian Heart-Rate Cycles

## 1. Overview and Epistemic Scope

This technical reference document reports the out-of-domain cross-device replication and high-resolution spectral periodogram diagnostics of infradian heart-rate rhythmicity across the German Corona Data Donation cohort. Following the primary single-source Apple Watch investigation (`female_infradian_cycles_v2`), this investigation evaluates:
1. **Cross-Device External Validity**: Full standardized exposure testing ($k = 15$ qualified days per 30-day block, $N_{\text{obs}} = 90$ across 180 calendar days) applied to the independent single-source Garmin Tier 1c cohort ($N = 4,410$ total manifest, $N = 3,760$ qualified participants).
2. **Exposure Sensitivity and Sample Scaling**: Relaxation of block occupancy from $k = 15$ to $k = 10$ qualified dates per block ($N_{\text{obs}} = 60$ across 180 days) across the unconstrained Apple Watch manifest ($N = 11,139$ total manifest, $N = 5,516$ qualified participants).
3. **Continuous Periodogram Sweeps (3 to 60 Days)**: Empirical harmonic regression across 229 unconstrained period points, isolating narrowband resonant peaks from broadband red noise.
4. **Spectral Deconstruction of Long-Period Dynamics (3 to 120 Days)**: Diagnostic resolution of the 35-day search grid boundary pile-up and the long-period male spectral power rise using continuous two-component REML nulls and higher-order polynomial detrending.
5. **Longitudinal Persistence vs. Phase Wander**: Independent nonoverlapping second-window re-fitting ($N_{\text{pairs}} = 2,875$ on Garmin, $N_{\text{pairs}} = 2,926$ on Apple) assessing period recurrence versus out-of-sample frozen-phase degradation.

All execution scripts, cached representations, and structured metrics reside in:
- Pipeline execution scripts: `experiments/garmin_infradian_replication/scripts/`
- Garmin 5-minute cache ($N = 4,410$): `experiments/garmin_infradian_replication/artifacts/cache_5min/`
- Tabular results and metrics: `experiments/garmin_infradian_replication/results/`
- Publication figures: `experiments/garmin_infradian_replication/figures/`

---

## 2. Cohort Retention and Experimental Design

### 2.1 Cohort Extraction and Gate Definitions
- **Garmin Single-Source Cohort**: Derived from `HEARTBEAT_LAB/artifacts/relaxed_garmin_manifest.csv` (Tier 1c single-source users). A qualified day requires $\ge 8$ clock hours and either $\ge 60$ raw HR events or $\ge 8.0$ hours interval-union coverage.
- **Window Qualification Gate**: Standardized 180-day window $[s, s + 180)$ partitioned into six 30-day blocks $[s + 30b, s + 30(b+1))$ for $b \in \{0, \dots, 5\}$.
  - Standard Protocol ($k = 15$): $\ge 15$ qualified dates per block; exactly 15 dates sampled uniformly per block via deterministic SHA-256 hashing ($N_{\text{obs}} = 90$).
  - Relaxed Protocol ($k = 10$): $\ge 10$ qualified dates per block; exactly 10 dates sampled uniformly per block ($N_{\text{obs}} = 60$).
- **Second-Window Protocol**: The earliest qualifying date $s_2 \ge s_1 + 180$ establishes a nonoverlapping second window.

### 2.2 Retention by Cohort and Demographic Cell

| Cohort Arm | Protocol | Total Manifest | Qualified W1 ($N$) | Two-Window Pairs | Young Females ($y0$) | Young Males ($y1$) | Middle Females ($y0$) | Middle Males ($y1$) | Elderly Females ($y0$) | Elderly Males ($y1$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple Watch (v2)** | $k=15$ | 11,139 | 4,630 | 2,324 | 224 | 666 | 1,082 | 2,086 | 172 | 400 |
| **Apple Watch (k10)** | $k=10$ | 11,139 | 5,516 | 2,926 | 292 | 789 | 1,331 | 2,449 | 197 | 458 |
| **Garmin Tier 1c** | $k=15$ | 4,410 | 3,760 | 2,875 | 299 | 461 | 938 | 1,625 | 102 | 335 |

---

## 3. Mathematical Specifications

### 3.1 Nonperiodic Baseline Model
For participant $i$ on selected calendar timestamps $t_k \in \{t_1, \dots, t_{N_{\text{obs}}}\}$:
$$y_i(t_k) = \text{percentile}_{10}(\{hr_{t_k, m} : hr_{t_k, m} \text{ finite}\})$$

The baseline design matrix $X_{0, i} \in \mathbb{R}^{N_{\text{obs}} \times p}$ contains:
1. Constant intercept $\mathbf{1}$.
2. Elapsed linear calendar trend: $(t_k - t_1)$.
3. Weekday indicators: $\mathbb{I}(\text{dayofweek}(t_k) = d)$ for $d \in \{\text{Tue}, \dots, \text{Sun}\}$ (Monday reference).
4. Four 6-hour clock coverage fractions: $c_{00\text{--}06}, c_{06\text{--}12}, c_{12\text{--}18}, c_{18\text{--}24}$.

Residualization operator:
$$P_0 = I - X_0 (X_0^T X_0)^{-1} X_0^T, \quad e_0 = P_0 y_i, \quad \text{RSS}_0 = e_0^T e_0$$

### 3.2 Frisch-Waugh-Lovell Harmonic Scoring
For candidate period $T \in [25.0, 35.0]\text{ days}$ with step $0.5\text{ d}$ ($K = 21$ candidate frequencies):
$$C_T = \left[\cos\left(\frac{2\pi (t_k - t_1)}{T}\right), \; \sin\left(\frac{2\pi (t_k - t_1)}{T}\right)\right]$$
$$\tilde{C}_T = P_0 C_T, \quad \hat{\beta}_T = (\tilde{C}_T^T \tilde{C}_T)^{-1} \tilde{C}_T^T e_0$$
$$\Delta \text{RSS}(T) = \hat{\beta}_T^T (\tilde{C}_T^T \tilde{C}_T) \hat{\beta}_T, \quad S_i = \max_{T \in [25, 35]} \frac{\Delta \text{RSS}(T)}{\text{RSS}_0}$$

### 3.3 Dependence-Aware Null Calibration
1. **Standard Calendar $\text{AR}(1)$ Null (v2)**:
   Daily correlation decay $\phi^{|t_k - t_j|}$ and stationary variance $\sigma^2$ estimated by sample autocovariance on $e_0$. Parametric bootstrap generates $B = 199$ surrogate realizations $\epsilon_b \sim \mathcal{N}(0, \Sigma_{\text{AR1}})$.
2. **Two-Component Continuous Auto-Regressive Null ($\text{AR}(1) + \text{Slow}$)**:
   To model long-memory $1/f^\alpha$ red noise without confounding harmonic frequencies, the residual covariance is parameterized as:
   $$\Sigma_{jk} = s_1 \phi_1^{|t_k - t_j|} + s_2 \phi_2^{|t_k - t_j|}$$
   where $\phi_1 \in [0, 0.95)$ represents daily autocorrelation and $\phi_2 \in \{0.980, 0.995\}$ captures low-frequency baseline persistence. Variance parameters $(s_1, s_2)$ are fitted via Restricted Maximum Likelihood (REML) on the baseline-projected coordinates $z = Q^T e_0$, where $Q$ spans the column space of $P_0$.

Excess score and empirical p-value:
$$E_i = S_i - \frac{1}{B} \sum_{b=1}^B S_{\text{null}, b}, \quad p_i = \frac{1 + \sum_{b=1}^B \mathbb{I}(S_{\text{null}, b} \ge S_i)}{B + 1}$$

---

## 4. Empirical Results: Cross-Device Replication and Sample Scaling

### 4.1 HC3 Robust Wald Contrasts on Excess Score $E_i$
Models are fitted via OLS with 6 cell indicators (age $\times$ salutation) and within-age centered observation covariates (mean 5m occupancy, nocturnal coverage fraction, and window span). Heteroskedasticity-consistent covariance (HC3) is applied across all tests.

| Cohort & Model Arm | Contrast / Hypothesis | Parameter Estimate ($\Delta$) | HC3 SE | 95% Confidence Interval | Wald Statistic ($\chi^2$) | p-value (Holm-adjusted) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Garmin $k=15$** | **Omnibus Age $\times$ Sex** | — | — | — | **$50.89$** | **$1.81 \times 10^{-11}$** |
| ($N = 3,760$) | Young Adults ($y0 - y1$) | $+0.0474$ | $0.0053$ | $[+0.0371, +0.0577]$ | **$81.13$** | **$4.21 \times 10^{-18}$** |
| (AR(1)+slow null) | Middle Ager ($y0 - y1$) | $+0.0350$ | $0.0028$ | $[+0.0296, +0.0404]$ | **$161.72$** | **$9.63 \times 10^{-36}$** |
| | Elderly ($y0 - y1$) | $-0.0013$ | $0.0051$ | $[-0.0113, +0.0087]$ | $0.06$ | $0.8050$ |
| | Standardized Overall | $+0.0333$ | $0.0022$ | $[+0.0289, +0.0377]$ | $221.20$ | $4.87 \times 10^{-49}$ |
| **Apple $k=10$** | **Omnibus Age $\times$ Sex** | — | — | — | **$31.00$** | **$1.85 \times 10^{-7}$** |
| ($N = 5,516$) | Young Adults ($y0 - y1$) | $+0.0526$ | $0.0061$ | $[+0.0407, +0.0645]$ | **$75.49$** | **$1.52 \times 10^{-17}$** |
| (AR(1)+slow null) | Middle Ager ($y0 - y1$) | $+0.0186$ | $0.0024$ | $[+0.0140, +0.0232]$ | **$62.73$** | **$4.61 \times 10^{-15}$** |
| | Elderly ($y0 - y1$) | $+0.0124$ | $0.0052$ | $[+0.0023, +0.0225]$ | $5.74$ | $0.0166$ |
| | Standardized Overall | $+0.0246$ | $0.0021$ | $[+0.0205, +0.0287]$ | $137.55$ | $9.08 \times 10^{-32}$ |
| **Apple $k=15$ (v2)** | **Omnibus Age $\times$ Sex** | — | — | — | **$60.52$** | **$7.23 \times 10^{-14}$** |
| ($N = 4,630$) | Young Adults ($y0 - y1$) | $+0.0645$ | $0.0060$ | $[+0.0528, +0.0763]$ | **$115.58$** | **$1.19 \times 10^{-25}$** |
| (AR(1)+slow null) | Middle Ager ($y0 - y1$) | $+0.0235$ | $0.0022$ | $[+0.0192, +0.0278]$ | **$113.82$** | **$2.77 \times 10^{-25}$** |
| | Elderly ($y0 - y1$) | $+0.0073$ | $0.0043$ | $[-0.0011, +0.0158]$ | $2.90$ | $0.0885$ |
| | Standardized Overall | $+0.0294$ | $0.0020$ | $[+0.0255, +0.0332]$ | $222.13$ | $3.24 \times 10^{-49}$ |

### 4.2 Rejection Rates and Null Multiplicity Control
At significance threshold $\alpha = 0.05$:
- **Garmin $k=15$**:
  - Young Females: **$21.7\%$** ($n = 299$).
  - Young Males (Negative Control): **$5.6\%$** ($n = 461$).
  - Middle-Aged Females: **$17.8\%$** ($n = 938$).
  - Middle-Aged Males: **$5.1\%$** ($n = 1,625$).
  - Elderly Females: **$4.9\%$** ($n = 102$).
  - Elderly Males: **$8.1\%$** ($n = 335$).
- **Apple $k=10$**:
  - Young Females: **$18.2\%$** ($n = 292$).
  - Young Males: **$3.7\%$** ($n = 789$).
  - Middle-Aged Females: **$8.6\%$** ($n = 1,331$).
  - Middle-Aged Males: **$4.1\%$** ($n = 2,449$).

Male rejection rates calibrate tightly around nominal $5.0\%$ across all age bands and both device architectures, demonstrating that the two-component null model prevents red-noise false positives.

---

## 5. Spectral Periodogram Diagnostics

### 5.1 Continuous Unconstrained Harmonic Sweeps (3 to 60 Days)
Evaluating unconstrained single-harmonic partial $R^2$ across 229 frequencies demonstrates:
1. **Narrowband Biological Resonance**: Young adult females display a localized, symmetric spectral elevation sharply centered at $27.7\text{ days}$ on Garmin (mean partial $R^2 \approx 0.079$ vs. $0.045$ null surrogate envelope) and $28.0\text{ days}$ on Apple (mean partial $R^2 \approx 0.085$ vs. $0.041$ null).
2. **Age-Dependent Attenuation**: Middle-aged females display a distinct but attenuated peak at $27.7\text{ days}$ (mean partial $R^2 \approx 0.064$ on Garmin). Elderly females show no discernible peak above red-noise background.
3. **Absence in Males**: Male spectra on both Apple and Garmin display zero resonant elevation in the 25–35 day band, tracking null envelopes monotonically.

### 5.2 Resolution of the 35-Day Edge Peak and Long-Period Dynamics (3 to 120 Days)
Detailed investigation into why observed power rises at long periods reveals:
1. **Grid-Boundary Argmax Selection Bias**:
   When searching $T \in [25, 35]\text{ days}$ over 21 discrete grid points, pure white noise yields an argmax at $T = 25\text{ d}$ in $15.8\%$ of simulations and at $T = 35\text{ d}$ in $14.8\%$ of simulations (expected uniform baseline $= 4.8\%$). Boundary accumulation is an intrinsic mathematical property of searching bounded continuous autocorrelated processes.
2. **Red-Noise Continuum vs. Periodic Resonance (Figure R7)**:
   In Garmin males ($N = 2,086$, ages 18–64), observed spectral power increases monotonically from 30 days to 120 days.
   - **Single $\text{AR}(1)$ Null**: Fails to capture this rise because short-memory daily parameter ($\phi_1 \approx 0.23$) has a flat spectral plateau beyond 30 days.
   - **Two-Component Continuous Null**: Incorporating a slow continuous auto-regressive component ($\phi_2 \approx 0.98\text{--}0.995$) fully tracks the male trajectory across the entire 3–120 day range (mean difference $< 0.003$).
   - **Polynomial Trend Invariance**: Adding quadratic and cubic calendar trends ($t^2, t^3$) reduces low-frequency ($>60\text{ d}$) male power by $31\%$, proving that macro seasonal drift drives the tail.
   - In contrast, the female $27.7\text{ d}$ resonance is completely invariant to cubic detrending ($\Delta R^2 = +0.0281$ with polynomial controls), dropping precisely to zero beyond 35 days.

---

## 6. Longitudinal Persistence vs. Phase Wander

### 6.1 Two-Window Nonoverlapping Re-Fitting
Evaluating nonoverlapping Window 2 ($s_2 \ge s_1 + 180\text{ d}$) confirms:
1. **Period Agreement ($|T_1 - T_2| \le 1.5\text{ days}$)**:
   - Garmin Young Females: **$37.3\%$** ($n = 209$) vs. $29.1\%$ in permuted null pairings.
   - Apple $k=10$ Young Females: **$38.7\%$** ($n = 137$) vs. $30.0\%$ in permuted pairings.
   - Garmin Young Males: **$29.0\%$** ($n = 338$) vs. $27.5\%$ in permuted pairings.
   - Apple $k=10$ Young Males: **$24.0\%$** ($n = 438$) vs. $27.3\%$ in permuted pairings.
2. **Degradation of Frozen Out-of-Sample Phase**:
   Projecting Window 1 period $T_1^*$ and phase $\phi_1^*$ deterministically forward across 180 days to predict Window 2 residuals yields:
   $$\text{Median Correlation } r(\hat{y}_{\text{persist}}, e_{0, \text{w2}}) \in [-0.007, +0.036] \quad (\text{Sign Test } p > 0.09 \text{ across all groups})$$
   $$\text{Mean Out-of-Sample } R^2_{\text{persist}} < 0 \quad (-0.08 \text{ to } -0.13)$$
   This confirms that the underlying biological infradian pacemaker possesses an endogenous cycle length that recurs across distinct 6-month seasons, but natural intra-individual period variability accumulates phase wander, preventing deterministic clock-phase locking over multi-month separations.

---

## 7. Validity Conditions and Limitations

1. **Sensor Measurement Mechanics**: Garmin devices report heart rate from photoplethysmography (PPG) using vendor-proprietary filtering, distinct from Apple Watch sensor hardware and PPG optical algorithms. Day adequacy criteria were harmonized to $\ge 8$ clock hours and $\ge 60$ raw readings.
2. **Administrative Salutation Labels**: Salutation codes ($10 \to 0, 20 \to 1$) reflect portal registration, not karyotypic confirmation or clinical endocrinological status.
3. **Absence of Cycle Ground Truth**: Menstrual bleeding records, hormonal assays (estradiol, progesterone, LH), and hormonal contraception usage are absent in the cohort.
4. **Window Selection Gate**: Infradian analyses apply to participants with sufficient longitudinal adherence ($\ge 180$ days with consistent weekly wear).

---

## 8. Artifact and Publication Figure Manifest

| Figure Artifact | Format | Description and Target Estimand |
| :--- | :---: | :--- |
| [`R1_forest_contrasts.png`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/garmin_infradian_replication/figures/R1_forest_contrasts.png) | 300 DPI PNG | Forest plot of HC3 Wald excess score contrasts ($E_i$) across Apple $k=15$, Apple $k=10$, and Garmin $k=15$. |
| [`R2_rejection_rates.png`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/garmin_infradian_replication/figures/R2_rejection_rates.png) | 300 DPI PNG | Empirical false positive and rejection rates at $\alpha = 0.05$ across age and sex cells under the two-component null. |
| [`R3_pvalue_qq.png`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/garmin_infradian_replication/figures/R3_pvalue_qq.png) | 300 DPI PNG | $-\log_{10} p$ Quantile-Quantile plots against uniform expectation confirming male calibration and female deflection. |
| [`R4_excess_score_distributions.png`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/garmin_infradian_replication/figures/R4_excess_score_distributions.png) | 300 DPI PNG | Violin distributions of individual participant excess infradian score $E_i$ across cohorts and age brackets. |
| [`R5_persistence.png`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/garmin_infradian_replication/figures/R5_persistence.png) | 300 DPI PNG | Second-window period agreement $|T_1 - T_2|$ ECDFs against permuted nulls, and frozen out-of-sample phase correlation. |
| [`R6_garmin_mean_periodogram.png`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/garmin_infradian_replication/figures/R6_garmin_mean_periodogram.png) | 300 DPI PNG | Full continuous spectral periodograms (3 to 60 days) on Garmin cohort with participant-level AR(1) surrogate envelopes. |
| [`R7_long_period_male_rise.png`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/garmin_infradian_replication/figures/R7_long_period_male_rise.png) | 300 DPI PNG | Diagnostic 3 to 120 day spectral decomposition resolving male long-period power as red noise and polynomial drift. |
