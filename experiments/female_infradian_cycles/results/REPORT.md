# Female Infradian Rhythm Dynamics on Long-Term Apple Watch Data: Technical Reference Report

**Reconciled 2026-10-20 from saved outputs.** No extraction, spectral detection or group-test script was rerun. Descriptive counts, means, SDs and proportions below were checked against the participant CSV; confidence intervals and adjusted coefficients are quoted from the existing contrast CSV, not re-estimated. Its p-value fields contain inconsistencies, described in §7.

**Cohort**: Apple Watch longitudinal data donation cohort ($N = 11,139$ total; $N = 3,855$ female, $N = 7,284$ male).  
**Screened Cohort**: Participants with $\ge 60$ adequate calendar days (at least 120 occupied 5-minute bins/day) with full cached last-minus-first calendar span $\ge 60$ days ($N = 8,425$; $75.64\%$ retention).  
**Artifact Dependencies**:
- Demographics: `experiments/apple_cohort_scaling/artifacts/apple_full_manifest.csv`
- Cache files: `experiments/apple_cohort_scaling/artifacts/cache_5min/customer_{cid}.npz`
- Output tables:
  - Participant-level metrics: `experiments/female_infradian_cycles/results/infradian_metrics_by_participant.csv`
  - Group contrast statistics: `experiments/female_infradian_cycles/results/group_contrasts_age_and_sex.csv`
- Verification suite: `experiments/female_infradian_cycles/scripts/00_verify_infradian_methods.py`

---

## 1. Formal Architecture and Hypotheses

### 1.1 Biological Hypothesis and Comparison Groups

The exploratory hypothesis is that selected 25–35-day HR variation is greater in younger female-coded participants than in comparison groups. The observed variables are registration salutation (`10 → y=0`, `20 → y=1`), broad `age_group` and `bmi_group`. “Female” and “male” below retain the saved-table labels; they do not establish reproductive anatomy, ovarian function or gender identity. `Young Adults`, `Middle Ager` and `Elderly` are recorded categories, not individual measured ages. The outputs contain no exact ages or verified menopausal status.

The target is female-coded `Young Adults`; comparison groups are male-coded `Young Adults` and female-coded `Elderly`. These provide age-category and label comparisons, not individually age-matched or clinically verified negative controls. The original reproductive/menopausal labels are therefore withdrawn. Candidate mean-amplitude contrasts are target minus each comparison group. Wear-derived signals are descriptive comparators; equality or inequality of their independently optimized amplitudes cannot by itself exclude recording artifacts.

---

### 1.2 Daily Signal Definitions & Preprocessing

For each participant $u$ on each calendar day $d$, 5-minute heart rate bins $x_{d, b} \in [25, 230]\text{ bpm}$ ($b \in \{0, \dots, 287\}$) are screened for at least 120 occupied bins. This is 10 hours of bin positions, not verified continuous wear duration:

$$\text{Valid}(d) = \mathbb{I}\left(\sum_{b=0}^{287} \mathbb{I}(x_{d, b} \text{ is finite}) \ge 120\right)$$

Time coordinate $t_d = \text{date}_d - \text{date}_0 \in \mathbb{Z}_{\ge 0}$ tracks elapsed calendar days since initial observation, preserving true intervals across missing wear days. Seven daily HR and recording time series are extracted. `hr_p10` is the 10th percentile of observed bin means, not a validated resting HR measurement. The clock windows do not establish sleep, recumbency or activity state:

1. **Low observed HR quantile (`hr_p10`, bpm; resting-HR proxy)**:
   $$y_{\text{hr\_p10}}(d) = \text{Quantile}_{0.10}\left(\{x_{d, b} : x_{d, b} \text{ is finite}\}\right)$$
2. **Nocturnal Heart Rate (`night_hr`, bpm)**:
   Mean observed heart rate during clock window 00:00–06:00 ($b \in [0, 71]$), requiring $\ge 12$ occupied bins (one hour of bin positions), else NaN:
   $$y_{\text{night\_hr}}(d) = \frac{1}{|\mathcal{B}_{\text{night}}|} \sum_{b \in \mathcal{B}_{\text{night}}} x_{d, b}, \quad \text{where } \mathcal{B}_{\text{night}} = \{b \in [0, 71] : x_{d, b} \text{ is finite}\}$$
3. **Daytime Heart Rate (`day_hr`, bpm)**:
   Mean heart rate during daytime window 09:00–18:00 ($b \in [108, 215]$), requiring $\ge 36$ occupied bins (three hours of bin positions), else NaN:
   $$y_{\text{day\_hr}}(d) = \frac{1}{|\mathcal{B}_{\text{day}}|} \sum_{b \in \mathcal{B}_{\text{day}}} x_{d, b}, \quad \text{where } \mathcal{B}_{\text{day}} = \{b \in [108, 215] : x_{d, b} \text{ is finite}\}$$
4. **Nocturnal Dipping Ratio (`dipping_ratio`, dimensionless)**:
   $$y_{\text{dipping}}(d) = 1.0 - \frac{y_{\text{night\_hr}}(d)}{y_{\text{day\_hr}}(d)} \quad (\text{defined when } y_{\text{day\_hr}} > 0)$$
5. **Daily Wear Occupancy Rate (`occ_daily`, fraction in $[0, 1]$)**:
   $$y_{\text{occ}}(d) = \frac{1}{288} \sum_{b=0}^{287} \mathbb{I}(x_{d, b} \text{ is finite})$$
6. **Nocturnal Wear Compliance (`nocturnal_compliance`, fraction in $[0, 1]$)**:
   $$y_{\text{noct\_comp}}(d) = \frac{1}{72} \sum_{b=0}^{71} \mathbb{I}(x_{d, b} \text{ is finite})$$
7. **Daily Wear-to-Gap Transitions (`gap_episodes`, count)**:
   $$y_{\text{gap}}(d) = \sum_{b=0}^{286} \mathbb{I}\left(x_{d, b} \text{ is finite} \land x_{d, b+1} \text{ is NaN}\right)$$

---

### 1.3 Matched Infradian Detection Suite

Periodicity is evaluated across the candidate grid $T \in [25.0, 35.0]\text{ days}$ ($\Delta T = 0.5\text{ days}$, $K=21$ frequencies) using two complementary formulations:

#### 1. Harmonic Cosinor Regression
For each candidate period $T_k$, a single-harmonic linear regression with linear drift is solved via OLS:

$$y(t_d) = \beta_0 + \beta_{\text{trend}} t_d + \beta_c \cos\left(\frac{2\pi t_d}{T_k}\right) + \beta_s \sin\left(\frac{2\pi t_d}{T_k}\right) + \epsilon(t_d)$$

Design matrix $X_k = [\mathbf{1}, \mathbf{t}, \cos(2\pi\mathbf{t}/T_k), \sin(2\pi\mathbf{t}/T_k)] \in \mathbb{R}^{N \times 4}$. The coefficients are estimated numerically by least squares. For a full-rank design they satisfy the normal equations $(X_k^T X_k)\hat{\beta}_k = X_k^T \mathbf{y}$; the corresponding OLS solution is unique. The saved summaries do not include conditioning diagnostics:
- **Physical Amplitude**: $A(T_k) = \sqrt{\hat{\beta}_{c, k}^2 + \hat{\beta}_{s, k}^2}$
- **Acrophase**: $\phi(T_k) = \text{atan2}(\hat{\beta}_{s, k}, \hat{\beta}_{c, k}) \in (-\pi, \pi]$
- **Incremental Variance Explained ($\Delta R^2$)**:
  $$\Delta R^2(T_k) = \frac{SS_{\text{res, null}} - SS_{\text{res, full}}(T_k)}{SS_{\text{tot}}}$$
  where null model is linear trend $y = \beta_0 + \beta_{\text{trend}} t$.
- **Best-Fit Period**: $T^*_{\text{cos}} = \arg\max_{T_k \in [25, 35]} \Delta R^2(T_k)$.
- **Extra-Sum-of-Squares $F$-Test**:
  $$F = \frac{(SS_{\text{res, null}} - SS_{\text{res, full}}(T^*_{\text{cos}})) / 2}{SS_{\text{res, full}}(T^*_{\text{cos}}) / (N - 4)}, \quad p = \left(1 + \frac{2}{N-4} F\right)^{-(N-4)/2}$$

#### 2. Detrended Lomb-Scargle Periodogram
Linear trend $\hat{\beta}_0 + \hat{\beta}_{\text{trend}} t$ is subtracted. Townsend (2010) normalized power $P_{\text{LS}}(\omega) \in [0, 1]$ is evaluated over angular frequencies $\omega_k = 2\pi / T_k$:
- **Peak Period**: $T^*_{\text{LS}} = \arg\max_{T_k \in [25, 35]} P_{\text{LS}}(2\pi / T_k)$.
- **Signal-to-Noise Ratio (SNR)**:
  $$\text{SNR} = \frac{P^*_{\text{LS}}}{\text{Median}\left(\{P_{\text{LS}}(2\pi / T_f) : T_f \in [14, 22] \cup [38, 56]\text{ days}\}\right)}$$
  The flanking-band ratio is a descriptive contrast. It is not a calibrated red-noise false-alarm probability and does not guarantee freedom from spectral leakage.

---

The nominal F tail is evaluated at the period selected by maximizing fit across 21 candidates. Its fixed-period reference distribution does not account for this search. No autocorrelation-preserving null calibration or multiplicity adjustment is saved. Thus the stored p-values and threshold proportions below are nominal summaries, not validated rhythm-detection rates. Amplitudes and fit strengths are in-sample maxima within a restricted band; a selected period near 29 days is not independent evidence of a menstrual origin.

## 2. Demographic Breakdown and Screening Statistics

Screening requires at least 60 eligible dates after the occupied-bin filter and `timespan_days >= 60`. The saved span is last minus first date in the original cache, without adding one, before this filter; it is not necessarily the span of the signal-specific observations. Date spacing remains irregular. Individual signals additionally need at least 10 finite observations and variance greater than 10⁻¹² to be fitted. Thus the participant screen does not guarantee 60 usable night-HR or dipping dates. The table is recomputed descriptively from the saved participant rows; no signal was re-extracted.

| Saved label / age group | All N | Screened N | Retention | Mean eligible dates | Mean span, days |
|---|---:|---:|---:|---:|---:|
| Female / Young Adults | 697 | 502 | 72.02% | 222.7 | 482.7 |
| Male / Young Adults | 1,619 | 1,209 | 74.68% | 247.5 | 497.1 |
| Female / Middle Ager | 2,774 | 2,088 | 75.27% | 237.2 | 500.3 |
| Male / Middle Ager | 4,696 | 3,634 | 77.39% | 254.8 | 499.4 |
| Female / Elderly | 384 | 289 | 75.26% | 259.5 | 489.9 |
| Male / Elderly | 969 | 703 | 72.55% | 253.1 | 497.3 |
| Female / all age groups | 3,855 | 2,879 | 74.68% | 236.9 | 496.2 |
| Male / all age groups | 7,284 | 5,546 | 76.14% | 253.0 | 498.6 |
| All / all age groups | 11,139 | 8,425 | 75.64% | 247.5 | 497.8 |

There are 11,139 unique participant IDs and 8,425 screened participants. One screened elderly male has no finite `hr_p10` fit, leaving 8,424 available resting-proxy amplitudes. Endpoint-specific denominators differ, particularly for night HR and dipping. The previous middle-aged and elderly male registered counts and several screening means were incorrect.

## 3. Nominal 25–35-Day F-Test Threshold Proportions

These are counts of stored `cos_pvalue_hr_p10` values, with finite-fit denominators. They do not establish menstrual-cycle prevalence, because the period search and correlated errors are not calibrated.

| Saved label / age group | Available p10 fits | Nominal p < 0.05 | Nominal p < 0.01 |
|---|---:|---:|---:|
| Female / Young Adults | 502 | 420/502 (83.67%) | 329/502 (65.54%) |
| Male / Young Adults | 1,209 | 836/1209 (69.15%) | 499/1209 (41.27%) |
| Female / Elderly | 289 | 235/289 (81.31%) | 147/289 (50.87%) |
| Female / Middle Ager | 2,088 | 1677/2088 (80.32%) | 1200/2088 (57.47%) |
| Male / Middle Ager | 3,634 | 2598/3634 (71.49%) | 1541/3634 (42.41%) |
| Male / Elderly | 702 | 549/702 (78.21%) | 354/702 (50.43%) |

The previous report's p < 0.01 percentages were inconsistent with both CSVs. High nominal positivity also occurs in the comparison groups; it cannot support biological specificity by itself.

## 4. Physiological HR Dynamics and Amplitude Contrasts

### 4.1 Reproduced descriptive means

Each cell is mean ± sample SD of the saved participant fit, with the finite-fit N in parentheses. Fractions are dimensionless. These are amplitudes (half the fitted peak-to-trough sinusoidal range), not peak-to-trough swings.

| Signal | Female / Young Adults | Male / Young Adults | Female / Elderly |
|---|---:|---:|---:|
| hr_p10 (bpm) | 1.6402 ± 0.7152 (N=502) | 1.3704 ± 0.7320 (N=1209) | 1.0530 ± 0.6246 (N=289) |
| night_hr (bpm) | 3.3650 ± 2.9025 (N=408) | 3.4645 ± 3.7846 (N=1074) | 1.8974 ± 2.1818 (N=239) |
| day_hr (bpm) | 2.0459 ± 0.8950 (N=502) | 2.0998 ± 0.9288 (N=1209) | 1.5518 ± 0.8208 (N=289) |
| dipping_ratio (fraction) | 0.0428 ± 0.0404 (N=408) | 0.0480 ± 0.0491 (N=1072) | 0.0269 ± 0.0265 (N=238) |

For `hr_p10`, mean incremental R² is 0.08403 / 0.04617 / 0.05473 and mean Lomb–Scargle peak power is 0.09022 / 0.04877 / 0.05723 in the same group order. Both were checked against stored participant values. This is stronger relative selected-band variation among the younger female-coded group, with broad overlap and no demonstrated individual diagnostic validity.

### 4.2 Stored group contrasts

The following mean differences, intervals and standardized effects are quoted from `group_contrasts_age_and_sex.csv`. Means agree with the participant CSV; inferential intervals were not recalculated or validated in this reconciliation. The summarizer constructs approximate Welch intervals, not bootstrap or autocorrelation-aware intervals. Individual amplitudes are already selected over the period grid.

| Signal / comparison | Group Ns | Mean difference | Stored 95% interval | Stored Cohen's d |
|---|---:|---:|---:|---:|
| hr_p10 / Target_vs_Control1 | 502 / 1209 | +0.26978 | [+0.19474, +0.34482] | +0.3710 |
| hr_p10 / Target_vs_Control2 | 502 / 289 | +0.58723 | [+0.49166, +0.68279] | +0.8591 |
| hr_p10 / MiddleAger_F_vs_M | 2088 / 3634 | +0.08591 | [+0.05403, +0.11779] | +0.1431 |
| hr_p10 / Overall_F_vs_M | 2879 / 5545 | +0.10558 | [+0.07694, +0.13423] | +0.1650 |
| night_hr / Target_vs_Control1 | 408 / 1074 | -0.09958 | [-0.46136, +0.26220] | -0.0279 |
| night_hr / Target_vs_Control2 | 408 / 239 | +1.46752 | [+1.07197, +1.86307] | +0.5518 |
| day_hr / Target_vs_Control1 | 502 / 1209 | -0.05387 | [-0.14817, +0.04044] | -0.0586 |
| day_hr / Target_vs_Control2 | 502 / 289 | +0.49413 | [+0.37108, +0.61719] | +0.5688 |
| dipping_ratio / Target_vs_Control1 | 408 / 1072 | -0.00520 | [-0.01011, -0.00029] | -0.1110 |

`Target_vs_Control1` means younger female-coded minus younger male-coded; `Target_vs_Control2` means younger female-coded minus elderly female-coded. The latter is an age-group comparison, not a measured menopause effect.

### 4.3 Additive age/BMI adjustment

The saved p10-amplitude ANCOVA coefficient for `y=1` is −0.11811 bpm, SE 0.01454, t=−8.125, stored p=4.4409×10⁻¹⁶, model R²=0.0426. No coefficient confidence interval is saved. These are quoted model outputs, not independently refitted estimates. The implemented OLS includes an intercept, male-coded label, `Middle Ager`/`Elderly` indicators and `overweight`/`obese`/`underweight` indicators. Other BMI categories share the all-zero indicator pattern with the reference; the earlier formula implying a fully specified separate category for every BMI group was too broad.

The model is additive: it has no age×label interaction, observation-count/span/density adjustment, occupancy covariate or serial-null correction. Adjustment for these categories does not eliminate confounding or establish an ovarian mechanism. The summary repeats the same overall adjusted coefficient on every contrast row; it is not a separate age-specific adjustment for each pair. The prior report's additional age coefficients are not present in the saved contrast table and are withdrawn.

### 4.4 Night versus day interpretation

Young female-coded night-HR amplitude is numerically greater than its daytime amplitude, but these are different signals with different finite-fit participant sets and no saved paired night-minus-day contrast. Young male-coded night-HR amplitude is also high and slightly greater in mean than the young female-coded value. Thus greater absolute nocturnal amplitude is not a female-specific finding. Relative night-HR fit strength does differ: stored young female-minus-male LS power is +0.04106 [0.02562, 0.05649], and incremental R² is +0.03555 [0.02011, 0.05099]. These distinguish normalized fit strength from bpm amplitude; they do not identify sleep stage, progesterone or autonomic pathways.

## 5. Recording Behavior and Artifact Comparisons

| Recording signal | Young female mean amplitude | Young male mean amplitude | Elderly female mean amplitude | Young female − male [stored 95% interval] | Young − elderly female [stored 95% interval] |
|---|---:|---:|---:|---:|---:|
| occ_daily | 0.02233 | 0.02431 | 0.02218 | -0.00198 [-0.00306, -0.00089] | +0.00015 [-0.00141, +0.00172] |
| nocturnal_compliance | 0.05170 | 0.05600 | 0.04911 | -0.00430 [-0.00800, -0.00059] | +0.00259 [-0.00315, +0.00834] |
| gap_episodes | 3.99662 | 4.22592 | 3.82350 | -0.22930 [-0.40796, -0.05063] | +0.17312 [-0.08843, +0.43467] |

The occupancy amplitudes are similar between young and elderly female-coded groups and lower in young female-coded than male-coded participants. This describes the fitted recording signals. Absence of a mean difference is not an equivalence test. More importantly, each signal's best period is optimized separately, without a saved joint HR–mask phase/coherence analysis or perturbation control. These comparisons cannot exclude sampling artifacts, MNAR selection, calendar routines or coupling between observed HR and recording. The earlier “decisive disentanglement” and behavioral-independence claims are withdrawn.

## 6. Representative Stored Fits

These examples are selected illustrations, not an unbiased estimate of prevalence. No daily trajectory plot is saved here, and the tabulated metrics alone do not validate a repeating ovulatory waveform.

| Participant | Saved label / age group | Eligible dates / full cached span | Selected p10 period, days | p10 amplitude, bpm | Incremental R² |
|---|---|---:|---:|---:|---:|
| 1077368 | Female / Young Adults | 128 / 728 | 26.5 | 3.7258 | 0.5311 |
| 516923 | Female / Young Adults | 121 / 498 | 30.0 | 3.9078 | 0.3776 |
| 1219228 | Female / Young Adults | 87 / 90 | 30.5 | 3.8226 | 0.3314 |
| 1128549 | Male / Young Adults | 71 / 441 | 35.0 | 7.5854 | 0.2658 |
| 951579 | Male / Young Adults | 125 / 669 | 31.5 | 5.3775 | 0.1959 |
| 1217923 | Female / Elderly | 376 / 385 | 25.0 | 0.0177 | 0.0108 |
| 256184 | Female / Elderly | 261 / 261 | 26.5 | 0.0692 | 0.0124 |

Participant 1077368 has 128 eligible observations within a cache whose full span is 728 days; participant 516923 has 121 eligible observations within a cache whose full span is 498 days. These cached spans need not equal the spans of the retained observations. They should not be described as continuous wear or directly observed consecutive cycles. Participant 1217923 has low p10 amplitude but night-HR amplitude 1.38545 bpm, so “complete flatlining of HR rhythmicity” is too broad. Exact ages and clinical interpretations attached to these IDs in the previous report are unsupported by the saved metadata.

## 7. Empirical Conclusions, Critique and Verification Boundary

### 7.1 What the saved data support

The selected-band p10 amplitude and relative fit strength are greater in female-coded `Young Adults` than in male-coded `Young Adults` or female-coded `Elderly`. The mean p10 amplitude difference between label groups is smaller in `Middle Ager`. Absolute nocturnal and daytime amplitudes do not show the same young female-minus-male excess. Recording signals have different group patterns. These are descriptive associations compatible with several biological and sampling explanations; menstrual specificity, menopausal attenuation and independence from observation patterns remain unestablished.

### 7.2 Material limitations

1. **Selected-frequency inference:** choosing the maximum of 21 periods and then using a fixed-period F tail omits the search. Serially correlated residuals, trends, weekly habits and the observation window can further alter the reference distribution. Nominal significance percentages are not calibrated detection rates.
2. **Multiplicity and optimization:** seven signals, four summary metrics and five contrast families produce 140 saved contrast rows. No correction, prespecified primary contrast, held-out phase/period validation or independent replication is documented in these outputs. All participants are analyzed descriptively; this is not the allocation-based classification benchmark.
3. **Observation support:** the screen controls eligible-date count and the full cached calendar span, but not continuity, longest gaps or number of well-observed cycles. Occupied-bin thresholds are not verified wear hours. Signal-specific missingness changes denominators. The single-frequency fit assumes a stationary period over the full span.
4. **Phenotype metadata:** salutation and broad age/BMI groups do not establish ovarian function, menstrual phase, pregnancy, menopause or hormone exposure. Contraception/HRT and other causes of HR variation are unobserved. Their direction or size of bias is not determined by these outputs.
5. **Recording dependence:** a low quantile of observed bins is sampling-dependent, and clock-night HR is not measured sleep HR. Separate recording-amplitude contrasts do not remove this dependence. No artifact-injection, mask-transplant or count-matched perturbation result is saved.
6. **Inferential artifact inconsistencies:** the stored contrast CSV includes `hr_p10/ls_power/Overall_F_vs_M` with t=17.338 but Welch p=1.0000, and `occ_daily/cos_amplitude/Target_vs_Control1` with t=−3.570, a negative interval excluding zero, but Welch p=1.0000. The summarizer explicitly implements a two-sided p-value. These fields are internally inconsistent. Other p-values are rounded to zero. Neither the source nor the saved inference was repaired or rerun here; retain interval/adjusted-model outputs only as stored, unverified inference. The raw CSV remains unchanged.
7. **Evidence not saved:** no trajectory figures, null/surrogate calibration distributions, held-out temporal checks, Garmin validation or saved verification-suite log exists in this directory. The presence of test code is not evidence that checks passed. No biological diagnosis or causal mechanism follows from the illustrative fits.

### 7.3 Reconciliation performed on 2026-10-20

Read the existing report, participant CSV, contrast CSV and relevant method code. Checked 11,139 unique IDs, screened and endpoint-specific counts, group means/sample SDs, nominal p-threshold counts and example rows using the saved participant metrics. Corrected report counts, rates, means, SDs, secondary contrasts and example spans; removed unsupported exact ages, significance-based prevalence, clinical-status labels and causal interpretations. Stored confidence intervals, coefficients and inferential p-values were not recomputed. No extraction/detection/group-test script, model or verification suite was executed; no source CSV was edited.

The source CSVs are [participant metrics](infradian_metrics_by_participant.csv) and [group contrasts](group_contrasts_age_and_sex.csv). The extraction implementation is [01_extract_and_detect_infradian.py](../scripts/01_extract_and_detect_infradian.py), and the group summarizer is [02_summarize_infradian_contrasts.py](../scripts/02_summarize_infradian_contrasts.py). Rerunning the original summarizer can overwrite contrast outputs; this reconciliation is not an endorsement of its inferential fields.

The [v2 experimental plan](../../female_infradian_cycles_v2/PLAN.md) specifies follow-up work separately. Planned controls and validation must not be read as completed results.
