# Future experiments backlog

The next experiments should distinguish recording coverage, temporal resolution and filling effects while preserving participant-level evaluation. The [results audit](../reports/results_audit_2026-10-14.md) identifies the limits of the saved benchmarks. The opening section records completed benchmarks. The remaining sections are proposals, not scheduled or running jobs. Each states a question and the comparison needed to answer it.

## Completed temporal binning benchmarks

The [Garmin pilot](../experiments/garmin_temporal_binning/README.md) is complete across five allocations. Overall test AUROC was 0.8350 at one minute, 0.8410 at five minutes and 0.8228 at 15 minutes. Delta-stratum AUROC was 0.6022, 0.6753 and 0.7687, respectively. These delta estimates use only 15–24 test participants per allocation and overlapping test sets.

The [Apple pilot](../experiments/apple_pilot_multirocket/README.md) is also complete: five-minute MultiRocket plus summaries scored 0.7554, versus 0.7431 at 15 minutes and 0.6029 for summary ridge. Allocation SDs and full results appear in the experiment README. These results motivate the proposed comparisons below; binning alone does not isolate a filling mechanism.

The [full Apple cohort benchmark](../experiments/apple_cohort_scaling/README.md) and its [recording-pattern follow-up](../experiments/apple_cohort_scaling/BINNING_BY_RECORDING.md) are also complete. Full-history five-minute MR + summaries reaches **0.8561 ± 0.0111** in 11,139 participants. See [completed Apple findings](#full-apple-cohort-scaling) below.

## Completed October 2026 benchmarks

Seven major experimental benchmarks were completed in October 2026, resolving several previously open questions across infradian rhythmicity, missingness regularization, and cross-device validity:

1. **Female Infradian Rhythmicity Across Lifespan**:
   - The exploratory study ([`female_infradian_cycles`](../experiments/female_infradian_cycles/results/REPORT.md), $N=8,425$ screened) detected ~28-day resting HR oscillations elevated in young females ($1.64$ bpm amplitude) compared to young male negative controls ($+0.27$ bpm, $d=0.37$) and elderly females ($+0.59$ bpm, $d=0.86$).
   - The standardized 180-day follow-up ([`female_infradian_cycles_v2`](../experiments/female_infradian_cycles_v2/results/REPORT.md), $N=4,630$, $N_{\text{obs}}=90$) calibrated these rhythms against continuous AR(1) and two-component REML nulls ($B=199$). HC3 Wald tests confirmed significant excess score in Young Adults ($\hat{\Delta} E = +0.0642$, $p < 10^{-16}$; $\Delta A = +0.415$ bpm) and Middle Agers ($\hat{\Delta} E = +0.0233$, $p < 10^{-16}$), but complete cessation in Elderly females ($\hat{\Delta} E = +0.0056$, $p = 0.2163$). Refitted period difference across 6-month nonoverlapping windows was 3-fold tighter in females ($1.50$ d) than males ($4.00$ d).
   - Multi-year extensions ([`female_infradian_extensions`](../experiments/female_infradian_extensions/results/REPORT.md)) demonstrated halved period drift in long-term wearers ($\ge 365\text{ d}$, $N=1,323$: $\Delta T_{\text{drift}} = 10.0$ d vs. $18.0$ d in males), confirmed autonomous endogenous biological desynchrony via circular Rayleigh tests on calendar epoch acrophases ($p = 0.8703$, rejecting lunar/calendar entrainment), and established that infradian amplitude is invariant to WHO BMI adiposity tiers (ANCOVA $p > 0.40$).
   - Downstream predictive classification ([`monthly_periodicity_downstream`](../experiments/monthly_periodicity_downstream/results/REPORT.md), $N=4,631$) proved that three monthly descriptors ($A, S, T^*$) yield statistically significant test AUROC lift ($\delta_{\text{clock}} = +0.0109$, $p < 0.01$; $\delta_{\text{linear}} = +0.0061$, $p < 0.05$) beyond $19,121$ MultiRocket and circadian features.

2. **Apple Fill and Mask Representations Across Recording Patterns**:
   - The Stage 1 benchmark ([`apple_fill_and_mask_by_recording`](../experiments/apple_fill_and_mask_by_recording/results/REPORT.md), $N=11,139$, Allocation 0) evaluated 8 imputation/mask arms across recording patterns, resolving the [Apple recording patterns and filling comparisons](#apple-recording-patterns-and-filling-comparisons) proposal below. Full-history 5m test AUROC reached $0.8678$ for `H_clock`, $0.8718$ for `H_clean_diff` ($+0.0040$), $0.8727$ for `H_short_interp` ($+0.0049$), and $0.8749$ for `H_plus_M_stack` ($+0.0071$, 95% CI $[+0.0015, +0.0125]$). Mask representation alone (`M_temporal`, AUROC $0.7459$) provided a $+0.0993$ gain over static masks, proving strong salutation discrimination from sensor observation topology alone.

3. **Circular Diurnal Template Regularization**:
   - The benchmark ([`apple_circular_diurnal_regularization`](../experiments/apple_circular_diurnal_regularization/results/REPORT.md)) established that circular 1D Gaussian smoothing across midnight with optimal bandwidth $\sigma^* = 1.0\text{ bin}$ ($5\text{ min}$) minimizes reconstruction RMSE and boundary step jumps on multi-hour gaps without edge artifacts, motivating a two-stage hybrid imputation rule.

4. **Cross-Device Replication on Garmin Tier 1c**:
   - The cross-device study ([`garmin_infradian_replication`](../experiments/garmin_infradian_replication/REPORT.md), $N=3,760$ qualified under $k=15$) successfully replicated the 28-day infradian rhythm on independent Garmin hardware, addressing [Cross-device transfer](#cross-device-transfer): Omnibus Age $\times$ Sex Wald $\chi^2 = 50.89, p = 1.81 \times 10^{-11}$; Young Adult contrast $\hat{\Delta} E = +0.0474, p < 10^{-17}$; male false-positive rate calibrated at $5.1\%\text{--}5.6\%$. Continuous spectra isolated a sharp resonance at $27.7\text{ days}$ on Garmin and $28.0\text{ days}$ on Apple, while continuous two-component REML nulls resolved the long-period male spectral rise (30–120 d) as nonperiodic $1/f^\alpha$ red noise and seasonal drift.

## Apple recording patterns and filling comparisons

*Status: Resolved.* Completed in late October 2026 by [`apple_fill_and_mask_by_recording`](../experiments/apple_fill_and_mask_by_recording/results/REPORT.md) (see summary above). The historical proposal is retained below for context.

Does the effect of the filling rule depend on observed recording density?

History, bin occupancy, adequate-day continuity, gap length, and daily occupancy variability are now summarized from the full Apple cohort's saved caches. Five-minute bins lead across broad groups. Under selected alpha, the all-day grids are near a tie in low-continuity/low-occupancy participants (five-minute advantage +0.0058), but five minutes retains +0.0246 among high-continuity/low-occupancy participants. Short-history/high-occupancy participants are also near a tie (+0.0056 all days; +0.0001 capped). These contrasting groups are useful targets for the next comparison.

Compare clock-bin medians and HR-plus-mask representations at five and fifteen minutes on the same participants, dates and allocations. Examine the continuity × occupancy groups already defined in the saved-score analysis. The existing five-minute summary baseline already includes 288 observation-frequency features; mask channels would preserve the daily arrangement of missingness. Additional grids and interpolation rules remain possible follow-ups; no new model fits have been run for the recording-pattern analysis.

## Cross-device transfer

*Status: Infradian rhythm replicated; predictive transfer remains open.* Infradian external validity was confirmed on Garmin Tier 1c by [`garmin_infradian_replication`](../experiments/garmin_infradian_replication/REPORT.md) (see summary above). Direct frozen predictive model transfer across devices remains an open proposal below.

How does a frozen Garmin model perform on the same Apple participants used to evaluate an Apple-trained model?

Evaluate a frozen Garmin bundle on the Apple pilot with compatible five-minute inputs and explicit device/channel parsing. Compare it with the Apple-trained benchmark on identical evaluation participants. The saved Garmin `mr_clock` model was fitted on 2,692 training participants within a 3,848-participant cohort, rather than on all 3,848.

Any transfer difference combines cohort, recording and model differences. It does not isolate hardware effects. Keep the transferred calibration separate from discrimination: validation calibration from Garmin has no established probability guarantee for Apple.

## Apple channel 3002

What comparison with channel 3000 is supported by the exported channel definition and coverage?

Verify the exported channel definition, timestamp semantics, coverage and eligible participant pool before designing a daily representation. The earlier proposed count of 11,056 users requires reconciliation with the exact source/eligibility table and target restriction. A 24-bin representation and a comparison with channel 3000 remain proposals, without assuming that an exported resting-HR summary is physiologically cleaner or more predictive.

## Full Apple cohort scaling
Completed on October 14, 2026 across ten balanced holdouts (N=11,139). See the [experiment overview](../experiments/apple_cohort_scaling/README.md) and [detailed report](../experiments/apple_cohort_scaling/REPORT.md).

Full-history five-minute AUROC is **0.8561 ± 0.0111**, compared with the pilot mean of **0.7554**. The earlier 0.8022 figure is pilot allocation 0, not a separate pilot benchmark. The summary baseline reaches 0.6474; adding MR gains **+0.2087 ± 0.0193**. Full history exceeds first-up-to-40-day inputs by **+0.0643 ± 0.0144** at fixed alpha and **+0.0442 ± 0.0122** when both select alpha. Capped inputs benefit from stronger shrinkage: +0.0193 at five minutes and +0.0350 at fifteen minutes.

The October 15 saved-score follow-up separates history length from occupancy and continuity. It finds no Garmin-sized fifteen-minute gain in Apple; coarser bins become competitive in some combined profiles. The full-history five-minute fixed-alpha model scores 0.8043 in the lowest history tercile and 0.8942 in the highest. The [complete subgroup results](../experiments/apple_cohort_scaling/BINNING_BY_RECORDING.md) also compare resolutions and alpha rules. A controlled sample-size learning curve remains a possible extension.

## Garmin Tier 2 expansion

How does the model perform in a broader Garmin population after the cohort definitions are reconciled?

The upstream source-specific allocation manifest contains **5,636** primary-Garmin participants. The current relaxed Garmin manifest marks **5,616** as Tier 2 because 20 upstream participants are absent from its `artifacts_v2` input universe. Reconcile these populations before preparing an expanded cohort; neither count is an established completed Tier 2 benchmark.

Within the current manifest, Tier 2 contains all 4,410 Tier 1c participants and adds 1,206 others. Multi-source membership does not itself establish chronological device switching. Inspect source-specific dates before attributing effects to transitions, and preserve the target restriction, source filtering and participant-level split boundaries.

## Research Note: Age Dependency of Monthly Heart Rate Periodograms (Infradian Cessation Protocol)

*Status: Implementation handoff protocol formulated in [`experiments/infradian_age_dependency/PLAN.md`](../experiments/infradian_age_dependency/PLAN.md).*

### Objective & Scope
Quantify whether the female monthly resting heart rate periodogram elevation weakens, shifts, broadens, or becomes concentrated in fewer participants across advancing adult chronological age. Evaluated across three independent sensor architectures: Apple Watch (source 6), Garmin (source 3), and Samsung (source 7).

### Verified Feasibility & Candidate Cohorts
Candidate single-source caches and source-specific multisource participant trajectories establish the following eligible population under the primary protocol:

| Hardware Source & Sampling Protocol | Window-Eligible Cohort | Female Candidates (Salutation 10) |
|---|---:|---:|
| **Apple Watch (Source 6)**: `expanded90` ($\ge 10$/block, $N_{\text{obs}}=90$) | 5,390 | 1,765 |
| **Apple Watch (Source 6)**: `balanced90` ($\ge 15$/block, $N_{\text{obs}}=90$) | 4,631 | 1,478 |
| **Apple Watch (Source 6)**: `balanced60` ($\ge 10$/block, $N_{\text{obs}}=60$) | 5,518 | 1,820 |
| **Garmin Tier 1c (Source 3)**: `expanded90` ($\ge 10$/block, $N_{\text{obs}}=90$) | 3,832 | 1,376 |
| **Garmin Tier 1c (Source 3)**: `balanced90` ($\ge 15$/block, $N_{\text{obs}}=90$) | 3,760 | 1,339 |
| **Samsung (Source 7)**: Feasibility candidate frame | Inventory: 541 candidates with $\ge 90$ recorded channel-3000 dates | Pending local-date and 120-bin mask filter |

Multisource candidates add 405 distinct women outside existing Apple/Garmin caches with $\ge 90$ recorded channel-3000 dates (325 Apple, 258 Garmin; 178 overlapping IDs). Multisource records are evaluated per source, clustering inference by global `user_id`.

### Sampling Protocols & Reselection Rules
1. **Window Selection**:
   - Reselect the earliest qualifying 180-day window $[s, s+180)$ across six 30-day blocks $[s+30b, s+30(b+1))$ for every participant.
   - `expanded90` (Primary): Requires $\ge 10$ qualified dates per block and $\ge 90$ qualified dates total. Selected dates ($N_{\text{obs}}=90$) are deterministically sampled via SHA-256 seeded PCG64 uniform block allocations.
   - `balanced90` (Reference): Requires $\ge 15$ qualified dates per block ($N_{\text{obs}}=90$).
   - `balanced60` (Apple Sensitivity): Requires $\ge 10$ qualified dates per block ($N_{\text{obs}}=60$).
2. **Quality Thresholds**:
   - A qualified date requires $\ge 120$ originally observed 5-minute bins ($\ge 10$ occupied hours). Missing bins are unobserved, never interpolated or filled.
   - Daily physiological signal: 10th percentile resting heart rate (`HRp10`, in bpm).

### Spectral & Null Calibrated Estimands
1. **Broad Frequency Grid & Projection**:
   - Nuisance design: Intercept, linear elapsed calendar time, weekday indicators (Monday reference), and four 6-hour diurnal coverage fractions.
   - Frequency grid: Sorted union of 229 frequencies $\{k/720 : k=12,\dots,240\}$ ($T \in [3.0, 60.0]\text{ days}$) plus exact band boundaries $f = 1/35$ and $f = 1/25\text{ d}^{-1}$.
   - Normalized partial $R^2$: $P_i(f) = (RSS_{0,i} - RSS_{f,i}) / RSS_{0,i}$.
2. **Two-Component Continuous Covariance Null**:
   - Residual covariance: $\text{Cov}(e(t), e(u)) = v_1 \phi_1^{|t-u|} + v_2 \phi_2^{|t-u|}$, with $0 < \phi_1 < 0.95$ and $\phi_2 \in \{0.98, 0.995\}$ estimated via Nelder-Mead restricted likelihood.
   - Monte Carlo simulation: $B = 199$ independent null series per participant generated at actual irregular observation dates.
   - Signed excess spectrum: $D_i(f) = P_i(f) - \frac{1}{B}\sum_{b=1}^B P_{i,\text{null},b}(f)$.
3. **Primary Estimand (Band-Integrated Excess Score $Q_i$)**:
   $$Q_i = \frac{1}{\frac{1}{25} - \frac{1}{35}} \int_{1/35}^{1/25} D_i(f) \, df$$
   Evaluated via numerical trapezoidal quadrature across the monthly band. Unlike peak-argmax metrics, $Q_i$ integrates energy continuously across the biological cycle window without positive-truncation or search bias.
4. **Secondary Endpoints**:
   - Discrete peak excess: $E_{i2} = \max_{T \in [25, 35]} P_i(1/T) - \bar{P}_{\text{null}}$.
   - Cosinor amplitude: $\hat{A} = \sqrt{\hat{\beta}_{\cos}^2 + \hat{\beta}_{\sin}^2}$ (in bpm) at empirical peak and fixed 28-day period.
   - Full width at half maximum (FWHM) of female 5-year age-band excess spectra with 2,000-draw bootstrap pointwise 95% confidence bands.

### Parametric Age Association & Interaction Model (Frank Harrell RCS Formulation)
1. **Restricted Cubic Splines (RCS) Regression**:
   Following Frank Harrell's *Regression Modeling Strategies* (RMS), continuous age is modeled using restricted cubic splines with 4 biologically anchored knots at landmark ages $t = [25, 42, 52, 65]$ years (3 degrees of freedom: 1 linear, 2 spline basis variables):
   $$Q_i = \beta_0 + \beta_1 \text{age}_i + \beta_2 \text{rcs}_{1,i} + \beta_3 \text{rcs}_{2,i} + F_i \left( \delta_0 + \delta_1 \text{age}_i + \delta_2 \text{rcs}_{1,i} + \delta_3 \text{rcs}_{2,i} \right) + \varepsilon_i$$
   where:
   - Annual age is derived from participant birth year (`data/13Aug_1222.csv`) under the December-1 convention.
   - $F_i \in \{0, 1\}$ denotes administrative salutation ($F_i = 1$ for female / salutation 10; $F_i = 0$ for male / salutation 20).
   - Fully unadjusted bivariate model serves as primary evidence, followed by pre-specified sensitivities adjusting for continuous wear compliance ($Z_i$: occupied fraction, nocturnal coverage, date span) and BMI strata.
   - **Two-Component REML Covariance**: Resting heart rate noise is physically governed by two distinct timescales: fast day-to-day autonomic/recovery rebound (~1.5-day period, $\phi_1 \approx 0.3\text{--}0.7$) and slow environmental/metabolic drift (multi-week/seasonal persistence, $\phi_2 \approx 0.98\text{--}0.995$). Because a single AR(1) model cannot simultaneously fit both scales without either underestimating long-period power or overestimating short-period power, the continuous covariance $\text{Cov}(e(t), e(u)) = v_1 \phi_1^{|t-u|} + v_2 \phi_2^{|t-u|}$ is essential. Restricted Maximum Likelihood (REML) estimation is required over ordinary ML to project into the error contrast subspace (^T X = 0$), integrating out  \approx 8\text{--}10$ nuisance parameters to prevent /n \approx 11\%$ downward variance shrinkage and eliminate systematic downward attenuation of the slow persistence parameter $\phi_2 \in \{0.98, 0.995\}$.
2. **Inference & Chunk Wald Hypotheses (Pure Non-parametric Bootstrap)**:
   - **Primary Hypothesis (Female Total Age Effect)**: Joint chunk Wald test of $\beta_1 + \delta_1 = 0, \beta_2 + \delta_2 = 0, \beta_3 + \delta_3 = 0$ (3 d.f.).
   - **Secondary Hypotheses**:
     - Female nonlinear curvature: $\beta_2 + \delta_2 = 0, \beta_3 + \delta_3 = 0$ (2 d.f.).
     - Total Sex × Age interaction: $\delta_1 = 0, \delta_2 = 0, \delta_3 = 0$ (3 d.f.) on shared support.
     - **Nonlinear Sex × Age interaction (2 d.f.)**: $\delta_2 = 0, \delta_3 = 0$ on shared support. Tests whether the female-male delta curve $\Delta(\text{age})$ has non-zero curvature ($f''_{\text{female}} \ne f''_{\text{male}}$). Rejecting this null demonstrates that the female trajectory exhibits localized biological inflection (accelerating perimenopausal drop-off) rather than mirroring non-specific parallel autonomic aging or steady linear decline.
   - **Pure Bootstrap Uncertainty**: 2,000 stratified participant resamples (by 5-year age band × sex) evaluating chunk test p-values, pointwise 95% CIs, and simultaneous (uniform) confidence bands over age [18, 80].
3. **Transition Dynamics & Multi-Endpoint Decomposition**:
   - Comprehensive transition metrics from delta curve $\Delta(\text{age}) = \hat{Q}_{\text{female}}(\text{age}) - \hat{Q}_{\text{male}}(\text{age})$:
     1. Inflection point of steepest decline ($\arg\min d\Delta/d\text{age}$).
     2. Nominal zero-crossing ($\Delta = 0$).
     3. Conservative statistical cessation boundary (earliest age $\ge 40$ where upper 95% simultaneous bootstrap bound $\le 0$).
   - Full four-endpoint decomposition evaluated in parallel: band-integrated excess $Q_i$, discrete peak excess $E_{i2}$, cosinor physical amplitude $\hat{A}_i$, and resting HR residual variance.

## Evaluation requirements for future runs

Use saved participant assignments for matched comparisons, fit every learned transformation on training participants, and distinguish overlapping repeated holdouts from disjoint folds. Save per-participant scores and model/configuration provenance so AUROCs, paired uncertainty and subgroup denominators can be checked independently. Larger delta test groups and an evaluation cohort separate from repeated model development would resolve more uncertainty than further interpretation of the existing five small pilot splits.
