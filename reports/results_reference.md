# Saved results reference

This reference lists the authoritative saved results, updated on 2026-10-20 to incorporate the October 2026 infradian rhythm investigations, downstream monthly periodicity classification, Apple fill and mask representations across recording patterns, circular diurnal template regularization, cross-device transfer pilot, and the infradian age dependency protocol. Each comparison includes its cohort, input history, and evaluation design. Compiled laboratory reference PDFs and master manifests are indexed in the [scientific provenance catalog](scientific_provenance_pdfs/CATALOG.md). Implementation and lineage details are in the [experiment index](../learnings/experiment_lineage_worktree.md). The [results audit](results_audit_2026-10-14.md) covers earlier benchmark iterations.

## Terms and conventions

| Term | Meaning here |
|---|---|
| Target | Recorded salutation: `10 → 0/F`, `20 → 1/M` |
| Participant | The unit assigned to training, validation, or test |
| Allocation | One saved assignment of participants to training, validation, and test |
| Arm | One model or preprocessing condition evaluated within an experiment |
| Repeated holdouts | Several allocations of the same cohort; their test participants can overlap |
| Adequate HR day | At least eight observed local clock-hours, plus either eight hours of interval-union coverage or 60 valid readings |
| First 40 days | The first 40 adequate days in date order; not necessarily consecutive calendar days |
| First up to 40 days | The first `min(n_adequate, 40)` days; the Apple cohort retains participants with only 14–39 adequate days |
| Sampled days | A fixed number of adequate days drawn from the history specified by that experiment |
| All days / full history | Every adequate day available for a participant; history lengths vary between participants |
| Frozen model | A previously fitted model applied without retraining |
| Clock-bin median fill | Missing HR bins filled with training-derived medians for the corresponding clock positions |
| `±` | Sample SD across allocations, unless explicitly stated otherwise |
| Band-integrated excess score $Q_i$ | $Q_i = \frac{1}{1/25 - 1/35} \int_{1/35}^{1/25} D_i(f) df$: Mean signed excess spectral power of participant $i$ across the infradian frequency band $f \in [1/35, 1/25]\text{ cycles/day}$ ($T \in [25, 35]\text{ days}$), integrated via trapezoidal quadrature over signed excess $D_i(f) = P_i(f) - \bar{P}_{\text{null}, i}(f)$ |
| Peak excess score $E_i$ | $E_i = \max_{T \in [25, 35]} P_i(1/T) - \bar{P}_{\text{null}}$: Discrete argmax excess partial $R^2$ score over candidate periods $T \in [25, 35]\text{ days}$ above the mean bootstrap null envelope $\bar{P}_{\text{null}}$ |
| Cosinor amplitude $A_i$ | $A_i = \sqrt{\beta_{\cos}^2 + \beta_{\sin}^2}$ (in bpm): Half of the fitted peak-to-trough sinusoidal excursion estimated by single-harmonic cosinor linear regression on baseline-residualized daily resting HR proxy |
| Two-component continuous REML null | $\text{Cov}(e(t), e(u)) = v_1 \phi_1^{|t-u|} + v_2 \phi_2^{|t-u|}$: Continuous calendar-time auto-regressive error model combining a fast daily autocorrelation component ($\phi_1 \in [0, 0.95)$) and a slow long-memory drift component ($\phi_2 \in \{0.980, 0.995\}$), with positive variance parameters $(v_1, v_2)$ fitted via Restricted Maximum Likelihood (REML) on the baseline-projected residual subspace |
| 1D Circular Gaussian template bandwidth $\sigma$ | Kernel standard deviation (measured in 5-minute bins) for periodic circular convolution on $\mathbb{T}_{288} = \mathbb{Z}/288\mathbb{Z}$ across the midnight wraparound boundary ($23:55 \to 00:00$) |
| $M_{\text{temporal}}$ vs $H$ | $M_{\text{temporal}}$ denotes sequence features extracted strictly from binary observation indicator series $M_t = \mathbb{I}(\text{HR}_t \text{ is finite})$ without physiological values; $H$ denotes sequence features extracted from filled heart-rate values |

AUROC uses participant scores unless a row says **day-level**. SD across repeated holdouts is not a confidence interval or variation across independent cohorts. Compare models within the same experiment. Absolute scores across different experiments need not estimate the same quantity.

## Model and experiment names

| Name in reports or code | Meaning |
|---|---|
| RF | Random forest |
| MR / MultiRocket | Sequence transform; the reported temporal arms also include summary features |
| B40 | The historical 305-feature participant summary and observation-mask block; later models apply the same recipe to other history lengths |
| Profile24 / P24 | Hourly HR profile summaries appended to B40 |
| Profile288 | Five-minute HR profile summaries appended to B40 |
| R1a | Circadian-feature experiment; its feature block is evaluated alongside that study’s BASE model |
| R1b / P40 | Date-aware experiment and its participant-level feature block; P40 is distinct from P24 and B40 |
| BASE | The baseline defined by the named experiment; do not assume that BASE and B40 are interchangeable |
| Combined | B40 + MultiRocket + HYDRA in the first-40-day temporal experiment |
| `win` | The named default RF arm quoted in the source-specific summary |
| `mr_clock` / `mr_global` | Saved Heartbeat Lab bundles using clock-bin / global median filling |
| `female_infradian_cycles` | Exploratory Lomb-Scargle and harmonic cosinor screening on Apple Watch ($N=8,425$) |
| `female_infradian_cycles_v2` | Standardized 180-day exposure ($N_{\text{obs}}=90$, $k=15$) and continuous AR(1) bootstrap null calibration on Apple Watch ($N=4,630$) |
| `female_infradian_extensions` | Multi-year sliding-window period stability ($\ge 365\text{ d}$, $N=1,323$), circular calendar acrophase Rayleigh uniformity, and WHO BMI invariance |
| `garmin_infradian_replication` | Independent cross-device replication on Garmin Tier 1c ($N=3,760$), exposure sensitivity ($k=10$, $N=5,516$), and 3–120 day spectral diagnostics resolving male red-noise rise |
| `monthly_periodicity_downstream` | Downstream sex classification on Allocation 0 test set ($N=4,631$ cohort, $N_{\text{test}}=725$) augmenting MultiRocket with 3 monthly infradian descriptors ($A, S, T^*$) and recording controls ($Q$) |
| `apple_fill_and_mask_by_recording` | 8-arm imputation and observation-mask architecture sensitivity benchmark on Apple Allocation 0 ($N=11,139$) |
| `apple_circular_diurnal_regularization` | 1D periodic circular Gaussian smoothing benchmark across 11,900 injected gaps ($N=200$) optimizing bandwidth $\sigma$ across midnight boundaries |
| `apple_garmin_transfer_pilot` | Cross-device transferability pilot evaluating mutual generalization of 40-day static HR and recording summary models between Apple and Garmin ($N=1,000$ pilots) |
| `infradian_age_dependency` | Implementation handoff protocol for continuous age modeling (unpenalized quadratic age OLS, `expanded90` protocol, band-integrated excess $Q_i$) across Apple, Garmin, and Samsung cohorts |

For feature construction, see the [MultiRocket model notes](../learnings/multirocket_garmin_synthesis.md). For each study’s original naming, see its report in the [experiment index](../learnings/experiment_lineage_worktree.md).

## Cohorts

Device provenance is recorded as Garmin **source 3**, Apple **source 6**, and Samsung **source 7**. Core channels are HR **3000**, resting HR **3001**, and hourly resting HR **3002**. The source identifies the recording origin; the channel identifies the exported quantity.

Pooled v1 includes **20,485** participants; pooled v2 includes **20,423**. Base eligibility requires a retained daily source, valid epoch measurements, and recorded salutation 10/20. It also requires at least 14 adequate days in one core channel. Source-specific analyses then select participants with exactly one retained epoch source.

| Source | Base v2 single-source cohort | Strict cohort |
|---|---:|---:|
| Garmin / 3 | 4,410 | 3,848 |
| Apple / 6 | 11,143 | 4,446 |
| Samsung / 7 | 1,498 | 541 |

Source: [source_cohorts.csv](../experiments/artifacts_sq/source_cohorts.csv).

The strict gate searches 13-week windows. A qualifying window contains at least 11 adequate weeks and 65 strict days. An adequate week contains at least five strict days. A strict day has at least 12 hours of interval-union coverage in the cached channel data.

Each core channel is assessed separately. The final strict cohort is the union of participants who pass on any core channel. Intervals are assigned to their start date without midnight clipping, so the cached coverage quantity can exceed 24 hours.

| Garmin label in code | Cohort |
|---|---|
| Tier 0 | Strict cohort, N=3,848 |
| Delta | Added participants who failed the strict gate, N=562 |
| Tier 1c | Strict cohort plus added participants, N=4,410 |

The adequate-day rule differs from the strict longitudinal gate. Forty adequate days need not be consecutive calendar days. The strict Garmin full-history cache contains 80–979 adequate days per participant. Eighty is the observed minimum, not another eligibility cutoff.

In October 2026 infradian analyses, cohorts apply window-qualification gates over a 180-calendar-day interval $[s, s+180)$ partitioned into six 30-day blocks:
- **Standard Exposure ($k=15$, $N_{\text{obs}}=90$)**: Requires $\ge 15$ qualified dates ($\ge 120$ occupied 5-minute bins, i.e., $\ge 10\text{ h}$) per block, sampling exactly 15 dates per block. Retains $N=4,630$ on Apple Watch and $N=3,760$ on Garmin Tier 1c.
- **Relaxed Exposure ($k=10$, $N_{\text{obs}}=60$)**: Requires $\ge 10$ qualified dates per block, sampling exactly 10 dates per block. Retains $N=5,516$ on Apple Watch.
- **Multi-Year Cohort**: Requires initial screening pass and continuous wear history span $\ge 365\text{ days}$. Retains $N=1,323$ participants on Apple Watch.

## Garmin first 40 adequate days

**Cohort:** 3,848 strict Garmin participants. **History:** first 40 adequate days. **Evaluation:** ten repeated holdouts, each with 2,696/579/573 training/validation/test participants.

| Model | Mean test AUROC across allocations |
|---|---:|
| Summary ridge | 0.7157 |
| Summary RF | 0.7031 |
| Hourly profile ridge | 0.7370 |
| MultiRocket | 0.8361 |
| HYDRA | 0.8442 |
| B40 + MultiRocket + HYDRA | 0.8565 |
| MultiRocket with within-day HR values shuffled | 0.6643 |

B40 contains 305 summary and observation-mask features. MultiRocket and HYDRA add sequence features to B40. The combined model does not include the separate profile blocks.

These are the corrected v2 results. Preprocessing is fitted within each allocation; regularization was selected on allocation-0 validation and reused. Some later test participants belonged to that earlier validation set.

Sources: [report](../experiments/temporal_garmin/results/REPORT.md), [metrics](../experiments/temporal_garmin/results/temporal_metrics.csv).

## Garmin day budgets

**Cohort:** the same 3,848 participants. **History:** days sampled across available adequate history. **Evaluation:** the same ten allocations. **Model:** MultiRocket plus B40, ridge alpha 1000.

| Days per participant | Test AUROC, mean ± SD |
|---|---:|
| 40 | 0.8583 ± 0.0115 |
| 80 | 0.8854 ± 0.0115 |
| All, mean 425 | 0.9264 ± 0.0055 |

The all-days endpoint reuses allocation-0 fill values and MultiRocket biases across all allocations. Later test participants therefore contributed to fitted preprocessing. Its effect on AUROC is unknown. This endpoint also uses a variable history budget and one kernel draw.

Sources: [day-budget report](../experiments/temporal_garmin/results/DAYS_SCALING_REPORT.md), [metrics](../experiments/temporal_garmin/results/days_scaling_all_metrics.csv).

## Saved Heartbeat Lab models

**Cohort:** 3,848 strict Garmin participants. **History:** all adequate days. **Evaluation:** one 2,692/578/578 split.

| Saved bundle | Fill | Validation AUROC | Test AUROC |
|---|---|---:|---:|
| `mr_clock` | Clock-bin median | 0.8962 | 0.9177 |
| `mr_global` | Global median | 0.9034 | 0.9225 |

Both models fit preprocessing on training participants. They use different fitting choices from the historical endpoint and cannot quantify its evaluation bias.

Sources: local `HEARTBEAT_LAB/artifacts/models/`, [Lab audit](../HEARTBEAT_LAB/AUDIT_REPORT.md), [model notes](../learnings/multirocket_garmin_synthesis.md).

## Expanded Garmin cohort

**Cohort:** 4,410 single-source Garmin participants. **History:** all adequate days. **Evaluation:** ten repeated holdouts, each with 662 test participants.

| Model and test population | Test AUROC, mean ± SD |
|---|---:|
| MultiRocket, all participants | 0.9094 ± 0.0058 |
| MultiRocket, strict cohort within test | 0.9235 ± 0.0069 |
| MultiRocket, added participants within test | 0.8156 ± 0.0448 |
| Summary ridge, all participants | 0.7196 ± 0.0179 |

The mean paired MultiRocket minus summary difference is **+0.1898**. Source: [Tier1c metrics](../HEARTBEAT_LAB/artifacts/tier1c_cv_results/tier1c_10_alloc_results.csv).

A separate evaluation applies the frozen strict-cohort `mr_clock` model to all 562 added participants. Its AUROC is **0.7850**. That evaluation and expanded-cohort retraining use different models and test samples. Their difference does not isolate the effect of retraining. Source: [frozen-model results](../HEARTBEAT_LAB/artifacts/eval_delta_562/delta_evaluation_report.json).

## Garmin and Apple bin-size pilots

Each pilot has **1,000 participants** and five repeated 700/150/150 holdouts. Both use all adequate days. Garmin histories range from 14–979 days, mean 395.5; Apple histories range from 14–887 days, mean 264.9.

| Model | Garmin test AUROC, mean ± SD | Apple test AUROC, mean ± SD |
|---|---:|---:|
| MultiRocket, 1 minute | 0.8350 ± 0.0369 | Not evaluated |
| MultiRocket, 5 minutes | 0.8410 ± 0.0380 | 0.7554 ± 0.0594 |
| MultiRocket, 15 minutes | 0.8228 ± 0.0411 | 0.7431 ± 0.0437 |
| Summary ridge | 0.7143 ± 0.0360 | 0.6029 ± 0.0631 |

The populations are not matched across devices. Their source cohorts, recording histories, and sampling procedures differ.

The Garmin added-participant test groups contain only 15–24 participants per allocation. Their mean AUROCs are 0.6022, 0.6753, and 0.7687 at 1, 5, and 15 minutes, respectively. The paired 15-minute minus one-minute difference is **+0.1665 ± 0.1743 SD**.

The Apple five-minute minus summary difference is **+0.1526 ± 0.0423 SD**. All 1,000 sampled participants produced adequate-day caches. This retention result applies to the selected pilot, not the full source population.

Tier1c and both pilots save allocation metrics without participant predictions. Their reported means and SDs can be checked, but their AUROCs cannot be reconstructed independently from saved scores.

Sources: [Garmin report](../experiments/garmin_temporal_binning/README.md), [Apple report](../experiments/apple_pilot_multirocket/README.md). See [recording-pattern notes](../learnings/wear_phenotypes_and_missingness_synthesis.md) for interpretation.

## Full Apple cohort benchmark

**Cohort:** 11,139 qualified single-source Apple participants from 11,143 upstream candidates. **History:** all adequate days or first up to 40. **Evaluation:** ten repeated 7,797/1,671/1,671 train/validation/test holdouts. **Target:** recorded salutation. Apple days require eight observed clock hours and 60 valid type-3000 readings; participants require at least 14 adequate days.

Mean history is 266.4 adequate days (median 261, range 14–902). The 925 participants (8.30%) with 14–39 days contribute identical input histories to the two budget arms. Mean participant-level occupied-bin fractions are 54.0% at five minutes and 68.3% at fifteen minutes.

| Representation | History | Bins | Fixed α = 1000 test AUROC | Validation-selected α test AUROC |
|---|---|---|---:|---:|
| MultiRocket + summaries | All days | 5 min | 0.8561 ± 0.0111 | 0.8553 ± 0.0123 |
| MultiRocket + summaries | All days | 15 min | 0.8348 ± 0.0079 | 0.8378 ± 0.0106 |
| MultiRocket + summaries | First up to 40 | 5 min | 0.7918 ± 0.0148 | 0.8110 ± 0.0131 |
| MultiRocket + summaries | First up to 40 | 15 min | 0.7636 ± 0.0113 | 0.7985 ± 0.0097 |
| Summary ridge | All days | 5 min | 0.6474 ± 0.0180 | 0.6468 ± 0.0151 |
| Summary ridge | First up to 40 | 5 min | 0.6298 ± 0.0164 | 0.6340 ± 0.0143 |

The executed alpha candidates are `{100, 500, 1000, 5000, 10000}`. Training participants determine all learned preprocessing and model fits; validation selects alpha and a balanced-accuracy threshold. The separate summary baseline has 305 five-minute features. MultiRocket includes analogous summaries at its own grid, with 305 features at five minutes and 113 at fifteen minutes, plus pooled sequence features.

| Paired comparison | Mean test AUROC difference ± SD |
|---|---:|
| MR + summaries minus summaries; 5 min, all days, fixed α | +0.2087 ± 0.0193 |
| All days minus capped; 5 min, fixed α | +0.0643 ± 0.0144 |
| All days minus capped; 5 min, selected α | +0.0442 ± 0.0122 |
| 5 min minus 15 min; all days, fixed α | +0.0213 ± 0.0068 |
| 5 min minus 15 min; all days, selected α | +0.0175 ± 0.0082 |
| Selected minus fixed α; capped, 5 min | +0.0193 ± 0.0055 |
| Selected minus fixed α; capped, 15 min | +0.0350 ± 0.0072 |

The follow-up on saved predictions finds a positive mean five-minute advantage in each low/middle/high history, occupancy, and continuity group. Selected-alpha full-history differences are +0.0058 in the low-continuity/low-occupancy group, +0.0246 in high-continuity/low-occupancy, and +0.0056 in low-history/high-occupancy. Exact subgroup definitions, counts, paired SDs, and capped-history results appear in [binning by recording pattern](../experiments/apple_cohort_scaling/BINNING_BY_RECORDING.md).

Sources: [benchmark report](../experiments/apple_cohort_scaling/REPORT.md), [metrics](../experiments/apple_cohort_scaling/artifacts/apple_benchmark_metrics.csv), [recording-pattern summary](../experiments/apple_cohort_scaling/artifacts/binning_by_recording/summary.csv). The full benchmark saves 120 metric rows, 300 alpha-candidate rows, and 200,520 participant test prediction rows; the follow-up uses these existing predictions without model refitting.

## Infradian rhythms across lifespan

This line of investigation evaluates monthly heart-rate oscillations ($T \in [25.0, 35.0]\text{ days}$) across demographic categories (salutation $10 \to \text{female}$, $20 \to \text{male}$; age categories `Young Adults`, `Middle Ager`, `Elderly`) under progressively rigorous null calibration, multi-year observation, and cross-device testing.

### 1. Exploratory Lomb-Scargle discovery ($N=8,425$)

**Cohort:** Apple Watch longitudinal cohort ($N=11,139$ total). **Screening requirement:** $\ge 60$ adequate calendar days ($\ge 120$ occupied 5-minute bins, i.e., $\ge 10\text{ h}$ of valid HR in 25–230 bpm) with cached span $\ge 60$ calendar days. Retains $N=8,425$ participants ($75.64\%$ retention).

**Protocol:** Daily resting HR proxy ($hr_{p10}$, 10th percentile of valid 5-min bins), nocturnal HR ($night_{hr}$, 00:00–06:00), daytime HR ($day_{hr}$, 09:00–18:00), and nocturnal dipping ratio. Periodicity evaluated across 21 frequencies ($T \in [25.0, 35.0]\text{ days}$, step $0.5\text{ d}$) via single-harmonic OLS cosinor regression with linear drift:
$$y(t_d) = \beta_0 + \beta_{\text{trend}} t_d + \beta_c \cos(2\pi t_d / T_k) + \beta_s \sin(2\pi t_d / T_k) + \epsilon(t_d)$$
and Townsend-normalized detrended Lomb-Scargle periodograms.

| Signal | Female / Young Adults ($N=502$) | Male / Young Adults ($N=1,209$) | Female / Elderly ($N=289$) |
|---|---:|---:|---:|
| $hr_{p10}$ amplitude (bpm) | $1.6402 \pm 0.7152$ | $1.3704 \pm 0.7320$ | $1.0530 \pm 0.6246$ |
| $night_{hr}$ amplitude (bpm) | $3.3650 \pm 2.9025$ ($N=408$) | $3.4645 \pm 3.7846$ ($N=1,074$) | $1.8974 \pm 2.1818$ ($N=239$) |
| $day_{hr}$ amplitude (bpm) | $2.0459 \pm 0.8950$ | $2.0998 \pm 0.9288$ | $1.5518 \pm 0.8208$ |
| Dipping ratio amplitude (fraction) | $0.0428 \pm 0.0404$ ($N=408$) | $0.0480 \pm 0.0491$ ($N=1,072$) | $0.0269 \pm 0.0265$ ($N=238$) |

| Signal / comparison | Group Ns | Mean difference | Stored 95% Welch interval | Stored Cohen's d |
|---|---:|---:|---:|---:|
| $hr_{p10}$ / Target vs Control 1 (Young F − Young M) | 502 / 1,209 | +0.2698 bpm | [+0.1947, +0.3448] | +0.3710 |
| $hr_{p10}$ / Target vs Control 2 (Young F − Elderly F) | 502 / 289 | +0.5872 bpm | [+0.4917, +0.6828] | +0.8591 |
| $hr_{p10}$ / Middle Ager F vs M | 2,088 / 3,634 | +0.0859 bpm | [+0.0540, +0.1178] | +0.1431 |
| $hr_{p10}$ / Overall F vs M | 2,879 / 5,545 | +0.1056 bpm | [+0.0769, +0.1342] | +0.1650 |

*Methodological caveat:* Maximizing unconstrained harmonic fit across 21 frequencies without autocorrelation calibration inflates nominal F-test significance: 69.15% of Young Males exhibit nominal $p < 0.05$ due to underlying red noise. This finding necessitated standardized exposure and surrogate null calibration. Source: [exploratory report](../experiments/female_infradian_cycles/results/REPORT.md).

### 2. Standardized 180-day follow-up ($N=4,630$, $N_{\text{obs}}=90$, $k=15$ and $k=10$ sensitivity)

**Cohort:** Apple Watch single-source participants. **Exposure design:** Earliest 180-calendar-day interval $[s, s+180)$ partitioned into six 30-day blocks. The standard protocol ($k=15$) requires $\ge 15$ qualified dates per block and samples exactly 15 dates uniformly via deterministic SHA-256 seeding ($N_{\text{obs}}=90$ dates). Retains $N=4,630$ qualified participants (Young F: 224, Young M: 666, Middle F: 1,082, Middle M: 2,086, Elderly F: 172, Elderly M: 400).

**Null calibration:** Baseline nuisance design matrix $X_0$ (intercept, linear calendar trend, weekday indicators, four 6-hour clock coverage fractions) residualizes daily $hr_{p10}$ via projection $P_0$. Frisch-Waugh-Lovell harmonic score $S_i = \max_{T \in [25, 35]} \Delta \text{RSS}(T)/\text{RSS}_0$ is calibrated against $B=199$ surrogate series simulated from a participant-specific continuous calendar-time AR(1) process ($e_0(t_k) = \phi^{\Delta_k} e_0(t_{k-1}) + \sigma \sqrt{1 - \phi^{2\Delta_k}} z_k$). Excess score is $E_i = S_i - \frac{1}{B} \sum_{b=1}^B S_{\text{null}, b}$.

| Contrast / Hypothesis | Parameter estimate $\hat{\Delta}$ | $\text{SE}_{\text{HC3}}$ | 95% Bootstrap CI | Wald $\chi^2$ | df | $p_{\text{unadj}}$ | $p_{\text{Holm}}$ | Rejection ($\alpha=0.05$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Age $\times$ Sex Omnibus** | — | — | — | 56.763 | 2 | $4.72 \times 10^{-13}$ | $9.44 \times 10^{-13}$ | **Reject $H_0$** |
| **Young Adults (y0 vs y1)** | +0.0642 | 0.0063 | [+0.0520, +0.0767] | 102.434 | 1 | $< 10^{-16}$ | $< 10^{-16}$ | **Reject $H_0$** |
| **Middle Ager (y0 vs y1)** | +0.0233 | 0.0023 | [+0.0189, +0.0276] | 107.405 | 1 | $< 10^{-16}$ | $< 10^{-16}$ | **Reject $H_0$** |
| **Elderly (y0 vs y1)** | +0.0056 | 0.0045 | [-0.0038, +0.0151] | 1.529 | 1 | 0.2163 | 0.2163 | Fail to Reject |
| **Standardized Overall** | +0.0290 | 0.0020 | [+0.0252, +0.0329] | 201.310 | 1 | $< 10^{-16}$ | $< 10^{-16}$ | **Reject $H_0$** |

| Cohort | Female Mean $A_{y0}$ | Male Mean $A_{y1}$ | Difference $\hat{\Delta}_A$ (bpm) | $\text{SE}_{\text{HC3}}$ | 95% CI (bpm) | Wald $\chi^2$ | $p$-value |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Young Adults** | 1.850 bpm | 1.440 bpm | +0.415 bpm | 0.056 | [+0.306, +0.525] | 55.54 | $9.16 \times 10^{-14}$ |
| **Middle Ager** | 1.384 bpm | 1.302 bpm | +0.082 bpm | 0.021 | [+0.041, +0.124] | 15.01 | $1.07 \times 10^{-4}$ |
| **Elderly** | 1.146 bpm | 1.110 bpm | +0.037 bpm | 0.050 | [-0.061, +0.134] | 0.55 | 0.4592 |
| **Standardized Overall** | 1.427 bpm | 1.307 bpm | +0.141 bpm | 0.019 | [+0.103, +0.178] | 53.73 | $2.30 \times 10^{-13}$ |

*Sensor wear controls:* Daily occupancy rate ($62.97\%$ F vs $63.17\%$ M, $p=0.759$) and nocturnal coverage ($63.02\%$ F vs $59.00\%$ M, $p=0.0793$) do not differ significantly between young adults. Adjusting for occupancy, nocturnal fraction, and calendar span yields $\hat{\Delta} = +0.0642$ ($\text{SE} = 0.0063$), accounting for $<1.1\%$ of the contrast magnitude.

*Two-component continuous null recalibration:* Adding slow baseline persistence ($\phi_2 \in \{0.980, 0.995\}$) via REML covariance $\Sigma_{jk} = s_1 \phi_1^{|t_k - t_j|} + s_2 \phi_2^{|t_k - t_j|}$ preserves the Young Adult contrast: $\hat{\Delta} = +0.0645$, $\text{SE} = 0.0060$, Wald $\chi^2 = 115.58$, $p < 10^{-25}$. Male empirical rejection rates at $\alpha=0.05$ remain at $4.6\%\text{--}5.5\%$.

*Window relaxation sensitivity ($k=10$, $N_{\text{obs}}=60$):* Retains $N=5,516$ Apple participants (+19.1%; Young F $n=292$, +30.4%). Young Adult contrast remains $\hat{\Delta} = +0.0526$ ($\text{SE} = 0.0061$, Wald $\chi^2 = 75.49, p < 10^{-17}$). Sources: [standardized report](../experiments/female_infradian_cycles_v2/results/REPORT.md), [calibrated metrics](../experiments/female_infradian_cycles_v2/results/calibrated_group_contrasts.csv).

### 3. Multi-year stability, circular acrophase, and metabolic invariance

**Cohorts:** Apple Watch longitudinal participants with continuous wear history $\ge 365\text{ days}$ ($N=1,323$; Young Adult F $N=324$) and reproductive-age females with valid BMI records ($N=502$).

**Longitudinal stability (90-day sliding window, 15-day step):** Dominant period $T^*_w = \arg\max_{T \in [14, 56]} P_{LS, w}(1/T)$ tracked across valid windows ($N_w \ge 45\text{ days}$). Period drift is $\Delta T_{\text{drift}} = \text{IQR}(T^*_w)$. Infradian persistence ratio $\Pi_{\text{target}}$ measures the fraction of windows with $T^*_w \in [25.0, 35.0]\text{ days}$.

| Demographic cohort | Evaluation $N$ | Median period drift $\text{IQR}(T^*)$ | Target persistence $\Pi_{\text{target}}$ | Wear occupancy persistence control |
|---|---:|---:|---:|---:|
| Young adult females | 324 | 10.00 days | 35.0% | 19.8% |
| Young adult males | 785 | 18.00 days | 21.2% | 21.1% |
| Elderly females | 214 | 18.25 days | 24.0% | 20.2% |

*Circular acrophase uniformity:* Relative cosinor acrophase projected to calendar epoch coordinate: $\Phi_{\text{cal}, i} = (\phi_{\text{rel}, i} - \frac{2\pi}{T^*_i} \text{date}_0^{(i)}) \pmod{2\pi}$. Analytical Rayleigh test evaluated on reproductive females ($N=502$):
- Calendar epoch phase $\Phi_{\text{cal}}$ (`hr_p10`): Rayleigh $Z = 0.1391$, Mardia-Jupp $p = 0.8703$ (fails to reject circular uniformity, ruling out seasonal or calendar entrainment).
- Nocturnal HR acrophase (`night_hr`): Rayleigh $Z = 0.9791$, $p = 0.3759$.
- Daily wear occupancy control (`occ_daily`): Rayleigh $Z = 0.0514$, $p = 0.9499$.
- 28-day reference cycle modulo $\theta_{28}$ (`hr_p10`): Rayleigh $Z = 3.6324$, nominal $p = 0.0264$.

*Metabolic invariance across WHO BMI tiers:* Evaluated on reproductive females ($N=502$: Underweight $N=19$, Normal weight $N=296$, Overweight $N=123$, Obese $N=64$):
- Omnibus Kruskal-Wallis test on resting HR amplitude ($A_{10}$): $H = 1.8261$, df = 3, $p = 0.6093$.
- Omnibus Kruskal-Wallis test on cycle period ($T^*$): $H = 3.2462$, df = 3, $p = 0.3552$.
- Multivariable ANCOVA with HC3 standard errors (adjusting for mean daily occupancy and observation span, Normal weight reference):
  - Overweight effect: $\hat{\beta} = -0.0566\text{ bpm}$, $\text{SE}_{\text{HC3}} = 0.0677$, $p = 0.4037$.
  - Obese effect: $\hat{\beta} = -0.0365\text{ bpm}$, $\text{SE}_{\text{HC3}} = 0.0898$, $p = 0.6844$.
  - All BMI contrasts exceed $p > 0.40$; infradian amplitude is invariant to adiposity. Source: [extensions report](../experiments/female_infradian_extensions/results/REPORT.md).

### 4. Cross-device replication on Garmin Tier 1c ($N=3,760$)

**Cohort:** Garmin Tier 1c single-source users ($N=4,410$ total manifest, $N=3,760$ qualified at $k=15$, $N_{\text{obs}}=90$: Young F 299, Young M 461, Middle F 938, Middle M 1,625, Elderly F 102, Elderly M 335).

**Protocol:** Identical 180-day standardized exposure, Frisch-Waugh-Lovell harmonic scoring, and two-component continuous REML null bootstrap calibration ($B=199$).

| Hypothesis / Contrast | Point estimate $\hat{\Delta}E$ | $\text{SE}_{\text{HC3}}$ | 95% Confidence Interval | Wald Statistic ($\chi^2$) | df | $p_{\text{Holm}}$ | Rejection ($\alpha=0.05$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Omnibus Age $\times$ Sex** | — | — | — | 50.89 | 2 | $1.81 \times 10^{-11}$ | **Reject $H_0$** |
| **Young Adults (y0 − y1)** | +0.0474 | 0.0053 | [+0.0371, +0.0577] | 81.13 | 1 | $4.21 \times 10^{-18}$ | **Reject $H_0$** |
| **Middle Ager (y0 − y1)** | +0.0350 | 0.0028 | [+0.0296, +0.0404] | 161.72 | 1 | $9.63 \times 10^{-36}$ | **Reject $H_0$** |
| **Elderly (y0 − y1)** | -0.0013 | 0.0051 | [-0.0113, +0.0087] | 0.06 | 1 | 0.8050 | Fail to Reject |
| **Standardized Overall** | +0.0333 | 0.0022 | [+0.0289, +0.0377] | 221.20 | 1 | $4.87 \times 10^{-49}$ | **Reject $H_0$** |

*Multiplicity control and false-alarm rates:* At $\alpha = 0.05$, Garmin young females reject the null in $21.7\%$ of participants ($n=299$), while young male negative controls reject in $5.6\%$ ($n=461$), calibrating tightly to nominal $5.0\%$.

*Spectral deconstruction of 30–120 day male red noise:* Continuous harmonic sweeps across 229 frequencies (3 to 120 days) resolve the apparent male long-period power rise:
- Garmin males ($N=2,086$, ages 18–64) display a monotonic rise in spectral power beyond 30 days. Single AR(1) nulls plateau and fail to capture this continuum.
- The two-component continuous null ($\phi_2 \approx 0.98\text{--}0.995$) fully tracks the male trajectory across the entire 3–120 day continuum (mean residual $<0.003$).
- Higher-order polynomial calendar detrending ($t^2, t^3$) reduces $>60\text{ d}$ male power by $31\%$, proving that macro-seasonal drift accounts for the tail.
- In contrast, female $27.7\text{ d}$ biological resonance is completely invariant to cubic detrending ($\Delta R^2 = +0.0281$), dropping to zero beyond 35 days. Source: [Garmin replication report](../experiments/garmin_infradian_replication/REPORT.md).

## Monthly periodicity downstream classification

**Cohort:** Apple Watch Allocation 0 participants passing the standardized 180-day exposure gate ($N=4,631$ total eligible). Fixed split: 3,211 training, 695 validation, 725 held-out test participants (224 female, 501 male).

**Predictors:** Base representations ($H_{\text{fill}}$, 19,121 features: 305 static circadian + 18,816 MultiRocket sequence features), observation controls ($Q$, 20 features: nonperiodic resting HR mean/SD/trend, four 6-hour clock coverage fractions, total span, longest gap, mask cosinor features), monthly infradian descriptors ($R_{\text{obs}}$, 3 features: amplitude $A$, fit score $S$, best-fit period $T \in [25, 35]\text{ d}$), and categorical age indicators (3 features).

**Evaluation:** Ridge classifier with alpha selected from $\{100, 500, 1000, 5000, 10000, 100000\}$ via validation AUROC. Evaluated on the 725 held-out test participants. Pointwise 95% confidence intervals and standard errors computed from 2,000 stratified participant bootstrap resamples.

| Head | Description | Features ($d$) | Selected $\alpha$ | Validation AUROC | Test AUROC | Test AP | 95% Bootstrap CI | Bootstrap SE |
|---|---|:---:|:---:|---:|---:|---:|:---:|:---:|
| $H_{\text{clock}}$ | MultiRocket + Static Circadian (Clock Fill) | 19,121 | 1,000 | 0.8422 | 0.8288 | 0.9018 | [0.7954, 0.8610] | 0.0165 |
| $B_{\text{clock}}$ | $H_{\text{clock}} + Q + \text{Age}$ (Primary Baseline) | 19,144 | 1,000 | 0.8437 | 0.8309 | 0.9026 | [0.7980, 0.8627] | 0.0165 |
| $B_{\text{clock}} + R_{\text{obs}}$ | Primary Infradian-Augmented Arm | 19,147 | 1,000 | 0.8505 | **0.8418** | 0.9071 | [0.8097, 0.8724] | 0.0160 |
| $B_{\text{linear}}$ | $H_{\text{linear}} + Q + \text{Age}$ (Linear Sensitivity Baseline) | 19,144 | 5,000 | 0.8450 | 0.8547 | 0.9198 | [0.8242, 0.8829] | 0.0150 |
| $B_{\text{linear}} + R_{\text{obs}}$ | Linear Sensitivity Augmented Arm | 19,147 | 5,000 | 0.8517 | **0.8607** | 0.9217 | [0.8304, 0.8881] | 0.0148 |
| $B_{\text{clock}}$ (no age) | $H_{\text{clock}} + Q$ (Unconditional Baseline) | 19,141 | 1,000 | 0.8431 | 0.8291 | 0.9015 | [0.7959, 0.8607] | 0.0165 |
| $B_{\text{clock}}$ (no age) $+ R_{\text{obs}}$ | Unconditional Augmented Arm | 19,144 | 1,000 | 0.8494 | 0.8393 | 0.9054 | [0.8073, 0.8702] | 0.0162 |
| Age only | Categorical Age One-Hot (Diagnostic) | 3 | 100,000 | 0.5274 | 0.5629 | 0.7255 | [0.5629, 0.5629] | 0.0000 |

| Contrast | Estimand Formulation | Point Estimate | Bootstrap SE | 95% Pointwise CI | 97.5% Bonferroni CI | Statistically Significant |
|---|---|:---:|:---:|:---:|:---:|:---:|
| $\delta_{\text{clock}}$ | $\text{AUROC}(B_{\text{clock}} + R_{\text{obs}}) - \text{AUROC}(B_{\text{clock}})$ | **+0.010889** | 0.003741 | [+0.003555, +0.018294] | [+0.002548, +0.019258] | **Yes** ($p < 0.01$) |
| $\delta_{\text{linear}}$ | $\text{AUROC}(B_{\text{linear}} + R_{\text{obs}}) - \text{AUROC}(B_{\text{linear}})$ | **+0.006050** | 0.002684 | [+0.000713, +0.011178] | [+0.000205, +0.011967] | **Yes** ($p < 0.05$) |
| $\delta_{\text{no\_age}}$ | $\text{AUROC}(B_{\text{no\_age}} + R_{\text{obs}}) - \text{AUROC}(B_{\text{no\_age}})$ | **+0.010176** | 0.003745 | [+0.002860, +0.017679] | [+0.001888, +0.018892] | **Yes** ($p < 0.01$) |

*Age subgroup breakdown:* Test AUROC within Young Adults ($N=141$) improves from $0.86283$ ($B_{\text{clock}}$) to **$0.87769$** ($B_{\text{clock}} + R_{\text{obs}}$, lift $+0.01486$), and from $0.89633$ ($B_{\text{linear}}$) to **$0.90708$** ($B_{\text{linear}} + R_{\text{obs}}$, lift $+0.01075$). Training-age-weighted AUROC ($\sum_a w_a \text{AUROC}_a$) increases by $+0.01105$ under clock filling and $+0.00704$ under linear filling. Source: [downstream report](../experiments/monthly_periodicity_downstream/results/REPORT.md).

## Apple fill and mask representations across recording patterns

**Cohort:** Apple single-source cohort Allocation 0 ($N=11,139$ total: 7,797 train, 1,671 validation, 1,671 test). **Target:** recorded administrative salutation (`10 → 0/F`, `20 → 1/M`). Evaluated across eight feature arms, two resolutions (5m and 15m), and two history budgets (`all_days` and `up_to_40d`) using Ridge regression with validation-selected extended alpha grid (`ridge_selected_ext`).

| Arm | Description | All days 5m | All days 15m | 5m − 15m | Up to 40d 5m | Up to 40d 15m |
|---|---|---:|---:|---:|---:|---:|
| `H_clock` | Observed HR retained; gaps filled with training clock-bin medians | 0.8678 | 0.8448 | +0.0229 | 0.8168 | 0.7972 |
| `H_global` | Same feature blocks filled with single training global median | 0.8586 | 0.8438 | +0.0148 | 0.8140 | 0.8059 |
| `H_clean_diff` | Clock-filled base; difference branch is zero across missing gaps | 0.8718 | 0.8538 | +0.0180 | 0.8247 | 0.7992 |
| `H_short_interp` | Short linear interpolation ($\le 30\text{ min}$); clock medians elsewhere | 0.8727 | 0.8504 | +0.0223 | 0.8246 | 0.8026 |
| `M_static` | Across-day mask mean and SD profiles ($S(M)$), no HR values | 0.6466 | 0.6282 | +0.0185 | 0.6342 | 0.6131 |
| `M_temporal` | Mask profiles + MultiRocket sequence features on binary mask ($T(M, \Delta M)$) | 0.7459 | 0.7280 | +0.0180 | 0.7107 | 0.6843 |
| `H_plus_M_concat` | Joint Ridge fit on concatenated HR and mask sequence features | 0.8723 | 0.8517 | +0.0206 | 0.8284 | 0.8122 |
| `H_plus_M_stack` | Convex score blend ($w=0.20$) of `H_clock` and `M_temporal` | **0.8749** | 0.8517 | +0.0232 | 0.8255 | 0.8097 |

| Occupancy tercile | Test N | Clock 5m AUROC | Clock 15m AUROC | Global − clock [95% CI] | Stack − clock [95% CI] |
|---|---:|---:|---:|---:|---:|
| Low (<47.4% valid bins) | 575 | 0.8581 | 0.8530 | +0.0092 [-0.0115, +0.0287] | +0.0108 [+0.0033, +0.0194] |
| Middle (47.4%–62.5%) | 580 | 0.8721 | 0.8316 | -0.0198 [-0.0413, +0.0016] | +0.0063 [-0.0028, +0.0151] |
| High (>62.5% valid bins) | 516 | 0.8616 | 0.8365 | -0.0184 [-0.0427, +0.0071] | +0.0038 [-0.0073, +0.0170] |

*Key empirical findings:*
1. Observation patterns alone (`M_temporal`) achieve test AUROC **0.7459** (vs. 0.6466 for static mask `M_static`, increment $+0.0993$), demonstrating that temporal wear dynamics discriminate recorded salutation without physiological HR values.
2. Stacking HR and temporal mask scores (`H_plus_M_stack`) reaches **0.8749** on all days 5m (increment $+0.0071$ over `H_clock`).
3. Clock filling outperforms global filling overall by $+0.0092$ at 5m, though global filling exhibits a point-estimate crossover in the low-occupancy tercile ($+0.0092$ vs. $-0.0198$ in middle occupancy). Source: [fill and mask report](../experiments/apple_fill_and_mask_by_recording/results/REPORT.md).

## Circular diurnal template regularization

**Problem formulation:** Discretized diurnal cycle $\mathbb{T}_{288} = \mathbb{Z}/288\mathbb{Z}$ ($L=288$ five-minute bins). Standard 1D Gaussian filtering induces boundary truncation and slope discontinuities across midnight ($23:55 \to 00:00$). Periodic circular convolution $(x \circledast K_\sigma)[t] = \sum_{k=0}^{287} x[(t-k) \pmod{288}] K_\sigma[k]$ enforces $C^\infty$ wraparound boundary conditions with kernel bandwidth $\sigma$ (in bins).

**Imputation fidelity benchmark:** Evaluated on $N=200$ test participants sampled from Apple Allocation 0 test set, stratified across occupancy tiers. Exactly 11,900 contiguous artificial gaps were injected into verified observed segments (short: 1–3 bins / 5–15 min, $N=4,000$; medium: 6–12 bins / 30–60 min, $N=3,980$; long: 24–48 bins / 2–4 h, $N=3,920$), totaling 59,500 evaluations.

| Imputation Method | Bandwidth parameter $\sigma$ | MAE (bpm) | MAE SEM | RMSE (bpm) | RMSE SEM | Boundary Jump $\Delta$ (bpm) | Jump SEM |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| $H_{\text{linear}}$ (Linear Baseline) | Segment endpoints | **7.085** | 0.080 | **8.224** | 0.088 | **1.942** | 0.041 |
| $H_{\text{clock}}$ (Raw Median Template) | $\sigma = 0$ bins | 8.625 | 0.083 | 9.650 | 0.088 | 8.664 | 0.091 |
| $H_{\text{circ\_gauss\_s1}}$ ($\sigma^*$, Recommended) | $\sigma = 1\text{ bin}$ ($5\text{ min}$) | 8.640 | 0.083 | **9.628** | 0.088 | **8.637** | 0.091 |
| $H_{\text{circ\_gauss\_s2}}$ | $\sigma = 2\text{ bins}$ ($10\text{ min}$) | 8.659 | 0.083 | 9.639 | 0.088 | 8.654 | 0.091 |
| $H_{\text{circ\_gauss\_s4}}$ | $\sigma = 4\text{ bins}$ ($20\text{ min}$) | 8.686 | 0.083 | 9.661 | 0.088 | 8.682 | 0.091 |

| Gap length stratum | Gap count | $H_{\text{clock}}$ RMSE | $H_{\text{circ\_gauss\_s1}}$ RMSE | $H_{\text{clock}}$ Boundary Jump | $H_{\text{circ\_gauss\_s1}}$ Boundary Jump |
|---|---:|---:|---:|---:|---:|
| Short (1–3 bins, 5–15 min) | 4,000 | 8.771 bpm | **8.769 bpm** | 8.443 bpm | **8.414 bpm** |
| Medium (6–12 bins, 30–60 min) | 3,980 | 10.053 bpm | **10.019 bpm** | 8.848 bpm | **8.812 bpm** |
| Long (24–48 bins, 2–4 hours) | 3,920 | 10.138 bpm | **10.109 bpm** | 8.702 bpm | **8.688 bpm** |

*Downstream predictive impact:* On Apple Allocation 0 test participants ($N_{\text{test}}=1,671$), regularizing median templates with $\sigma^*=1$ bin yields $\Delta\text{AUROC} = +0.000395$ (primary $d=595$, $p=0.600$), with positive buffering in low-occupancy participants ($<50\%$ valid bins, $N=789$, $\Delta\text{AUROC} = +0.001817$). Larger bandwidths ($\sigma \ge 2$) oversmooth morning awakening peaks and nocturnal troughs, monotonically increasing error. Source: [circular regularization report](../experiments/apple_circular_diurnal_regularization/results/REPORT.md).

## Apple-Garmin cross-device transfer pilot

**Cohorts:** Apple full cohort ($N=10,214$ eligible with $\ge 40$ adequate days; test $N=1,530$) and Garmin binning pilot ($N=979$ eligible; test $N=148$). Evaluated on earliest 40 adequate calendar days per participant using Ridge classification with alpha selected from $\{100, 500, 1000, 5000, 10000\}$ via source validation.

**Estimand:** Target-cohort test AUROC of the imported source-fitted score and its paired difference from a locally fitted score on identical target test participants.

| Target ← imported source | Feature block ($d$) | Local target AUROC | Imported target AUROC | Imported − local [95% CI] |
|---|---|---:|---:|---:|
| Apple ← Garmin | HR summaries (10) | 0.5947 | 0.5405 | −0.0542 [−0.0780, −0.0311] |
| Apple ← Garmin | Recording summaries (30) | 0.6328 | 0.4935 | −0.1393 [−0.1847, −0.0901] |
| Apple ← Garmin | Combined (41) | 0.6596 | 0.5583 | −0.1013 [−0.1319, −0.0702] |
| Garmin ← Apple | HR summaries (10) | 0.7811 | 0.7114 | −0.0697 [−0.1399, +0.0065] |
| Garmin ← Apple | Recording summaries (30) | 0.5162 | 0.4768 | −0.0394 [−0.2004, +0.1143] |
| Garmin ← Apple | Combined (41) | 0.7595 | 0.6617 | −0.0977 [−0.2017, +0.0032] |

| Target ← imported source (Size-matched source) | Feature block ($d$) | Local target AUROC | Imported target AUROC | Imported − local [95% CI] |
|---|---|---:|---:|---:|
| Apple ← Garmin (size-matched) | HR summaries (10) | 0.5654 | 0.5405 | −0.0248 [−0.0537, +0.0034] |
| Apple ← Garmin (size-matched) | Recording summaries (30) | 0.6007 | 0.4935 | −0.1072 [−0.1535, −0.0560] |
| Apple ← Garmin (size-matched) | Combined (41) | 0.6250 | 0.5583 | −0.0666 [−0.1025, −0.0313] |
| Garmin ← Apple (size-matched) | HR summaries (10) | 0.7811 | 0.6952 | −0.0859 [−0.1430, −0.0350] |
| Garmin ← Apple (size-matched) | Recording summaries (30) | 0.5162 | 0.4439 | −0.0723 [−0.1929, +0.0506] |
| Garmin ← Apple (size-matched) | Combined (41) | 0.7595 | 0.5575 | −0.2019 [−0.2924, −0.1075] |

*Key transfer conclusions:* Physiological HR summary representations transfer ranking signal across device ecosystems (imported HR AUROC $0.7114$ on Garmin, $0.5405$ on Apple). Conversely, recording summaries exhibit device-specific calibration failure and zero transportability across manufacturer hardware (imported AUROC $0.4768\text{--}0.4935$, indistinguishable from chance). Source: [cross-device transfer report](../experiments/apple_garmin_transfer_pilot/REPORT.md).

## Infradian age dependency protocol (implementation handoff)

**Document status:** Implementation handoff protocol (dated 2026-10-20; execution pending). Complete technical specifications reside in [`experiments/infradian_age_dependency/PLAN.md`](../experiments/infradian_age_dependency/PLAN.md).

**Objective:** Estimate whether the female monthly HR periodogram peak weakens, shifts, broadens, or concentrates into fewer participants with advancing chronological age, analyzing Apple (source 6), Garmin (source 3), and Samsung (source 7) separately.

### 1. Protocols and candidate feasibility

Window length is standardized to 180 calendar days $[s, s+180)$ partitioned into six 30-day blocks:
- `expanded90` (Primary): Every block contains $\ge 10$ qualified dates ($\ge 120$ occupied 5-minute bins); total window contains $\ge 90$ qualified dates. Quotas are initialized to $q_b = 10$, and 30 remaining dates are allocated to blocks with smallest quota below capacity. Exactly $N_{\text{obs}} = 90$ dates are sampled deterministically via PCG64 SHA-256 seeding.
- `balanced90` (Reference): Every block contains $\ge 15$ qualified dates; exactly 15 sampled per block ($N_{\text{obs}} = 90$).
- `balanced60` (Apple sensitivity): Every block contains $\ge 10$ qualified dates; exactly 10 sampled per block ($N_{\text{obs}} = 60$).

| Source and rule | Window-eligible participants | Female participants ($y=0$) |
|---|---:|---:|
| Apple, `balanced90` (15/block) | 4,631 | 1,478 |
| Apple, `expanded90` (10/block, 90 total) | 5,390 | 1,765 |
| Apple, `balanced60` (10/block, 60 total) | 5,518 | 1,820 |
| Garmin, `balanced90` (15/block) | 3,760 | 1,339 |
| Garmin, `expanded90` (10/block, 90 total) | 3,832 | 1,376 |
| Garmin, `balanced60` (10/block, 60 total) | 3,840 | 1,380 |

*Additional candidates:* 405 additional female candidates with $\ge 90$ recorded channel-3000 dates identified outside current single-source caches (399 multisource, 6 single-source). Samsung contains 541 female candidates with $\ge 90$ recorded dates. Actual window qualification must be evaluated from raw event bins.

### 2. Mathematical specification

1. **Age transformation:** Integral birth year parsed from metadata; age at window start evaluated under the December-1 convention:
   $$\text{age} = \text{start.year} - \text{birth\_year} - 1 + \mathbb{I}(\text{start.month} = 12)$$
   Normalized age predictor is $a = (\text{age} - 45) / 10$ (centered at 45 years, decade scale).
2. **Primary spectral estimand ($Q_i$):** Broad frequency grid $\{k/720 : k \in \{12, \dots, 240\}\} \cup \{1/35, 1/25\}$ cycles/day. Signed spectral excess over two-component continuous REML null:
   $$D_i(f) = P_i(f) - \frac{1}{B} \sum_{b=1}^B P_{i, \text{null}, b}(f)$$
   Band-integrated excess score $Q_i$ evaluated via numerical trapezoidal quadrature across the monthly frequency band:
   $$Q_i = \frac{1}{1/25 - 1/35} \int_{1/35}^{1/25} D_i(f) df$$
3. **Statistical model:** Unpenalized quadratic age OLS model:
   $$Q_i = \beta_0 + \beta_1 a_i + \beta_2 a_i^2 + \gamma_1 \tilde{c}_{\text{occ}, i} + \gamma_2 \tilde{c}_{\text{night}, i} + \gamma_3 \tilde{c}_{\text{span}, i} + \varepsilon_i$$
   with heteroskedasticity-consistent HC3 standard errors.
4. **Support and reporting rules:** Female age curves are displayed only across five-year age bands containing $\ge 30$ women. Sex interaction models ($a \times \text{salutation}$) are fitted strictly on shared-support bands with $\ge 30$ participants in each salutation group (minimum 3 shared bands).

## Earlier summary and feature experiments

| Experiment | Cohort and evaluation | Saved result |
|---|---|---|
| [Pooled v1](../experiments/artifacts/model_report.md) | N=20,485; one 80/10/10 split | Test AUROC 0.7132 |
| [Pooled v2](../experiments/artifacts_v2/model_report.md) | N=20,423; one 80/10/10 split | Test AUROC 0.6833 |
| [Source-specific default RF](../experiments/artifacts_sq/sidequest_report.md) | Strict cohorts; one split per source | `win` arm: Garmin 0.6918; Apple 0.6814; Samsung 0.6609 |
| [Informative recording](../experiments/2026-09-20_informative_recording_features/README.md) | Ungated N=25,784; 30 repeated splits | Recording + values 0.7239; paired gain over native-NaN values +0.0167 |
| [Balanced-allocation R1a](../experiments/r1a_allocfolds_garmin_2026-09-21/results/REPORT.md) | Strict Garmin; ten allocations | BASE 0.7415; BASE + R1a 0.7509 |
| [Date-aware R1b](../experiments/r1b_dateaware_garmin_2026-09-21/results/REPORT.md) | Strict Garmin; ten allocations | Day-level A40 0.6589; C40 0.6782; participant BASE + P40 0.7513 |
| [Recording controls](../experiments/recording_structure_garmin/results/REPORT.md) | Strict Garmin; first 40 adequate days; three allocations | Mask-only temporal model 0.6166; count-only temporal model 0.6332 |
| [Filling controls](../experiments/filling_sensitivity_garmin/results/REPORT.md) | Strict Garmin; first 40 adequate days; three allocations | Four fitted pipelines: mean test AUROC 0.8464–0.8485 |

Pooled v1 and v2 change both cohort and predictors. The source-specific `win` scores describe one named arm, not the best arm for each source. R1b BASE contains longer-history features; BASE + P40 is not a 40-day-only model.

The [experiment index](../learnings/experiment_lineage_worktree.md) also links RF tuning, feature selection, feature engineering, age, day-level aggregation, and the broader 50/50 source experiment.

## Scientific provenance and compiled reports

Authoritative experimental investigations are compiled into publication-grade reference PDF documents and cataloged in [`reports/scientific_provenance_pdfs/CATALOG.md`](scientific_provenance_pdfs/CATALOG.md).

| Category | Date | Source Experiment | Compiled PDF Document |
|:---|:---:|:---|:---|
| **01 Exploratory Discovery** | 2026-10-20 | [`female_infradian_cycles`](../experiments/female_infradian_cycles/results/REPORT.md) | [`01_exploratory_discovery/2026-10-20_female-infradian-cycles_exploratory-spectral-discovery.pdf`](scientific_provenance_pdfs/01_exploratory_discovery/2026-10-20_female-infradian-cycles_exploratory-spectral-discovery.pdf) |
| **02 Calibrated Hypothesis Testing** | 2026-10-20 | [`female_infradian_cycles_v2`](../experiments/female_infradian_cycles_v2/results/REPORT.md) | [`02_calibrated_hypothesis_testing/2026-10-20_female-infradian-cycles-v2_calibrated-hypothesis-testing.pdf`](scientific_provenance_pdfs/02_calibrated_hypothesis_testing/2026-10-20_female-infradian-cycles-v2_calibrated-hypothesis-testing.pdf) |
| **03 Confirmatory Mechanisms** | 2026-10-20 | [`female_infradian_extensions`](../experiments/female_infradian_extensions/results/REPORT.md) | [`03_confirmatory_mechanisms/2026-10-20_female-infradian-extensions_confirmatory-mechanisms.pdf`](scientific_provenance_pdfs/03_confirmatory_mechanisms/2026-10-20_female-infradian-extensions_confirmatory-mechanisms.pdf) |
| **04 Predictive Utility Benchmarks** | 2026-10-20 | [`monthly_periodicity_downstream`](../experiments/monthly_periodicity_downstream/results/REPORT.md) | [`04_predictive_utility_benchmarks/2026-10-20_monthly-periodicity-downstream_predictive-utility-benchmark.pdf`](scientific_provenance_pdfs/04_predictive_utility_benchmarks/2026-10-20_monthly-periodicity-downstream_predictive-utility-benchmark.pdf) |
| **05 Imputation and Regularization** | 2026-10-20 | [`apple_fill_and_mask_by_recording`](../experiments/apple_fill_and_mask_by_recording/results/REPORT.md) | [`05_imputation_and_regularization/2026-10-20_apple-fill-and-mask-by-recording_imputation-architecture.pdf`](scientific_provenance_pdfs/05_imputation_and_regularization/2026-10-20_apple-fill-and-mask-by-recording_imputation-architecture.pdf) |
| **05 Imputation and Regularization** | 2026-10-20 | [`apple_circular_diurnal_regularization`](../experiments/apple_circular_diurnal_regularization/results/REPORT.md) | [`05_imputation_and_regularization/2026-10-20_apple-circular-diurnal-regularization_template-regularization.pdf`](scientific_provenance_pdfs/05_imputation_and_regularization/2026-10-20_apple-circular-diurnal-regularization_template-regularization.pdf) |
| **06 Cross-Cohort and Scaling** | 2026-10-19 | [`apple_garmin_transfer_pilot`](../experiments/apple_garmin_transfer_pilot/REPORT.md) | [`06_cross_cohort_and_scaling/2026-10-19_apple-garmin-transfer-pilot_device-generalization.pdf`](scientific_provenance_pdfs/06_cross_cohort_and_scaling/2026-10-19_apple-garmin-transfer-pilot_device-generalization.pdf) |
| **06 Cross-Cohort and Scaling** | 2026-10-14 | [`apple_cohort_scaling`](../experiments/apple_cohort_scaling/REPORT.md) | [`06_cross_cohort_and_scaling/2026-10-14_apple-cohort-scaling_full-cohort-benchmark.pdf`](scientific_provenance_pdfs/06_cross_cohort_and_scaling/2026-10-14_apple-cohort-scaling_full-cohort-benchmark.pdf) |
| **06 Cross-Cohort and Scaling** | 2026-10-20 | [`garmin_infradian_replication`](../experiments/garmin_infradian_replication/REPORT.md) | [`06_cross_cohort_and_scaling/2026-10-20_garmin-infradian-replication_cross-device-and-spectral.pdf`](scientific_provenance_pdfs/06_cross_cohort_and_scaling/2026-10-20_garmin-infradian-replication_cross-device-and-spectral.pdf) |
