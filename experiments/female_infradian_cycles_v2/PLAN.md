# Age-accounted validation of 25–35-day structure in observed heart rate

Date: 2026-10-20. Status: protocol only. Feasibility EDA is complete; implementation, spectral analyses, null simulations, group inference and temporal prediction below have **not run**. The v1 cohort and outcomes have already been explored, so v2 is exploratory validation, not a pristine confirmatory study.

## 1. Question, population and existing evidence

Does observed daily HRp10 contain reproducible excess 25–35-day harmonic structure, and does the recorded-salutation contrast vary across age categories after controlling observation geometry and calibrating period search against serially dependent noise?

The target grouping is administrative salutation: 10 → y0 and 20 → y1. Age is categorical: `Young Adults`, `Middle Ager`, `Elderly`. Neither exact age nor menstrual events, ovulation, hormone use or menopause status is available in these artifacts. No category is treated as a biological reproductive state. The estimand concerns observed HR, not unobserved resting physiology.

Inputs are the v1 Apple source-6/channel-3000 participant table and original 5-minute caches. [EDA.md](EDA.md) records demographics, missingness, overlap and provenance; [WINDOW_FEASIBILITY.md](WINDOW_FEASIBILITY.md) describes actual-date checks. Descriptive findings motivate the design without validating v1 significance:

| Existing observation | Saved-data estimate |
|---|---:|
| Selected HRp10 amplitude mean difference y0−y1, young / middle / elderly | +0.2698 / +0.0859 / +0.0042 bpm |
| Selected amplitude versus qualified-day count, Spearman ρ | −0.463 |
| Selected ΔR² versus qualified-day count, Spearman ρ | −0.564 |
| HRp10 best period at 25 or 35-day boundary | 20.24% |
| Exact LS/cosinor HRp10 period agreement | 96.3% |

LS and cosinor use related harmonic representations on the same observations; agreement is not independent replication. Larger selected amplitudes with shorter histories motivate exposure control, not a causal explanation.

## 2. Freeze calendar exposure before inspecting v2 outcomes

A qualified date has at least 120 finite 5-minute bins in the original cache. This is occupied-bin coverage, not verified ten-hour wear. Candidate window starts are qualified dates in chronological order. A qualifying window is `[start, start+180)` calendar days, partitioned into six successive half-open 30-day blocks. Each block must contain at least 15 qualified dates. Select the first qualifying window without using HR magnitudes, salutation or fitted periods.

Within each block, select exactly 15 qualified dates uniformly without replacement, using sorted dates and a reproducible participant-specific stream derived from seed 20261020 and the numeric participant ID. Save the precise RNG specification before execution. This gives **90 observations in a common 180-day design**. Actual first-to-last selected span can be shorter than 180 days and must be reported. All primary representations, adjustments and nulls reuse these dates and original masks.

Actual-date feasibility among the 8,425 v1-screened participants is:

| Age category | First-window y0 / y1 | Two-window y0 / y1 |
|---|---:|---:|
| Young Adults | 224 / 666 | 106 / 348 |
| Middle Ager | 1082 / 2086 | 503 / 1081 |
| Elderly | 172 / 401 | 91 / 195 |
| Total | 4631 | 2324 |

The primary target population is these first-window eligible participants, not all Apple users. Young y0 retention is 44.6% for one window and 21.1% for two, relative to the v1-screened group. Coverage protocols requiring 10 or 20 dates per block are prespecified sensitivities with different populations and 60 or 120 selected dates; do not pool them or choose the most favorable result.

Save candidate/selected windows, all qualified and selected dates, daily occupied-bin counts, four clock-coverage fractions, calendar gaps and selection flags. For every secondary signal save its actual finite dates, count and span. Original cached span is not selected-signal span.

## 3. Age-specific and standardized estimands

For participant endpoint E defined below, estimate three contrasts:

`Δ_a = E(E_i | y0, age=a) − E(E_i | y1, age=a)`.

Fit participant-level OLS with categorical age, salutation and their interaction. The adjusted two-degree-of-freedom age×salutation omnibus and the three adjusted within-age salutation contrasts form **one family of four HC3 robust Wald tests**, with Holm adjustment across all four at α=0.05. The omnibus remains the main scientific question; it has no separate uncorrected significance claim. Report effect estimates regardless of the adjusted test outcomes. Unadjusted within-age contrasts are descriptive/secondary, without selecting whichever analysis gives the smallest p-value. Days are not independent participants.

Use 1,000 participant bootstrap resamples, stratified by age×salutation, for 95% contrast intervals. These intervals are **pointwise, not familywise**; all four hypothesis tests use HC3 Wald p-values and the shared Holm rule. Refit group models each draw; retain fixed age weights. Null-calibration uncertainty conditional on the available B simulations must be disclosed.

Before outcome analysis, set `w_a = N_a / 4631` from pooled primary eligibility; the descriptive/secondary standardized contrast is `Δ_std = Σ_a w_a Δ_a`. Freeze observation adjustment to mean selected-day occupancy, mean selected night-clock coverage and selected-date span; center within pooled age strata and evaluate adjusted salutation contrasts at the same pooled covariate distribution within each age. Report overlap and extrapolation. These adjustments define conditional/standardized associations, not causal effects.

Add pooled categorical BMI effects as a sensitivity, retaining age interactions. Do not match exactly on sparse age×BMI cells or collapse categories after seeing results: v1 elderly underweight support is only 5 y0 / 2 y1. Broad age adjustment leaves within-category age confounding; there is no exact-age matching claim.

## 4. Daily endpoint, harmonic score and dependence-aware null

Primary daily HRp10 is the tenth percentile of original observed bin means on each selected date, in bpm. All bins are equally weighted; event-level resting HR is not reconstructed.

The nonperiodic baseline has an intercept, linear elapsed calendar time, weekday indicators and four nonoverlapping six-hour observed-bin fractions. Clock fractions condition on sampling placement and occupied-bin count. Remove redundant columns deterministically in a declared order, check rank and save retained columns. No daily HR imputation is permitted.

For each `T ∈ {25,25.5,…,35}` days, add `cos(2πt/T)` and `sin(2πt/T)` jointly to that baseline and fit numerical least squares. Define

`S_i = max_T (RSS_0 − RSS_T) / RSS_0`.

This is partial R² relative to baseline residual variation, unlike v1's total-variance denominator. Handle zero baseline RSS as undefined and record exclusions. Amplitude at selected T is `A_i = sqrt(β_cos² + β_sin²)` bpm; peak-to-trough is `2A_i`. Physical amplitude and normalized fit strength answer different questions.

The primary null is a baseline plus stationary calendar-time AR(1) residual noise: a latent daily process with correlation `φ^|t−s|`, time in days, `0≤φ<1`. One-day φ describes persistence between adjacent dates; gaps weaken correlation by elapsed calendar lag. Estimate noise parameters from the nonperiodic baseline with missing-date likelihood/state-space treatment; do not fill missing HR or compress time to observation index. Harmonic scores use ordinary RSS; the AR model calibrates their null distribution. A stationary AR(2) daily-noise model is the prespecified colored-noise sensitivity, with stationarity constraints recorded.

Use an approximate plug-in parametric bootstrap: estimate noise parameters once from the observed nonperiodic residuals, then simulate on each participant's actual selected calendar/covariates, preserving missingness and masks. Refit the baseline and complete 21-period search in every replicate. Noise parameters do not enter the OLS score, so refitting them inside these score replicates is unnecessary. Noise-parameter uncertainty is a separate sensitivity; the empirical tail probability is not exact. Use B=999 final replicates and

`p_i = (1 + Σ_b 1[S_i,b ≥ S_i]) / (B+1)`.

This is a model-conditional null-tail measure, not a probability of a true cycle or an estimate of biological prevalence. Primary group endpoint:

`E_i = S_i − mean_b S_i,b`,

without zero truncation. It measures excess fit under the chosen null. Report A and null mean selected A separately; an amplitude-excess sensitivity is not an unbiased latent amplitude estimate. Boundary hits and null Monte Carlo precision must accompany results.

Before group inference, validate false-positive behavior and power on synthetic nulls and injected sinusoids using actual timestamp geometries balanced across age×salutation groups. Report null calibration and power separately in each of the six age×salutation cells. Include equal-physical-amplitude sinusoid injections across actual observation geometries to quantify detectability and selection bias. Identical injected A need not produce identical normalized S: higher relative fit strength does not establish a larger true physical amplitude. Include noise-parameter estimation uncertainty, colored-noise and AR(1)/AR(2) misspecification checks. Do not choose a null to maximize the y0−y1 contrast. If calibration fails, stop inferential interpretation, document the failure and revise assumptions explicitly.

A fixed-frequency F test does not calibrate the selected maximum; see [Astropy peak-significance documentation](https://docs.astropy.org/en/stable/timeseries/lombscargle.html#false-alarm-approximations). Dependence assumptions matter: [Delisle et al. (2020)](https://arxiv.org/abs/2001.10319) derive peak-significance calculations for correlated noise. Neither source validates this dataset's proposed noise model.

## 5. Sampling and signal specificity

Prespecify quadratic versus linear elapsed-time trend as a slow-drift sensitivity, recalibrating the complete score under its corresponding null. Inspect first-window start season and calendar year by age×salutation before outcome inference. Compare a baseline without clock fractions with the recording-conditioned baseline on identical participants/dates. Treat attenuation as representation sensitivity, not proof of behavioral mediation. Prespecify a common clock-support HRp10 sensitivity using mask availability alone to choose clock strata; check feasibility before fitting and record changed populations. Do not silently substitute a sleep signal.

Night/day HR are secondary and use paired common-date eligibility, with their own counts and spans. Analyze HR and recording signals at a common selected period with explicit phase/coupling summaries; independently maximizing each signal cannot establish independence. Do not interpret wear controls as a complete confounding adjustment.

Plot 14–60-day spectra, observation-window functions and boundary frequencies. Any formal broader-band claim requires repeating that broader search in its null simulation. Flanking bands are not guaranteed biological nulls. Keep secondary signal tests in a separately declared BH family and label them exploratory; they cannot redefine the primary outcome.

## 6. Temporal coherence and optional device replication

Among the 2,324 two-window eligible participants, use the earliest second qualifying window whose start is at least 180 days after the first start. Apply the same 15-dates-per-block rule. First-window HR alone selects T, phase, coefficients and preprocessing. Freeze them and predict second-window HR at actual elapsed dates; compare prediction error with the frozen nonperiodic baseline. Offset or amplitude adaptation is a separately labeled secondary analysis, never hidden inside the frozen prediction endpoint.

A chronological-versus-random-partition control must retain equal observation counts and comparable calendar/clock coverage where feasible, and describe remaining spacing differences. These estimands differ: random partitions can interpolate across an era while chronological prediction extrapolates. Inspect calendar era/season within age groups. No future HR, labels or performance select windows or detrending. Phase failure challenges stationary coherence; it does not disprove variable-period physiology.

Garmin replication is optional after protocol freeze, using the same calendar exposure, age categories and null calibration. Count eligibility first. The existing strict first-40-day cache cannot supply 90 observations over 180 days; inspect the long-history Garmin pilot cache or commission extraction separately. No Garmin replication has run under this protocol.

## 7. Stages, deliverables and stopping boundaries

1. **Done:** saved-data EDA, actual-date feasibility and v1 report correction. No detector implementation was added.
2. **Not run:** freeze manifests/RNG, null-validation fixtures and numerical/rank checks. Stop if calibration fails or clock-support feasibility changes the target materially.
3. **Not run:** primary participant null metrics and age-specific/interaction inference.
4. **Not run:** temporal coherence, secondary sensitivities and optional Garmin replication.

Future outputs must include age×salutation exclusions, masks/dates and source provenance, participant scores/null summaries/counts/spans/boundary flags, age contrasts/interactions, and matched-window plots. Generate tables from saved outputs and verify them independently; avoid fixed narrative numbers. Record model failures and empty/sparse groups. Preserve v1 artifacts. This planning stage authorizes no implementation or experiment execution.

## 8. Next stage: imputation sensitivity and other-source checks

**Status: planned only.** This extension changes neither the observed-HRp10 primary endpoint nor the four-test age family above. Current work comprises code/output inventory and documentation; no new extraction, implementation, imputation, model or simulation has run. [NEXT_STAGE_INVENTORY.md](NEXT_STAGE_INVENTORY.md) records supporting paths, source identities and readiness limits.

### 8.1 Distinguish the missing-value operations

Within-day HR-bin filling changes the daily signal before extracting a quantile. Filling whole missing calendar-day outcomes changes longitudinal exposure and is excluded from the primary matched experiment. Participant-feature median imputation fills missing summary columns before a classifier; this is the only imputation used by `apple_garmin_transfer_pilot`, which did not clock-fill HR bins or daily outcomes.

The next-stage arms use identical eligible participants, first 180-day windows, 90 actual selected dates, original 5-minute masks and raw observed values:

| Arm | Within-day operation |
|---|---|
| Observed reference | HRp10 over original observed bins; current primary definition. |
| Global median | Fill missing bins with the training observed-cell median; diagnostic. |
| Clock median | Fill with training clock-bin medians, falling back to the training global median; main sensitivity. |
| Short linear + clock | Interior linear interpolation over 1–5 missing 5-minute bins, with bounding-center separation≤30 minutes; clock fallback elsewhere; main sensitivity. |
| Three-tier | Strict-Garmin rule: one missing bin linear; 2–12 bins four-point PCHIP using two original observed neighbors on each side, otherwise whole-gap linear; longer/boundary gaps clock-filled; secondary. |

Twelve missing 5-minute bins span 60 minutes of missing positions but 65 minutes between observed endpoints. Do not replace this rule with the earlier Garmin six-bin interpolation policy. Participant-specific medians would be a new optional method, not a method already evaluated by these experiments.

Never cross midnight, introduce dates, requalify days using filled coverage, or anchor interpolation on imputed values. Preserve observed HR exactly and retain the original mask even in completed arrays. Where caches use zero sentinels, interpret zeros through the original mask. Record filled fractions and interpolation tiers by age×salutation and source. A quantile over all 288 completed bins is a **clock-completed pseudoquantile**, a different estimand from observed-bin HRp10, not recovered true resting HR.

### 8.2 Template fitting and time boundaries

Use five participant folds, stratified by age×salutation only to balance fold allocation. For each evaluated participant, learn clock/global templates exclusively from other-fold participants' first-window selected 90 dates. Templates are pooled across ages and labels, never conditioned on them. Reuse folds across methods. Pool original observed cells as existing code does; this weights observed-cell exposure rather than taking a median of participant medians.

Freeze these templates before second-window evaluation; neither evaluated-person data nor later-window HR may train them. A strictly prospective calendar claim additionally requires donor observations available by the training cutoff. Otherwise describe the analysis as retrospective cross-fitting. Device-native cross-fitted templates are the primary replication setting. Frozen Apple→Garmin template transport is a distinct secondary question, separate from classifier transfer and from device-native filling.

### 8.3 Validate the measurement transformation before inference

First verify exact preservation of observations, dates and masks. On complete or high-coverage empirical days, impose artificial masks and assess daily-quantile distortion against the retained empirical reference, with uniform-mask controls at equal retained counts. This reference is not unobserved physiological truth. Prior feasibility found 1,753 strict Garmin participants with at least 14 complete 5-minute-grid days within their first 40 selected days. That pool can test intraday distortion, not 180-day periodicity.

Next construct explicit bin-level known-truth nulls and injections. One proposed generator normalizes complete 288-bin donor profiles to HRp10=0, assigns profiles independently of monthly phase/calendar, and adds a known baseline HR level plus daily AR variation with or without a 25–35-day sinusoid. The complete-bin reference HRp10 therefore has the assigned daily component and harmonic amplitude by construction. Use disjoint donors; document independent-profile, weekday-stratified and correlated-profile variants rather than assuming a generator reproduces physiology. Retain actual target dates and original masks. Run the entire sequence: mask → fill → daily quantile → nuisance fit → full period search. Template fitting/refitting must remain confined to designated donor training pools and never access target unobserved truth.

Inject equal physical amplitudes across all six age×salutation cells. Assess false-positive calibration, quantile/amplitude bias, attenuation, period/boundary recovery and phase prediction across noise and mask conditions. Choose methods for calibrated recovery and robustness, never the largest female-coded contrast. A daily AR simulation applied after filled HRp10 calibrates that completed representation conditionally; it does **not** repeat raw-bin filling or establish that filling cannot create cycles. The bin-level controls are required before that interpretation.

### 8.4 Paired representation inference

Each arm needs its own daily endpoint and full search calibration. Pair participants and reuse simulated truths, masks and random draws where possible. Report changes in E and physical amplitude A separately; larger values after filling are not inherently better biological measurements.

Prespecify `D_clock = E_clock − E_observed` and `D_linear = E_short_linear − E_clock`. For each D, estimate three observation-adjusted within-age y0−y1 contrasts and the adjusted age×salutation interaction. These two representation contrasts times four HC3 Wald tests form **one eight-test Holm family at α=0.05 for this next stage**. Use pointwise 95% participant bootstrap intervals stratified by age×salutation; they are not familywise intervals. Global and three-tier results remain diagnostic/secondary unless a separate finite family is frozen before outcomes. No project-wide error guarantee across stages is claimed.

Show all six cells, original coverage and categorical BMI sensitivities, and the pooled age-standardized summary using existing fixed weights. Retain selected-population and within-category age-confounding limits. Do not redefine contrasts after observing which arm increases an effect.

### 8.5 Device and endpoint readiness gates

Garmin source 3 has a ready long-history 5-minute pilot cache for 1,000 participants; the strict first-40-day cache cannot support this protocol. Before v2 screening, young y0/y1 counts are 84/124, middle 252/429, elderly 26/85. Apply the unchanged 180-day/90-date rule and actual signal-coverage requirements, then report six-cell retention and power before replication claims. Expansion may be needed; metadata describing 5,812 raw Garmin HR users does not establish usable windows. Do not promise elderly replication from at most 26 eligible female-coded candidates before screening.

Samsung source 7 has 2,089 raw channel-3000 users in metadata, without a verified ready per-user 5-minute cache. Its strict 541-person cohort mostly uses channel 3002 (540 participants), with only one strict channel-3000 case. Audit native channel-3000 coverage, timezone, units and source provenance before any later cache construction. Channel 3002 is a separate vendor-derived hourly resting-HR endpoint: an optional convergent-endpoint study needs its own native 24-hour grid, summary, adequacy and null protocol frozen before analysis. Never relabel or upsample it as raw 5-minute HR or pool it with HRp10. Source IDs 9/13 remain vendor-unmapped; identity/channel/units/timestamp verification is an inclusion gate.

Primary between-device replication cohorts must be participant-disjoint globally, including template/reference donor folds, unless explicitly modeled as a repeated-subject design. Reserve multisource users for the separate paired extension; never let an ID cross source-fitting/evaluation folds. Existing cached-cohort disjointness does not establish disjointness of newly constructed Samsung or unmapped-source cohorts, so verify identity overlap before inclusion. Main replication remains channel 3000 with matched quality and budgets, common age/salutation definitions, original masks and calendar-era/season checks. Compare within-age effects by device and device heterogeneity rather than raw pooled y0/y1 differences. Keep common fixed age weights; absent age cells cannot silently trigger reweighting and a claim of full replication. Freeze the finite source×endpoint×representation hypothesis list and stage-level multiplicity before outcomes.

### 8.6 Multisource extension and outputs

Raw metadata overlaps are 1,102 Garmin/Apple users, 72 Garmin/Samsung and five Apple/Samsung; single-source caches exclude these overlaps. These counts do not establish concurrent paired observations. First audit simultaneous qualifying calendar windows, source switches and lineage: aggregator imports or duplicated records are not independent sensors. The pooled `epoch_days` table lacks a source key; source-specific raw-record extraction would be required later.

Same-person, same-calendar comparisons on source-qualified common support could better isolate acquisition differences. Nonconcurrent switches confound source with season, aging and behavior and are not paired hardware validation. If feasible, estimate per-person source contrasts and age×salutation heterogeneity; assume no paired sample size in advance.

Save source/six-cell exclusions, coverage/fill-tier distributions, folds and template provenance, arm-specific metrics/null validation, paired contrasts and plots distinguishing observations from imputations. Generate reports from saved tables. Stop mechanistic claims if nulls are inflated, injection bias differs materially between cells, or age overlap fails; report negative findings. All these tasks remain planned, with the accepted observed-data core unchanged.
