# Longitudinal spectral deep dive: within-person age change in the 40–50 age band

**Status:** Planned; not executed.
**Plan date:** 2026-10-22.
**Scope:** Multisource infradian replication (`experiments/multisource_infradian_replication`).
**Interpretation assumption:** “40–50 band” means chronological age at the window midpoint, not a 40–50-day spectral period band. If the intended band is period days, the design below is unchanged; only the spectral integration band changes.

## 1. Question

Among people with repeated eligible infradian windows, how do spectral properties change **within the same person** as age advances through the 40–50 age band, separately by device source and endpoint?

Primary question:

> For each estimable source × endpoint × protocol arm, estimate the within-person annualized change in the primary spectral endpoint \(Q_i\) as a smooth function of age in \([40,50)\), and test whether that annualized change itself varies across the band.

Secondary questions:

1. Does the within-person age slope agree with the derivative of the existing cross-sectional age curve in the same arm and age band?
2. Do spectral shape properties — peak excess, peak period, sub-band power, period-domain centroid, and period-domain spread — change with age in the same direction as \(Q_i\)?
3. Are the patterns consistent across dense device arms, and do they differ by administrative sex?
4. How much of the observed change is compatible with measurement noise and short-interval sampling error?

This is an observational within-person association, not a causal aging effect. With two windows separated by about six months, age and calendar period are collinear. The estimand is **change per elapsed year of age**, interpreted as age-plus-period change unless a period sensitivity supports a narrower interpretation.

## 2. Current cohort evidence and design consequence

The executed repeat-window run `artifacts/runs/full_repeat_b199_boot1000_20261022` provides the current cohort. These are planning counts, not eligibility for a new run.

| Quantity | Current value |
|---|---:|
| Paired participant-arm observations | 45,279 |
| Unique source-participants | 16,571 |
| Pairs with midpoint age in `[40,50)` | 13,565 |
| Unique source-participants represented in those pairs | 5,575 |
| Median window gap | 0.493 years |
| 95th percentile window gap | 1.52 years |
| Maximum window gap | 2.24 years |
| Integer age change 0 / 1 / 2 years in the 40–50 pairs | 49.7% / 49.5% / 0.75% |
| First-window calendar years | 2020, 2021, 2022 |

The gap distribution is heavily concentrated at 180 days. The integer age difference between windows is therefore **zero for about half the pairs**. Integer age change is not a usable denominator or exposure. Use exact elapsed time between window starts for the change denominator, and use an age midpoint for the age axis.

Current arm-specific counts in the 40–50 midpoint band, `expanded90` only, unique source-participants per arm:

| Source | Endpoint | Female | Male | Total | Planned tier |
|---|---|---:|---:|---:|---|
| Apple | RHR | 795 | 1,596 | 2,391 | A |
| Garmin | RHR | 503 | 906 | 1,409 | A |
| Garmin | HRp10 | 396 | 690 | 1,086 | A |
| Apple | HRp10 | 278 | 695 | 973 | A |
| Fitbit | RHR | 549 | 324 | 873 | A |
| Nokia | RHR | 32 | 87 | 119 | B |
| Samsung | RHR | 38 | 62 | 100 | B |
| Oura | HR_during_sleep | 10 | 15 | 25 | C |
| Nokia | HRp10 | 1 | 9 | 10 | C |
| Google Fit | RHR | 1 | 7 | 8 | C |
| Google Fit | HRp10 | 0 | 5 | 5 | C |
| Samsung | HRp10 | 0 | 1 | 1 | C |

The totals sum arm-participant records and double-count people across arms. Tier definitions are in §8. Counts must be refreshed after the final eligibility scan; they are not permission to relax support rules.

## 3. Data and eligibility

### 3.1 Arm definitions

- Sources: Fitbit `2`, Garmin `3`, Apple `6`, Samsung `7`, Nokia `9`, Google Fit `13`, Oura `19`.
- Endpoints: dense channel-3000 `HRp10`; daily type-65 `RHR`; daily type-66 `HR_during_sleep`.
- Protocols: `expanded90` primary; `balanced90` reference sensitivity. Do not combine protocols or count both as independent participants.
- Analysis key: `(source, endpoint, protocol, user_id, window_index)`.
- Do not concatenate devices, endpoints, or protocols into one series. Do not pool source-specific estimates into a single causal estimate. Cross-device figures are descriptive forests.

### 3.2 Window sequence

**Tier 1 (immediate, uses existing artifacts).** Reuse the current first and second windows in `paired_windows.parquet`. This is the fast path and validates the analysis code. It is underpowered for age curvature because the median age change is only 0.493 years.

**Tier 2 (recommended longitudinal design).** Enumerate all eligible non-overlapping 180-day windows for each participant, source, endpoint, and protocol. For a participant with sorted candidate starts, select the earliest eligible start, then the earliest eligible start at least 180 days after the previous start, and repeat. Require the same source, endpoint, protocol, and participant identity across the sequence. This preserves the existing window and selection rules and avoids overlapping-window dependence.

A sliding-window design with starts less than 180 days apart may be run only as a sensitivity analysis. Adjacent windows overlap; observations are not independent; report it separately and do not call it a replication.

Tier 2 requires new manifest and endpoint stages. Do not silently reinterpret the existing `window_index=1,2` rows as an all-window record.

### 3.3 Age and age change

- Available age is year-resolution: birth year plus the date convention. Primary convention is December 1, matching the existing pipeline.
- `age_w1` and `age_w2` are integer ages at window starts. The actual birth date is unknown; age is interval-censored within roughly one year.
- Exact elapsed age change is
  \[
  \Delta a_i = \frac{s_{i2}-s_{i1}}{365.25},
  \]
  where \(s\) is the window start day. This is exact from dates. Do **not** use `age_w2 - age_w1` as the denominator; it is zero in about half the pairs.
- Age at transition midpoint:
  \[
  a_{i,\mathrm{mid}} = a_{i1} + \Delta a_i/2,
  \]
  with \(a_{i1}\) the year-resolution age at the first window start. Report sensitivity using first-window age, second-window age, and January 1 / December 31 birthday conventions.
- Primary band membership uses \(a_{i,\mathrm{mid}}\in[40,50)\). Sensitivities:
  - both windows in \([40,50)\);
  - either window in \([40,50)\);
  - age interval overlaps \([40,50)\) under the birthday uncertainty.
- Report the joint distribution of age midpoint, gap, and integer age change by arm, sex, and calendar year.
- Calendar period and age are collinear within person. Do not fit both as separately identified linear effects. Report start-year distribution and a period sensitivity; do not claim to separate aging from secular change.

## 4. Spectral properties

### 4.1 Primary and secondary endpoints

The primary endpoint remains the normalized signed null-adjusted excess partial \(R^2\) over periods 25–35 days:
\[
Q_i=\frac{1}{|B|}\int_B D_i(f)\,df,\qquad B=[1/35,1/25],
\]
where \(D_i(f)=P_i(f)-\overline{P_{i,\mathrm{null}}}(f)\). This is already computed in `endpoints.parquet`.

Secondary properties, derived from the same \(D_i(f)\) grid:

| Property | Definition or source | Role |
|---|---|---|
| \(E_{i2}\) | existing discrete peak excess over 25–35 days | secondary |
| \(T_{\mathrm{peak}}\) | existing discrete peak period | secondary |
| \(\hat A_i\) | existing cosinor amplitude at selected peak | secondary |
| \(Q_i^{25\text{–}30}\) | average \(D_i(f)\) over periods 25–30 days | shape, secondary |
| \(Q_i^{30\text{–}35}\) | average \(D_i(f)\) over periods 30–35 days | shape, secondary |
| \(T_c\) | period-domain centroid of positive excess over 20–45 days | exploratory |
| \(S_T\) | period-domain spread of positive excess over 20–45 days | exploratory |
| \(\sigma_{\mathrm{resid}}\) | existing residual SD | context, not a spectral shape claim |

Define, for \(w(T)=\max(D_i(1/T),0)\) on a fixed period grid over 20–45 days,
\[
T_c=\frac{\sum_T T\,w(T)}{\sum_T w(T)},\qquad
S_T=\sqrt{\frac{\sum_T (T-T_c)^2 w(T)}{\sum_T w(T)}}.
\]
If \(\sum_T w(T)=0\), set \(T_c\) and \(S_T\) missing. The positive-part transform is used only for shape summaries; \(Q_i\) remains signed. Flag peaks at the 25- or 35-day boundary; shape summaries are not interpretable when the peak is censored at a boundary.

Do not add a composite spectral score, PCA score, or machine-learning classifier in the primary analysis. Keep one primary endpoint, a small secondary family, and clearly exploratory shape metrics.

### 4.2 Feature extraction and alignment

`spectra_broad.npz` stores `broad_freqs`, `P_obs`, `P_null_mean`, `P_null_var`, and `D_excess` per arm. It does not store participant keys. Row alignment must therefore be validated, not assumed:

1. Read the arm endpoint table in its original row order.
2. Retain rows with `status == "ok"` and `null_status == "converged"` and finite required metrics; this is the order used for the spectra arrays.
3. Assert row counts match `D_excess.shape[0]`.
4. Recompute \(Q_i\) from `D_excess` by trapezoidal integration and compare to stored `Q_i`. Current spot check on Apple RHR `balanced90`: maximum absolute difference \(5.4\times10^{-9}\) and correlation \(1.0\) after filtering two failed rows.
5. Persist a keyed feature table with `(source, endpoint, protocol, user_id, window_index, selected_dates_hash)` and a feature-definition version. Do not rely on positional order downstream.

All feature extraction uses the same selected dates, nuisance design, and null fit already stored for the arm. No endpoint is imputed.

## 5. Estimands

For participant \(i\), arm \(a\), adjacent windows \(w,w+1\), spectral property \(Y\):

\[
\Delta Y_{iaw}=Y_{ia,w+1}-Y_{iaw},\qquad
R_{iaw}=\frac{\Delta Y_{iaw}}{\Delta a_{iaw}},\qquad
\bar a_{iaw}=a_{ia,w}+\Delta a_{iaw}/2.
\]

Here \(R\) is the within-person annualized rate of change, and \(\bar a\) is the age at the midpoint of the transition.

For each arm and sex stratum, estimate the smooth function
\[
\mu_a(t)=\mathbb E[R_{iaw}\mid \bar a_{iaw}=t]
\]
over \(t\in[40,50)\). Report:

1. **Mean within-person rate** in the band:
   \[
   \theta_{\mathrm{mean}}=\frac{1}{10}\int_{40}^{50}\mu_a(t)\,dt.
   \]
2. **Change in the annualized rate across the band**:
   \[
   \theta_{\mathrm{curv}}=\mu_a(48)-\mu_a(42),
   \]
   or an equivalent prespecified linear contrast. This is the longitudinal analogue of curvature in the cross-sectional age curve.
3. **Cross-sectional derivative** at the same ages, \( \beta_{\mathrm{cross}}(t)=f_a'(t) \), from the existing expanded90 RCS age curve. Compare \(\mu_a(t)\) with \(\beta_{\mathrm{cross}}(t)\) descriptively. Agreement supports a consistent age pattern; disagreement may reflect period effects, selection, or measurement differences. The comparison is not a formal test of aging.
4. **Variance decomposition** in the 40–50 band: between-person SD, within-person transition SD, and the fraction attributable to null Monte Carlo error where available. Report the intraclass correlation for each spectral property.
5. **Sex contrast** in \(\theta_{\mathrm{mean}}\) and \(\theta_{\mathrm{curv}}\), only where sex-specific age support is adequate.
6. **Arm-specific results**. No pooled estimand across sources or endpoints; a forest of arm estimates is descriptive only.

For a single transition per participant, \(R_i\) is estimated with noise. The design estimates a **mean derivative**, not an individual causal slope. Do not present person-specific slopes as reliable unless Tier 2 provides enough repeated transitions.

## 6. Statistical models

### 6.1 Tier 1: two-window first-difference model

For each arm and sex stratum, with one transition per participant:
\[
R_i=\beta_0+\beta_1(\bar a_i-45)+\beta_2(\bar a_i-45)^2+\varepsilon_i.
\]
Primary fit is OLS with HC3 robust covariance; bootstrap by participant. A linear model is used where the quadratic term is not supported. Report the fitted curve and both \(\theta_{\mathrm{mean}}\) and \(\theta_{\mathrm{curv}}\).

Measurement-error sensitivity:

- \(Q_i\) has known null Monte Carlo standard error `Q_i_mc_se_w1/w2`. Approximate the measurement variance of \(R_i\) by
  \[
  v_i^{\mathrm{meas}}=\frac{\mathrm{se}_{i1}^2+\mathrm{se}_{i2}^2}{\Delta a_i^2}.
  \]
- Fit weighted least squares with weights \(1/(v_i^{\mathrm{meas}}+\tau^2)\), estimating \(\tau^2\) from the residual variance. Report unweighted as primary and weighted as sensitivity. Current 40–50 \(Q_i\) Monte Carlo SEs are about \(1.4\times10^{-3}\), much smaller than the observed annualized-change SD of about \(6.8\times10^{-2}\), so this sensitivity is expected to be small but must be reported.
- Do not treat the Monte Carlo SE as the full measurement error of \(Q_i\); it measures null simulation error only.

### 6.2 Tier 2: multi-window transition model

If the feasibility gate supports at least three non-overlapping windows for a meaningful subset, build all adjacent transitions. For each arm and sex stratum:
\[
R_{ij}=\mu_a(\bar a_{ij})+u_i+\varepsilon_{ij},
\]
with participant random intercept \(u_i\). Use cluster-robust inference or a participant bootstrap that resamples participants and retains all their transitions. A random slope is added only when each included participant has enough transitions and the model is identified; otherwise report the random-intercept model. The same quadratic or prespecified RCS specification for \(\mu_a\) is used.

Tier 2 is preferred for the age-curvature question because it increases within-person age leverage and separates persistent participant effects from transition noise. It does not solve age–period collinearity.

### 6.3 Inference and multiplicity

- Participant bootstrap: 2,000 draws stratified by five-year age band × administrative sex, resampling participants as clusters. Reuse the same participant draws across metrics within an arm where possible.
- Report pointwise 95% percentile intervals. Do not call them simultaneous bands.
- Primary inference family: \(Q_i\) in Tier A arms, \(\theta_{\mathrm{mean}}\) and \(\theta_{\mathrm{curv}}\). Secondary spectral properties use Benjamini–Hochberg FDR within a prespecified family; exploratory shape metrics are labeled exploratory and are not used for confirmatory claims.
- Report all arms, including non-estimable and sparse arms, in inventory tables. Do not drop an arm because its estimate is null or inconvenient.
- Model diagnostics: convergence, residual heteroskedasticity, influence of high-leverage ages, age support, and bootstrap failure counts.

### 6.4 Power and precision gate

Before fitting the full model family, simulate from the observed 40–50 paired residuals:

- Preserve arm sample sizes, age distribution, gap distribution, and sex mix.
- Inject a range of \(\theta_{\mathrm{mean}}\) and \(\theta_{\mathrm{curv}}\) effects.
- Estimate the minimum detectable effect for 80% power at two-sided \(\alpha=0.05\), with and without FDR.
- Report expected 95% CI widths for each Tier A arm.

If an arm cannot detect a scientifically plausible effect, report it as descriptive and do not present a null result as evidence of no change.

## 7. Selection, missingness, and bias

- The paired cohort is selected on two eligible, converged windows. Report paired fraction by arm, age band, sex, and calendar year.
- Compare paired and all eligible participants on first-window \(Q_i\), `E_i2`, age, sex, wear/coverage, and gap. Report standardized differences.
- Report all `rss0_degenerate` and non-converged windows; exclude them from models; do not remediate or rerun without an explicit request.
- Report the distribution of `n_valid_windows` and time span. Longer histories may have more windows and different wear behavior; this can induce informative observation times.
- Sensitivity: inverse-probability weighting for inclusion in the paired set, if positivity is adequate. Label as sensitivity, not primary.
- Sensitivity: restrict to 180-day gaps (the dominant design) to remove gap-related heterogeneity. Report longer-gap pairs separately.
- Sensitivity: expanded90 primary versus balanced90 reference. Do not switch the primary protocol after inspecting results.
- Calendar period: 2020–2022 includes pandemic-era changes. Report start-year distribution and a period-stratified sensitivity. Do not claim that a within-person change over six months is pure aging.

## 8. Device and arm support

Support rules are fixed before seeing 40–50 model results:

| Tier | Rule | Use |
|---|---|---|
| A | At least 500 unique source-participants in the 40–50 band, with at least 100 per sex | sex-specific curves; primary inference for \(Q_i\) |
| B | 100–499 unique source-participants, or fewer than 100 per sex | sex-pooled or limited model; descriptive |
| C | Fewer than 100 unique source-participants | inventory and descriptive summaries only; no inferential curve |

Additional cell rule: a five-year age × sex cell needs at least 30 participants for an interaction or cell-specific estimate. Otherwise collapse ages or report the arm as not estimable for that contrast. Do not relax thresholds after inspecting spectra.

Current Tier A arms: Apple RHR, Garmin RHR, Garmin HRp10, Apple HRp10, Fitbit RHR. Tier B: Nokia RHR, Samsung RHR. Tier C: Oura sleep HR, Nokia HRp10, Google Fit RHR/HRp10, Samsung HRp10. These tiers are provisional until the refreshed eligibility scan.

No cross-source pooling. A descriptive forest may show arm-specific \(\theta_{\mathrm{mean}}\) and \(\theta_{\mathrm{curv}}\) with intervals; do not compute a single “all-device” effect. Endpoint definitions and nuisance designs differ, so a pooled estimate would not have a common estimand.

## 9. Implementation stages

### Stage 0 — Freeze

Write this plan into the repository, freeze:
- primary endpoint \(Q_i\);
- age band \([40,50)\) by midpoint;
- expanded90 primary, balanced90 sensitivity;
- support tiers and multiplicity families;
- Tier 1 and Tier 2 decision rule.

Proposed implementation units (not yet written):
- `multisource/spectral_features.py`: keyed extraction of \(Q\), \(E_{i2}\), peak period, amplitude, sub-band \(Q\), \(T_c\), and \(S_T\).
- `multisource/longitudinal_spectral.py`: transition construction, age/gap derivation, Tier 1 and Tier 2 models, bootstrap.
- `scripts/11_longitudinal_spectral.py`: stage entry point and artifact writing.
- `tests/test_longitudinal_spectral.py`: alignment, age arithmetic, synthetic recovery, and bootstrap reproducibility.

### Stage 1 — Feasibility and feature extraction

1. Refresh arm × sex × age counts from the current repeat run and, if requested, from the raw eligible-date inventory.
2. Count non-overlapping 180-day windows per participant/arm from existing manifests and caches. Report the distribution of window counts and the number of participants with ≥3 windows in the 40–50 band.
3. Extract `Q_i`, `E_i2`, `peak_period`, `A_hat`, `sigma_resid`, sub-band \(Q\), \(T_c\), and \(S_T\) from each `spectra_broad.npz`.
4. Validate row alignment and recompute \(Q_i\) against `endpoints.parquet`.
5. Write keyed `spectral_features.parquet`; no positional assumptions downstream.
6. Run the precision simulation and report minimum detectable effects.

**Gate 1:** feature alignment passes; \(\max|Q_{\mathrm{recomputed}}-Q_{\mathrm{stored}}|<10^{-6}\); no duplicate keys; keyed features cover all converged rows.

### Stage 2 — Tier 1 analysis

1. Build transition table from existing `paired_windows.parquet`: exact gap, age midpoint, integer-age-change diagnostics, band flags.
2. Fit arm- and sex-specific first-difference models for \(Q_i\); secondary features separately.
3. Bootstrap 2,000 participant draws; save curves, coefficients, contrasts, CIs, diagnostics.
4. Compare with the cross-sectional derivative from `curves_point.parquet` (expanded90).
5. Produce selection and availability tables.

**Gate 2:** Tier A arms meet support; model convergence; no age-boundary-only estimates; sensitivity tables complete.

### Stage 3 — Multi-window extension

Run only if Stage 1 shows enough participants with ≥3 non-overlapping windows in the 40–50 band.

1. Add a new manifest mode for all non-overlapping windows; preserve the existing first/second window definitions.
2. Compute endpoints for all windows with the same engine, `B=199`, and the same endpoint-specific nuisance design.
3. Build adjacent transitions and fit the Tier 2 random-intercept model.
4. Compare Tier 1 and Tier 2 estimates; report differences as design sensitivity, not as a new truth.

If the gate fails, report Tier 1 and record the multi-window extension as infeasible with counts.

### Stage 4 — Reporting and figures

Required outputs:

- `spectral_features.parquet` and feature-definition metadata.
- Transition table with age, gap, midpoint, band flags, protocol, and selection flags.
- Model coefficient table: arm, sex, metric, \(\theta_{\mathrm{mean}}\), \(\theta_{\mathrm{curv}}\), CIs, n, support tier.
- Curves: \(\mu_a(t)\) in 40–50 with pointwise 95% intervals, dense arms, female and male separately.
- Overlay: within-person \(\mu_a(t)\) versus cross-sectional derivative \(f_a'(t)\).
- Forest: arm-specific \(\theta_{\mathrm{mean}}\) at age 45 and \(\theta_{\mathrm{curv}}\), descriptive.
- Availability: paired fraction, SMD, gap distribution, integer-age-change distribution, window-count distribution.
- Sensitivity: expanded vs balanced, birthday conventions, 180-day restriction, measurement-error weighting, IPW, period stratification.

### Stage 5 — Validation

- Unit tests for \(R\), age midpoint, exact gap, band membership, sub-band integration, \(T_c\), \(S_T\), and feature alignment.
- Synthetic recovery: generate known linear and quadratic age effects with observed noise; verify model recovery and CI coverage.
- Exact reproduction of current \(Q_i\) from spectra.
- Bootstrap reproducibility with fixed seeds.
- No silent changes to `window_index` semantics, selected dates, or endpoint definitions.

## 10. Decision gates and stop rules

1. **Alignment failure:** if spectra arrays cannot be keyed or \(Q_i\) does not reproduce, stop before modeling.
2. **Support failure:** if a Tier A arm cannot support the 40–50 curve with at least 30 participants per five-year sex cell, report the arm as descriptive and do not force a curve.
3. **Multi-window failure:** if fewer than a prespecified number of participants have ≥3 non-overlapping windows in the band, do not run Tier 2; report Tier 1.
4. **Sensitivity instability:** if the sign of \(\theta_{\mathrm{mean}}\) or \(\theta_{\mathrm{curv}}\) reverses across expanded/balanced protocols, age conventions, or measurement-error weighting, report the result as unstable rather than selecting the favorable specification.
5. **No post hoc metric rescue:** do not replace \(Q_i\) with a more favorable feature after seeing results.

## 11. Out of scope

- Causal aging, menopause onset, ovarian function, or clinical cycle status.
- Vendor RHR equivalence or validation.
- Pooling sources, endpoints, or protocols into one effect.
- Unwindowed/all-available endpoint estimation; that remains a separate deferred estimand.
- Full-cohort RHR-definition sensitivity; provenance remains unresolved and those results are not to be presented as verified.
- Person-specific prediction or clinical use.

## 12. Open decisions for the user

1. Confirm that “40–50 band” means age in years, not period in days.
2. Confirm `expanded90` as primary and `balanced90` as sensitivity, or switch primary with a documented rationale.
3. Choose Tier 1 only (fast, current two windows) or Tier 2 as the primary longitudinal design (all non-overlapping eligible windows, new endpoint run).
4. Choose 2,000 bootstrap draws for final inference, consistent with the historical protocol, or retain 1,000 for lower compute.
