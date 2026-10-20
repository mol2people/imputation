# Project: Continuous Age RCS Modeling of Heart Rate Infradian Rhythmicity

## Architecture
This project executes a rigorous, full-lifespan continuous age dependency study of human infradian resting heart rate rhythmicity across Apple Watch ($N=11,139$), Garmin Tier 1c ($N=4,410$), and Samsung cohorts.
Age is parameterized continuously using Frank Harrell's Restricted Cubic Splines (RMS best practices, Stone-Koo basis, landmark knots at $[25, 42, 52, 65]$ years).
Statistical inference is governed by pure non-parametric participant bootstrap (2,000 stratified resamples), uniform simultaneous confidence bands over $[18, 80]$, four Chunk Wald hypothesis tests on shared age support, and three physiological transition metrics.

```
Data Ingestion & Caches (Apple, Garmin, Samsung)
                    │
                    ▼
[M1: Cohort Extraction & Window Selection (expanded90 / balanced90)]
                    │
                    ▼
[M2: Four-Endpoint Suite & Two-Component REML Null Calibration (B=199)]
                    │
                    ▼
[M3: Continuous RCS OLS Regression & Chunk Wald Tests (Harrell RMS)]
                    │
                    ▼
[M4: 2,000 Stratified Bootstrap Inference, Simultaneous Bands, Transition Metrics]
                    │
                    ▼
[M5: Multi-Panel Publication Figures & Diátaxis Scientific Report]
                    │
                    ▼
[M6: E2E Dual-Track Verification & Forensic Integrity Audit]
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Existing cache validation | Verify Apple ($N=11,139$) and Garmin Tier 1c ($N=4,410$) 5m arrays and demographics (`13Aug_1222.csv`) | M1 | Survey 2 |
| 2 | Samsung cache builder | Extract channel-3000 Samsung candidate records from `out/`, bin into 5m arrays with $\ge 120$ valid bins/day | M1 | Survey 2 |
| 3 | Demographics & Age derivation | Join birth year, calculate continuous age via December-1 convention with Jan-1/Dec-31 sensitivities | M1 | Survey 2 |
| 4 | Standardized window selection | Implement `expanded90` (greedy 10-quota allocation to 90 total) and `balanced90` (15/block) | M1 | Survey 2 |
| 5 | Deterministic PRNG seeding | PCG64 stream from SHA-256 hash `20261020\|infradian_age\|{source}\|{user_id}\|{window_start_day}\|dates_block_{b}` | M1 | Survey 2 |
| 6 | Daily signal extraction | Extract resting HR proxy `HRp10` (10th percentile linear interpolation) on days with $\ge 120$ bins | M2 | Survey 3 |
| 7 | Nuisance baseline regression | Detrend via intercept, linear time, weekday indicators, and four 6-hour clock coverage fractions; sequential QR | M2 | Survey 3 |
| 8 | Four-endpoint estimation | Compute $Q_i$ (band-integrated signed excess), $E_{i2}$ (peak excess), $\hat{A}_i$ (cosinor amplitude), residual SD | M2 | Survey 3 |
| 9 | Broad frequency grid & quadrature | 231-frequency grid with exact boundaries $1/35$ and $1/25$, normalized trapezoidal quadrature for $Q_i$ | M2 | Survey 3 |
| 10 | Two-component REML null covariance | Model $\text{Cov}(e(t), e(u)) = v_1 \phi_1^{\|t-u\|} + v_2 \phi_2^{\|t-u\|}$ via 6-start Nelder-Mead optimization | M2 | Survey 3 |
| 11 | Null simulation pipeline | Generate $B=199$ null paths via continuous Markov transitions in $O(N)$ operations with Monte Carlo SE | M2 | Survey 3 |
| 12 | Stone-Koo RCS basis construction | Pre-specified knots at $[25, 42, 52, 65]$ years, linear tails outside $[25, 65]$, normalization by $(t_4-t_1)^2=1600$ | M3 | Survey 1 |
| 13 | Bivariate RCS regression model | $Q_i = \beta_0 + \beta_1 a_i + \beta_2 rcs_{1,i} + \beta_3 rcs_{2,i} + F_i (\delta_0 + \delta_1 a_i + \delta_2 rcs_{1,i} + \delta_3 rcs_{2,i}) + \varepsilon_i$ | M3 | Survey 1 |
| 14 | Chunk Wald hypothesis tests | Four tests: (1) Total female age (3 df), (2) Female curvature (2 df), (3) Total Sex×Age (3 df), (4) Nonlinear Sex×Age (2 df) | M3 | Survey 1 |
| 15 | Shared age support enforcement | Restrict interaction tests strictly to 10 contiguous 5-year bands with $\ge 30$ of each sex ($[25, 75)$ years) | M3 | Survey 1 |
| 16 | Wear-compliance & BMI sensitivities | Secondary adjustment for occupied bin fraction, nocturnal coverage, day span, and BMI stratification | M3 | Survey 1 |
| 17 | Stratified bootstrap resampling | 2,000 participant bootstrap resamples stratified by 5-year age band $\times$ sex | M4 | Survey 1 |
| 18 | Simultaneous uniform confidence bands | Studentized sup-norm deviation over continuous age $[18, 80]$ ($c_{0.95} \approx 2.63$) vs pointwise 95% CIs | M4 | Survey 1 |
| 19 | Three physiological transition metrics | (1) Inflection point $\arg\min d\Delta/da$, (2) Nominal zero-crossing $\Delta=0$, (3) Statistical cessation boundary | M4 | Survey 1 |
| 20 | Second-window repeatability | Test temporal persistence and frozen harmonic prediction on two-window participants ($N=2,324$) | M4 | Survey 2 |
| 21 | Multi-panel publication figures | Continuous partial effect curves with uniform bands, age-by-period excess heatmaps, FWHM traces | M5 | Survey 1,3 |
| 22 | Diátaxis scientific reference report | Comprehensive, ASD-STE100 Diátaxis report (`REPORT.md`) with empirical tables, effect sizes, caveats | M5 | Survey 1,2,3 |
| 23 | Opaque-box E2E testing suite | Dual-track verification covering Tiers 1-4 tests and adversarial hardening | M6 | Orchestrator |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Cohort Extraction & Windowing | Apple/Garmin validation, Samsung cache builder, demographics join, `expanded90`/`balanced90` window selection | none | PLANNED |
| M2 | Four Endpoints & REML Nulls | Daily HRp10, nuisance projector, broad/narrow spectra, 6-start REML two-component nulls ($B=199$), $Q_i, E_{i2}, \hat{A}_i, \sigma_{\text{resid}}$ | M1 | PLANNED |
| M3 | Continuous RCS Modeling & Tests | Stone-Koo basis, primary OLS regression, 4 Chunk Wald tests on shared support $[25, 75)$, compliance/BMI models | M2 | PLANNED |
| M4 | Bootstrap Inference & Transitions | 2,000 stratified resamples, uniform confidence bands, 3 transition metrics, second-window repeatability | M3 | PLANNED |
| M5 | Report & Publication Figures | Continuous curves with uniform bands, excess heatmaps, FWHM traces, Diátaxis `REPORT.md` | M4 | PLANNED |
| M6 | E2E Testing & Integrity Audit | Opaque-box test suite across Tiers 1-4 and independent forensic integrity verification | M1-M5 | PLANNED |

## Interface Contracts

### M1 ↔ M2 Contract
- **Artifacts**:
  - `participants.parquet`: columns `[user_id, source, salutation, birth_year, age_dec1, age_jan1, age_dec31, weight, height, bmi]`
  - `windows.parquet`: columns `[user_id, source, protocol, window_index, start_day, end_day, num_qualified_dates, is_eligible]`
  - `selected_dates.parquet`: columns `[user_id, source, protocol, window_index, selected_dates_hash, block_index, date, occupied_bins]`
- **Semantics**:
  - `protocol` is `'expanded90'` (primary) or `'balanced90'` (reference).
  - Exactly 90 unique dates per eligible window for both protocols.

### M2 ↔ M3 Contract
- **Artifacts**:
  - `endpoints.parquet`: columns `[user_id, source, protocol, age, sex, Q_i, E_i2, A_hat, sigma_resid, Q_i_mc_se, null_status, v1, v2, phi1, phi2]`
  - `spectra/`: participant observed spectra and null means across the 231-point broad grid.
- **Semantics**:
  - $Q_i$ is normalized trapezoidal quadrature of signed excess over $[1/35, 1/25]\text{ d}^{-1}$.
  - Finite status flags recorded for every participant; failed fits flagged, not silently zeroed.

### M3 ↔ M4 Contract
- **Artifacts**:
  - `age_models/model_fits.parquet`: coefficient vectors $\boldsymbol{\beta}, \boldsymbol{\delta}$, covariances, $R^2$, and degrees of freedom.
  - `age_models/chunk_wald_tests.csv`: Wald statistics, degrees of freedom, asymptotic and bootstrap p-values.
  - `shared_support_mask.parquet`: boolean indicator for participants in the 10 contiguous bands $[25, 75)$ years.

### M4 ↔ M5 Contract
- **Artifacts**:
  - `bootstrap/bootstrap_curves.parquet`: 2,000 resampled continuous curves over age grid $a \in [18, 80]$ (step 0.25y).
  - `bootstrap/confidence_bands.parquet`: pointwise 95% CIs and simultaneous uniform bands.
  - `bootstrap/transition_metrics.csv`: point estimates and 95% CIs for inflection point, zero-crossing, and cessation boundary.

## Code Layout
All implementation code, scripts, artifacts, and figures reside strictly under:
`/Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/infradian_age_dependency/`

- `scripts/`:
  - `00_verify_environment_and_fixtures.py` (pre-flight unit verification)
  - `01_build_manifests_and_windows.py` (M1: ingestion, Samsung cache builder, window selection)
  - `02_compute_endpoints_and_nulls.py` (M2: HRp10, nuisance baseline, REML calibration, endpoints)
  - `03_fit_rcs_models.py` (M3: Stone-Koo basis, OLS regression, Chunk Wald tests, sensitivities)
  - `04_run_bootstrap_inference.py` (M4: 2,000 stratified bootstrap resamples, uniform bands, transition metrics)
  - `05_generate_publication_figures.py` (M5: multi-panel figures)
  - `06_generate_diataxis_report.py` (M5: Diátaxis reference report generator)
  - `tests/test_e2e_suite.py` (M6: Opaque-box E2E test suite)
- `artifacts/`:
  - `cache_5min/` (newly built Samsung 5m `.npz` files)
  - `manifests/`
  - `windows/`
  - `endpoints/`
  - `models/`
  - `bootstrap/`
- `figures/`:
  - `fig1_rcs_continuous_partial_effects.png`
  - `fig2_excess_heatmaps_and_fwhm.png`
  - `fig3_delta_curve_transitions.png`
  - `fig4_source_and_protocol_comparisons.png`
- `REPORT.md`: Primary Diátaxis scientific reference report.
