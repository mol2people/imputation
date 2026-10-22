# RHR definition sensitivity: percentile arms and supplier daily RHR

## Question

Does the female-specific 25–35 d spectral concentration in resting heart rate, and its age pattern, depend on how daily RHR is defined?

Three alternatives to the saved channel-3000 HRp10 endpoint are compared:

| arm | definition | source | cohort |
| --- | --- | --- | --- |
| HRp5 | 5th percentile of observed 5-min bin means | channel 3000, same cache and selected dates | full fixed-date cohort |
| HRp1 | 1st percentile of observed 5-min bin means | channel 3000, same cache and selected dates | full fixed-date cohort |
| RHR | supplier-reported daily resting heart rate (daily type 65) | supplier daily export | exact-date complete cases only |

The primary estimand is the paired difference in the RCS age curve of `Q_i`, and in the female-minus-male contrast, on the common participant set for each source/protocol/window. `Q_i` is the partial R² at the peak of the 25–35 d band after projection on the fixed nuisance design. The analysis is cross-sectional; it does not identify individual aging, menopause onset, or vendor equivalence.

## Runs and design

- Percentile arms: `artifacts/rhr_sensitivity/run_20261020/endpoints`. Full cohort: 15,074 expanded90 and 13,598 balanced90 windows; 199 null replicates; 2,000 bootstrap draws. Common-cohort sizes 2,076–5,162.
- Supplier arm: `artifacts/rhr_sensitivity/run_20261021/endpoints`. Exact-date complete cases: 9,406 expanded90 and 8,672 balanced90 windows; 199 null replicates; 2,000 bootstrap draws. Common-cohort sizes 1,019–2,804.
- Supplier retention: 18,078 of 28,672 windows (62.7 %) have a finite supplier value on every one of the 90 selected dates. One `rss0_degenerate` row per protocol is excluded by `null_status`.
- Inference is run separately for the percentile arms and the supplier arm. `run_secondary` uses one common cohort across all requested arms; mixing a 62.7 % exact-date subset with the full percentile cohort would shrink the percentile secondary analyses.
- All arms use the same nuisance design: intercept, linear elapsed time, six weekday indicators, and four clock-coverage fractions from channel 3000. For the supplier arm the coverage covariates are inherited from channel 3000; they do not describe the supplier's own missingness (all 90 dates are present by construction). Keeping the design identical makes the endpoint the only changed input.

## Daily agreement

| contrast | paired daily values | Pearson r | median absolute difference |
| --- | ---: | ---: | ---: |
| HRp5 − HRp10 | 2,580,480 | 0.989 | 1.35 bpm |
| HRp1 − HRp10 | 2,580,480 | 0.934 | 3.50 bpm |
| RHR − HRp10 | 1,627,020 | 0.854 | 2.00 bpm |

Supplier RHR minus HRp10: mean −0.77 bpm, SD 4.59 bpm (expanded90). Correlation does not establish equivalence; the two endpoints are related but not interchangeable.

## Q_i pattern

### HRp10 reference

Female `Q_i` exceeds male `Q_i`, and the female level declines with age. In the supplier-matched expanded90 cohort:

| sex | age | mean Q_i HRp10 | mean Q_i RHR |
| --- | --- | ---: | ---: |
| F | 20–35 | 0.0290 | 0.0220 |
| F | 35–50 | 0.0239 | 0.0165 |
| F | 50–75 | 0.0070 | 0.0052 |
| M | 20–35 | 0.0016 | −0.0001 |
| M | 35–50 | 0.0020 | 0.0011 |
| M | 50–75 | 0.0022 | 0.0017 |

The ordering F20–35 > F35–50 > F50–75 and female > male is present in both arms. Cross-arm within-participant `Q_i` correlation is 0.79 in F20–35, 0.73 in F35–50, 0.61–0.65 in F50–75, and 0.50–0.63 in males.

### Paired curve differences, female, ages 40–55

| arm | max absolute difference in Q_i | simultaneous 95 % band excludes zero | direction |
| --- | ---: | ---: | --- |
| HRp5 | 0.0001–0.0017 | 2 of 8 strata | negligible |
| HRp1 | 0.0019–0.0101 | 6 of 8 strata | HRp1 lower |
| RHR | 0.0036–0.0079 (mean difference) | 8 of 8 strata | RHR lower |

The supplier RHR leaves female `Q_i` at roughly 69–76 % of the HRp10 level at ages 20–50 (24–31 % lower), and the paired difference is simultaneous-significant in every stratum. HRp5 is nearly indistinguishable from HRp10. HRp1 is generally intermediate: it lowers `Q_i` less than the supplier endpoint in most strata.

### Amplitude and residual scatter

- `A_hat` is similar between RHR and HRp10 in F20–35 (1.92 vs 1.96 bpm, expanded90) and higher for RHR in older women and men. Paired RHR − HRp10 `A_hat` in females 40–55 is +0.036 to +0.180 bpm and simultaneous-significant.
- `sigma_resid` is higher for RHR by 0.34–0.46 bpm in every sex/age stratum.

The attenuation of `Q_i` is therefore consistent with regression dilution from a noisier daily endpoint, not with a different age pattern. This mechanism is suggestive; the analysis does not isolate measurement error from a genuine difference in what the supplier RHR measures.

## Transition landmarks

Paired inflection-age point estimates differ by −0.6 to +0.45 years. Every paired 95 % interval includes zero. Intervals are wide (up to 30 years in one boundary stratum), so the analysis has little power to detect a real shift. The inflection-age comparison does not establish invariance or a shift.

## Fixed-period spectral excess (secondary)

Female participants aged 42 to 47, within-person change in excess at fixed periods, window 1 to window 2:

| period | HRp10 mean change | RHR mean change | paired RHR − HRp10 (95 % CI) |
| --- | ---: | ---: | --- |
| 26.67 d, source 3 balanced90 | −0.0115 | +0.0031 | +0.0146 (0.0031, 0.0269) |
| 26.67 d, source 3 expanded90 | −0.0117 | +0.0023 | +0.0140 (0.0026, 0.0255) |
| 26.67 d, source 6 balanced90 | −0.0169 | −0.0024 | +0.0144 (−0.0005, 0.0306) |
| 26.67 d, source 6 expanded90 | −0.0116 | −0.0004 | +0.0112 (−0.0001, 0.0247) |

At 20 d and 45 d the paired contrasts are mostly null. The fixed-period spectral-change estimate is therefore endpoint-dependent: HRp10 shows a decrease at 26.67 d, while RHR shows little or no decrease. The two-component null is fit separately per arm, so a difference in excess can reflect null calibration as well as signal; this secondary result is weaker than the age-curve result.

For comparison, the percentile arms at 26.67 d change the estimate by −0.0031 to +0.0051 (HRp5) and +0.0009 to +0.0087 (HRp1); most intervals include zero.

## Interpretation

Established within this dataset:

- The female-specific 25–35 d concentration and its age ordering are not artifacts of the RHR definition (percentile choice or supplier pipeline). The sign and ordering survive HRp5, HRp1, and a supplier daily RHR from a different pipeline.
- HRp5 is nearly indistinguishable from HRp10 in `Q_i`. HRp1 and supplier RHR attenuate the level while preserving the pattern.

Suggestive:

- The attenuation is consistent with higher measurement noise in the alternate endpoints (higher `sigma_resid`, regression dilution). The analysis does not separate measurement error from a difference in the underlying construct.

Not established:

- Endpoint invariance of the magnitude of `Q_i`.
- Endpoint invariance of the fixed-period 26.67 d spectral change in perimenopausal women.
- A shift or invariance of the `Q_i` inflection age.
- That any endpoint is more accurate, or that the supplier RHR validates a vendor algorithm.

## Limits

- The supplier arm is selected on exact-date data availability, not at random; 62.7 % of windows are retained. Differential retention by device, age, or sex is not excluded.
- Coverage covariates for the supplier arm are inherited from channel 3000 and are a fixed design choice, not a model of supplier missingness.
- No menstrual-cycle logs are available. The analysis cannot distinguish a menstrual mechanism from other infradian sources.
- One window per participant is used in the primary tests; the two-window secondary analyses are restricted to participants with both windows in every requested arm.
- The cross-sectional RCS curves do not identify within-person aging or menopause onset.

## Artefacts

- `experiments/infradian_age_dependency/artifacts/rhr_sensitivity/run_20261020/inference/REPORT.md` (percentile arms).
- `experiments/infradian_age_dependency/artifacts/rhr_sensitivity/run_20261021/inference_supplier/REPORT.md` (supplier arm).
- Key tables: `cohort_counts.parquet`, `curves.parquet`, `paired_curve_differences.parquet`, `paired_transitions.parquet`, `secondary_spectral_contrasts.parquet` in each inference directory.
- Run provenance: `endpoints/<arm>/<protocol>/provenance.json`; null replicates 199; fixed date manifest hash `0edc65a6…`.
