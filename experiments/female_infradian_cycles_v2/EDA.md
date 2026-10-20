# Saved-data EDA for an age-accounted infradian protocol

2026-10-20. Descriptive aggregation only. No detection scripts, periodograms, models, null simulations, resampling or hypothesis tests were run. The previous experiment's fitted values are inputs, not validated detections. This document does not specify or implement the v2 protocol.

## Available demographics and measurement fields

The participant table has 11,139 unique IDs and 79 columns. The manifest contains `customer_id,user_id,salutation,y,age_group,bmi_group,csv_path`; it supplies no continuous age, birth date, cycle status, menopause, hormone use or menstrual events. `y=0` is recorded salutation10; `y=1` is20. Calling age categories reproductive/perimenopausal/postmenopausal is unsupported. The three observed age labels are `Young Adults`, `Middle Ager`, `Elderly`; no age-category missingness. Four BMI labels are `normal,overweight,obese,underweight`; no missing BMI categories. Metric-table y/age/BMI agree exactly with matched manifest rows. Continuous-age availability elsewhere in the raw database was not assessed.

Metadata columns are `customer_id,y,sex,age_group,bmi_group,n_total_days,n_adequate_days,timespan_days,screened_pass`. Each of seven signals has ten saved metrics with prefixes `ls_period,ls_power,ls_amplitude,ls_snr,cos_period,cos_amplitude,cos_acrophase,cos_delta_r2,cos_f_stat,cos_pvalue`. Signals are `hr_p10,night_hr,day_hr,dipping_ratio,occ_daily,nocturnal_compliance,gap_episodes`. No signal-specific observation count, selected dates, signal-specific span, daily values, mean level, uncertainty or raw residuals are saved here. Original mask occupancy is joined from `cache_5min/customers.csv`: it describes all upstream cached adequate days, not only the experiment's ≥120-bin days.

## Screening and age overlap

The old screen requires ≥60 days with ≥120 occupied five-minute bins plus cached span≥60 days. Its span is last minus first date across all upstream cached days; the span after the ≥120-bin filter is not saved. Occupied bins do not establish continuous wear. Of 8,425 screened participants, 8,424 have finite HRp10 amplitudes; one elderly y1 participant does not. Signal-specific fits need only ten nonmissing daily observations in the old code. Across screened participants, finite amplitudes number 7,033 for night HR, 7,019 for dipping, 8,413 for nocturnal compliance, and 8,425 for day HR, occupancy and gap episodes. Missing fits must not be counted as negative detections.

| age_group | y | n_pre | n_screened | retained_fraction |
| --- | --- | --- | --- | --- |
| Elderly | 0 | 384 | 289 | 0.7526 |
| Elderly | 1 | 969 | 703 | 0.7255 |
| Middle Ager | 0 | 2774 | 2088 | 0.7527 |
| Middle Ager | 1 | 4696 | 3634 | 0.7739 |
| Young Adults | 0 | 697 | 502 | 0.7202 |
| Young Adults | 1 | 1619 | 1209 | 0.7468 |

Age-stratified comparisons are supported by both salutation groups in all three categories, but exact-age matching is impossible with these artifacts. Within-category residual age confounding remains. The detailed age×BMI table reports finite cell support; elderly underweight has only 5 y0 / 2 y1 and middle-category underweight 30 / 9. Do not rely on finely stratified subgroup effects in those cells. Weighting/matching should define a target population and inspect overlap before fitting; no matching was performed here.

## Recording histories and observation selection

Medians below are among screened participants. `adequate_density_over_cached_span` divides selected high-occupancy day count by the inclusive span of all upstream cached days. It is a descriptive coverage proxy, not continuity within a prespecified analysis window.

| age_group | y | n_total_days | n_adequate_days | timespan_days | adequate_density_over_cached_span | occupancy_rate_5m |
| --- | --- | --- | --- | --- | --- | --- |
| Elderly | 0 | 331.0000 | 257.0000 | 479.0000 | 0.5546 | 0.5883 |
| Elderly | 1 | 309.0000 | 232.0000 | 490.0000 | 0.5479 | 0.6161 |
| Middle Ager | 0 | 331.0000 | 213.0000 | 490.0000 | 0.5053 | 0.5301 |
| Middle Ager | 1 | 331.0000 | 236.0000 | 489.0000 | 0.5440 | 0.5832 |
| Young Adults | 0 | 320.5000 | 197.5000 | 442.5000 | 0.4745 | 0.5307 |
| Young Adults | 1 | 338.0000 | 219.0000 | 488.0000 | 0.5319 | 0.5459 |

Young y0 participants have shorter median histories and spans than young y1 participants. Age-specific amplitude contrasts can therefore mix different precision, opportunities for period selection, and temporal coverage. The full CSV contains quartiles and range; medians do not establish overlap of individual sampling schedules.

## Existing fitted summaries

HRp10 values below are selected-period native-bpm amplitudes from the old model, not menstrual amplitudes. A period exactly 25 or 35 days is a boundary maximum within the constrained search; it need not indicate a 25/35-day oscillator. Per-signal finite-fit counts, quartiles/means and separate boundary counts are in `signal_summaries_by_age_and_salutation.csv`. No nominal p-values were used to select these rows.

| age_group | y | n_finite_fit | amplitude_median | period_boundary_fraction |
| --- | --- | --- | --- | --- |
| Elderly | 0 | 289 | 0.9103 | 0.1799 |
| Elderly | 1 | 702 | 0.9295 | 0.2108 |
| Middle Ager | 0 | 2088 | 1.1676 | 0.2079 |
| Middle Ager | 1 | 3634 | 1.0530 | 0.2053 |
| Young Adults | 0 | 502 | 1.5159 | 0.1474 |
| Young Adults | 1 | 1209 | 1.2126 | 0.2076 |

The period search itself forces a maximum into 25–35 days even for nonperiodic series. Boundary frequencies and between-group amplitude contrasts require a calibrated dependence-aware null and sampling controls before mechanistic interpretation. Broad age and BMI adjustment alone cannot supply that calibration.

## Feasibility limits for v2

`metadata_threshold_counts.csv` enumerates descriptive retention for minimum selected-day counts 60/90/120/180 and cached spans 120/180/270/365 days. These are candidate-count upper bounds for a future fixed-duration or rolling-window analysis: globally sufficient days/span do not guarantee enough observations within any particular window, let alone within night/day signals. For example, ≥120 selected days and cached span≥180 days retains the following participants:

| age_group | y | n |
| --- | --- | --- |
| Elderly | 0 | 217 |
| Elderly | 1 | 517 |
| Middle Ager | 0 | 1565 |
| Middle Ager | 1 | 2765 |
| Young Adults | 0 | 351 |
| Young Adults | 1 | 900 |

A future extraction should save actual selected calendar dates, per-signal finite counts and spans, observation fractions by window, and exclusions by age/salutation. A window with a prespecified duration should enforce its own observed-day density and longest-gap constraints. Existing metadata cannot choose a scientifically justified exact window or guarantee six observed cycles. Avoid treating the currently available global span as analyzed continuous history.

An age-accounted analysis can prespecify age-stratum-specific contrasts plus a common age-standardized summary; retain y×age interaction rather than describing one additive y coefficient as age invariant. BMI can be included as a coarse covariate with explicit sparse-cell handling. Coverage, history and selection opportunities should be matched or standardized within age categories, with the estimand stated. These are design recommendations, not analyses performed.

## Provenance

All outputs in this folder are new descriptive derivatives. Original caches, prior result tables and code were not modified. Source methods inspected: `../female_infradian_cycles/scripts/01_extract_and_detect_infradian.py` (screening 486–530), `00_verify_infradian_methods.py` (period search and fixed-period F test 576–594), and `02_summarize_infradian_contrasts.py` (age/BMI additive model 312–368). No old-script execution or coefficient/p-value recalculation occurred. Group summaries are unadjusted descriptive statistics of existing stored metrics.

Input SHA256:

- `experiments/female_infradian_cycles/results/infradian_metrics_by_participant.csv`: `be416c0a2b85cdd6a63183e23ecff1c1f7b731e80ce318b512df60549d27ef22`
- `experiments/apple_cohort_scaling/artifacts/apple_full_manifest.csv`: `0635bff111fedab241dd02ebca53b14b4b58752ce249d37052110b748ccfeb74`
- `experiments/apple_cohort_scaling/artifacts/cache_5min/customers.csv`: `f5016b5a6c477eec1a439929685fc5c64a0e4d5fd48f1132af23b0116d77f09a`
