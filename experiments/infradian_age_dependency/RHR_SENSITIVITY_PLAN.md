# RHR Definition Sensitivity Plan

**Status:** Planned; not executed.
**Plan date:** 2026-10-20
**Purpose:** Test whether the reported infradian age patterns depend on the daily RHR summary derived from channel 3000 (`HeartRate`). This is a measurement-definition sensitivity analysis. It does not validate a menopausal mechanism or establish vendor RHR equivalence.

**Implementation instruction (2026-10-20):** Treat the existing saved HRp10 results as trusted and reuse them; the replay requirement below is superseded. The authorized work is implementation and bounded validation only. Do not run a full-cohort analysis.

**Baseline provenance:** Prepared against repository HEAD `d554c1f` (`docs(infradian-age): flag two-timescale RHR serial correlation as an open empirical benchmarking item`); live GitHub verification confirmed `refs/heads/main` at the same commit. The existing report and its prior documentation of HRp10, the onset-mixture interpretation, and longitudinal findings remain unchanged.

## Question and estimands

The primary question is whether female age-associated infradian excess and nonlinear sex-by-age differences remain similar when daily RHR is summarized using lower percentiles of the same valid 5-minute channel-3000 bin means. The reference is current `HRp10`; planned alternatives are `HRp5` and `HRp1`.

For each arm, estimate the age-specific female mean endpoint curve and female-minus-male curve on the existing RCS basis with knots `[25, 42, 52, 65]`, evaluated on the same shared support `[25, 75)`. The primary endpoint is `Q_i`, the normalized band-integrated signed spectral excess against a participant- and arm-specific two-component continuous covariance null. The primary paired contrasts are arm-minus-HRp10 differences in these curves, the four chunk Wald tests, and the inflection age defined as the minimum derivative of the female-minus-male curve over the existing `[35, 65]` search interval. Report zero crossing only as a secondary descriptive landmark.

The inferential target remains a cross-sectional association among participants eligible under the fixed cohort and windows. It is not an individual aging effect or a direct estimate of menopause onset.

## RHR arms

All three derived arms use the same channel-3000 cache, valid-bin mask, selected dates, and NumPy linear quantile interpolation (`np.percentile` default method):

| Arm | Daily value |
|---|---|
| HRp10 reference | 10th percentile of finite 5-minute bin means |
| HRp5 | 5th percentile of finite 5-minute bin means |
| HRp1 | 1st percentile of finite 5-minute bin means |

The existing day qualification remains at least 120 finite bins out of 288. Do not treat HRp1 as a minimum: under linear interpolation, with 120–288 observed bins it lies roughly between the second and fourth ordered observed bin means. Its greater sensitivity to a few low values is a measurement concern and motivation for the sensitivity, not an established performance result. Do not add a daily minimum arm to the primary protocol; it is a more extreme statistic and may be considered only as an explicitly labeled optional bound.

Supplier-derived candidates are conditional arms, not assumed available or equivalent: channel 3001 (`HeartRateResting`), channel 3002 (`HeartRateRestingHourly`), daily type 65 (mapped in `src/config.py` as RHR), and daily type 66 (mapped there as HR during sleep). The code mapping identifies exported labels, not derivation, units, timestamp semantics, or physiological equivalence. Channel 3002 requires a prespecified daily aggregation if its hourly values are to be reduced to daily observations. Do not silently mix these data with channel-3000 bins or call them HRp10.

## Stage 0: availability and provenance gate

Before implementation, inspect current raw inputs, cache/index coverage, and mappings for the exact analysis population and dates used by this experiment. This is an inventory step only; do not run endpoint models. Record, by source and candidate endpoint: file/table presence, participant coverage, timestamp/timezone and local-date mapping, duplicate handling, units/ranges, missingness by date, and provenance/processing description available in the repository. Confirm whether channels 3001/3002 and daily types 65/66 can be reliably linked to the same participant, source, local date, and existing window dates.

Proceed with the supplier comparison only for an endpoint with verified participant/date linkage and a defensible daily value definition. Keep each supplier endpoint as its own arm and source stratum. If provenance or date mapping is unresolved, report the limitation and stop that arm; do not infer vendor algorithms from the code labels. The age-dependency endpoint script uses full Apple and Garmin 5-minute caches at `experiments/apple_cohort_scaling/artifacts/cache_5min` and `experiments/garmin_infradian_replication/artifacts/cache_5min`; these are the relevant channel-3000 inputs. The separate 1,000-person Garmin temporal-binning cache is not the current age-analysis input. Samsung's age-dependency cache is partial and requires task-to-file coverage audit before use. Current inventory documentation cautions that channel quantities are distinct and that old strict-gate counts do not establish current-window eligibility (`experiments/female_infradian_cycles_v2/NEXT_STAGE_INVENTORY.md`, §§ channel distinctions and planning boundary).

## Fixed sample and date matching

Use `expanded90` as primary and `balanced90` as the prespecified protocol sensitivity. First replay HRp10 from the current inputs and reproduce saved endpoints within recorded numerical and Monte Carlo tolerance. Record those tolerances and diagnostics before deriving alternatives. For HRp5 and HRp1, preserve the original participant lists, window indices, boundaries, selected dates, and date hashes from the existing manifests. No arm may change participant eligibility, select dates according to its HR values, or use a different age support. Recompute the daily series and all downstream quantities on the same selected dates.

For each supplier endpoint, first use an **exact-date complete-case comparison**: retain only participants for whom the supplier value is available and valid on every one of the 90 selected dates for the corresponding existing window (and, for longitudinal analyses, every selected date in both existing windows). Apply the same participant subset to all compared arms and report retention and age-by-sex support before fitting. This subset is a conditional paired estimand and may differ materially from the primary cohort. For any paired estimate, use the common-converged participant set across all compared arms; report arm-specific convergence/failure counts. Also report HRp10 on the matched restriction and its full cohort so restriction and arm effects remain distinct. Do not silently exclude failed participants from a paired comparison.

If exact-date complete cases do not support the planned curves or endpoints, a separate **common-date reselection feasibility fallback** may be evaluated. Define dates using only channel-3000 qualification and verified supplier availability, before examining RHR values or spectral endpoints. Within the original 180-calendar-day interval and six fixed 30-day blocks, select exactly 90 common dates meeting the existing `expanded90` quota/capacity rule; apply this identical common-date list to every arm. For the `balanced90` reference, require and select 15 dates in each block. Keep original participants and window boundaries where feasible; any participant/window failing the common-date rule is excluded from every arm in that fallback. Report this analysis separately from exact-date complete cases and from the fixed-date HRp5/HRp1 comparison. If supplier availability changes shared age support, recompute support once on the supplier-complete matched cohort, then fit both supplier and HRp10 arms on that identical restricted support. Also report HRp10 on its original full support/cohort separately so restriction effects are visible. If common-date feasibility or shared age support fails, stop without relaxing quotas or redefining support post hoc.

## Matched signal processing and inference

For every arm and window, recompute its daily values, nuisance projection, residual variance, two-component REML null fit, spectral operators, and endpoints from that arm's signal. Preserve the existing nuisance design (intercept, linear elapsed time, weekday indicators, and four clock-coverage fractions), QR rank-screening rule, frequency grids, period grid, and endpoint formulas. Coverage covariates come from the original channel-3000 valid-bin mask in all arms, including supplier arms; this holds the observation-geometry adjustment fixed and must be described as such. Supplier values may have different temporal support or vendor filtering, so do not interpret channel-3000 coverage fractions as a complete measure of their observation process. Refit the null separately by arm; do not reuse HRp10 covariance parameters for another daily series.

For paired arm comparisons, generate and reuse the same independent standard-normal innovation arrays for both covariance components, by participant, selected-date vector, and null replicate across arms; propagate each arm's separately fitted covariance parameters through those innovations. Freeze the current `B=199` null replicates and `2,000` participant bootstrap resamples for the main comparison. Use common deterministic seed inputs that omit arm identity, and save the seed/provenance. This couples Monte Carlo error without forcing covariance parameters to match. Assess Monte Carlo precision for paired changes; if changes are comparable to null Monte Carlo error, report that limitation and run a larger replicate subset as a precision check, using common innovations across arms. For participant bootstrap inference, reuse the same participant cluster resample indices in every arm, stratified by the original 5-year age band × sex cells. Resample participants with both windows as a cluster for longitudinal comparisons. Do not bootstrap individual dates as independent observations.

Retain the existing RCS basis, model specification, robust covariance, and four tests on the same support:

1. Total female age association (3 df).
2. Nonlinear female curvature (2 df).
3. Total sex × age interaction (3 df).
4. Nonlinear sex × age interaction (2 df; primary test of nonparallel nonlinear curves).

Report coefficients/curves and paired arm-minus-reference contrasts with participant-bootstrap intervals, effect sizes, sample counts, and null-fit failure counts. Retain the current bootstrap construction and report pointwise and simultaneous bands for the primary age curves. The four chunk tests are per-arm supporting tests of within-arm hypotheses; they are not paired tests of arm differences. Report paired bootstrap intervals for curve/endpoint contrasts separately. Present p-values as supporting summaries, not as the criterion for robustness. Do not invent an equivalence margin or claim equivalence from non-significance of paired differences. Report endpoint agreement and measurement agreement separately; high correlation between daily values does not imply invariant spectral estimands.

## Outcomes

**Primary:** `Q_i` female and female-minus-male curves; paired curve contrasts HRp5−HRp10 and HRp1−HRp10; four per-arm chunk tests; derivative-minimum age with paired bootstrap uncertainty. These outcomes address whether the core age pattern and nonlinear interaction are sensitive to low-quantile choice. For the derivative minimum, report bootstrap boundary hits, multiple minima, and replicates without an identifiable minimum; do not discard these cases silently. For zero crossing, report nonexistence within the evaluation range rather than dropping those replicates.

For interpretation, show each arm's daily-value distribution and within-participant HRp10-to-alternative agreement alongside endpoint curves. Summarize paired changes in endpoint units and uncertainty. Do not define robustness by whether each arm independently crosses a significance threshold.

**Secondary spectral endpoints:** Repeat the matched age-curve and four-test summaries for `E_i2` and physical cosinor amplitude `A_hat`. Keep `sigma_resid` as a scale/comparator endpoint only. Do not use a nonsignificant interaction for `sigma_resid` as evidence of invariance or as proof of biological specificity; report all four tests and intervals, since the current report includes a bootstrap-significant Apple nonlinear interaction and Garmin nonlinear female curvature.

**Secondary longitudinal summaries:** On participants with both existing nonoverlapping windows, report paired `Q_i` change and annualized rate using the report calculation `(Q_w2 - Q_w1) / ((start_w2 - start_w1)/365.25)`, implemented in `scripts/06_generate_diataxis_report.py`. Preserve the report's sex strata and age bins `[20,35,42,47,52,60,75)` years. Report window-to-window endpoint correlations and frozen first-window harmonic forecast correlation using the same absolute calendar phase. For paired spectra, preserve the report's female `42 ≤ age_w1 < 48` cohort and paired source/protocol/window matching; calculate excess as `max(0, P_obs - P_null_mean)` and report changes at the existing 26.67-day peak and flanks (20 and 45 days). Preserve the same participants and dates across arms. Distinguish window-to-window association from repeatability and from a longitudinal causal effect. Use the same participant-level bootstrap clusters and resamples across arms. These analyses are secondary and conditional on two-window availability.

## Interpretation limits to carry into any report

- HRp10/p5/p1 are alternative summaries of the same observed PPG bins. Their differences can reflect the order statistic and the daily distribution of wear times, activity, sleep, and artifact. Keeping coverage adjustment fixed does not remove unmeasured behavior or prove equal measurement error across arms.
- Cross-sectional curves remain vulnerable to cohort/adoption differences, fitness and health differences, age-related wear patterns, and unrecorded endocrine exposures (including contraception, HRT, and relevant surgery). Age is based on birth year under a convention, not exact date of birth.
- Channel 3001/3002 or daily type 65/66 agreement is a supplier-endpoint comparison only after provenance and date semantics are verified. It cannot establish equivalence to HRp10 or to one another.
- The report's conversion of the spline inflection to population menopause-onset age depends on assumed symmetric unimodal onset distributions and transition-profile assumptions (REPORT.md §§2.6 and 4.3). Its Gaussian peak-attenuation approximation likewise depends on the assumed distributional shape and variance-to-peak argument. Proxy-definition sensitivity does not validate those assumptions or identify onset ages. Describe the inflection as a cross-sectional curve feature; present any menopause-onset interpretation as conditional, not empirically established.
- Residual-SD nonsignificance is not evidence of invariance. The current table has mixed interaction evidence, including Apple nonlinear sex×age bootstrap `p=0.0455` and Garmin nonlinear female curvature `p=0.0045` (REPORT.md Table 3). No biological-specificity claim follows from these comparisons alone.

## Reporting rules

- Label the channel-3000 percentile arms as derived RHR proxies and supplier values by their exported endpoint name and source.
- Keep the main fixed-date HRp5/HRp1 results separate from supplier complete-case and common-date fallback results.
- For supplier comparisons, show HRp10 restricted to the supplier-matched cohort beside the supplier arm; show the original full-cohort HRp10 estimate as a separate reference.
- Include paired participant counts, selected-date counts, age-by-sex cell counts, null convergence, and endpoint missingness for each arm and window.
- Display absolute curve estimates and paired curve differences with intervals; avoid conclusions based on significance changes alone.
- If the same-data derived arms are strongly correlated, report the correlation descriptively and retain the endpoint contrasts as the inferential comparison.
- Treat p1's sensitivity to a small number of low bins as a plausible measurement concern; do not state that p1 is less accurate without external validation.
- State explicitly that supplier-endpoint agreement cannot identify the vendor's internal filtering, sleep detection, or resting-state algorithm.
- Any claim about invariance requires a prespecified scientifically meaningful margin and suitable precision; this plan specifies neither, so no equivalence conclusion is planned.

## Staged execution and deliverables

1. **Inventory/provenance:** Complete Stage 0; publish a compact availability and local-date linkage table. Mark each supplier arm feasible, infeasible, or unresolved with evidence.
2. **Replay and traceability:** Reproduce HRp10 saved endpoints. Before comparing longitudinal or spectral summaries, reconcile script 06 and its source artifacts with report tables: verify participant IDs, source/protocol/window matching, endpoint and null versions, age-bin boundaries, denominators, and summary formulas. If these do not reproduce, stop those secondary comparisons and document the unresolved lineage; do not infer participant strata from printed report values.
3. **Derived-quantile validation:** Implement HRp5 and HRp1 from the unchanged cached channel-3000 matrices; verify identical participants, selected-date hashes, finite-day qualification, and expected order `HRp1 ≤ HRp5 ≤ HRp10` for every selected date. This is implementation validation, not an outcome analysis.
4. **Fixed-date primary sensitivity:** Recompute both window indices and both existing protocols; fit arm-specific nulls; produce primary curves, paired contrasts, tests, diagnostics, and cohort/retention table.
5. **Conditional supplier arm:** Run only after Stage 0 supports an endpoint; attempt exact-date complete cases first. Use common-date reselection only as the separately reported fallback if exact-date support is inadequate.
6. **Secondary longitudinal and spectral summaries:** Run only after paired windows, endpoint completeness, and Stage 2 traceability are confirmed for each arm.
7. **Reporting:** Save code/configuration provenance, date hashes, common RNG/bootstrap identifiers, null convergence diagnostics, result tables, and a short sensitivity report in separate run-specific output directories. Do not overwrite existing caches or artifacts. Do not alter `REPORT.md`; any later report update is a separate documented change after results exist.

**Planned deliverables:** source/endpoint availability table; arm-specific endpoint artifact with fixed provenance; paired primary and secondary result tables; cohort/support and convergence diagnostics; figures for age curves and paired contrasts; and a concise report stating what remains conditional or unvalidated.

No execution date or result is implied by this plan.
