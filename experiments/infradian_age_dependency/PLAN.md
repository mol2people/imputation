# Age dependency of the monthly heart rate periodogram

Date: 2026-10-20. Status: implementation handoff; this experiment has not run. Existing Apple and Garmin data have already been explored. This is an exploratory extension, including Samsung replication, not an untouched confirmatory study.

## 1. Objective and agreed scope

Estimate whether the female monthly HR periodogram bump weakens, shifts, broadens or becomes concentrated in fewer participants with increasing age. Analyze Apple/source 6, Garmin/source 3 and Samsung/source 7 separately. “Female” means administrative salutation 10; salutation 20 supplies the male comparison. These labels do not verify reproductive anatomy or clinical cycle status.

Use birth-year-derived annual age and an **unpenalized quadratic age model**, not splines. The main expanded protocol retains 180 calendar days and exactly 90 selected observations, requiring at least 10 qualified dates in each of six 30-day blocks and at least 90 across the whole window. Reselect the earliest qualifying window for **every participant**, including those already analyzed. Do not append newly eligible results to old endpoints computed on different windows.

Include source-specific trajectories from multisource participants. Include Samsung in the implementation even though its qualified sample size is not yet known. Keep the original 15-dates-per-block protocol as a reference and the existing Apple 60-date protocol as a sensitivity. Do not shorten windows or lower daily coverage thresholds in this experiment.

This task produces a protocol only. Subsequent implementation should create new scripts and artifacts in this directory, leaving upstream caches, reports, allocations and plans intact. PDF catalog reconciliation and downstream salutation-classifier experiments are outside scope.

## 2. Verified starting point and remaining feasibility

The following counts were obtained on 2026-10-20 from saved tables and finite-mask/date scans. They describe current single-source caches, before admitting additional source-specific histories.

| Source and rule | Window-eligible participants | Female participants |
|---|---:|---:|
| Apple, 15/block | 4,631 | 1,478 |
| Apple, 10/block and 90 total | 5,390 | 1,765 |
| Apple, 10/block and 60 selected | 5,518 | 1,820 |
| Garmin, 15/block | 3,760 | 1,339 |
| Garmin, 10/block and 90 total | 3,832 | 1,376 |
| Garmin, 10/block and 60 selected | 3,840 | 1,380 |

Saved fitted Apple counts are 4,630 for the original protocol and 5,516 for the 60-date protocol; eligibility does not guarantee a finite fitted endpoint. Female counts coincide in these saved runs. Expanded 90-date eligibility adds 324 women across the existing two caches, from 2,817 to 3,141. These new windows have not yet been analyzed.

Saved source metadata identify 405 additional distinct women outside both current Apple/Garmin caches with at least 90 recorded channel-3000 dates: 399 multisource and six single-source. Apple and Garmin source-specific counts are 325 and 258, with 178 overlapping IDs. Raw files exist for all 405. Samsung has 541 female candidates with at least 90 recorded dates outside the current Apple/Garmin caches; seven overlap the 405. These are inventory counts, **not qualified-window counts or strict upper bounds under different local-date semantics**. Source-tagged records, valid local dates and original bin coverage must establish actual eligibility. Do not add source counts as if they represented independent people.

Birth year is present for all 4,630 fitted Apple and 3,760 fitted Garmin participants and for the inventoried additional female candidates. Refresh birth-year availability for both salutation groups in every candidate source, including additional men and Samsung, before determining age-model eligibility. Refresh counts and hashes when implementing; these feasibility calculations did not create persistent manifests.

## 3. Inputs and identity

Paths below are relative to the repository root.

| Input | Use |
|---|---|
| `data/13Aug_1222.csv` | Unique `user_id`, salutation and `birth_date`; the latter stores birth **year**, not full date of birth. |
| `experiments/artifacts_v2/epoch_sources.parquet` | Enumerate source/channel-3000 participants and prioritize extraction. Recorded-date totals are not qualified-date totals. |
| `experiments/artifacts_v2/cohort_manifest.parquet` | Participant metadata and historical source membership; retain exclusions and provenance, not the old single-source restriction. |
| `experiments/apple_cohort_scaling/artifacts/cache_5min/` | Existing full-history Apple arrays and customer index. |
| `experiments/garmin_infradian_replication/artifacts/cache_5min/` | Existing full-history Garmin arrays for the 4,410-person expanded cohort. |
| `experiments/garmin_infradian_replication/artifacts/garmin_tier1c_cached_manifest.csv` | Garmin cache identities and demographic labels. |
| `out/<customer_id>.csv` | Source-tagged raw events for additional trajectories and Samsung. Resolve actual paths from existing manifests where supplied. |
| `experiments/female_infradian_cycles_v2/results/window_eligibility.parquet` | Historical windows and selected dates for reconciliation only. |
| `experiments/female_infradian_cycles_v2/results/raw_periodogram/spectra.npz` | Historical Apple observed and single-AR(1) spectral arrays. |
| `experiments/female_infradian_cycles_v2/results/recalibration/` | Historical two-component-null participant endpoints and spectral arrays. |
| `experiments/garmin_infradian_replication/results/garmin_k15/` | Historical Garmin endpoints; saved full spectral null is single AR(1), despite two-component scalar endpoints. |
| `experiments/female_infradian_cycles_v2/results/extended_k10/participants_k10.parquet` | One consistent historical Apple 60-date sensitivity run. |

Verify the `customer_id` to `user_id` mapping before joins. Normalize numeric identifiers to a declared canonical representation without silently truncating nonintegral values. Require uniqueness of the demographic record and agreement of existing labels; conflicting IDs or labels are quarantined with reasons. Never infer a match from row order.

The population frame is participants with channel-3000 metadata for sources 3, 6 or 7, a usable local raw file or validated cache, salutation 10/20 and valid birth year. Do not require historical `cohort_category == eligible`, availability of daily channels, complete BMI or single-source status. Do not hard-filter using the inventory's 90 recorded dates: earlier date-count semantics can differ from the local-date construction below. Eligibility is determined from actual source-specific bins.

One person can contribute one primary window to each eligible source. The key is `(user_id, source, protocol, window_number)`; report both unique people and source-specific records. Never concatenate devices into a single HR trajectory. Within-source analyses have one row per person. Any combined-source summary or cross-source uncertainty must cluster/resample by global `user_id`.

## 4. Source-specific cache construction

Reuse existing Apple/Garmin caches after checking identity, schema and bin semantics. Build new source-specific caches for missing trajectories and Samsung. The older strict Garmin 40-day cache is insufficient. Pooled `epoch_days.parquet` lacks a source column and cannot reconstruct source-specific HRp10 or masks.

Use the existing Apple/Garmin start-time binning convention:

1. Retain the requested source and type 3000, finite HR in 25–230 bpm, positive finite start timestamp and finite timezone offset. Log exclusion counts by reason. HR values retain their native scale; verify Samsung field names and units from export provenance before interpreting values as bpm. Do not silently adapt a different channel or unit.
2. Local timestamp is `startTimestamp + 60000 * timezoneOffset` in milliseconds. Use the exported per-record offset, including changes; no guessed timezone or UTC fallback. Construct local dates and 288 five-minute clock bins from this timestamp.
3. Each bin is the arithmetic mean of retained readings whose **start** falls in that bin. Do not duration-expand interval records, interpolate or fill. Preserve exported duplicate-reading semantics consistently with the existing builders; report duplicate counts instead of silently changing weights.
4. Store `date[D]` as integer calendar days since 1970-01-01 and `hr[D,288]`, with original unobserved bins missing. Verify unique ordered dates and finite-mask consistency. A qualified date has at least 120 originally observed bins.
5. Retain source provenance, filtering counts and cache hashes. A multisource person's source-7 cache contains source-7 readings only. Samsung channel 3002 and its historical strict RF cohort are not substitutes for channel 3000.

The historical cache day filters cannot remove a date with 120 occupied five-minute bins: it necessarily has at least 120 readings and at least 10 occupied clock-hours. Reusing those caches is compatible with the qualified-date rule. Still report cache failures and identity discrepancies explicitly.

Samsung is a required source-specific feasibility and analysis branch. If no usable windows or insufficient age/label support remain, produce its exclusion/support report and mark the affected analyses not estimable. Do not weaken thresholds to obtain a favorable Samsung result.

## 5. Exact window and date selection

Define protocols `expanded90` (primary), `balanced90` (reference) and `balanced60` (Apple sensitivity).

For a sorted unique set of qualified dates, enumerate candidate starts at those dates in ascending order. A window is `[s,s+180)` with blocks `[s+30b,s+30(b+1))`, `b=0,...,5`.

- `expanded90`: every block has at least 10 qualified dates and the full window has at least 90.
- `balanced90`: every block has at least 15 qualified dates.
- `balanced60`: every block has at least 10 qualified dates.

Select the earliest qualifying start separately for each protocol. Existing participants do not retain old windows by default. Window 2, used only for repeatability, is the earliest qualifying start at least 180 days after that protocol's window-1 start. Search actual calendar time; do not compress missing dates.

For `expanded90`, let `c_b` be block capacity. Initialize six selected-date quotas `q_b=10`. Allocate the remaining 30 dates one at a time to a block with the smallest current quota among those with `q_b<c_b`; break ties by ascending block index. This terminates at 90, preserves at least 10/block and gives exactly 15/block whenever every capacity is at least 15. `balanced90` and `balanced60` use quotas 15 and 10 respectively.

Within each block, uniformly sample its quota without replacement from sorted qualified dates. Use `numpy.random.Generator(PCG64(seed))`, where seed is the unsigned little-endian integer from the first eight SHA-256 bytes of:

`20261020|infradian_age|{source}|{user_id}|{window_start_day}|dates_block_{b}`

Generate one permutation and take its first `q_b` entries, then sort the selected dates. Do not put protocol in this seed: identical source/person/window/block with the same quota must yield identical selections across protocols. Source and window start are included intentionally; this differs from the historical Apple-only hash. Persist all selected dates, quotas, capacities, window starts and hashes. Define `selected_date_hash` as SHA-256 of the sorted selected dates serialized as little-endian signed int64 calendar days.

Recompute endpoints for changed dates. Historical spectra can be reused only if source, selected-date hash, daily endpoint, nuisance design, frequency grid and null specification all match; match by ID, never array position. Keep the historical results as context when parity fails.

## 6. Age and support

Parse `birth_date` as an integral birth year. Reject missing/nonintegral values and birth years later than the selected window's calendar year. Require the derived primary age to be in [0,120] years as a prespecified plausibility check, reporting exclusions rather than silently correcting values. Do not impute age or substitute category midpoints. At each selected window start, use the established December-1 convention:

`age = start.year - birth_year - 1 + I(start.month == 12)`.

This is a year-resolution estimate with birthday uncertainty, not exact age. Set `a=(age-45)/10`; 45 is a coefficient-reference age and 10 sets decade units. Neither is a biological transition parameter. Sensitivities use January-1 and December-31 birthday conventions at the same selected window, without changing HR data.

Before examining new spectral outcomes, tabulate source × protocol × salutation × five-year age-band counts on the fixed grid `[0,5),[5,10),...,[120,125)`, displaying bands covering all valid observed ages. Do not impose an adult-age cutoff by assumption. Report metadata availability, window retention, fit failures, coverage, actual date span and calendar timing. Show finer age distributions around 35–65, without choosing cutoffs from spectral results.

Fit the primary female-age model on all valid participant endpoints, but display female curves only over observed ages in bands with at least 30 women. For interaction inference and female-minus-male curves, refit the same model on participants in shared-support bands having at least 30 of each group; require at least three such bands. This avoids basing the interaction test on disjoint female/male age ranges. Mark unsupported intervals rather than bridge them with confident curves. Show all descriptive counts, including sparse bands. Require a full-rank model and at least 50 of each salutation, with each group represented in at least three five-year bands, for the primary combined quadratic inferential model. Otherwise report descriptive estimates and the specific support failure. All support counts for fitted curves refer to finite endpoint rows; separately report eligibility counts.

## 7. Daily endpoint and spectral estimands

Primary daily signal is the tenth percentile of originally observed five-minute bin means, `HRp10`, in bpm, using NumPy's linear quantile interpolation. Missing bins are excluded, never filled. Every selected date contributes one value; each participant contributes one primary spectrum regardless of history length.

The nuisance design contains intercept, linear elapsed calendar days, weekday indicators (Monday reference), and four original six-hour clock-coverage fractions. Use numerical least squares and the existing deterministic redundant-column handling in `female_infradian_cycles_v2/scripts/00_verify_v2_methods.py`; record retained columns, numerical rank and tolerances. Verify the residual projector against direct least-squares fitted values. Nonfinite results or effectively zero residual RSS receive explicit failed/undefined status, not score zero.

At frequency `f` in cycles/day, fit cosine and sine jointly with that nuisance design and define:

`P_i(f) = (RSS_0,i - RSS_f,i) / RSS_0,i`.

This is dimensionless partial R², not a power spectral density. Preserve numerical precision; clamp only tiny floating-point excursions outside [0,1] under a documented tolerance and fail substantive violations.

For the broad spectrum use the sorted union of frequencies `{k/720: k=12,...,240}` and exact boundaries `1/35` and `1/25`. Save all 3–60-day results; display 15–60 days for the main age-period plot. Adding exact band boundaries prevents grid-dependent endpoint truncation.

With the primary null in section 8, define signed excess:

`D_i(f) = P_i(f) - mean_b P_i,null,b(f)`.

The primary participant endpoint is the **mean signed excess across the monthly frequency band**:

`Q_i = integral_[1/35,1/25] D_i(f) df / (1/25 - 1/35)`.

Use trapezoidal quadrature on the specified frequency grid, including exact boundaries. This is a numerical band summary, not an exact continuous integral. Do not integrate uniformly over period, truncate negative excess, maximize the band or retain only “significant” participants.

Secondary endpoints:

- Recompute `E_i2 = max_T P_i(1/T) - mean_b max_T P_i,null,b(1/T)` on the historical `T=25,25.5,...,35` day grid. This is distinct from `Q_i`.
- Report selected harmonic amplitude `sqrt(beta_cos^2+beta_sin^2)` in bpm, residual SD and amplitude at fixed 28 days. Selected amplitude has search bias; fixed-period amplitude remains a noisy nonnegative estimate. Do not interpret either as calibrated cycle prevalence.
- Show participant `Q_i` distributions and quantiles by age band. Rejecting a noise null is not an estimate of the fraction with biological cycles.
- From each female age-band mean excess spectrum with at least 30 participants, describe the largest positive peak in 25–35 days and its full width at half maximum using the broad grid and linear interpolation. Resample women with replacement within each source/protocol/five-year band 2,000 times, retaining their complete spectra and fixed null means, to obtain pointwise percentile 95% intervals for the mean spectrum. Require the lower bound at the selected peak to exceed zero for a numerical width; this is a descriptive reporting rule, not a selection-corrected test. Find the nearest half-height crossings on each side; if either is absent within 15–60 days, label width censored/undefined. Report boundary peaks explicitly. Group width cannot distinguish within-person instability from between-person period dispersion.

## 8. Null calibration and computation

Use the same two-component calendar-time covariance for all three sources:

`Cov(e(t),e(u)) = v1 * phi1^abs(t-u) + v2 * phi2^abs(t-u)`,

with positive variances, `0<phi1<0.95`, and `phi2` selected from `{0.98,0.995}` by the restricted likelihood. This is a specified noise model, not a proven physiological model. Adapt the implementation in `female_infradian_cycles_v2/scripts/07_two_component_null_recalibration.py`; its existing unchecked optimizer result is not sufficient evidence of convergence.

Fit to the nonperiodic residual subspace on the selected actual dates. Use three fixed starts for the fast-component variance share (0.3, 0.5, 0.7), initializing total variance to `RSS_0 / residual_rank`, the existing single-AR(1) estimate clipped to `[1e-4,0.95-1e-4]`, and both slow-persistence candidates. Use the existing log-variance/logistic-persistence parameterization with Nelder-Mead, `maxiter=2000`, `xatol=1e-6`, `fatol=1e-6`. Retain the converged finite solution with the best objective; record all statuses and boundary behavior. Report that this is numerical optimization with no established global-optimum guarantee. If every start fails, retain the participant in eligibility counts with null-fit failure, not an alternative silently substituted null.

Generate B=199 independent null series per participant/window/protocol at their actual selected dates, preserving the nuisance design and masks. Project/refit the nuisance baseline and compute the complete spectrum and narrow-grid maximum for each draw. Parameters are plug-in estimates fitted once; they are not re-estimated per draw. Use an independent PCG64 stream derived by the section-5 SHA-256 rule with suffix `null_two_component|{selected_date_hash}` instead of `dates_block_{b}`. Thus identical selected dates and methods reuse identical null draws across protocols. Fit a single-AR(1) null as a specified sensitivity, with suffix `null_single_ar1|{selected_date_hash}`.

Persist observed spectra, null mean and variance by frequency, per-draw null band summaries and maxima, fitted noise parameters, selected dates and numerical diagnostics. Full per-draw broad spectra need not be retained. Save sufficient provenance to regenerate them.

Estimate the Monte Carlo SE of each null mean band summary from its 199 draws. Propagate these independent simulation errors through the age-model coefficient map, conditional on its design, and report their contribution relative to participant-bootstrap SEs. Finite-B and fitted-noise-parameter uncertainty are limitations of the primary conditional inference. More draws can be a separately documented precision follow-up; do not increase B selectively until a desired significance result appears.

Do not reuse the existing single-AR(1) Garmin spectral envelope as a two-component envelope. All sources and both 90-date protocols require matching spectral/null definitions.

## 9. Parametric age analysis

For each source separately, use `F=1` for salutation 10 and `F=0` for 20. Fit by unpenalized OLS:

`Q_i = beta0 + beta1*a_i + beta2*a_i^2 + F_i*(delta0 + delta1*a_i + delta2*a_i^2) + gamma' Z_i + error_i`.

There is no ridge alpha or spline smoothing parameter. The primary adjusted `Z` contains mean selected-day occupied-bin fraction, mean selected night coverage (00:00–06:00) and actual first-to-last selected-date span. Center covariates at pooled within-source eligible means. Display adjusted curves at those covariate means and assess observed covariate overlap by age; these are conditional associations, not causal adjustment or correction for all selection effects.

Fit the same quadratic model without `Z` as the unadjusted comparison. Add categorical BMI and calendar window-start year plus annual sine/cosine of window-start day-of-year as separate specified sensitivities. Keep missing BMI as an explicit category rather than shrinking the primary cohort. No outcome-driven covariate selection.

Primary per-source hypothesis: female age association, jointly

`beta1 + delta1 = 0` and `beta2 + delta2 = 0`.

Secondary per-source hypothesis, from the shared-age-support refit defined in section 6: different age dependence between labels, jointly

`delta1 = 0` and `delta2 = 0`.

Use HC3 Wald tests. The three source-specific primary tests form one Holm family; reserve all three slots, assigning an unavailable source p=1 for adjustment while labeling its estimate not estimable. The three interaction tests form a separate secondary Holm family with the same fixed slots and unavailable-source rule. Report estimates and raw/adjusted p-values. Other endpoint, curvature-only, sensitivity and age-specific contrasts are descriptive; do not present their unadjusted p-values as multiplicity-controlled findings.

Use 2,000 participant bootstrap resamples, stratified within source by original broad age category × salutation, to refit group models. Treat absent/unknown broad age category as an explicit `unknown` stratum; annual age eligibility does not require that metadata field. Freeze support membership, age transformation and covariate reference values at their original sample values. Report pointwise percentile 95% bands plus simultaneous 95% bands over the displayed annual-age grid: estimate each grid point's SE from the bootstrap fits, compute each draw's maximum absolute deviation from the original fitted curve divided by that SE, and use the 95th percentile maximum as the common multiplier. Clearly distinguish these bands and disclose failed/rank-deficient resamples. Treat zero-variance grid points separately rather than dividing by zero.

Estimate each source independently; no pooled coefficient is primary. For any cross-source contrast, use global-person resampling with shared multiplicities across all that person's source records; do not reuse independent source bootstraps as though multisource participants were independent.

The female-minus-male curve is `delta0+delta1*a+delta2*a^2` from the shared-support model. Keep its coefficients and analysis population separate from the primary full-population model. Test the female association and interaction directly, not “significant in women but not men.” A linear-age version is a fixed sensitivity; do not select linear versus quadratic by p-value. Report turning points only descriptively, inside supported ages, and never as menopause estimates. Sparse tails can create artificial quadratic upturns.

Repeat the same model specification for secondary `E_i2`, physical amplitude and residual variability, labeling the endpoint change. This helps distinguish an age change in absolute HR variation from a change in its prominence above background variation.

## 10. Reference comparisons and repeatability

Run both 90-date protocols on the full source-specific candidate frame, including Samsung. Report their common participants, entrants, selected-window shifts, age shifts and fit failures. Also report legacy-cache-only strata to distinguish population expansion from inclusion of multisource participants. These are different observation protocols and conditional populations, not interchangeable rows in one analysis.

For common participants, compare `Q_i` and fitted age curves using paired participant resampling. Window changes and date-geometry changes are part of the protocol contrast; do not attribute their combined effect solely to adding people. Where selected dates coincide, require endpoint parity under identical methods.

Use the historical Apple 60-date run as a secondary scalar-endpoint comparison after joining annual ages at its actual earliest qualifying windows. Reconstruct its windows from cache dates when the saved table omits starts; do not assign age from the new 90-date window. Do not splice together the two saved 60-date runs, which use different null RNG tags. A comparable 60-date `Q_i` curve requires new spectral/null computation and should be labeled separately from reuse of saved `E_i2`.

For the primary expanded90 population with a second nonoverlapping window, estimate window-specific `Q_i`, monthly selected-period differences and frozen first-window harmonic prediction in window 2. Retain first-window phase in absolute calendar time; do not re-estimate it from window 2 for the forecasting metric. Describe repeatability by annual age and report second-window retention. Repeated windows do not increase the number of independent participants or identify longitudinal aging from this short observation interval.

## 11. Implementation sequence and artifacts

Use the repository's Python environment and existing numerical libraries. Build a small, inspectable runner with separately callable stages; avoid copying a whole historical pipeline whose defaults differ.

1. **Inventory and extraction:** freeze config/input hashes, build global participant/source manifest, validate birth-year/label joins, reuse compatible caches and construct missing source-specific caches. Generate source × label × age retention tables, including Samsung.
2. **Window selection:** generate both 90-date protocols, quotas and selected dates before inspecting new HR spectral outcomes. Establish source support and report changes for existing participants.
3. **Focused method checks:** test the selection rules and endpoint/null implementation below on small deterministic examples, then a small cache-based engineering sample. This is not model selection.
4. **Participant endpoints:** run spectra and null calibration with checkpointed, unique source/person/protocol keys. Save errors as statuses; never silently drop exceptions.
5. **Age models and comparisons:** fit the fixed quadratic models, uncertainty calculations and sensitivities; complete second-window repeatability separately.
6. **Report and figures:** generate numbers from saved tables and render the figures below. Provide exact commands, package versions, seeds, runtime and resumability instructions in a short README.

Required artifacts under this experiment:

- `config.json`, `provenance.json`, `participants.parquet`: frozen specification, source/input/code hashes, metadata and all exclusion statuses.
- `windows.parquet`, `selected_dates.parquet`: one window key per row and selected original dates, capacities, quotas, coverage and date hashes.
- `endpoints.parquet`, `spectra/`, `null_diagnostics.parquet`: participant estimates, grids, null summaries/draw-level band statistics and numerical statuses.
- `age_support.csv`, `age_models/`, `contrasts.csv`, `bootstrap/`: analysis populations, ordered model columns, coefficients/covariances, fixed references, tests and reproducible group-resampling outputs.
- `protocol_comparisons.csv`, `repeatability/`, `figures/`, `REPORT.md`: paired protocol results, second-window results and narrative.

Never overwrite a completed run with a changed config. Checkpoint/resume must validate configuration and input hashes. Source metadata must remain separate from exported figures; omit direct participant identifiers from plots.

## 12. Required figures and completion checks

Main figure: three source columns, with (1) female age-by-period observed and excess heatmaps, (2) female adjusted/unadjusted quadratic curves with uncertainty, (3) female-minus-male curves, and (4) female/male age support and retention. Use consistent frequency grids, color scales and units. Five-year-bin spectra are descriptive; they do not replace the annual-age regression.

Supporting figures: spectral slices across age, amplitude versus background variability, participant `Q_i` distributions, reference-versus-expanded protocol comparisons and second-window repeatability. Show Samsung's support failure in place of an unavailable panel rather than omitting the source.

Completion requires:

- Exact 180-day half-open boundaries; 90 unique selected dates; expanded quotas at least 10/block; balanced quotas exactly 15; original-mask threshold at least 120 bins/date. Include boundary fixtures and a capacity vector such as `[10,11,12,17,20,20]` that qualifies only under the expanded rule.
- Deterministic date selections under reordered input, correct earliest qualifying starts, nonoverlapping second windows and correct Dec-1 age rollover. Confirm the January/December sensitivity leaves HR inputs unchanged.
- Global-ID/source isolation and correct unique-person counts; no source pooling, filled-bin qualification or category-midpoint age imputation.
- Direct-OLS versus projected-harmonic agreement; pure-nuisance and known-harmonic checks; null covariance/simulation checks at irregular calendar gaps; finite Monte Carlo summaries and reported optimizer failures.
- Exact band-boundary inclusion, frequency-weighted trapezoid normalization and preservation of negative excess. Verify that band average and selected maximum are separate endpoints.
- Quadratic-model design/test contrast checks, no penalization, correct source multiplicity handling and failed-bootstrap reporting. Changing age centering/scaling must leave fitted values and joint tests unchanged within numerical tolerance.
- Saved-table reproduction of counts, headline estimates and plotted curves; a written accounting of Samsung eligibility, missing metadata and all failed fits.

The final interpretation must distinguish cross-sectional age associations, cohort/calendar differences, recording selection, model-dependent null excess and physical amplitude. A monthly bump or a fitted age transition does not establish menstruation, ovulation or menopause.
