# Monthly descriptors and downstream recorded-salutation prediction

Date: 2026-10-20. **Plan only:** no monthly extraction, transforms, fitting or inference has run for this extension. This allocation-0 pilot is exploratory because these cohorts/test sets have already been inspected. Existing artifact facts are in [READINESS.md](READINESS.md).

## 1. Question and scope

Do low-dimensional monthly HR descriptors improve recorded-salutation prediction beyond the existing HR representation, recording patterns and coarse age, and does their incremental value depend on within-day filling?

The target remains administrative salutation: 10 → y0, 20 → y1. Predictive utility is not proof of biological cycles. **Feature augmentation is primary.** Periodicity-aware filling and fitted-rhythm removal are later, separate experiments. This plan complements the [age-accounted infradian plan](../female_infradian_cycles_v2/PLAN.md); it does not replace its noise-calibration or physical-amplitude validation.

## 2. Cohorts, splits and matched exposure

Analyze Apple and Garmin separately. Retain existing allocation-0 assignments:

| Source/cohort | Before eligibility | Train / validation / test |
|---|---:|---:|
| Apple full cohort | 11,139 | 7,797 / 1,671 / 1,671 |
| Garmin full-history binning pilot | 1,000 | 700 / 150 / 150 |

Use 5-minute source-specific channel-3000 caches. A qualified date has at least 120 original observed bins; this does not establish continuous ten-hour wear. Select the first qualifying `[start,start+180)` calendar window with at least 15 qualified dates in each of six successive 30-day blocks, with candidate starts on qualified dates. Uniformly select 15 dates per block without replacement using seed 20261020, participant-ID-derived streams and sorted dates. Freeze the precise RNG specification before implementation. No labels or HR magnitudes choose dates/windows.

All arms use the same eligible people, 90 dates, masks, windows and allocation labels. Apple feasibility found 4,631 first-window eligible participants among the v1-screened population; allocation-0/cell retention remains unaudited. Garmin eligibility is unknown. Its elderly y0 test count is only four before this screen. Do not substitute v1-screened counts for monthly-window eligibility. Audit six age×salutation cells before fitting and report exclusions.

Strict Garmin holdout allocation 0 does exist: 2,696/579/573 participants in the r1b/recording/filling protocol. Its ready HR cache contains only the first 40 selected days. A later larger replication requires full-history sequences for that cohort while retaining its assignments. The strict k5 production bundles fit all 3,848 participants and cannot serve as held-out allocation-0 models. Full-history monthly features appended to a 40-day baseline constitute an extra-history study, not this primary comparison.

## 3. HR baseline and filling

`H_fill` mirrors the compatible Apple/Garmin 5-minute representation: 305 static features plus 18,816 MultiRocket features, computed using these same 90 dates. Static features are across-day means/sample SDs of native daily mean, median, SD, minimum, maximum, event count, coverage hours and occupied hours; weekday-minus-weekend HR; and 288 original-mask frequencies. MultiRocket transforms each daily base/difference series and pools mean/sample SD across dates. No chronological between-day feature precedes the new monthly block. Preserve native statistic semantics and original recording information.

Compare clock-bin median filling with original-anchor short linear interpolation plus clock fallback. Linear filling covers interior 1–5-bin gaps with observed bounding-center separation≤30 minutes, never midnight. Fit clock/global-fallback templates only on eligible source-training participants' selected dates, without age/salutation conditioning. Preserve observed HR and original masks; filled bins never requalify dates. Global/PCHIP arms do not expand this primary grid.

Use identical training-sampled transform days and seeds across fills, but fit each fill's own transform biases and head. Within a fill, freeze its exact H matrix and preprocessing across baseline/monthly augmentation. Thus fill comparisons include refitted-transform effects, while augmentation compares a fixed base representation.

Existing runners did not persist reusable participant matrices. Controlled re-extraction and train-only fitting are required; saved test scores alone cannot implement augmentation. See the [Apple fill/mask](../apple_fill_and_mask_by_recording/PLAN.md) and [Garmin filling](../filling_sensitivity_garmin/PLAN.md) protocols for antecedents, not identical new evaluation settings.

## 4. Monthly HR descriptors and controls

Compute `R_obs` from **original observed-bin HRp10**, independently of H filling, on the same actual 90 dates. Fit baseline intercept, linear elapsed time, weekday indicators and four nonoverlapping six-hour original clock-coverage fractions, with deterministic redundant-column removal/rank checks. Add sin/cos harmonics for 21 periods `25,25.5,…,35` days. Select the maximum partial fit improvement:

`S = max_T (RSS_0−RSS_T)/RSS_0`, `A = sqrt(βcos²+βsin²)` bpm.

The three fixed features are A, S and selected T. Save boundary hits diagnostically. Phase is excluded because participant origins are arbitrary. Do not use nominal p-values, “significant-cycle” flags or cyclic-person selection. Rank failure/zero residual variance produces prespecified missing status; retain participants and use source-training median imputation, reporting failures by age. Put monthly-HR fit-availability and zero-variance flags in shared Q, so both B and B+R receive the same data-quality information; the augmented arm must not gain merely from a new missing-fit indicator. Undefined T remains missing until training-median imputation. Do not silently assign T=28.

These label-independent within-person descriptors may use test participants' predictor histories for retrospective salutation classification; they may not use test labels or cohort-fitted parameters. This is not future-HR forecasting. The v1 full-history table has different exposure/normalization and is not a drop-in feature source. Validated null-adjusted E may be added later as a separate sensitivity; this small ML screen does not require 999 participant null simulations or claim biological significance from raw searched features.

Freeze Q before test inspection. Its nonperiodic p10 controls are mean, sample SD (ddof=1), elapsed-time OLS slope and linear-only residual SD `sqrt(RSS_linear/(90−2))`, where the detrending model has intercept and linear time only. Observation controls are four original clock-coverage means, actual selected-date span and longest within-window gap; mean occupied-bin count is derivable from the coverage fractions and need not duplicate them. Add A/S/T descriptors for daily occupancy and nocturnal-mask fraction using the same 90 selected dates and 21-period grid, treating them as sampling-periodicity controls. **Resolved handoff clarification:** these recording outcomes use intercept + linear elapsed time + weekday effects only, excluding the four clock-coverage fractions. Daily occupancy is their average and nocturnal-mask fraction is the first fraction; including them would deterministically eliminate the recording outcomes' residual variation. The HR R baseline retains all four clock-coverage fractions. Reuse already identical H columns instead of duplicates; save the final ordered schema. Q includes low-tail variability and sampling geometry so R does not merely introduce those measurements.

Age is a three-level categorical one-hot block; exact age is unavailable. `B_fill = H_fill + Q + age` is the primary coarse-age-conditioned baseline. R and Q remain identical across fills; only H changes.

## 5. Eight heads per source

| Head | Purpose |
|---|---|
| H_clock | Legacy-feature reference on the new matched exposure |
| B_clock | Primary controlled baseline |
| B_clock + R_obs | Primary augmentation |
| B_linear | Filling sensitivity baseline |
| B_linear + R_obs | Filling sensitivity augmentation |
| B_clock without age | Unconditional diagnostic baseline |
| B_clock without age + R_obs | Age-proxy check |
| Age only | Diagnostic |

There are two transform fits and eight core heads per source; runtime has not been measured. Feature matrices are shared within matched comparisons. A later `R_fill` calculated over completed 288-bin days changes the daily endpoint and must be compared separately with H fixed; it is not substituted for R_obs here.

An optional out-of-band diagnostic adds one `B_clock+R_control` head using A/S/T from 21 equally spaced periods over 38–56 days. Decide whether to include it in the pre-run freeze, before validation or test results from this pilot are inspected. This gives eight core heads plus one diagnostic if included. Never add it in response to a validation/test increment while retaining this frozen-arm protocol. It checks flexible-search capacity, not a guaranteed biological null; do not choose a favorable band.

## 6. Fitting and selection boundaries

Use a new common Ridge grid `{100,500,1000,5000,10000,100000}` for every head. Select alpha by source-validation AUROC, using the exact existing Apple `np.isclose(atol=1e−12)` helper/default relative tolerance and larger-alpha near-ties. This does not reproduce every historical Garmin grid. No test-selected v1 alpha is carried over and there is no train-plus-validation refit.

Fit feature imputation, variance filtering and scaling on eligible source-training participants only. Freeze H preprocessing within paired arms; preprocess Q, R and age blocks separately using training data. Record shared-Q missingness/status indicators and retained columns. No PCA or preprocessing may learn from all users. Source-training template fitting replaces the descriptive v2 subject-cross-fitting scheme for this supervised prediction task; validation/test participants never train templates.

Validation chooses alpha only. Evaluate test after freezing eligibility, schema, arms and selection. Raw Ridge scores support the primary AUROC estimand. No probability calibration or Brier claims are needed; a later probability assessment would require a source-validation-fitted calibrator.

## 7. Evaluation and interpretation

For each source, the primary increment is

`δ_clock = AUROC(B_clock+R_obs) − AUROC(B_clock)`.

Use identical test IDs for all comparisons. Run 2,000 participant bootstrap resamples stratified by age×salutation, with the same draws across all arms. Report 95% pointwise intervals and 97.5% Bonferroni per-source intervals for the **two primary source contrasts**, giving a nominal conditional-bootstrap 95% family target. This is approximate inference conditional on fitted models; training/selection uncertainty is excluded.

Secondary quantities are δ_linear and `η = δ_linear−δ_clock`, using joint four-score resampling. Report both fills, not whichever works best. Report overall AUROC, three within-age AUROCs, and `Σ_a w_a AUROC_a` excluding cross-age ranking pairs. Freeze common weights from pooled eligible **training** age counts across sources before test inspection. If an age stratum lacks either label, declare its AUROC/complete weighted summary not estimable; do not silently reweight. Very small Garmin cells remain unstable even when computable and cannot establish age independence. Expanded strict Garmin history may be necessary.

Report unconditional and coarse-age-conditioned increments, age-only performance, counts/retention and recording/exposure distributions. Optional balanced accuracy requires a validation-selected threshold applied unchanged to test, never a test median.

A gain disappearing within age or with Q can reflect age/sampling proxies. A retained gain still cannot eliminate within-category age confounding. A null increment means no marginal utility for these features/model/history, not absence of cycles. Positive η indicates a prediction/imputation interaction, not better physiological reconstruction. The v2 physical-fidelity validation remains separate.

The primary H representation plus explicit recording controls is not the strongest earlier HR/temporal-mask stack. A claim of improvement beyond the best existing architecture would require a separately declared matched-90-day strong-baseline follow-up: recompute HR and temporal-mask scores, choose stacking architecture/weights on validation only, then compare that baseline with its monthly augmentation on the same participants. Older full-cohort stack scores are not a substitute. Defer this check rather than enlarging the primary factorial; until then, limit conclusions to the specified B baseline.

## 8. Later experiments and deliverables

After freezing this pilot's results, a new factorial experiment may compare observed/clock/linear/phase-aware filling. Fit phase only from original anchors within training/temporal boundaries and validate known-truth recovery first. Fitted-rhythm removal is a separate refitted-downstream ablation with matched data and phase-shuffled/distortion controls; AUROC subtraction is not a causal signal partition. Neither extension is implemented here.

Do not expand allocations automatically. Review the pilot first; if Garmin age support fails, prioritize full-history strict-cohort allocation 0 rather than claim replication from sparse cells.

Future outputs: allocation×age×label eligibility, selected-date/mask hashes, ordered feature definitions, preprocessing/transform artifacts, all train/validation/test scores and alphas, paired/age/control metrics, runtime/cost and reports generated from saved tables. Checks must establish split/test-ID parity, original-mask/observed-value invariance, exact date matching, source-only fitting and bit-identical H blocks across augmented pairs. Reconcile the newly fitted primary baseline against its own saved features; old full-cohort scores are context, not parity targets for changed eligibility/exposure. This task authorizes documentation only, not implementation or execution.
