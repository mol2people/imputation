# Recorded-salutation prediction from wearable data: research synthesis

Updated 22 September 2026 after the filling-sensitivity experiment. This is the current synthesis, superseding the earlier HTML/PDF reports and temporal roadmap. It distinguishes saved results from interpretation and proposed work. Review used text artifacts, selected logs and code; no experiments were rerun and no PDFs were opened.

## Assessment

**The repository establishes substantial predictive signal in within-day ch3000 sequences, but has not established its physiological origin or its transportability to participants with missing salutation.** The strongest advance came from temporal representations, not increasingly elaborate summary features. The latest experiment is a useful robustness check: changing the tested filling convention barely changes aggregate AUROC. Its short-gap interventions affect very little input, however, and it is not a general validation of imputation.

The research style is productive: small matched experiments, reusable caches, explicit controls and fast decisions. The main weakness is interpretive overreach. Predictive adjustments become claims of deconfounding; negative concatenation results become claims of redundancy; small filling differences become a claim that the filling question is closed. Better wording and experiments aimed at a specific competing explanation matter more than additional significance calculations on these reused participants.

The target throughout is **recorded salutation, 10 versus 20**, not validated biological sex or gender identity. Here, *wearable-value imputation* means reconstructing missing measurements; *target-label imputation* means assigning a missing salutation. Success at one does not validate the other.

## What the campaign establishes

The [README](../README.md) indexes all 16 stages. The principal findings are:

| Evidence | Result | Scientific reading |
|---|---|---|
| Summary-feature studies | Primary feature-engineering gains at most approximately +0.0034 AUROC; selection gave no consistent gain | The tested summary extensions have limited value; this is not evidence of a prediction ceiling. |
| Broader informative-recording cohort | Values/context 0.7072; recording + values 0.7239 test AUROC | Recording features can complement a value model in that population and representation. |
| Garmin circadian studies | R1b day-level 0.6589 → 0.6782; participant baseline 0.7415 → 0.7513 | Hourly profiles and curve features add useful prediction. The experiment does not isolate cosinor alone. |
| Garmin temporal ladder, ten allocations | Summary ridge 0.7157; MultiRocket 0.8361; HYDRA 0.8442; combined 0.8565 test AUROC | Temporal representations recover substantially more predictive structure on matched participants and windows. |
| Recording-only pilot, three allocations | Mask 0.6166; counts 0.6332 test AUROC | Availability and event-count arrangement carry signal without HR values. |

Sources: [feature engineering](../experiments/artifacts_sq/feng/feat_eng_report.md), [informative recording](../experiments/2026-09-20_informative_recording_features/README.md), [R1b](../experiments/r1b_dateaware_garmin_2026-09-21/results/REPORT.md), [temporal ladder](../experiments/temporal_garmin/results/REPORT.md), [recording pilot](../experiments/recording_structure_garmin/results/REPORT.md). Cohorts, windows and learners differ across rows; compare interventions within their own experiment.

The temporal value-shuffle control drops MultiRocket test AUROC from 0.8361 to 0.6643 while preserving the day's observed-value multiset and mask. It destroys local order and value-to-clock alignment jointly. Recording-only position shuffles give smaller losses, approximately 0.031 for masks and 0.036 for counts. Together these results implicate temporal arrangement; they do not separate physiology, activity, device behavior and sampling effects. AUROC differences are not fractions of information.

Appending recording blocks to the HR model gives −0.0170 validation and −0.0107 test AUROC. That is a negative result for this concatenation and regularization scheme, not proof of redundancy. The older report's claim that mask inputs make MultiRocket bias quantiles degenerate over {0,1} is incorrect: the [vendored implementation](../experiments/temporal_garmin/cache/vendor/multirocket/multirocket/multirocket.py) takes quantiles of convolution responses. A second transform could still be useful, but that argument does not compel it.

## New result: filling sensitivity

The pilot holds 3,848 Garmin participants, their first 40 adequate days, 288 five-minute bins per day, masks/counts, B40 summaries and allocations 0–2 fixed. Every arm refits MultiRocket and a validation-selected ridge head. B40 includes recording features throughout. This estimates sensitivity of the **refitted pipeline**, not a frozen model's response to perturbation.

| Fill | Mean validation AUROC | Mean test AUROC | Mean test Δ versus clock median |
|---|---:|---:|---:|
| Training clock-bin median | 0.8311 | 0.8464 | Reference |
| Overall training median | 0.8268 | 0.8485 | +0.0021 |
| Linear, eligible short gaps | 0.8307 | 0.8470 | +0.0007 |
| Local PCHIP, linear fallback | 0.8303 | 0.8473 | +0.0009 |

Short gaps are complete interior runs of 1–6 missing bins, bounded within the same day. PCHIP uses two original observations immediately on each side, otherwise linear interpolation. Ineligible gaps retain clock-bin medians. [Plan](../experiments/filling_sensitivity_garmin/PLAN.md), [metrics](../experiments/filling_sensitivity_garmin/results/metrics.csv), [experiment report](../experiments/filling_sensitivity_garmin/results/REPORT.md).

**PCHIP provides no practically useful advantage here.** Its paired test gain over linear is +0.0002, versus −0.0004 on validation. Saved test-score correlations are 0.9993–0.9995. There is no reason to prefer its added complexity from these results.

**Exposure limits the interpolation conclusion.** Of 5,089,133 missing bins, only 197,148 (3.87%) qualify for interpolation—approximately 0.44% of all input bins. PCHIP handles 174,198 bins; linear fallback handles 22,950. The other 96.13% of missing bins retain clock-median filling. The global-median arm is the broader test because it changes the filling rule at every missing position, although some resulting values may coincide.

**Retain clock-bin filling as the reference.** Global median has the highest mean test AUROC but lower validation AUROC in every allocation. Its allocation-0 test difference is approximately 10⁻¹⁶, a numerical tie, not meaningful evidence of a third positive result. Across the tested arms and allocations, the largest paired test difference versus clock filling is approximately +0.00325. These are observed differences, not an upper bound on future effects or an equivalence result. Comparing them with the approximately 0.016 cross-allocation SD is not a paired uncertainty analysis.

Stable AUROC also does not mean identical individual rankings: alternatives versus clock median have Spearman correlations around 0.89–0.90 in allocation 1, where clock filling selects alpha 1,000 and alternatives select 10,000. Different regularization plausibly contributes, but the report's attribution of the validation/test sign pattern entirely to alpha selection is unproven; global median also has opposite validation/test signs in allocation 2 with matched alpha. No fixed-alpha control was run, and none is needed merely to report the present result accurately.

The decision is therefore **stop optimizing these short-gap fills on this cohort, while keeping broader imputation sensitivity as the main scientific direction**. Constant fills still create missingness-related boundaries; masks, density, observed values and B40 recording features remain. The results do not show that the signal resides exclusively in observed HR or that missing HR was reconstructed accurately.

## What exactly was the cosinor model?

The implemented model is a three-harmonic, fixed-period regression of hourly HR means:

\[
\bar y_h=M+\sum_{k=1}^{3}\left[a_k\cos(2\pi kh/24)+b_k\sin(2\pi kh/24)\right]+\epsilon_h,
\qquad
\hat\theta=\arg\min_\theta\sum_{h\in\mathcal O}n_h[\bar y_h-f_\theta(h)]^2.
\]

Here h is the integer local-hour bin, n_h its retained event count, and O the observed hours. The components have periods 24, 12 and 8 hours. M is the fitted intercept; A_k = sqrt(a_k² + b_k²) gives each harmonic's amplitude. The reported `cosinor_acro_h` is the maximum of the **full fitted curve** on a 15-minute grid, not the first harmonic's analytic phase. The reported `cosinor_resid_sd` is intended as weighted residual RMS, without a degrees-of-freedom correction.

**Preprocessing before fitting:** upstream code filters valid HR to 25–230 bpm and valid positive start timestamps, applies source exclusions and forms local time using the recorded timezone offset (the original hourly cache substitutes UTC when that offset is missing). Events contribute by their start hour. Hourly HR is an event-count-weighted mean, not duration weighting. R1a pools retained history by participant/hour; R1b fits each participant/day separately. R1b's targeted scan explicitly restricts events to Garmin/source 3; R1a reads a cache without a source column, so Garmin participant membership alone does not establish Garmin-only provenance of its hourly curve.

There is **no detrending, smoothing, z-scoring or interpolation before cosinor fitting**. Missing hours receive zero fitting weight; implementation zeros at those positions are bookkeeping, not zero-HR observations. Afterwards, feature preprocessing handles missing derived features. Coverage residualization is a separate, training-fitted adjustment of already extracted curve features using hourly count shares and total count; it is not detrending the HR series. Event-count weights emphasize densely recorded hours and should not be interpreted as independent-observation precision weights.

Sources: [R1a fitting and residualization](../experiments/r1a_circadian_garmin_2026-09-21/run_r1a.py), [R1b fitting](../experiments/r1b_dateaware_garmin_2026-09-21/run_r1b.py), [R1b scan](../experiments/r1b_dateaware_garmin_2026-09-21/scan_perday.py), [original cache builder](../src/epoch_worker.py).

**Is detrending worth testing? Yes, as a focused sensitivity, not a default correction.** A trend fitted and removed from one 24-hour profile can absorb part of the daily harmonic structure, especially with uneven coverage. Prefer jointly fitting a slow trend and the harmonic terms on chronological observations spanning multiple days, using actual elapsed time. Retain level and trend as separate predictors so the comparison does not simply discard potentially useful signal. Do not infer longitudinal drift from R1a's already collapsed 24-hour profile. The present results do not demonstrate that a trend correction is needed.

Before extending these features, correct the documented residual-RMS bug: NaN weights multiplied by zero still propagate NaN, affecting 35.5% of R1b days according to its report. Use valid-hour weights in the numerator and compare the corrected feature on existing folds. This is a real feature-construction defect, not a reason to invalidate the other cosinor coefficients or the separate temporal experiment. No code correction was made in this review.

## Is naive Bayes warranted?

**Reasonable as a cheap secondary baseline; not a missing prerequisite for the main finding.** Gaussian naive Bayes asks whether class-specific marginal Gaussian feature distributions suffice under a conditional-independence approximation. It is inexpensive and interpretable. Dependence among hourly features and convolutional outputs is a plausible source of misspecification, although within-class dependence has not been quantified here; its probabilities should not be assumed calibrated. [Official model documentation](https://scikit-learn.org/stable/modules/naive_bayes.html).

Use a small, fixed continuous summary block, with the same block evaluated under the existing ridge/RF baselines, the same participant splits and training-only preprocessing. Start with GaussianNB defaults; no broad search. Avoid making a 19,000-column convolutional naive Bayes model the main comparison. Existing summary ridge and RF already show that the temporal gain is not just a weak RF baseline. Priority is below multichannel work and the broader imputation study.

## Revised next steps

1. **Main scientific direction—imputation sensitivity on a larger, more representative dataset.** The required larger dataset is not currently available to this workspace. Include participants with lower recording coverage and the target population with missing salutation, rather than only the strict Garmin cohort. Study long interior gaps, day-edge gaps, missingness burden and sampling density. Separately assess whether alternative target-label imputations change the substantive downstream estimates once the required outcomes and covariates are available. Extra allocations of the current cohort do not resolve this access limitation.
2. **Next feasible direction—multichannel modelling.** The earlier ch3000-only restriction is superseded for future work. Start with HR plus steps/activity, then add sleep/resting-HR channels where definitions and coverage support it. Compare on identical participant/day sets and include availability-only controls; otherwise channel completeness changes can masquerade as predictive gains. Vendor-derived channels require explicit measurement semantics. This is a direction to plan, not a claim of a likely 0.90 AUROC or an executed experiment.
3. **Engineering track—evaluate learned imputers before committing to them.** A small local feasibility test is possible; representative imputation validation remains dependent on broader data. Prioritize TabPFN-TS-style imputation, with TiRex as a secondary candidate, against clock/global medians, linear/PCHIP and a simple seasonal baseline. Preserve original masks and measured values, and never condition wearable reconstruction on salutation labels.
4. **Small optional additions:** the summary-feature GaussianNB baseline and a joint trend-plus-cosinor sensitivity. Correct the known residual-RMS feature before interpreting additional cosinor comparisons. Avoid expanding this into an exhaustive learner or preprocessing search.

For the imputation study, hide originally observed blocks with lengths and clock positions resembling real gaps. Use identical masks across methods, separate training/validation/test participants, and report reconstruction error by gap regime alongside downstream AUROC and changes in substantive analyses. Check whether imputers suppress variability or invent activity peaks. Artificial masking evaluates recovery where measurements were available; it does not establish performance under naturally informative missingness. Keep that distinction explicit.

### TiRex and TabPFN-TS as engineering candidates

**TabPFN-TS is the more direct fit to interior-gap reconstruction.** Published work applies time-index features and a pretrained TabPFN regressor to observed time/value pairs, querying missing timestamps. This supports use of observations on both sides of an interior gap and optional covariates. That literature warrants a trial, not a claim of success on Garmin HR. The forecasting package and the imputation-study adaptation are distinct implementations; pin the chosen version and feature construction. [Imputation study, HTML](https://arxiv.org/html/2511.05980v2), [study code](https://github.com/taharnbl/tsfm_imputation), [official forecasting package](https://github.com/PriorLabs/tabpfn-time-series).

**TiRex is primarily a pretrained forecasting model.** Its documented forecast interface predicts beyond a supplied context; this alone is not a bidirectional interior-gap imputer. A possible adaptation is forward forecasting from the left, backward forecasting on reversed right context, then a predefined blend. That is an unvalidated wrapper proposal requiring its own masked-block check. Edge gaps would use one-sided predictions. [Official TiRex repository](https://github.com/NX-AI/tirex).

No installation or benchmark was performed. With the current no-GPU constraint, benchmark a small fixed set of windows locally for wall time, memory and error before scaling. Neither zero-shot use nor a small number of windows guarantees cheap CPU inference. Better reconstruction may leave classification AUROC unchanged; higher AUROC alone does not establish more faithful reconstruction.

## Material limitations and verification record

- **Population and reuse:** current temporal results describe 3,848 selected participants with substantial histories. The first 40 adequate days need not be consecutive or fit into 40 calendar days. Existing validation/test folds have repeatedly informed development; these are exploratory comparisons, not independent confirmation or estimates of performance where salutation is missing.
- **Mechanism:** filling, sampling and recording behavior remain entangled with measured HR. Coverage residualization is predictive adjustment, not causal deconfounding. There is no measured numerical performance ceiling.
- **Hyperparameters:** the ten-allocation temporal benchmark used a narrower alpha grid than the later pilots. The combined model's allocation-0 improvement to 0.8784 in the [extended-grid addendum](../experiments/temporal_garmin/results/alpha_addendum.md) is promising but not a revised ten-allocation estimate.
- **Latest-run evidence:** all 12 metric rows are present; the saved fixture/provenance checks pass. Runtime reference comparisons differ by at most approximately 5.6×10⁻⁸, below the 10⁻⁶ tolerance, with matching alphas and AUROCs. The report's zero differences refer to saved-score comparisons, so use “reproduced within tolerance” rather than claiming exact computation throughout. Recorded allocation times total approximately 7.7 minutes, excluding setup; logged peak RSS is 3,817 MiB.
- **Reporting interruption:** the saved log completes all allocations, then fails in report generation because predictions use `allocation`, not `alloc`. The current script uses the corrected column and report/provenance files exist. This supports completion of fits, but the saved log does not establish an uninterrupted successful end-to-end command. The review did not rerun report generation or recompute metrics from participant predictions.

Latest evidence: [metrics](../experiments/filling_sensitivity_garmin/results/metrics.csv), [provenance](../experiments/filling_sensitivity_garmin/results/repro.json), [run log](../experiments/filling_sensitivity_garmin/cache/run_stdout.log), [implementation](../experiments/filling_sensitivity_garmin/run_filling.py). Historical experiment outputs are preserved; the interpretation above supersedes their stronger causal, redundancy and blanket robustness claims.
