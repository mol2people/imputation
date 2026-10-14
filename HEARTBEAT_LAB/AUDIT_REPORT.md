# Evaluation risks and visual diagnostics for the impute sex project

The temporal models show substantial predictive signal within the selected Garmin
cohort. The main unresolved question is how much transfers to later or differently
recorded customers. The historical 0.9264 endpoint is not a clean estimate of that
quantity: some held-out customers influenced its fitted representation, the same
cohort supported repeated model development, and histories varied greatly in length.

This review focuses on the strongest temporal experiments, their preprocessing and
splits, and the new Heartbeat Lab implementation. Earlier RF/day-level results were
reviewed through their reports and selected code; this is not an exhaustive audit of
every historical script. Observations below distinguish confirmed problems from
plausible explanations. Original experiment files have not been changed.

## Findings requiring correction or qualification

### Historical all days preprocessing crossed allocation boundaries

**Confirmed, high impact on evaluation interpretation.** In
[`day_scaling_all.py`](../experiments/temporal_garmin/day_scaling_all.py), Phase B
fits fill values and MultiRocket biases using allocation-0 training customers,
then reuses that representation under all ten allocations. Directly comparing the
saved membership files shows that, for allocations 1–9, **380–413 of 573 test
customers** were in allocation 0's transform-training set, mean **402.4 / 573,
or 70.2%**. Validation overlap is 383–423 of 579 customers.

The mechanism is unsupervised exposure to held-out HR, not direct use of their
labels to fit the transform. Its effect on AUROC has not been isolated and need
not be large or have a known direction. It does invalidate a fully inductive
interpretation of those nine endpoint evaluations. Allocation 0 does not have
this particular problem. The main 40-day implementation fits fill and kernels
within each allocation; do not incorrectly extend this defect to all its fits.

**Smallest useful correction:** fit the entire representation within each split,
or report the endpoint as a transductive exploratory analysis. Heartbeat Lab's new
bundles perform train-only fitting. This follows the standard rule that fitted
preprocessing must exclude evaluation observations, including transformations
that do not use labels. [Scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

### Shared selection and reused customers limit independence

**Confirmed protocol; optimism magnitude unknown.** `run_temporal.py` selects alpha
on allocation 0 validation and carries it into other allocations. Between **80 and
98 of each later allocation's 573 test customers** belonged to that original
validation set. This is a shared hyperparameter-selection design, not fully nested
selection for each allocation. It differs from the unsupervised endpoint issue above.

The reports also document reusing customers/folds across successive feature,
representation and filling experiments. A new random split of those same people
does not make a previously explored dataset an untouched external test cohort.
This is a risk of adaptation to an evaluation set even when individual fits keep
customers disjoint. Model-selection overfitting is distinct from training-set
overfitting. [Cawley and Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html)

**Next test:** freeze the full recipe and evaluate later, previously unused
customers. For development, repeat the complete fitting/selection procedure in
outer customer folds. Do not globally fit kernels before cross-validation.

### Some project interpretations are stronger than their evidence

**Confirmed wording problems.** The root README calls 0.9264 a floor and treats
80 adequate days as a cohort gate. The scan selects the existing adequate-day
pool; 80 is its observed minimum. The scaling script asserts that this pool can
support k=80, rather than filtering customers with a newly defined 80-day gate.
The fitted cohort remains strongly selected by earlier eligibility rules, but
the eligibility description should preserve the actual sequence of gates.

Untuned regularization does not establish a lower bound on future-population
performance. Extrapolating a learning-curve ceiling above the tested fixed budgets
is a heuristic, not an identified asymptote. The same-statistics comparisons also
do not establish that recording behavior or engagement cannot explain temporal
signal. Preserve the reported numbers; qualify those interpretations.

**Smaller confirmed inconsistencies:**

- The notebook maps **10→F, 20→M**. Current cached labels give median adequate-day
  counts **F=467, M=482**, whereas the day-scale report describes M=467/F=482.
  That narrative reverses the mapping used by the current notebook and app.
- `run_temporal.py` contains `assert np.array_equal(mask_pu, mask_pu)` as a shuffle
  gate. This assertion is tautological. It does not demonstrate a wrong shuffle;
  it fails to test the claimed invariant. Test pre/post masks and within-day
  observed-value multisets instead. The new perturbation tests do this.
- `scan_bins_all.py` assumes extending beyond the first 40 days does not change
  the span distribution, without repeating the span audit. This is an unverified
  assumption, not evidence that intervals are mishandled. Audit spans by period
  before attributing behavior to physiological timing.

## What the new runs establish

These new models use all 1,635,698 adequate days from the same 3,848 customers,
with 2,692/578/578 customers in train/validation/test. Every learned transformation
is fitted on training customers. These are new exploratory fits, not a numerical
replay or external validation of the historical experiment.

| Full history model | Validation AUROC | Test AUROC |
|---|---:|---:|
| Summary ridge | 0.7247 | 0.7394 |
| MultiRocket with clock-bin median fill | 0.8962 | 0.9177 |
| MultiRocket with global median fill | 0.9034 | 0.9225 |

This supports substantial predictability under a fit that avoids the endpoint's
specific preprocessing exposure. It does **not** quantify how much the old exposure
inflated its score: the splits, seeds and fill-estimation windows also differ.

The paired global-minus-clock test AUROC difference is **+0.00477**, with a
1,000-replicate stratified customer bootstrap percentile interval
**[−0.00467, +0.01403]**. This does not establish a filling advantage. That interval
conditions on the selected cohort and already-fitted models; it excludes model
selection, retraining and transport uncertainty.

Additional cheap checks used no HR values: a two-feature first/last-date ridge
achieved test AUROC **0.5123**; six recording/coverage summaries achieved **0.4799**;
their combination achieved **0.4840**. Permuting training-customer labels 100 times
for that combined nuisance model gave mean test AUROC **0.4968**, with central 95%
of runs **[0.4637, 0.5323]**. This is a wiring check for this small control only.
It does not rule out nonlinear recording shortcuts: the existing recording-structure
experiment obtains approximately 0.62–0.63 using temporal masks/counts.

Reproducible aggregate evidence is under `artifacts/audit/`, generated by
[`04_audit_project.py`](scripts/04_audit_project.py), with model metadata and
held-out predictions in `artifacts/models/`. Customer artifacts stay gitignored.

## How to interpret the visual stress tests

The app keeps the customer, fitted model and unselected days fixed. Randomness is
within the selected dates. Each run uses the same modified inputs across selected
models. The band is a conditional randomization distribution, not a population
confidence interval and not proof of overfitting or correctness.

| Intervention now available | What it changes | What a score change can reveal | Main limitation |
|---|---|---|---|
| Raw HR shuffle within day | Value-to-timestamp assignment | Dependence on ordering, timing and aggregation | Rebinning can alter the distribution of bin means |
| Shuffle observed five-minute means | Bin-value placement; original summaries/mask fixed | Dependence on bin order/clock alignment at fixed bin-value multiset | Jointly disrupts local order and clock alignment |
| Empirical daily resampling | Replacement samples from that day's raw values | Sensitivity to sampled values and lost temporal dependence | IID draws discard dependence; moments need not match exactly |
| Observation thinning | Independent loss of raw readings | Sensitivity to recording density and quality gates | Removed events can remove entire days; inspect retention |
| Circular clock shift | Time alignment of values and mask together | Sensitivity to clock placement | Moving the mask confounds HR phase with recording phase |

The app overlays original scores, median perturbed scores and 5th–95th percentile
bands. It also shows score-change histograms, day-retention fractions, an example
generated day and its value distribution. Failed quality gates create missing
predictions, not replacement scores. Histograms condition on surviving selected
days, so informative dropout must be read alongside them.

An extreme response to an implausible shuffled series can be expected from a
well-generalizing model. Stability can likewise reflect a model that ignores a
useful variable. Neither direction alone diagnoses overfitting. To separate
short-range dynamics from clock placement, a useful next control is a block
shuffle or circular rotation of values with the mask fixed; retain the current
mask-moving shift as a separate intervention.

## Time based validation and retraining

The app can launch a local background fit with a customer budget, day budget,
model, fill and seed. Its temporal protocol uses **first adequate recorded day**,
not account enrollment:

1. Training customers first appear before cutoff A; their usable days are < A.
2. Validation customers first appear in [A,B); their usable days stay in [A,B).
3. Test customers first appear at or after B; their usable days are ≥ B.

Customers never cross splits. Histories with fewer than two usable days are
excluded and recorded. Cutoffs operate on the original start-assigned local days;
intervals crossing midnight retain the historical coverage convention. The
retrospective strict-cohort selection still uses eventual history, so this is not
a fully prospective recruitment simulation.

A 200-customer, at-most-40-day pilot with A=2021-08-10 and B=2021-12-22 retained
125/33/32 customers after truncation. Summary ridge scored **0.714 validation /
0.549 test**, and MultiRocket **0.571 / 0.565**. These tiny tests also change the
training population and effective day budget. **Do not attribute the drop from
the full-history random split to temporal drift alone.** A matched-budget, larger
rolling-cutoff experiment is needed, with class counts and day counts alongside
each AUROC.

Measured on this machine: the 200-customer summary run took **0.57 s** and
MultiRocket **2.63 s**, excluding interpreter/import startup and with warm compilation caches. Full-history
MultiRocket bundle builds took approximately **697–705 s each while two ran
concurrently**, measured from artifact directory creation to metadata completion.
Those are workload-specific measurements, not a general latency promise.

Later days of customers already used for training answer a different question:
stability for known customers. That design should be named separately and must
not be presented as generalization to new customers. It is not the app's temporal
validation protocol.

## Next visual checks in priority order

1. **Matched day budgets across calendar cutoffs.** Overlay learning curves for
   random customer splits and several forward splits at the same k, with the same
   feature budget and comparable training size. Show exclusions and label prevalence.
2. **Customer-label permutation with complete refitting.** Permute one label per
   customer before joining days; repeat the full selection procedure if that is
   what the claim concerns. Plot observed AUROC against the null distribution.
   Do not shuffle identical labels within each customer's rows: that changes
   nothing. Scikit-learn's `groups` permutation restriction must be interpreted
   carefully for this reason. [Permutation-test API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html)
3. **Representation-level recording controls.** Compare HR, mask/count-only and
   matched mask-swap controls on the same folds. Plot errors against observed bins,
   gaps and activity-shaped bursts. A weak linear nuisance model cannot exclude
   these nonlinear shortcuts.
4. **Customer-level error strata.** The app now shows score distributions by F/M,
   score versus available history, and history-quartile AUROC/error tables. Add
   age/BMI, calendar period and device strata when their provenance is available.
   Report subgroup sample sizes and uncertainty; tiny groups can look extreme.
5. **Validation-selected calibration and abstention.** Ridge outputs are scores.
   The app's 0.5 F/M threshold is a display convention, not a calibrated operating
   point. Fit calibration/thresholds on disjoint validation customers and inspect
   calibration and coverage on a frozen evaluation set.

The individual-day plot applies a history-trained model to a singleton day,
including undefined between-day summaries. Its value is diagnostic. If daily
classification is the intended deployed task, add the project's genuinely
day-trained model as a separate bundle and evaluate that estimand directly.
