# Minimal experiment: predicting recorded SALUTATION from epoch data

Draft protocol, 2026-09-18. This document plans the experiment; no splits, features, or fitted models have been generated. Confirmed decisions: **80% training / 10% validation / 10% test**, participant-level assignment, **salutation 10 versus 20 only**, and **exclusion of sources 38, 46 and 48** from both daily and epoch records. Code 30 is excluded for this experiment.

Fit **one random forest with default learning parameters**, with a fixed random seed for reproducibility. There is **no hyperparameter tuning, cross-validation, model comparison or ablation study**. Use **10 bootstrap resamples** of test participants for a rough assessment of metric variability.

## 1. Question and target population

Estimate how accurately epoch-derived features predict a previously unseen participant's recorded SALUTATION among participants represented in `15Sep_2230_PG_type65-66.csv` after source exclusions who meet the prespecified epoch-coverage requirement using allowed sources and have an observed salutation of 10 or 20.

This is retrospective classification of recorded salutation. It does not establish prediction of biological sex, gender identity, future participants from another population, or participants with missing salutation. The latter would require a subsequent transportability assessment.

The prediction and evaluation unit is the participant. All records, devices, days, and any derived windows belonging to a participant must remain in the same split.

## 2. Data and label provenance

| Input | Role |
| --- | --- |
| `15Sep_2230_PG_type65-66.csv` | Defines the initial participant set and daily source profile; daily measurement values are not model predictors. |
| `13Aug_1222.csv` | Supplies the actual `salutation`, joined by `user_id`. |
| `df_whoOneAverage.csv` | Supplies `age_group` and `bmi_grp`, joined using `user == user_id`. |
| `out/<user_id>.csv` | Supplies epoch predictors: `customer`, `startTimestamp`, `endTimestamp`, `type`, `longValue`, `source`, `timezoneOffset`. |

Use `salutation == 10` as outcome 0 and `salutation == 20` as outcome 1. Store the original codes alongside the encoding. The existing notebook maps these codes to F and M; retain code-based names in the experiment artifacts so that the target remains explicit.

Do not substitute the linked table's `gender` for salutation. In the current binary candidate cohort after daily source filtering, 60 participants have salutation 20 and linked gender Female; 25 have salutation 10 and linked gender Male. Retain their original salutation and flag the disagreement. Its cause and the relative timing of the two exports are unverified.

Use the existing age and BMI group labels for allocation. The linked column named `age` contains birth years, not ages; do not use it as a numerical age. Group boundaries and their reference date must be documented from provenance before interpreting them as clinical age/BMI bands. Numerical BMI is not present in this linked table, so balancing `bmi_grp` cannot guarantee equal mean numerical BMI.

## 3. Eligibility and observed cohort counts

Apply source exclusions at the record level before deriving eligibility or source profiles. Remove daily and epoch rows whose source is 38, 46 or 48. Retain a mixed-source participant only if they still have an allowed-source daily record and usable allowed-source epoch data. Do not remove an otherwise eligible participant solely because they also contributed excluded-source records.

These are observed file/label inventory counts after daily source filtering, before a complete epoch validity, allowed-source and minimum-coverage scan. They are candidate counts, not final analysis-cohort counts:

| Step | Participants |
| --- | ---: |
| At least one daily type-65/66 record, before source exclusions | 27,256 |
| At least one daily record after excluding sources 38/46/48 | 26,939 |
| Nonempty corresponding epoch file, before inspecting its allowed-source content | 25,806 |
| Matching record in the original salutation export | 25,795 |
| Salutation 10 or 20, excluding eight code-30 records | **25,787** |
| Salutation 10 | 9,662 |
| Salutation 20 | 16,125 |

Daily filtering removes 23,052 rows (20,575 from source 46 and 2,477 from source 48) and leaves 317 participants with no allowed-source daily record. Source 38 is absent from this daily export but is still excluded from epoch processing. Of the remaining daily participants, 1,133 have no nonempty epoch file. Among the 25,806 with a nonempty file, 11 lack a matching salutation record. Report these exclusions and resulting coverage; do not manufacture epoch predictions for them or fill labels from `gender`.

Before allocation, stream the candidate epoch files, remove excluded-source records, and check schema, parsability, participant identity, timestamp units, event types and usable recording coverage. Neither a nonempty file nor a single valid measurement establishes eligibility. Empty and insufficiently observed recordings are excluded from all three primary splits before balancing.

Maintain separate exclusion flags for missing files, zero-byte files, header-only files, unreadable/invalid files, no valid allowed-source measurements after filtering, and insufficient recording coverage. Save all applicable flags and use a fixed precedence when tabulating mutually exclusive cohort-flow counts. Do not turn an empty recording into an all-imputed feature row.

The full `out/` inventory contains 3,282 zero-byte files, 119 nonempty files of at most 200 bytes, and 458 files of 201–1,024 bytes. These counts cover all exported users, not just the candidate cohort. A deterministic sample of small files included one with only 18 records on 14 distinct UTC dates (assuming the apparent millisecond timestamp encoding), and another nonempty file with no remaining records after source filtering. These observations support a content/coverage audit; file size and observed-date counts alone are not eligibility rules.

**Provisional primary coverage gate:** require at least **14 adequately observed days** in at least one predeclared core measurement channel. Days may be nonconsecutive. Fourteen is a pragmatic starting design choice, not a validated threshold for reliable salutation prediction. A date with an isolated sample does not qualify automatically. Do not pool unrelated channels to manufacture the required day count.

Before splitting, the channel/source audit must identify the qualifying core channel(s) and freeze a concrete adequate-day rule for each supported representation. For sampled signals, specify valid measurement coverage, admissible gaps and, where meaningful, expected sampling completeness. For interval data, use the union of valid intervals; do not count overlapping devices twice or equate the first-to-last timestamp span with observed duration. For event-driven channels, a low event count may be valid and does not itself establish poor coverage. Rules may reflect documented source-specific recording semantics, but must not be chosen to equalize class retention or improve model performance. The gate is not executable until these definitions are recorded in the configuration.

The epoch inventory must report raw and retained event counts, distinct observed dates, adequate-day counts by core channel, relevant coverage/gap summaries, source/channel availability and exclusion reasons. Freeze the quality rules without examining predictive performance or epoch-value associations with salutation. Then report retained/excluded counts by salutation, source, age group and BMI group to show how coverage restrictions change the target population.

Participants failing the primary coverage gate remain in the exclusion inventory and do not enter training, validation or testing. No additional recording-length sensitivity analysis is required for this minimal experiment.

Use all valid epoch history in the frozen export for this retrospective experiment. Record each participant's first/last observation and observed days. The daily export spans 2020-04-08 through 2023-01-02; do not assume the epoch export has the same bounds.

Eligibility, technical validity and minimum-coverage rules must be fixed before allocation and before fitting the model. Do not exclude participants because their physiology or predicted label appears atypical. The final sample size can be smaller than 25,787 after epoch source filtering, technical validation and sparse-recording exclusions; calculate split counts and balance diagnostics from that final cohort.

## 4. Source representation and balancing variables

For each participant derive from the daily file after removing sources 38, 46 and 48:

- `daily_source_set`: the sorted set of observed source codes.
- One binary membership indicator per observed source code.
- `daily_primary_source`: the source with the largest number of distinct observed dates, with the smallest numeric code breaking ties. Count dates, not rows, so repeated measurements and the presence of both daily types do not inflate coverage.
- `daily_multisource`: whether more than one source is present.

The current binary candidate cohort has 3,328 participants with multiple remaining daily sources. Preserve them. The remaining daily source codes are 2, 3, 6, 7, 9, 13 and 19.

The epoch inventory must additionally produce allowed-source membership, a primary source based on distinct valid epoch days, and a multi-source flag. Use the same minimum-code tie rule for primary source. Compare daily and epoch source profiles. Balance the principal epoch source indicators as well as the requested daily source variables, because daily-source balance alone does not ensure balance of the actual predictor data.

The required balance variables are salutation, daily source, age group, and BMI group. Preserve their associations with salutation across splits by additionally targeting salutation-by-source, salutation-by-age-group, and salutation-by-BMI-group distributions. Do not balance salutation classes against each other: preserve the cohort's observed class prevalence.

## 5. Allocation: closest feasible balance, with explicit diagnostics

Literal equality of all descriptive statistics cannot generally be achieved. An 80/10/10 allocation requires compatible integer counts, and strata with fewer than three participants cannot be represented in all three splits. Non-significant balance tests would not establish equality.

This is already a concrete issue: the current 25,787-person candidate cohort has **475 occupied salutation × daily source-set × age-group × BMI-group strata**. Of these, 184 contain fewer than three participants, and 302 contain fewer than ten. Even replacing source sets with the primary daily source leaves 13 of 148 strata with fewer than three participants.

Use a seeded constrained allocation on a table with one row per participant:

1. Fix final split totals using largest-remainder rounding of 80/10/10; ties go to validation before test. Require disjoint membership and complete coverage of the eligible cohort.
2. Jointly allocate salutation counts to these totals, keeping each count within floor/ceiling of its proportional target where feasible. Do not round each class independently without checking the split totals.
3. Allocate integer counts from covariate strata. Use hard total/class constraints, then lexicographically minimize imbalance in: (a) one-variable category margins, including daily/epoch source membership, primary source, and multi-source status; (b) salutation-by-covariate distributions; (c) joint strata with at least ten participants. Full source sets can remain sparse; their counts do not need to be positive in every split. Also balance pooled rarity groups (joint-stratum sizes 1–2, 3–9 and at least 10), using seeded tie-breaking within the constrained allocation. This avoids a joint-cell rounding objective that systematically pushes singleton profiles into training. Report rare-profile representation explicitly.
4. Initially request category counts within floor/ceiling of their proportional targets. If those simultaneous constraints are infeasible, retain total/class constraints and minimize explicitly recorded deviations in the stated priority order. Do not silently drop rare categories, oversample participants across splits, or merge outcomes.
5. Randomly assign participants within the allocated strata using a fixed seed, provisionally `20260918`. A practical implementation can aggregate identical indicator profiles before integer optimization. The aggregation key must include every categorical allocation indicator, including primary source and the epoch source profile; a source set alone does not determine a primary source. Save the solver status and optimality gap if a solver is used; report a heuristic result as such if optimality is unproven.

For illustration only, allocating all 25,787 candidates would give the following compatible totals and class counts. These are pre-coverage counts, not an executed split or the expected final sample size; recompute them after applying the coverage gate:

| Split | Salutation 10 | Salutation 20 | Total |
| --- | ---: | ---: | ---: |
| Train | 7,729 | 12,900 | 20,629 |
| Validation | 966 | 1,613 | 2,579 |
| Test | 967 | 1,612 | 2,579 |

Before freezing the manifest, produce a balance report with counts/proportions by split for every category, the full joint table, and all pairwise absolute proportion differences. For salutation-by-covariate tables, report both joint proportions and distributions conditional on salutation.

The practical target is **at most one percentage point pairwise difference in each main category's proportion**, with proportions as close to integer equality as feasible. This is a design tolerance, not an established statistical cutoff. Report residuals and infeasibility explicitly if it cannot be met. Sparse conditional subgroups get their actual counts and discrepancies, not an unsupported claim of balance.

Also report birth-year distributions and epoch observation length/calendar coverage as supplemental diagnostics. Use mean/SD, median/IQR and standardized mean differences for continuous quantities; an absolute SMD below 0.05 is a supplementary target, not a proof of identical distributions.

Covariate-based allocation may use labels for stratification. It must not use epoch-value/label associations, classifier results, or test performance to select an attractive split. Freeze the participant manifest before model development. Participant separation addresses repeated observations from the same subject; see the [scikit-learn discussion of grouped evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

## 6. Epoch processing and feature construction

Stream one participant file at a time; the full linked directory is approximately 283 GiB. Generate a compact participant feature table rather than concatenate raw exports into memory. Keep raw files immutable and write derived artifacts inside this project.

First document the meanings, units and source-specific semantics of types 3000, 3001 and 3002. The export establishes their codes, not their physiological interpretation. Verify timestamp units, interval versus point-event semantics, timezone-offset units/sign, and whether any vendor-derived channel uses a user-entered sex/salutation in its calculation. Such a channel could make the prediction circular; exclude it from the measurement-based primary feature set until provenance is resolved.

Filter sources 38, 46 and 48 before any epoch summaries, counts, coverage measurements or feature extraction. Technical processing must check duplicate events, invalid timestamps, interval durations where applicable, missing values, and implausible values under documented channel-specific rules. Overlapping intervals or parallel devices must not automatically be summed. Record timezone availability; a missing offset does not establish UTC local time. Omit local-hour features when local time cannot be established.

Begin with one feature row per eligible participant. Keep measurement types and incompatible source definitions separate. Core longitudinal summaries use adequately observed days under the frozen rules. Missing or insufficiently observed optional channels remain missing; do not interpret their absence as a measured zero. Predeclare a modest feature set after the channel audit:

- Measurement summaries appropriate to each channel: quantiles, median, mean, variability and range after validity filtering.
- For sampled rate measurements, daily summaries followed by participant summaries, with observed days receiving equal weight. Use duration-weighted summaries only if the recording semantics justify them.
- For interval totals, rates, sleep or event counts, use the corresponding channel-specific aggregation; do not treat all `longValue` values as interchangeable measurements.
- Between-day variability and weekday/weekend contrasts; local time-of-day profiles only when timezone information is usable.
- Acquisition summaries: observed days, event density, recording span, channel/source availability, and missingness. Identify these separately in the feature dictionary.

Source availability can be encoded because it is present in epoch data, but it must be distinguished from measured values. IDs, filenames, salutation, linked gender, age, BMI, WHO scores and survey-session counts are excluded from model inputs. Age/BMI remain allocation and reporting variables.

Fixed per-participant calculations can be computed independently for every split. Use a simple preprocessing pipeline: training-median imputation for numerical features, missingness indicators, and one-hot encoding for any categorical predictors with unknown levels ignored. Drop measurement columns that are entirely missing in training using a training-derived column mask. Fit these transformations on training participants and apply them unchanged to validation/test, following the [scikit-learn data-leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). No scaling, learned feature selection or representation learning is required. Verify the final transformed feature names; the existing notebook accidentally drops its numerical predictors.

## 7. One default random forest

Use one participant feature row and one outcome per eligible participant. Fit the preprocessing pipeline and a single random forest on the training 80% only:

```python
RandomForestClassifier(random_state=20260918)
```

Leave all learning parameters at their scikit-learn defaults; the seed is fixed solely for reproducibility. In particular, do not carry over the old notebook's 500 trees, minimum leaf size of 5, or balanced class weighting. Record the installed scikit-learn version and `get_params()` output. The documented defaults include 100 trees, unrestricted maximum depth, minimum leaf size 1 and no class weighting. See the [RandomForestClassifier reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

Fit once, with equally weighted participant rows. No hyperparameter search, cross-validation, additional fitted baselines, feature ablations, probability calibration or threshold tuning.

Keep the agreed validation 10% as a separate descriptive evaluation of this same fixed model; it is not used to select parameters or features. Then evaluate the same model once on the test 10%. Do not refit on train+validation or adapt the experiment after inspecting either score.

Use `predict_proba` for the probability of outcome 1 (salutation 20), locating its column through `classes_`. Use the estimator's default `predict` for hard classifications, including its default handling of tied probabilities.

## 8. Evaluation and interpretation

Report participant-level **AUROC**, accuracy, balanced accuracy, per-class precision/recall/F1 and a confusion matrix, separately for validation and test. For context, report the accuracy of always predicting the training majority class as a direct calculation; no additional classifier is fitted.

For test-set variability, generate exactly **10 bootstrap resamples** using a fixed seed. Within each salutation class, sample test participants with replacement, preserving the original class counts. Recompute AUROC, accuracy and balanced accuracy using the already saved predictions. Resampling units are participants, not epoch records. Do not refit the RF, regenerate features or change the split in these resamples. The RF's internal tree bootstrap remains at its default and is separate from these ten evaluation resamples.

Save all ten metric values and summarize their mean, sample SD and minimum/maximum alongside the original test-set point estimates. Ten resamples give only a rough variability check; do not present their percentile range as a reliable 95% confidence interval. These summaries condition on the fitted model, fixed split and observed class counts.

No subgroup performance study, calibration analysis or source-only model is required for this first pass. Keep the cohort and split descriptive tables. Source–salutation associations can contribute to prediction even with balanced splits; this experiment measures overall predictability and does not isolate a physiological contribution.

## 9. Implementation outputs and completion checks

Produce these artifacts when the experiment is implemented:

1. `cohort_manifest.parquet`: labels with provenance, all eligibility/exclusion flags, balance covariates, daily/epoch source profiles, retained event counts and adequate-day/coverage summaries; include the frozen channel-specific quality configuration.
2. `split_manifest.csv`: one immutable `user_id, split` assignment per eligible participant, plus configuration containing seed, constraints and tolerances.
3. `split_balance.csv` and a short report: category counts/proportions, pairwise discrepancies, continuous summaries and unresolved infeasibility.
4. `epoch_features.parquet` and a feature dictionary: units, aggregation rules, channel/source provenance, acquisition flags and missing-value semantics.
5. The fitted preprocessing pipeline and single RF, its package version/default parameters and random seed, and validation results.
6. A short test report with participant predictions, overall metrics, majority-class reference accuracy, all ten bootstrap metric rows and their descriptive summary, and cohort coverage.

Assert before fitting that participant sets are disjoint, labels are exclusively 10/20, sources 38/46/48 contribute no eligibility records or predictors, all included participants pass the frozen coverage gate, every eligible participant appears exactly once, every feature row has the correct owner, and no prohibited metadata enter the predictor matrix. Verify actual numerical features reach the estimator. Record input paths, resolved symlink targets, file inventory, processing version, package versions and random seeds for reproducibility.

Immediate implementation order: validate epoch/channel provenance; freeze and apply the coverage gate, documenting empty/sparse exclusions; build the balanced 80/10/10 manifest from the retained cohort; check its balance; extract the declared features; fit one default RF on training; report validation and test performance; run ten bootstrap resamples of test participants' fixed predictions.
