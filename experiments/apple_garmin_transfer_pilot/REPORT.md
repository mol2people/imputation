# Apple ↔ Garmin: preliminary transfer of recorded-salutation predictors

Recorded 2026-10-20 (Europe/Berlin).

The shared HR-summary representation transfers some ranking information in both directions, but imported models perform worse than locally trained models on these test cohorts. Recording summaries show little transport. This is an exploratory result for a small static baseline, not a test of the stronger MultiRocket pipelines or a demonstration of device-independent physiology.

## Question and estimand

The question is whether a predictor trained in one device cohort retains discrimination of recorded salutation in the other. The estimand here is target-cohort test AUROC of the imported source-fitted score and its difference from a locally fitted score on the identical target participants. It is conditional on eligibility, these allocation-0 splits, the chosen representation and fitted models. It is not an effect of changing device within a person.

## Preliminary inventory

The full Apple allocation has 11,139 users (7,284 class 1, 3,855 class 0); the Garmin binning pilot has 1,000 (638/362). Both cache calendar ranges span 2020-04-12 to 2022-12-17. Median full adequate history is 261 versus 448.5 days. Mean 5-minute occupancy is 0.5400 versus 0.8673; medians are 0.5293 versus 0.9435. These are pre-restriction inventory values, not pilot test estimates.

A separate strict Garmin cache contains 3,848 users (2,464/1,384), exactly 40 days each (153,920 rows), spanning 2020-04-12 to 2022-11-01. Apple has zero participant-ID overlap with either Garmin cohort; the Garmin pilot overlaps the strict cohort by 881 users. Strict and relaxed Garmin eligibility must not be conflated.

The common manifest contains age groups and BMI groups, but no continuous-age field. Full-cohort age counts are Apple 7,470 Middle Ager, 2,316 Young Adults, 1,353 Elderly; Garmin pilot 681, 208, 111. These distributions and recording differences permit selection confounding despite similar label prevalence. Shared per-user caches contain `hr[D,288]` float32, integer daily `date[D]`, and `stats[D,8]` float32 in the same daily-statistic order. The Apple caches lack native sub-bin counts. Saved stronger Garmin production bundles exist, but this pilot trains a common simple representation in both devices instead of treating separately fitted pipelines and cohort definitions as directly transportable.

Saved-model compatibility was inspected read-only. The strict Garmin median production bundle at `experiments/k5_strict_garmin_interpolation/track_40days/median/production_bundle.joblib` contains a 288-bin fill template, MultiRocket base/difference transforms, imputer, variance filter, scaler, a 19,121-feature classifier (alpha 5000), and calibration/threshold objects. Apple cohort-scaling artifacts have no saved joblib model; its runner saves predictions. Both architectures conceptually combine 305 static features with 18,816 sequence features, and both use the same implemented daily-statistic and weekday-minus-weekend definitions. The two vendored MultiRocket source files have identical SHA256 `71d1e61bf59a6a7ff21c76b36d73e88dbf4e6307f1cea547cb342373687a7786`. These checks establish conceptual compatibility, not equality of fitted transforms. Strict Garmin fitting also uses class-balance weights while the Apple runner does not. Its cohort gate requires at least 65 strict days with at least 12 hours coverage within a 91-day span of at least 11 weeks. These differences, plus the absent Apple model serialization, mean full saved-model transport would require a separate controlled implementation; it was not run here.

## Protocol

Inputs are existing Apple full-cohort and Garmin 1,000-person binning-pilot caches. Participants need at least 40 adequate days; features use exactly the earliest 40 dates. Existing allocation-0 train/validation/test assignments are retained after this restriction. Apple has 10,214 eligible participants and Garmin 979, with no shared IDs. These are distinct device cohorts, not paired device recordings.

| Device | Train | Validation | Test | Test class 0 / 1 |
|---|---:|---:|---:|---:|
| Apple | 7,169 | 1,515 | 1,530 | 542 / 988 |
| Garmin | 684 | 147 | 148 | 54 / 94 |

The target is recorded administrative salutation: 10 → 0 and 20 → 1. Age groups are available; continuous age is not used. No age or BMI predictor is included.

Three prespecified feature blocks use the same definitions in both devices:

- **HR (10):** across-day mean and sample SD of daily HR mean, median, SD, minimum and maximum.
- **Recording (30):** across-day mean and sample SD of daily event count, covered hours and occupied clock hours, plus 24 hourly mean observation frequencies from the original 5-minute masks.
- **Combined (41):** both blocks plus weekday-minus-weekend daily mean HR.

All summary statistics use cached native values; HR features remain dependent on which observations each device records. For each source device, median imputation, standardization and RidgeClassifier are fitted on source training participants. Alpha is selected by source-validation AUROC from {100, 500, 1000, 5000, 10000}; reference `np.isclose` ties favor larger alpha. There is no train-plus-validation refit, target tuning, calibration, raw SQL or MultiRocket computation. Each target's own and imported models are evaluated on exactly the same target test participants.

## Full source samples

Values in the last column are imported-minus-local AUROC with exploratory conditional 95% intervals. Each interval uses 1,000 class-stratified paired participant resamples at fixed class counts, seed 20261019 reset per contrast. The same sampled indices are used for both scores. Intervals omit model-fitting, selection and cohort-sampling uncertainty and have no multiplicity correction.

| Target ← imported source | Features | Local AUROC | Imported AUROC | Imported − local [95% interval] |
|---|---|---:|---:|---:|
| Apple ← Garmin | HR | 0.5947 | 0.5405 | −0.0542 [−0.0780, −0.0311] |
| Apple ← Garmin | Recording | 0.6328 | 0.4935 | −0.1393 [−0.1847, −0.0901] |
| Apple ← Garmin | Combined | 0.6596 | 0.5583 | −0.1013 [−0.1319, −0.0702] |
| Garmin ← Apple | HR | 0.7811 | 0.7114 | −0.0697 [−0.1399, +0.0065] |
| Garmin ← Apple | Recording | 0.5162 | 0.4768 | −0.0394 [−0.2004, +0.1143] |
| Garmin ← Apple | Combined | 0.7595 | 0.6617 | −0.0977 [−0.2017, +0.0032] |

HR ranking transfers better than recording ranking on the observed point estimates. Adding recording information to HR improves imported performance on Apple, but reduces it on Garmin; the combined representation is not uniformly preferable. Absolute imported-AUROC intervals from the same bootstrap draws are HR Garmin→Apple 0.5405 [0.5102, 0.5698], HR Apple→Garmin 0.7114 [0.6296, 0.7902], combined Garmin→Apple 0.5583 [0.5273, 0.5869], and combined Apple→Garmin 0.6617 [0.5705, 0.7447]. These exploratory intervals lie above 0.5 in this run; they remain conditional on the fitted scores and do not establish a population-invariant mechanism. The paired intervals compare imported with local performance, a different estimand.

Selected full-mode alphas are Apple HR 100, recording 1000, combined 100; Garmin HR 100, recording 100, combined 500. Size-matched Apple selects 1000, 1000 and 500 respectively. All candidates and validation AUROCs are saved; test results did not select among them.

## Source-size control

Apple training and validation sets were downsampled once, separately and stratified by label, to Garmin's 684 and 147 participants. Sampling preserves Apple class proportions approximately; seed 20261019 was fixed in advance. Test sets remain unchanged. Garmin fits are identical to the full-source analysis. This single draw controls source sample sizes; it does not equate cohort composition or learning difficulty.

| Target ← imported source | Features | Local AUROC | Imported AUROC | Imported − local [95% interval] |
|---|---|---:|---:|---:|
| Apple ← Garmin | HR | 0.5654 | 0.5405 | −0.0248 [−0.0537, +0.0034] |
| Apple ← Garmin | Recording | 0.6007 | 0.4935 | −0.1072 [−0.1535, −0.0560] |
| Apple ← Garmin | Combined | 0.6250 | 0.5583 | −0.0666 [−0.1025, −0.0313] |
| Garmin ← Apple | HR | 0.7811 | 0.6952 | −0.0859 [−0.1430, −0.0350] |
| Garmin ← Apple | Recording | 0.5162 | 0.4439 | −0.0723 [−0.1929, +0.0506] |
| Garmin ← Apple | Combined | 0.7595 | 0.5575 | −0.2019 [−0.2924, −0.1075] |

For the size-matched Apple→Garmin models, imported HR AUROC is 0.6952 [0.6139, 0.7735] and imported combined AUROC is 0.5575 [0.4651, 0.6491]. Garmin→Apple imported scores remain identical across modes. Absolute intervals for all local and imported scores are saved in `paired_transfer_effects.csv`.

The smaller Apple local HR model narrows the Garmin→Apple comparison. Conversely, Apple→Garmin HR transfer remains below the local Garmin model with equal source sample sizes in this draw. Domain shift and sample-size effects therefore cannot be reduced to one explanation.

Age-stratified HR point estimates are in `test_metrics.csv`. On Apple, imported/local AUROCs are 0.6138/0.6470 in Elderly (N=184), 0.5334/0.5785 in Middle Ager (N=1,029), and 0.5315/0.6083 in Young Adults (N=317). Garmin age strata contain only 17, 100 and 31 test participants; the Elderly stratum has four class-0 participants. These descriptive strata do not provide adjusted transport estimates.

## Interpretation and limits

The common signal may combine physiology, behavior, demographic composition and device-dependent sampling of HR. Day eligibility differs: Apple requires at least 60 readings in eight clock hours; Garmin requires eight clock hours and either eight covered hours or 60 readings. Thus recording and HR-summary distributions are not measurement-equivalent. Restricting both cohorts to 40 adequate days matches history quantity, not calendar periods, occupancy or day-selection mechanisms. First-40-day selection also changes the population relative to the earlier full Apple benchmark.

The Garmin pilot is a sample from a relaxed Garmin cohort, not the strict 3,848-person Garmin cohort. No result here establishes biological sex classification, causal behavioral effects or transport to other populations. The small Garmin test set and one allocation materially limit precision. HR coefficient signs are not evidence of common mechanisms because the summaries are correlated and scaled within source.

Proposed, not run: a useful next experiment is to impose observed Apple-like masks on high-coverage Garmin days, comparing the same Garmin observations under mask timing versus uniform thinning at matched retained counts. That intervention would test sampling sensitivity on fixed participants, complementing this between-cohort transport comparison. Mask exposure and original observations must remain explicit; it would not identify missing Apple HR values.

See [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md) for the separate feasibility checks and proposed follow-up designs; those experiments were not run in this pilot.

## Artifacts and verification

Run `/Users/bulat/micromamba/envs/datenspende/bin/python experiments/apple_garmin_transfer_pilot/run_transfer.py` from the repository. The script only reads existing caches and fits these small static models. Runtime for this run was approximately 20 seconds.

- `participant_features.csv`: exact first-40-day participant features and retained split labels.
- `fit_participants_{mode}_{device}.csv`: exact source fitting/validation participants, including the size control.
- `alpha_selection.csv`, `coefficients.csv`: source-only selection scores and standardized coefficients.
- `test_predictions.csv`, `test_metrics.csv`, `paired_transfer_effects.csv`: saved scores, age-stratified metrics and paired intervals.
- `provenance.json`: source and code hashes, selected-array hashes, library versions and protocol.

Checks cover cache shapes/date ordering, 40-day eligibility counts, ID non-overlap, uniqueness of saved prediction keys, paired target score alignment, and raw-score AUROC reproduction. Independent review reproduced all 24 overall AUROCs, fitting/test disjointness, source and feature hashes, identical Garmin predictions across modes, and both full-mode HR paired intervals to 1e−14. The initial run stopped while exporting coefficients because the installed scikit-learn returns a one-dimensional binary coefficient array; flattening the array fixed the export. No valid result from that interrupted run was retained as final evidence. The script and saved outputs were then checked without running the upstream pipelines. No pre-existing artifacts were modified.

Input paths relative to the repository are `experiments/apple_cohort_scaling/artifacts/{apple_10_allocations.parquet,cache_5min/customers.csv,cache_5min/customer_*.npz}` and the corresponding `experiments/garmin_temporal_binning/artifacts/{garmin_binning_5_allocations.parquet,cache_5min/customers.csv,cache_5min/customer_*.npz}`. Only selected earliest-40-day arrays are consumed. The source-array hashes concatenate sorted participant IDs, selected dates, daily statistics and HR grids. The provenance records exact versions and raw metadata hashes. No trained-model serialization is required to reproduce this small fit; `coefficients.csv` does not replace fitted preprocessing.
