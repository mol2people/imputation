# Recorded-salutation prediction from wearable epoch data

Experiment overview, reviewed 21 September 2026. This summary covers the reports for all 12 experiment stages, including local output folders and the newest Garmin studies. Selected logs and scripts were inspected where interpretation needed clarification. No experiments were rerun for this review.

The prediction target is recorded salutation, codes 10 versus 20. Results do not establish biological sex, gender identity, or causal physiological effects. Cohorts and evaluation designs differ across experiments, so their absolute AUROCs are not directly interchangeable.

## Experiment results

| Experiment | Main result |
|---|---|
| [Pooled v1](REPORT.md) | 20,485 participants; default RF test AUROC **0.7132**, balanced accuracy **0.5896**. Strongly asymmetric class recall. |
| [Pooled v2](artifacts_v2/model_report.md) | Added demographics/source indicators and changed source exclusions; AUROC **0.6833**. Different cohort and split prevent attributing the difference to features alone. |
| [Strict source-specific models](artifacts_sq/sidequest_report.md) | Garmin **3,848**, Apple **4,446**, Samsung **541** participants. Compared demographics, full-recording, 91-day and rolling representations. Epoch summaries outperform demographics; useful representations differ by source. |
| [RF tuning](artifacts_sq/tuned_500_comparison.md) | Regularization improved performance, particularly for wide matrices. Established the shared configuration: `max_features=0.4`, `min_samples_leaf=10`, `class_weight="balanced_subsample"`, unrestricted depth. Later repeated-split studies used 100 trees. |
| [Feature selection](artifacts_sq/fsplit/feat_sel_report.md) | Across 30 repeated splits, hygiene reduced matrix width without a detected performance loss. Correlation pruning and top-k selection offered no consistent improvement; selected subsets were unstable. |
| [Feature engineering](artifacts_sq/feng/feat_eng_report.md) | Tested interactions, channel ratios and rolling-window differences. Primary gains were small—at most **+0.0034 AUROC**—with intervals crossing zero. |
| [Informative recording / missingness](2026-09-20_informative_recording_features/README.md) | Ungated cohort **25,784**. Recording model **0.6826**, values with source/gate context **0.7072**, combined **0.7239**. Explicit recording features add **+0.0167** beyond the value model. Native NaNs offer no detectable advantage over median imputation. |
| [Continuous age versus age groups](experiments/age_vs_agegroup_garmin_2026-09-20/results/REPORT.md) | Garmin AUROC **0.7445 → 0.7475**. Paired gain **+0.0029**, interval **[−0.0008, +0.0067]**: promising but not established superiority. |
| [OR-Tools 50/50, flipped](2026-09-21_ortools_source_50_50_salutation_rf/PLAN.md) | Broader ungated source cohorts; age/BMI/salutation-balanced allocation. AUROCs in the two directions: Garmin **0.7290 / 0.7094**, Apple **0.6875 / 0.6787**, Samsung **0.6289 / 0.6493**. This experiment fixes 100 trees and `max_depth=10`, alongside the shared feature-fraction, leaf-size and class-weight settings. |
| [R1a circadian features](experiments/r1a_circadian_garmin_2026-09-21/results/REPORT.md) | Added 35 hourly-profile/cosinor features. Garmin baseline **0.7434**, combined **0.7463**, coverage-residualized combination **0.7470**. Small positive increments. |
| [Day-scale modelling](experiments/dayscale_garmin_2026-09-21/results/REPORT.md) | Raw-day predictions aggregated per participant rise from **0.6653** with one sampled day to **0.7269** with 32 and **0.7302** with all days. Personal-baseline features help most at small k. |
| [R1a with balanced allocator folds](experiments/r1a_allocfolds_garmin_2026-09-21/results/REPORT.md) | Ten allocations: baseline **0.7415**, combined **0.7509**, residualized combination **0.7475**. Both increments were positive in all ten allocations and their Bonferroni intervals exclude zero. |
| [R1b date-aware circadian](experiments/r1b_dateaware_garmin_2026-09-21/results/REPORT.md) | Per-day curves over each user's first 40 adequate days, on the R1a-AF folds. Day-level AUROC **0.6589 → 0.6782** (**+0.0193**, coverage-residualized **+0.0181**); participant baseline **0.7415**, combined **0.7513** (**+0.0098**). All three increments Bonferroni-positive; day-level gain mostly coverage-independent. |

The informative-recording and OR-Tools result summaries above were read from each experiment's local `output/report.md`. Those output directories are excluded from Git because they also contain participant-level artifacts; the links point to the published experiment documentation.

## What the results support

The strongest new findings are that **recording patterns and HR values contribute complementary predictive information**, and that **hourly/circadian representations add information beyond the existing Garmin summaries**. The older claim that feature-side improvements were exhausted is too strong. R1b extends this to the day level: per-day circadian shape adds **+0.019** AUROC over daily summaries on individual days, largely independent of within-day coverage, and participant-level aggregation preserves a **+0.010** increment on top of the existing baseline.

Feature hygiene remains useful for reducing matrix width. The tested feature-selection and feature-engineering methods do not support a general performance improvement. Continuous age is a compact alternative to age groups, but its observed gain does not establish superiority.

The informative-recording experiment supports using observation patterns as predictors. It does not identify an MNAR mechanism or establish performance among participants whose salutation is missing. Its executed protocol is documented in [the informative-recording plan](CODEX_INFORMATIVE_RECORDING_FEATURE_PLAN_2026-09-20.md); [MNAR_PLAN.md](MNAR_PLAN.md) is the earlier proposal and should not be treated as the exact executed design.

## Interpretation qualifications

- **Day-scale results are retrospective sampling results.** The script samples k days **with replacement** across the available record. Arm B additionally uses up to **60 separate baseline days**, also sampled from that participant's record. Its “one-day” performance therefore is not performance with only one day of history, and the scaling curve does not establish performance during the first month. R1b's chronological first-k curve addresses the prospective-window question on the R1b model (first day **0.669**, near-plateau by day 32). See the [day-scale implementation](experiments/dayscale_garmin_2026-09-21/run_dayscale.py).
- **Coverage residualization is a predictive adjustment, not proof of deconfounding.** The R1a-AF claim that approximately 37% of the increment is mediated by composition/activity is not identified by these AUROC differences. Likewise, the larger gain under balanced folds does not establish why it increased. The [R1a implementation](experiments/r1a_circadian_garmin_2026-09-21/run_r1a.py) fits a linear adjustment for hourly coverage shares and total recording volume on training participants.
- **The comparison designs differ.** V1/v2 and the original sidequest used optimized allocations; feature selection, engineering, age and original R1a used repeated label-stratified random splits. R1a-AF uses demographic balancing. These remain evaluations on reused cohorts, not independent validation. Repeated-split intervals describe split/model randomization conditional on the observed cohort.
- **The September 20 meta-report is now incomplete.** It still describes missingness modelling as unrun and predates the newer experiments. It does, however, contain corrected simultaneous intervals for feature selection/engineering that supersede the older reports' mislabeled intervals. R1a and day-scale reports also contain explicit errata. See the [meta-report PDF](reports/recorded_salutation_meta_report_2026-09-20.pdf) and its [HTML source](reports/recorded_salutation_meta_report_2026-09-20.html).

## Cohort allocation package

[`cohort_allocator`](cohort_allocator/README.md) is the separate reusable OR-Tools package. It accepts a participant-level demographic summary, variables to balance, optional independent partitions, and relative fold sizes, then exports assignments, prepared cohorts and balance diagnostics. R1a-AF is the completed experiment demonstrating its use for the later feature comparisons.
