# Repository results audit

Audit date: 2026-10-14. See the [results reference](results_reference.md) for the current numerical summary.

**The checked metrics agree with saved predictions.** The main corrections concern experiment labels, cohort definitions, uncertainty, and interpretation. One historical full-history experiment has confirmed preprocessing exposure across allocations. Several newer benchmarks save only aggregate metrics, which limits verification.

## What was checked

Checks used source inspection, saved predictions, allocation files, metadata, and cached input summaries. No existing analysis, training, extraction, or plotting scripts were run. No experiment modules were imported. Data, fitted models, and numeric result files were unchanged.

AUROC was recomputed from saved labels and scores with `sklearn.metrics.roc_auc_score`. Calculations used each experiment's arm, allocation or repeat, and split. Additional checks covered participant keys, labels, split membership, and saved hashes where available.

| Result family | Checks against saved outputs | Result and limit |
|---|---|---|
| Pooled v1/v2 and source-specific default/tuned RF | 58 AUROCs; 116 accuracy/balanced-accuracy cells; 76 prediction files inspected | Agreement to numerical precision; unique participant rows and disjoint saved split assignments |
| Original R1a | 60 validation/test AUROCs | Exact agreement; predictions exist for only one of five arms |
| R1a with balanced allocations | 30 test AUROCs | Exact agreement; saved predictions cover three of five arms |
| Age experiment | 300 validation/test AUROCs | Exact agreement |
| R1b | 40 headline test AUROCs and 160 first-k AUROCs | Exact agreement; checked predictions match the recorded test participants |
| Day-level aggregation | Three saved repeat-0 day AUROCs; 90 all-day participant AUROCs | Agreement; other per-day prediction repeats are not saved in the inspected artifact |
| Informative recording | 540 repeated-split AUROCs and eight frozen-v1 bridge AUROCs, plus accuracy checks | Agreement; all 60 repeated split files match their test predictions |
| Source-specific 50/50 | Six AUROCs and 12 accuracy/balanced-accuracy values from 21,588 predictions | Exact agreement; both allocation directions match their saved test membership |
| Temporal v2 | 160 validation/test AUROCs | Exact agreement; all 92,160 prediction rows match recorded allocations |
| Recording-structure controls | 48 validation/test AUROCs | Exact agreement |
| Filling controls | 24 validation/test AUROCs | Exact agreement; clock-fill predictions reproduce the recording experiment's HR arm exactly |
| Heartbeat Lab bundles and delta evaluation | Six bundles' saved AUROCs, split/label consistency, assignment and calibration hashes; three delta models | Agreement; all 562 delta participants are disjoint from the full-study bundles' training/validation/test assignments |
| Tier1c, Apple pilot, Garmin binning | Recomputed means, sample SDs, and paired differences from all 10/5/5 allocation rows; inspected splits and fitting code | Aggregate agreement after documentation corrections. Participant scores are not saved by these benchmark scripts, so their AUROCs cannot be independently reconstructed here. |
| Full-history day scaling | Per-allocation metric summaries, cached days, and transform-fitting code | Mean/SD verified; endpoint predictions are unavailable, so endpoint AUROC itself was not independently recomputed |

The historical and temporal checks above include **1,527 AUROC comparisons**. The historical audit also checked **1,224 accuracy/balanced-accuracy values**. The largest discrepancy in those metric comparisons was **5.55 × 10⁻¹⁷**. No duplicate prediction keys, inconsistent labels, or split-membership mismatches were found in the checks performed. These are checks of saved artifacts, not a claim that every result file or all upstream processing was independently reproduced.

The four hashes in `experiments/temporal_garmin/results/temporal_repro.json` match the current cached inputs. The full-history cache contains **1,635,698 adequate participant-days from 3,848 people**, with no duplicated participant/date pairs and exact alignment between cached day rows and bin arrays. Adequate days per person range from **80 to 979**, mean **425.08**, median **476**.

## Confirmed evaluation issue

### Full-history endpoint preprocessing crosses allocation boundaries

In [`day_scaling_all.py`](../experiments/temporal_garmin/day_scaling_all.py), Phase B fits clock-bin fill values and MultiRocket biases on allocation-0 training participants and reuses that representation for all ten allocations. In allocations 1–9, **380–413 of the 573 test participants** had contributed to this preprocessing; the mean exposed proportion is **70.23%**.

The reported **0.9264 ± 0.0055** is correctly aggregated from saved endpoint metrics. The `0.0055` is sample SD across allocations. Labels were not used to fit the shared transform, but held-out participants' HR entered its fitted representation. The direction and magnitude of the effect on AUROC are unknown.

The ordinary first-40-day v2 analysis fits fill values and transform biases separately within each allocation. The endpoint defect must not be attributed to that analysis. The saved Heartbeat Lab bundles also use training-participant preprocessing, but differ in split and other fitting choices; their scores cannot quantify the endpoint's bias.

**Required before using this endpoint as clean held-out evidence:** fit the complete representation separately within each allocation and retain its predictions. This requires new fitting.

### Shared model selection and repeated participant use

Historical temporal regularization was selected using allocation-0 validation and reused across allocations. **80–98 later test participants per allocation** also belonged to that earlier validation set. This is a separate source of evaluation exposure from the unsupervised endpoint preprocessing.

The allocations are repeated holdouts. The historical ten-allocation temporal analysis has **5,730 test appearances but 3,097 distinct test participants**. Tier1c has **3,533 distinct test participants** across its ten allocations; Apple and Garmin pilots have **560** and **546**, respectively, across five allocations. The remaining participants never enter those test sets. These are not disjoint K-fold predictions or independent validation cohorts.

SDs describe variation across the observed allocations. Repeated-split t intervals in historical reports concern split/model randomization conditional on the observed cohort; they do not establish population or external-validation uncertainty. Clean per-allocation fitting also does not remove the effects of repeated development on the same people.

## Corrections to the repository overview

| Previous statement | Verified correction | Evidence |
|---|---|---|
| Pooled v2 N=20,485 | **N=20,423**; v1 is 20,485 | [`cohort_flow.csv`](../experiments/artifacts_v2/cohort_flow.csv), saved manifests |
| MultiRocket first-40 AUROC 0.8565 | **MultiRocket 0.8361**; **B40 + MultiRocket + HYDRA 0.8565** | [`temporal_metrics.csv`](../experiments/temporal_garmin/results/temporal_metrics.csv), `run_temporal.py` |
| Canonical clock-fill model 0.9100 | Saved `mr_clock` test AUROC **0.9177**; `mr_global` **0.9225** | Local bundle metadata and recomputed predictions; [`Lab audit`](../HEARTBEAT_LAB/AUDIT_REPORT.md) |
| Strict Samsung RF 0.6289 | Named default `win` arm **0.6609**. The **0.6289** belongs to the broader 50/50 experiment. | Source-7 predictions and [`50/50 results`](../experiments/2026-09-21_ortools_source_50_50_salutation_rf/output/model_metrics.csv) |
| Historical Apple cohort 4,367, only channel 3002 | **4,446 any-core participants**; 4,367 is the channel-3002 passer count | [`source_cohorts.csv`](../experiments/artifacts_sq/source_cohorts.csv) |
| Apple pilot establishes eligibility of all 11,143 | All **1,000 sampled** pilot participants produced adequate-day caches; full-cohort processing was not established | Pilot manifest and cache customer summaries |
| Garmin binning benchmark still running | All five allocation rows are saved; completed means replace the earlier partial means | [`binning metrics`](../experiments/garmin_temporal_binning/artifacts/garmin_binning_benchmark_results.csv) |
| Saved pilots compare eight static features | The current benchmark implementation pools eight daily statistics into mean/SD, adds a weekday contrast and bin-mask means: **305 summary features at five-minute resolution** | `heartbeat_lab/model.py`, pilot benchmark code |

Historical `temporal_*_v1` outputs retain the zero-sentinel defect; day-scaling `*_v1mrbug` outputs retain the separate omitted-B40 defect. They remain as provenance and are not mixed into the corrected series.

## Newer cohort and binning results

The expanded Garmin cohort uses all available adequate cached days. Its ten allocation rows give MultiRocket **0.9094 ± 0.0058**, summary **0.7196 ± 0.0179**, and a mean paired difference of **+0.1898**. Within MultiRocket test sets, strict participants score **0.9235** and added participants **0.8156**. The earlier frozen-model score on all 562 added participants is **0.7850**. Subtracting these two added-cohort scores does not isolate retraining: the trained models, evaluation samples, and allocation structures differ.

Both 1,000-person pilots also use **all adequate cached days**, not a common 40-day budget. Apple histories range from **14 to 887 days**, mean **264.91**; Garmin histories range from **14 to 979**, mean **395.49**. Sampling four days per training participant is a transform-bias fitting step, not the participant prediction budget.

### Apple

Saved test means are **0.7554** for five-minute MultiRocket, **0.7431** for 15-minute MultiRocket, and **0.6029** for summary ridge. The paired five-minute minus summary difference is **+0.1526**, with sample SD **0.0423**, correcting the former **0.0412**. The five-minute minus 15-minute difference is **+0.0123**, SD **0.0350**, positive in three of five allocations. These five allocations do not establish a general preference for one resolution.

### Garmin

| Grid | Overall mean ± SD | Strict stratum mean ± SD | Added stratum mean ± SD |
|---|---|---|---|
| 1 minute | 0.8350 ± 0.0369 | 0.8645 ± 0.0335 | 0.6022 ± 0.2150 |
| 5 minutes | 0.8410 ± 0.0380 | 0.8647 ± 0.0319 | 0.6753 ± 0.1245 |
| 15 minutes | 0.8228 ± 0.0411 | 0.8320 ± 0.0344 | 0.7687 ± 0.0753 |

Several former subgroup SDs did not match the CSV. The added stratum has only **15–24 test participants**, including **4–12 class-0 participants**, per allocation. Its paired 15-minute minus one-minute gain is **+0.1665**, SD **0.1743**, ranging from **+0.0185 to +0.4615**. The mean difference is a saved result; its large variability needs to remain visible.

The former summary-baseline subgroup values **0.7289/0.6225** are not present in the benchmark CSV and are not computed by its saved script. They were removed from the current summary rather than treated as verified.

## Interpretation corrections

**Day-budget curves do not establish performance bounds.** Fitting `a + b log₂(k)` or `L − C/√k` describes the measured fixed-budget points. Neither curve establishes an upper or lower bound on attainable AUROC. Averaging fitted `AUROC(k)` across participants' individual budgets does not identify heterogeneous-budget AUROC, which depends on positive–negative participant pairs. There is no fixed-budget measurement above 80 days, and the all-days endpoint also changes transform fitting.

**Binning does not isolate an imputation mechanism.** The pilot scripts aggregate observed raw events into each grid before filling empty bins. They change sequence length, clock resolution, missing-bin frequency, and effective convolution scales together. Long gaps remain empty at coarse resolution. In the Garmin pilot, participant-mean occupancy in the added stratum changes only from **64.70% at one minute to 66.53% at 15 minutes**. Claims that this recovers physiology or proves damage from median filling do not follow from these results.

**Recording groups are not observed behaviors.** The 562 added Garmin participants include 128 with spans shorter than 13 weeks, 432 failing the adequate-week requirement, and two failing the strict-day total. The last two were incorrectly described as empty non-wearers: they have **140 and 264 adequate days**. Short spans do not establish dropout, and absent night readings do not establish charging or off-wrist behavior. The revised notes use observed recording patterns and gate failures instead.

**Clock-bin medians are not a constant flat fill.** They vary with clock time; the historical fill range is documented in the temporal report. The claim that all imputed gaps necessarily yield zero differences is incorrect. Differences in AUROC under binning do not establish a missingness mechanism or explain vendor differences.

**Filling sensitivity concerns complete fitted pipelines.** Each fill arm refits biases, preprocessing, and the ridge head, with fill-specific regularization selection. Similar AUROC across four fills in dense data does not prove equivalence or locate all predictive information in observed bins. The interpolation intervention changes only a small subset of missing bins. For global minus clock filling, one of three test allocations is an exact ranking tie; a floating-point `1.11 × 10⁻¹⁶` difference had been counted as positive.

**Day-level and participant-level estimands differ.** Pooled day AUROC weights participants according to their available days; one-day-per-participant and aggregated participant AUROCs use different weighting. The older day-scale arm B additionally reserves **20–60 baseline days per participant**, so its `k=1` result does not describe prediction from one observed day. Similar bootstrap mean and point estimate does not prove estimator unbiasedness. The day-scale report now states these distinctions.

**Coverage audit zeros were mislabeled.** The source-specific report's `max_cov_s` column takes the maximum only among days exceeding 24 hours, then fills absent cells with zero. Zero therefore means no such over-24-hour days, not that a channel has no interval coverage. The table heading has been corrected. This resolves the apparent contradiction between zero entries in that audit column and positive channel-3002 strict-pass counts.

## Remaining limits and useful next checks

1. **Endpoint evaluation correction:** refit its preprocessing within each split before treating 0.9264 as clean held-out evidence.
2. **New benchmark auditability:** save participant scores, labels, split, model configuration, and relevant input hashes when Tier1c or the pilots are next run. The current CSV means are checked, but their underlying AUROCs cannot be reconstructed from saved predictions.
3. **Prospective or external performance:** use participants not involved in development and define the available-history budget in advance. A new split of the same cohort does not provide external validation.
4. **Mechanism questions:** use a matched design that varies the filling rule or representation while holding participants, splits, day budgets, and other processing fixed. Existing binning comparisons cannot answer why the differences occur.

App behavior, fresh-training reproducibility, raw-data reconstruction, and external validity remain untested. Existing figures and HTML/PDF exports were left unchanged and may retain older wording. Use the Markdown results reference and linked reports for the current summary.
