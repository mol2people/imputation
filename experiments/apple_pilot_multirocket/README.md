# Apple Watch MultiRocket pilot benchmark

MultiRocket plus summary features achieved mean test AUROC **0.7554** on a pilot of 1,000 selected single-source Apple participants, compared with **0.6029** for summary ridge. The target is recorded salutation: `10 → 0/F`, `20 → 1/M`.

The five saved allocations are complete. Results are exploratory within this selected cohort; the [results audit](../../reports/results_audit_2026-10-14.md) covers provenance and evaluation limits.

## Cohort and preprocessing

The pilot was sampled from single-source Source 6 participants already marked eligible in `experiments/artifacts_v2/cohort_manifest.parquet`, with a valid target and an available raw CSV. Sampling was proportional to recorded salutation and age group. The pilot contains 346 class-0 and 654 class-1 participants.

Source 6, type 3000 readings are retained at 25–230 bpm with finite timestamps and timezone offsets. A local day requires at least eight observed clock hours and 60 readings; participants require at least 14 such days. All 1,000 sampled participants passed this preparation, with mean **264.9 adequate days**. This 100% retention is conditional on the already eligible sampling pool, rather than an estimate for all Apple users.

Raw events are averaged independently within 5-minute or 15-minute bins before filling. Empty bins remain missing in the saved caches. Mean participant-level occupancy is **54.1%** at 5 minutes and **68.3%** at 15 minutes. Participant features use all cached adequate days; sampling up to four days per training participant applies only to fitting MultiRocket biases.

## Benchmark results

Each of five OR-Tools allocations has 700 training, 150 validation and 150 test participants. Clock-bin filling, MultiRocket bias quantiles, feature imputation, scaling and ridge fitting use training participants only. Ridge alpha is fixed at 1000. These are repeated holdouts with overlapping test sets, rather than five disjoint cross-validation folds; 560 distinct participants appear in test at least once.

| Allocation | MultiRocket 5 min | MultiRocket 15 min | Summary ridge | Paired gain of 5 min over summary |
|---|---:|---:|---:|---:|
| 0 | 0.8022 | 0.7987 | 0.6847 | +0.1175 |
| 1 | 0.7704 | 0.7102 | 0.5599 | +0.2106 |
| 2 | 0.8181 | 0.7824 | 0.6489 | +0.1692 |
| 3 | 0.6949 | 0.7123 | 0.5895 | +0.1054 |
| 4 | 0.6915 | 0.7121 | 0.5314 | +0.1601 |
| Mean ± sample SD | **0.7554 ± 0.0594** | **0.7431 ± 0.0437** | **0.6029 ± 0.0631** | **+0.1526 ± 0.0423** |

Values come from [`pilot_benchmark_results.csv`](artifacts/pilot_benchmark_results.csv). The SD describes variation across allocations; it is not a confidence interval or an independent replication standard error. The script saves aggregate metrics, without participant predictions or fitted bundles for these arms, so the AUROCs cannot be independently reconstructed from saved scores.

The summary baseline has **305 features**: means and sample SDs of eight daily statistics, a weekday–weekend mean contrast, and 288 observation-frequency features. MultiRocket adds participant means and SDs of daily transformed features. The comparison therefore measures the increment from adding MultiRocket to an HR-and-recording summary baseline.

## Interpretation

The 5-minute arm exceeded summary ridge in all five allocations. Its mean advantage over 15 minutes was **0.0123**, with allocation differences ranging from −0.0206 to +0.0602; this does not establish a stable resolution advantage. Mean elapsed arm times were 172.5 s and 70.4 s, respectively, including preprocessing, transformation of all splits, fitting and evaluation.

The corresponding 1,000-participant Garmin pilot reached 0.8410 at 5 minutes; the larger 4,410-participant Garmin experiment reached 0.9094. These are different cohorts, with different histories and training sample sizes. Their difference from Apple performance does not identify a hardware or sampling mechanism. Missingness and clock-time arrangement may contribute, but charging behavior and physiological causes require separate evidence.

## Scripts

- [`01_sample_pilot_cohort.py`](scripts/01_sample_pilot_cohort.py): sample the selected eligible pool.
- [`02_build_apple_cache.py`](scripts/02_build_apple_cache.py): prepare adequate days and both raw-event bin grids.
- [`03_split_pilot_ortools.py`](scripts/03_split_pilot_ortools.py): generate five demographic-balanced repeated holdouts.
- [`04_run_apple_pilot.py`](scripts/04_run_apple_pilot.py): fit the three arms within each allocation and save metrics.
