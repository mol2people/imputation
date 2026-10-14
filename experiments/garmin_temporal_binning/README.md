# Garmin temporal binning benchmark

This completed benchmark compares 1-minute, 5-minute and 15-minute MultiRocket inputs in 1,000 selected single-source Garmin participants. Mean test AUROC was **0.8350**, **0.8410** and **0.8228**, respectively. The target is recorded salutation: `10 → 0/F`, `20 → 1/M`.

## Cohort and evaluation

Participants were sampled proportionally to recorded salutation and age group from the 4,410-member Tier 1c cohort. The sample contains 881 strict-gate participants and 119 delta participants who failed that earlier gate; sampling did not explicitly match these strata or match Apple participants. Five OR-Tools allocations each contain 700 training, 150 validation and 150 test participants. Test sets overlap: 546 distinct participants appear in test at least once.

Source 3, type 3000 HR is filtered to 25–230 bpm and finite timestamps and timezone offsets. Adequate local days require at least eight observed clock hours and either 60 readings or eight hours of interval-union coverage; participants require at least 14 adequate days. The same dates and raw events feed all resolutions. Participant features use all cached adequate days, averaging 395.5 days per participant; sampling up to four training days per participant applies only to fitting MultiRocket biases.

Each grid averages observed raw events within its bins **before filling**. Empty bins remain missing until clock-bin medians are fitted on training participants. MultiRocket biases, feature imputation, variance filtering, scaling and ridge regression are also fitted within each training allocation. Alpha is fixed at 1000. Train-only fitting avoids direct cross-split preprocessing exposure; repeated use of the study cohort still makes this an exploratory benchmark.

## Saved results

| Arm | Overall test AUROC | Strict stratum | Delta stratum | Mean elapsed arm time |
|---|---:|---:|---:|---:|
| MultiRocket 1 min | 0.8350 ± 0.0369 | 0.8645 ± 0.0335 | 0.6022 ± 0.2150 | 356.4 s |
| MultiRocket 5 min | **0.8410 ± 0.0380** | **0.8647 ± 0.0319** | 0.6753 ± 0.1245 | 84.7 s |
| MultiRocket 15 min | 0.8228 ± 0.0411 | 0.8320 ± 0.0344 | **0.7687 ± 0.0753** | 39.3 s |
| Summary ridge | 0.7143 ± 0.0360 | Not saved | Not saved | 0.013 s |

Entries are means ± sample SD across five overlapping repeated holdouts, not confidence intervals. Elapsed time includes train-derived filling, feature transformation of training/validation/test participants, ridge fitting and evaluation. Full allocation metrics are in [`garmin_binning_benchmark_results.csv`](artifacts/garmin_binning_benchmark_results.csv).

The summary baseline has **305 features** at 5 minutes: means and sample SDs of eight daily statistics, a weekday–weekend contrast, and 288 observation-frequency features. Each MultiRocket arm includes analogous summaries at its own resolution, so the mask-feature width also changes with resolution. The saved CSV omits participant predictions, fitted bundles and validation metrics, limiting independent verification beyond aggregate arithmetic and source inspection. See the [results audit](../../reports/results_audit_2026-10-14.md).

## Interpretation

Five minutes and one minute have similar mean strict-stratum AUROC. Fifteen minutes has higher delta-stratum AUROC in all five comparisons with one minute and four of five comparisons with five minutes. Mean paired differences are **+0.1665** and **+0.0934**, respectively.

The delta test groups contain only **15–24 participants**, including **4–12 class-0 participants**, per allocation. The large 1-minute variation includes an AUROC of 0.2418 in one allocation. These small, overlapping subgroups support an exploratory resolution comparison, without establishing a reliable population gain or its mechanism.

Mean participant-level 5-minute occupancy is **89.6%** in the strict stratum and **65.3%** in delta. Delta occupancy increases from **64.7%** at one minute to **66.5%** at 15 minutes. Coarser bins combine nearby observations but do not fill multi-hour gaps with observed data. Resolution changes smoothing, kernel time scales, missingness and mask features simultaneously; the results do not isolate imputation artifacts, autonomic dynamics or wear behavior as the cause.

## Scripts

- [`01_sample_garmin_pilot.py`](scripts/01_sample_garmin_pilot.py): sample 1,000 Tier 1c participants.
- [`02_build_garmin_binning_caches.py`](scripts/02_build_garmin_binning_caches.py): prepare raw-event means at all three resolutions.
- [`03_split_garmin_pilot_ortools.py`](scripts/03_split_garmin_pilot_ortools.py): generate five repeated holdouts.
- [`04_run_binning_benchmark.py`](scripts/04_run_binning_benchmark.py): fit all arms within each allocation and save metrics.
