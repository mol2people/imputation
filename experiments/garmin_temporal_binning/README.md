# Garmin Temporal Granularity Benchmark: 1-min vs. 5-min vs. 15-min MultiRocket

This experiment evaluates the effect of temporal binning resolution on MultiRocket discrimination across continuous vs. fragmented wear regimes in the German COVID-19 Data Donation (*Datenspende*) Garmin cohort.

---

## 1. Experimental Setup

- **Cohort:** Matched sample of $N = 1,000$ Garmin single-source participants sampled with identical stratification as the Apple Watch pilot benchmark (stratified across Strict Tier 0 vs. Relaxed Wear-Fail Delta).
- **Allocations:** 5 demographic/label-balanced allocations generated via OR-Tools integer programming (70% train, 15% validation, 15% test).
- **Compute:** 8 CPU threads (`NUMBA_NUM_THREADS = 8`).
- **Data Hygiene:** Strictly leakage-free. Imputation clock medians, scalers, and transform biases were fitted exclusively on training sets.
- **Compared Configurations:**
  1. **1-minute bins ($L = 1,440$ bins/day):** Raw export resolution.
  2. **5-minute bins ($L = 288$ bins/day):** Gold standard repository baseline.
  3. **15-minute bins ($L = 96$ bins/day):** Temporal boxcar downsampling.
  4. **Static Summary Baseline:** 8 tabular daily statistics (mean, median, sd, vmin, vmax, n, cov_h, hours).

---

## 2. Benchmark Results

Across all 5 balanced allocations:

| Model / Resolution | Grid Length ($L$) | Overall Test AUROC | Strict Stratum AUROC | Delta Stratum AUROC | Mean Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MultiRocket 1-min** | 1,440 | $0.8350 \pm 0.0369$ | $0.8645 \pm 0.0347$ | $0.6022 \pm 0.2173$ | 356.4 s |
| **MultiRocket 5-min (Baseline)** | 288 | **$0.8410 \pm 0.0380$** | **$0.8647 \pm 0.0326$** | $0.6753 \pm 0.1246$ | 84.7 s |
| **MultiRocket 15-min** | 96 | $0.8228 \pm 0.0411$ | $0.8320 \pm 0.0336$ | **$0.7687 \pm 0.0717$** | **39.3 s** |
| **Static Summary Baseline** | 8 statics | $0.7143 \pm 0.0360$ | $0.7289 \pm 0.0360$ | $0.6225 \pm 0.0637$ | 0.01 s |

*Full per-allocation metrics are preserved in [`artifacts/garmin_binning_benchmark_results.csv`](artifacts/garmin_binning_benchmark_results.csv).*

---

## 3. Key Findings

### 1. High Resolution Wins on Continuous Wearers
For the Strict cohort (continuous wearers with $>95\%$ daily bin occupancy), **5-minute binning achieves the highest performance ($0.8647$)**, slightly edging out 1-minute ($0.8645$) while running **$4.2\times$ faster** (84.7s vs. 356.4s). Coarse 15-minute binning loses $-0.0327$ AUROC because it smooths out fine-grained autonomic recovery and transient slopes.

### 2. Coarse Binning Rescues Fragmented / Intermittent Wearers
On the wear-fail Delta stratum (dominated by Phenotype A intermittent wearers with 10-hour daily absences), 1-minute binning collapses to **$0.6022$** (and crashed to $0.2418$ in Fold 1) because flat population-median imputation injects vast synthetic flatlines ($\Delta x = 0$) and step discontinuities.

Moving to 15-minute boxcar binning acts as **physical observation aggregation**: any single heart rate measurement within a 900-second window populates the bin with true physiology. This lifts Delta stratum AUROC to **$0.7687$ (+0.1665 over 1-minute, +0.0934 over 5-minute)** with lower variance ($\sigma = 0.0717$ vs. $0.2173$).

---

## 4. Scripts
- [`scripts/01_sample_garmin_binning_cohort.py`](scripts/01_sample_garmin_binning_cohort.py): Samples matched 1,000 Garmin cohort.
- [`scripts/02_build_garmin_multigrid_cache.py`](scripts/02_build_garmin_multigrid_cache.py): Builds 1m, 5m, and 15m compressed `.npz` caches.
- [`scripts/03_split_garmin_ortools.py`](scripts/03_split_garmin_ortools.py): Generates 5 OR-Tools balanced demographic splits.
- [`scripts/04_run_binning_benchmark.py`](scripts/04_run_binning_benchmark.py): Runs train-only MultiRocket benchmark across all 5 allocations on 8 cores.
