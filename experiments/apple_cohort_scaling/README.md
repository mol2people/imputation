# Full Apple cohort benchmark

Five-minute MultiRocket plus summary features reaches **0.8561 ± 0.0111 test AUROC** across ten allocations of **11,139 Apple participants**, versus **0.6474 ± 0.0180** for summary ridge. The target is recorded salutation: `10 → 0/F`, `20 → 1/M`.

The benchmark completed on **2026-10-14**. The [recording-pattern exploration](BINNING_BY_RECORDING.md), added on October 15, compares existing five- and fifteen-minute predictions by history length, within-day occupancy, and continuity across adequate dates.

## Results

Values are mean ± sample SD across ten allocations. MR includes summary features. Alpha selection uses validation AUROC.

| Representation | History | Bins | Fixed α = 1000 | Selected α |
|---|---|---|---:|---:|
| MR + summaries | All adequate days | 5 min | **0.8561 ± 0.0111** | 0.8553 ± 0.0123 |
| MR + summaries | All adequate days | 15 min | 0.8348 ± 0.0079 | 0.8378 ± 0.0106 |
| MR + summaries | First up to 40 adequate days | 5 min | 0.7918 ± 0.0148 | **0.8110 ± 0.0131** |
| MR + summaries | First up to 40 adequate days | 15 min | 0.7636 ± 0.0113 | 0.7985 ± 0.0097 |
| Summary ridge | All adequate days | 5 min | 0.6474 ± 0.0180 | 0.6468 ± 0.0151 |
| Summary ridge | First up to 40 adequate days | 5 min | 0.6298 ± 0.0164 | 0.6340 ± 0.0143 |

The full-history MR increment over summaries is **+0.2087 ± 0.0193** at fixed alpha. Five-minute full history exceeds capped history by **+0.0643 ± 0.0144** at fixed alpha, or **+0.0442 ± 0.0122** when both select alpha. Shorter inputs benefit from stronger regularization: selected alphas are 5,000–10,000 for five-minute capped inputs and 10,000 for all ten fifteen-minute capped fits.

Five-minute bins lead on average across low/middle/high recording-pattern groups. Their selected-alpha advantage is only **+0.0058** when both occupancy and continuity are low, but **+0.0246** when within-day occupancy is low and continuity is high. Short-history/high-occupancy participants are near a tie (+0.0056 all days; +0.0001 capped). See [binning by recording pattern](BINNING_BY_RECORDING.md) for subgroup definitions, paired SDs, and complete saved tables.

## Cohort and inputs

Of 11,143 eligible single-source Apple candidates, 11,139 qualify. Source 6, type 3000 HR is retained at 25–230 bpm; an adequate local day has at least eight observed clock hours and 60 readings. Participants need at least 14 adequate days. Mean history is **266.4 days**, median **261**, range **14–902**; **925 participants (8.30%)** have 14–39 days and therefore contribute identical input histories to the two budget arms.

Raw readings are averaged independently into five- or fifteen-minute bins before filling. Mean participant occupancy is **54.0% / 68.3%**. Each OR-Tools allocation contains **7,797 training, 1,671 validation, and 1,671 test participants**.

Clock-bin fills, MultiRocket biases, feature imputation, variance filtering, scaling, and ridge fitting use training participants. Each resolution's summaries include the means and SDs of eight daily statistics, a weekday–weekend contrast, and observation frequencies: 305 features at five minutes or 113 at fifteen minutes. MultiRocket adds means and SDs of daily transformed features. The separate summary baseline uses five-minute features.

The executed alpha grid is **`{100, 500, 1000, 5000, 10000}`**, replacing the grid originally proposed in [PLAN.md](PLAN.md). Validation selects alpha and a balanced-accuracy threshold; models are not refitted on train plus validation. The saved outputs cover **120 conditions, 300 candidate fits, and 200,520 test prediction rows**.

## Files

- [Detailed benchmark report](REPORT.md): paired effects, balanced accuracy, and age/history subgroups.
- [Recording-pattern findings](BINNING_BY_RECORDING.md): new five- versus fifteen-minute comparisons from saved scores.
- [Benchmark metrics](artifacts/apple_benchmark_metrics.csv) and [alpha selection](artifacts/alpha_selection.csv).
- [Benchmark implementation](scripts/04_run_apple_benchmark.py) and [saved-score summary implementation](scripts/06_summarize_binning_by_recording.py).
- Local inputs: `artifacts/scaling_predictions_test.parquet`, `artifacts/apple_10_allocations.parquet`, and `artifacts/cache_5min/` / `cache_15min/`.

The October 15 additions are maintained in these documents. Running `05_audit_and_report.py` regenerates the original README and report, replacing those additions; the separate recording-pattern report and CSVs are unaffected.
