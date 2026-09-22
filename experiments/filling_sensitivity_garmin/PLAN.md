# ch3000 filling sensitivity

2026-09-22. Plan only; implementation and execution are outside this change.

## Question and scope

Does participant-level HR prediction depend materially on how missing five-minute bins are filled? The recording-structure pilot found mask/count signal (~0.62–0.63 test AUROC), but did not establish how much the HR model (~0.85) depends on recording structure. This experiment changes filling only and refits the same model family.

Garmin/source 3, ch3000 only; recorded salutation 10→0, 20→1. Four arms × allocations **0, 1, 2**. No extra channels, HYDRA, raw-epoch scan, density manipulation, threshold calibration or exhaustive statistics.

## Inputs and fixed quantities

Read-only inputs, relative to repository root:

- `experiments/temporal_garmin/cache/day_bins.npz`: HR bin means, original masks/counts and user/date keys.
- `experiments/r1b_dateaware_garmin_2026-09-21/cache/day40_table.parquet` and `experiments/r1b_dateaware_garmin_2026-09-21/cache/folds.parquet`: designated days, daily statistics and participant splits.
- `artifacts_sq/cohort_manifest_model_sources.parquet` and `artifacts_sq/split_manifest_sq.parquet`: cohort and labels.
- `experiments/recording_structure_garmin/results/{metrics,predictions}.csv`: arm H reproduction reference.

Keep the same 3,848 participants, first 40 adequate days × 288 bins, and 2,696/579/573 train/validation/test participants per allocation. Retain original bin means at every observed position, original masks/counts and the existing 305-column B40. Never recompute B40 from filled sequences. Identify missing bins by the mask, not by HR values. No new inclusion gate or dropped days.

## Four fill methods

All arms use **B40 + pooled MultiRocket(HR)** followed by ridge.

| Arm | Missing-bin rule |
|---|---|
| H_clock | Current corrected fill: median observed HR bin mean at that clock bin across training participants/days; overall training median fallback. |
| H_global | One median over all observed HR bin means from training participants/days, regardless of clock time. |
| H_linear | Linear interpolation for every eligible short gap; H_clock fill everywhere else. |
| H_pchip | Local PCHIP for eligible short gaps with sufficient observed context; linear fallback for other eligible short gaps; H_clock fill everywhere else. |

Compute fill medians separately within each allocation using training participants only. Medians weight observed bins equally, not by event count.

Both interpolation arms use identical eligibility: complete interior missing runs of **1–6 consecutive five-minute bins**, bounded by originally observed bins within the same participant/day. This is 5–30 minutes of missing bins, with bounding bin centres 10–35 minutes apart; six bins is a fixed pilot convention, not a physiological threshold. Interpolate the entire eligible run; do not partially interpolate longer gaps, cross midnight, extrapolate day edges, or use previously filled values as endpoints. Same-participant observations are available across the full day for this retrospective task; this is not a real-time predictor.

- **H_linear:** for a run of k missing bins bounded by observed values x_L and x_R, fill position j with `x_L + j/(k+1) * (x_R - x_L)`, j=1,...,k.
- **H_pchip:** require two consecutive originally observed bins immediately before the gap and two immediately after it, all within the same day. Fit `scipy.interpolate.PchipInterpolator(..., extrapolate=False)` to exactly those four bin centres and observed HR means; evaluate only the missing positions. If either outer context bin is absent or missing, use the same linear rule as H_linear for the entire gap. Never seek more distant context across another gap.

Audit intervention exposure once from the masks: per-participant missing-bin fraction, fraction of missing bins eligible for interpolation, and fraction of days containing an eligible gap. For H_pchip also report fractions of all missing bins receiving PCHIP, linear fallback and H_clock, plus the PCHIP fraction among eligible bins. Report median/range and overall fractions; mark zero-denominator fractions unavailable. This distinguishes a robust result from an intervention that changed very little input.

## Matched fitting

- Implement separately in `experiments/filling_sensitivity_garmin/`. Reuse the recording pilot's loading, MR fit/pool and alpha-selection helpers plus the corrected temporal fill/B40 helpers. Do not invoke the full recording runner or modify existing experiment outputs.
- Match the **executed recording pilot's H**, including its actual seeds: day selection `comp_seed(r, 0)`, MR base `comp_seed(r, 1)`, difference `comp_seed(r, 2)`, where `comp_seed` uses `SeedSequence([20260921, 3, r, component])`. This differs from the older temporal runner's difference-seed convention; do not substitute that convention.
- Same four selected training days per participant and same fit seeds across fill arms. Refit MR separately on each arm's filled training inputs. Use the installed pinned implementation and existing budget: 1,250 features per transformation, 9,408 daily outputs; base + first differences; participant pooling by mean/sample SD over the same days. Keep the existing HR day-gate based on original masks.
- Fit each temporal block's imputation/zero-variance removal/scaling on training participants only. Share the unchanged, training-preprocessed B40 across arms within an allocation.
- Ridge with intercept, no class weighting. Select alpha separately per arm/allocation on validation AUROC from **[1e-3, 1, 1e3, 1e4, 1e5, 1e6]**; smallest alpha wins ties. Record boundary optima; no grid expansion or nested resampling. Validation maxima are selected estimates; test comparisons remain exploratory on this reused cohort.
- CPU only, ≤8 active threads; arms/allocations sequential, participant-batched transforms with immediate pooling. Reuse fitted blocks for the alpha sweep. Target <8 GiB; reduce batching if necessary. No ridge optimization project or broad environment setup.

## Checks, readout and stopping point

1. Before full fits, check exact user/date/fold alignment, unchanged observed HR/masks/counts/B40, finite filled inputs and training-only medians. Use a tiny fixture with a 1-bin gap, a 6-bin gap, a 7-bin gap and boundary gaps. Include sufficient four-point context, missing outer context and a turning point; verify the linear formula, PCHIP values staying within the bounding HR values, exact linear fallback and H_clock for ineligible gaps. Confirm identical interpolation eligibility across H_linear and H_pchip.
2. Fit H_clock first in allocation 0. Compare selected alpha, AUROCs and saved val/test scores with recording-pilot H (score max absolute difference ≤1e-6). Resolve any mismatch before interpreting alternative fills. Compare the remaining H_clock allocations when produced; do not silently redefine the reference.
3. Report per-allocation validation/test AUROC and paired test deltas **H_global − H_clock**, **H_linear − H_clock**, **H_pchip − H_clock** and **H_pchip − H_linear**, plus mean, range and direction agreement. Also report paired test-score Spearman correlations for each alternative against H_clock and H_pchip against H_linear: stable AUROC can conceal changed participant rankings. No bootstrap, p-values or multiplicity correction.
4. Treat a consistent change around **0.01 AUROC** as worth investigating, not a significance threshold. A gain or loss shows sensitivity to the filling convention under refitting. Small changes support robustness only to these tested alternatives; high interpolation ineligibility weakens conclusions about interpolation. Neither result quantifies physiological signal or excludes density, device/behavior effects, or interactions with recording structure. B40 still contains recording features throughout.
5. Stop after all **12 arm/allocation fits and one short report**. Recommend one next step based on the result; do not automatically extend allocations, add methods or start density experiments.

Write incremental `results/metrics.csv`, `results/predictions.csv` (user, allocation, split, arm, label, score), `results/repro.json` (config, seeds, alpha traces, input provenance, checks, timings/RSS and fill exposure), and `results/REPORT.md`. Keep caches local/gitignored and participant rows out of logs. Report what was observed separately from explanations. No publication or Git operation is part of this plan-writing task.
