# Garmin MultiRocket models

The model converts each adequate Garmin HR day into sequence features, pools those features across a participant’s available days, and fits a ridge regression score for recorded salutation. The sequence features supplement the participant summary and recording-mask features.

This note defines the inputs, transform, feature dimensions, and score. It then distinguishes the historical day-budget experiment from the saved Heartbeat Lab bundles. The [results reference](../reports/results_reference.md) compares experiments; the [results audit](../reports/results_audit_2026-10-14.md) records the earlier verification and evaluation issues.

## Target and input

- **Prediction target:** recorded salutation (codes `10` → class 0 [F], `20` → class 1 [M], per the notebook mapping).
- **Caveat:** predictability in this selected cohort; not biological sex, gender identity, or a causal physiological effect.
- **Input:** Garmin (source 3) channel-3000 epoch HR, 5-minute clock bins ($L = 288$ bins per local day).

## Cohort and history

- **Size:** 3,848 participants (strict-coverage Garmin cohort, `experiments/artifacts_sq/cohort_manifest_model_sources.parquet`).
- **History length:** adequate days per participant: mean 425, min 80, p50 476, max 979. **80 is the observed minimum of the pool, not a newly applied ≥80-day gate** — the cohort was selected by earlier eligibility rules (see `HEARTBEAT_LAB/AUDIT_REPORT.md`). Historical references to a "≥80-day cohort gate" are imprecise.
- **Adequate day:** ≥8 observed clock-hours AND (≥60 events OR ≥8 h interval-union coverage); HR kept in 25–230 bpm. The historical day-selection predicate is verified in `experiments/r1b_dateaware_garmin_2026-09-21/run_r1b.py:363–366`; `HEARTBEAT_LAB` uses the same adequacy thresholds. This does not assert identical raw-event handling in every pipeline version.
- **Training / validation / test:**
  - historical experiments: 10 demographically balanced OR-Tools allocations, ≈70/15/15; 2,696 / 579 / 573 participants per allocation
  - `mr_clock` and `mr_global` bundles below: 2,692 / 578 / 578 (single split)

## Transform and feature dimensions

```text
One adequate day: 288 five-minute HR bins
    → fill missing bins with training-derived values
    → MultiRocket on HR and its first differences
    → 9,408 daily sequence features
All selected days for one participant
    → mean and sample SD of each daily feature
    → 18,816 sequence features + 305 summary/mask features
    → 19,121 columns before feature filtering
    → ridge regression → one continuous score
```

Vendored `ChangWeiTan/MultiRocket`, commit `3ccaa4f8`; called with `num_features=1250`, `max_dilations_per_kernel=32`, `n_features_per_kernel=4`.

### Base kernels

- Length 9; weights −1 at six positions and +2 at three positions (sum = 0).
- One kernel for each choice of the three +2 positions: $\binom{9}{3} = 84$.

### Dilations and fitted biases

- `num_features // 84 = 1250 // 84 = 14` dilation slots per kernel.
- Slots are spaced log-uniformly on $[0, \log_2((L-1)/8)]$ and then truncated to integers, which gives duplicates. Distinct values: {1, 2, 3, 5, 6, 9, 11, 15, 20, 27, 35}, with 1 used 3×, 3 used 2×, and the rest once (sum 14). The same set results for the difference series ($L = 287$).
- Biases: one bias per (kernel, dilation slot), taken as a quantile of the convolution output on a randomly chosen training day. These are the only fitted, data-dependent parameters of the transform.
- 84 × 14 = **1,176** kernel/dilation/bias channels per transform; repeated dilation slots use different bias thresholds.

### Daily features

- Inputs: filled HR series $X$ and the first difference $\Delta X$, each with its own fitted biases.
- Pooling per convolution output: PPV (proportion of positive values), MPV (mean of positive values), MIPV (mean of indices of positive values), LSPV (longest stretch of positive values).
- Per day: 1,176 × 4 × 2 = **9,408** features.

### Participant features and ridge score

- Across days: mean and sample SD of every daily feature → 18,816 features.
- Plus 305 B40 static features (daily-stat means/SDs, coverage, weekday-minus-weekend mean-HR contrast, 288 per-bin observation frequencies) → **19,121 columns before feature filtering**. For the historical full-history experiment, 19,109–19,121 remain after training-derived feature filtering across the saved budget/allocation grid.
- **Input widths:** historical B40 and the current Lab summary implementation each contain 305 features. Saved bundle input widths are 305 for summary and 19,121 for `mr_clock` and `mr_global`, confirmed from their serialized `n_features_in_` values.
- Head: train-only median imputation → variance filter → `StandardScaler` → **`sklearn.linear_model.Ridge(alpha=1000)` regressed on 0/1 labels**. This is not `RidgeClassifier`; the output is a continuous score, used directly for AUROC.

## Saved model results

The evaluated transforms are MultiRocket and HYDRA. References to MiniRocket in older figure comments do not describe these fitted models.

### Historical day-budget experiment

- Script: [`day_scaling_all.py`](../experiments/temporal_garmin/day_scaling_all.py) · Report: [`DAYS_SCALING_REPORT.md`](../experiments/temporal_garmin/results/DAYS_SCALING_REPORT.md) · Metrics: [`days_scaling_all_metrics.csv`](../experiments/temporal_garmin/results/days_scaling_all_metrics.csv)
- MR, α = 1e3, test AUROC, mean ± SD over 10 allocations, $k$ days sampled uniformly from each participant’s full history:

| $k$ | 4 | 14 | 40 | 60 | 80 | all (mean 425) |
|---|---|---|---|---|---|---|
| test AUROC | 0.7132 ± 0.0268 | 0.7990 ± 0.0198 | 0.8583 ± 0.0115 | 0.8753 ± 0.0139 | 0.8854 ± 0.0115 | **0.9264 ± 0.0055** |

- Endpoint per-allocation test AUROC: **0.9187–0.9347** (all 10 > 0.91); mean validation AUROC 0.9235.
- At fixed $k$, only one allocation exceeds 0.90: **allocation 5** (k = 60: 0.9036; k = 80: 0.9060, or 0.9118 with α = 3e3). These are best-of-10 allocations, not representative performance.
- **Endpoint qualifiers:** single kernel draw (SD measures allocation variation conditional on that draw); variable per-participant budget (80–979 days); and the transform-exposure issue described below.

### Saved Heartbeat Lab bundles

Local path: `HEARTBEAT_LAB/artifacts/models/`. Each bundle has `model.joblib`, `metadata.json`, `predictions.csv`, `calibration.json`. All adequate days per participant (training histories 80–960 days).

| Bundle | Fill | Val AUROC (n=578) | Test AUROC (n=578) |
|---|---|---:|---:|
| `mr_global` | global median | 0.9034 | **0.9225** |
| `mr_clock` | clock-bin median | 0.8962 | 0.9177 |

- Fill values use all available training-participant days. MultiRocket bias fitting samples four days per training participant, with seed 20260930. Both steps exclude validation and test participants.
- This does **not** make them leakage-free in the broad sense: α = 1e3 and the overall recipe were chosen through earlier work on these same 3,848 people, and this is a single split with no external validation. Global − clock test Δ = +0.0048, bootstrap 95% [−0.0047, +0.0140] — no established fill advantage.

## Controls and evaluation limits

1. **Within-day value shuffle** (first-40-day window, v2): permuting observed HR values among each day's observed bins keeps that day's value multiset and mask, and refits the transform.
   - Test AUROC: MR 0.8361 → Shuffled 0.6643 (Δ = −0.172).
   - Validation AUROC: 0.8284 → 0.6619 (Δ = −0.166).
   - Interpretation: the fitted model is sensitive to value-to-clock-position arrangement. This does **not** identify the mechanism (circadian physiology vs. wear/behavior routines), and the shuffle jointly disrupts local order and clock alignment.
2. **Representation ladder** (first-40-day window, test AUROC): Summary RF 0.703 · Profile24 0.737 · Profile288 0.733 · MultiRocket 0.836 · HYDRA 0.844 · Combined (B40 + MR + HYDRA, without profile blocks) 0.857. The historical full-history MR endpoint reaches 0.926, subject to the exposure qualification below.
   - Going from hourly to 5-minute per-bin means did not help (Profile288 − Profile24 val Δ = −0.011, CI excludes 0). The tested convolutional representations outperform per-bin moment summaries at the same input resolution; this does not identify the physiological or behavioral mechanism.
3. **Transform exposure in the historical 0.9264 endpoint (confirmed in the audit):** Phase B fitted fill values and MR biases on allocation-0 training participants and reused them for all 10 allocations. For allocations 1–9, 380–413 of 573 test participants (mean 70.2%) had been in that transform-fitting set.
   - This is unsupervised exposure to held-out participants’ HR; labels were not used. The size and direction of its effect on AUROC are unknown.
   - The `HEARTBEAT_LAB` bundles avoid this specific issue. They are new fits (different split, seeds and fill windows), not a corrected replay, so their 0.9225 cannot be used to quantify the historical endpoint’s bias.
4. **Shared hyperparameter selection:** α was selected on allocation-0 validation and frozen for all allocations, and 80–98 of each later allocation's test participants were in that validation set. Together with repeated reuse of the same cohort, this means test AUROCs are exploratory, not independent confirmation.
5. **Uncertainty and extrapolation:** the ten allocations overlap, and their SDs/paired t-intervals do not provide independent-cohort population uncertainty. No fixed-budget point above 80 days was measured. The fitted log-linear and 1/√k curves do not bound the performance ceiling, and averaging a fixed-budget AUROC curve over individual history lengths does not identify heterogeneous-budget AUROC.
