# MultiRocket Sequence Modeling on Garmin Wearables

*Compiled 2026-10-14 from repository artifacts; audited against source CSVs/code the same day. No experiment code was run to produce these numbers — all values are read from saved outputs, except the dilation set (§4.2), which was recomputed from the vendored `_fit_dilations` logic.*

## 1. Problem Definition & Target

- **Prediction target:** recorded salutation (codes `10` → class 0 [F], `20` → class 1 [M], per the notebook mapping).
- **Caveat:** predictability in this selected cohort; not biological sex, gender identity, or a causal physiological effect.
- **Input:** Garmin (source 3) channel-3000 epoch HR, 5-minute clock bins ($L = 288$ bins per local day).

---

## 2. Garmin Cohort

- **Size:** 3,848 participants (strict-coverage Garmin cohort, `artifacts_sq/cohort_manifest_model_sources.parquet`).
- **History length:** adequate days per user: mean 425, min 80, p50 476, max 979. **80 is the observed minimum of the pool, not a newly applied ≥80-day gate** — the cohort was selected by earlier eligibility rules (see `HEARTBEAT_LAB/AUDIT_REPORT.md`). The README's "≥80-day cohort gate" wording is flagged there as imprecise.
- **Adequate day (as implemented in `HEARTBEAT_LAB`, notebook 01):** ≥ 8 observed clock-hours AND (≥ 60 events OR ≥ 8 h interval-union coverage); HR kept in 25–230 bpm. I did not verify that the historical experiments used an identical rule.
- **Splits:** 10 demographically balanced allocations (OR-Tools allocator), ≈70/15/15:
  - historical experiments: 2,696 / 579 / 573
  - `HEARTBEAT_LAB` bundles: 2,692 / 578 / 578 (single split)

---

## 3. Models Reaching > 0.90 AUROC

**Naming:** I found no MiniRocket implementation in the repo. "MiniRocket" appears only in one commit message / figure docstring (`figs_ppt/exp_garmin_temporal.py`) and one user question in a session transcript; the plotted data and all code use **MultiRocket** (Tan et al.). Caveat: my text search skipped `cache/`, `out/` and `data/` (large/symlinked dirs).

**Cohort size:** 3,848 is the *small strict-coverage subset* relative to the ~25k pooled cohort, but it is not small in absolute terms. The only genuinely small cohort found (`temporal_mr_200_40`, 125/33/32 customers) scored **0.56** test AUROC.

### A. Days-scaling, full-history pool (historical experiment)
- Script: [`day_scaling_all.py`](../experiments/temporal_garmin/day_scaling_all.py) · Report: [`DAYS_SCALING_REPORT.md`](../experiments/temporal_garmin/results/DAYS_SCALING_REPORT.md) · Metrics: [`days_scaling_all_metrics.csv`](../experiments/temporal_garmin/results/days_scaling_all_metrics.csv)
- MR, α = 1e3, test AUROC, mean ± SD over 10 allocations, $k$ days sampled uniformly from each user's full history:

| $k$ | 4 | 14 | 40 | 60 | 80 | all (mean 425) |
|---|---|---|---|---|---|---|
| test AUROC | 0.7132 ± 0.0268 | 0.7990 ± 0.0198 | 0.8583 ± 0.0115 | 0.8753 ± 0.0139 | 0.8854 ± 0.0115 | **0.9264 ± 0.0055** |

- Endpoint per-allocation test AUROC: **0.9187–0.9347** (all 10 > 0.91); mean validation AUROC 0.9235.
- At fixed $k$, only one allocation exceeds 0.90: **allocation 5** (k = 60: 0.9036; k = 80: 0.9060, or 0.9118 with α = 3e3). These are best-of-10 folds, not representative performance.
- **Endpoint qualifiers:** single kernel draw (CI is fold-split-only); variable per-user budget (80–979 days); and the transform-exposure defect in §5.3.

### B. Pre-trained bundles in `HEARTBEAT_LAB/artifacts/models/`
Each has `model.joblib`, `metadata.json`, `predictions.csv`, `calibration.json`. All adequate days per customer (training histories 80–960 days).

| Bundle | Fill | Val AUROC (n=578) | Test AUROC (n=578) |
|---|---|---:|---:|
| `mr_global` | global median | 0.9034 | **0.9225** |
| `mr_clock` | clock-bin median | 0.8962 | 0.9177 |

- Fill values and MultiRocket kernels are fitted on training customers only (4 random days per training customer, seed 20260930), which avoids the §5.3 exposure.
- This does **not** make them leakage-free in the broad sense: α = 1e3 and the overall recipe were chosen through earlier work on these same 3,848 people, and this is a single split with no external validation. Global − clock test Δ = +0.0048, bootstrap 95% [−0.0047, +0.0140] — no established fill advantage.

---

## 4. Kernel and Feature Architecture

Vendored `ChangWeiTan/MultiRocket`, commit `3ccaa4f8`; called with `num_features=1250`, `max_dilations_per_kernel=32`, `n_features_per_kernel=4`.

### 4.1 Base kernels: 84
- Length 9; weights −1 at six positions and +2 at three positions (sum = 0).
- One kernel for each choice of the three +2 positions: $\binom{9}{3} = 84$.

### 4.2 Dilations: 14 per kernel, 11 distinct values
- `num_features // 84 = 1250 // 84 = 14` dilation slots per kernel.
- Slots are spaced log-uniformly on $[0, \log_2((L-1)/8)]$ and then truncated to integers, which gives duplicates. Distinct values: {1, 2, 3, 5, 6, 9, 11, 15, 20, 27, 35}, with 1 used 3×, 3 used 2×, and the rest once (sum 14). The same set results for the difference series ($L = 287$).
- Biases: one bias per (kernel, dilation slot), taken as a quantile of the convolution output on a randomly chosen training day. These are the only fitted, data-dependent parameters of the transform.
- 84 × 14 = **1,176** convolution outputs per transform.

### 4.3 Two inputs × four pooling operators
- Inputs: raw series $X$ and the first difference $\Delta X$, each with its own fitted biases.
- Pooling per convolution output: PPV (proportion of positive values), MPV (mean of positive values), MIPV (mean of indices of positive values), LSPV (longest stretch of positive values).
- Per day: 1,176 × 4 × 2 = **9,408** features.

### 4.4 Participant aggregation and classifier
- Across days: mean and sample SD of every daily feature → 18,816 features.
- Plus 305 B40 static features (daily-stat means/SDs, coverage, weekend–weekday contrast, 288 per-bin mask means) → **19,121 columns before hygiene**. After the train-only `VarianceThreshold`, 19,109–19,121 remain depending on the allocation.
- Head: train-only median imputation → variance filter → `StandardScaler` → **`sklearn.linear_model.Ridge(alpha=1000)` regressed on 0/1 labels**. This is not `RidgeClassifier`; the output is a continuous score, used directly for AUROC.

---

## 5. Controls & Audit Findings

1. **Within-day value shuffle** (first-40-day window, v2): permuting observed HR values among each day's observed bins keeps that day's value multiset and mask, and refits the transform.
   - Test AUROC: MR 0.8361 → Shuffled 0.6643 (Δ = −0.172).
   - Validation AUROC: 0.8284 → 0.6619 (Δ = −0.166).
   - Interpretation: value-to-clock-position arrangement carries most of MR's increment. This does **not** identify the mechanism (circadian physiology vs. wear/behavior routines), and the shuffle jointly disrupts local order and clock alignment.
2. **Representation ladder** (first-40-day window, test AUROC): Summary RF 0.703 · Profile24 0.737 · Profile288 0.733 · MultiRocket 0.836 · HYDRA 0.844 · Combined 0.857. On the full history, MR reaches 0.926.
   - Going from hourly to 5-minute per-bin means did not help (Profile288 − Profile24 val Δ = −0.011, CI excludes 0). The gain comes from the convolutional representation, not from bin resolution alone.
3. **Transform exposure in the historical 0.9264 endpoint (confirmed in the audit):** Phase B fitted fill values and MR biases on allocation-0 training users and reused them for all 10 fold-splits. For allocations 1–9, 380–413 of 573 test users (mean 70.2%) had been in that transform-fitting set.
   - This is unsupervised exposure to held-out users' HR; labels were not used. The size and direction of its effect on AUROC are unknown.
   - The `HEARTBEAT_LAB` bundles avoid this specific issue. They are new fits (different split, seeds and fill windows), not a corrected replay, so their 0.9225 cannot be used to quantify the old inflation.
4. **Shared hyperparameter selection:** α was selected on allocation-0 validation and frozen for all allocations, and 80–98 of each later allocation's test users were in that validation set. Together with repeated reuse of the same cohort, this means test AUROCs are exploratory, not independent confirmation.
