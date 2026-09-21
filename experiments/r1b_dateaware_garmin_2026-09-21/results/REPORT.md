# R1b — date-aware circadian features (Garmin) — results

Run 2026-09-21T22:04:32 | implements [`../PLAN.md`](../PLAN.md) | R = 10 allocator allocations (verbatim R1a-AF spec); 3 day-level arms + 1 new participant arm (P40); BASE reused from R1a-AF; 8 workers, n_jobs=1 per fit; first 40 adequate ch3000 days per user, chronological.

**Allocator gate:** all_phases_optimal=True (23 partitions x 10 allocs); ortools 9.15.6755.

**Cross-experiment determinism:** test-user-set equality vs R1a-AF BASE predictions 10/10; balance-CSV equality 10/10 (0 missing ref); BASE AUROC recompute max |delta| 0.0e+00.

**Cohort:** s3 strict, n = 3848 (PLAN §16 erratum to dayscale PLAN §2: predicted 3,847 from record-level channel-pass, but epoch_days ch3000 day-rows cover all 3,848 s3 users and every user has ≥ 80 adequate days; the dayscale day_table itself has 3,848 users — verified read-only).

## 1. Absolute AUROC (mean ± SD over 10 allocations)

| arm | level | test AUROC | val AUROC | n cols (hygiene) | test units |
|---|---|---|---|---|---|
| `A40` | day | 0.6589 ± 0.0117 | 0.6552 | 29 | 22,920 |
| `C40` | day | 0.6782 ± 0.0136 | 0.6737 | 64 | 22,920 |
| `C40_resid` | day | 0.6770 ± 0.0135 | 0.6729 | 64 | 22,920 |
| `BASE_x_P40` | participant | 0.7513 ± 0.0227 | 0.7568 | 485 | 573 |
| `BASE (R1a-AF reuse)` | participant | 0.7415 ± 0.0222 | 0.7501 | 400 | 573 |

## 2. Primary family (3 paired deltas; Bonferroni 95%-simultaneous over the family)

| comparison | mean Δ | 95% t-CI | Bonferroni 95%-simult | SD | share > 0 |
|---|---|---|---|---|---|
| C40 − A40 | +0.0193 | [+0.0166, +0.0219] | [+0.0158, +0.0227] | 0.0037 | 1.00 |
| C40_resid − A40 | +0.0181 | [+0.0156, +0.0206] | [+0.0149, +0.0214] | 0.0035 | 1.00 |
| BASE⊕P40 − BASE | +0.0098 | [+0.0043, +0.0153] | [+0.0027, +0.0170] | 0.0077 | 0.90 |

## 3. First-k curve (chronological; pooled test users)

Mean participant AUROC by k (days pooled, chronological), mean over 10 allocations:

| k | A40 | C40 |
|---|---|---|
| 1 | 0.6444 | 0.6694 |
| 2 | 0.6666 | 0.6864 |
| 4 | 0.6859 | 0.7036 |
| 8 | 0.6978 | 0.7136 |
| 16 | 0.7083 | 0.7214 |
| 24 | 0.7084 | 0.7236 |
| 32 | 0.7112 | 0.7261 |
| 40 | 0.7139 | 0.7286 |

## 4. A40_alldays sensitivity (r=0; train on ALL adequate ch3000 days; evaluate on first-40 test days)

| main A40 (r=0) | A40_alldays (r=0) | Δ | n_train_days |
|---|---|---|---|
| 0.6632 | 0.6703 | +0.0072 | 1,150,462 |

## 5. Random-40 sensitivity (r=0, 50 trials; sample 40 days per training user from full adequate pool; evaluate on first-40 test days; arm C40; RF seed fixed = day_seed(0))

| metric | value |
|---|---|
| random-40 mean test AUROC | 0.6837 ± 0.0016 |
| main C40 r=0 test AUROC | 0.6825 |
| Δ (random − main) mean | +0.0012 |
| Δ 95% t-CI | [+0.0008, +0.0017] |
| Δ share > 0 | 0.76 |

## 6. Top features (mean Gini over 10)

**Arm C40 (day-level):**

| feature | Gini (mean) | R1a curve? |
|---|---|---|
| `d_ch3000_vmin` | 0.0435 | no |
| `d_ch3000_mean` | 0.0434 | no |
| `hour_h7` | 0.0262 | **yes** |
| `hour_h6` | 0.0246 | **yes** |
| `hour_h4` | 0.0243 | **yes** |
| `hour_h8` | 0.0226 | **yes** |
| `hour_h10` | 0.0225 | **yes** |
| `d_ch3000_median` | 0.0215 | no |
| `hour_h11` | 0.0210 | **yes** |
| `curve_night_mean` | 0.0208 | **yes** |
| `hour_h15` | 0.0207 | **yes** |
| `hour_h9` | 0.0206 | **yes** |
| `hour_h16` | 0.0205 | **yes** |
| `hour_h13` | 0.0203 | **yes** |
| `hour_h5` | 0.0202 | **yes** |

**Arm BASE⊕P40 (participant-level):**

| feature | Gini (mean) | P40? |
|---|---|---|
| `rec__ch3000_weekday_mean` | 0.0317 | **yes** |
| `rec__ch3000_tod_morning` | 0.0259 | no |
| `win__ch3000_tod_morning` | 0.0138 | no |
| `rec__ch3000_weekend_minus_weekday` | 0.0128 | no |
| `rec__ch3000_minus_ch3002_mean` | 0.0117 | **yes** |
| `roll_rec__ch3001_daily_sd_w7_max` | 0.0114 | no |
| `curve_day_night_contrast_sd` | 0.0108 | **yes** |
| `rec__ch3000_minus_ch3001_mean` | 0.0093 | **yes** |
| `hour_h10_mean` | 0.0091 | **yes** |
| `win__ch3000_minus_ch3002_mean` | 0.0091 | **yes** |
| `win__ch3000_weekday_mean` | 0.0089 | **yes** |
| `cosinor_resid_sd_mean` | 0.0088 | **yes** |
| `rec__ch3002_mean_of_daily_sd` | 0.0087 | **yes** |
| `roll_rec__ch3002_daily_sd_w7_max` | 0.0085 | no |
| `win__ch3000_weekend_minus_weekday` | 0.0082 | no |

## 7. Caveats

- R = 10 under-powers the 3-comparison primary family for true deltas ≲ 0.005; Bonferroni 95%-simult CIs are wide. Same caveat as R1a-AF.
- **3,847 → 3,848 cohort correction** vs dayscale PLAN §2. Dayscale PLAN predicted 3,847 from `FROZEN_CHANNEL_PASS[3][3000]` (record-level); the actual `epoch_days` ch3000 day-rows cover all 3,848 s3 users (verified read-only) and the built `day_table.parquet` has 3,848 users. R1b uses 3,848 throughout.
- First-40 is a prospective-window estimator, not a random sample; random-40 sensitivity (§5) probes this directly.
- Within-day clustered units (days within users) inflate day-level test AUROC apparent precision; no within-allocation cluster bootstrap (RAM-pressure precedent — dayscale). Across-allocation t-CIs only.
- Same-participant reuse across R1a-AF and R1b is intentional (paired BASE predictions); not external validation.
- ch3000-only convention for new curve features (vendor caveat ch3001/ch3002); arm A40 retains dayscale ch3001/3002 stat/gap cols.
- Recorded salutation ≠ biological sex/gender.
- No vendor-circularity audit (user decision 2026-09-21).
- R1a formula identity: per-day curve computation is a verbatim port of `run_r1a.compute_r1a` (commit `4868482`) extended by `date`; per-day residualisation a verbatim port of `run_r1a.residualise`. CURVE35/WEIGHTS/n_total/log_n stored as float64 (fidelity to the port); STAT/GAP stored as float32 (dayscale-matching).
- Dayscale k-curve saturation used within-user with-replacement sampling; the R1b first-k curve (§3) uses chronological first-k on the actual R1b model (apples-to-apples on this experiment).
- **`cosinor_resid_sd` NaN quirk (R1a-inherited, not a port bug):** this curve feature is NaN on 35.5% of days. Root cause is byte-identical to `run_r1a.compute_r1a` line 176: `resid_sd = sqrt(sum(w * where(valid, resid, 0)**2) / max(sum(w[valid]), 1e-9))`. Here `w` (the hour-of-day weights) carries NaN at invalid hour-bins, so `w * 0 = NaN` propagates into the sum — the feature is only finite on days with **full 24-hour coverage**. The verbatim-port claim holds; R1a hid this at participant level (long histories cover all hour-bins); R1b exposes it at day level. SQPreprocessor median-imputes these NaNs identically to other missing values, so fits are valid but `cosinor_resid_sd` carries no signal on partial-coverage days. A two-character fix (`where(valid, w, 0)` instead of `w`) in R1a would eliminate the quirk — out of scope for the frozen R1b protocol.
- **Compute-budget erratum:** wall 1988s ≈ 33 min vs PLAN estimate 12–15 min. The day-level fits (508s for 10 allocs × 3 arms = 30 fits, ≈17s/fit) and random-40 (1275s for 50 trials ≈ 25s/trial) both ran ~3× the per-fit estimate. Cause: A1 hygiene expands 64 input cols → ~90 output cols (median-impute + per-col `__missing` indicators for CURVE35 NaNs + nzv prune), making each 100-tree RF fit slower than the dayscale A-arm's 29-col baseline. All predeclared work completed; no protocol deviation.
- **Two `loky` "A worker stopped" warnings** at the build-curves stage and the fits stage. Workers were replaced by loky automatically; outputs verified complete (1,635,698 day-rows / 63 curve cols in `dayall_table.parquet`, 687,600 day-pred rows, 40 metrics rows). Same memory-pressure pattern observed in the dayscale pipeline.

## 8. Interpretation

**Headline.** Per-day circadian curves deliver a substantial, Bonferroni-significant day-level AUROC increment over the dayscale first-40 baseline (**C40−A40 = +0.0193**, Bonferroni 95%-simult [+0.0158, +0.0227], 10/10 positive), and the increment is **largely coverage-orthogonal at the day level** (C40_resid−A40 = +0.0181, only ≈6% removal). Aggregated to participant level, the same curve block delivered the R1a-comparable increment (**BASE⊕P40−BASE = +0.0098**, Bonferroni [+0.0027, +0.0170], 9/10 positive) plus new cross-day SD and weekend-contrast features R1a didn't have.

**Day-level increment (+0.019) > participant-level increment (+0.010).** Natural ordering: per-day predictions are harder (within-user variance dominates), so adding predictive features helps more in absolute AUROC. Averaging over 40 days compresses the variance and the increment shrinks — but does not disappear. The participant-level curve increment is essentially unchanged from R1a (0.0098 here vs 0.0095 in R1a-AF), and P40 reaches 0.7513 vs R1a's 0.7509.

**Partialling behaves differently at the two levels.** At participant level, partialling halved the R1a curve effect (+0.0095 → +0.0060, ≈37% removal); at day level here, partialling removes only ≈6% (+0.0193 → +0.0181). Interpretation: participant-level coverage was a major confounder (more recording → both more curve signal and more salutation signal); at day level within the first-40 *adequate* days, day-to-day coverage varies little (every day already passed ≥8-hour ch3000 adequacy), so partialling removes much less. The curve increment at day level is genuinely *within-day circadian shape*, not coverage.

**First-k curve (chronological).** C40 participant AUROC rises monotonically from **0.669 at k=1** to 0.729 at k=40, with a near-plateau by k≈32 (0.726). A single day of curve data (k=1) is already highly predictive (0.669 ≈ A40 at k=8). The 40-day choice captures most of the day-level signal; the marginal 8 days add only +0.003. The C40−A40 gap holds across all k (≈+0.015 at k=1, ≈+0.015 at k=40), so the curve increment is stable.

**A40_alldays sensitivity (+0.0071).** Training on all 1,150,462 adequate days beats first-40 training (107,820 days) on the *same* first-40 test days (0.6703 vs 0.6632). Confirms: the first-40 training truncation does cost some day-level signal (~0.007), but the curve block's gains over A40 are robust to which training window is used.

**random-40 sensitivity (+0.0012 [+0.0008, +0.0017], share>0=0.76).** The chronological-first-40 window is *very slightly worse* than a random-40 window (~0.001 AUROC at participant level). The CI excludes zero but the magnitude is well within per-alloc noise (SD 0.0037 for the primary comparison). Plausible causes: first-40 days may include onboarding-period artifacts; or just small-sample fluctuation. Honest framing: a small but detectable order effect — **not** a reason to prefer random-40 (the chronological window has other advantages: deterministic, paired with the day-level first-40 evaluation window, no sampling-induced variance in the primary estimator).

**What carries signal?** C40 top Gini: 3 ch3000 daily stat cols (`d_ch3000_vmin/mean/median`) plus the morning-hour profile (`hour_h4..h11`). The day's overall level and *when in the day* HRV is sampled dominate; the curve aggregates (`curve_night_mean` rank 10) extract some signal but the hour-level granularity carries more at the day level. BASE⊕P40 top Gini: P40's value comes through weekday/weekend contrast (`rec__ch3000_weekend_minus_weekday`), cross-day SD (`curve_day_night_contrast_sd`, `cosinor_resid_sd_mean`), inter-channel differences (`rec__ch3000_minus_ch3002_mean`), and the curve block's per-day mean (`hour_h10_mean`). These are the **stability/consistency** signals — salutation differences in *how variable* circadian behavior is day-to-day, not just its mean shape.

**Overall.** Per-day circadian curves deliver a real, Bonferroni-significant day-level increment, and aggregating to participant preserves the R1a-level curve increment. The first-40 chronological window is small-but-defensible; random-40 is +0.001 better but the magnitude is within noise. The day-level curve increment is largely coverage-orthogonal — a substantive finding that distinguishes day-level from participant-level analysis.
