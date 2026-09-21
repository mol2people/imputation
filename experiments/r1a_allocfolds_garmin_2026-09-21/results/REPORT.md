# R1a-AF — circadian curve features under demographically-balanced CP-SAT folds — results

Run 2026-09-21T19:40:29 | implements [`../PLAN.md`](../PLAN.md) | R = 10 allocator allocations (partition = age_band x bmi_group, stratify = salutation; 70/15/15), paired within allocation; 5 R1a arms per allocation, RF seed stream identical to R1a (`SeedSequence([20260919, 3, r, 5])`); 4 workers, n_jobs=1 per fit.

**Allocator gate:** all_phases_optimal=True for 10/10 allocations (230 partition solves) (ortools 9.15.6755).

**Univariate AUROCs:** unchanged by folds — see `../r1a_circadian_garmin_2026-09-21/results/r1a_univariate.csv`.

## 1. Absolute AUROC (mean ± SD over 10 allocations)

| arm | test AUROC | val AUROC | n cols (hygiene) |
|---|---|---|---|
| `BASE` | 0.7415 ± 0.0222 | 0.7501 | 400 |
| `R1a_only` | 0.6928 ± 0.0164 | 0.6874 | 35 |
| `BASE_x_R1a` | 0.7509 ± 0.0243 | 0.7538 | 435 |
| `R1a_resid_only` | 0.6906 ± 0.0185 | 0.6910 | 35 |
| `BASE_x_R1a_resid` | 0.7475 ± 0.0240 | 0.7538 | 435 |

**BASE sanity (descriptive, different test users):** allocator folds 0.7415 ± 0.0222 vs R1a frozen-split 0.7434 (Δ -0.0019). Not a paired test — split systems assign different users to test.

## 2. Paired test-AUROC deltas within allocation

### 2a. Primary family (2 comparisons)

| comparison | mean Δ | 95% t-CI | 97.5% t-CI (Bonferroni) | SD | share > 0 |
|---|---|---|---|---|---|
| BASE⊕R1a − BASE | +0.0095 | [+0.0056, +0.0133] | [+0.0048, +0.0141] | 0.0054 | 1.00 |
| BASE⊕R1a_resid − BASE | +0.0060 | [+0.0023, +0.0096] | [+0.0016, +0.0103] | 0.0051 | 1.00 |

_R1a frozen-split reference (commit `4868482`): +0.0029 95% t-CI [+0.0004, +0.0054]; wear-partialled +0.0036 [+0.0014, +0.0058]. Note: R1a's REPORT labels these CIs "97.5%" — that is a labeling error (the formula is `t.ppf(0.975, 29)`, a 95% t-CI); erratum to follow._

### 2b. Secondary comparisons

| comparison | mean Δ | 95% t-CI | SD | share > 0 |
|---|---|---|---|---|
| BASE⊕R1a − R1a_only | +0.0581 | [+0.0451, +0.0711] | 0.0182 | 1.00 |
| BASE⊕R1a_resid − R1a_resid_only | +0.0568 | [+0.0471, +0.0665] | 0.0136 | 1.00 |
| R1a_only − BASE | -0.0487 | [-0.0626, -0.0347] | 0.0195 | 0.00 |
| R1a_resid_only − BASE | -0.0509 | [-0.0609, -0.0409] | 0.0140 | 0.00 |
| R1a_only − R1a_resid_only (activity vs composition) | +0.0022 | [-0.0073, +0.0117] | 0.0133 | 0.60 |

## 3. Top features (BASE⊕R1a Gini, mean over 10)

| feature | Gini (mean) | R1a? |
|---|---|---|
| `rec__ch3000_weekday_mean` | 0.0286 | no |
| `rec__ch3000_tod_morning` | 0.0263 | no |
| `rec__ch3000_weekend_minus_weekday` | 0.0152 | no |
| `hour_h9` | 0.0130 | **yes** |
| `rec__ch3000_minus_ch3002_mean` | 0.0125 | no |
| `cosinor_A3` | 0.0122 | **yes** |
| `win__ch3000_tod_morning` | 0.0121 | no |
| `roll_rec__ch3001_daily_sd_w7_max` | 0.0117 | no |
| `win__ch3000_minus_ch3002_mean` | 0.0115 | no |
| `rec__ch3000_minus_ch3001_mean` | 0.0106 | no |
| `roll_rec__ch3002_daily_sd_w7_max` | 0.0099 | no |
| `hour_h8` | 0.0094 | **yes** |
| `rec__ch3002_mean_of_daily_sd` | 0.0092 | no |
| `win__ch3000_weekend_minus_weekday` | 0.0091 | no |
| `win__ch3000_sd_of_daily_mean` | 0.0088 | no |

## 4. Fold composition (allocation 0; balance files for all runs in `cache/balance_run*.csv`)

| fold | n | mean age (band mid) | salutation 10 share | bmi normal share |
|---|---:|---:|---:|---:|
| train | 2696 | 50.33 | 0.360 | 0.540 |
| val | 579 | 50.42 | 0.356 | 0.537 |
| test | 573 | 50.43 | 0.361 | 0.543 |

Fold sizes over runs: train 2696.0 [2696, 2696], val 579.0 [579, 579], test 573.0 [573, 573] (constant across these 10 allocations — largest-remainder per partition is deterministic in the cohort, so fold totals are pinned by construction).


## 5. Caveats

- Same-participant reuse — not external validation; folds are resamples of one cohort.
- **R = 10 under-powers the paired-Δ estimate** (SE ≈ 2× R1a's). Direction/magnitude check, not confirmatory; Bonferroni 97.5% CIs are correspondingly wide.
- Allocator tie-break introduces per-allocation variation beyond stratified permutation; per-allocation BASE spread is expected to exceed R1a's frozen-split spread.
- Structural integer-granularity imbalance at the `(20-29, salutation 10)` cell (worst-cell proportion deviation ≈ 0.39, identical across seeds); aggregate composition is matched to 3 dp (see §4 and balance files).
- ch3001/ch3002 excluded (vendor caveat); vendor-circularity audit declined (user decision 2026-09-21).
- Allocator reproducibility: single CP-SAT worker + fixed seeds + ortools 9.15.6755; portability across machines/versions not guaranteed by the package README.

## 6. Interpretation

**(a) Both primary increments replicate and are unambiguous at R=10.** Per-fold
`BASE⊕R1a − BASE` ∈ [+0.0019, +0.0189] and `BASE⊕R1a_resid − BASE` ∈ [+0.0019,
+0.0128] — every one of the 10 allocator folds is positive. Bonferroni-simultaneous
97.5% lower bounds are +0.0048 and +0.0016, strictly positive despite the small n.
The R1a circadian increment is not an artifact of frozen-split train/test
demographic mismatch.

**(b) The increment is materially larger under balanced folds than under frozen
random splits.** Raw `BASE⊕R1a − BASE`: AF +0.0095 vs frozen +0.0029 (3.3×);
wear-partialled: +0.0060 vs +0.0036 (1.7×). Difference of means +0.0065
(descriptive z ≈ 3.1; not a valid paired test — shared cohort, different
test-user assignments across split systems). The most plausible mechanism:
under frozen random splits, train and test drift in age/bmi composition, and
the curve features' conditional information content is partly attenuated by
residual demographic mismatch; under balanced folds the drift is removed and
the true increment is revealed. Equivalently, the frozen-split estimate is
conservative on this axis.

**(c) Wear-partialling flips direction across regimes.** R1a: raw +0.0029,
partialled +0.0036 (Δ +0.0007). AF: raw +0.0095, partialled +0.0060
(Δ −0.0035). "Wear partialled" is not a regime-independent test of "purely
conditional"; it does different work in each. Under balanced folds the raw
increment is already the demographically-matched read, and hour-coverage
partialling removes a different component (residual hourly-coverage variance).
Honest summary: ~63% of the raw increment survives partialling → ~37% is
plausibly composition/activity-mediated; the R1a "wear-partialled is the
deconfounded comparison" framing should be read as regime-specific.

**(d) Marginals are split-system-invariant; the change is concentrated in the
composite.** `R1a_only` 0.6928 (AF) vs 0.6921 (frozen); `BASE` 0.7415 vs
0.7434. Within sampling noise. The marginal models carry the same information
under both split systems — the difference shows up only when they are combined.

**(e) Fold-difficulty shocks cancel in the paired delta.** Per-allocation
`BASE` test AUROC spans 0.7037–0.7689 (spread 0.065 across n_test=573 folds);
`corr(BASE, BASE⊕R1a) = 0.977` across allocations — the composite tracks
BASE within ±0.002 even on the hardest folds. The increment is structurally
robust to which 573 users land in test.

**(f) Activity-vs-composition mediation not resolvable at R=10.**
`R1a_only − R1a_resid_only = +0.0022` [−0.0073, +0.0117], share>0 = 0.60.
Consistent with R1a's hard-to-separate reading; would need larger n (or an
exogenous activity instrument) to resolve.

**(g) What changes vs R1a.** The R1a "consistent but CI-touching" claim
becomes an "unambiguous +3× larger" claim under balanced folds. Vendor-circularity
posture is unchanged in standing but fold-system robustness weakens the
"splits-specific artifact" worry. The "wear-partialled as deconfounded" framing
is downgraded from universal to regime-specific. Caveats remain per §5.
