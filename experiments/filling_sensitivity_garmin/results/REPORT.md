# ch3000 filling sensitivity — results

**Audit update, 2026-10-14.** Saved metrics were checked against saved predictions.
The allocation-0 H_global/H_clock AUROCs are an exact ranking tie; a floating-point
difference previously counted it as positive. Observed small differences are
whole-pipeline sensitivity estimates, not equivalence bounds. See the
[results audit](../../../reports/results_audit_2026-10-14.md).

Implements [`../PLAN.md`](../PLAN.md). 4 fill arms × 3 allocations (0/1/2, chosen before results); cached 5-min bins; Garmin/source 3, ch3000 only; all arms `B40 + pooled MultiRocket(HR) + Ridge` with one shared seed stream per allocation (fit days `comp_seed(r,0)`, MR base `comp_seed(r,1)`, MR diff `comp_seed(r,2)` — the executed recording pilot's convention). Reference: recording pilot commit `f6123c9` (arm H).


## 1. Arm table (per-alloc validation / test AUROC)

| arm | alloc | val AUROC | test AUROC | n cols | α | boundary |
|---|---|---|---|---|---|---|
| `H_clock` | 0 | 0.8304 | 0.8525 | 19,121 | 10000 | no |
| `H_clock` | 1 | 0.8153 | 0.8285 | 19,121 | 1000 | no |
| `H_clock` | 2 | 0.8476 | 0.8581 | 19,115 | 10000 | no |
| `H_clock` | **mean ± SD** | 0.8311 ± 0.0161 | 0.8464 ± 0.0157 | 19,119 | — | — |
| `H_global` | 0 | 0.8234 | 0.8525 | 19,121 | 10000 | no |
| `H_global` | 1 | 0.8110 | 0.8316 | 19,121 | 10000 | no |
| `H_global` | 2 | 0.8458 | 0.8613 | 19,115 | 10000 | no |
| `H_global` | **mean ± SD** | 0.8268 ± 0.0176 | 0.8485 ± 0.0153 | 19,119 | — | — |
| `H_linear` | 0 | 0.8282 | 0.8529 | 19,121 | 10000 | no |
| `H_linear` | 1 | 0.8143 | 0.8295 | 19,121 | 10000 | no |
| `H_linear` | 2 | 0.8497 | 0.8588 | 19,115 | 10000 | no |
| `H_linear` | **mean ± SD** | 0.8307 ± 0.0178 | 0.8470 ± 0.0155 | 19,119 | — | — |
| `H_pchip` | 0 | 0.8278 | 0.8529 | 19,121 | 10000 | no |
| `H_pchip` | 1 | 0.8134 | 0.8297 | 19,121 | 10000 | no |
| `H_pchip` | 2 | 0.8498 | 0.8592 | 19,115 | 10000 | no |
| `H_pchip` | **mean ± SD** | 0.8303 ± 0.0183 | 0.8473 ± 0.0155 | 19,119 | — | — |

## 2. Paired TEST AUROC deltas (3 allocations; no p-values, PLAN §"Readout")

| comparison | alloc 0 | alloc 1 | alloc 2 | mean | range | direction |
|---|---|---|---|---|---|---|
| H_global − H_clock | 0.0000 | +0.0031 | +0.0032 | +0.0021 | [0.0000, +0.0032] | 2 +, 1 tie |
| H_linear − H_clock | +0.0003 | +0.0010 | +0.0007 | +0.0007 | [+0.0003, +0.0010] | all 3 + |
| H_pchip − H_clock | +0.0004 | +0.0012 | +0.0011 | +0.0009 | [+0.0004, +0.0012] | all 3 + |
| H_pchip − H_linear | +0.0001 | +0.0002 | +0.0004 | +0.0002 | [+0.0001, +0.0004] | all 3 + |

### 2b. Paired VALIDATION AUROC deltas (secondary)

| comparison | mean | range | direction |
|---|---|---|---|
| H_global − H_clock | -0.0044 | [-0.0070, -0.0018] | all 3 − |
| H_linear − H_clock | -0.0004 | [-0.0022, +0.0021] | 1/3 + |
| H_pchip − H_clock | -0.0008 | [-0.0027, +0.0022] | 1/3 + |
| H_pchip − H_linear | -0.0004 | [-0.0009, +0.0001] | 1/3 + |

## 3. Spearman ρ of paired test scores (per alloc)

| comparison | alloc 0 | alloc 1 | alloc 2 |
|---|---|---|---|
| H_global − H_clock | 0.9817 | 0.8940 | 0.9820 |
| H_linear − H_clock | 0.9927 | 0.9022 | 0.9907 |
| H_pchip − H_clock | 0.9920 | 0.9029 | 0.9906 |
| H_pchip − H_linear | 0.9994 | 0.9995 | 0.9993 |

## 4. H_clock reproduction vs recording pilot H (PLAN §"Checks" item 2)

| alloc | n | max |Δscore| | α | AUROC val (mine/ref) | AUROC test (mine/ref) |
|---|---|---|---|---|---|
| 0 | 1152 | 0.00e+00 | 10000 / 10000 | 0.8304 / 0.8304 | 0.8525 / 0.8525 |
| 1 | 1152 | 0.00e+00 | 1000 / 1000 | 0.8153 / 0.8153 | 0.8285 / 0.8285 |
| 2 | 1152 | 0.00e+00 | 10000 / 10000 | 0.8476 / 0.8476 | 0.8581 / 0.8581 |

## 5. Intervention exposure (from masks; alloc- and arm-independent)

- Missing bins (3,848 users × 40 days × 288 bins): **5,089,133** in **189,241** runs (boundary-touching 58,563; length > 6: 105,376).

- Interpolation-eligible (interior 1–6-bin runs bounded by observed bins): **77,539** runs / **197,148** bins = **3.87%** of missing bins.

- With 4-point PCHIP context: **68,705** runs / **174,198** bins = **3.42%** of missing; linear fallback (eligible, context incomplete): **8,834** runs / **22,950** bins = **0.45%**.

- Remaining on H_clock fill (ineligible gaps): **4,891,985** bins = **96.13%** of missing.

- Run-length histogram (1..9, >9): [28400, 19277, 13376, 9511, 7288, 6013, 5033, 4428, 4040, 91875].


### Per-participant exposure (median; [range])

| fraction | median | range | n users |
|---|---|---|---|
| missing bins / all bins | 0.0617 | [0.0014, 0.6020] | 3848 |
| eligible / missing | 0.0489 | [0.0000, 1.0000] | 3848 |
| days containing an eligible gap | 0.3250 | [0.0000, 1.0000] | 3848 |
| PCHIP-filled / missing (H_pchip) | 0.0436 | [0.0000, 1.0000] | 3848 |
| linear-fallback / missing (H_pchip) | 0.0000 | [0.0000, 0.2674] | 3848 |
| clock-filled / missing (all arms) | 0.9511 | [0.0000, 1.0000] | 3848 |
| PCHIP / eligible (H_pchip) | 0.9897 | [0.0000, 1.0000] | 3751 |

Zero-denominator fractions (users with no missing bins, or no eligible bins for the last row) are excluded and counted in the n-users column; users with zero missing bins: 0.


## 6. Checks (PLAN §"Checks")

| check | status |
|---|---|
| data: mask_eq_cnt_gt_0 | PASS |
| data: observed_hr_finite | PASS |
| data: bins_d40_user_date_multiset | PASS |
| data: exactly_40_bin_rows_per_user | PASS |
| data: fold_sizes_2696_579_573_all_allocs | PASS |
| data: B40_shape_305_finite | PASS |
| fixture: eligibility classification (1/6/7-bin, boundary, no-ctx, turning point) | PASS |
| fixture: overlays == naive loop reference (positions + values) | PASS (linear Δ 0.0e+00; pchip Δ 0.0e+00) |
| fixture: closed-form linear values (r0, r1) | PASS (Δ 0.0e+00) |
| fixture: PCHIP within 4-pt [min, max] (incl. turning point) | PASS |
| fixture: exact linear fallback + H_clock at ineligible + observed preserved | PASS (True, True, True) |
| fixture: fill_clock/fill_global train-only | PASS (True, True) |
| observed HR/masks/counts/B40 unchanged; finite filled inputs and Z (every arm, every alloc) | PASS (per-arm assertions) |
| identical interpolation eligibility across H_linear / H_pchip | PASS (single shared computation; fixture asserts pchip ∪ fallback = eligible, disjoint) |
| H_clock score match vs recording pilot H ≤ 1e-6 (α, AUROC equal) | PASS (run-time gate; see §4) |

## 7. Interpretation

**Pipeline validation is exact.** H_clock's saved scores are textually identical to the recording pilot's H arm across all 1,152 val+test users × 3 allocations (§4: max |Δscore| = 0.00e+00, α and AUROC equal), confirming alignment of the reference pipelines. Each fill arm also refits MR biases, feature hygiene and ridge, and selects its validation α; the comparison measures the fitted pipeline's response to the fill intervention.

**Observed AUROC differences are small in this pilot.** Mean test AUROC spans
0.8464–0.8485 across the four fills, with cross-allocation SD ≈0.016. The largest
mean paired test delta is +0.0021 (H_global − H_clock; two allocations positive,
one exact tie). With three allocations and reused splits, this observed range
is not an equivalence test or a bound on effects in other users or wear regimes.

**The two interpolation arms give very similar saved test rankings.** Spearman ρ(H_pchip, H_linear) = 0.9993–0.9995 on paired test scores (§3); mean test delta +0.0002. This pilot does not establish a useful predictive advantage for either method.

**H_global changes the most input bins.** It fills every missing bin with one
overall training median (≈70.2 bpm), compared with the interpolation arms' 3.87%
of missing bins. Its mean test AUROC differs from H_clock by +0.0021, while its
mean validation AUROC differs by −0.0044. Similar aggregate discrimination does
not identify where the signal resides or exclude use of recording patterns.

**Validation and test differences have opposite mean signs.** On allocation 1,
H_clock selected α=1e3 and the alternatives α=1e4, which can contribute to score
differences. This does not establish an α-selection mechanism: for example,
allocation-0 H_global validation AUROC is 0.0070 below H_clock despite both
selecting α=1e4. Fixed-α comparisons would be needed to separate this component.
Post-hygiene widths are equal across arms (19,121 / 19,121 / 19,115), although
this alone does not prove identical selected columns.

**Conclusion.** These four fitted pipelines have similar saved AUROCs on this
strict-coverage cohort. The interpolation intervention reaches only 3.87% of
missing bins; global filling reaches all missing bins. The observed mean test
range is about 0.0021, with no established advantage for a replacement fill.
This supports retaining H_clock for alignment with the recording pilot, while
leaving broader population sensitivity unquantified.


## 8. Caveats

- 3 allocations, R = 3; no bootstrap, p-values, multiplicity families or population-confidence claims (PLAN §"Readout"). Validation maxima are selected estimates; test comparisons are exploratory on this reused cohort.

- Splits are reused from R1b / the recording pilot; validation-alpha selection and test evaluation share folds — an explicit deviation authorized by this plan, matched to the recording pilot protocol.

- B40 contains per-clock-bin mask means throughout (recording features); they are fill-invariant, so any fill sensitivity is attributable to the T(HR) block alone.

- Six bins (30 min) is a fixed pilot convention, not a physiological threshold; PCHIP context is strictly local (two observed bins each side, same day, never across another gap).

- Interpolation arms change only ~4% of missing bins overall; robustness conclusions are bounded by that exposure (§5). Neither result quantifies physiological signal or excludes density, device/behaviour effects, or interactions with recording structure.

- Recorded salutation ≠ biological sex/gender.


## 9. Recommended next step

Close the fill question and stop. Per PLAN §"Stop and decide": the observed mean test range is 0.0021 across these four fitted pipelines. This is sufficient for retaining the reference convention in this pilot, without claiming statistical equivalence or predicting what additional allocations would show. In particular, *do not* change the fill convention used by the recording pilot (H_clock, training-only clock-bin medians): the small H_global test increment does not establish an advantage, and H_clock keeps this experiment byte-identically aligned with the recording pilot's H arm. Use this report as a robustness audit of that convention; the next ch3000 question is the recording-structure axis already addressed by the recording pilot's other arms, not a further fill axis.
