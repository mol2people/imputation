# Recording structure in ch3000 — pilot results

Implements [`../PLAN.md`](../PLAN.md). 8 arms × 3 allocations (0/1/2, chosen before results); cached 5-min bins; mask (M), log1p-count (C) and HR MultiRocket blocks with shuffle controls; Ridge with per-arm-per-alloc validation-alpha selection. R1b cohort/folds/first-40-adequate-days. Frozen plan commit `c970732`.


## 1. Arm table (mean ± SD over 3 allocations)

| arm | val AUROC | test AUROC | n cols | α selected (0/1/2) | boundary |
|---|---|---|---|---|---|
| `M_static` | 0.5741 ± 0.0137 | 0.5465 ± 0.0313 | 576 | 10000/10000/10000 | no |
| `M_temporal` | 0.6186 ± 0.0153 | 0.6166 ± 0.0265 | 19,391 | 100000/100000/10000 | no |
| `M_shuffled` | 0.5860 ± 0.0135 | 0.5859 ± 0.0225 | 19,382 | 100000/10000/10000 | no |
| `C_static` | 0.5745 ± 0.0184 | 0.5498 ± 0.0308 | 576 | 10000/10000/10000 | no |
| `C_temporal` | 0.6192 ± 0.0221 | 0.6332 ± 0.0283 | 19,392 | 100000/100000/10000 | no |
| `C_shuffled` | 0.5980 ± 0.0150 | 0.5976 ± 0.0145 | 19,392 | 10000/10000/10000 | no |
| `H` | 0.8311 ± 0.0161 | 0.8464 ± 0.0157 | 19,119 | 10000/1000/10000 | no |
| `H_recording` | 0.8142 ± 0.0218 | 0.8357 ± 0.0076 | 57,614 | 10000/10000/10000 | no |

## 2. Paired test deltas (3 allocations; no p-values, PLAN §"Readout")

| comparison | alloc 0 | alloc 1 | alloc 2 | mean | range | direction |
|---|---|---|---|---|---|---|
| M_temporal − M_static | +0.0836 | +0.0428 | +0.0838 | +0.0701 | [+0.0428, +0.0838] | all 3 + |
| C_temporal − C_static | +0.0954 | +0.0639 | +0.0911 | +0.0835 | [+0.0639, +0.0954] | all 3 + |
| M_temporal − M_shuffled | +0.0340 | +0.0348 | +0.0232 | +0.0307 | [+0.0232, +0.0348] | all 3 + |
| C_temporal − C_shuffled | +0.0491 | +0.0483 | +0.0095 | +0.0357 | [+0.0095, +0.0491] | all 3 + |
| C_temporal − M_temporal | +0.0162 | +0.0228 | +0.0111 | +0.0167 | [+0.0111, +0.0228] | all 3 + |
| H_recording − H | -0.0081 | +0.0034 | -0.0273 | -0.0107 | [-0.0273, +0.0034] | 1/3 + |

## 3. Paired validation deltas (3 allocations; no p-values, PLAN §"Readout")

| comparison | alloc 0 | alloc 1 | alloc 2 | mean | range | direction |
|---|---|---|---|---|---|---|
| M_temporal − M_static | +0.0419 | +0.0405 | +0.0511 | +0.0445 | [+0.0405, +0.0511] | all 3 + |
| C_temporal − C_static | +0.0422 | +0.0372 | +0.0549 | +0.0448 | [+0.0372, +0.0549] | all 3 + |
| M_temporal − M_shuffled | +0.0290 | +0.0265 | +0.0423 | +0.0326 | [+0.0265, +0.0423] | all 3 + |
| C_temporal − C_shuffled | +0.0207 | +0.0065 | +0.0366 | +0.0213 | [+0.0065, +0.0366] | all 3 + |
| C_temporal − M_temporal | +0.0059 | -0.0070 | +0.0030 | +0.0007 | [-0.0070, +0.0059] | 2/3 + |
| H_recording − H | -0.0211 | -0.0202 | -0.0096 | -0.0170 | [-0.0211, -0.0096] | all 3 − |

## 4. Recording-only absolute AUROCs (is there signal at all)

| arm | alloc 0 val | alloc 1 val | alloc 2 val | mean val | mean test |
|---|---|---|---|---|---|
| `M_static` | 0.5880 | 0.5607 | 0.5735 | 0.5741 | 0.5465 |
| `M_temporal` | 0.6300 | 0.6012 | 0.6246 | 0.6186 | 0.6166 |
| `M_shuffled` | 0.6010 | 0.5747 | 0.5824 | 0.5860 | 0.5859 |
| `C_static` | 0.5938 | 0.5570 | 0.5727 | 0.5745 | 0.5498 |
| `C_temporal` | 0.6359 | 0.5942 | 0.6276 | 0.6192 | 0.6332 |
| `C_shuffled` | 0.6152 | 0.5877 | 0.5910 | 0.5980 | 0.5976 |

## 5. Interpretation and recommended next probe

**Recording-only signal exists, and arrangement matters.** Even without HR values, the temporal MultiRocket block on mask (M) and log1p-count (C) sequences reaches val AUROC ~0.62 and test AUROC ~0.62–0.63 — well above chance and far above the static-only baseline (0.55–0.57). The signal comes from two layers:

1. **MR width over per-day distributions** (M/C_temporal − M/C_static ≈ +0.07/+0.08 test, all-3-+): a nonlinear expansion of the average clock profile plus per-day value multisets.
2. **Local arrangement beyond clock alignment and per-day multisets** (M/C_temporal − M/C_shuffled ≈ +0.031/+0.036 test, all-3-+, ≈44% of the temporal increment). The shuffled control preserves each day's count multiset and the static cross-day profile; the residual increment is what MR picks up from local stretch structure that within-day position permutation destroys. Shuffling breaks both local order *and* clock alignment together (PLAN §"Readout") — the +0.03 conflates the two and is not a causal mechanism.

**Density adds weakly over presence.** C_temporal − M_temporal test mean +0.017, all-3-+; val mean +0.001, 2/3-+ (fragile). Log1p-count carries a marginal increment over binary mask on test, not enough to claim density is the driver at n=3.

**H already contains recording texture; H_recording adds no complementarity.** H_recording − H is negative on val (all-3, mean −0.017) and on test (1/3 +, mean −0.011). The H block includes the cross-day mean mask inside B40, and the HR MultiRocket transform sees the same observed/missing pattern implicitly; the additional ~38k recording columns either duplicate that information or add shrinkage burden at the selected α. This is consistent with redundancy, not independence — recording-only models (C_temporal 0.63) sit ~0.22 below H (0.85) and likely track a weaker projection of the same source.

**Recommended next probe — HYDRA confirmation of the arrangement deltas.** The arrangement claim (+0.03 test, all-3, ≈44% of the temporal increment) rests on MultiRocket applied to binary/count sequences where the per-kernel bias quantiles are degenerate (over {0,1} for M; near-degenerate for low-count C bins). Plan option 3 (HYDRA confirmation) is the targeted test: re-run C_temporal − C_static and C_temporal − C_shuffled under HYDRA on the same three allocations (k=8, g=64, univariate, official SparseScaler). If the arrangement delta holds under a second transform on count sequences, the recording-structure finding is robust to transform choice; if it collapses, the MR-on-binary/count interaction is the likely explanation and the static+dynamic decomposition should not be over-interpreted. Extension of the C ladder to allocations 3–9 (precision rather than robustness) is the alternative for narrowing the confidence interval on C_temporal − C_shuffled; H_recording is not worth extending (no complementarity signal at the current α selection rule).


## 6. Minimal checks (PLAN §"Minimal checks")

| check | status |
|---|---|
| cohort ⊂ folds users; splits disjoint per alloc; sizes 2696/579/573 | PASS (assertions in load_data/fit_alloc) |
| mask == (cnt > 0) | PASS (load_data assertion) |
| recording arms exclude B40 and T(HR) | PASS (fit_alloc assertion) |
| no NaN/Inf at the classifier (post-hygiene) | PASS (per-block assertion) |
| train-only fitted preprocessing | PASS (hygiene_std fit on tr) |
| shuffle: occupied-bin count per day preserved | PASS (run_all) |
| shuffle: cnt multiset per day preserved (sample 200 rows) | PASS (run_all) |
| M_shuffled == (C_shuffled > 0) | PASS (run_all) |
| pooling chunk invariance (chunk 64 vs 256/direct) | PASS (fixture) |
| same-seed repeat scores within 1e-6 | PASS (fixture) |

_fixture (200 users × 40 days)_: T(M) wall 0.82s, chunk-invariance max|Δ| 0.0e+00, repeat max|Δ| 0.0e+00; T(C) wall 0.89s.


## 7. Caveats

- Pilot, 3 allocations; no bootstrap, p-values, multiplicity families or population-confidence claims (PLAN §"Readout").

- Both splits are reused from prior experiments (R1b, temporal). Validation alpha selection and test evaluation on the same folds is an explicit deviation from the temporal plan's retirement statement, authorized by this plan.

- MR bias quantiles on binary M input are degenerate (over {0,1}); the transform still runs, and the shuffled control isolates arrangement from the preserved value distribution.

- Pooling: M/C/M_shuf/C_shuf pool all 40 days unconditionally; T(HR) keeps the temporal mask day-gate. Block-local, intentional.

- Shuffling breaks local order AND clock alignment; it does not separate their contributions (PLAN §"Readout").

- Ridge dual solve emitted LinAlgWarning (ill-conditioned kernel, rcond ~1e-8) and a singular-matrix least-squares fallback at the smallest grid alphas (1e-3, 1) during selection — expected for n=2,696 < p (up to 57,617). Every **selected** alpha is ≥1e3 (§1), where the solve is well-regularized; warnings attach to unselected grid points only (log: `cache/run_stdout.log`).

- Recorded salutation ≠ biological sex/gender.
