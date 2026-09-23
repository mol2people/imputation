# Side-quest report: strict-coverage cohorts and per-source RF benchmarks

Generated 2026-09-19 from `src/sidequest/`; all outputs under `experiments/artifacts_sq/`. Verification: **135/135 checks passed** (`verification_sq.json`; re-derives eligibility, selection, windows, splits, features, schemas, predictions, and the bootstrap from raw inputs).

Plan-status note: the 2026-09-19 decision made plan section 1 gate semantics authoritative; the count rows in section 2 and the Apple/Samsung rows of section 4 are superseded. Corrected anchors live in `src/sidequest/sq_config.py`. One frozen-table correction was required: source 7 train sal10 is **167**, not 166 (166+266 != 433 and 166+21+21 != 209).

## 1. Cohorts (strict-coverage, any-core)

| source | base v2-eligible single-source | ch3000 | ch3001 | ch3002 | any-core pass | sal10 | sal20 | disposition |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 4,410 | 3,847 | 2,148 | 3,181 | 3,848 | 1,384 | 2,464 | model |
| 6 | 11,143 | 28 | 105 | 4,367 | 4,446 | 1,373 | 3,073 | model |
| 7 | 1,498 | 1 | 0 | 540 | 541 | 209 | 332 | model |
| 9 | 396 | 0 | 0 | 9 | 9 | 4 | 5 | audit only |
| 13 | 543 | 19 | 1 | 26 | 41 | 5 | 36 | audit only |

Any-core union **8,885**; model union (sources 3/6/7) **8,835**. Any-core membership is exactly the set union of independent per-channel passers (verified). Sources 9 and 13 fail the minimum class-count rule (any-core 9 and 41) and are audit-only.

## 2. Coverage semantics audit (frozen cov_s artifact semantics)

| source | channel | day rows | days cov_s>24h | share | max cov_s (s) |
|---|---|---:|---:|---:|---:|
| 3 | 3000 | 1,780,918 | 6,160 | 0.346% | 166245 |
| 3 | 3001 | 1,550,523 | 12 | 0.001% | 121143 |
| 3 | 3002 | 1,550,588 | 0 | 0.000% | 0 |
| 6 | 3000 | 3,105,475 | 4 | 0.000% | 118789 |
| 6 | 3001 | 2,953,966 | 1,017 | 0.034% | 172539 |
| 6 | 3002 | 2,667,200 | 0 | 0.000% | 0 |
| 7 | 3000 | 386,638 | 0 | 0.000% | 0 |
| 7 | 3001 | 295,151 | 0 | 0.000% | 0 |
| 7 | 3002 | 295,149 | 0 | 0.000% | 0 |
| 9 | 3000 | 111,375 | 1 | 0.001% | 87183 |
| 9 | 3001 | 78,736 | 0 | 0.000% | 0 |
| 9 | 3002 | 78,736 | 0 | 0.000% | 0 |
| 13 | 3000 | 87,952 | 397 | 0.451% | 162240 |
| 13 | 3001 | 66,086 | 181 | 0.274% | 156444 |
| 13 | 3002 | 66,066 | 0 | 0.000% | 0 |

`cov_s` is the frozen v2 artifact quantity: the union of epoch intervals attributed to the interval-start local date, without midnight clipping, so values above 86,400 s are possible. It is treated as audit-only and is never reinterpreted.

## 3. Participant allocation (frozen 80/10/10, per source)

Targets: split totals `T = largest_remainder(FRACS, N)` and class-1 (sal20) apportioned by the same rule with ties to the lowest split index; sal10 = T - sal20. This reproduces the untouched Garmin row and the recorded Apple row exactly. Floor/ceiling proportional marginals for age group, BMI group, and qualifying-channel pass-pattern were requested; the two-stage smallest-violation MILP found total violation **0** (no nonzero bound violations; relaxation machinery recorded in `split_violations_sq.csv`).

**Source 3** (N = 3,848):

| split | total (target) | sal10 (target) | sal20 (target) |
|---|---:|---:|---:|
| train | 3,078 (3,078) | 1,107 (1,107) | 1,971 (1,971) |
| val | 385 (385) | 138 (138) | 247 (247) |
| test | 385 (385) | 139 (139) | 246 (246) |

Max within-marginal proportion spread across splits: age_group 0.12; bmi_grp 0.26; selected_channel 0.26; pass_pattern 0.26 (percentage points; joint y x age x BMI cells minimised as the MILP secondary objective). Per-level tables: `source_3/split_balance_report.md`.

**Source 6** (N = 4,446):

| split | total (target) | sal10 (target) | sal20 (target) |
|---|---:|---:|---:|
| train | 3,557 (3,557) | 1,098 (1,098) | 2,459 (2,459) |
| val | 445 (445) | 138 (138) | 307 (307) |
| test | 444 (444) | 137 (137) | 307 (307) |

Max within-marginal proportion spread across splits: age_group 0.19; bmi_grp 0.23; selected_channel 0.22; pass_pattern 0.23 (percentage points; joint y x age x BMI cells minimised as the MILP secondary objective). Per-level tables: `source_6/split_balance_report.md`.

**Source 7** (N = 541):

| split | total (target) | sal10 (target) | sal20 (target) |
|---|---:|---:|---:|
| train | 433 (433) | 167 (167) | 266 (266) |
| val | 54 (54) | 21 (21) | 33 (33) |
| test | 54 (54) | 21 (21) | 33 (33) |

Max within-marginal proportion spread across splits: age_group 1.89; bmi_grp 2.08; selected_channel 0.23; pass_pattern 0.23 (percentage points; joint y x age x BMI cells minimised as the MILP secondary objective). Per-level tables: `source_7/split_balance_report.md`.

## 4. Feature blocks

| block | prefix | width |
|---|---|---:|
| D demographics | `demo__` | 2 |
| A recording-quality (epoch span) | `rec__` | 78 |
| B 91-day window summaries | `win__` | 78 |
| C rolling 4/7/15/30-day, recording span | `roll_rec__` | 120 |
| C rolling 4/7/15/30-day, window + pre-lookback | `roll_win__` | 120 |
| all | - | 398 |

Per-block widths asserted at build time and re-verified; one row per model-union participant in every block; prefixes disjoint; demographics appear only in D and exactly once per design matrix; `feature_dictionary_sq.csv` documents all 398 columns.

Rolling-value interpretation (the one interpretive choice flagged for review): the plan defines rolling-block *definedness* only; the rolling value is implemented as the **trailing-window mean of the non-missing daily series values** (mean/median/SD/hours series, missing on days failing the v2 feature-day rule), with windows truncated at the grid start kept valid. This matches the plan's pre-window lookback convention.

## 5. Models and bootstrap (exploratory)

RandomForestClassifier(`random_state=20260918`, all other hyperparameters at the installed scikit-learn default), new preprocessing + forest per (source, variant); training-fitted preprocessing: all-missing column drop, median imputation with missingness indicators, demographic mode imputation + one-hot with unknown-level handling, zero-variance term drop; no scaling. Bootstrap: exactly 100 within-class test resamples per source, SeedSequence([20260918, source_id]), identical resamples across variants, recomputed from saved predictions.

### Source 3

| variant | val AUROC | test AUROC | boot mean | boot SD | 95% percentile (exploratory) | test bal. acc. |
|---|---:|---:|---:|---|---:|---:|
| demo | 0.5590 | 0.5743 | 0.5730 | 0.0271 | [0.5236, 0.6184] | 0.5067 |
| rec | 0.6904 | 0.6568 | 0.6538 | 0.0315 | [0.5970, 0.7242] | 0.6176 |
| win | 0.7093 | 0.6918 | 0.6883 | 0.0310 | [0.6328, 0.7543] | 0.6391 |
| roll_rec | 0.6342 | 0.5928 | 0.5889 | 0.0317 | [0.5367, 0.6547] | 0.5423 |
| roll_win | 0.6658 | 0.6445 | 0.6407 | 0.0321 | [0.5819, 0.7089] | 0.5995 |
| all | 0.6934 | 0.6318 | 0.6291 | 0.0311 | [0.5808, 0.6997] | 0.6047 |

Paired test-AUROC deltas (draw-level, same resamples):

| comparison | point | draw mean | draw SD |
|---|---:|---:|---:|
| recording vs demographics (`rec-demo`) | +0.0825 | +0.0808 | 0.0407 |
| window vs recording (`win-rec`) | +0.0350 | +0.0345 | 0.0195 |
| rolling window vs rolling recording (`roll_win-roll_rec`) | +0.0518 | +0.0518 | 0.0241 |
| all blocks vs recording (`all-rec`) | -0.0249 | -0.0247 | 0.0154 |

### Source 6

| variant | val AUROC | test AUROC | boot mean | boot SD | 95% percentile (exploratory) | test bal. acc. |
|---|---:|---:|---:|---|---:|---:|
| demo | 0.5637 | 0.5618 | 0.5659 | 0.0257 | [0.5130, 0.6155] | 0.5057 |
| rec | 0.7056 | 0.7224 | 0.7242 | 0.0279 | [0.6750, 0.7749] | 0.5405 |
| win | 0.6455 | 0.6814 | 0.6814 | 0.0274 | [0.6364, 0.7344] | 0.5474 |
| roll_rec | 0.7062 | 0.7012 | 0.7029 | 0.0294 | [0.6512, 0.7623] | 0.5729 |
| roll_win | 0.6146 | 0.6503 | 0.6557 | 0.0296 | [0.5949, 0.7093] | 0.5360 |
| all | 0.7012 | 0.7184 | 0.7193 | 0.0277 | [0.6755, 0.7717] | 0.5815 |

Paired test-AUROC deltas (draw-level, same resamples):

| comparison | point | draw mean | draw SD |
|---|---:|---:|---:|
| recording vs demographics (`rec-demo`) | +0.1606 | +0.1583 | 0.0362 |
| window vs recording (`win-rec`) | -0.0410 | -0.0428 | 0.0277 |
| rolling window vs rolling recording (`roll_win-roll_rec`) | -0.0509 | -0.0473 | 0.0276 |
| all blocks vs recording (`all-rec`) | -0.0040 | -0.0049 | 0.0254 |

### Source 7

| variant | val AUROC | test AUROC | boot mean | boot SD | 95% percentile (exploratory) | test bal. acc. |
|---|---:|---:|---:|---|---:|---:|
| demo | 0.5902 | 0.5750 | 0.5653 | 0.0635 | [0.4422, 0.6881] | 0.5000 |
| rec | 0.7439 | 0.6638 | 0.6580 | 0.0803 | [0.5036, 0.7810] | 0.5736 |
| win | 0.7807 | 0.6609 | 0.6493 | 0.0901 | [0.4838, 0.8190] | 0.5649 |
| roll_rec | 0.4812 | 0.6392 | 0.6468 | 0.0818 | [0.4998, 0.7875] | 0.5671 |
| roll_win | 0.6263 | 0.6061 | 0.6133 | 0.0762 | [0.4652, 0.7620] | 0.4957 |
| all | 0.7186 | 0.5880 | 0.5845 | 0.0857 | [0.4426, 0.7360] | 0.5108 |

Paired test-AUROC deltas (draw-level, same resamples):

| comparison | point | draw mean | draw SD |
|---|---:|---:|---:|
| recording vs demographics (`rec-demo`) | +0.0887 | +0.0927 | 0.1004 |
| window vs recording (`win-rec`) | -0.0029 | -0.0087 | 0.0865 |
| rolling window vs rolling recording (`roll_win-roll_rec`) | -0.0332 | -0.0335 | 0.0763 |
| all blocks vs recording (`all-rec`) | -0.0758 | -0.0735 | 0.0649 |

### Reading

- Demographics alone sit near chance in every source (test AUROC 0.55-0.59); recording-quality features carry the signal, consistent with the v2 pooled finding (pooled test AUROC 0.68333, 10-draw SD 0.00873, as background context only).
- Per-source point estimates are exploratory: the plan pre-specifies no test-set winner selection, no vendor ranking, and no expected AUROC; the 100-draw percentile intervals are coarse. Samsung's test set has n = 54 (21/33) - its intervals span roughly 0.28 in AUROC and must not be read as rankings.
- The identical participant split is reused for every variant within a source, so paired deltas are the comparable quantity; they are recomputed from saved predictions and re-verified.

## 6. Measured runtime and provenance

| stage | wall time | note |
|---|---:|---|
| raw-file manifest + parser benchmark | ~752 s (aggregate single-worker) | 186.2 GiB across 8,835 files; all present, 0 parse failures |
| window mini-scan (8 workers) | 177 s | 625,743 hour rows, 0 failures |
| feature blocks | ~24 s | rolling over 26,500 user/channel series |
| allocation (3 sources, 2-stage MILP) | <1 s | 0 bound violations |
| 18 RF fits + 300 bootstrap draws | ~13 s | sklearn 1.9.0 |
| 18 RF fits + 300 bootstrap draws | ~13 s | sklearn 1.9.0 |
| tuned-500 extension (48-config × 5-fold × 500-tree CV, 18 cells) | ≈1.5–2 h aggregate across three launches | two crash-resumes; not precisely metered; log `tune_run.log` |
| verification contract | ~27 s | 135 checks, 0 failures |

Full provenance (config echo, package versions, SHA-256 of scripts and inputs): `reproducibility_sq.json`. Default-vs-tuned side-by-side with paired bootstrap deltas: `tuned_500_comparison.md`. Phase archives: `plans/sidequest_strict_coverage_frozen_split_2026-09-19.md`, `plans/sidequest_feat_sel_repeated_splits_2026-09-19.md`.

## 7. Tuned-500 extension (added 2026-09-19)

Protocol: grid of **48 configs** — `max_features` ∈ {sqrt, 0.2, 0.4} × `min_samples_leaf` ∈ {1, 2, 5, 10} × `max_depth` ∈ {None, 12} × `class_weight` ∈ {None, `balanced_subsample`} (`min_samples_split` dropped as dominated by leaf size). Selection by **pooled 5-fold OOF AUROC** (StratifiedKFold, shuffle, seed 20260918) on the **train+val pool** (3,463 / 4,002 / 487 per source); preprocessing refit per fold; ties → first grid order. Final fit: best config × `n_estimators=500` on the full pool, one test prediction, **identical 100-draw bootstrap seed stream** as the default run (draw-level cross-run comparisons valid). Test fold untouched by tuning. Full per-source tables: `tuned_500_comparison.md`; artifacts `source_<s>/tuned_500/`.

### 7.1 Test AUROC delta, tuned-500 vs default-100 (frozen test fold)

| source | demo | rec | win | roll_rec | roll_win | all |
|---|---:|---:|---:|---:|---:|---:|
| 3 | +0.0000 | +0.0414 | +0.0361 | +0.0484 | +0.0273 | +0.0750 |
| 6 | +0.0001 | +0.0327 | +0.0657 | +0.0191 | +0.0307 | +0.0530 |
| 7 | +0.0245 | +0.0058 | **−0.0267** | −0.0065 | +0.0491 | +0.0916 |

Tuned levels: s3 demo 0.5743 / rec 0.6981 / win 0.7279 / roll_rec 0.6412 / roll_win 0.6719 / all 0.7068; s6 0.5619 / 0.7551 / 0.7472 / 0.7203 / 0.6809 / **0.7714**; s7 0.5996 / 0.6696 / 0.6342 / 0.6328 / 0.6551 / 0.6797.

### 7.2 Hyperparameter landscape (8 surviving full 48-config surfaces)

- **`min_samples_leaf` is the only monotone main effect**: mean CV AUROC rises 1→10 in every surface (+0.005 to +0.014; one flat exception, s7 `rec`). `max_features=0.4` adds ~+0.005 on wide blocks (wash on narrow); `class_weight=balanced_subsample` adds +0.002–0.006 on average; **`max_depth` is dead** (≤0.002 mean effect, sign flips; rarely binds once leaf ≥ 5); `n_estimators` 100→500 is variance reduction only.
- **A single global config G1** (`max_features=0.4`, `min_samples_leaf=10`, `max_depth=None`, `class_weight=balanced_subsample`) sits within **≤0.006 CV AUROC** of every cell's searched optimum (mean gap ≈ 0.003; s3 `all` +0.0002, s6 `all` +0.0000). Per-cell tuning buys ~nothing beyond a good fixed config.
- **Winner's curse quantified**: best−median per surface is 0.006–0.011 vs per-config OOF SE ≈ 0.008 at pool n ≈ 3.5–4k. **Samsung is a noise floor** (pool 487 → SE ≈ 0.03): the entire s7 `demo` CV surface sits at 0.459–0.474 (below chance); s7 `win`'s CV winner *is* the default config and still regresses −0.0267 on test; chosen-config scatter across s7 cells is noise, not signal.
- The default sklearn config ranks **46–48/48** in every full surface → the tuned gains are almost entirely **regularization** (leaf, then class weighting), not tree count.

### 7.3 Interpretation (agreed framing)

- **Honest primary claim: the CV AUROC gain of +0.015–0.020** over defaults (pre-registered grid, clean fold structure, no test involvement).
- **Test deltas are corroborative/descriptive only**: they are a post-default second look on one frozen fold. Samsung's mixed signs at n = 54 (test SE ≈ 0.07) — `all` +0.0916 alongside `win` −0.0267 — are the same noise mechanism, not signal.
- `all` gains most from tuning: wide, diluted feature spaces are where defaults hurt, and leaf regularization recovers the C-block signal (s6 `all` 0.7714 is now the best Apple cell, above `rec` 0.7551).

## 8. Feature-space diagnostics (read-only; motivate the FS phase)

From the fitted s3 `all` tuned pipeline and train pool:

- **Missingness indicators are dead weight**: 131 of 534 transformed terms are `{col}__missing`; their **summed Gini importance is 0.000** (best rank ≈ 400/534). Mechanism: the strict-coverage gate makes missingness rare by construction, and `min_samples_leaf=10` renders features separating < 10 rows unusable. No block exceeds 50% missing in the train pool (median 0%) — the cohorts have data by design.
- **Near-zero-variance design artifacts**: the caret nzv rule (dominant value > 95% of rows AND unique-value fraction < 10%) flags **137/534** columns — `win__ch3000_observed_days`, `win__overall_span_days`, near-constant indicators, `roll_rec__ch3000_daily_hours_w7_max`. Window-completeness features are near-constant *by eligibility*, i.e. design artifacts, not signal.
- **Signal structure is wear-pattern, not physiology**: top Gini importances are `rec__ch3000_weekday_mean` (0.045), `rec__ch3000_tod_morning` (0.035), the `ch3000−ch3002` contrast (0.018), weekend−weekday (0.015). Rolling features are heavily redundant across window widths (w7/w15/w30 of the same series).

## 9. Feature selection under repeated splits (completed 2026-09-19)

Spec: `plans/sidequest_feat_sel_repeated_splits_2026-09-19.md` (repo copy; the frozen-phase archive is `plans/sidequest_strict_coverage_frozen_split_2026-09-19.md`). Design, per user rulings:

- **Repeated stratified splits replace the frozen single-look test**: R = 30 train/val/test (70/15/15) resamples per source, no allocation matching. Every arm sees every repeat → the paired Δ(arm − A0) over repeats is the primary estimand and the multiplicity tangle dissolves. No per-fit bootstrap.
- **Uniform config G1 at 100 trees** (no per-cell tuning; per-cell tuning shown to buy ≤ 0.006 CV AUROC, below selection noise). 100 vs 500 trees verified on our own data at G1: AUROC deltas ±0.005 with flipped signs across sources, importance ρ ≈ 0.98, top-20 overlap 17/20, 4–5× cheaper; per-cell RF seed shared across arms makes paired deltas deterministic given the split.
- **Arms**: A0 baseline / A1 hygiene (drop all `__missing` indicators + caret-nzv) / A2 = A1 + |ρ| > 0.95 dedup / A3 top-k ∈ {50, 100} by permutation importance **on that repeat's validation fold** (val-based selection per repeat replaces nested CV entirely; test used once per arm-repeat).
- Driver `src/sidequest/feat_sel.py` (resumable per source-repeat); results in `experiments/artifacts_sq/fsplit/`, comparison in `fsplit/feat_sel_report.md`. Feature engineering is a follow-on phase spec'd after these results (same split seeds → paired across phases).

### 9.1 Outcome (R = 30; full tables in `fsplit/feat_sel_report.md`)

- **Feature selection buys essentially no AUROC.** A1 (hygiene) is AUROC-neutral in **18/18** source × variant cells (|mean Δ| ≤ 0.007, every CI straddles zero) while halving the matrices (s6 `all` 736 → 401 columns; 335 dropped, ~330 of them indicators). A2 (dedup) is **source-inconsistent**: the only significant gains are Garmin `rec` +0.0050 [+0.0030, +0.0070] and `win` +0.0035 [+0.0017, +0.0053], while it is significantly *negative* for Apple `roll_rec`/`roll_win` (−0.0034/−0.0083) and Samsung `roll_win`/`all` (−0.0200/−0.0124) — no arm is uniformly non-negative.
- **Permutation-importance top-k is unstable and sometimes harmful**: top-50 sets share only Jaccard 0.14–0.21 (`all`) across splits, and A3 is significantly negative in 2 cells (s6 `all` k50 −0.0081, s7 `rec` k50 −0.0110), positive in 1 (s3 `rec` +0.0040). Many near-equivalent subsets exist; which exact 50 you get is split luck.
- **Retroactive context for every single-look number above**: for the two frozen tuned cells whose selected config *is* G1, the frozen test draw sits **−2.7 SD** (s3 `rec` 0.6981 vs resample mean 0.7465 ± 0.0177) and **+2.3 SD** (s6 `all` 0.7714 vs 0.7188 ± 0.0226) from its source's resampling distribution — the frozen-fold numbers quoted in §5–§7 carry real draw luck in both directions, which the repeated-splits design removes. (Frozen runs trained on 90% of data vs 70% here and 500 vs 100 trees — both favor the frozen side, making the gaps conservative.)
- **Adopted defaults**: drop the `__missing` indicators (free, AUROC-neutral, ~45% smaller matrices); do **not** adopt correlation dedup or top-k as pipeline defaults. Feature engineering (phase 2, same split seeds) is the remaining lever, with expectations tempered accordingly.

### 9.2 Primary-family readout (adopting the Codex proposal's inference discipline)

`plans/CODEX_feature_selection_repeated_holdouts_2026-09-19.md` (alternative spec of the same design, not executed) prescribes a stricter inference discipline; adopted here retroactively for reporting. Its **primary family** — the only winner-capable comparisons — is source 3 `all` and source 6 `all`, arm `A2 − A0`, with two-sided **97.5% t intervals** giving a Bonferroni-simultaneous 95% family:

| cell | mean paired Δ | SD | MC SE (SD/√30) | median | IQR | share > 0 | 97.5% t CI |
|---|---:|---:|---:|---:|---|---:|---|
| s3 `all` A2−A0 | +0.0023 | 0.0083 | 0.0015 | +0.0026 | [−0.0041, +0.0092] | 0.57 | [−0.0008, +0.0054] |
| s6 `all` A2−A0 | +0.0021 | 0.0076 | 0.0014 | +0.0018 | [−0.0035, +0.0058] | 0.57 | [−0.0007, +0.0050] |

**Conclusion under the prespecified primary family: no detectable feature-selection effect** — both intervals straddle zero (MC SE ≈ 0.0015; interval half-width ≈ 0.003, so the design would have resolved effects of ~0.003 AUROC or larger). Every other cell in §9.1 — including the nominally significant s3 `rec`/`win` dedup gains and the roll-block negatives — is **secondary/exploratory** and is not used to declare winners; source-specific results are preserved and never pooled or ranked.

**Estimand and interpretation limits** (proposal §1/§13, adopted in substance): the intervals quantify the prespecified split/model randomization conditional on the observed strict-coverage cohort and the already-constructed outcome-blind feature tables. They do **not** capture participant-sampling, transportability, or analyst-adaptivity uncertainty. The target is recorded salutation, not biological sex or gender identity; strict coverage selects sustained recorders, and source-specific channels may carry vendor processing or acquisition-pattern information. Feature selection improves predictive efficiency within this selected export only — it cannot isolate physiology, establish causality, or support vendor comparisons. This dataset and its original frozen split have already informed analysis choices; all results from this phase are exploratory algorithm-development evidence.

**Feature-engineering guard (phase 2, proposal §12)**: the FE candidate families were pre-registered in the phase plan §10 *before* this phase's test summaries were produced; any FE choice additionally informed by this phase's results is labeled **adaptive and exploratory**. Reusing the same test partitions supports descriptive paired comparisons but never independent validation; confirmatory evidence requires new participants or a genuinely external dataset.


## 10. Feature engineering under repeated splits (completed 2026-09-19)

Phase spec pre-registered in `experiments/artifacts_sq/plans/sidequest_feature_engineering_repeated_splits_2026-09-19.md` (frozen before any FE evaluation; one arithmetic correction — E3 = 18 columns, not 36 — made after a structural smoke run, before any test summary was inspected). Driver: `src/sidequest/feat_eng.py`; artifacts: `experiments/artifacts_sq/feng/` (90/90 cells, 2,700 rows, 12.7 min, log `feng_run.log`); full tables: `experiments/artifacts_sq/feng/feat_eng_report.md`. The phase is **adaptive relative to the cohort's analysis history and pre-registered relative to the FS phase's test summaries**; it is never independent validation (guard above).

**Design.** Identical splits and RF seeds as the FS phase (verified: arm **B0** = hygiene on the original matrix reproduces FS arm A1 with max |ΔAUROC| 5.6e-17 across all 270 cells — the two phases are bit-compatible, so all FE deltas are paired against the adopted baseline). Families (54 columns, added as raw columns before the standard preprocessor): **E1** wear-pattern interactions `weekday_mean × tod_b` (24), **E2** channel ratios `chA/chB`, denominator > 0 guarded (12), **E3** drift deltas `w30_mean − w7_mean` (18). Arms: B0 / **F1** (+ all applicable families) / **F2** (= F1 + \|ρ\|>0.95 dedup, **originals-first** so engineered terms can be evicted but never evict their parents) / **F1_E{1,2,3}** single-family attribution arms (no-ops for `demo`).

**Primary family** (s3/s6 `all`; F1−B0 and F2−B0; two-sided 97.5% t, Bonferroni-simultaneous 95% over the four comparisons):

| comparison | mean Δ | 97.5% t CI | share>0 |
|---|---:|---|---:|
| s3 `all` F1−B0 | +0.0009 | [−0.0014, +0.0032] | 0.63 |
| s3 `all` F2−B0 | +0.0009 | [−0.0015, +0.0032] | 0.67 |
| s6 `all` F1−B0 | +0.0028 | [−0.0003, +0.0059] | 0.63 |
| s6 `all` F2−B0 | +0.0034 | [−0.0000, +0.0068] | 0.60 |

**No detectable feature-engineering effect in the primary family**: all four intervals contain 0; the s6 point estimates (+0.0028/+0.0034, MC SE ≈ 0.0015) kiss the zero boundary and do not survive the multiplicity-controlled threshold. Detection floor remains ≈ 0.003 AUROC at this split count.

**Secondary readout** (exploratory):
- **Family attribution**: E1 interactions are informationally void — dedup evicts 87–95% of them in every cell (near-duplicates of their parent level/share columns), and F1_E1−B0 is ≤ +0.0005 in all nine cells where E1 applies (`rec`/`win`/`all` × 3 sources; significantly *negative* in s3 `win` −0.0025 and s6 `rec` −0.0032). E2 ratios are largely evicted on s6 (83%) but survive partly on s3/s7; F1_E2−B0 is significant only in s3 `rec` (+0.0034 [+0.0016, +0.0052]).
- **E3 drift deltas are the only family carrying genuinely new information — and it hurts where it survives.** They survive dedup fully on s6/s7 (0/270 and 0/270 evictions) but are significantly negative on s7 (`roll_rec` F1−B0 −0.0175 [−0.0305, −0.0045]; `roll_win` F2−B0 −0.0279 [−0.0457, −0.0101]; `all` F1−B0 −0.0103 [−0.0201, −0.0004]) and inconsistent on s6 (`roll_rec` +0.0041 [+0.0006, +0.0075] vs `roll_win` F2 −0.0076 [−0.0118, −0.0034]). The one significant positive (s6 `roll_rec` E3-only) does not transport across the parallel block or the neighboring source — the same source-inconsistency pattern that disqualified FS A2.
- **F2 vs F1**: wherever F2 moves, it reproduces the known dedup effects rather than revealing an engineered-feature benefit (s3 `rec` F2−B0 +0.0054 [+0.0035, +0.0074] ≈ FS A2−A0 +0.0050 [+0.0030, +0.0070] on the same splits; cross-repeat corr of the paired deltas is only 0.29 because the baselines differ — A0 keeps indicators, B0 drops them — but the location agrees). Original-vs-original pruning inside F2 removes 17–206 originals per repeat×cell (e.g., s3 `all` ≈ 206/repeat), i.e., F2 ⊃ A2 semantics as designed.

**Outcome ruling.** Feature engineering is **not adopted**: the pre-registered primary family shows no effect, the only novel family (E3) is source-inconsistent with a negative Samsung signature, and E1/E2 are redundant with existing columns by construction and by measurement. The A1 preprocessing default stands unchanged; combined with §7 (hyperparameters solved at G1 @ 100 trees) and §9 (feature selection neutral at the adopted resolution), the model-side levers for this cohort are exhausted at the current noise floor: remaining AUROC movement would require new signal sources (external features, participants, or labels), not rearrangements of the existing ones.
