# R1a-AF — circadian-curve features under demographically-balanced CP-SAT folds

Status: exploratory follow-up to R1a (commit `4868482`, pushed). Exploratory; reuses the
R1a feature cache + model code by importing the committed R1a runner.

## 1. Question

Does the R1a circadian-curve increment replicate when train/val/test folds are
demographically-balanced CP-SAT allocations (via the user's `cohort_allocator` package)
rather than the frozen FS-phase label-stratified random splits?

- If the increment rides on a demographic mismatch between the R1a train and test folds,
  it should shrink (or vanish) under allocator folds.
- If it survives demographically-matched folds, the R1a claim is strengthened.
- Descriptive side-comparison: R1a BASE absolute AUROC on the allocator test folds vs
  the R1a-frozen-split value (0.7434). Different test users across split systems — not a
  paired test; CI-overlap descriptive only.

## 2. User rulings (carry-overs)

- All experiment code in its own folder (`experiments/r1a_allocfolds_garmin_2026-09-21/`).
  No raw time-series (`out/`) committed.
- 5-worker cap (libreralised this run to N_JOBS=4 user grant 2026-09-21, alongside the
  active dayscale run).
- ch3000-only convention; recorded salutation label (not biological sex/gender).
- **Do not modify the `cohort_allocator` package.** Work around its limits via coarser
  stratify bands.
- Coarser age bands; **70+ merged into a single band** (`age_band_10y_with_70plus`).
- Vendor-circularity audit declined; do not propose or run.
- This run: **R = 10** (user ruling 2026-09-21; sanity-check scope).

## 3. Cohort & inputs

- s3 strict cohort, n = 3,848 (2,464 y=1 / 1,384 y=0), identical to R1a.
- Demographics built deterministically in-runner (cached to `cache/demographics.parquet`):
  - `user_id`, `salutation` (string), `birth_year` (numeric, from
    `13Aug_1222.csv::birth_date` year-only), `age` (years, **Dec-1 cutoff convention**
    from the age experiment: `age = ws.year − birth_year − 1 + [ws.month ≥ 12]`,
    `window_start` interpreted as **days since epoch** via
    `pd.to_timedelta(window_start.astype(int), "D")`).
  - `age_band` ∈ {`20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70+`} (10-year bands with
    70+ merged).
  - `bmi_group` ← `features_all.demo__bmi_grp` cast to `string` (single **categorical**
    column, not one-hot; invert via `.astype('string')`, NOT `idxmax`).
- Frozen FS artefacts: `artifacts_sq/{split_manifest_sq,features_all}.parquet` (s3 only).
- R1a feature cache (READ-ONLY): `experiments/r1a_circadian_garmin_2026-09-21/cache/
  r1a_features_epoch_hours.parquet` (sha-verified into repro); rebuilt by importing the
  R1a script's compute path if missing (fail-fast message points to R1a experiment).

## 4. Allocator spec (frozen)

```
partition_columns = ('age_band', 'bmi_group')   # 23 partitions, ≤ 601 users
stratify_columns  = ('salutation',)             # 2 joint cells per partition
fold_sizes        = {'train': 70, 'val': 15, 'test': 15}
seed              = derived per allocation       # see §6
time_limit_seconds = 300                        # bench worst 3.8 s; safety margin
```

Decomposition math (validated in bench, 5/5 seeds OPTIMAL, 1.7–5.7 s): with
`partition_columns = (age_band, bmi_group)`, the allocator solves independently within
each `(age_band, bmi_group)` cell. Within each partition, only `salutation` is
stratified — so the marginal phase is *skipped* (single stratify column) and only
joint + tie-break phases run, each on ≤ 601 users with 2 joint cells. Each phase solves
to OPTIMAL trivially.

Balance equivalence to the full three-way `(salutation × age_band × bmi_group)` J-spec:
each `(age, bmi)` partition receives an exact proportional slice in every fold, so age
and bmi marginals are balanced by construction; salutation × bmi × age cells are
balanced within each partition. Summed across partitions, the fold composition matches
the J-spec balance up to integer-rounding per cell.

Benched worst-cell deviation (intentional integer granularity, structural and identical
across seeds): cell `(age_band=20-29, salutation=10)` carries the largest per-cell
proportion deviation (~0.39; ~30 users → train fold gets ~22 by largest-remainder). All
aggregate composition is essentially perfect — mean age 50.33/50.42/50.43 across folds,
bmi shares identical to 3 dp, salutation ≈ 35.7%/64.3% in every fold. Documented in
`cache/balance_run{0..9}.csv` per allocation.

## 5. Protocol per allocation

Per allocation r ∈ 0..R-1:

1. Build the demographics table (§3) — cached once.
2. Run the allocator with `seed_alloc(r)` to obtain `assignments` (user_id → fold).
3. **Gate:** assert `all_phases_optimal == True` for every partition
   (fail-loud; the bench shows this holds at this spec for all 5 pilot seeds). Record
   `alloc_summary_run{r}.json` with phase results per partition, total wall time, and
   `ortools` version.
4. Map assignments → integer fold index arrays aligned to the R1a-sorted user order
   (assert `set(assignments.user_id) == set(users_sorted)`).
5. Load the R1a feature cache; residualise `r1a_full` against the allocation's train
   fold to obtain `r1a_resid_full` (reuses `run_r1a.residualise`).
6. Fit the five R1a arms (BASE / R1a_only / BASE⊕R1a / R1a_resid_only /
   BASE⊕R1a_resid) on the allocation's train/val/test indices using
   `run_r1a.apply_arm_transform` + `run_r1a.rf_seed(r)` (= `SeedSequence([20260919, 3,
   r, 5])`, identical stream to R1a's first R repeats — **same RF seed, different
   folds**).
7. Record per arm: `auroc_val`, `auroc_test`, `n_features_after_A1`, Gini for the
   composite arms. Store test predictions for BASE / BASE⊕R1a / BASE⊕R1a_resid only
   (small; enables future paired analyses).
8. Persist `cache/balance_run{r}.csv` (fold × age × bmi × sal cross-tab) and the
   allocator summary.

## 6. Inference

- Primary family: 2 paired deltas (`BASE⊕R1a − BASE` and `BASE⊕R1a_resid − BASE`)
  per allocation; mean ± 95% t-CI over R = 10 allocations (program convention:
  `t.ppf(0.975, 9)·sd/√10`, matching `feat_sel._mean_ci`).
- Secondary: Bonferroni-simultaneous 97.5% t-CI for the 2-comparison family
  (`t.ppf(0.9875, 9)`) — honest family-wise inference, not the program default.
- Secondary: paired-within-allocation deltas for the 3 un-composited comparisons
  (R1a_only − BASE, R1a_resid_only − BASE, BASE⊕R1a_resid − BASE⊕R1a) and the
  R1a_only vs R1a_resid_only activity-vs-composition mediation read.
- Sanity: allocator BASE mean over R = 10 vs R1a's frozen-split value 0.7434
  (descriptive CI-overlap; different test users across split systems → not a paired
  test).
- Share > 0 (sign stability) for each paired delta.

## 7. Gates

1. Allocator internal asserts (exact fold sizes, complete assignment) — raises on
   violation.
2. `all_phases_optimal == True` per allocation (asserted; fail-loud — bench shows
   this holds at the chosen spec).
3. Fold sizes across allocations: largest-remainder per partition yields per-allocation
   totals that can vary by ±1–2 users (documented in §4); reported in the balance
   files, not a fail criterion.
4. BASE sanity: allocator BASE mean vs 0.7434 (descriptive; not a fail criterion — the
   test users differ).

The R1a frozen-split bit-exact `BASE` gate no longer applies: test users differ across
split systems. Rigour transfer: FS splits → allocator determinism (single CP-SAT worker
+ fixed seed + version) + per-allocation `all_phases_optimal == True`.

## 8. Compute

- 10 allocations × ~2–6 s ≈ 20–60 s for CP-SAT.
- 5 arms × 10 allocations = 50 RF fits at 100 trees ≈ 3–5 min on 4 workers
  (RF fits are the bottleneck; estimator cache is small).
- Total: ~5–8 min wall. Runs in parallel with the dayscale run (N_JOBS = 4 per user
  grant 2026-09-21; dayscale occupies 4 workers; 8 total on a 10-core / 4-perfcore
  machine → expect some perf-core oversubscription, acceptable for R = 10).

## 9. Outputs

- `results/r1aaf_metrics.csv` — per-arm × per-allocation AUROC val/test, n cols,
  Gini, `alloc_seed`, `all_opt`.
- `results/r1aaf_predictions.csv` — test predictions for BASE, BASE⊕R1a,
  BASE⊕R1a_resid (compact).
- `results/r1aaf_top_features.csv` — top permutation-importance features per
  composite arm per allocation (reuse R1a's logic).
- `results/r1aaf_univariate.csv` — OMITTED (allocation-independent; identical to R1a's
  output and reused via path).
- `results/r1aaf_repro.json` — versions, seeds, allocator spec, ortools version,
  demographics sha, r1a script sha, r1a cache sha, wall time per allocation.
- `results/run_log.txt` — human log.
- `results/REPORT.md` — §1 absolute AUROC; §2 paired deltas (95% + 97.5% Bonferroni);
  §3 top features; §4 caveats; §5 interpretation (allocator reproducibility note,
  balance audit, R = 10 power, R1a-label erratum pointer).
- `cache/demographics.parquet` (gitignored).
- `cache/balance_run{0..9}.csv` (gitignored).
- `cache/alloc_summary_run{0..9}.json` (gitignored).

## 10. Caveats

- **R = 10 under-powers the paired-Δ estimate.** Expected SE ~2× R1a's frozen-split SE
  (≈ 0.0022 vs 0.0012). Honest framing: this is a sanity check on the *direction* and
  *magnitude* of the increment, not a confirmatory test. The 95% CI likely straddles 0
  even at the R1a magnitude (~ +0.003) due to smaller n; the Bonferroni 97.5% CI is even
  wider. Report both, do not over-claim.
- **Frozen-split BASE 0.7434 ≠ allocator-folds BASE.** Different test users; per-allocation
  BASE is expected to sit near the frozen value but with a wider spread across allocations
  (allocator's seeded tie-break introduces per-allocation variation not present in
  stratified permutation). Descriptive only.
- **R1a label erratum.** R1a's REPORT lines 19/21/77 say "97.5% t-CI"; the formula there
  is `t.ppf(0.975, 29)` which yields a **95%** t-CI — the program's convention (cf.
  `feat_sel._mean_ci`, `feat_sel.build_report`, age experiment REPORT). Fixed in a
  follow-up commit after R1a-AF lands.
- **No vendor-circularity audit** (user decision 2026-09-21). The R1a increment is
  ch3000-only; vendor-channel caveat applies to BASE absolute AUROC, not the paired Δ.
