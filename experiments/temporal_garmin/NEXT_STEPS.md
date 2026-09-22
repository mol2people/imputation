# Temporal ladder — open gaps and next steps

Written 2026-09-22, mid-run (5/10 allocations complete, `run_temporal.py run` executing;
results below are the 5-alloc snapshot and will shift ±0.01 at 10/10). Companion to the
frozen [`PLAN.md`](PLAN.md); this file is a working gap analysis and roadmap, not a frozen
plan. The next experiment requires its own frozen plan before execution.

## 1. Where we stand

**v2 results (post-erratum; this file's 10-alloc snapshot).** Same cohort/folds/test-users as R1b
(3,848 Garmin, 10 allocations, first-40-day window). R1b BASE⊕P40 anchors at
0.7568 val / 0.7513 test (paired test AUROCs across 10 allocs in REPORT §4).

| arm | val (mean ± SD, 10 allocs) | test (mean ± SD, 10 allocs) |
|---|---|---|
| Combined | **0.8489 ± 0.0141** | **0.8565 ± 0.0146** |
| HYDRA | 0.8374 ± 0.0149 | 0.8442 ± 0.0113 |
| MultiRocket | 0.8284 ± 0.0184 | 0.8361 ± 0.0149 |
| Profile24 | 0.7248 ± 0.0233 | 0.7370 ± 0.0157 |
| Profile288 | 0.7140 ± 0.0181 | 0.7327 ± 0.0159 |
| Summary_linear | 0.7077 ± 0.0193 | 0.7157 ± 0.0204 |
| Summary_RF | 0.6962 ± 0.0132 | 0.7031 ± 0.0192 |
| Shuffled_MR | 0.6619 ± 0.0139 | 0.6643 ± 0.0183 |

Primary family deltas vs `Summary_RF` (Bonferroni 4-arm, df=9, all 10/10 positive, REPORT §2):
Profile288 +0.018, MultiRocket +0.132, HYDRA +0.141, Combined +0.153.
Within-day placement signal: MultiRocket − Shuffled_MR = **+0.166** (10/10, REPORT §3).
Test pairing vs R1b BASE⊕P40: Combined − BASE⊕P40 = **+0.1051** pooled, 95% t-CI
[+0.0878, +0.1225], share > 0 = 1.00, df=9 (REPORT §4).

Selective classification (REPORT §10; `results/selective_classification.{md,csv}`):
- Combined raw @95%: cov₁ 21.4 ± 9.8%, cov₀ 3.8 ± 3.4%, cov_total **25.2 ± 10.5%**
  (test precision 0.956 / 0.937 — class-1 holds at the 95% target; class-0 mild
  small-set boundary drop).
- Combined cpc (Clopper-Pearson LCB, δ=0.10) @95%: cov₁ **8.1 ± 9.2%** (~5/10 allocs
  attain, test precision 0.959 over attained); class-0 cpc unattainable everywhere;
  @98% cpc unattainable in all 10 allocs.
- Combined − BASE⊕P40 raw @95%: **+12.4 pp** cov_total (25.2 vs 12.8, ~×2.0);
  cpc @95%: **+8.1 pp** (8.1 vs 0 — R1b BASE can't certify anything at n_val=579).
- Binormal projection (AUROC 0.86, π₁=0.64, symmetric): 22.2% raw cov_total @95%.
  Observed 25.2% slightly exceeds it because class-1 is stronger than binormal; class-0
  is weaker than binormal. Asymmetric per-class threshold shape is real.
- Random sanity (seeded uniform scores on real users): raw cov 0.2%, test precision ≈
  prevalence, cpc rejects everything — threshold mechanic is sane.

v2 determinism: `verify` gate PASS under relaxed 1e-6 standard (1–2 ULP per arm,
threaded-BLAS reduction order; MR/HYDRA/seed streams bit-stable).
v2 plateau RSS 8197 MiB (+0.06% over the 8192 MiB gate, recorded in REPORT §8).

Open: **α extended-grid addendum** (predeclared below) — alloc-0 α froze at the grid
boundary (1e3) for every family in v1 *and* v2; wide arms were still climbing.

## 2. What we're missing — open threats, ranked

### 2.1 Device/behavior artifact (the main interpretive threat)

Mechanism: Garmin writes dense continuous HR during logged activities (workout-shaped
curves: ramp–peak–cool-down) and sparser sampling otherwise. MultiRocket/HYDRA kernels can
latch onto workout shapes. "User logs runs" is behaviorally sex-correlated, so part of the
+0.15 may be *device-usage texture*, not physiology. **The Shuffled control does not rule
this out** — it preserves observation placement and only permutes values.

Discriminating probes (cheap; fold into the next experiment as sensitivity arms):

1. **Night-only arm** — transform bins 0–72 (00:00–06:00 local) only. Sleep is the least
   behavior-contaminated window. Night-retained gain → physiology-leaning; day-only-retained
   gain → behavior-leaning.
2. **Activity-hours exclusion** — mask out high-motion windows (needs activity tables or
   the walk/run binaries as proxy).
3. **Importance-by-dilation** — ridge coefficient mass by kernel dilation (5-min texture vs
   multi-hour shape).

### 2.2 Sampling-density texture (confounder, data on disk)

`day_bins.npz` stores per-bin `cnt`; we used only the binary mask. Users differ in epochs
per bin → bin means differ in smoothing (jagged vs averaged). If density correlates with
salutation (device model, wear habits), part of the "placement signal" is smoothing
texture. Probes: add cnt-profile block to B40; or homogenize by per-bin downsampling.

### 2.3 Label-noise ceiling (unmeasured)

Recorded salutation ≠ biological sex. If a fraction p of labels is effectively arbitrary,
max AUROC ≈ 1 − p/2. Probes: contradiction/duplicate-registration audit; inspection of
top-confidence errors of the best arm for physiological plausibility. Directly bounds the
0.95 ambition.

### 2.4 Recorded statistical caveats (non-fatal, go in REPORT.md)

- Alloc-0 α was selected *on* alloc-0 val → alloc-0 val delta optimistically biased for
  ridge arms (~+0.003 on one of ten allocs; RF has no α).
- Test split exploratory by predeclared rule (prior model development used it).
- Combined − BASE⊕P40 pairing is test-only (R1b saved test-only participant predictions).
- Epoch-span audit: start-bin assignment kept (weighted frac >5 min = 0.0004; 1-min epochs).
- NaN-tz rejection divergence had zero effect (0 events).

### 2.5 Missing analyses

- Coverage interaction: does the transform gain concentrate in high-coverage users?
- Fill-patch sensitivity: pool only ≥50%-covered days (currently partially-observed days
  contain train-median-patched segments).
- **Selective classification (the applied estimand) — see §2.6 below.** Calibration in the
  narrow score-calibration sense is subsumed by the per-alloc threshold selection there.

### 2.6 Selective classification — the imputation estimand, unmeasured

The applied goal is **missing-label imputation**, not ranking. AUROC measures ranking
quality; the imputable fraction at target accuracy is what downstream study design needs.
Reframing:

> Per alloc r, per arm: what fraction of participants can be **labeled** (class-1 vs
> class-0) at per-class precision ≥ 95% or ≥ 98%, with thresholds selected on val and
> evaluated on untouched participants?

This is selective classification (Chow 1970; El-Yaniv & Wiener 2010; Angelopoulos & Bates
2022 for the conformal version). Coverage@precision is **not** a function of AUROC
alone — it depends on score concentration at extremes — so must be measured, not
inferred from the AUROC table.

**Class asymmetry matters** with π₁ ≈ 0.64 (val prevalence): minority-class coverage
(`cov₀`) is far smaller than majority-class coverage (`cov₁`) at any given precision,
because the class-0 prior is lower and the threshold to achieve high precision for
class-0 must sit farther into the tail.

**Binormal illustrative expectations** (equal-variance Gaussian scores, π₁=0.64; real
ridge/RF scores may deviate):

| metric | AUROC 0.86 (d'=1.527) | AUROC 0.93 (d'=2.088) | gain |
|---|---|---|---|
| cov₁ @ 95% precision | 17.9% | 48.8% | ×2.7 |
| cov₀ @ 95% precision | 4.3% | 10.4% | ×2.4 |
| cov_total @ 95% precision | 22.2% | 59.2% | ×2.7 |
| cov₁ @ 98% precision | 9.6% | 30.1% | ×3.1 |
| cov₀ @ 98% precision | 0.6% | 5.2% | ×8.7 |
| cov_total @ 98% precision | 10.2% | 35.3% | ×3.5 |

These motivate Step 2 (multichannel) in the applied currency: a 0.86→0.93 AUROC gain is
roughly **×2.7** at 95% precision and **×3.5** at 98% precision overall, with **×8.7** on
the minority class — the biggest payoff is specifically on the harder class.

## 3. Channel inventory (ClickHouse exploration 2026-09-22)

Source: `rocs.vital_data_epoch` backup (read-only, HDD `bz_2tb`), scoped to the 3,848-user
cohort, `source=3` (Garmin). Designated-day probe = 100-user sample on their exact 40
day40 dates.

**Available on designated days:**

| channel | type | cohort users | ≥39/40 days (sample) | median rows/40d |
|---|---|---|---|---|
| Steps | 1000 | 3,848/3,848 | 68% | 1,702 |
| MET | 1012 | 3,848 | 73% | 5,213 |
| WalkBinary | 1115 | 3,848 | 68% | 1,620 |
| ActivityType (+details) | 1200–1202 | 3,848 | 79% | 2,744 |
| SleepState/InBed (≡1112) | 2000/2001 | 3,800 | 55% | 1,062/1,274 |
| SleepDeep/Light/Awake | 2003/2005/2006 | 3,794–3,799 | 49/55/10% | 281/794/178 |
| HRResting (dense) | 3001 | 3,848 | 24% (29 median days) | 18,799 |
| HRRestingHourly | 3002 | 3,848 | 24% | 412 |
| SPO2 | 3009 | 1,676 (44%) | — | — |
| RespirationSleep | 4002 | 2,074 (54%) | — | — |
| SedentaryBinary | 1104 | 3,846 cohort / 67% in-window | patchy | 1,348 |

**Absent (zero rows for cohort):** Height (5030), Weight (5020), InterbeatIntervals
(3029), RMSSD (3100), SleepREM (2002), sleep latency/interruption (2007/2008/2102),
Cadence/Speed (1260/1261), BurnedCalories total (1010).

Implications: the anthropometrics and HRV levers are dead in this export; steps + activity
binaries + sleep architecture + Garmin resting-HR are the multichannel levers. Data ends
~2022-12-17 (backup 2023-10). All values: Steps→doubleValue, HR→longValue,
Sleep→booleanValue (pipeline `parse_value_format_selection`).

## 4. Roadmap

**Step 0 — lock the current run** (DONE 2026-09-22): allocs 0–9 → `verify` gate (PASS,
relaxed 1e-6, 1–2 ULP) → REPORT.md (§1–§10) → README row (research-direction section
untouched) → commit code + results + NEXT_STEPS.md → CONTEXT.md retirement banner.
v2 = post-erratum primary record; v1 kept as `*_v1` erratum record (commit `168b9d7`).

**Step 0.5 — selective-classification analysis** (DONE 2026-09-22; numbers above in §1;
procedure below as predeclared 2026-09-22 before any coverage/precision numbers were
computed; outputs `results/selective_classification.{md,csv}`,
`risk_coverage_curves.csv`, `abstention_composition.csv`, embedded REPORT §10).
Recorded deviation: the `cpc` variant uses the Clopper-Pearson 90% LCB on val precision
(`L = B(δ, s, f+1)` per candidate threshold; CRC was predeclared in outline but CP-LCB
is the correct finite-sample control for a precision *ratio* — CRC bounds bounded-loss
*expectations*, not precision ratios; recorded as a deviation from the predeclared
outline, formula in `selective.py`).
Aggregation-bug fixes recorded post-numbers (display layer only; thresholds/targets/
selection rule untouched): unattainable thresholds now report precision as undefined
(not 0) and are excluded from precision means; cov_total/abst now grouped per-arm
correctly for non-crossfit arms. Abstention composition verdict: abstention is NOT
wear-driven (labeled vs abstained mask coverage 0.90 vs 0.88, ~254 vs 261 obs
bins/day); the separator is mean HR (71.1 vs 74.0 bpm) — ambiguous physiology, not
data scarcity. So the minimum-wear-time upstream lever (§2.5) is NOT the binding one.

**Step 0.7 — α extended-grid addendum (DONE 2026-09-22; `results/alpha_addendum.{json,md}`).**
Grid [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6] on alloc-0 val only; matrices rebuilt
via `fit_alloc(0)` (same seeds/fill/hygiene as v2). **Frozen decision rule result:
Combined val gain +0.0152 ≥ 0.01 → USER DECISION required on v3 full rerun.**

| family | frozen α | val @ frozen | α* | val @ α* | gain | test @ frozen | test @ α* |
|---|---|---|---|---|---|---|---|
| Summary_linear | 1e3 | 0.7161 | 3e2 | 0.7176 | +0.0015 | 0.7107 | 0.7118 |
| Profile24 | 1e3 | 0.7459 | 1e3 | 0.7459 | +0.0000 | 0.7404 | 0.7404 |
| Profile288 | 1e3 | 0.7204 | 1e4 | 0.7343 | +0.0138 | 0.7339 | **0.7178** |
| MultiRocket | 1e3 | 0.8234 | 3e3 | 0.8348 | +0.0114 | 0.8507 | 0.8626 |
| HYDRA | 1e3 | 0.8194 | 1e4 | 0.8468 | +0.0274 | 0.8401 | 0.8674 |
| Combined | 1e3 | 0.8446 | 1e4 | 0.8598 | **+0.0152** | 0.8625 | **0.8784** |

Notes: α* interior in the grid for every family (MR 3e3, HYDRA/Combined 1e4 — not at
the 1e6 boundary → no further extension needed). MR/HYDRA/Combined test gains track
val gains (+0.0119/+0.0273/+0.0159), test-corroborated. **Profile288 is the exception**
(val +0.0138 but test −0.016) — the textbook alloc-0 val-α selection-overfit signature
(§2.4 caveat materializes here); do NOT carry its gain forward. Sanity vs v2 metrics csv
at frozen α: max |Δval| = 0.0, max |Δtest| = 0.0 (the rebuild is bit-identical at α=1e3).
Wall 429 s; peak RSS 6308 MiB (6.16 GiB; darwin `ru_maxrss` is bytes, initial run
divided by 1024 — unit-corrected, json note).

**Decision needed:** v3 full rerun (72 min, extended grid from alloc-0 freeze, no other
changes) vs record-only. See message 2026-09-22 for the full cost/benefit framing.

**Step 0.5 procedure — as predeclared 2026-09-22 (archived verbatim, unchanged):**

Procedure (frozen, will not be adjusted after seeing numbers):
- **Per alloc r ∈ {0,…,9}, per arm.** Primary: `Combined`. Reference: `Summary_RF`,
  `BASE⊕P40` (applied delta vs previous best). Plus a `Random` baseline sanity check.
- **Calibration:** on val (n=579/alloc), per assigned class:
  - Class-1: smallest τ₁ s.t. empirical P(y=1 | score ≥ τ₁) ≥ target.
  - Class-0: largest τ₀ s.t. P(y=0 | score ≤ τ₀) ≥ target.
  - All tied scores included on both sides (no within-tie cherry-picking or randomized
    tie-breaking — ties are deterministic in the procedure).
  - Report two variants: (a) **raw empirical** threshold; (b) **conformal risk-controlled
    (CRC)** threshold chosen so that P(test precision ≥ target) ≥ 1−δ with δ=0.10,
    calibration-finite-sample (Angelopoulos et al. 2022, Bates et al. 2021). For
    bounded monotone precision-type losses the CRC correction is the
    `(n_cal+1)`-denominator calibration with the standard finite-sample bound; recorded
    by formula in the analysis script.
- **Targets:** per assigned class, 95% and 98% (four numbers per alloc per arm).
- **Evaluation on test (n=573/alloc):** per-class precision (Wilson 95% CI), per-class
  coverage, overall coverage, abstention fraction.
- **Full curve:** also report the risk-coverage curve (precision vs coverage across all
  thresholds) per arm per alloc — the complete operating-point object for downstream
  selection. The four per-class target numbers are summary statistics of the curve.
- **BASE⊕P40 specific:** R1b saved preds are test-only (`r1b_predictions_part.csv`), no
  val calibration available. Use 50/50 **half-sample cross-fit on test** (thresholds on
  one seeded half, evaluation on the other; both halves independently evaluated for
  averaging). Flagged as exploratory; honest about the calibration mismatch.
- **Aggregates:** cross-alloc mean ± SD (alloc = repeat unit, as for AUROC); alloc-0 val
  α-contamination flagged separately.
- **Abstention composition:** report mean mask coverage, observed-bins-per-day, B40 stats
  for abstained vs labeled subsets — wear-coverage interaction with the abstention
  region. If abstained users are systematically low-coverage, that's a separate
  engineering lever (require minimum wear-time before prediction).
- **Compute:** zero model retraining; reads `temporal_predictions.csv` +
  `r1b_predictions_part.csv`. Implementation predeclared (script committed) before any
  numbers computed.
- **Targets are candidates, not population promises.** With split-conformal at δ=0.10
  the 95% targets become finite-sample guarantees per alloc (exchangeability within
  the random val/test split, conditional on the split). At 98% target with n_val=579
  the conformal correction is non-trivial — expect coverage shrinkage vs raw empirical.
  Population-level promise is further limited by (i) cohort selection (coverage-gated
  Garmin users — not a random sample), (ii) test-split exploratory status (prior model
  development touched test participants), (iii) label-noise floor per §2.3 (if ≥2% of
  salutations are effectively arbitrary, 98% precision against recorded labels sits at
  or below the noise floor).

Reframes Step 2: the multichannel push should be justified in **coverage@95% precision**
terms, not AUROC alone. The ×2.7 / ×3.5 expected gains from §2.6 table make the
applied-case payoff concrete.

**Step 1 — artifact-discrimination probes** (§2.1, §2.2): night-only / day-only transform
arms + cnt-profile arm. Small; can be sensitivity arms inside Step 2 rather than standalone.

**Step 2 — MULTICHANNEL experiment** (the main next experiment; new frozen plan required):
- Scan steps (1000), MET (1012), walk (1115), activity-type (1200), sleep binaries
  (2000/2001/2003/2005/2006), resting-HR (3001/3002) into the same 288-bin × 40-day grid
  (same designated user-days, per-event timezone convention as scan_bins.py).
- Arms (all with the shared B40 block, same hygiene/α-precalibration/matched-folds
  protocol): Steps_only, Sleep_only, Steps+Sleep, HR+Steps, HR+Sleep, HR+Steps+Sleep
  (MultiRocket or HYDRA pooled, plus handcrafted daily-aggregate blocks).
- Multichannel HYDRA is native (conv1d over `(N, d, L)`) — stack [HR, steps, sleep-stage]
  as parallel channels; MultiRocket stays per-channel pooled.
- Coverage gate per channel predeclared before fitting (e.g., ≥20/40 days with data for
  inclusion; SPO2/respiration as partial-coverage arms or excluded).
- Bonferroni family adjusted for the new arm count.
- Expected: 0.89–0.93 combined AUROC. 0.95 AUROC unlikely without anthropometrics
  (absent) — reframe target as **coverage@95% precision** per §2.6 / Step 0.5.

**Step 3 — tuning ladder** (ranked by expected value, all val-only, new frozen plan):
1. Pooling augmentation across 40 days (quantiles p10/50/90, fraction-above-median) in
   place of mean/SD only.
2. Multichannel HYDRA (above).
3. Day-stacked long input (length 11,520) for cross-day structure.
4. Channel additions (Step 2).
5. Seed-ensembling of transforms (3–5 seeds; principled variance reduction, not tuning).
6. Kernel budget up (HYDRA g→256; MR num_features→5,000).
7. Head: SVD→ridge, elastic-net.
8. α grid widening (keep alloc-0 freeze rule).

**Step 4 — label-noise probe** (§2.3): contradiction audit + confident-error inspection;
sets the honest ceiling for any target number.

## 5. Appendix A — Step 2 multichannel plan (DRAFT, pending freeze; informed by §3 channel inventory + v2 Step 0.5 gap)

The binormal projection at v2 framing promised ×2.7 coverage@95% precision at AUROC 0.93 vs ×1 at AUROC 0.86; the realized gain from BASE⊕P40 (0.75) to Combined (0.86) was only **×2.0** because the projection is symmetric and the realized operating characteristic is asymmetric (strong class-1, weak class-0). The next experiment must close that gap in the applied currency. New channels are the lever; this is the plan to evaluate them.

**A.1 Estimand & primary metric.** Reuse v2 estimand. Primary: AUROC. **Co-primary: coverage@95% precision** (Step 0.5 procedure, raw + CP-LCB variants, val threshold → test evaluation). The coverage@precision metric is the binding one for the applied goal (imputing labels for downstream study design) and is not a monotone function of AUROC — must be measured, not inferred.

**A.2 Cohort / folds / test users.** Identical to v2: R1b's 3,848 Garmin cohort, same designated 40-day window, same 10 allocs, same fold/test-user set. Two pairing anchors recorded: against R1b BASE⊕P40 (inter-experiment), against v2 Combined (intra-experiment, the within-design delta). Folds parquet + r1b_predictions_part.csv read-only.

**A.3 Channels + coverage gate.** Predeclared per-bin aggregation (channel-specific analogues of `scan_bins.py`):
- **Steps (1000)**: mean steps per 5-min bin (integer, float-cast). Mask-aware fill (train-median per clock-bin).
- **MET (1012)**: mean MET per bin (float). Same fill.
- **WalkBinary (1115) / SedentaryBinary (1104)**: mean per bin ∈ [0,1] (fraction of bin marked active). Same fill.
- **ActivityType (1200–1202)**: 1200 dominates; encode as 5 binary channels (walk/run/cycle/other/inactive) per bin → mean per binary. Or treat 1200 as ordinal (1–7 mapped to bins). Decide at freeze.
- **Sleep stages (2000/2001/2003/2005/2006)**: 5 binary channels per bin (in-bed, light, deep, REM, awake). Same fill.
- **HRResting (3001)**: mean resting HR per bin. Native scale; same fill.
- **HRRestingHourly (3002)**: hourly resolution; 24 bins/day with 12× upsample or separate arm at native 24-bin × 40-day.
- **Excluded from primary** (predeclared): SPO2 (3009, 44% coverage skew), RespirationSleep (4002, 54% coverage skew) — record as exploratory partial-coverage arms if material; otherwise excluded.

**Coverage gate** (predeclared): include a channel in the multichannel HYDRA stack only if ≥ 80% of users have ≥ 30/40 designated days with ≥ 12 bins of data on that channel. Failing channels → partial-coverage arm restricted to the qualifying subset (record selection bias in REPORT; do not pool).

**A.4 Multichannel representation.** HYDRA is natively multichannel (`hydra_multivariate.py` in vendored repo, or stack channels and forward Hydra per-channel — verify at bench). Pool across 40 days exactly as v2 (mask-aware nanmean + nanstd per day-feature). MultiRocket is 1D; per-channel pooled (apply_mr_pooled per channel separately, concatenate feature blocks).

**A.5 Arms (proposed primary family; exploratory arms noted).**

| arm | channels | conv kernel | role |
|---|---|---|---|
| v2_Combined (replay) | HR | MR pooled + HYDRA pooled | control (within-experiment anchor) |
| Steps_only | Steps | MR + HYDRA | primary |
| Sleep_only | Sleep (5 stages) | MR + HYDRA | primary |
| RestingHR_only | HRResting | MR + HYDRA | primary (partial-coverage, exploratory tag) |
| HR+Steps | HR, Steps | multichannel HYDRA + per-channel MR | primary |
| HR+Sleep | HR, Sleep | multichannel HYDRA + per-channel MR | primary |
| HR+Steps+Sleep | HR, Steps, Sleep | multichannel HYDRA + per-channel MR | primary |
| HR+Steps+Sleep+RestingHR | 4 channels | multichannel HYDRA + per-channel MR | primary (if 4-channel bench feasible) |
| MultiChannel_Combined | stacked feature blocks across all channels | mixed | primary summary arm |
| Shuffled_MC | permuted placement within channels (each channel shuffled independently) | MR | control (placement-signal sanity in the multichannel setting) |

**Primary family (frozen Bonferroni correction, 4 paired deltas, df=9, t.ppf(1−0.05/8, 9))**: HR+Steps+Sleep − v2_Combined, HR+Steps+Sleep − BASE⊕P40, MultiChannel_Combined − v2_Combined, MultiChannel_Combined − BASE⊕P40. All other pairwise comparisons are exploratory.

**A.6 Hygiene & head.** Reuse v2 protocol verbatim: train-only SimpleImputer → VarianceThreshold → SparseScaler (HYDRA block) / StandardScaler (others); Ridge with α frozen per family via alloc-0 precalibration on the **extended grid** [1e2..1e6]. **The Step 0.7 addendum verdict is in (Combined val gain +0.0152 ≥ 0.01) and the conv-arm α optima are interior in that grid (MR 3e3, HYDRA/Combined 1e4), so the multichannel experiment uses the extended grid from the start — no v2-style 1e-3..1e3 grid retreat.** Summary_RF control carries forward unchanged.

**A.7 Determinism + gates.** Same standards as v2: byte-identical OR max diff ≤ 1e-6 per arm (relaxed gate with the `verify` CLI dispatched and hard-failing); seed streams stable across all components; peak RSS ≤ 8 GiB (v2 plateau 8197 MiB; multichannel HYDRA forward cost scales ~linearly with D — bench first).

**A.8 Statistics.** Bonferroni two-sided 95%-simultaneous CIs on the 4 primary deltas (df=9). Report arm AUROCs (mean ± SD across 10 allocs); primary paired deltas; coverage@95% and coverage@98% (raw + cpc, per-arm per-alloc); MultiChannel_Combined vs BASE⊕P40 paired test deltas (fold/test-user gate identical to v2). Risk-coverage curves for the 3-4 most promising arms per Step 0.5 procedure. Abstention composition for MultiChannel_Combined (wear-driven or physiology-driven?).

**A.9 Expected gain vs v2 — honest framing.** Binormal projection at AUROC 0.86 → 0.92 gives ×2.2 coverage@95% (smaller than the original ×2.7 because the symmetric projection is now discounted by the realized asymmetry). **Realistic primary target: AUROC ≥ 0.90 AND coverage@95% precision raw ≥ 35% (vs Combined 25.2%) — and CP-LCB-certified coverage ≥ 15% (vs Combined 8.1%).** These are the bridging numbers; do not promise physiological interpretation, only predictability from the export (README discipline).

**A.10 Compute projection.** v2 was 72 min for 7 arms × 10 allocs; multichannel adds ~6 new arms and multichannel HYDRA forward costs that scale with D. Bench first: 100-user × 40-day multichannel HYDRA forward for D ∈ {1, 2, 3, 4}; full-projection = 10×. If D=4 HYDRA wall exceeds 8 min/alloc combined, restrict D=4 to exploratory status or drop it.

**A.11 Pitfalls (predeclared).** (i) The binormal ×2.7–3.5 was optimistic at v2 (realized ×2.0); do not promise ×3 at v3. (ii) RestingHR's 24% coverage creates selection bias; the partial-coverage arm is NOT comparable to the full-coverage arms at the cohort level. (iii) Per-channel hygiene must remain train-only — no test leakage through the multichannel fill or scaler. (iv) Multichannel HYDRA padding (same 9-tap, dilations {1,2,4,8,16,32}, padding 4d); output feature count scales linearly with D — record per-arm n_cols; VarianceThreshold may drop more constant columns as D grows (e.g., count_min on flat binary channels). (v) Integer/binary channel distributions challenge SparseScaler's sqrt assumption — reuse as-is, validate the per-arm AUROC sanity vs a StandardScaler variant at bench (record the choice). (vi) Activity-type mapping is a free parameter — fix at freeze (recommend 5-binary or 1-ordinal, not both).

**A.12 Open freeze checklist.** Before promoting this draft to `PLAN.md` (frozen):
1. Primary family size: 4 paired deltas (above) vs full pairwise Bonferroni — user.
2. RestingHR: partial-coverage primary arm vs excluded.
3. Max D for multichannel HYDRA: 2/3/4 — bench-gated.
4. Coverage gate thresholds (K_min, % users): confirm or tighten.
5. α grid: **RESOLVED by Step 0.7** — extended grid [1e2..1e6], α* interior for every
   family; P288's val-only gain is selection noise (test −0.016) and is not carried
   forward.
6. Activity-type encoding (5-binary vs 1-ordinal) — pick one.
7. Sleep stages encoding: 5 binary channels or coarser (in-bed/light/deep/REM/awake → 3-level ordered?).
8. Pairing targets confirmed (v2 Combined + BASE⊕P40).
9. Wear-coverage threshold for "primary cohort inclusion" (currently all 3,848; v2 had no exclusion).
10. Bonferroni factor: 4 (primary family) or 8 (×2 for raw+cpc on coverage)?

**A.13 What does NOT change.** Same cohort, same designated 40-day window, same folds, same fold/test-user gate, same train/val/test hygiene, same ridge α-precalibration discipline, same v2 erratum (mask-aware fill) carried forward, same RSS gate, same vendor SHAs, same determinism gate.

## 6. Discipline reminders

- Every step above = new frozen plan (estimand, arms, seeds, gates, stats predeclared
  before results are seen); deviations recorded as errata, never silent.
- Tuning on validation only; test stays exploratory; matched folds/test-user gates against
  saved predictions where pairing is claimed.
- README claims stay proportional: "predictability from the export", not physiology
  (vendor processing caveat, §2.1); recorded salutation ≠ sex/gender.
- Runtime/resource gates (CPU-only torch, ≤8 GiB peak, byte-identical reruns) carry over.
