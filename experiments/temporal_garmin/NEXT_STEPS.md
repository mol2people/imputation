# Temporal ladder — open gaps and next steps

Written 2026-09-22, mid-run (5/10 allocations complete, `run_temporal.py run` executing;
results below are the 5-alloc snapshot and will shift ±0.01 at 10/10). Companion to the
frozen [`PLAN.md`](PLAN.md); this file is a working gap analysis and roadmap, not a frozen
plan. The next experiment requires its own frozen plan before execution.

## 1. Where we stand

Same cohort/folds/test-users as R1b (3,848 Garmin, 10 allocations, first-40-day window).
Reference anchors: R1b BASE⊕P40 = 0.7568 val / 0.7513 test.

| arm | val (mean of 5) | test (mean of 5) |
|---|---|---|
| Combined | **0.8490** | **0.8610** |
| HYDRA | 0.8369 | 0.8477 |
| MultiRocket | 0.8295 | 0.8407 |
| Profile24 | 0.7263 | 0.7311 |
| Profile288 | 0.7160 | 0.7220 |
| Summary_linear | 0.7104 | 0.7027 |
| Summary_RF | 0.6987 | 0.6933 |
| Shuffled_MR | 0.6648 | 0.6564 |

Established so far (all primary deltas positive in every completed allocation, an order of
magnitude above the Bonferroni MDD ≈ 0.01):

- Within-day temporal placement of 5-min HR values carries ~+0.15 AUROC beyond summary
  statistics and hourly profiles. MultiRocket − Shuffled_MR ≈ +0.16 isolates placement.
- Hourly aggregation destroys it (Profile24 ≈ Profile288 ≈ 0.72, both ≈ Summary + 0.02).
- Combination adds modestly over the best single transform (+0.01–0.02).

Pending lock-in: allocs 5–9; byte-identical alloc-0 `verify` gate; REPORT.md; README row;
commit. Peak RSS plateaued at 7.46 GiB (under the 8 GiB gate).

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

**Step 0 — lock the current run** (in flight): allocs 5–9 → `verify` gate → REPORT.md →
README row (respect the 14-line research-direction section) → commit code + results +
NEXT_STEPS.md → retire/rewrite CONTEXT.md (numba-seed gotcha superseded by `_nb_seed`
erratum in vendor_setup.py).

**Step 0.5 — selective-classification analysis (post hoc on saved predictions; procedure
predeclared 2026-09-22 before any coverage/precision numbers are computed):**

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

## 5. Discipline reminders

- Every step above = new frozen plan (estimand, arms, seeds, gates, stats predeclared
  before results are seen); deviations recorded as errata, never silent.
- Tuning on validation only; test stays exploratory; matched folds/test-user gates against
  saved predictions where pairing is claimed.
- README claims stay proportional: "predictability from the export", not physiology
  (vendor processing caveat, §2.1); recorded salutation ≠ sex/gender.
- Runtime/resource gates (CPU-only torch, ≤8 GiB peak, byte-identical reruns) carry over.
