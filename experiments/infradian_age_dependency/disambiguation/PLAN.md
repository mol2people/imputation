# Lunar and seasonal disambiguation of the 25–35 day infradian HR excess

**Status: frozen before the run (2026-10-20).** Every analysis choice that could affect the
result is pre-specified here. Deviations must be recorded in the REPORT with a reason.

## Question

The 25–35 day band contains the lunar synodic month (29.530589 d), the lunar anomalistic
month (27.554550 d), the 12th harmonic of the annual cycle (30.4375 d), a 4-week alias
(28.0 d), and the menstrual-cycle-length distribution (population median 29.3 d; only 12.4 %
exactly 28 d; Bull 2019). The saved HRp10 endpoints condition only on intercept, linear
trend, weekday, and clock coverage. This experiment asks:

> Does the young-female 25–35 day excess survive removal of the lunar (synodic + anomalistic)
> and seasonal (quadratic + cubic trend) components, and which pre-specified candidate period,
> if any, absorbs it?

## Exposure and cohort

- Manifests: `artifacts/manifests/participants.parquet`, `artifacts/windows/windows.parquet`,
  `artifacts/windows/selected_dates.parquet` (unchanged, read-only).
- Protocols: `balanced90` (180-d windows, 15 dates per 30-d block) and `expanded90`
  (270-d windows, 10 dates per 30-d block). Both have 90 dates per window.
- Sources: Apple Watch (6) and Garmin Tier 1c (3). Samsung (7) has 8 eligible windows and is
  excluded.
- Arm: HRp10 only (10th percentile of 5-min HR; identical extraction path as the saved
  reference endpoints).
- Task count: 8,397 participant tasks per protocol (same as the `rhr_sensitivity` run).

## Nulls

Unchanged engine: two-component continuous-time AR(1) fitted by REML on the residual
subspace, B = 199 surrogates on the exact selected dates, projected through the *same*
projector P₀ as the observed series. The extended baseline changes P₀; observed and null
share that design, so the excess D(f) = P_obs(f) − P̄_null(f) remains unbiased under the null.

## Regressors (the only change to the pipeline)

Baseline X₀ (unchanged, 12 columns): intercept, linear elapsed time, weekday indicators
(Tue–Sun, Mon reference), four 6-hour clock-coverage fractions.

Additive blocks, appended before the same deterministic QR screening:

| Block | Columns | Justification |
|---|---|---|
| L (lunar) | cos ψ, sin ψ, cos ν, sin ν | ψ = true geocentric Sun–Moon elongation (0 at new moon, 180° at full); ν = Moon true anomaly (0 at perigee). Meeus truncated series, ~0.3° accuracy. Synodic = illumination mechanism; anomalistic = distance/tidal mechanism. |
| S (seasonal) | τ², τ³ with τ = 2(t − t_min)/(t_max − t_min) − 1 | 180/270-d windows span a seasonal arc; a full annual sinusoid is unidentifiable in half a cycle. Cubic detrending was the strongest low-frequency control in the Garmin replication. |
| W (weekly) | none | **Omitted by design.** At integer day resolution every sinusoid at an integer harmonic of 1/7 is exactly 7-periodic and therefore lies in the span of the six weekday dummies already in X₀. A separate weekly block is linearly redundant (verified algebraically in `validate.py`). |

Six extra columns; retained rank ≈ 18, leaving ~72 residual degrees of freedom per window.

## Endpoints

Primary: `Q_i`, the trapezoidal band excess over [1/35, 1/25] d⁻¹, paired per window against
the saved HRp10 reference endpoint (join on user_id, source, protocol, window_index,
selected_dates_hash).

Secondary: `E_i2` (max-over-narrow-grid excess), `peak_period`, `A_hat` (cosinor amplitude),
and the discrete p-value.

Pre-specified tests, computed in `analyze.py` from saved daily values and the reconstructed
design (Holm-corrected across the five periods):

| Candidate | Period (d) | Hypothesis |
|---|---|---|
| synodic | 29.530589 | photic lunar |
| anomalistic | 27.554550 | tidal lunar |
| 4-week | 28.000000 | weekly sub-harmonic alias |
| annual harmonic | 30.437500 | 365.25/12 |
| menstrual median | 29.300000 | reproduction |

Each is a 2-d.f. partial F for adding cos/sin at that fixed period to the extended baseline.
Periods closer than one Rayleigh resolution element are reported with their pairwise design
correlation and interpreted jointly, not as independent identifications.

Diagnostic (new; the existing calendar-epoch acrophase uniformity does not discriminate
because each window starts at a random lunar phase): for windows with `p_value_discrete ≤ 0.05`
and `peak_period ∈ [27, 30.5]`, compute the cosinor acrophase and evaluate ψ and ν at that
timestamp. A Rayleigh test across participants on {ψ(t_acrophase)} and {ν(t_acrophase)}:
concentration indicates lunar phase-locking; uniformity indicates an endogenous rhythm.

Blockwise absorbed variance: partial R² of L and of S against the baseline-only design, per
window, summarised by stratum. Placebo: the same quantity at ghost periods (synodic ± 0.5 d,
anomalistic ± 0.5 d); the true period must absorb more.

## Strata

F 20–35, F 36–50, F 51–75, M 20–35, M 36–50, M 51–75 (age = `age_dec1`; sex = salutation 10).

## Validation gate (before interpreting results)

1. Lunar phase model: |ψ| < 6° at a table of known new moons 2015–2026; spectral peak of
   cos ψ at 29.53 ± 0.05 d over 2000–2026; spectral peak of cos ν at 27.55 ± 0.15 d.
2. Weekly redundancy: max residual norm of 7-d sin/cos (and 2/7, 3/7, 4/7 harmonics) after
   projection onto the 12-column baseline space < 1e-8 on real selected-date sets.
3. Injection: pure 29.530589-d and 27.554550-d sinusoids + AR(1) noise on real date sets are
   absorbed by L (partial R² > 0.9) and leave no in-band excess; a 28.5-d injection is *not*
   absorbed and retains band excess.
4. Negative-control FPR: male rejection rate at `p_value_discrete ≤ 0.05` under the extended
   baseline ≈ 5 % (the reference run gives ≈ 5 %).

If test 1–4 fails, stop and diagnose; do not interpret the real-data result.

## Decision rules

- **Robust**: young-female band excess after extension ≥ 70 % of the baseline-only value,
  absorbed R²(L+S) < 10 % of in-band variance, male FPR ≈ 5 %.
- **Environmental collapse**: young-female excess removed by L or S. A female-only collapse
  with an unchanged male null is mechanistically implausible and must be reported as such
  (a shared environmental driver should appear in the male negative control).
- **Dissociation**: Garmin peak absorbed by the anomalistic block but Apple peak not (or vice
  versa) — report both periods and the absorbed shares; do not average across devices.

## Limitations (stated in advance)

- No menstrual-onset data: the strongest available claim is "not attributable to the tested
  environmental candidates", not "identified as the menstrual cycle".
- No second (1/f) null: not run, by decision.
- At 180-d span the Rayleigh resolution in period near 28 d is ≈ 4.4 d (≈ 2.9 d at 270 d), so
  the periodogram cannot separate 27.55/28.5/29.53. Separation here is by regression on
  near-orthogonal basis functions (design correlations ≈ 0.4 at 180 d, ≈ 0.25 at 270 d), not
  by spectral resolution. A genuinely resolving test requires the multi-year cohort
  (ΔT ≈ 0.7 d), deferred to phase 2.
- The lunar phase model is a truncated series; residual phase error < 6° reduces absorbed R²
  negligibly (cos 6° = 0.995).

## Run

```bash
PY=HEARTBEAT_LAB/.venv/bin/python
RUN=experiments/infradian_age_dependency/artifacts/disambiguation/run_20261020

$PY experiments/infradian_age_dependency/disambiguation/validate.py
$PY experiments/infradian_age_dependency/disambiguation/run_disambiguation.py \
  --protocols balanced90 expanded90 --workers 8 --output-dir "$RUN"
$PY experiments/infradian_age_dependency/disambiguation/analyze.py --run "$RUN"
```

Outputs refuse to overwrite an existing run directory.

---

# Addendum 1 (2026-10-20, before any real-data run): the lunar block is a band-stop

The validation gate was run before the endpoint job. It passed (lunar phase error 0.20 deg
against 10 eclipse epochs; weekly redundancy 5.6e-12; engine consistency exact) and produced a
result that invalidates the original decision rules.

## Measurement

Recovery fraction of a unit sinusoid after projection on the extended design, real 90-date
windows, 40 windows per protocol:

| Period (d) | S-only (tau^2, tau^3) | L+S (lunar + trend) |
|---|---|---|
| 23.0 | 0.978 | 0.925 |
| 27.5546 (anomalistic) | 0.975 | 0.003 |
| 28.0 | — | 0.005 |
| 28.5 | 0.977 | 0.008 |
| 29.5306 (synodic) | 0.973 | 0.003 |
| 30.4375 (annual/12) | — | 0.027 |
| 36.0 | 0.964 | 0.879 |
| 45.0 | 0.949 | 0.894 |

The lunar block (cos/sin of the true elongation and of the true anomaly) absorbs 99.2-99.7 %
of *any* sinusoid in 27.5-30.5 d. The reason is structural, not a coding artifact: two
bracketing frequencies synthesize an intermediate frequency over a short window, and the
collinearity is a sinc function of df * T.

Span required to separate 28.5 d from 29.53 d: 1/df = 820 d. Observed cache spans: Apple
median 400 d, max 892 d, median 147 qualified days (n=25 sample); Garmin median 682 d, max
905 d, median 497 qualified days (n=15). Period identification at 28.5 vs 29.53 is therefore
not achievable in this repository, at either protocol length.

## Consequences (pre-registered here, before the run)

1. The L+S endpoint arm is **not run**. A collapse under L+S cannot distinguish "lunar" from
   "menstrual": it is guaranteed by the design. Reporting it as an environmental null would be
   a false inference.
2. The S-only arm (2.3-2.5 % absorption in band) remains available as an identified
   seasonal/trend control; it is not part of this run.
3. Executed instead: the two discriminators that do not require period separation.

### T1 lunar-direction test (primary)

For each window, regress the baseline residual on (cos psi, sin psi) and, separately, on
(cos nu, sin nu). The coefficient angle is the lunar phase at which HR peaks. Under lunar
forcing the angle is concentrated across participants; under an endogenous rhythm it is
uniform. Not tautological: psi and nu are external physical variables, not calendar time.

### T2 acrophase test (secondary)

Estimate the period freely on a 0.05 d grid over 25-35 d; map the cosinor acrophase to psi and
nu. Null: acrophase uniform inside each participant's window (400 draws). Attenuated by
period-estimation error; reported as a cross-check, not as the primary.

### T3 circaseptan control

Day-of-week of the empirical acrophase, chi-square(6). A 28-d alias of a weekly component
would concentrate weekdays.

### Male-cohort bound (independent of the phase tests)

Under no sex-by-lunar interaction, the lunar-attributable share of the young-female band
excess is bounded by Q_male / Q_young_female, estimated on the saved HRp10 endpoints with a
participant-level bootstrap. Sex-independent lunar forcing predicts a ratio near 1.

## Analysis choices frozen for T1/T2/T3

- One window per participant (lowest window_index) for independence.
- Strata: F20-35, F36-50, F51-75, M20-35, M36-50, M51-75 (age = age_dec1).
- Primary family: T1_lunar and T1_ano in F20-35 and M20-35, Holm-corrected within protocol.
- T2 and T3 are descriptive; no multiplicity correction.
- Both protocols reported separately; they reuse participants and are not independent, so no
  pooling by Stouffer.

## Artifacts

- `calibration_recovery_balanced90.csv`, `calibration_recovery_expanded90.csv`
- `<run>/phase_records_{protocol}.parquet`, `phase_lock_summary.csv`,
  `phase_lock_provenance.json`
- `male_bound.csv`, `male_bound_strata.csv`

---

# Addendum 2 (2026-10-20, after the run): outcomes against the pre-registered rules

Run `artifacts/disambiguation/run_20261020`; both protocols; 13,598 and 15,074 windows.

| Pre-registered rule | Outcome |
|---|---|
| T1 primary, Holm across {T1_lunar, T1_ano} x {F20-35, M20-35} | **Void.** The male negative control is the most positive cell (M20-35 Rbar_nu = 0.152, p = 1e-10); Holm p = 1.0 at the only nominal hit. |
| Rotation null as the calibrated null | **Invalidated during analysis.** A per-participant rotation destroys any calendar-referenced common structure whether or not it is lunar, so it cannot test specificity. Computed, reported, not used. |
| T2 acrophase, T3 weekday as secondary | Both null: permutation p = 0.015-0.41 (psi) and 0.06-0.48 (nu), 12 tests; weekday chi2 n.s. in 11/12 cells. |
| Male bound: sex-independent forcing predicts ratio near 1 | **Ratio 0.027-0.061** (young male), CI includes 0 in both expanded-protocol cells. |

## Unregistered analysis added after the T1 male control failed

The T1 failure forced a specificity test that was not in the original plan: the
cross-participant phase-coherence spectrum Rbar(T) over 25-35 d, with a red/white-noise
null computed on the real date grid.

- Null: 2,507 synthetic windows, Rbar never exceeds the 95 % line (exceedance 0.000).
- Real, one window per participant: exceedance 0.21-1.00; peaks at 27.0-28.0 d and 34.3 d.
  **No stratum peaks at the lunar period.** Rbar(29.53) = 0.020-0.079, below the 27.5-28.0 d
  lobe in five of six strata.
- At 28.0 d the phase is stable across start quarters (84-106 deg); at 29.53 d it moves
  75 -> 195 -> 303 -> 288 deg, i.e. annual leakage.
- At 28.0 d, Rbar(calendar-referenced) = 0.119 vs Rbar(window-relative) = 0.012
  (expectation 0.010, 95 % line 0.018, n = 9,230): absolute-time, not window geometry.
- Present in men at the same or greater strength as in young women (M36-50 0.126 versus
  F20-35 0.100).

An earlier all-window pass (15,074 windows, not independent) gave a sharper lunar trough and
was discarded; every number above is participant-level.

This is a real ~28-d shared component and a new confound for the band. It is not lunar.
Its origin (four-weekly behaviour vs coverage-mediated measurement) is unresolved and is
stated as such in `REPORT.md` §5.

## Decision rules as applied

1. Lunar attribution rejected on the male bound and on phase non-specificity, not on the
   L+S endpoint collapse (which was never run).
2. The lunar regression arm stays dropped.
3. The S-only arm was not needed for this verdict and remains unrun at this point. (Superseded by Addendum 3: it was run afterwards on request.)
4. The 27.5-28.0 d component is the next required control, not the Moon.

---

# Addendum 3 (2026-10-20, after the main run): the S-only arm was executed

Addendum 1, consequence 2, left the S-only arm available but unrun. It was run at the user's
request after the main analysis; the pre-registered rule is unchanged.

## What was run

- `seasonal_arm.py`: the identical engine (`scripts/02_compute_endpoints_and_nulls.py`) with
  `build_and_screen_nuisance_baseline` replaced by the 12-column baseline plus tau^2, tau^3,
  screened by the same sequential QR. Both protocols, both devices, B = 199, same per-window
  seed.
- Determinism control: with the original 12 columns the same engine reproduces the saved
  `Q_i` and `p_value_discrete` exactly on 18 windows (max |dQ| = 0, max |dp| = 0), so the
  pairing with the reference endpoint is exact.
- One window per participant, ages 20-75, identical to the main strata.

## Pre-registered rule and outcome

Rule (Decision rules, above): robust if the young-female band excess after extension is
>= 70 % of the baseline-only value, absorbed in-band variance < 10 %, male FPR ~ 5 %.

| Protocol | F20-35 Q_ref | Q_S | Q_S/Q_ref (95 % CI) | male FPR (ref -> S) |
|---|---|---|---|---|
| balanced90 | 0.0280 | 0.0287 | 1.026 (0.990-1.064) | 4.7 -> 3.9 % |
| expanded90 | 0.0273 | 0.0274 | 1.005 (0.971-1.040) | 4.6 -> 4.1 % |

Design absorption of a unit 28-d sinusoid is 2.6-2.7 % (recovery 0.973-0.974). F36-50 ratio
1.03-1.04, F51-75 0.83-0.89, all above 0.70. Male ratios are uninterpretable because
Q_male ~ 0; the FPR is the male diagnostic and stays at 4 % (slightly conservative).

**Robust.** The excess is not a seasonal arc or a smooth within-window trend.

## Residual gap

S is a cubic polynomial. It does not exclude a non-smooth in-band seasonal pattern, in
particular the 30.4375-d 12th annual harmonic, which is a pre-specified candidate rather than
a nuisance. The coherence spectrum has no peak at 30.4375 d, but the 4.4-d Rayleigh resolution
at these spans prevents a formal period assignment.
