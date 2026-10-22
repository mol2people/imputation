# Lunar and shared-environmental disambiguation of the 25–35 d HR excess

Run: `artifacts/disambiguation/run_20261020`. Protocols `balanced90` and `expanded90`,
sources Apple (6) and Garmin (3), `HRp10`, one window per participant for all tests.
Written 2026-10-20. All results below are complete; no run is pending.

## 1. Answer

Two discriminators were run because the pre-specified lunar regression cannot work here
(Addendum 1 of `PLAN.md`: the lunar + trend block absorbs 99.2–99.7 % of any 27.5–30.5 d
sinusoid, and every window spans 156–179 d, so 28.5 d and 29.53 d are separated by less
than the 4.4-d Rayleigh resolution).

**The 25–35 d excess in young women is not attributable to the Moon.**

1. The male-cohort bound caps any sex-independent environmental share at **3–6 %** of the
   young-female excess (CI includes zero on both devices, expanded protocol).
2. Lunar phase-locking is **not specific**: the cross-participant coherence spectrum has no
   peak at 29.53 d anywhere (its peaks are at 27–28 d and 34 d), the coherence there is lower
   than the 27.5–28.0 d lobe in five of six strata, and its mean phase rotates with the season
   (the signature of annual leakage). The anomalistic phase behaves the same.
3. A pre-registered S-only seasonal/trend control (τ², τ³) leaves the excess intact:
   Q_S/Q_ref = 1.03 [0.99, 1.06] (balanced90) and 1.01 [0.97, 1.04] (expanded90) in F20–35,
   against a pre-registered robustness threshold of 0.70. The design absorbs only 2.6–2.7 % of
   a unit 28-d sinusoid, so this is a real control, not a formality.
4. The concentration that the naive lunar direction test reports is a **real shared component
   centred at 27.5–28.0 d, plus a second lobe near 34 d, present equally in men**, not at the
   lunar period.

The unexpected result is item 4: a calibrated, sex- and age-independent, absolute-time
component at ~28 d (and a second lobe near 34 d) that survives every control run here. It is
not lunar. Its energetic weight is small: with R̄ = 0.04–0.16 and mean fixed-period amplitude
0.76–1.24 bpm, the coherent amplitude is R̄·A ≈ 0.04–0.12 bpm (0.10–0.12 in five of six
strata), i.e. R̄² ≈ 1–2.4 % of the 28-d power. It is therefore a confound for *phase*-based tests (T1–T3, coherence), not for the
band statistic `Q_i`: its variance (≤ 0.12²/2 ≈ 0.007 bpm²) is ≈ 0.07 % of the residual
variance (σ ≈ 3.2 bpm).

## 2. Male-cohort bound

`Q_i` is a band excess measured against a calibrated per-subject null, so a sex-independent
component appears in men at the same absolute magnitude. Under no sex × lunar interaction the
lunar-attributable share of the young-female excess is bounded by `Q_male / Q_young-female`.
One window per participant, 4,000 participant-level bootstrap draws.

| Protocol | Device | Q F20–35 | Q M20–35 | Ratio (95 % CI) |
|---|---|---|---|---|
| balanced90 | Apple | 0.0276 (n=189) | 0.00169 (n=593) | 0.061 (−0.0002, 0.127) |
| balanced90 | Garmin | 0.0283 (n=252) | 0.00107 (n=398) | 0.038 (−0.027, 0.108) |
| expanded90 | Apple | 0.0249 (n=230) | 0.00070 (n=676) | 0.027 (−0.033, 0.091) |
| expanded90 | Garmin | 0.0294 (n=255) | 0.00081 (n=406) | 0.028 (−0.034, 0.089) |

Less well-matched controls give the worst case: male 36–50 ratio 0.079–0.095
(CI excludes 0), male 51–75 0.048–0.134, postmenopausal women 0.144–0.165. The shared band
excess in middle-aged men is small but detectable (mean 0.0023–0.0027, SE 0.0006–0.0007); in young men it is not (balanced90 t = 2.0
Apple, 1.1 Garmin; both t < 1 in expanded90).

A lunar explanation of the female peak would therefore require a 16–37-fold sex × lunar
interaction. No literature supports one.

## 3. Lunar phase-lock

### 3.1 Direction test (T1)

Each window is regressed on (cos ψ, sin ψ) and (cos ν, sin ν), ψ the true Sun–Moon
elongation, ν the true anomaly. The coefficient angle is the phase at which HR peaks.
Rayleigh `R̄`, n = participants per stratum after taking one window each.

| Protocol | Stratum | n | R̄ ψ | R̄ ν |
|---|---|---|---|---|
| balanced90 | F20–35 | 441 | 0.057 (p=0.24) | 0.086 (p=0.038) |
| balanced90 | M20–35 | 991 | 0.081 (p=0.0014) | 0.152 (p=1.0e-10) |
| balanced90 | F36–50 | 1268 | 0.065 (p=0.005) | 0.035 (p=0.21) |
| balanced90 | M36–50 | 2382 | 0.057 (p=4e-4) | 0.100 (p=4e-11) |
| balanced90 | F51–75 | 1095 | 0.076 (p=0.002) | 0.122 (p=8e-8) |
| balanced90 | M51–75 | 2131 | 0.058 (p=7e-4) | 0.151 (p=1e-21) |

`expanded90` differs by ≤ 0.012 in every cell except F20–35 ψ (0.057 → 0.025). **The male negative control is the
most strongly positive cell in the table.** That alone makes the test uninterpretable as a
lunar test.

### 3.2 Specificity

The same statistic at non-lunar trial periods, same windows:

| Protocol | Stratum | Lunar ψ (29.53 d) | Sham 28.0 d | Sham 29.0 d |
|---|---|---|---|---|
| balanced90 | F20–35 | 0.057 | 0.089 | 0.036 |
| balanced90 | M20–35 | 0.081 | 0.133 | 0.079 |
| balanced90 | M36–50 | 0.057 | 0.122 | 0.079 |
| balanced90 | M51–75 | 0.058 | 0.171 | 0.105 |
| expanded90 | F20–35 | 0.025 | 0.097 | 0.036 |

A period of 28.0 d is *more* phase-coherent than the lunar period in every stratum shown
here; the one exception in the full spectrum is F36–50, whose peak is at 34.3 d (§4.2). The
lunar phase therefore carries no specific information about HR phase in this cohort.

The rotation null (`N_ROT = 32` random phase offsets per window) was computed and is
**reported but not used for inference**. It returns p = 0.03 almost everywhere, because a
per-participant rotation destroys any calendar-referenced common structure regardless of
its origin. It cannot separate lunar from non-lunar; the sham periods and the coherence
spectrum below do.

### 3.3 Acrophase test (T2) and weekday control (T3)

Free period on a 0.05-d grid, 25–35 d; acrophase mapped to ψ and ν; null from a uniform
acrophase inside each window (400 draws). No stratum survives: permutation p = 0.015–0.41
for ψ and 0.06–0.48 for ν (balanced90), similar in expanded90; with 12 tests, the smallest raw p
(0.015) does not survive any multiplicity correction. Median `T*` is 28.2–29.35 d
in women and 29.55–29.70 d in men — the expected winner's-curse spread, not a resolution of
the lunar period. Weekday of the acrophase: χ²(6) p = 0.25–0.84 in 11 of 12 cells; the
exception (M20–35, expanded90, p = 0.0045) is one of twelve tests and has no counterpart in
balanced90.

## 4. Cross-participant phase-coherence spectrum (the informative analysis)

For each window, the cosinor coefficient at trial period T has a direction. Averaging unit
vectors across participants gives `R̄(T)`. An endogenous rhythm with random phases gives no
coherence; an environmental driver imposes a common phase reference; a design artifact gives
broad-band coherence.

### 4.1 Null calibration

The identical pipeline (real date grid, real window geometry, baseline projection, 101
periods) on synthetic data, 2,507 windows:

| Null | R̄ median | R̄ max | Fraction of periods above the 95 % line |
|---|---|---|---|
| two-component AR(1), φ=0.70/0.99 | 0.015 | 0.031 | 0.000 |
| white, sd 8 bpm | 0.015 | 0.030 | 0.000 |

The pipeline generates no spurious coherence. Real-data exceedance is 0.19–1.00 across both
protocols (0.21–1.00 in expanded90).

### 4.2 Real spectrum

`R̄(T)`, expanded90, one window per participant, exact values at the periods quoted:

| Stratum | n | 27.5 | 28.0 | 28.5 | 29.0 | **29.53** | 30.0 | 33.5 | 34.5 | Peak (T) | 95 % line |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F20–35 | 485 | 0.114 | 0.097 | 0.068 | 0.036 | **0.023** | 0.037 | 0.062 | 0.065 | 0.120 (27.0) | 0.079 |
| F36–50 | 1413 | 0.038 | 0.042 | 0.052 | 0.048 | **0.062** | 0.061 | 0.062 | 0.081 | 0.082 (34.3) | 0.046 |
| F51–75 | 1227 | 0.111 | 0.142 | 0.124 | 0.096 | **0.068** | 0.049 | 0.093 | 0.115 | 0.142 (28.0) | 0.049 |
| M20–35 | 1082 | 0.139 | 0.122 | 0.088 | 0.080 | **0.080** | 0.070 | 0.127 | 0.105 | 0.140 (27.6) | 0.053 |
| M36–50 | 2600 | 0.099 | 0.126 | 0.116 | 0.091 | **0.068** | 0.054 | 0.085 | 0.105 | 0.126 (28.0) | 0.034 |
| M51–75 | 2322 | 0.137 | 0.156 | 0.134 | 0.100 | **0.060** | 0.035 | 0.082 | 0.112 | 0.157 (27.9) | 0.036 |

Band median 0.033–0.077. Strata are 20–35, 36–50 and 51–75 on the age at window end; the n
matches the phase-lock records. **No stratum peaks at the lunar period.** The 27.5–28.0 d lobe exceeds
the lunar value in five of six strata (exception F36–50, whose peak is at 34.3 d and where
29.53 d sits on a rising flank). The coherence is **equal or larger in men than in young
women**: 0.126 in men 36–50 (n = 2600) versus 0.097 in women 20–35 (n = 485, 95 % line
0.079). Young women show the weakest, not the strongest, 28-d coherence of any stratum.

### 4.3 Is the 28.0-d coherence absolute-time or window-relative?

| Quantity | Value |
|---|---|
| R̄(calendar-referenced phase at 28.0 d, pooled) | **0.1185** |
| R̄(window-relative phase at 28.0 d, pooled) | 0.0124 |
| 95 % line for random directions (n = 9,129) | 0.0181 |
| R̄(calendar-referenced phase at 29.53 d, pooled) | 0.0618 |

The coherence exists only in absolute time. The window-relative phase is consistent with
uniform (0.012 against a random-direction expectation of 0.011 and a 95 % line of 0.018), so
the signal is not a waveform tied to the window start or to the 30-day block construction.

### 4.4 Season split

| Start quarter (expanded90, pooled) | n | R̄(28.0 d) | Mean phase | R̄(29.53 d) | Mean phase |
|---|---|---|---|---|---|
| Jan–Mar | 1,865 | 0.111 | 91.9° | 0.046 | 68° |
| Apr–Jun | 1,025 | 0.065 | 106.5° | 0.041 | 193° |
| Jul–Sep | 2,992 | 0.168 | 83.9° | 0.153 | 303° |
| Oct–Dec | 3,247 | 0.095 | 88.8° | 0.054 | 288° |

At 28.0 d the phase is **stable across the year** (84–107°, within 23°). At 29.53 d it moves
from 68° to 193° to 303° to 288° across quarters — the signature of an annual component
leaking into the lunar period, not of a lunar rhythm.

## 5. Interpretation

**Established in this cohort.**
- The female band excess is not lunar, by two independent routes: the male bound (3–6 %) and
  phase specificity (28 d > 29.53 d in five of six strata, in both sexes).
- A sex- and age-independent, absolute-time, phase-coherent component exists in the
  27.5–28.0 d lobe (R̄ = 0.10–0.16 in five of six strata; pooled 0.119), with a season-stable
  phase (within 23°), plus a second lobe near 34 d. It is not produced by red or white noise,
  is not window-relative, and is not removed by the weekday dummies (1/28 d is a subharmonic
  of the week, not a harmonic; 4×7 = 28 d is a period, whereas the weekly harmonics are at
  k/7 c/d). Its coherent amplitude is R̄·A ≈ 0.04–0.12 bpm (0.10–0.12 in five of six strata;
  R̄² ≈ 1–2.4 % of power), so it is detectable only because n is large and it does not carry
  the band excess.
- In young women the 28-d amplitude excess over same-age men is the broad-band excess, not a
  28-d line: the F/M amplitude ratio is 1.41 at 27.0 d, 1.40 at 28.0 d, 1.38 at 29.0 d and
  1.17 at 34.0 d (expanded90, one window per participant). The female-specific share of
  fixed-period power, 1 − A²_M/A²_F, is 0.41–0.57 at 27–29 d and 0.16–0.40 at 34.0 d, and the
  excess adds amplitude without coherence (F20–35 R̄ = 0.097 at 28.0 d, below every male
  stratum).
- At the lunar period there is no coherence peak and the phase is season-dependent
  (68° → 193° → 303° → 288°), consistent with annual leakage rather than a lunar rhythm.

**Suggestive, not established.**
- The component is most plausibly a four-weekly human behavioral cycle (roster, pay period,
  four-weekly shift pattern) rather than an environmental oscillation: the lobe is centred at
  27.5–28.0 d, and no geophysical period is there (solar rotation 27.3 d, synodic month
  29.53 d). A 21/7 contraceptive cycle would be female-only and cannot explain the male
  component. Because 28 ≡ 0 (mod 7), any 28.0-d
  signal is automatically weekday-phase-locked, so that property neither supports nor
  refutes this reading.
- A coverage-mediated mechanism is not supported (§7): the raw coverage coherence at 28.0 d
  is 0.007 (night fraction) and 0.033 (total) against 0.119 for HR, and the implied HRp10
  slope is 44–390 bpm per unit coverage.

**Not established.**
- The origin of the 27.5–28.0 d component. It needs its own experiment before it is named.

## 6. Consequences for the 25–35 d claim

1. The claim can no longer be defended against an environmental explanation by adding lunar
   regressors: in this design those regressors remove the menstrual signal by construction.
   The defensible evidence is the male bound, phase specificity, and the seasonal control (§8).
2. The band contains a sex-independent 27.5–28.0 d coherent component, stronger in men than in
   young women in the coherence metric but only ≈1–2.4 % of the power. A replication must still
   show that the female excess is (a) not phase-coherent across women and (b) not localised at
   28 d. (a) is supported here (young women's R̄ at 28.0 d is 0.097, below every male stratum,
   while their amplitude is 1.40× men's); (b) is supported within 27–34 d by the flat female
   share at 27–29 d and out of band by check A (§7): the F/M ratio falls to 0.87–0.95 at
   18/48/55/60 d. A per-participant nuisance pair at 28.0/34.0 d cannot change `Q_i` materially
   and deletes the slices at which `T*` and the phase tests are evaluated; do not add one.
3. The menstrual interpretation remains viable but unproven: 28.2–29.35 d median `T*` in
   women versus 29.55–29.70 d in men is consistent with an endogenous female rhythm, but at
   156–179 d spans this is not a period estimate.

## 7. Post-hoc checks

Two threats left open by §5 were tested after the main analysis. Both pass.

**Coverage-mediated coherence (check B).** The coherence null (§4.1) used random Dirichlet
coverage fractions, so a four-weekly cycle in wear time was not excluded. `R̄(T)` of the
*raw* daily clock-coverage series, same windows and periods, is the upper bound on the
induced coherence: a memoryless nonlinear map creates harmonics (higher frequencies), not
subharmonics, and a lagged map shifts phase without changing the magnitude. At 28.0 d,
R̄ = 0.007 (night fraction, 00–06) and 0.033 (total, 0–24) pooled, against 0.119 for HR.
Per stratum the total fraction is 0.055 (F20–35), 0.065 (F36–50), 0.033 (F51–75), 0.043
(M20–35), 0.029 (M36–50) and 0.016 (M51–75) — the strata where HR coherence is strongest
(F51–75, M36–50, M51–75) have the weakest coverage coherence. The HRp10 slope needed to
generate the HR coherence is 44–50 bpm per unit total coverage and 390 bpm per unit night
coverage (pooled), against a plausible few bpm. Per stratum the required total-coverage
slope is 9–109 bpm per unit, and 31–51 in F20–35. With a generous 10 bpm limit the coverage
channel accounts for ≈5 % of the pooled coherent power (0.1 % for night; 11 % in F20–35 for
the total fraction). Coverage-mediated coherence is not supported. The bound is weakest in
F36–50, where the total-coverage `R̄(28.0) = 0.065` exceeds the HR value of 0.043 and the
required slope is 9 bpm per unit.

**Band-localisation (check A).** If young women had larger HR variability at all infradian
periods, the 25–35 d excess would be the top of a broad hump. The F/M amplitude ratio is
1.97–2.02 at 28.0 d, 1.28–1.41 at 34.0 d (the band edge), and 0.87–0.95 at 18, 48, 55 and
60 d (expanded90 and balanced90). Relative to the in-band value the out-of-band ratio is
0.44–0.48 (95 % CI excludes 1.0) and the 34-d value is 0.65–0.70. The excess is
band-localised, not a global variance scaling.

Both checks are in `placebo_bands.py` / `placebo_bands.csv` and
`coverage_coherence.py` / `coverage_coherence.csv`. Figures 1–4 reproduce the spectra and
the two checks from stored artefacts (`figures.py`; captions in `figures/CAPTIONS.md`).

## 8. Seasonal/trend control (S-only arm)

Addendum 1 left the S-only arm available but unrun (consequence 2). It was executed after the
main analysis with the pre-registered rule unchanged: the young-female excess is robust if
`Q_S / Q_ref ≥ 0.70`, with male FPR near 5 %. The block is τ², τ³ over the selected-date span.

| Protocol | Stratum | n | Q_ref | Q_S | Q_S/Q_ref (95 % CI) |
|---|---|---|---|---|---|
| balanced90 | F20–35 | 441 | 0.0280 | 0.0287 | **1.026 (0.990–1.064)** |
| balanced90 | F36–50 | 1268 | 0.0210 | 0.0218 | 1.038 (1.012–1.064) |
| balanced90 | F51–75 | 1095 | 0.0044 | 0.0039 | 0.886 (0.752–1.003) |
| balanced90 | M20–35 | 991 | 0.0014 | 0.0008 | 0.584 (−1.028–0.972)\* |
| balanced90 | M36–50 | 2382 | 0.0025 | 0.0022 | 0.895 (0.734–1.037) |
| balanced90 | M51–75 | 2131 | 0.0025 | 0.0021 | 0.847 (0.677–0.985) |
| expanded90 | F20–35 | 485 | 0.0273 | 0.0274 | **1.005 (0.971–1.040)** |
| expanded90 | F36–50 | 1413 | 0.0203 | 0.0208 | 1.025 (1.001–1.052) |
| expanded90 | F51–75 | 1227 | 0.0039 | 0.0032 | 0.833 (0.661–0.961) |
| expanded90 | M20–35 | 1082 | 0.0007 | 0.0001 | 0.150 (−10.41–7.62)\* |
| expanded90 | M36–50 | 2600 | 0.0022 | 0.0018 | 0.843 (0.666–0.991) |
| expanded90 | M51–75 | 2322 | 0.0025 | 0.0022 | 0.834 (0.676–0.959) |

\* `Q_male ≈ 0`, so the male ratio is not interpretable; the male diagnostic is the FPR.

Design recovery of a unit sinusoid under baseline + S: 0.973–0.974 at 28.0 d (2.6–2.7 %
absorbed) and 0.969 at 34.0 d (3.1 %), over 200 windows per protocol (range 0.88–1.00 across
windows). The expected ratio for a pure 28-d sinusoid is therefore ≈0.974; the observed
1.00–1.03 in F20–35 is at or above that, so no detectable part of the excess is a smooth
within-window trend. The small excess above 1 is a null-recalibration effect: Q is a signed
excess over a re-fitted AR(1) null, and removing the trend changes both the observed and the
null spectra.

The male ratio is not interpretable because `Q_male ≈ 0` (its CI spans −10 to +7); the male
check is the FPR, which falls from 4.6–5.7 % to **3.9–4.3 %**, i.e. the extension is if
anything slightly conservative. F51–75 loses 11–17 % of a small excess (0.004), still above
the 0.70 threshold. Engine determinism was verified before the run: the same engine with the
original 12 columns reproduces the saved `Q_i` and `p_value_discrete` bit-for-bit
(max |ΔQ| = 0, 18 windows), so the pairing is exact.

**Verdict: robust.** The young-female 25–35 d excess is not a seasonal arc or a smooth
within-window trend. This removes the last smooth low-frequency alternative; a non-smooth
in-band seasonal pattern (the 30.44-d annual harmonic) is not formally excluded.

## 9. Limitations

- Single window per participant (first `window_index`); participants with several windows are
  represented once. The all-window pass (15,074 windows) is not independent and is not used.
- 156–179 d spans give ΔT ≈ 4.4 d at 28 d. No statement about 28.5 vs 29.53 vs 30.4 d is
  possible, in either direction.
- `R̄` is unweighted: it measures phase agreement, not amplitude. With n = 485 the young-female
  95 % line is 0.079, so a 0.097 estimate is only 1.2× that line. A small coherent component
  and a large one give the same `R̄` if the phases agree.
- The coherence null used random coverage fractions. Check B (§7) bounds the coverage channel
  with the raw coverage series but does not model the HR–coverage map directly.
- S is a smooth polynomial. It excludes a seasonal arc or trend, not a non-smooth in-band
  seasonal pattern such as the 30.44-d 12th annual harmonic, which is a pre-specified in-band
  candidate, not a nuisance. The coherence spectrum has no peak at 30.44 d.
- Samsung (18 windows) is excluded.
- The rotation null is invalid for this question and is reported only for completeness.
- Endpoint definition: the excess is computed from channel-3000 HRp10. HRp5 and HRp1 (same
  bins, different percentiles) and a supplier daily RHR reproduce the female > male age
  ordering but attenuate the level; the fixed-period 26.67 d within-person change is
  endpoint-dependent. See `learnings/rhr_definition_sensitivity.md`.

## 10. Files

| File | Content |
|---|---|
| `phase_lock.py` | compute (T1/T2/T3, rotation null, sham periods, coverage spectrum) and analyze |
| `coherence.py` | `R̄(T)` summary per stratum |
| `null_coherence.py`, `null_coherence.csv` | calibrated null for the coherence spectrum |
| `male_bound.py`, `male_bound.csv`, `male_bound_strata.csv` | male-cohort bound |
| `placebo_bands.py`, `placebo_bands.csv` | out-of-band F/M amplitude ratio (check A) |
| `coverage_coherence.py`, `coverage_coherence.csv` | raw clock-coverage coherence (check B) |
| `seasonal_arm.py`, `seasonal_arm.csv`, `seasonal_design_recovery.csv` | S-only seasonal/trend arm (§8) |
| `figures.py`, `figures/` | periodogram figures 1–4 and captions |
| `regressors.py`, `validate.py` | lunar phase model, gate, recovery calibration |
| `artifacts/disambiguation/run_20261020/phase_records_*.parquet` | per-window results |
| `.../phase_coherence_*.npz` | per-window HR angles plus coverage angles/amplitudes, 101 periods |
| `.../phase_lock_summary.csv`, `coherence_summary.csv`, `coherence_spectrum.csv` | tables |
| `.../phase_lock_provenance.json` | frozen analysis choices |
| `../rhr_sensitivity/`; `learnings/rhr_definition_sensitivity.md` | HRp5/HRp1 percentile arms and supplier daily RHR sensitivity (separate runs) |
