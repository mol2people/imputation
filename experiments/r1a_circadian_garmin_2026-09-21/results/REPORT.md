# R1a — participant-level circadian curve (Garmin) — results

Run 2026-09-21T18:03:39 | implements [`../PLAN.md`](../PLAN.md) | 30 FS-phase repeats (same splits/seeds), paired; 2 workers, n_jobs=1 per fit.

**BASE gate:** max |delta AUROC| vs frozen FS A1 (s3 all) = 0.00e+00 across 30/30 repeats.

**Cache:** `epoch_hours.parquet` | cohort coverage 1.0000 (3848/3848).

## 1. Absolute AUROC (mean ± SD over 30 repeats)

| arm | test AUROC | val AUROC | n cols (hygiene) |
|---|---|---|---|
| `BASE` | 0.7434 ± 0.0155 | 0.7466 | 400 |
| `R1a_only` | 0.6921 ± 0.0200 | 0.6885 | 35 |
| `BASE_x_R1a` | 0.7463 ± 0.0179 | 0.7509 | 435 |
| `R1a_resid_only` | 0.6885 ± 0.0157 | 0.6868 | 35 |
| `BASE_x_R1a_resid` | 0.7470 ± 0.0173 | 0.7500 | 435 |

## 2. Paired test-AUROC deltas (97.5% t-CI)

| comparison | mean Δ | 97.5% t-CI | SD | share > 0 |
|---|---|---|---|---|
| BASE⊕R1a − BASE | +0.0029 | [+0.0004, +0.0054] | 0.0067 | 0.60 |
| BASE⊕R1a_resid − BASE | +0.0036 | [+0.0014, +0.0058] | 0.0058 | 0.70 |
| BASE⊕R1a − R1a_only | +0.0542 | [+0.0490, +0.0594] | 0.0140 | 1.00 |
| BASE⊕R1a_resid − R1a_resid_only | +0.0585 | [+0.0531, +0.0640] | 0.0146 | 1.00 |

## 3. Top features (BASE⊕R1a Gini, mean over 30)

| feature | Gini (mean) | R1a? |
|---|---|---|
| `rec__ch3000_tod_morning` | 0.0276 | no |
| `rec__ch3000_weekday_mean` | 0.0248 | no |
| `hour_h9` | 0.0157 | **yes** |
| `roll_rec__ch3001_daily_sd_w7_max` | 0.0144 | no |
| `win__ch3000_tod_morning` | 0.0141 | no |
| `rec__ch3000_weekend_minus_weekday` | 0.0137 | no |
| `rec__ch3000_minus_ch3002_mean` | 0.0137 | no |
| `cosinor_A3` | 0.0117 | **yes** |
| `rec__ch3000_minus_ch3001_mean` | 0.0107 | no |
| `win__ch3000_minus_ch3002_mean` | 0.0105 | no |
| `win__ch3000_weekend_minus_weekday` | 0.0103 | no |
| `roll_rec__ch3002_daily_sd_w7_max` | 0.0100 | no |
| `hour_h8` | 0.0097 | **yes** |
| `win__ch3000_minus_ch3001_mean` | 0.0089 | no |
| `roll_rec__ch3001_daily_sd_w30_max` | 0.0080 | no |

## 4. Univariate AUROCs (descriptive, pooled cohort)

| feature | AUROC |
|---|---|
| `cosinor_acro_h` | 0.5651 |
| `cosinor_A2` | 0.5324 |
| `curve_range` | 0.5307 |
| `cosinor_A1` | 0.5206 |
| `curve_day_night_contrast` | 0.5080 |
| `cosinor_resid_sd` | 0.4913 |
| `curve_morning_slope` | 0.4849 |
| `curve_entropy` | 0.4823 |
| `cosinor_A3` | 0.4315 |
| `hour_h20` | 0.4054 |
| `hour_h21` | 0.4015 |
| `hour_h19` | 0.3977 |
| `hour_h22` | 0.3938 |
| `hour_h23` | 0.3879 |
| `hour_h18` | 0.3832 |

## 5. Caveats

- Same-partition reuse — not independent validation.
- Curve features correlate with wear pattern by construction; `BASE⊕R1a_resid` is the deconfounded comparison.
- ch3001/ch3002 excluded (vendor caveat).
- Weekday/weekend contrast not derivable from the hour-of-day cache (lives in R1b, deferred).

## 6. Interpretation

**Primary family (both comparisons Bonferroni-simultaneous 95%, i.e. 97.5% t-CIs):
both clear zero.** The wear-partialled curve is the stronger and tighter
increment: `BASE⊕R1a_resid − BASE = +0.0036 [+0.0014, +0.0058]` (share > 0 in
70% of repeats) vs `+0.0029 [+0.0004, +0.0054]` raw. Magnitude is at the
detection floor (~0.003 at R=30) — the same order as the age counter
(+0.0029) — but resolved, and it is the first *feature-side* lever to clear
the floor since the FS-phase saturation point.

**Standalone: 35 curve features alone reach 0.6921** vs 0.7434 for the full
400-col matrix — 79% of the participant-level excess signal
((0.6921−0.5)/(0.7434−0.5)) from the hour-of-day profile alone. This is not
independent of the existing matrix (tod/coverage features share variance by
construction) but it establishes the curve as one of the densest single
feature families in the program — for free, from an existing cache, no
rescan.

**The deconflation result is the scientifically interesting one.** Adding
raw curve features to BASE yields little (+0.0029) because the existing
tod-share/coverage block already absorbs their wear-pattern component.
Linearly partialling the 24 hour-of-day coverage shares (plus log total)
*out* of each curve feature before adding it to BASE gives a *larger and
tighter* increment (+0.0036, CI lower bound +0.0014 vs +0.0004). The
increment is therefore curve *shape* — relative timing and form of the
hourly profile — not coverage. Operationally: coverage-partialling is a
cheap denoiser worth carrying into R1b and any day-curve model.

**What the shape says (label: 1 = recorded-female salutation, 64%
prevalence):**

| feature | class 0 (male) | class 1 (female) |
|---|---|---|
| `cosinor_acro_h` (peak time) | 13.76 h | 14.53 h (+0.77 h later) |
| `curve_night_mean` (00–05) | 64.8 bpm | 60.8 bpm |
| `hour_h9` | 79.6 bpm | 75.1 bpm |
| `hour_h20` | 77.6 bpm | 75.1 bpm |
| `cosinor_A3` (8h amplitude) | 1.77 | 1.57 |

Best univariate is acrophase (0.565); evening hours h18–h23 univariate
0.38–0.41 (≈0.59–0.62 reversed) are the strongest single cells; `hour_h9`
and `hour_h8` enter the combined model's top-15 Gini outright. The pattern:
**male curves run hotter at essentially every clock hour and peak earlier;
female curves peak ~46 min later at lower absolute levels with similar
day–night contrast.** Note this direction is *opposite* to laboratory
resting-physiology sex differences (women: higher resting HR, earlier
circadian phase). The discriminative signal is therefore unlikely to be
textbook circadian dimorphism; more plausibly it is mediated by activity
load, cohort composition (age/fitness/BMI differ by salutation class), and
wear behaviour — consistent with the program-wide wear-pattern finding.
The label is recorded salutation, not biological sex/gender.

**Positioning.** This is the *trait* complement to the dayscale experiment's
*state* result: participant-level curve shape (this experiment, +0.0036
beyond summaries) and within-person day-level deviations (dayscale arm B)
are separate levers on the same problem. Both are small at the participant
level — the day level has more headroom (day-table base ≈0.65 vs 0.743
here), which is where R1b (per-day curves, weekday/weekend contrast,
minute-scale dynamics, episodes) should be evaluated first. The R1b rescan
also produces the per-day curve cache that doubles as the D2 warm-start
input.

**Caveats beyond §5.** Secondary comparisons in §2 are not family-wise
corrected; the effect sits at the detection floor, so a modest
specification change could move it across zero; the residualisation is
linear only (the RF on BASE can partial nonlinearly, so the raw-vs-resid
gap is a lower bound on the wear-confound role); n_total differs by class
(female 557k vs male 535k minute-observations), so coverage-partialling is
not just a denoiser but also a mild class-balancer on the wear axis.
