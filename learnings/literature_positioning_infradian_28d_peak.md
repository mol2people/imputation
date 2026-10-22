# Positioning the ~28.5-day infradian heart-rate peak in the current literature

**Reviewer memo — external positioning of the infradian HR-peak finding, written 2026-10-20.**
This memo reviews the `female_infradian_cycles` / `female_infradian_cycles_v2` line of work as a PhD-advisor would review a student's report: what is established by others, what is new here, what is over-claimed, and what must change before the finding is publishable. It complements the internal [results audit](../reports/results_audit_2026-10-14.md) and the [results reference](../reports/results_reference.md).

---

## 1. The claim, sharpened

The student's claim is not one thing; it is a stack of four sub-claims that must be evaluated separately:

| Sub-claim | Status |
|---|---|
| A cohort-averaged Lomb-Scargle argmax at **T\* = 28.5 d** (search band 25–35 d) in daily `hr_p10` (a resting-HR proxy), Apple Watch, young recorded-females (N = 502). | **Exploratory, unvalidated as stated.** The argmax over a 21-period grid followed by a fixed-period F-tail is the canonical "post-hoc period selection" error. The raw LS peak is a *descriptive* cohort statistic, not evidence of a mechanism. |
| The finding survives a calibrated null: per-subject continuous AR(1) surrogate (B = 199) applied to the **max-over-grid** statistic; male false-positive rate ≈ 5% (well-calibrated negative control); HC3 Wald group contrast ΔE = +0.0642, p_Holm < 1e-16. | **The strongest part of the work.** This is the correct Baluev/VanderPlas-style calibration, done per-subject on gappy wearable data, with a negative-control FPR check. Most wearable-HRV spectral studies in the literature do not do this. |
| Cross-device replication (Garmin peak ≈ 27.7 d, same age/sex pattern). | **A weak control, not a replication.** Apple→Garmin share PPG modality, the same "resting = lowest/sleep-anchored window" construction, the same wear regime, and the same environment. This rules out an Apple-specific sensor artifact and nothing else. A genuinely independent replication requires a different modality (chest-strap ECG), a different population, or a mechanistic anchor (phase-lock to menses). |
| "The 28.5-day LS peak **is** the menstrual cycle." | **Not yet defensible.** The 25–35-day band is saturated with competing periods: lunar synodic (29.53 d), lunar anomalistic (27.55 d), 4× the circaseptan week (4×7 = 28 d), the annual harmonic (365/12 ≈ 30.4 d), and the menstrual cycle itself (population median 29.3 d, Bull 2019; only 12.4% exactly 28 d). A single argmax at 28.5 d does not identify which of these generated the peak. |

**Verdict up front.** The finding is real, internally consistent, and methodologically better than the field norm — but the claim as framed is one step ahead of the evidence. What is established is: *a sex- and age-specific excess of 25–35-day power in young recorded-females' continuous resting HR, surviving a per-subject red-noise null, attenuating across the lifespan, replicating across devices, and carrying incremental predictive signal.* What is not established is: that this specific peak is the menstrual cycle rather than a lunar/circaseptan alias or a shared endogenous rhythm. The single highest-value missing experiment is **phase-locking the peak to recorded menses** in the women who have them.

---

## 2. What is established (the student's direction is the known one)

The luteal-phase RHR elevation is one of the most replicated findings in wearable physiology:

- **Goodale et al. 2019** (Ava, n = 193, 708 cycles, LH-confirmed): RHR +2.0/+2.5 bpm early/late luteal vs. menses; −1.5 bpm follicular vs. menses — a ~4 bpm peak-to-trough swing.
- **Sims et al. 2021** (WHOOP, n = 4,594, 13,535 cycles): RHR and RR-interval rise follicular → luteal (P < 0.001), with HRV ~70 ms → <64 ms across cycle quartiles.
- **Jasinski et al. 2024** (WHOOP, n = 11,590, 45,811 cycles): population RHR offset −1.83 bpm at cycle day 5 → +1.64 bpm at day 26 (~3.5 bpm peak-to-trough); mean individual RHR amplitude **2.73 bpm, 93.6% positive**; RMSSD offset +3.57 ms (day 5) → −3.22 ms (day 27), amplitude 4.65 ms, 80.6% positive. Amplitude linked to age and BMI.
- **Alzueta et al. 2022** (Oura, n = 26, LH-confirmed): nocturnal HR +2.38 bpm (95% CI 0.98–3.78) mid-luteal vs. menses; +2.49 bpm late-luteal; rMSSD −5.5/−6.0 ms luteal vs. menses.
- **Alzueta et al. 2024** (Oura, n = 116): HRV modulation **only in young** group, not midlife.
- **de Jager et al. 2026** (*Sports Med*, living systematic Review, 16 studies, n ≈ 19,322 women): time-domain HRV 3–9% **lower in the latter half of the cycle** in naturally cycling women; lower in OC users; declines after menopause. This is the HRV companion to the RHR finding.
- **Mechanism:** progesterone thermogenesis — luteal core temperature +0.3–0.56 °C → HR +7–10%/°C → luteal RHR +1.5–2.5 bpm (Goodale 2019; Alzueta 2024; Alzueta 2022). Estrogen opposes this (ovulatory temperature dip −0.093 °C, P = 0.05).

So the student's *direction* (HR higher in luteal, young women, attenuating with age) is the known one. The 28.5-day argmax is a *spectral confirmation* of an established phase-contrast phenomenon — not a new biological fact.

**What IS new here (three genuine contributions):**

1. **A sharper sex/age specificity than any prior report.** De Silva et al. 2026 found 25–35-day rhythms in 48.9% of menstruating women, 27.3% of menopausal women, and 16.7% of men — i.e., **not menstrual-specific, and present in men**. The student reports it as absent in elderly and males. This is stronger than anything in the prior literature, and the calibration is why.
2. **A methodological advance**: per-subject continuous-time AR(1) surrogate null applied to the max-over-21-periods statistic, with the male negative-control FPR checked (≈5%). Most prior wearable-HRV spectral studies use either no null, a fixed-period null, or wavelet red-noise backgrounds that leave residual weekly/monthly contamination.
3. **The first demonstration that the peak carries incremental downstream predictive utility** (+0.0109 AUROC, p < 0.01, Bonferroni-corrected CI excludes zero; replicated on Garmin at +0.006 AUROC; BMI-invariant).

---

## 3. What is NOT established (the student's burden)

The burden is: **the 25–35-day band contains several candidate periods, and at these record lengths they cannot be separated spectrally.** This section now reports measurements rather than expectations (experiment: `experiments/infradian_age_dependency/disambiguation`, 2026-10-20).

| Candidate period | Source | Separable at 180 d? |
|---|---|---|
| ~28.0 d | 4× the circaseptan (7-d) week | redundant with weekday dummies (exact, 5.6e-12 residual) |
| ~28.5 d | The student's LS argmax (Apple) | — |
| ~27.7 d | Garmin replication peak | no (same resolution cell) |
| 29.53 d | Lunar synodic month | no (ΔT resolution 4.4 d at 180 d, 2.9 d at 270 d) |
| 27.55 d | Lunar anomalistic month | no |
| ~29.3 d | Population menstrual median (Bull 2019) | no |
| ~30.4 d | Annual harmonic (365/12) | no |

**Resolution.** The Rayleigh resolution in period at T ≈ 28 d is ΔT ≈ T²/T_span = 4.4 d (180-d windows) and 2.9 d (270-d windows). Separating 28.5 d from 29.53 d by any method requires a span of 1/Δf = 820 d. The caches do not provide it: Apple median span 400 d (max 892 d, median 147 qualified days, n = 25); Garmin median 682 d (max 905 d, median 497 qualified days, n = 15). **Period identification at 28.5 vs 29.53 is not possible in this repository.**

### 3.1 The lunar regression is a band-stop, not an identification

Measured recovery of a unit sinusoid after projection on the extended design (real 90-date windows, 40 per protocol):

| Period (d) | Seasonal block only | Lunar + seasonal |
|---|---|---|
| 23.0 | 0.978 | 0.925 |
| 27.5546 | 0.975 | 0.003 |
| 28.0 | — | 0.005 |
| 28.5 | 0.977 | 0.008 |
| 29.5306 | 0.973 | 0.003 |
| 36.0 | 0.964 | 0.879 |

Two bracketing lunar frequencies synthesize any intermediate frequency over a short window; the collinearity is a sinc function of Δf·T. Consequence: **a collapse of the female excess after lunar regression cannot be read as "environmental"** — it is guaranteed by the design, and would occur equally for a genuine menstrual rhythm. The seasonal block, by contrast, is nearly orthogonal to the band (2.3–2.7 % absorption) and is the only cleanly identified regression control.

**It was run** (after the main analysis, `seasonal_arm.py`). In F20–35 the band excess is 1.026 × the baseline-only value under balanced90 (95 % CI 0.990–1.064) and 1.005 × under expanded90 (0.971–1.040), against a pre-registered robustness threshold of 0.70. Design absorption of a unit 28-d sinusoid is 2.6–2.7 % (recovery 0.973–0.974 on real windows), so the control is informative rather than nominal. Male FPR stays at 3.9–4.3 %. The excess is not a seasonal arc or a smooth trend.

### 3.2 The circaseptan confound is already absorbed

At integer-day resolution every sinusoid at an integer harmonic of 1/7 is exactly 7-periodic and lies in the span of the six weekday dummies already in the baseline. Verified: the relative residual norm of the 7-d, 3.5-d, 2.33-d, and 1.75-d harmonics after projection on the weekday space is 5.6e-12. A separate circaseptan regressor is linearly redundant. The earlier "4×7 = 28 d alias" concern is not supported: 1/28 d⁻¹ is the fourth *subharmonic* of the weekly frequency 1/7 d⁻¹, not a harmonic of it, so a 28.0-d sinusoid is outside the weekday span. (4×7 = 28 d is a period; the weekly harmonics are at k/7 d⁻¹.)

### 3.3 The male-cohort bound

A sex-independent lunar component would appear in the calibrated male band excess at the same absolute magnitude as in women. Under no sex × lunar interaction, the lunar-attributable share of the young-female excess is bounded by Q_male / Q_young-female (saved HRp10 endpoints, one window per participant, participant bootstrap, 4,000 draws):

| Reference group | Apple balanced90 | Apple expanded90 | Garmin balanced90 | Garmin expanded90 |
|---|---|---|---|---|
| male 20–35 | 0.061 [−0.0002, 0.127] | 0.027 [−0.033, 0.091] | 0.038 [−0.027, 0.108] | 0.028 [−0.034, 0.089] |
| male 36–50 | 0.084 [0.042, 0.133] | 0.083 [0.043, 0.131] | 0.095 [0.049, 0.147] | 0.079 [0.035, 0.126] |
| male 51–75 | 0.124 [0.074, 0.185] | 0.134 [0.082, 0.197] | 0.048 [0.003, 0.098] | 0.058 [0.015, 0.104] |
| female 51–75 | 0.153 [0.090, 0.227] | 0.144 [0.079, 0.219] | 0.165 [0.087, 0.254] | 0.151 [0.076, 0.235] |

The best-matched control (male 20–35) bounds the shared environmental share at **3–6 %** of the young-female excess, with a CI that includes zero on both devices in the expanded protocol. A lunar explanation of the female peak therefore requires a sex × lunar interaction of roughly 20-fold — implausible a priori and with no supporting literature.

### 3.4 Lunar phase-lock: run, and negative for non-specificity

Two directed tests were run on both protocols (one window per participant, 156–179 d spans).

- **Direction test.** Regress the baseline residual on (cos ψ, sin ψ) and on (cos ν, sin ν), ψ the true Sun–Moon elongation, ν the true anomaly. The coefficient angle is the lunar phase at which HR peaks. Under lunar forcing it concentrates across participants; under an endogenous rhythm it is uniform.
- Result: R̄(ψ) = 0.057–0.081, R̄(ν) = 0.086–0.152 across strata, nominally significant almost everywhere — **including the male negative control, which is the most positive cell (M20–35, R̄ = 0.152, p = 1e-10)**.
- **Specificity.** The identical statistic at a sham 28.0 d period is *higher* than at the lunar period in every stratum (0.089–0.171 vs 0.057–0.081), and the acrophase test is null (permutation p = 0.10–0.66). A per-participant rotation null was computed and discarded: it destroys any calendar-referenced common structure, lunar or not.

**Conclusion: lunar phase carries no specific information about HR phase in this cohort.** The concentration the naive test finds is a shared component centred at 27.5–28.0 d (with a second lobe near 34 d), not at 29.53 d.

### 3.5 A newly identified confound: a 27.5–28.0 d component, not lunar

The cross-participant phase-coherence spectrum R̄(T) over 25–35 d, calibrated on red and white noise with the real date grid (2,507 synthetic windows; the null never exceeds the 95 % line, exceedance 0.000):

- Real data, one window per participant: exceedance 0.21–1.00; peaks at 27.0–28.0 d and 34.3 d. **No stratum peaks at the lunar period.** R̄ at 29.53 d is 0.023–0.080 and is below the 27.5–28.0 d lobe in five of six strata (exception F36–50, whose peak is at 34.3 d).
- At 28.0 d the mean phase is **stable across start quarters** (84–107°); at 29.53 d it moves 68° → 193° → 303° → 288° across quarters — annual leakage, not a lunar rhythm.
- At 28.0 d, R̄ = 0.119 in absolute time versus 0.012 in window-relative time (random-direction expectation 0.011, 95 % line 0.018, n = 9,129): the component is calendar-referenced, not a window-geometry artifact.
- It is present in men at the same or greater strength as in young women (M36–50 0.126 at n = 2,600 versus F20–35 0.097 at n = 485, 95 % line 0.079).
- Its energetic weight is small: fixed-period amplitude is 0.76–1.24 bpm, so the coherent amplitude R̄·A is ≈ 0.04–0.12 bpm (0.10–0.12 in five of six strata) and the coherent power fraction R̄² ≈ 1–2.4 %. It is a confound for phase-based tests, not for the band-excess statistic.
- The young-female amplitude excess is the broad-band excess, not a 28-d line: the F/M amplitude ratio is 1.41 at 27.0 d, 1.40 at 28.0 d, 1.38 at 29.0 d and 1.17 at 34.0 d; the female share of fixed-period power is 0.41–0.57 at 27–29 d and 0.16–0.40 at 34.0 d. The 34-d lobe is not the annual 11th harmonic: its phase is season-stable (195°, 201°, 183°, 198° across start quarters), unlike 29.53 d.

A sex- and age-independent, absolute-time coherent component at 27.5–28.0 d (with a second lobe near 34 d) therefore exists in the same band as the female excess. Two readings are live: a four-weekly behavioural cycle (28 d is exactly four weeks; no geophysical period is there), or coverage-mediated measurement. The coverage reading is now bounded: the raw daily clock-coverage series has R̄(28) = 0.007 (night, 00–06) and 0.033 (total) against 0.119 for HR, and the implied HRp10 slope is 44–390 bpm per unit coverage, far above a plausible few bpm. The behavioural reading remains the leading one but is unproven. Either way, **the band contains a ~28-d coherent component that the weekday dummies do not remove** — 1/28 d is a subharmonic of the week, not a harmonic, so it is outside the span of the six weekday indicators — though at ≈1–2.4 % of power it does not explain the excess.


---

## 4. The amplitude question (a reviewer's first question)

The established luteal-vs-follicular RHR swing is **~2.5–3.9 bpm peak-to-trough** (Goodale 2019: 2.0–2.5 bpm; Alzueta 2022: +2.38/+2.49 bpm vs. menses; Jasinski 2024: mean individual amplitude 2.73 bpm, 93.6% positive). The student reports **0.415 bpm** — roughly **5–6× smaller**.

This is *defensible* on three grounds:
- `hr_p10` is a **compressed** statistic (10th percentile of daily HR distribution), not the mean — it under-represents the swing relative to mean-RHR analyses.
- The 0.415 bpm is a **first-harmonic** amplitude of a near-square-wave cycle waveform — the first harmonic of a square wave of amplitude A is (4/π)·A ≈ 1.27·A, not A itself.
- Inter-cycle period variance (SD 2–4 d in WHOOP data) smears power across the 25–35-day band, reducing the argmax amplitude.

But a reviewer will ask: **why is the p10 amplitude 5× smaller than the mean-RHR amplitude in the same cohort?** The student should show the mean-RHR first-harmonic amplitude alongside the p10 amplitude, or at minimum state the compression argument explicitly.

---

## 5. How to reframe the claim so it is defensible

**Defensible (current evidence supports this):**
> "Young recorded-females show a sex- and age-specific excess of 25–35-day power in continuous resting heart rate that survives per-subject red-noise calibration, attenuates across the lifespan, replicates across two device ecosystems, and carries incremental predictive signal. This is consistent with — and provides a spectral characterization of — the established luteal-phase resting-HR elevation."

**Not yet defensible:**
> "The 28.5-day peak is the menstrual cycle."

The distinction is between a *spectral confirmation of a known phenomenon* and a *mechanistic identification*. The student has done the former. The latter requires the disambiguation experiments in §3.

---

## 6. What would change the verdict

The single highest-value missing experiment: **phase-lock the 28.5-day rhythm to recorded menses.** In the women who have menstrual-cycle logs (if available in the cohort), compute the acrophase of the 28.5-day cosinor relative to menstrual onset, and test with a Rayleigh/circular-uniformity test. If the phase is locked to onset (Rayleigh P < 0.05, after correction for the number of cycles tested), the claim upgrades from "consistent with the menstrual cycle" to "phase-locked to menses, strongly implicating the menstrual cycle as the driver." If not, the finding is an endogenous ~28.5-day rhythm of unknown but probably non-reproductive origin — a different (and arguably more interesting) paper.

---

## 7. Literature positioning summary (what a reviewer must know)

| Thread | Status | Key references |
|---|---|---|
| Luteal-phase RHR/HRV elevation | **Established** (multiple independent wearable cohorts, LH-confirmed) | Goodale 2019; Sims 2021; Alzueta 2022, 2024; Jasinski 2024; de Jager 2026 |
| HRV lower in luteal phase | **Established** | de Jager 2026 (living systematic review, 16 studies) |
| Age attenuation of cycle-amplitude | **Established** | Jasinski 2024 (β = −0.04 bpm/yr); Alzueta 2024; Gombert-Labedens 2026 |
| ~28.5-day *spectral peak* in continuous HR | **New — first report** | This work; no prior spectral-peak study on daily RHR exists |
| Per-subject AR(1) surrogate null on max-over-grid LS | **Methodological advance** | Baluev 2008; Frescura 2008; VanderPlas 2018 — the astronomy/period-detection literature, applied here for the first time to wearable RHR |
| Lunar effects on human physiology | **Weak-to-null** when controlled | Multiple negative studies; see Casiraghi 2021, Hocking 2024 for recent null results with adequate power |
| Circaseptan (7-day) rhythms in HR/BP | **Established as exogenous** (weekly social calendar); endogenous origin debated | Halberg et al. 1965–2005; the transplant-rejection ~7-day periodicity literature |
| Non-menstrual ~28-day rhythms in men | **Reported, debated** | De Silva 2026 (16.7% of men); Halberg's circatrigintan literature; confirmed here by the negative-control calibration |
| Menstrual cycle length distribution | **Established** | Bull 2019 (n > 600,000 cycles, Flo app): median 29.3 d, only 12.4% exactly 28 d |

**The one prior work that must be cited and engaged:** De Silva, Stirling, Naim-Feil, Puri, Paratz, Karoly. *Infradian rhythms of human heart rate reveal distinct multiday cycle periods in healthy adults.* **eBioMedicine 2026;131:106444** (preprint: bioRxiv 2026.03.01.708870). Wavelet spectral analysis on daily RHR, n = 523 + 70. Four infradian "chronotypes" in the 4–45-day band. **Found 25–35-day rhythms in 48.9% of menstruating women, 27.3% of menopausal women, 16.7% of men** — i.e., the band is *not* menstrual-specific and *is present in men*. Monthly rhythms **collapsed after regressing out lunar and seasonal effects** (monthly prevalence ~12% → ~4%). Menses synchronized to the HR rhythm (SI median 0.75, n = 28 with ≥5 cycles).

**The student's work is sharper than De Silva's on specificity** (per-subject AR(1) null vs. wavelet red-noise background; the male FPR ≈ 5% check is exactly the negative-control validation De Silva lacks). But De Silva did the **lunar/seasonal regression** that the student has not yet done. The two papers are complementary; both must be cited; the student's disambiguation experiments (§3) would close the gap.

---

## 8. Ranked list of what to do next (for the student)

1. **Do not add lunar regressors.** They absorb 99.2–99.7 % of any 27.5–30.5 d sinusoid at these window lengths (§3.1) and would remove a genuine menstrual signal; separation of 28.5 from 29.53 d needs 820-d spans and the windows span 156–179 d. The controls that actually carry information are the male-cohort bound (§3.3), phase specificity (§3.4), and the S-only seasonal control (§3.1, now run: ratio 1.01–1.03).
2. **Characterise the 27.5–28.0 d shared component (§3.5).** It is sex-independent, absolute-time, season-stable, and not reproduced by red or white noise. Until it is explained, the 25–35 d band is not clean. The coverage test has been run: R̄(28) = 0.007 (night) and 0.033 (total) against 0.119 for HR, with an implied slope of 44–390 bpm per unit coverage, so coverage is disfavoured and a four-weekly behavioural cycle is the leading reading.
3. **Add the mean-RHR amplitude alongside the p10 amplitude** (§4) to reconcile with the established ~2.7-bpm luteal swing.
4. **If menstrual-cycle logs exist for any sub-cohort: do the phase-lock test** against cycle onset (§6). This is the mechanistic anchor that would upgrade the claim from "consistent with" to "identified as", and it is the only test that can.
5. **Report the result under a second null** — a phase-randomized (AAFT) or fitted 1/f null — to address the AR(1)-vs-1/f model-mismatch concern. A skeptical reviewer will ask; a result that survives both nulls is the one that matters.
6. **Soften the cross-device claim** from "replication" to "rules out Apple-specific sensor artifact." A genuinely independent replication requires a different modality or a mechanistic anchor.
7. **Raise B to ≥1,000** for any per-subject p-value claims, or report only the calibrated detection *rate* (the male ≈ 5% FPR is already the right statistic to feature).

If the student does (1)–(4), the finding moves from "an interesting spectral peak with good calibration" to "a disambiguated, mechanistically anchored infradian rhythm in continuous wearable HR — the first spectral-peak demonstration of the menstrual cycle's autonomic signature in free-living wearable data." That is a genuinely publishable, novel contribution.

---

*Prepared 2026-10-20. Literature searched via web (2020–2026 peer-reviewed + preprints). Some 2026 references are preprint-stage; verify DOIs before final citation. WHOOP-derived numbers (Jasinski 2024, Sims 2021, Gonzalez 2026) are company-held and not independently replicable; treat as vendor-reported.*
