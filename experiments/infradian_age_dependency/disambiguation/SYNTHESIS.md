# Sintesi: infradian disambiguation

One-page synthesis of `REPORT.md`. Numbers are from the run of 2026-10-20
(`artifacts/disambiguation/run_20261020/`), one window per participant, unless stated.

## Question and design

Does the 25–35 d HR excess in women 20–35 have a lunar or seasonal driver, or is it
reproductive? Outcome: daily HRp10. Windows: 180 d = six 30 d blocks, 90 qualified days
(expanded90: ≥10/block, greedy to 90; balanced90: 15/block), spans 156–179 d. Protocols:
balanced90 and expanded90; devices Apple (6,955 / 8,252 windows) and Garmin (6,635 / 6,812). Statistic: `Q_i`, band excess over a calibrated per-subject null.
Identification rests on a male negative control, phase specificity at non-lunar trial periods,
and the cross-participant coherence spectrum; the naive lunar direction test is not used.

## Results

**1. The female excess is real, age-dependent, and closes at the menopause band.**

| Stratum | `Q` (Apple) | `Q` (Garmin) |
|---|---|---|
| F20–35 | 0.0249–0.0276 | 0.0283–0.0294 |
| F36–50 | 0.0180–0.0185 | 0.0229–0.0235 |
| F51–75 | 0.0036–0.0042 | 0.0044–0.0047 |

**2. Male-cohort bound.** Young male `Q` is 0.0007–0.0017; the ratio to the young-female
excess is 0.027–0.061 (95 % CI includes zero on both devices, expanded90). Middle-aged men
carry a small, detectable shared excess (ratio 0.079–0.095, CI excludes zero); postmenopausal
women 0.144–0.165. A lunar explanation would need a 16–37× sex × lunar interaction.

**3. Lunar tests fail specificity.** The male negative control is the most strongly positive
cell of the direction test (anomalistic R̄ 0.152 in M20–35 vs 0.086 in F20–35, p = 1e-10).
A sham 28.0 d period is more coherent than the lunar period in every stratum (0.089–0.171 vs
0.057–0.081). The acrophase and weekday tests are null after multiplicity. The rotation null
is invalid for specificity (it destroys any calendar-referenced structure) and is not used.

**4. A small absolute-time coherent component exists at ~28 d and ~34 d.** R̄(T) peaks at
27.5–28.0 d (0.10–0.16 in five of six strata) with a second lobe at 33.5–34.5 d (five of six
strata); no stratum peaks at 29.53 d (0.023–0.080). It is calendar-referenced, not
window-relative (0.119 vs 0.012; 95 % line 0.018). The 28-d phase is season-stable (84–107°);
the lunar phase rotates (68° → 193° → 303° → 288°), the signature of annual leakage.

**5. Energetic weight and age decomposition (new).** Coherent amplitude is R̄·A ≈ 0.04–0.12 bpm
(0.10–0.12 in five of six strata), i.e. R̄² ≈ 1–2.4 % of the 28-d power: the component is a
confound for phase tests, not for `Q`.
The young-female amplitude excess is broad-band, not a 28-d line: the F/M amplitude ratio is
1.41 at 27.0 d, 1.40 at 28.0 d, 1.38 at 29.0 d and 1.17 at 34.0 d; the female share of
fixed-period power, 1 − A²_M/A²_F, is 0.41–0.57 at 27–29 d and 0.16–0.40 at 34.0 d. Young
women's 28-d coherence is the *lowest* of any stratum (0.097), so the excess adds amplitude
without phase.

## Post-hoc checks

**A. Band-localisation.** The F/M amplitude ratio is 1.97–2.02 at 28 d and 0.87–0.95 at
18/48/55/60 d; relative to the in-band value the out-of-band ratio is 0.44–0.48 (95 % CI
excludes 1). The excess is band-localised, not a global variance scaling.

**B. Coverage.** Raw clock-coverage coherence at 28 d is 0.007 (night, 00–06) and 0.033
(total) against 0.119 for HR; the implied HRp10 slope is 44–390 bpm per unit coverage.
Coverage does not explain the coherent component.

**C. Seasonal/trend.** The pre-registered S-only block (τ², τ³) leaves the young-female
excess intact: Q_S/Q_ref = 1.03 [0.99, 1.06] (balanced90) and 1.01 [0.97, 1.04] (expanded90),
against a 0.70 robustness threshold. The design absorbs 2.6–2.7 % of a unit 28-d sinusoid;
male FPR stays at 3.9–4.3 %. Not a smooth trend or seasonal arc.

## Verdicts

**Established.** The 25–35 d excess in young women is real and age-dependent. It is not lunar:
the male bound is 3–6 %, and the lunar period carries no coherence peak and no stable phase.
The excess is band-localised (out-of-band F/M ratio 0.87–0.95), not coverage-mediated
(raw coverage R̄(28) ≤ 0.033), and not a seasonal arc or smooth trend (Q_S/Q_ref 1.01–1.03). A sex- and age-independent, absolute-time coherent component
exists at 27.5–28.0 d and near 34 d; its coherent amplitude is ~0.1 bpm. The female excess is
not that component: it is broad-band 27–32 d and phase-incoherent.

**Suggestive.** The 28-d coherent component is a four-weekly behavioural cycle; the coverage
reading is now disfavoured (check B). The 34-d lobe is a second absolute-time cycle, not the
annual 11th harmonic (33.2 d): its phase is season-stable (195°, 201°, 183°, 198° across
start quarters).

**Not established.** The origin of the 28/34-d coherent component. The menstrual attribution
of the female excess: no cycle logs exist, so no phase-lock to cycle onset is possible. The
28.5 vs 29.53 d distinction: 156–179 d spans give ΔT ≈ 4.4 d at 28 d; separating 28.5 from
29.53 d needs ~820 d.

**Speculative.** That the female excess is the menstrual cycle. The age gradient (closing at
51–75) and the incoherent phase are consistent with it; neither proves it.

## Limits

- One window per participant; the all-window pass (15,074) is not independent and is discarded.
- The coherence null used random Dirichlet coverage fractions. Check B bounds the raw
  coverage channel but does not model the HR–coverage map directly.
- The out-of-band periods (18/48/55/60 d) are >= 1.3 spectral-kernel widths from the band
  edges; a leak from the 35-d edge into 48 d is possible but small.
- Inference is a difference-in-differences in power plus phase specificity, not a lunar
  phase-lock; the latter is non-identified in this design.

## Consequences

1. Do not add lunar or 28-d nuisance columns: they absorb a narrow rhythm in 27.5–30.5 d
   almost completely (99.2–99.7 % measured) and cannot change `Q` materially, while deleting
   the frequency slices at which `T*` and the phase tests are evaluated.
2. If a cleaned per-participant endpoint is needed, use a two-level fit: fixed common vector
   on the 28/34-d pair plus participant random deviation; subtract the common part only.
3. Remaining work: cycle logs, the only mechanistic test. The seasonal/trend and coverage
   alternatives are now closed; the 30.44-d annual-harmonic candidate is not formally excluded
   at these spans (in-band, 4.4-d resolution).
