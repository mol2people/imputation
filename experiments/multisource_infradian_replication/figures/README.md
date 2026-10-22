# Full-run figures

These figures summarize the full primary-window run
`full_b199_boot1000_20261022` (`B=199`, `n_boot=1000`, sources
`2 3 6 7 9 13 19`, both protocols). Unless stated, they use `balanced90`.
They are descriptive: age is cross-sectional, not within-person follow-up.

The full run produced 65,449 endpoint rows, of which 65,443 converged. Six
rows are `rss0_degenerate`; see
`artifacts/runs/full_b199_boot1000_20261022/RUN_NOTES.md`.

## Female and male `Q_i` age curves

![Female and male Q_i age curves](fig1_q_i_age_curves_balanced90.png)

`Q_i` is mean spectral power in the 25–35 d band. Curves are source- and
endpoint-specific RCS fits; shaded bands are pointwise 95% participant
bootstrap intervals. Source 9 is omitted because the sex-interaction model has
no shared female–male support; its female-only `Q_i` support is 55–60 y and its
total-age term is not significant (`P=0.28`).

## Female age-association tests

![Female age-association heatmap](fig2_age_test_heatmap_balanced90.png)

HC3 chi-square P values for the female-only RCS model: total-age and nonlinear
terms separately, for `Q_i`, `E_i2`, `A_hat`, and `sigma_resid`. Color is
`-log10(P)` capped at 40; printed values are untransformed. No multiplicity
adjustment. Source 9 is sparse; only its `A_hat` total-age term is nominally
significant (`P=0.022`).

## Age of female−male slope divergence

![Female-male divergence forest](fig3_sex_divergence_forest.png)

Points show the age minimizing `d/dage [female(age) - male(age)]`, i.e. where
the female−male slope difference is most negative. Bars are 95% participant
bootstrap intervals. This is a sex-difference transition, not the inflection
point of either sex-specific curve. `balanced90` and `expanded90` are shown
separately. Source 9 has no shared female–male support. `n.e.` means the point
was not estimable but a bootstrap interval was available (source 7, `E_i2`,
`expanded90`).

## Interpretation limits

- These are cross-sectional age curves; they do not identify within-person
  aging or causal effects.
- `Q_i` is band-integrated spectral power and `E_i2` is peak excess. Neither
  estimates the prevalence of a 25–35 d rhythm; null rejection is a separate
  analysis.
- The 46–50 y cluster in the third figure is a derived sex-difference
  transition, not evidence for menopause timing or a causal adult transition.
- Device-source consistency is not independent replication; users may overlap
  across sources.
- Intervals are pointwise, `n_boot=1000`, and P values are unadjusted.
