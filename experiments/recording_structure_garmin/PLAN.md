# Recording structure in ch3000

2026-09-22. Plan only; no implementation or execution in this change.

## Question

How well can ch3000 recording patterns predict recorded salutation, and do they add information to the HR temporal model? Follow with targeted perturbations to test the HR representation's dependence on filling and sampling density.

Garmin/source 3, ch3000 only; salutation 10→0, 20→1. No additional physiological channels. This is exploratory signal characterization, not precision certification or a causal decomposition.

## First pass: cached data, three allocations, MultiRocket

Implement separately in `experiments/recording_structure_garmin/`; preserve existing experiments and results. Reuse corrected helpers from `experiments/temporal_garmin/run_temporal.py` without invoking its full runner, which also computes HYDRA and shuffled arms.

Read-only inputs, paths relative to repository root:

- `experiments/temporal_garmin/cache/day_bins.npz`: `hr`, `cnt`, `mask`, `user`, `date`.
- `experiments/r1b_dateaware_garmin_2026-09-21/cache/day40_table.parquet`: daily statistics for B40 and designated dates.
- `experiments/r1b_dateaware_garmin_2026-09-21/cache/folds.parquet`: use allocations **0, 1, 2**, chosen before results.
- `artifacts_sq/cohort_manifest_model_sources.parquet` and `artifacts_sq/split_manifest_sq.parquet`: cohort and labels.

Keep all 3,848 participants and the same first 40 adequate days × 288 five-minute bins. Each allocation has 2,696/579/573 train/validation/test participants. These are adequate days, not consecutive calendar days. No new eligibility gate, epoch scan or database query in the first pass.

`cnt` counts retained events after the existing HR-validity filters; `mask = cnt > 0`. They measure recording availability, not directly wear time. Numeric HR values never enter recording-only arms, although the validity filter can influence their inputs.

Define daily sequences `M = mask.astype(float)` and `C = log1p(cnt)`. Zeros are meaningful; do not fill them. Let `S(X)` be per-clock-bin mean and sample SD across the 40 days (576 features). Let `T(X)` be daily MultiRocket features pooled across days by mean and sample SD.

| Arm | Participant features | Role |
|---|---|---|
| M_static | S(M) | Average observation pattern |
| M_temporal | S(M) + T(M) | Arrangement of observed/missing stretches |
| M_shuffled | S(M) + T(M_shuffled) | Mask arrangement control |
| C_static | S(C) | Average count pattern |
| C_temporal | S(C) + T(C) | Temporal recording density, including presence |
| C_shuffled | S(C) + T(C_shuffled) | Count arrangement control |
| H | B40 + T(HR) | Matched HR reference |
| H_recording | H + S(M) + S(C) + T(M) + T(C) | Complementarity with HR |

B40 is the corrected temporal experiment's existing 305-column block, including HR statistics and mean masks. Use it **only in H and H_recording**. In H_recording, omit the mean-mask part of S(M), which is already in B40; retain its SD part. Counts already encode presence; C_temporal − M_temporal is a representation comparison, not a pure density effect.

## Shuffle control

For each participant/day, randomly permute **all 288 positions, including zeros**. Apply the same permutation to M and C. This preserves each day's occupied-bin count and count distribution, including total event count, while breaking local arrangement and clock alignment. Never shuffle across days or participants. Use one fixed permutation per participant/day, generated with `SeedSequence([20260921, 3, 14, user, date])`, reused across allocations. IDs/dates key randomness only; they are not predictors.

Keep S(M) and S(C) from the **original, unshuffled** data in the corresponding control arms. Thus the contrast changes only the temporal block; average clock profiles remain available. Refit MR on shuffled training sequences using the same selected training days, feature budget and fit seeds as its unshuffled counterpart; fit each head with the same alpha-selection procedure. Do not feed shuffled sequences through an MR transform fitted on original inputs. HR and H_recording remain unshuffled.

## Fitting and compute

- Start with **MultiRocket only**: reuse the installed pinned module-level implementation, feature budget 1,250 per transformation (9,408 actual daily outputs), base + first differences. Defer HYDRA until a result needs confirmation with another representation.
- Fit separate MR parameters for HR, M, C and the two shuffled inputs using four seeded days per training participant; use the same selected days across inputs. Share fitted feature blocks across arms within an allocation. HR uses the corrected training-only clock-bin median fill.
- Reuse the HR seed streams from the temporal experiment. Reserve new component IDs 10/11 for M base/difference fits and 12/13 for C; use `SeedSequence([20260921, 3, allocation, component])`. Seed numba through the existing `_nb_seed` helper, not just numpy's global RNG.
- Fit median fallback for any undefined pooled features, zero-variance removal and standard scaling on training participants only, separately by block. Pool all 40 recording sequences; an all-zero recording day is valid information, not a day to omit.
- Ridge on participant labels, intercept enabled, no class weighting. Per arm and allocation, select alpha from **[1e-3, 1, 1e3, 1e4, 1e5, 1e6]** by validation AUROC; smallest alpha wins ties. Save the trace; record boundary optima without extending the grid during this pass. Refit H under this same rule; older alpha=1,000 scores are context only.
- This deliberately uses ordinary validation tuning rather than nested resampling. Validation maxima are selected estimates. Evaluate the selected model on that allocation's test participants; both splits remain part of a reused exploratory cohort. This new plan explicitly permits limited fold reuse despite the earlier temporal plan's retirement statement.
- CPU only, at most eight active threads, allocations sequential. Transform and pool in participant batches; discard daily feature slabs. Cache pooled blocks with allocation, seeds and input provenance so ridge fitting does not repeat transforms. Do not reuse training-fitted blocks across allocations.
- Start with a small fixture for binary/count inputs and one complete allocation; record actual wall time and peak RSS, then finish allocations 1 and 2. Target <8 GiB; reduce batch size if needed. No broad environment setup, vendor download or unrelated refactoring.

## Readout and advancement

Save per-allocation validation/test AUROC and paired test deltas: M_temporal − M_static, C_temporal − C_static, M_temporal − M_shuffled, C_temporal − C_shuffled, C_temporal − M_temporal, H_recording − H. Report the three allocation values for each contrast, their mean/range and direction agreement. No bootstrap, p-values, multiplicity families or population confidence claims in this pilot.

Use **0.01 AUROC as a practical prioritization scale**, not a significance threshold or minimum scientifically possible effect. Recommend follow-up only for findings that affect the interpretation: a roughly ≥0.01 gain with the same sign in all three allocations, or an inconsistent result whose resolution would change the next experiment. Recommend which arms, if any, to extend to the remaining allocations with the same settings. Stop after the three-allocation pilot and report; do not automatically extend it.

Interpretation:

- Strong recording-only AUROC establishes predictive information in recording structure within this cohort.
- Temporal over static gains establish a useful representation increment; nonlinear feature expansion can contribute. Unshuffled over shuffled gains support sensitivity to recording arrangement beyond the original static profiles and preserved daily count distributions. Shuffling breaks both local order and clock alignment, so it does not separate their contributions or establish a causal mechanism.
- H_recording improving on H supports complementary information. No improvement does not establish independence: H already contains masks and may encode recording texture.
- Weak recording-only models do not rule out interactions between recording structure and HR values.
- Never convert AUROC differences into a percentage of physiological signal explained. Do not infer non-wear, biological sex, or a performance ceiling.

## Targeted follow-ups, outside the initial deliverable

Choose the next probe in the pilot report; these are alternatives to another large arm search.

1. **Filling sensitivity (cached data first).** Compare H's current fill with an overall training median, and within-day linear interpolation for interior gaps of at most six bins (30 min), with the original clock-bin training median for remaining gaps. Preserve observed bins and masks exactly. Refit each transform/head; hold B40 fixed. A delta measures sensitivity of the temporal representation to this convention, not proof of artifact.
2. **Density sensitivity (requires retained epoch values).** Replace each occupied bin's mean by one uniformly sampled retained event from that bin, preserving dates and masks. Use three fixed sampling seeds; no other channel. Refit the HR transform/head, holding B40 fixed. This removes variation in the number of events averaged per bin but adds noise and retains sampling-time differences; a drop is not a pure density effect. Cache sampled grids once if pursued.
3. **HYDRA confirmation.** Repeat only the informative recording/HR comparison if the conclusion may depend on MR's handling of binary or count sequences. It is not required to complete the pilot.

## Minimal checks and deliverables

Before full participant fits: exact user/date/fold alignment and disjoint participants within each allocation; `mask == (cnt > 0)`; recording-arm schemas exclude HR values and B40; no NaN/Inf at the classifier; train-only fitted preprocessing. Check shuffle preservation of each day's count multiset and occupied-bin count, `M_shuffled == (C_shuffled > 0)`, and unchanged static blocks. On a small fixture, check pooling against direct aggregation and same-seed repeat scores within 1e-6. Reuse existing numerical conventions; log any nonfinite transform handling. No exhaustive verification suite.

Write incrementally under this experiment: `results/metrics.csv`, `results/predictions.csv` (user, allocation, split, arm, label, score), `results/repro.json` (configuration, alpha traces, seeds, input provenance, checks, timing/RSS), and a short `results/REPORT.md` with the arm table, paired deltas, interpretation and one recommended next probe. Raw/pooled caches stay local and gitignored. No participant rows in logs. Initial completion means the eight-arm, three-allocation pilot and report, not automatic follow-up experiments or publication.
