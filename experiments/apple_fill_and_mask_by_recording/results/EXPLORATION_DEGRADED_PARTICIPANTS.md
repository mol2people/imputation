# Degraded participants: threshold-dependent comparison of older predictions

Corrected on 2026-10-19. This analysis uses `apple_cohort_scaling/artifacts/scaling_predictions_test.parquet`, not the fill-and-mask benchmark predictions. Configuration: allocation 0 test, `all_days`, 5m, `ridge_selected`; summary versus MultiRocket, 1,671 participants. Use [REPORT.md](REPORT.md) for the new benchmark.

The following counts were independently reproduced from the older saved scores. Each model is thresholded at its own empirical test-score median, as implemented in [07_analyze_degraded_participants.py](../scripts/07_analyze_degraded_participants.py), lines 132–156.

| Classification outcome | Total | Class 0 | Class 1 |
|---|---:|---:|---:|
| Both correct | 839 | 343 | 496 |
| Rescued by MultiRocket | 428 | 162 | 266 |
| Degraded by MultiRocket | 176 | 36 | 140 |
| Both incorrect | 228 | 38 | 190 |

Class labels represent recorded administrative salutation, not verified biological sex or gender identity. The previous report's both-correct counts (278/561) were incorrect. Its physiological tables and feature-mechanism explanations were not independently established and have been withdrawn.

The median thresholds are 0.6723859 for MultiRocket and 0.6548844 for summaries. Each predicts class 1 for 836/1,671 participants (50.03%), whereas 1,092/1,671 (65.35%) have class 1 labels. Thus 140/176 degraded participants having class 1 does not establish an intrinsic model failure for a physiological subgroup. The counts depend on an exploratory, test-derived threshold and do not describe performance under the benchmark's validation-selected decision rule.

The net 252 additional correct classifications applies only to these thresholds and older models. It does not establish a gain in calibrated probability accuracy or identify which temporal features caused a change. A prespecified threshold comparison and held-out subgroup evaluation would be needed to assess an actionable degradation pattern. No models or pipelines were rerun for this correction.
