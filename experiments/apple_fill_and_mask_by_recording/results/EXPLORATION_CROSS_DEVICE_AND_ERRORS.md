# Cross-device and error exploration: superseded numerical synthesis

Corrected on 2026-10-19. Use [REPORT.md](REPORT.md) for the verified Apple fill-and-mask benchmark.

[05_explore_physiology_and_errors.py](../scripts/05_explore_physiology_and_errors.py) combines older Apple cohort-scaling predictions and binning summaries with hard-coded Apple/Garmin benchmark values (lines 55–70). The previous cross-device table and causal interpretations are withdrawn: its claimed three-allocation Apple readout is not supported by the available fill-and-mask prediction artifact, and Garmin values were not independently verified in this reconciliation.

Older scaling outputs are separate analyses, with their own allocations, penalties, and comparators. They must not be used as evidence for new fill-and-mask arm effects. Differences between devices also combine cohort composition, sampling, recording history, preprocessing, and model differences; a device-specific physiological or missingness mechanism is not identified by comparing their AUROCs.

For the independently checked, threshold-dependent comparison of older summary and MultiRocket predictions, see [the degraded-participant note](EXPLORATION_DEGRADED_PARTICIPANTS.md). No models or pipelines were rerun for this correction.
