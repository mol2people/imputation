# Circadian and mask dynamics: exploratory provenance

Corrected on 2026-10-19. The previous interpretation is superseded by [REPORT.md](REPORT.md) for this experiment's verified benchmark results.

[04_explore_circadian_and_mask_dynamics.py](../scripts/04_explore_circadian_and_mask_dynamics.py) computes descriptive mask and observed-HR features from the older `apple_cohort_scaling/artifacts/cache_5min` caches, restricted to allocation 0 training participants. Its reported clock-window associations are training-cohort descriptions, not held-out tests of mechanisms. Those cache-derived numerical summaries were not independently recomputed during this reconciliation and are omitted here.

The script's auxiliary five-fold comparisons use different feature sets and classifiers; standardization is fitted before splitting into folds (lines 286–329). They do not isolate the source of the benchmark MultiRocket increment. The previous benchmark gains and blending-weight claims must be checked against the corrected main report rather than reused from this exploration.

Mask changes identify transitions between observed and missing bins. They do not identify physical watch removal, charging, sleep, or activity without corresponding measurements. Predictive mask features can reflect recording behavior, sensor sampling, participant characteristics, and their associations. Their mechanism and independence from HR are not established by univariate associations or a score blend. The H_clock benchmark also includes recording summaries; it is not a physiology-only comparator.

No exploratory models, cache summaries, or benchmark models were rerun for this correction.
