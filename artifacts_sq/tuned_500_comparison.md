# Default (100 trees, untuned) vs tuned (500 trees, CV-selected)

**Documentation update:** The paired tuned-minus-default summaries were added
directly from the saved draw tables. Per instruction, `compare_models.py` was
not modified; rerunning that generator would restore its earlier, less complete
layout.

Tuning used pooled five-fold out-of-fold AUROC on train+validation over 48 configurations, followed by a 500-tree fit on the full train+validation pool. The frozen test participants were not used for tuning, but this is a second analysis of a test fold already examined by the default run; all test comparisons are descriptive rather than confirmatory. Tuned validation predictions are in-sample and are not compared here.

Default and tuned runs use the same ordered test participants and the same 100 within-class participant resamples. The paired tuned-minus-default bootstrap deltas below condition on this cohort, split, and the two fitted models; they do not include split, fitting, or tuning-selection uncertainty. Percentile intervals are exploratory.

## Source 3 - test AUROC

| variant | default 100 | tuned 500 | point delta | paired draw mean | paired SD | paired 95% pct |
|---|---:|---:|---:|---:|---:|---|
| demo | 0.5743 | 0.5743 | +0.0000 | +0.0001 | 0.0006 | [-0.0011, +0.0010] |
| rec | 0.6568 | 0.6981 | +0.0414 | +0.0421 | 0.0108 | [+0.0186, +0.0621] |
| win | 0.6918 | 0.7279 | +0.0361 | +0.0355 | 0.0117 | [+0.0130, +0.0585] |
| roll_rec | 0.5928 | 0.6412 | +0.0484 | +0.0473 | 0.0130 | [+0.0227, +0.0710] |
| roll_win | 0.6445 | 0.6719 | +0.0273 | +0.0256 | 0.0113 | [+0.0062, +0.0465] |
| all | 0.6318 | 0.7068 | +0.0750 | +0.0741 | 0.0124 | [+0.0494, +0.0986] |

## Source 3 - within-run feature-block contrasts

| comparison | default point | tuned point | tuned draw mean | tuned draw SD |
|---|---:|---:|---:|---:|
| recording vs demographics (`rec-demo`) | +0.0825 | +0.1238 | +0.1228 | 0.0394 |
| window vs recording (`win-rec`) | +0.0350 | +0.0298 | +0.0279 | 0.0140 |
| rolling window vs rolling recording (`roll_win-roll_rec`) | +0.0518 | +0.0307 | +0.0301 | 0.0180 |
| all blocks vs recording (`all-rec`) | -0.0249 | +0.0087 | +0.0073 | 0.0097 |

Chosen configurations (per variant). Each CV AUROC is the score used to select the maximum among 48 configurations, not an unbiased post-selection performance estimate:
- `demo`: {'max_features': 'sqrt', 'min_samples_leaf': 1, 'max_depth': None, 'class_weight': None} (cv_auroc 0.5554)
- `rec`: {'max_features': 0.4, 'min_samples_leaf': 10, 'max_depth': None, 'class_weight': 'balanced_subsample'} (cv_auroc 0.7539)
- `win`: {'max_features': 0.4, 'min_samples_leaf': 5, 'max_depth': None, 'class_weight': 'balanced_subsample'} (cv_auroc 0.7435)
- `roll_rec`: {'max_features': 0.4, 'min_samples_leaf': 10, 'max_depth': 12, 'class_weight': 'balanced_subsample'} (cv_auroc 0.7161)
- `roll_win`: {'max_features': 0.4, 'min_samples_leaf': 10, 'max_depth': None, 'class_weight': None} (cv_auroc 0.7086)
- `all`: {'max_features': 0.4, 'min_samples_leaf': 10, 'max_depth': 12, 'class_weight': 'balanced_subsample'} (cv_auroc 0.7490)

## Source 6 - test AUROC

| variant | default 100 | tuned 500 | point delta | paired draw mean | paired SD | paired 95% pct |
|---|---:|---:|---:|---:|---:|---|
| demo | 0.5618 | 0.5619 | +0.0001 | +0.0003 | 0.0017 | [-0.0025, +0.0037] |
| rec | 0.7224 | 0.7551 | +0.0327 | +0.0324 | 0.0167 | [+0.0020, +0.0590] |
| win | 0.6814 | 0.7472 | +0.0657 | +0.0662 | 0.0180 | [+0.0328, +0.0982] |
| roll_rec | 0.7012 | 0.7203 | +0.0191 | +0.0184 | 0.0140 | [-0.0066, +0.0428] |
| roll_win | 0.6503 | 0.6809 | +0.0307 | +0.0270 | 0.0167 | [+0.0001, +0.0597] |
| all | 0.7184 | 0.7714 | +0.0530 | +0.0535 | 0.0158 | [+0.0243, +0.0857] |

## Source 6 - within-run feature-block contrasts

| comparison | default point | tuned point | tuned draw mean | tuned draw SD |
|---|---:|---:|---:|---:|
| recording vs demographics (`rec-demo`) | +0.1606 | +0.1932 | +0.1903 | 0.0349 |
| window vs recording (`win-rec`) | -0.0410 | -0.0079 | -0.0089 | 0.0193 |
| rolling window vs rolling recording (`roll_win-roll_rec`) | -0.0509 | -0.0394 | -0.0387 | 0.0205 |
| all blocks vs recording (`all-rec`) | -0.0040 | +0.0163 | +0.0163 | 0.0185 |

Chosen configurations (per variant). Each CV AUROC is the score used to select the maximum among 48 configurations, not an unbiased post-selection performance estimate:
- `demo`: {'max_features': 0.2, 'min_samples_leaf': 10, 'max_depth': None, 'class_weight': None} (cv_auroc 0.5481)
- `rec`: {'max_features': 0.4, 'min_samples_leaf': 5, 'max_depth': None, 'class_weight': 'balanced_subsample'} (cv_auroc 0.7066)
- `win`: {'max_features': 0.4, 'min_samples_leaf': 5, 'max_depth': None, 'class_weight': 'balanced_subsample'} (cv_auroc 0.6809)
- `roll_rec`: {'max_features': 0.4, 'min_samples_leaf': 10, 'max_depth': 12, 'class_weight': 'balanced_subsample'} (cv_auroc 0.6995)
- `roll_win`: {'max_features': 'sqrt', 'min_samples_leaf': 10, 'max_depth': 12, 'class_weight': None} (cv_auroc 0.6558)
- `all`: {'max_features': 0.4, 'min_samples_leaf': 10, 'max_depth': None, 'class_weight': 'balanced_subsample'} (cv_auroc 0.7239)

## Source 7 - test AUROC

| variant | default 100 | tuned 500 | point delta | paired draw mean | paired SD | paired 95% pct |
|---|---:|---:|---:|---:|---:|---|
| demo | 0.5750 | 0.5996 | +0.0245 | +0.0273 | 0.0226 | [+0.0000, +0.0759] |
| rec | 0.6638 | 0.6696 | +0.0058 | +0.0017 | 0.0538 | [-0.0893, +0.0980] |
| win | 0.6609 | 0.6342 | -0.0267 | -0.0308 | 0.0327 | [-0.0988, +0.0240] |
| roll_rec | 0.6392 | 0.6328 | -0.0065 | -0.0109 | 0.0506 | [-0.0944, +0.0861] |
| roll_win | 0.6061 | 0.6551 | +0.0491 | +0.0415 | 0.0653 | [-0.0770, +0.1928] |
| all | 0.5880 | 0.6797 | +0.0916 | +0.0802 | 0.0630 | [-0.0468, +0.1855] |

## Source 7 - within-run feature-block contrasts

| comparison | default point | tuned point | tuned draw mean | tuned draw SD |
|---|---:|---:|---:|---:|
| recording vs demographics (`rec-demo`) | +0.0887 | +0.0700 | +0.0671 | 0.0864 |
| window vs recording (`win-rec`) | -0.0029 | -0.0354 | -0.0412 | 0.0699 |
| rolling window vs rolling recording (`roll_win-roll_rec`) | -0.0332 | +0.0224 | +0.0189 | 0.0689 |
| all blocks vs recording (`all-rec`) | -0.0758 | +0.0101 | +0.0050 | 0.0387 |

Chosen configurations (per variant). Each CV AUROC is the score used to select the maximum among 48 configurations, not an unbiased post-selection performance estimate:
- `demo`: {'max_features': 0.2, 'min_samples_leaf': 10, 'max_depth': None, 'class_weight': 'balanced_subsample'} (cv_auroc 0.4737)
- `rec`: {'max_features': 0.2, 'min_samples_leaf': 5, 'max_depth': 12, 'class_weight': 'balanced_subsample'} (cv_auroc 0.6226)
- `win`: {'max_features': 'sqrt', 'min_samples_leaf': 1, 'max_depth': None, 'class_weight': None} (cv_auroc 0.6382)
- `roll_rec`: {'max_features': 'sqrt', 'min_samples_leaf': 1, 'max_depth': None, 'class_weight': 'balanced_subsample'} (cv_auroc 0.5878)
- `roll_win`: {'max_features': 'sqrt', 'min_samples_leaf': 2, 'max_depth': None, 'class_weight': None} (cv_auroc 0.5762)
- `all`: {'max_features': 0.4, 'min_samples_leaf': 5, 'max_depth': 12, 'class_weight': None} (cv_auroc 0.6254)
