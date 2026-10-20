# Post hoc calibration audit: no held-out probability evaluation

Corrected on 2026-10-19. Use [REPORT.md](REPORT.md) for discrimination results supported by saved predictions.

The previous table and its calibration conclusions are withdrawn. In [08_evaluate_harrell_calibration.py](../scripts/08_evaluate_harrell_calibration.py), lines 76–79 fit a logistic score-to-probability mapping using the test labels and predict probabilities for those same participants. Lines 86–102 evaluate Brier score, log loss, and calibration slope on the same labels. These are post hoc, test-fitted apparent scores; they are not held-out probability-performance estimates.

A slope near 1 and intercept near 0 after fitting and reassessing the logistic mapping on the same observations do not demonstrate calibration of the original Ridge scores or the benefit of shrinkage. The reported intercept is from a free-slope logistic fit, not the fixed-slope offset model described in the script comment (lines 105–110). The previous numerical table was not independently reconciled and is not retained as evidence.

For a valid held-out assessment, fit the probability mapping on training-derived or validation data, freeze it, then evaluate probabilities on untouched test labels. This correction did not fit a replacement calibrator or rerun any model.
