# Model report: one default random forest

scikit-learn 1.9.0; seed RF=20260918; participants train/val/test = 16339/2042/2042.

Majority-class reference accuracy (training majority = class 1): 0.6691. Accuracy is inflated by the 66.9% class-1 prevalence; AUROC and balanced accuracy are primary.

## Validation (descriptive)

- AUROC 0.7064; accuracy 0.7067; balanced accuracy 0.5953
- class 0 P/R/F1 0.634/0.267/0.375
- class 1 P/R/F1 0.718/0.924/0.808
- confusion [[TN,FP],[FN,TP]] = [[180, 495], [104, 1263]]

## Test (evaluated once)

- AUROC 0.6833; accuracy 0.6939; balanced accuracy 0.5805
- class 0 P/R/F1 0.589/0.246/0.347
- class 1 P/R/F1 0.711/0.915/0.800
- confusion [[TN,FP],[FN,TP]] = [[166, 509], [116, 1251]]

## Test bootstrap (10 within-class resamples, seed 20260918)

- AUROC mean 0.6841 (sample SD 0.0087, min/max 0.6758/0.7000)
- accuracy mean 0.6892 (SD 0.0048)
- balanced accuracy mean 0.5747 (SD 0.0069)

Bootstrap summaries condition on the fitted model, fixed split and observed class counts; they are a rough variability check, not a reliable 95% CI.
