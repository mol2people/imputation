# Model report: one default random forest

scikit-learn 1.9.0; seed RF=20260918; participants train/val/test = 16388/2049/2048.

Majority-class reference accuracy (training majority = class 1): 0.6686. Accuracy is inflated by the 66.9% class-1 prevalence; AUROC and balanced accuracy are primary.

## Validation (descriptive)

- AUROC 0.6677; accuracy 0.6877; balanced accuracy 0.5737
- class 0 P/R/F1 0.569/0.236/0.333
- class 1 P/R/F1 0.706/0.912/0.796
- confusion [[TN,FP],[FN,TP]] = [[160, 519], [121, 1249]]

## Test (evaluated once)

- AUROC 0.7132; accuracy 0.7026; balanced accuracy 0.5896
- class 0 P/R/F1 0.625/0.255/0.362
- class 1 P/R/F1 0.715/0.924/0.806
- confusion [[TN,FP],[FN,TP]] = [[173, 505], [104, 1266]]

## Test bootstrap (10 within-class resamples, seed 20260918)

- AUROC mean 0.7131 (sample SD 0.0160, min/max 0.6877/0.7329)
- accuracy mean 0.7037 (SD 0.0051)
- balanced accuracy mean 0.5923 (SD 0.0073)

Bootstrap summaries condition on the fitted model, fixed split and observed class counts; they are a rough variability check, not a reliable 95% CI.
