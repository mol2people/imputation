# Heartbeat Lab: implementation and evaluation plan

**Implementation update:** local study-trained bundles, customer selection,
uploads, repeated perturbations, background retraining and temporal validation
are now implemented and exercised. See [README.md](README.md) for current use and
[AUDIT_REPORT.md](AUDIT_REPORT.md) for the evaluation audit. The original scope
and sequencing below are retained as the design record.

## 1. Reference and scope

The highest reported endpoint in this folder is full-history MultiRocket:
**0.9264 mean test AUROC**, using all available adequate days in the selected
3,848-person Garmin cohort. The multi-allocation first-40-days Combined model
reached **0.8565**; these are different day budgets/model recipes.

Evidence:
- [Full-history implementation](../experiments/temporal_garmin/day_scaling_all.py)
- [Full-history results and caveats](../experiments/temporal_garmin/results/days_scaling_all.md)
- [Temporal reference and corrected v2 results](../experiments/temporal_garmin/results/REPORT.md)
- [Filling comparison](../experiments/filling_sensitivity_garmin/results/REPORT.md)

The all-days endpoint reused allocation-0-trained kernels/fill across allocations.
For other allocations, validation/test customers can therefore have influenced
the transform. Its AUROC is exploratory; it is neither this demo's performance
nor a guaranteed lower bound. Observed histories range from 80 to 979 adequate
days; 80 was the observed minimum in that selected cohort, not an additional
eligibility rule implemented by the scanner.

Upstream selection was v2 eligibility → sole-source Garmin → strict longitudinal
coverage on any core channel (3000/3001/3002) → adequate type-3000 days. The strict
rule required a 13-week window with ≥11 weeks having ≥5 strict days and ≥65 strict
days overall, separately per channel; strict days had ≥12 hours union coverage.
An upload passing the daily gate is not thereby in that training population.

## 2. Small initial implementation

Use Streamlit, one installable Python package and numbered scripts. Keep uploads
in the process, with no database or custom frontend. Begin with:

| Choice | Representation | Missing HR bins | Fitted classifier |
|---|---|---|---|
| summary | B40-style summaries | unused | ridge α=1000 |
| mr_clock | B40 + pooled author MultiRocket | train clock-bin medians | ridge α=1000 |
| mr_global | same MultiRocket recipe | train global median | ridge α=1000 |

These are complete fitted bundles, not arbitrary inference-time preprocessing
toggles. Hold data, customer split, seed and feature budget fixed to compare fills.
The app can select or overlay up to three bundle timelines.

Deliberate differences from the historical endpoint: a fresh 70/15/15 stratified
customer split by default; fill estimated over all selected training days; seeded
kernel fit on up to four days per training customer; no reuse of fitted transforms
across splits. This follows the successful recipe, not historical random streams
or an exact numerical replay. Reproducing the original 0.9264 remains the job of
the original experiment script, with its caveats intact.

## 3. Inspect, then run the case study

1. Inspect this folder and run the two synthetic notebooks.
2. Prepare the existing cached all-days cohort using `01_from_repository.py`.
   Alternatively prepare a new labelled CSV manifest with `01_prepare.py`.
3. Save one customer allocation. For demographic balance use the existing
   `cohort_allocator`; for the minimal demo use `allocate_customers`.
4. Fit all three options through `02_train.py`. Inspect validation AUROC and
   paired scores on the same people; retain test as a final evaluation set.
   Repeatedly choosing models using test results makes those results exploratory.
5. If tuning is needed, use `customer_cv` on the outer training partition and
   fit the entire transform inside each fold. Freeze settings before final test.
6. Verify raw-upload features against cached study features for matched person/day
   pairs. Check timezone changes, gaps and interval semantics before claiming
   equivalence. Current synthetic parity checks are not that cohort-wide audit.
7. Assess performance by day budget and device before representing short-history
   or non-Garmin predictions as validated. For probabilities add calibration on
   held-out customers, report calibration, then evaluate on untouched test users.

## 4. Daily visualization

Default: cumulative score at each adequate date, using only history up to then.
Secondary: individual-day scores, visibly marked exploratory because the
full-history model was not validated on singleton days. Show data availability
and download the exact scores. Avoid claiming that someone's gender changes
when their model score changes.

A dedicated daily model is a later, separate experiment: keep all days of a
customer in one split, consider equal customer weighting, and distinguish
day-level from participant-level estimands. Do not create row-random CV.

## 5. Sharing

Local use first. After code/model review, choose a host and upload only code,
approved model bundles and synthetic examples. Real customer CSVs, assignments
and predictions stay private. Remote uploads necessarily reach the host; say so
in the interface. Local full-cohort retraining was subsequently authorized and
completed. No public deployment or model publication has been performed.

Acceptance: understandable source; installable package; reproducible split files;
train-only fitting; switchable model/preprocessing bundles; working CSV preview,
timeline and export; runnable explanatory notebooks; no historical AUROC attached
to an unvalidated new bundle.
