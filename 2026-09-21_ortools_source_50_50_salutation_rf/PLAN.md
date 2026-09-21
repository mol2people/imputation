# OR-Tools configurable source-specific salutation RF

**Date:** 2026-09-21  
**Status:** completed

## Question

Within each source-specific raw-epoch cohort, how well can a fixed 100-tree
random forest discriminate recorded salutation 10 versus 20 when evaluated in
both complementary folds of a balanced participant allocation?

## Cohort and representation

- Start from every nonzero `out/<user_id>.csv` with a 10/20 salutation label.
- A participant needs at least one valid core-HR event (`type` 3000/3001/3002,
  25--230 bpm, positive timestamp) from Garmin (3), Apple (6), or Samsung (7).
  There is no coverage-duration gate.
- Assign each participant once to the target source with the most valid target
  events (ties: Garmin, Apple, Samsung). Extract features only from that
  selected source's events. Source is never a predictor.
- Features are source-restricted, participant-level summaries: value,
  distribution, daily-recording, calendar-span, and local-time composition
  summaries for each core channel and overall. No identifiers, salutation,
  age, BMI, source code, or absolute dates enter the RF.
- `birth_date` stores a birth year only. Define the fixed
  `age_at_first_epoch_proxy` from the earliest valid core-HR event in the raw
  file, using a Dec-1 birthday convention and the event's local date (UTC if
  offset is missing). Five-year age bands are derived from that proxy before
  allocation. BMI uses linked `bmi_grp`; missing values form explicit balance
  categories.

## Allocation

For each source separately, OR-Tools CP-SAT assigns every participant to
`train` or `test`. The default is 50/50 with `salutation`, five-year age band,
and BMI group as allocation variables.

- `--balance-vars` selects a comma-separated subset of `salutation`,
  `age_band_5y`, and `bmi_group`. The selected joint cells are minimized
  first, then selected non-salutation margins. Including `salutation` imposes
  exact proportional salutation quotas in the test fold; omitting it does not.
- `--fold-sizes TRAIN,TEST` sets relative two-fold sizes, for example `3,1`
  gives 75/25. Integer targets use a deterministic largest-remainder rule.
- `--output-dir` isolates a changed allocation. It can reuse the completed
  default cohort cache, so an allocation change does not require a raw rescan.
- `STUDY_SEED = 20260921`; one CP-SAT worker gives reproducible allocation.
  The paired model evaluation remains deliberately two-fold.

## Model and evaluation

For each of Garmin, Apple, and Samsung:

- RF: `n_estimators=100`, `max_features=0.4`, `min_samples_leaf=10`,
  `max_depth=10`, `class_weight="balanced_subsample"`.
- Median imputation plus train-derived missingness indicators; preprocessing is
  fit separately in each training half.
- Raw parsing uses five processes; the final forests use one worker. This stays
  within the five-core cap and makes the saved probability vectors exactly
  reproducible in the local scikit-learn build.
- Fit direction A on allocated train and evaluate allocated test. Generate 10
  within-class bootstrap draws from saved test predictions, without refitting.
- Flip train/test, refit, and repeat the 10-draw bootstrap. Each participant is
  therefore out-of-sample exactly once.

The report labels this as prediction of recorded salutation, not a biological
or causal inference.
