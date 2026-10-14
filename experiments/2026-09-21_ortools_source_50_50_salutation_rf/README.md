# OR-Tools configurable source-specific salutation experiment

This completed experiment fits separate random forests for broader Garmin (source 3), Apple (source 6), and Samsung (source 7) cohorts. The cohort sizes are 5,636, 13,884, and 2,068, respectively. A balanced 50/50 assignment is evaluated in both directions: each half serves once as training and once as test.

These are broader cohorts than the strict source-specific RF study. Keep their results separate from the strict `win` arm in the [results reference](../../reports/results_reference.md#earlier-summary-and-feature-experiments). Detailed results are local files `output/report.md` and `output/model_metrics.csv`; see the [experiment index](../../learnings/experiment_lineage_worktree.md) for the other source-specific studies.

## Implementation and execution

Standalone experiment created 2026-09-21. It reads the raw `../../out/` epoch
files and linked salutation/BMI tables, but does not modify any existing module
or artifact.

Run from the repository root:

```sh
/Users/bulat/micromamba/envs/datenspende/bin/python \
  experiments/2026-09-21_ortools_source_50_50_salutation_rf/run_experiment.py --mode all
```

The allocation is configurable without changing the source code. Defaults
reproduce the completed experiment:

```sh
/Users/bulat/micromamba/envs/datenspende/bin/python \
  experiments/2026-09-21_ortools_source_50_50_salutation_rf/run_experiment.py \
  --mode allocate \
  --balance-vars salutation,age_band_5y,bmi_group \
  --fold-sizes 1,1
```

For example, this creates a separate 75/25 allocation balanced on salutation
and age band, reusing the existing raw-cohort cache:

```sh
/Users/bulat/micromamba/envs/datenspende/bin/python \
  experiments/2026-09-21_ortools_source_50_50_salutation_rf/run_experiment.py \
  --mode allocate \
  --output-dir output_75_25_salutation_age \
  --balance-vars salutation,age_band_5y \
  --fold-sizes 3,1
```

Allowed allocation variables are `salutation`, `age_band_5y`, and
`bmi_group`. Including `salutation` makes its proportional fold quotas hard
constraints; otherwise it is not enforced. The current modelling/evaluation
workflow supports exactly two folds (`train,test`). For an alternative
allocation, reuse the same `--output-dir` and choices for `--mode model`,
`report`, or `verify`.

The raw scan uses at most five processes. Final RF fits use one worker after a
repeatability check found tiny probability differences with five joblib workers
in the local scikit-learn build. Result directories named `output*` are
deliberately untracked: they contain participant-level assignments and
predictions.
