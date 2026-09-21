# cohort-allocator

A small, standalone package for turning a participant-level demographic summary
into complete, disjoint, balanced cohorts. It contains no project-specific raw
data logic and does not modify the existing salutation experiment.

It uses OR-Tools CP-SAT to assign every participant once. Within each optional
partition (for example, device source), it enforces exact whole-cohort fold
sizes and minimizes imbalance in:

1. the joint distribution of the requested split variables;
2. each requested variable's marginal distribution; then
3. a deterministic, seed-controlled tie-break.

The optimiser uses one CP-SAT worker. The same input, configuration, seed, and
OR-Tools version produce the same allocation.

## Input

Supply one CSV or Parquet table with exactly one row per participant. It must
include:

- a unique identifier;
- categorical fields to balance;
- optionally, one or more partition fields.

All supplied split fields are treated as categories. Pre-bin continuous
variables first: use age_band_5y, not raw floating-point age; use a BMI
category rather than raw BMI. Missing values form their own category.

## Install

From this repository:

~~~sh
/Users/bulat/micromamba/envs/datenspende/bin/python -m pip install -e cohort_allocator
~~~

For Parquet input/output, install the optional extra if the environment does
not already have PyArrow:

~~~sh
/Users/bulat/micromamba/envs/datenspende/bin/python -m pip install -e 'cohort_allocator[parquet]'
~~~

## Command line

This prepares a source-specific 50/50 train/test allocation from a demographic
summary:

~~~sh
cohort-allocate demographics.parquet \
  --id-column user_id \
  --stratify salutation,age_band_5y,bmi_group \
  --partition-by source_id \
  --fold-sizes train=1,test=1 \
  --seed 20260921 \
  --output-dir prepared_50_50
~~~

Named folds are recommended. Unequal and three-way allocations work directly:

~~~sh
cohort-allocate demographics.csv \
  --id-column participant_id \
  --stratify salutation,age_band_5y,bmi_group \
  --fold-sizes train=8,validation=1,test=1 \
  --output-dir prepared_80_10_10 \
  --format csv
~~~

Numeric-only sizes are also accepted and receive automatic names such as
fold_1, fold_2, and fold_3.

The output directory must be new or empty. This prevents a rerun from silently
replacing participant-level assignments.

## Outputs

For the first example:

~~~
prepared_50_50/
  assignments.parquet          # complete input summary plus fold label
  balance.csv                  # joint and marginal allocation audit
  allocation_summary.json      # configuration, solver outcomes, fold targets
  cohorts/
    train.parquet              # all rows assigned to train
    test.parquet               # all rows assigned to test
~~~

Assignments is the canonical split manifest. Each fold-specific cohort keeps
all supplied columns plus the fold label, so it can be joined to features or
used directly for downstream modelling.

## Python API

~~~python
import pandas as pd
from cohort_allocator import allocate_cohort, make_config, write_prepared_cohorts

summary = pd.read_parquet("demographics.parquet")
config = make_config(
    id_column="user_id",
    stratify_columns=("salutation", "age_band_5y", "bmi_group"),
    partition_columns=("source_id",),
    fold_sizes={"train": 1, "test": 1},
    seed=20260921,
)
result = allocate_cohort(summary, config)
write_prepared_cohorts(result, "prepared_50_50")
~~~

## Scope

This is an allocation utility, not an analysis pipeline. It does not derive
age, BMI, labels, wearable features, or eligibility from raw data. Users must
freeze those definitions in the supplied summary before splitting. If a tiny
partition or a highly sparse joint cell cannot appear in every fold, the exact
fold totals are still respected and balance.csv makes the residual imbalance
visible.
