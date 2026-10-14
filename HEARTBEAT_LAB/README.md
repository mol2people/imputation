# Heartbeat Lab

A local app for inspecting Garmin heart-rate records, comparing saved model scores, and testing sensitivity to changes in the input. The target is recorded salutation: **10 (F), 20 (M)**. The app uses source 3, type 3000 HR; its CSV and model conventions are defined [below](#csv-and-model-semantics).

The documented study setup contains summary, MultiRocket/clock-fill, and MultiRocket/global-fill bundles. It supports selection of all 3,848 study participants, including training participants whose predictions are labelled in-sample. The app reads original participant files through `../out` or accepts uploaded CSVs. These inputs and fitted bundles are local and may be absent from a fresh checkout.

To inspect existing results, start with the next section. For the scientific comparisons, read the [results reference](../reports/results_reference.md#saved-heartbeat-lab-models). [PLAN.md](PLAN.md) is the original design record; [VALIDATION.md](VALIDATION.md) records the checks performed at the time.

## Inspect existing models

With the existing local environment and saved bundles available, run from `HEARTBEAT_LAB/`:

```sh
streamlit run app.py --server.address 127.0.0.1
```

Open **http://127.0.0.1:8501** while the server is running. For a new environment, use [Setup](#setup) first. Data preparation and model training are separate operations described [below](#train-comparable-model-bundles).

Predictions appear automatically. The display threshold is 50% estimated P(M). Charts use fixed 0–100% axes, contrasting colors and line patterns. Stress plots separate models into panels. The score interpretation and calibration limits are described in [Scores and calibration](#scores-and-calibration).

The model selector, predictions and charts use readable names: **HR summaries**,
**MultiRocket · clock-time fill**, and **MultiRocket · global fill**. Names also
show the day budget and identify time-split experiments; duplicate configurations
have separate run numbers. Each selected model has a description and expandable
details with its actual split, dates, customer counts, training history lengths,
ridge penalty, calibration sample and customer-level AUROC. Raw bundle IDs are
retained in CSVs alongside a `model_label` column.

Open **Methods — from uploaded HR to predictions** near the top of the app for
the implemented filtering, feature construction, validation, calibration, timeline
and perturbation methods. This explanation is available before uploading a file.

### Participant selection and perturbations

Use **Study customer** to select an ID or **Upload CSV** for your own file.
The local-file directory can be changed with `HEARTBEAT_EPOCH_DIR`; the default
resolves this repository's `out` symlink. Both routes use the same CSV preparation.
The previous 20 MB demo limit was removed; the installed Streamlit uploader's
default 200 MB limit covers the study's files (largest approximately 94 MB).

In **Stress tests**, choose dates, an intervention and 1–200 repetitions:

- Shuffle raw readings within each day, with adjustable strength.
- Shuffle observed five-minute means while preserving their multiset and mask.
- Draw replacement values from each day's empirical distribution.
- Randomly thin observations, rerunning the original adequacy gate.
- Apply random circular clock shifts within each date.

Unselected dates stay unchanged, and models remain fitted and fixed. All selected
models receive the same random inputs. Plots show the baseline, median and 5th–95th
percentile band, score-change distributions and adequate-day retention. Export all
replicates and inspect an example generated day. Raw-event interventions can also
export the first generated customer CSV; a bin-level intervention cannot produce
an equivalent raw file in general. These bands describe sensitivity conditional
on one customer/model, not population uncertainty.

**Cohort diagnostics** shows saved validation/test score distributions by F/M,
errors versus available history, and history-quartile performance. The model audit
explains the limits of these descriptive checks.

## Scores and calibration

Each bundle has a `calibration.json`: a sigmoid fitted only to saved validation
customer scores (logistic regression with fixed C=1e6). Training creates this automatically.

Calibration uses one whole/budgeted history per customer. Daily, prefix and
perturbed predictions reuse that mapping and are labelled **pseudo-probabilities**;
calibration at those lengths or under those interventions is unvalidated. Uploads
also lack a transport guarantee. Raw ridge scores remain unchanged in downloads.
The cohort diagnostics include test reliability curves, bin counts, Brier score
and log loss. Validation curves reuse sigmoid-fitting data; test results still
belong to the previously explored study cohort. Small temporal pilots calibrate
on only 33 customers and can give poor probability estimates.

Read [AUDIT_REPORT.md](AUDIT_REPORT.md) for confirmed evaluation issues, saved
results, interpretation of the stress tests, and prioritized bias checks.

## Setup

From `HEARTBEAT_LAB/`, using a Python environment compatible with numpy/numba:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[demo]'
python scripts/00_reference.py
```

The reference script copies the existing pinned checkout from this repository,
including its GPL-3.0 license. Outside this repository, first obtain the author
checkout, then pass its path with `--source`:

```sh
git clone https://github.com/ChangWeiTan/MultiRocket.git /tmp/heartbeat-multirocket
git -C /tmp/heartbeat-multirocket checkout 3ccaa4f8ed11769b904af885e17018c39c89a2ac
python scripts/00_reference.py --source /tmp/heartbeat-multirocket
```

The reusable package is installable with `pip install -e .`; the demo is run
from this source folder. A non-editable package installation needs
`HEARTBEAT_MULTIROCKET` set to the prepared reference directory. The reference
imports torch through an unused classifier wrapper; this is why torch is a
dependency even though our fitted classifier is sklearn ridge.

Streamlit supplies [CSV uploads](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
and a [local Python server](https://docs.streamlit.io/develop/concepts/architecture/run-your-app),
so a separate frontend/API/database is unnecessary here.

## Train comparable model bundles

Choose **one** preparation route. For the existing study cohort:

```sh
python scripts/01_from_repository.py artifacts/prepared
```

This uses all adequate days and the existing selected cohort; it checks exact
customer/date alignment between day statistics and HR bins. `--days 40` gives
a faster, first-40-days experiment, which must be described separately.

For new labelled data, create `private_data/manifest.csv` with one row per person:

```csv
customer_id,salutation,csv_path
person_001,10,person_001.csv
person_002,20,person_002.csv
```

Then run `python scripts/01_prepare.py private_data/manifest.csv artifacts/prepared`.
CSV paths are relative to the manifest. There must be enough customers of both
classes for stratification. This route enforces daily adequacy and two days per
person, **not** the study's longitudinal cohort eligibility. Review exclusions.

Fit the three starting options on identical customers/splits/seeds:

```sh
python scripts/02_train.py artifacts/prepared artifacts/models/summary --model summary --assignments artifacts/splits.csv
python scripts/02_train.py artifacts/prepared artifacts/models/mr_clock --model multirocket --fill clock --assignments artifacts/splits.csv
python scripts/02_train.py artifacts/prepared artifacts/models/mr_global --model multirocket --fill global --assignments artifacts/splits.csv
```

The first call creates a 70/15/15 label-stratified customer split; subsequent
calls validate and reuse it. Every option refits **all** learned preprocessing
on training customers. Validation/test customers never fit fill values, kernels,
imputation, scaling or ridge. Alpha is fixed at 1000; no test-based tuning.
Raw parsing, eligibility, representation settings and seed stay fixed when
comparing fills. Summary ridge does not use filled HR, so its fill is N/A.

Each output folder contains `model.joblib`, `metadata.json`, customer assignments
and held-out predictions. These are private/gitignored. Metadata records inputs,
split hash, versions, settings and AUROC. The app discovers completed bundles
under `artifacts/models/`; select up to three to compare. Only load bundles you
created/trust: joblib is executable serialization, not a safe upload format.

### Calibrate an existing bundle

Training writes `calibration.json` automatically. To add calibration to existing bundles without retraining:

```bash
python scripts/03_calibrate.py artifacts/models/*
```

This command writes bundle calibration files. The [score definitions and limitations](#scores-and-calibration) also apply to these calibrations.

## Retrain from the app or command line

Expand **Retrain locally and test later customers**. Choose a model, filling rule,
customer/day budgets, seed and random or temporal validation. One app-launched job
runs in a background process; use **Refresh training status** to inspect logs,
split counts and results. Completed bundles become available in the selector.
Existing models are retained.

Temporal validation uses different customers and disjoint day windows: first
adequate day before A and observations before A for training; first adequate day
and observations in [A,B) for validation; first adequate day ≥B for test. Histories
with fewer than two usable days are excluded explicitly. This is conditional on
the existing retrospective cohort, not a new prospective recruitment study.

```sh
python scripts/02_train.py artifacts/prepared artifacts/models/my_temporal_model \
  --assignments artifacts/my_temporal_splits.csv --model multirocket \
  --protocol temporal --train-before 2021-08-10 --test-from 2021-12-22 \
  --max-customers 200 --days 40
```

Measured 200-customer/40-day runs took 0.57 s for summary ridge and 2.63 s for
MultiRocket after imports with warm caches; first-run compilation can add latency. Full-history MultiRocket took about 12 minutes per
bundle with two fits running concurrently. Different budgets change both runtime
and the experiment; compare matched budgets before attributing differences to time.

## CSV and model semantics

- Simple input: `timestamp,hr`, with an explicit UTC offset on every timestamp;
  optional `end_timestamp`. No timezone guessing. Assume type-3000-like Garmin
  measurements; we do not infer a device from the waveform.
- Native input: `startTimestamp,endTimestamp,timezoneOffset,longValue,source,type`.
  UTC milliseconds; offset minutes east of UTC; source 3, type 3000.
- Valid HR: 25–230 bpm. Unknown timezone or invalid values are excluded and counted.
  Native seconds accidentally supplied as milliseconds produce an error.
- Adequate day: at least 8 observed clock-hours AND (at least 60 events OR at least
  8 hours interval-union coverage). Missing end times imply point observations.
  Intervals are assigned to their start date without clipping at midnight,
  matching the source convention. DST repeated hours collapse into clock bins.
- 288 bins/day; arithmetic mean of events beginning in each bin. Empty bins stay
  missing until model-specific training-derived filling. Preserve duplicate
  observations; no implicit deduplication or normalization.
- Summary ridge uses the original 305-column B40 recipe over the available days:
  mean/sample SD of eight daily summaries, weekday-minus-weekend mean HR, and
  per-bin observation frequency. MultiRocket adds mean/sample SD of 9,408 daily
  reference-transform features (HR and first differences): 19,121 raw columns.
- Target: recorded salutation 10 (F)→0, 20 (M)→1. Raw ridge scores are unbounded.
  The app applies validation-fitted sigmoid calibration and displays M at ≥50%.
  This can change labels relative to the old raw-score threshold of 0.5.

The cumulative chart uses days available by each date. The daily chart feeds
single-day summaries to a history-trained model: undefined static summaries use
training medians; between-day MultiRocket SD is zero. This is deliberately
labelled exploratory. Neither singleton nor short-history performance follows
from the full-history result. For validated daily probabilities, train and
calibrate a separate daily model with customer-disjoint evaluation.

## Files and implementation

```text
app.py                      Streamlit upload, preview, comparison and download
src/heartbeat_lab/
  data.py                   CSV -> local days, quality gate, 5-minute HR bins
  splits.py                 customer allocation, split checks, customer CV
  model.py                  summary ridge / MultiRocket, fill, fit, timeline
scripts/
  00_reference.py            prepare the pinned MultiRocket author code
  01_prepare.py              labelled raw CSV manifest -> customer day files
  01_from_repository.py     alternative: reuse this study's cached days
  02_train.py                train/evaluate/save one complete model bundle
  04_audit_project.py        aggregate split-exposure and nuisance-model checks
notebooks/
  01_csv_to_days.ipynb       inspect a synthetic upload and its missingness
  02_models_and_splits.ipynb compare models, explain fitting and customer CV
examples/synthetic_hr.csv    fabricated data, safe to share
tests/test_lab.py            behavioural checks
```

## Reuse in future experiments

```python
from heartbeat_lab import allocate_customers, customer_cv

assignments = allocate_customers(customer_table, seed=42)
train_ids = assignments.loc[assignments.split.eq('train'), 'customer_id']
outer_train = day_rows[day_rows.customer_id.isin(train_ids)]
for train_rows, validation_rows in customer_cv(outer_train, n_splits=5):
    # Positional indices: use .iloc. Fit a NEW complete model within each fold.
    # Never fit the transform on all outer_train before running this loop.
    pass
```

For balancing age/BMI/source, keep using the existing
[`cohort_allocator`](../cohort_allocator/README.md); it already solves that task.
Rename its ID/fold columns to `customer_id,split`, with train/val/test values,
and pass the saved CSV to `02_train.py`. The simple helpers here do not silently
substitute approximate demographic balance for that optimizer.

For another model, implement the same `fit(customers,y)`, `predict(customers)`
and `timeline(days,hr)` interface and add one explicit CLI choice. For another
fill, add it beside clock/global and retrain a complete bundle. Preserve metadata
and the score meaning; probabilities from a future classifier must be clearly
distinguished from ridge scores. No plugin framework is needed.

## Checks

```sh
python -m unittest discover -s tests -v
jupyter lab notebooks/
```

See [VALIDATION.md](VALIDATION.md) for what was actually run and what remains.
