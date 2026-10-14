# Validation performed — 2026-09-30

This folder now includes local study-trained bundles and an interactive audit
workbench. The checks below distinguish the original scaffold from the subsequent
real-data extension.

## Real data extension

- Prepared all 3,848 customers and 1,635,698 adequate days. Trained summary ridge,
  MultiRocket/clock and MultiRocket/global on the same saved 2,692/578/578 split.
  New exploratory validation/test results are in model metadata and AUDIT_REPORT.md.
- Verified the 3,848-ID selector, original external customer CSV reading,
  training/in-sample display, actual-file upload, automatic predictions with F/M
  labels, persistent daily/cumulative switching and CSV downloads.
- Streamlit server started successfully at 127.0.0.1:8501.
- Eight unittest cases pass, covering perturbation invariants/reproducibility,
  empirical resampling, clock shifts, complete thinning/dropout, strictly truncated
  temporal windows and the original parsing/allocation/serialization checks.
- Efficient running-moment timelines agree with independent singleton/prefix
  predictions for summary, MultiRocket/clock and MultiRocket/global synthetic models.
- All five stress-test modes passed Streamlit AppTest on a real customer, with
  three repetitions each. Changing the score view retains completed runs. This
  verifies UI integration in addition to the numerical invariant tests.
- The real background-training button launched and completed a 200-customer,
  at-most-40-day temporal summary job while predictions remained available.
  An equivalent MultiRocket CLI run also completed and recorded timings.
- Aggregate historical split exposure and simple calendar/recording-only controls
  were computed; the report links their reproducible script. A paired customer
  bootstrap was run for the two new full-history filling variants.
- The initial perturbation test exposed a pandas set-to-datetime conversion error;
  converting the selected set to a list fixed it, and all tests subsequently passed.

## Passed

- Editable package build/install in `HEARTBEAT_LAB/.venv`. This local environment
  inherits the existing datenspende scientific stack; Streamlit 1.64.0 and
  notebook validation tools were installed locally. No clean-environment claim.
- Four unittest cases: native/simple CSV equivalence and local date handling;
  adequacy, missingness and invalid HR; deterministic customer allocation,
  expected 70/15/15 counts and customer-disjoint CV; summary-model serialization
  and cumulative predictions independent of future days.
- Both notebooks validate against the notebook schema. All code cells executed
  sequentially in a headless Python process. The second notebook fitted all
  three choices with the full 1,250-feature-per-transform MultiRocket setting.
  Synthetic models serialized and reloaded with identical prediction arrays.
- Raw CSV CLI preparation for 40 synthetic customers, followed by all three
  training CLI options. Bundles shared the same saved assignment hash and
  produced metadata, held-out predictions and fitted models.
- Streamlit AppTest: initial page; actual synthetic CSV upload; selecting three
  bundles; cumulative and individual-day views; both download controls;
  malformed CSV shown as an error instead of crashing. Temporary synthetic
  bundles were removed afterward. A first chart assertion used an obsolete test
  element name; inspecting the actual `vega_lite_chart` elements resolved that
  test-harness issue. No browser screenshot or remote-host test was run.
- Existing full-history day/bin cache alignment: 3,848 customers and 1,635,698
  rows, with matching customer/date order.
- One real customer, selected for a small raw file: upload parser vs historical
  cache across all **94 adequate days**. Exact date and observation-mask equality;
  HR bins match at rtol/atol 1e-6, eight daily summaries at rtol/atol 1e-5. No
  customer identifier or raw values were exported into this folder.
- Python syntax/compile checks, notebook JSON/schema, source whitespace review.
  Existing unrelated worktree files were left unchanged.

Small weekday-only unit fixtures emit sklearn's expected warning about dropping
an entirely missing weekday/weekend contrast. The pipeline handles that feature
consistently using its training-fitted imputer. Headless notebook plotting emits
the expected noninteractive-backend warning.

## Percentage display and calibration update

- All six existing bundles now have validation-only sigmoid calibration sidecars;
  original model files and raw-score CSVs are unchanged. Test Brier scores are
  0.1066 (MultiRocket clock), 0.1035 (global), and 0.1939 (summary). The small
  temporal pilots are substantially worse (0.3619–0.3706).
- Ten unit tests pass, including test-label/score independence of the calibrator,
  extreme-score stability, preserved NaNs, and removal of all readings on only
  selected dates. The latter exposed and fixed an empty-frame dtype issue.
- Streamlit AppTest passed for three-model comparison, repeated shuffling, daily
  and cumulative view changes, and complete thinning of selected dates. Chart
  specifications confirm separate stress panels and fixed percentage axes.
- A new 200-customer temporal summary training run produced calibration
  automatically. Downloads retain raw scores alongside mapped probabilities.
- Colors/line patterns are explicitly specified; no manual browser rendering was
  checked. Whole-history calibration does not validate daily/prefix/perturbed
  probabilities. Those displays explicitly say pseudo-probability.

## Remaining limits

- External independent predictive performance of these bundles; the current
  results still use the historically explored selected cohort.
- Raw-to-cache equivalence across the entire cohort; one-person parity is limited
  evidence and does not establish every timezone/interval edge case.
- Clean Python installation across versions/platforms; package metadata gives
  compatibility ranges, not a fully pinned cross-platform lockfile.
- Interactive Jupyter frontend/kernel launch (code cells themselves did execute).
- External probability calibration, singleton-day accuracy, short-history validity,
  transport to other devices/populations, and performance under concurrent load.
- A live remote deployment, manual browser rendering, and worst-case long-upload
  latency/memory. AppTest covers the UI programmatically.

Synthetic AUROC values are intentionally not advertised as model performance. Retain the
historical endpoint's cross-split transform-exposure caveat when presenting it.
