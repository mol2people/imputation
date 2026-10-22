# RHR sensitivity

The fixed-date arms derive HRp5 and HRp1 from the same channel-3000 cache,
selected dates, and valid-bin mask as saved HRp10. The endpoint step recomputes
each alternative signal and its null. Inference joins the alternatives to the
existing saved HRp10 endpoints by participant, source, protocol, window, and
date hash. It does not replay HRp10. The full-cohort percentile run is under
`artifacts/rhr_sensitivity/run_20261020/`; the bounded one-task runs under
`rhr_sensitivity/` are implementation validation only and are not used for
inference.

Completed runs: `artifacts/rhr_sensitivity/run_20261020/` (HRp5/HRp1, full
cohort) and `artifacts/rhr_sensitivity/run_20261021/` (supplier RHR,
exact-date complete cases). The synthesis is
`learnings/rhr_definition_sensitivity.md`.

## Run

From the repository root, use the project Python environment. Choose a new run
directory each time; endpoint output refuses to overwrite an existing one.

```bash
RUN=experiments/infradian_age_dependency/artifacts/rhr_sensitivity/run_YYYYMMDD

python experiments/infradian_age_dependency/rhr_sensitivity/endpoints.py \
  --arms HRp5 HRp1 \
  --protocols expanded90 balanced90 \
  --null-replicates 199 \
  --output-dir "$RUN/endpoints"

python experiments/infradian_age_dependency/rhr_sensitivity/run_inference.py \
  --endpoints \
    "$RUN/endpoints/HRp5/expanded90/endpoints.parquet" \
    "$RUN/endpoints/HRp1/expanded90/endpoints.parquet" \
    "$RUN/endpoints/HRp5/balanced90/endpoints.parquet" \
    "$RUN/endpoints/HRp1/balanced90/endpoints.parquet" \
  --output "$RUN/inference" \
  --n-boot 2000
```

Inference writes matched curves, contrasts, tests, longitudinal and spectral
summaries where supported, figures, and `REPORT.md` under the output directory.
Bounded real-cache validation is available with `validate_bounded.py --limit 1`
and a new `--output-dir`; it checks fixed date hashes, 90 dates, null
convergence, and percentile order. Do not interpret a one-task run as an
age-pattern analysis.

## Supplier endpoints

Supplier arms are conditional on verifying participant/source linkage,
calendar-date meaning, and the daily value definition. Export labels alone do
not establish those facts. Keep channel 3001 `HeartRateResting`, channel 3002
`HeartRateRestingHourly`, daily type 65 `RHR`, and daily type 66 `HR during
sleep` as separate endpoints. Do not derive a daily channel-3002 value without
a stated aggregation rule.

The endpoint CLI accepts a normalized long table with columns
`user_id,source,endpoint,date,value`. `date` must already be mapped to the
experiment's calendar-day convention; keep any native channel/type and
timezone metadata as extra columns. Supply a short JSON record only after the
mapping is checked:

```json
{"status":"verified", "participant_source_linkage":"...",
 "date_semantics":"...", "daily_value_definition":"...", "units":"..."}
```

```bash
python experiments/infradian_age_dependency/rhr_sensitivity/endpoints.py \
  --supplier-only \
  --supplier-daily /path/to/normalized_supplier_daily.parquet \
  --supplier-endpoint HeartRateResting \
  --supplier-provenance /path/to/provenance.json \
  --output-dir "$RUN/supplier_endpoints"
```

Run supplier inference separately from the percentile arms. The exact-date
complete cases are a subset, and `run_secondary` uses one common cohort across
all requested arms, so passing a supplier arm alongside HRp5/HRp1 would shrink
the percentile secondary cohort. Use `--arms HRp10 RHR`:

```bash
python experiments/infradian_age_dependency/rhr_sensitivity/run_inference.py \
  --endpoints \
    "$RUN/supplier_endpoints/RHR/expanded90/endpoints.parquet" \
    "$RUN/supplier_endpoints/RHR/balanced90/endpoints.parquet" \
  --arms HRp10 RHR \
  --reference-daily \
    "$RUN/supplier_endpoints/HRp10_matched_RHR/expanded90/daily_values.parquet" \
    "$RUN/supplier_endpoints/HRp10_matched_RHR/balanced90/daily_values.parquet" \
  --output "$RUN/inference_supplier" \
  --n-boot 2000
```

`--reference-daily` is optional: the supplier arm's `daily_values.parquet`
already contains the appended matched HRp10 rows. Keep the supplier output
separate from the percentile-arm output.

This attempts exact-date complete cases first and writes the matched saved
HRp10 subset beside the supplier output. Incomplete windows are excluded from
both arms; no values are imputed. `supplier.select_common_dates` implements the
separate six-block common-date feasibility fallback. It requires
channel-3000-qualified dates and verified supplier dates, and enforces the
existing 90-date quotas. Keep its results separate from exact-date comparisons
and the fixed-date percentile arms.

## Inventory and limits

```bash
python experiments/infradian_age_dependency/rhr_sensitivity/inventory.py
python experiments/infradian_age_dependency/rhr_sensitivity/inventory.py --sample-raw-files 25
```

The default inventory reads paths and headers only. The optional sample scans
at most the first 25 raw files and reports aggregate source/channel counts; it
is not cohort coverage. `--scan-daily` streams the full daily export but still
does not establish local-date semantics or participant/window linkage. Current
channel mappings in `src/config.py` identify exported labels, not vendor
algorithms or equivalence to HRp10. Channel-3000 coverage covariates remain a
fixed observation-geometry adjustment for supplier arms.
