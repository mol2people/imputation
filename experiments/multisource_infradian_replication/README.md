# Multisource infradian replication — implementation

This directory implements the staged analysis in [`PLAN.md`](PLAN.md). It is a
new run namespace: it reads historical raw exports, caches, and protocol
primitives, but does not overwrite historical
`infradian_age_dependency`, Apple, Garmin, or Samsung artifacts.

Status: implementation and bounded validation are complete. The full
`B=199` null / `2,000`-replicate bootstrap analysis has **not** been executed
here; see [Compute cost](#compute-cost).

## Scope and separation

Primary sources: Fitbit (2), Garmin (3), Apple (6), Samsung (7), Nokia (9),
Google Fit (13), Oura (19). Polar (4) is excluded. Inventory-only source IDs
46 and 48 are retained in inventory and excluded from endpoint analysis.

The three analyses are separate estimands and are never pooled:

| Analysis | Data | Endpoint |
|---|---|---|
| A | channel-3000 5-minute dense bins | `HRp10` with the historical coverage-adjusted nuisance |
| B | daily type 65 | `RHR` with the daily nuisance (no clock coverage) |
| C | daily type 66 | `HR_during_sleep` with the daily nuisance (no clock coverage) |

Daily arms use one resolved finite value in `[25, 230]` bpm per exported
calendar date. There is no imputation. Exact duplicate rows are collapsed;
conflicting values for the same `(user_id, source, endpoint, date)` key are
excluded, never averaged.

## Environment

Use the datenspende interpreter for all data work:

```bash
PY=/Users/bulat/micromamba/envs/datenspende/bin/python
```

The upstream protocol primitives (`expanded90`, `balanced90`, block sampler,
SHA-256 seeding, age conventions, RCS basis) are imported from
`experiments/infradian_age_dependency/scripts/common.py`. The endpoint
spectral/REML engine is imported from the historical endpoint script.

## Quick start

Run every stage under a new namespace:

```bash
cd experiments/multisource_infradian_replication
$PY scripts/run_all.py \
  --run-dir artifacts/runs/full_20261021 \
  --workers 9 \
  --null-replicates 199 \
  --n-boot 2000
```

`run_all.py` refuses to write into an existing run directory unless
`--overwrite` is passed. The default computation uses the primary window
(`window_index=1`); add `--include-repeat-windows` to compute and model the
secondary repeat window as well.

Partial daily-only run (useful for a bounded test):

```bash
$PY scripts/run_all.py \
  --run-dir artifacts/runs/daily_20261021 \
  --sources 3 --protocols expanded90 \
  --workers 9 --null-replicates 6 --n-boot 100 \
  --limit 3000 --skip-channel
```

The `--limit` argument bounds channel cache builds and endpoint tasks per arm.
It is intended for smoke tests, not for a substantive analysis.

## Stages

| Script | Output |
|---|---|
| `00_inventory.py` | `provenance.json`, daily and channel inventories, `daily_clean.parquet` |
| `01_build_caches.py` | source-tagged `customer_{user_id}_source_{source}.npz` caches, `raw_index.parquet`, `cache_index.parquet` |
| `02_build_manifests.py` | qualifying dates, `expanded90`/`balanced90` windows, eligibility, age support, demographics audit |
| `03_compute_endpoints.py` | per-arm endpoint rows, two-component nulls, `endpoints/endpoints.parquet` |
| `04_fit_models.py` | female RCS and sex-interaction RCS fits, HC3 chunk tests, support tables |
| `05_bootstrap.py` | participant bootstrap curves, simultaneous confidence bands, transition-metric distributions |
| `06_matched.py` | exact-date matched cohort, four paired-comparison safeguards, paired-difference intervals |
| `07_report.py` | `REPORT.md` |

Each stage can also be run directly; see `--help` for arguments.

### Channel caches

`01_build_caches.py --scan-mode all` scans every `out/*.csv` once for
channel-3000 rows. This is the exhaustive cohort source for Fitbit and Oura,
which are absent from the epoch metadata. Source-tagged caches are built when
no source-specific historical cache exists. Existing Apple/Garmin/Samsung raw
5-minute caches are reused read-only; endpoints and nulls are recomputed in the
new run namespace.

`--scan-mode candidates` is faster but conditions channel availability on the
daily/epoch candidate list. Use it only for debugging.

Channel-3000 row presence does not imply dense-day eligibility. In this export,
Fitbit type-3000 rows are often one 5-minute bin per day, so Analysis A Fitbit
is likely non-estimable; Oura channel coverage can also be sparse. The cache
index and channel eligibility tables report the actual retained counts rather
than assuming an arm is estimable from inventory alone.

### Models and support

Female-only RCS curves are fitted on valid women, then masked to five-year
bands with at least 30 women. Sex-interaction RCS models are fitted on the
shared-support bands with at least 30 women and 30 men per band, and at least
three such bands. Predictions and bootstrap bands outside supported bands are
`NaN`. Transition locations at the edge of the supported/search range are
reported as undefined rather than as boundary artifacts.

### Bootstrap

Participant resampling is stratified by five-year age band and sex. The
primary chunk-test inference is the HC3 Wald test in
`models/chunk_tests.csv` (`p_chi2`, `p_f`). `bootstrap/bootstrap_chunk_summary.csv`
reports the participant-resampling distribution of the fitted Wald statistic
(median, 2.5%/97.5% quantiles); these are **not null p-values** and must not be
interpreted as such. Individual resampled curve prediction arrays are saved in
`bootstrap/bootstrap_curves.npz` (shape `(n_boot, n_ages)` float32 matrices per arm/scope),
and pointwise/simultaneous confidence bands are in
`bootstrap/confidence_bands.parquet`. Detailed computational optimizations are
documented in [`EFFICIENCY_AUDIT.md`](EFFICIENCY_AUDIT.md).

### Matched comparisons

`06_matched.py` restricts to participants with the same source and both
endpoints on the same selected dense dates. It recomputes both series under
the same reduced nuisance design, fits a separate endpoint-specific null for
each arm, and enforces:

1. a common participant set with finite endpoints and converged nulls in every arm;
2. age support recomputed on that common set;
3. identical bootstrap participant draws across arms;
4. arm-specific failure/retention counts and paired-difference intervals.

Exact-date matched cohorts are often sparse. Empty comparisons are reported as
empty, not silently substituted with independently selected dates.

## Outputs

Run artifacts are written under `artifacts/runs/<name>/`:

- `manifests/`: eligibility, selected dates, windows, age support, demographics audit.
- `endpoints/`: per-arm endpoint rows and combined `endpoints.parquet`.
- `models/`: `model_summary.csv`, `chunk_tests.csv`, `age_support.csv`, `curves.parquet`.
- `bootstrap/`: curves, confidence bands, chunk summaries, transition metrics.
- `matched/`: requirements, arm status, retention, paired contrasts.
- `REPORT.md`: synthesis with provenance, inventories, eligibility, models, bootstrap, and matched sections.
- `stage*_summary.json`: machine-readable stage summaries.

## Compute cost

The full plan is compute-heavy:

- `B=199` null draws per endpoint arm. The prespecified bootstrap is `2,000`
  participant resamples; the full endpoint run used `n_boot=1,000` by user
  decision.
- Primary window only by default; adding `--include-repeat-windows` roughly
  doubles endpoint cost.
- Both `expanded90` and `balanced90` are computed by default. Restricting
  `--protocols expanded90` roughly halves endpoint cost.

On an 8–9 core machine, the final bounded validation (`--limit 2000`,
`--null-replicates 50`) completed in 6m39s with 18,696 Stage-3 tasks, confirming
worker parallelism. The unbounded primary-window run
(`full_b199_boot1000_20261022`, `B=199`, `n_boot=1000`, no `--limit`) completed
in **28m23s** with **65,449** Stage-3 tasks (**43,855** daily + **21,594**
channel), of which **65,443** converged. Stage 1b cache construction took about
**10m48s**, Stage 3 about **11m32s**, Stage 5 about **18s**, and Stage 6 matched
about **4m40s**; the run directory is **727 MB**. The earlier 46,430-task
estimate used the bounded channel count (2,575) and is superseded. Repeat
windows add further tasks and materially increase Stage-3 cost. Plan for a
dedicated run namespace and do not start it in an interactive foreground shell.

## Tests

`pytest` is not installed in the datenspende environment. Run the plain-assert
tests directly:

```bash
$PY -m compileall -q multisource scripts tests
$PY tests/test_pipeline.py
```

The suite covers protocol selection, daily QC and duplicate policy, daily
endpoint determinism, source/value-type namespaces, support masks, transition
boundaries, identical-draw paired bootstrap behavior, and exhaustive raw
scanning.

## Known limitations

- The optional common-date reselection described in `PLAN.md` is not
  implemented. Exact-date matching is the only matched cohort.
- Optional matched `HRp5`/`HRp1` sensitivities are not implemented; matched
  contrasts use `HRp10`.
- Unresolved source IDs 46 and 48 remain inventory-only.
- The full `B=199`, `n_boot=1,000` primary-window run is
  `artifacts/runs/full_b199_boot1000_20261022/`. It completed in **28m23s**
  with **65,449** Stage-3 rows and **65,443** converged. The prespecified
  bootstrap count was 2,000, so the full run's bootstrap precision is lower
  than the protocol reference.
- In that run, **6/65,449 endpoint rows** are flagged `rss0_degenerate`
  (`null_status=failed`): source 2 user `1124280` (RHR, both protocols) and
  source 6 user `1222449` (RHR and HRp10, both protocols). The user will inspect
  the source data. No remediation, exclusion, or rerun is planned for this run;
  these rows must not be treated as converged. See
  `artifacts/runs/full_b199_boot1000_20261022/RUN_NOTES.md`.
- Exact-date matched comparisons are non-empty only for sources 3 and 6. Source
  9 has zero paired rows, and sources 2, 7, 13, and 19 have no exact-date
  complete cases.
- Bootstrap chunk summaries are descriptive distributions, not null tests;
  use the HC3 chunk tests for inference.
