# Disambiguation of the 25–35 day infradian HR excess

Tests whether the young-female 25–35-d band excess in `HRp10` is attributable to lunar or
seasonal structure. Read `PLAN.md` first; Addendum 1 records why the original lunar-regression
design was not run.

## What was established before the run

- The lunar + seasonal design is a **band-stop** in 27.5–30.5 d (99.2–99.7 % absorption of any
  sinusoid there). It cannot separate a lunar from a menstrual rhythm.
- Separating 28.5 d from 29.53 d needs a record span ≥ 820 d; the caches provide ≤ 905 d and
  typically 400–680 d.
- The weekly harmonics are already absorbed by the weekday dummies (5.6e-12).
- The seasonal block alone absorbs 2.3–2.5 % in band: identified, usable.

## Executed analyses

| Script | What it does | Cost |
|---|---|---|
| `validate.py` | lunar phase model vs eclipse epochs; spectral peaks; weekly redundancy; recovery calibration; engine consistency | ~1 min |
| `phase_lock.py compute` | per-window lunar/anomalistic direction angle, rotation null, sham periods, free-period acrophase, cross-participant coherence spectrum | ~3 min, 8 workers |
| `phase_lock.py analyze` | Rayleigh/rotation tests by stratum, T2/T3 controls, Holm | ~1 min |
| `male_bound.py` | Q_male / Q_young-female bound from saved endpoints | ~1 min |
| `coherence.py` | R̄(T) per stratum from the angle archive | ~1 min |
| `null_coherence.py` | red/white-noise null for R̄(T) on the real date grid | ~10 min |
| `placebo_bands.py` | check A: F/M amplitude ratio at 18/48/55/60 d vs in band | ~1 min |
| `coverage_coherence.py` | check B: raw clock-coverage R̄(T) vs HR | ~1 min |
| `seasonal_arm.py` | S-only seasonal/trend arm (τ², τ³), both protocols | ~15 min |
| `figures.py` | periodogram figures 1–4 from stored artefacts | ~20 s |

## Run

```bash
PY=HEARTBEAT_LAB/.venv/bin/python
RUN=experiments/infradian_age_dependency/artifacts/disambiguation/run_$(date +%Y%m%d)

$PY experiments/infradian_age_dependency/disambiguation/validate.py
$PY experiments/infradian_age_dependency/disambiguation/phase_lock.py compute \
    --protocols balanced90 expanded90 --workers 8 --run "$RUN"
$PY experiments/infradian_age_dependency/disambiguation/phase_lock.py analyze \
    --run "$RUN" --protocols balanced90 expanded90
$PY experiments/infradian_age_dependency/disambiguation/coherence.py --run "$RUN"
$PY experiments/infradian_age_dependency/disambiguation/placebo_bands.py --run "$RUN"
$PY experiments/infradian_age_dependency/disambiguation/coverage_coherence.py --run "$RUN"
$PY experiments/infradian_age_dependency/disambiguation/seasonal_arm.py compute --workers 9 --run "$RUN"
$PY experiments/infradian_age_dependency/disambiguation/seasonal_arm.py analyze --run "$RUN"
$PY experiments/infradian_age_dependency/disambiguation/figures.py --run "$RUN"
$PY experiments/infradian_age_dependency/disambiguation/male_bound.py
$PY experiments/infradian_age_dependency/disambiguation/null_coherence.py --n 1500
```

## Outputs

- `calibration_recovery_{protocol}.csv` — recovery of a unit sinusoid vs period.
- `<run>/phase_records_{protocol}.parquet` — one row per window.
- `<run>/phase_coherence_{protocol}.npz` — cross-participant coherence spectrum R̄(T).
- `<run>/phase_lock_summary.csv`, `<run>/phase_lock_provenance.json`.
- `<run>/placebo_bands.csv` — check A, out-of-band F/M ratio and relative excess.
- `<run>/coverage_coherence.csv` — check B, coverage R̄(T) and implied slope.
- `<run>/seasonal_arm.csv`, `<run>/seasonal_design_recovery.csv` — S-only arm and its design absorption.
- `<run>/figures/` — `fig1` excess periodogram, `fig2` coherence, `fig3` placebo ratio,
  `fig4` coverage coherence (PNG + PDF), plus `CAPTIONS.md`.
- `male_bound.csv`, `male_bound_strata.csv`.

## Tests and their nulls

- **T1 (primary).** Direction of the (cos ψ, sin ψ) and (cos ν, sin ν) coefficients. Null:
  rotation of the phase basis by 32 random offsets per window (calibrated and reported as
  `p_rot`); Rayleigh p reported for reference only.
- **T2.** Acrophase at the freely estimated period mapped to ψ and ν. Null: acrophase uniform
  inside each window (400 draws).
- **T3.** Day-of-week of the acrophase, χ²(6).
- **Specificity.** `sham28`, `sham29` and the full R̄(T) spectrum; a genuine lunar alignment
  should stand out at 29.53 d against its immediate neighbours, an artifact should not.
