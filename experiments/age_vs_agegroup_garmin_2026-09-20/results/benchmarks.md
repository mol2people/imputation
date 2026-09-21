# Pre-run checks and engineering log — age_vs_agegroup_garmin

## Determinism / parity checks (before the official run)

1. **REF50 bit-exactness (n_jobs=1, repeat 0).**
   A stripped-down replica of the REF50 arm (frozen FS-phase A3_k50
   selection: same split seed `SeedSequence([20260919, 3, 0])`, same PI
   seeds, same G1 RF seed slot 3) reproduced the saved A3_k50 result
   exactly: max |ΔAUROC| = 0.00e+00 and identical top-50 column set.
   Runtime ≈ 14 s. This is the pre-registered validation gate (PLAN.md §6);
   the official run re-checks it internally (same gate, 30/30 repeats).

2. **Timing / determinism across n_jobs (all arms, repeat 0).**
   One exploratory pass with `n_jobs=-1` (10 cores) to calibrate runtime:
   REF50 ≈ 8 s, AGE50 / AGE50_nodemo ≈ 6 s, AGE_all ≈ 2 s per fit. Numbers
   identical to the n_jobs=1 replica — RF under a fixed `random_state` is
   deterministic regardless of `n_jobs`, so the official 5-worker layout
   (5 × n_jobs=1) adds no nondeterminism.

## Official run

5 joblib workers, n_jobs=1 per fit, 5 arms × 30 repeats: 300 s wall
(2026-09-20T18:02:12 → 18:07:12). See `run_log.txt` and `age_repro.json`.

## Engineering log (aborted attempts before the clean run)

- First full-run attempt completed all compute but crashed in the
  results-assembly step (prediction-array vs user-list shape mismatch in
  the force-arm path). No results were written; fixed with an explicit
  guard, then rerun from scratch — the committed results are from the
  clean rerun only.
- Cosmetic, not rerun: the report generator emits `nan` for two
  Table 1 cells that are undefined by construction (AGE50_force shares
  AGE50's hygiene fit; AGE_all's final fit IS the hygiene fit). Patched by
  hand with a footnote instead of burning another 300 s; CSVs untouched.

## Reproducibility

`age_repro.json` records script sha256, library versions, config, input
hashes and the validation line. Inputs are read-only repo artifacts; the
experiment writes only inside this folder.
