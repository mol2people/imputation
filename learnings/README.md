# Repository Learnings & Model Synthesis

This directory consolidates findings, model specifications, evaluation audits, and architectural details extracted from the `datenspende_IMPUTE_sex` repository.

## Documents

- [`multirocket_garmin_synthesis.md`](multirocket_garmin_synthesis.md): the >0.90 AUROC MultiRocket models on the 3,848-user strict Garmin cohort (no MiniRocket implementation exists in the repo). Contents:
  - Test AUROC by day budget ($k = 4 \ldots$ all) and per-allocation detail
  - The 84-kernel / 14-dilation configuration
  - Feature widths: 9,408 per day → 19,121 per participant before hygiene, fed to `Ridge` regression
  - Pre-trained bundles in `HEARTBEAT_LAB/artifacts/models/`
  - Shuffle control, representation ladder, and evaluation-exposure findings

- [`experiment_lineage_worktree.md`](experiment_lineage_worktree.md): complete visual DAG/worktree diagram of all experiments (ran, running, planned), hardware divergence points (Garmin continuous vs Apple point-events), and milestone results.

- [`future_experiments_backlog.md`](future_experiments_backlog.md): prioritized backlog of planned investigations, including:
  - Garmin temporal bin size & granularity ablation (1-min, 2.5-min, 5-min, 10-min, 15-min)
  - Cross-device transfer (Garmin → Apple Watch zero-shot)
  - Apple Channel 3002 hourly resting heart rate modeling
  - Full-scale Apple cohort scaling (N = 11,143)
  - Garmin Tier 2 multi-source device-transition expansion (N = 5,636)

All numbers are read from saved outputs; nothing was re-run.
