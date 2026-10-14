# Informative recording-feature experiment

This completed experiment asks whether recording-availability features add predictive information beyond HR values and context. It uses pooled device sources and compares the feature blocks within the same cohort and repeated participant splits.

In the ungated cohort, recording + values gives test AUROC **0.7239**, with a mean paired gain of **+0.0167** over native-NaN values. These are the reported results summarized in the [results reference](../../reports/results_reference.md#earlier-summary-and-feature-experiments). They concern the tested feature blocks and evaluation population.

The detailed report and paired comparisons are local files under `output/`: `report.md` and `paired_auroc_comparisons.csv`. The [experiment index](../../learnings/experiment_lineage_worktree.md) places this study among the earlier summary-feature experiments.

## Implementation and execution

Standalone implementation of [the experiment plan](../../plans/CODEX_INFORMATIVE_RECORDING_FEATURE_PLAN_2026-09-20.md).
It reads only the frozen v1 aggregate parquet/CSV artifacts in `../artifacts/` and writes
only under `output/` in this directory.  It does not import, edit, or regenerate the
existing analysis code or artifacts.

Run with the project environment:

```sh
/Users/bulat/micromamba/envs/datenspende/bin/python run_experiment.py --mode all
```

`--mode all` runs a seeded stratified smoke test twice, asserts identical sampled IDs,
splits, and hard predictions, and compares probabilities/metrics within `1e-12` before
starting the 30 repeated 80/20 splits.
The smoke sample has 200 randomly selected participants in each U `y x gate_pass`
stratum, using the fixed study seed.  The full analysis uses the complete frozen cohorts;
its random participant draws are the prespecified seeded stratified train/test splits.

The implementation is intentionally limited to the declared feature blocks, fixed
100-tree random forest, frozen inputs, and report tables.  It does not tune models,
modify existing functions, rescan raw epoch files, or start a new feature-selection loop.
