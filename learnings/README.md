# Documentation guide

The [repository overview](../README.md) is the entry point. Use the documents below according to what you need to do.

| Purpose | Document |
|---|---|
| Look up metrics and definitions | [Results reference](../reports/results_reference.md) |
| Locate code and result files | [Experiment map](experiment_lineage_worktree.md) |
| Inspect verification evidence and corrections | [Results audit](../reports/results_audit_2026-10-14.md) |
| Understand the model representation | [Garmin MultiRocket](multirocket_garmin_synthesis.md) |
| Interpret recording and binning comparisons | [Recording patterns](wear_phenotypes_and_missingness_synthesis.md) |
| Run or inspect the app | [Heartbeat Lab instructions](../HEARTBEAT_LAB/README.md) |
| Review proposed analyses | [Follow-up experiments](future_experiments_backlog.md) |

## Updating experiment summaries

Use a consistent order: cohort, input history, evaluation design, models, results, limitations, and source files. Keep numerical reference separate from interpretation and execution instructions. Put a limitation beside a result when it changes what that result estimates.

Use **participant** for the person assigned to a split. Use **allocation** for one assignment and **repeated holdouts** for a set of overlapping assignments. Preserve code identifiers such as `customer_id`, `Tier 1c`, and `B40` when pointing to implementation details.

Prefer short, complete sentences and concrete verbs. Give one main point per paragraph. Keep statistical terms when they carry a precise meaning. Avoid changing terms for variety or adding an interpretation to every table cell.

This approach draws on [Diátaxis reference guidance](https://www.diataxis.fr/reference/) and [ASD-STE100](https://www.asd-ste100.org/faq.html). It uses their relevant principles; it does not claim formal STE compliance.
