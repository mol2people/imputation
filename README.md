# Recorded salutation prediction from wearable data

This repository studies how well wearable heart-rate records predict recorded salutation in the German COVID-19 Data Donation study. The original analysis maps `10 → 0/F` and `20 → 1/M`. The target is a recorded profile field, not a validated measure of biological sex or gender identity.

The experiments compare participant summaries, recording patterns, circadian features, and within-day sequences. Later work tests different history lengths, expands the Garmin cohort, and adds an Apple pilot.

## Where to start

| Need | Document |
|---|---|
| Look up results, cohorts, and evaluation designs | [Results reference](reports/results_reference.md) |
| Find an experiment's code and outputs | [Experiment map](learnings/experiment_lineage_worktree.md) |
| Check which results were verified and what limits them | [Results audit](reports/results_audit_2026-10-14.md) |
| Understand the MultiRocket representation | [Model notes](learnings/multirocket_garmin_synthesis.md) |
| Understand recording coverage and bin-size comparisons | [Recording-pattern notes](learnings/wear_phenotypes_and_missingness_synthesis.md) |
| Inspect model predictions or run the local app | [Heartbeat Lab](HEARTBEAT_LAB/README.md) |
| Review proposed work | [Follow-up experiments](learnings/future_experiments_backlog.md) |

## Main findings

Sequence models outperform the tested summary baselines in several matched experiments. On the first 40 adequate Garmin days, test AUROC is 0.7157 for summary ridge and 0.8361 for MultiRocket. Adding HYDRA gives 0.8565. See the [temporal comparison](reports/results_reference.md#garmin-first-40-adequate-days).

More adequate days improve fixed-budget Garmin results through 80 days. The all-history score of 0.9264 has a preprocessing-exposure problem: later test participants contributed to preprocessing fitted for allocation 0. Its effect on AUROC is unknown. See the [day-budget results](reports/results_reference.md#garmin-day-budgets).

The expanded Garmin cohort and Apple pilot also show gains over summary models. Their populations and available histories differ. The [results reference](reports/results_reference.md) keeps each comparison with its cohort, history budget, and evaluation design.

These are exploratory results from repeatedly studied participants. They do not establish external validity or performance for people whose salutation is missing.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Pooled analysis and source-specific RF code in `src/sidequest/` |
| `experiments/` | Individual studies, plans, implementations, reports, metrics, and local caches |
| `HEARTBEAT_LAB/` | Saved models, later cohort experiments, and the local inspection app |
| `cohort_allocator/` | OR-Tools allocation package |
| `reports/` | Results reference, audit, and historical meta-reports |
| `learnings/` | Experiment map, method explanations, and proposed follow-ups |
| `plans/` | Earlier protocols |
| `figs_ppt/` | Presentation figures and their source scripts |
| `data/`, `out/` | Local input tables and participant epoch exports |

Many inputs, predictions, caches, and fitted models are local or Git-ignored. A checkout alone cannot reproduce every result.

The [September 22 meta-report](reports/recorded_salutation_meta_report_2026-09-22.md) covers the earlier RF and circadian work. It predates the temporal and expanded-cohort results. In `temporal_garmin/results/`, unversioned files contain corrected results; `_v1` and `_v1mrbug` files preserve earlier runs with documented defects.
