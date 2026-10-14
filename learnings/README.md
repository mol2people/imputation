# Documentation guide

The [repository README](../README.md) gives the experimental work tree, device sources, and main findings. The documents below serve different reading tasks.

| Task | Start here |
|---|---|
| Look up a result, cohort, or model name | [Results reference](../reports/results_reference.md) |
| Locate a study’s report, code, and outputs | [Experiment index](experiment_lineage_worktree.md) |
| Understand the MultiRocket inputs and feature construction | [Model notes](multirocket_garmin_synthesis.md) |
| Interpret recording coverage and bin-size comparisons | [Recording-pattern notes](wear_phenotypes_and_missingness_synthesis.md) |
| Inspect a saved model or use the local app | [Heartbeat Lab](../HEARTBEAT_LAB/README.md) |
| Read the completed verification record | [Results audit](../reports/results_audit_2026-10-14.md) |
| Review proposed experiments | [Follow-up experiments](future_experiments_backlog.md) |

## Current summaries and historical records

The results reference is the current cross-experiment numerical summary. Individual experiment reports retain the detailed methods, tables, and run history. Where their interpretation differs, read the qualifications in the current reference and linked audit.

`PLAN.md` records intended work; it does not establish that the work ran. Dated reports, validation logs, and retired context files describe the state at that time. The [September synthesis](../reports/recorded_salutation_meta_report_2026-09-22.md) and [temporal roadmap](../experiments/temporal_garmin/NEXT_STEPS.md) remain useful historical records. Use the current backlog for proposed work.

Saved inputs, predictions, and fitted models may be local or Git-ignored. Mark such paths as local when linking them. Distinguish “not saved by this experiment” from “not present in this checkout.” Reading documentation does not require regenerating these artifacts.

## Writing an experiment summary

Open with the scientific question and the observed result. Define the comparison before interpreting it: device/source and channel, participant cohort, history budget, models, split design, and unit of evaluation. Give the reference arm and the direction of any reported difference. Name the uncertainty measure and the population or repetitions it describes.

Use a short methods paragraph or compact fields when these make the design easier to inspect. Follow with the result table and the interpretation it supports. Put a limitation beside a result when it changes the conclusion. Separate proposed explanations from observed findings. Link the implementation and saved outputs; identify which are local.

Use **participant** for the person assigned to a split, **allocation** for one assignment, and **repeated holdouts** for overlapping assignments. Define feature-block names at first use or link the [shared definitions](../reports/results_reference.md#model-and-experiment-names). Preserve identifiers such as `customer_id`, `Tier 1c`, and `B40` when referring to code.

## Maintaining the summaries

Keep detailed results in the experiment report. When a new result changes the overview, update its row or section in the results reference, its entry in the experiment index, and the relevant node or finding in the README. Move completed proposals out of the active backlog. Preserve historical reports and label their scope instead of silently rewriting them as current accounts.

Use [Diátaxis](https://www.diataxis.fr/reference/) to separate reference, explanation, and instructions when that separation helps the reader. Use [ASD-STE100](https://www.asd-ste100.org/faq.html) principles for direct sentences and consistent terms while retaining scientific precision. These are writing principles, not a fixed template or a claim of formal STE compliance. Short notes and operational updates usually need only plain prose.
