"""Compare default-100-tree vs tuned-500-tree model runs.

Reads source_<s>/metrics_sq.json (default) and source_<s>/tuned_500/metrics_sq.json
(tuned), checks both share identical bootstrap draw IDs per variant, and writes
artifacts_sq/tuned_500_comparison.md with per-source test AUROC, bootstrap
intervals, and paired deltas side by side.

Run:  python src/sidequest/compare_models.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sq_config import ARTIFACTS_SQ, MODEL_SOURCE_IDS, VARIANTS  # noqa: E402

PAIRED = ["rec-demo", "win-rec", "roll_win-roll_rec", "all-rec"]
DELTA_LABELS = {"rec-demo": "recording vs demographics",
                "win-rec": "window vs recording",
                "roll_win-roll_rec": "rolling window vs rolling recording",
                "all-rec": "all blocks vs recording"}


def main() -> None:
    lines = ["# Default (100 trees, untuned) vs tuned (500 trees, CV-selected)",
             "",
             "Tuning: pooled 5-fold OOF AUROC on train+val, 48 configs; test "
             "split untouched and identical across both runs; identical 100 "
             "bootstrap draws. Val is part of the tuning pool in the tuned run "
             "(its metrics are not comparable).", ""]
    for src in MODEL_SOURCE_IDS:
        base = json.loads((ARTIFACTS_SQ / f"source_{src}" / "metrics_sq.json")
                          .read_text())["variants"]
        tuned = json.loads((ARTIFACTS_SQ / f"source_{src}" / "tuned_500" /
                            "metrics_sq.json").read_text())["variants"]
        dv = pd.read_csv(ARTIFACTS_SQ / f"source_{src}" / "tuned_500" /
                         "bootstrap_test_draws.csv")
        dd = pd.read_csv(ARTIFACTS_SQ / f"source_{src}" / "bootstrap_test_draws.csv")
        assert set(dv["variant"]) == set(VARIANTS) and \
            set(dd["variant"]) == set(VARIANTS)
        assert set(dv[dv["variant"] == "rec"]["draw_id"]) == \
            set(dd[dd["variant"] == "rec"]["draw_id"]), "draw IDs diverged"

        lines += [f"## Source {src} - test AUROC", "",
                  "| variant | default 100 | tuned 500 | delta | tuned boot SD | "
                  "tuned 95% pct (exploratory) |", "|---|---:|---:|---:|---:|---|"]
        for v in VARIANTS:
            b0, b1 = base[v]["test"]["auroc"], tuned[v]["test"]["auroc"]
            bt = tuned[v]["bootstrap_test"]["auroc"]
            lines.append(f"| {v} | {b0:.4f} | {b1:.4f} | {b1 - b0:+.4f} | "
                         f"{bt['bootstrap_sd']:.4f} | "
                         f"[{bt['q025']:.4f}, {bt['q975']:.4f}] |")
        lines += ["", f"## Source {src} - paired test-AUROC deltas", "",
                  "| comparison | default point | tuned point | tuned draw mean | "
                  "tuned draw SD |", "|---|---:|---:|---:|---:|"]
        for comp in PAIRED:
            d0 = base[comp.split("-")[0]]["paired_deltas_test"][comp]["auroc"]
            d1 = tuned[comp.split("-")[0]]["paired_deltas_test"][comp]["auroc"]
            lines.append(f"| {DELTA_LABELS[comp]} (`{comp}`) | {d0['point']:+.4f} | "
                         f"{d1['point']:+.4f} | {d1['draw_mean']:+.4f} | "
                         f"{d1['draw_sd']:.4f} |")
        lines += ["", "Chosen configs (per variant):"]
        for v in VARIANTS:
            bp = json.loads((ARTIFACTS_SQ / f"source_{src}" / "tuned_500" /
                             f"best_params_{v}.json").read_text())
            lines.append(f"- `{v}`: {bp['params']} (cv_auroc "
                         f"{bp['cv_auroc']:.4f})")
        lines.append("")

    out = ARTIFACTS_SQ / "tuned_500_comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"comparison written: {out}")


if __name__ == "__main__":
    main()
