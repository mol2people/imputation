#!/usr/bin/env python3
"""Build the final meta-report from frozen experiment artifacts.

This is a reporting-only program. It reads saved metrics and cohort tables,
creates figures, and writes a self-contained local HTML document. It does not
fit a model or modify any experiment artifact.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR = Path("/private/tmp/codex_meta_report_mplconfig")
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import t  # noqa: E402


OUT_HTML = REPORT_DIR / "recorded_salutation_meta_report_2026-09-20.html"
OUT_PDF = REPORT_DIR / "recorded_salutation_meta_report_2026-09-20.pdf"
VARIANTS = ["demo", "rec", "win", "roll_rec", "roll_win", "all"]
VARIANT_LABELS = ["Demo", "Recording", "91-day", "Rolling\nfull", "Rolling\n91-day", "All"]
SOURCE_NAMES = {3: "Garmin / source 3", 6: "Apple / source 6", 7: "Samsung / source 7"}

COLORS = {
    "navy": "#12324a",
    "blue": "#2676a8",
    "teal": "#2a9d8f",
    "gold": "#e9a23b",
    "red": "#c95c54",
    "gray": "#73808a",
    "light": "#e8eef2",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def savefig(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def signed(x: float, digits: int = 4) -> str:
    return f"{x:+.{digits}f}"


def table(headers: list[str], rows: list[list[object]], classes: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(v))}</th>" for v in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
    return f'<table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def paired_effect(
    frame: pd.DataFrame,
    source: int,
    variant: str,
    arm: str,
    base_arm: str,
    confidence: float,
) -> dict:
    sub = frame[(frame["source"] == source) & (frame["variant"] == variant)]
    base = sub[sub["arm"] == base_arm].sort_values("repeat")["auroc_test"].to_numpy()
    cur = sub[sub["arm"] == arm].sort_values("repeat")["auroc_test"].to_numpy()
    if len(base) != len(cur) or len(base) == 0:
        raise ValueError(f"missing paired rows: source={source}, variant={variant}, arm={arm}")
    delta = cur - base
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1))
    se = sd / np.sqrt(len(delta))
    critical = float(t.ppf(1 - (1 - confidence) / 2, len(delta) - 1))
    return {
        "mean": mean,
        "sd": sd,
        "se": se,
        "lo": mean - critical * se,
        "hi": mean + critical * se,
        "n": len(delta),
    }


def build_figures() -> dict[str, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )
    figures: dict[str, Path] = {}

    # Figure 1: v1 participant flow.
    labels = [
        "Daily candidates",
        "Allowed-source daily record",
        "Non-empty epoch file",
        "Valid allowed measurement",
        "Observed binary salutation",
        "Pass 14-day coverage gate",
    ]
    counts = np.array([27256, 26939, 25806, 25803, 25784, 20485])
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    y = np.arange(len(labels))
    shades = ["#d7e4ec", "#c7dce8", "#b3d2e3", "#9bc5dc", "#75afd0", COLORS["blue"]]
    ax.barh(y, counts, color=shades, edgecolor="white", height=0.72)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 29200)
    ax.set_xlabel("Participants")
    ax.set_title("Pooled v1 cohort construction")
    ax.grid(axis="x", color="#dfe5e8", linewidth=0.8)
    for yi, n in zip(y, counts):
        ax.text(n + 250, yi, f"{n:,}  ({n/counts[0]:.1%})", va="center", fontsize=9)
    ax.text(
        0,
        1.08,
        "Mutually exclusive losses: 317 no allowed daily source; 1,133 zero-byte epoch file; "
        "3 no valid measurement; 11 missing labels; 8 non-binary labels; 5,299 below gate.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["gray"],
    )
    figures["flow"] = savefig(fig, "01_pooled_cohort_flow.png")

    # Figure 2: strict source-specific cohorts.
    cohort = pd.read_csv(ROOT / "experiments" / "artifacts_sq" / "source_cohorts.csv")
    cohort = cohort[cohort["source_id"].isin([3, 6, 7])].copy()
    fig, ax = plt.subplots(figsize=(9.2, 3.7))
    y = np.arange(len(cohort))
    ax.barh(y, cohort["base_single_source"], height=0.58, color="#d8dee2", label="v2 single-source base")
    ax.barh(y, cohort["sal10"], height=0.36, color=COLORS["gold"], label="strict pass: salutation 10")
    ax.barh(
        y,
        cohort["sal20"],
        left=cohort["sal10"],
        height=0.36,
        color=COLORS["blue"],
        label="strict pass: salutation 20",
    )
    ax.set_yticks(y, [SOURCE_NAMES[int(s)] for s in cohort["source_id"]])
    ax.invert_yaxis()
    ax.set_xlabel("Participants")
    ax.set_title("Strict 91-day, sole-source cohorts")
    ax.grid(axis="x", color="#e2e7ea", linewidth=0.8)
    for yi, row in enumerate(cohort.itertuples(index=False)):
        rate = row.any_core_pass / row.base_single_source
        ax.text(row.any_core_pass + 130, yi, f"{row.any_core_pass:,} pass ({rate:.0%})", va="center", fontsize=9)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.38), fontsize=8.5)
    figures["strict_cohort"] = savefig(fig, "02_strict_source_cohorts.png")

    # Figure 3: default and tuned test AUROC on the frozen source-specific split.
    default: dict[int, list[float]] = {}
    tuned: dict[int, list[float]] = {}
    for source in (3, 6, 7):
        d0 = read_json(ROOT / "experiments" / "artifacts_sq" / f"source_{source}" / "metrics_sq.json")
        d1 = read_json(ROOT / "experiments" / "artifacts_sq" / f"source_{source}" / "tuned_500" / "metrics_sq.json")
        default[source] = [float(d0["variants"][v]["test"]["auroc"]) for v in VARIANTS]
        tuned[source] = [float(d1["variants"][v]["test"]["auroc"]) for v in VARIANTS]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.15), sharey=True)
    x = np.arange(len(VARIANTS))
    for ax, source in zip(axes, (3, 6, 7)):
        ax.plot(x, default[source], marker="o", linewidth=1.8, color=COLORS["gray"], label="Default RF")
        ax.plot(x, tuned[source], marker="o", linewidth=2.2, color=COLORS["blue"], label="CV-tuned, 500 trees")
        ax.axhline(0.5, color="#aeb7bd", linewidth=1, linestyle="--")
        ax.set_xticks(x, VARIANT_LABELS, rotation=35, ha="right", fontsize=8.2)
        ax.set_title(SOURCE_NAMES[source])
        ax.set_ylim(0.50, 0.81)
        ax.grid(axis="y", color="#e3e8eb", linewidth=0.8)
        if source == 3:
            ax.set_ylabel("Frozen-test AUROC")
        if source == 7:
            ax.text(0.98, 0.05, "test n = 54", transform=ax.transAxes, ha="right", fontsize=8, color=COLORS["red"])
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Per-source representation results: one frozen test split", y=1.03, fontsize=13)
    figures["auroc"] = savefig(fig, "03_source_variant_auroc.png")

    # Figure 4: corrected simultaneous intervals for repeated-split primary families.
    fs = pd.read_csv(ROOT / "experiments" / "artifacts_sq" / "fsplit" / "fsplit_metrics.csv")
    fe = pd.read_csv(ROOT / "experiments" / "artifacts_sq" / "feng" / "feng_metrics.csv")
    effects: list[tuple[str, str, dict]] = []
    for source in (3, 6):
        effects.append((f"FS  s{source} all: A2 - A0", "FS", paired_effect(fs, source, "all", "A2", "A0", 0.975)))
    for source in (3, 6):
        for arm in ("F1", "F2"):
            effects.append(
                (f"FE  s{source} all: {arm} - B0", "FE", paired_effect(fe, source, "all", arm, "B0", 0.9875))
            )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    yy = np.arange(len(effects))[::-1]
    for yi, (label, family, eff) in zip(yy, effects):
        color = COLORS["teal"] if family == "FS" else COLORS["blue"]
        ax.errorbar(
            eff["mean"],
            yi,
            xerr=[[eff["mean"] - eff["lo"]], [eff["hi"] - eff["mean"]]],
            fmt="o",
            color=color,
            capsize=4,
            linewidth=1.8,
            markersize=6,
        )
        ax.text(eff["hi"] + 0.00035, yi, f"{eff['mean']:+.4f}", va="center", fontsize=8.5)
    ax.axvline(0, color="#454d52", linewidth=1.1)
    ax.set_yticks(yy, [x[0] for x in effects])
    ax.set_xlabel("Mean paired change in test AUROC")
    ax.set_title("Repeated-split primary effects")
    ax.set_xlim(-0.0042, 0.0090)
    ax.grid(axis="x", color="#e2e7ea", linewidth=0.8)
    ax.text(
        0,
        -0.24,
        "FS: 97.5% intervals for its 2-comparison family. FE: 98.75% intervals for its 4-comparison family. "
        "All intervals include zero.",
        transform=ax.transAxes,
        fontsize=8.4,
        color=COLORS["gray"],
    )
    figures["effects"] = savefig(fig, "04_repeated_split_primary_effects.png")

    # Figure 5: observed association between gate status and recorded salutation.
    eligible_p0 = 6788 / 20485
    below_p0 = 2874 / 5299
    fail_10 = 2874 / (6788 + 2874)
    fail_20 = 2425 / (13697 + 2425)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    vals = [eligible_p0, below_p0]
    bars = axes[0].bar(["Pass gate", "Below gate"], vals, color=[COLORS["blue"], COLORS["gold"]], width=0.62)
    axes[0].set_title("Share recorded as salutation 10")
    axes[0].set_ylabel("Proportion")
    axes[0].set_ylim(0, 0.65)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.1%}", ha="center", fontweight="bold")
    vals = [fail_10, fail_20]
    bars = axes[1].bar(["Salutation 10", "Salutation 20"], vals, color=[COLORS["gold"], COLORS["blue"]], width=0.62)
    axes[1].set_title("Share failing the coverage gate")
    axes[1].set_ylim(0, 0.38)
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.1%}", ha="center", fontweight="bold")
    for ax in axes:
        ax.grid(axis="y", color="#e2e7ea", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Coverage and recording availability are associated with the recorded label", y=1.04, fontsize=13)
    figures["missingness"] = savefig(fig, "05_missingness_signal.png")
    return figures


def build_html(figures: dict[str, Path]) -> str:
    v1 = read_json(ROOT / "experiments" / "artifacts" / "metrics.json")
    v2 = read_json(ROOT / "experiments" / "artifacts_v2" / "metrics.json")
    cohorts = pd.read_csv(ROOT / "experiments" / "artifacts_sq" / "source_cohorts.csv")
    fs = pd.read_csv(ROOT / "experiments" / "artifacts_sq" / "fsplit" / "fsplit_metrics.csv")
    fe = pd.read_csv(ROOT / "experiments" / "artifacts_sq" / "feng" / "feng_metrics.csv")

    # Side-quest result rows and best tuned point per source.
    source_rows = []
    side_rows = []
    for source in (3, 6, 7):
        cohort = cohorts[cohorts["source_id"] == source].iloc[0]
        d0 = read_json(ROOT / "experiments" / "artifacts_sq" / f"source_{source}" / "metrics_sq.json")
        d1 = read_json(ROOT / "experiments" / "artifacts_sq" / f"source_{source}" / "tuned_500" / "metrics_sq.json")
        default_vals = {v: float(d0["variants"][v]["test"]["auroc"]) for v in VARIANTS}
        tuned_vals = {v: float(d1["variants"][v]["test"]["auroc"]) for v in VARIANTS}
        best_variant = max(tuned_vals, key=tuned_vals.get)
        test_n = int(d0["variants"]["all"]["test"]["n"])
        source_rows.append(
            [
                SOURCE_NAMES[source],
                f"{int(cohort['model_eligible'] and cohort['any_core_pass']):,}",
                f"{int(cohort['sal10']):,}",
                f"{int(cohort['sal20']):,}",
                f"{test_n:,}",
                f"{cohort['any_core_pass']/cohort['base_single_source']:.1%}",
            ]
        )
        side_rows.append(
            [
                SOURCE_NAMES[source],
                fmt(default_vals["demo"]),
                fmt(default_vals["rec"]),
                fmt(default_vals["win"]),
                fmt(default_vals["all"]),
                f"{html.escape(best_variant)} ({tuned_vals[best_variant]:.3f})",
            ]
        )

    fs_effects = [
        (3, paired_effect(fs, 3, "all", "A2", "A0", 0.975)),
        (6, paired_effect(fs, 6, "all", "A2", "A0", 0.975)),
    ]
    fe_effects = []
    for source in (3, 6):
        for arm in ("F1", "F2"):
            fe_effects.append((source, arm, paired_effect(fe, source, "all", arm, "B0", 0.9875)))
    effect_rows = []
    for source, eff in fs_effects:
        effect_rows.append(
            [f"Feature selection, s{source} all, A2 - A0", signed(eff["mean"]), f"[{signed(eff['lo'])}, {signed(eff['hi'])}]", "97.5%"]
        )
    for source, arm, eff in fe_effects:
        effect_rows.append(
            [f"Feature engineering, s{source} all, {arm} - B0", signed(eff["mean"]), f"[{signed(eff['lo'])}, {signed(eff['hi'])}]", "98.75%"]
        )

    figure_rel = {k: f"figures/{v.name}" for k, v in figures.items()}

    pooled_table = table(
        ["Experiment", "N", "Main change", "Validation AUROC", "Test AUROC", "Test balanced accuracy", "Bootstrap SD"],
        [
            [
                "Pooled v1",
                "20,485",
                "89 epoch/recording predictors; sources 38/46/48 excluded",
                fmt(v1["validation"]["auroc"]),
                f"<strong>{fmt(v1['test']['auroc'])}</strong>",
                fmt(v1["test"]["balanced_accuracy"]),
                fmt(v1["bootstrap_test"]["auroc_sd"]),
            ],
            [
                "Pooled v2",
                "20,423",
                "Adds age/BMI and source indicators; also excludes sources 2/4/19",
                fmt(v2["validation"]["auroc"]),
                f"<strong>{fmt(v2['test']['auroc'])}</strong>",
                fmt(v2["test"]["balanced_accuracy"]),
                fmt(v2["bootstrap_test"]["auroc_sd"]),
            ],
        ],
    )

    experiment_table = table(
        ["Stage", "Population and split", "Question", "Status"],
        [
            ["Pooled v1", "20,485 gate-passing participants; 80/10/10", "Can epoch summaries predict recorded salutation?", '<span class="tag done">Complete</span>'],
            ["Pooled v2", "20,423 after extra source exclusions; 80/10/10", "Do demographics and source membership change the pooled result?", '<span class="tag done">Complete</span>'],
            ["Strict per-source", "Sources 3/6/7; sole-source 91-day gate; 80/10/10", "Which time representation works within each source?", '<span class="tag done">Complete</span>'],
            ["Tuning", "Same frozen per-source test sets; CV on train+validation", "Can regularization repair the default RF?", '<span class="tag done">Complete</span>'],
            ["Feature selection", "30 repeated 70/15/15 splits per source", "Do hygiene, deduplication, or top-k selection help?", '<span class="tag done">Complete</span>'],
            ["Feature engineering", "Same 30 splits and RF seeds", "Do simple interactions, ratios, or drift terms help?", '<span class="tag done">Complete</span>'],
            ["Missingness study", "Gated and ungated pooled cohorts", "How much signal is in the observation process itself?", '<span class="tag proposed">Proposed, not run</span>'],
        ],
    )

    representation_table = table(
        ["Representation", "Raw inputs", "Plain-language description"],
        [
            ["Demographics (`demo`)", "2", "Age group and BMI group. Reference model; included once in every side-quest design."],
            ["Full recording (`rec`)", "80", "Demographics plus 78 acquisition, coverage, HR level, variability, weekday/weekend, and time-of-day summaries over the available record."],
            ["Selected window (`win`)", "80", "The same summary families inside the selected 91-day sustained-recording window."],
            ["Rolling full (`roll_rec`)", "122", "Demographics plus 7- and 30-day rolling summaries across the recording span."],
            ["Rolling window (`roll_win`)", "122", "Rolling endpoints inside the selected window, with up to 29 days of pre-window lookback."],
            ["All blocks (`all`)", "398", "Demographics and all four epoch blocks combined."],
        ],
    )

    strict_table = table(
        ["Source cohort", "Strict-pass N", "Salutation 10", "Salutation 20", "Frozen test N", "Retention from sole-source base"],
        source_rows,
    )
    side_table = table(
        ["Source cohort", "Default demo", "Default recording", "Default 91-day", "Default all", "Largest tuned test point"],
        side_rows,
    )
    effects_table = table(["Primary comparison", "Mean delta", "Simultaneous interval", "Per-comparison level"], effect_rows)

    css = """
    @page { size: A4; margin: 16mm 15mm 17mm 15mm; }
    * { box-sizing: border-box; }
    body { margin: 0; color: #1d2b34; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; font-size: 10.1pt; line-height: 1.47; background: white; }
    h1, h2, h3 { color: #12324a; margin-top: 0; line-height: 1.16; break-after: avoid; }
    h1 { font-size: 29pt; letter-spacing: -0.5px; }
    h2 { font-size: 18pt; border-bottom: 2px solid #dce7ed; padding-bottom: 5px; margin-top: 22px; }
    h3 { font-size: 12.5pt; margin-top: 17px; margin-bottom: 6px; }
    p { margin: 6px 0 10px; }
    ul, ol { margin-top: 5px; padding-left: 21px; }
    li { margin: 3px 0; }
    strong { color: #102f45; }
    .cover { min-height: 255mm; padding: 23mm 10mm 8mm; position: relative; }
    .eyebrow { text-transform: uppercase; letter-spacing: 1.5px; color: #2676a8; font-size: 9pt; font-weight: 700; }
    .subtitle { font-size: 15pt; color: #4c616e; max-width: 145mm; margin-top: 12px; }
    .cover-rule { width: 42mm; border-top: 5px solid #2a9d8f; margin: 20px 0; }
    .cover-box { margin-top: 22mm; border-left: 5px solid #2676a8; background: #eef5f8; padding: 14px 17px; max-width: 165mm; }
    .meta { color: #63737d; margin-top: 13mm; }
    .page-break { break-after: page; }
    .avoid { break-inside: avoid; }
    .lead { font-size: 12pt; color: #314955; }
    .callout { border-left: 4px solid #2a9d8f; padding: 9px 13px; background: #f1f8f6; margin: 12px 0; break-inside: avoid; }
    .warning { border-left-color: #e9a23b; background: #fff8ea; }
    .important { border-left-color: #2676a8; background: #eef5f8; }
    .cards { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin: 14px 0; }
    .card { border: 1px solid #dbe4e9; border-radius: 6px; padding: 11px 13px; break-inside: avoid; }
    .card .number { font-size: 19pt; font-weight: 750; color: #2676a8; }
    .card .label { color: #5a6d77; font-size: 9pt; }
    table { width: 100%; border-collapse: collapse; margin: 10px 0 15px; font-size: 8.7pt; break-inside: avoid; }
    th { background: #12324a; color: white; text-align: left; padding: 6px 7px; font-weight: 650; }
    td { border-bottom: 1px solid #dce3e7; padding: 6px 7px; vertical-align: top; }
    tr:nth-child(even) td { background: #f7f9fa; }
    .tag { display: inline-block; border-radius: 10px; padding: 1px 7px; font-size: 8pt; font-weight: 650; white-space: nowrap; }
    .done { background: #dff2ed; color: #1f6d61; }
    .proposed { background: #fff0d8; color: #8b5b12; }
    figure { margin: 15px 0 18px; break-inside: avoid; }
    figure img { display: block; width: 100%; max-height: 154mm; object-fit: contain; }
    figcaption { color: #586b76; font-size: 8.5pt; margin-top: 5px; line-height: 1.35; }
    .small { font-size: 8.6pt; color: #60717a; }
    .two-col { columns: 2; column-gap: 22px; }
    .two-col > * { break-inside: avoid; }
    .flowline { display: grid; grid-template-columns: repeat(4, 1fr); gap: 7px; margin: 13px 0; break-inside: avoid; }
    .flowstep { position: relative; background: #edf4f7; border: 1px solid #cfdee5; border-radius: 5px; padding: 10px; min-height: 67px; font-size: 8.7pt; }
    .flowstep b { display: block; color: #12324a; margin-bottom: 3px; }
    .flowstep:not(:last-child)::after { content: "→"; position: absolute; right: -10px; top: 22px; color: #2676a8; font-size: 16pt; z-index: 2; }
    .footer-note { border-top: 1px solid #d9e1e5; margin-top: 18px; padding-top: 7px; color: #6a7981; font-size: 8pt; }
    code { background: #f1f4f5; padding: 1px 3px; border-radius: 3px; font-size: 0.94em; }
    """

    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Recorded salutation from wearable epoch data</title><style>{css}</style></head>
<body>
<section class="cover page-break">
  <div class="eyebrow">Final meta-report · 20 September 2026</div>
  <h1>Recorded salutation from wearable epoch data</h1>
  <div class="subtitle">Cohort design, feature representations, random-forest experiments, and a proposed analysis of informative missingness</div>
  <div class="cover-rule"></div>
  <div class="cover-box">
    <strong>Central finding.</strong> Wearable epoch data contain modest information about recorded salutation. Most usable signal is tied to recording pattern, device context, and broad heart-rate summaries. Regularization helps. Feature selection and the tested hand-built transformations do not provide a reliable general improvement.
  </div>
  <p class="meta"><strong>Evidence base:</strong> frozen September 2026 artifacts in <code>experiments/artifacts/</code>, <code>experiments/artifacts_v2/</code>, and <code>experiments/artifacts_sq/</code>.<br>
  <strong>Outcome:</strong> recorded salutation code 10 versus 20.<br>
  <strong>Status:</strong> all model results are completed; the missingness analysis is a proposal and has not been run.</p>
</section>

<section>
<h2>Executive summary</h2>
<p class="lead">The experiments answer a prediction question: how well can summaries of wearable recordings separate two recorded salutation codes among participants represented in this export? They do not identify biological sex, gender identity, or causal physiology.</p>
<div class="cards">
  <div class="card"><div class="number">0.713</div><div class="label">Pooled v1 frozen-test AUROC; balanced accuracy 0.590.</div></div>
  <div class="card"><div class="number">0.55–0.60</div><div class="label">Demographic-only frozen-test AUROC across the three strict source cohorts.</div></div>
  <div class="card"><div class="number">≤ 0.0034</div><div class="label">Largest mean primary feature-engineering gain; its simultaneous interval includes zero.</div></div>
  <div class="card"><div class="number">54.2% vs 33.1%</div><div class="label">Share recorded as salutation 10 below the coverage gate versus among gate passers.</div></div>
</div>
<div class="callout important"><strong>Best-supported reading:</strong> the models use a mixture of recording behavior, device or vendor context, availability of processed channels, and heart-rate summaries. The current experiments cannot isolate a physiological component.</div>
<ol>
  <li>The pooled baseline showed above-chance but modest discrimination and strongly asymmetric class recall.</li>
  <li>Within source-specific cohorts, recording and heart-rate summaries clearly outperformed age/BMI alone. Which time representation worked best differed by source.</li>
  <li>Cross-validated regularization—especially a larger minimum leaf size—improved the random forest. A single fixed regularized configuration was nearly as good as cell-specific tuning.</li>
  <li>Across repeated splits, feature hygiene greatly reduced width without a detected mean AUROC change. Correlation pruning, top-k selection, and the tested engineered terms were not consistently useful.</li>
  <li>The next useful diagnostic is to model the observation process directly, including participants below the coverage gate. This should be described as an informative-missingness study, not as proof that the data are MNAR.</li>
</ol>
</section>

<section>
<h2>1. Study question and data</h2>
<p>The outcome was the recorded salutation field: code 10 was mapped to class 0 and code 20 to class 1. Code 30 and missing labels were excluded from fitted models. Core epoch channels were 3000 (heart rate), 3001 (resting heart rate), and 3002 (hourly resting heart rate). Measurements outside 25–230 bpm were excluded. Participant identity, not rows or days, defined every train/validation/test split.</p>
<p>For the pooled cohort, an adequate channel-day required at least 8 observed clock hours and either 8 hours of interval coverage or 60 valid measurements. A participant passed when at least one channel had 14 adequate days.</p>
<div class="callout warning"><strong>Scope:</strong> results are conditional on participants with observed labels and on the export, source mix, coverage rules, and analysis choices. Vendor-derived channels may themselves use proprietary processing or user-entered information. This is a classification study, not a biological or causal study.</div>
<figure><img src="{figure_rel['flow']}"><figcaption><strong>Figure 1.</strong> Pooled v1 cohort construction. The coverage gate removed 5,299 otherwise labelled participants, making the fitted cohort a selected population.</figcaption></figure>

<h3>Experiment outline</h3>
{experiment_table}

<div class="flowline">
  <div class="flowstep"><b>1 · Define cohorts</b>Filter invalid records, apply participant-level coverage rules, and freeze splits.</div>
  <div class="flowstep"><b>2 · Build representations</b>Summarize acquisition, HR level, variability, calendar pattern, and local time of day.</div>
  <div class="flowstep"><b>3 · Fit and compare</b>Use training-fitted preprocessing and random forests; keep comparisons paired within a split.</div>
  <div class="flowstep"><b>4 · Stress-test</b>Repeat splits for feature selection and engineering; separate stable findings from single-split variation.</div>
</div>
</section>

<section>
<h2>2. Feature representations</h2>
<p>The pooled experiments used one broad participant-level table. The side quest separated the same general information into blocks so that time scope and representation could be compared within each source.</p>
{representation_table}
<p>Numeric preprocessing was learned on training participants only. The original and default side-quest models used median imputation and per-column missingness indicators, followed by zero-variance removal. Categorical variables used most-frequent imputation and one-hot encoding. No scaling was required for random forests.</p>
<div class="callout"><strong>Important distinction:</strong> acquisition variables such as observed days, coverage, event counts, span, and channel availability describe how data were recorded. HR means and variability describe recorded values. They are correlated but are not the same kind of information.</div>
</section>

<section>
<h2>3. Pooled baseline experiments</h2>
{pooled_table}
<p>Both models separated the classes better than chance, but default 0.5-threshold predictions mostly favored the more common class 1. In v1, class-0 recall was 0.255 and class-1 recall was 0.924. The corresponding v2 recalls were 0.246 and 0.915.</p>
<p>The v1 and v2 test AUROCs should not be subtracted as if they were a paired treatment comparison. V2 changed the cohort, split, source exclusions, and predictors simultaneously. Its lower test point does, however, show that simply adding age/BMI and source-membership indicators did not create an obvious improvement.</p>
</section>

<section>
<h2>4. Strict per-source side quest</h2>
<p>The side quest required a sustained 91-day window within a single source. Within one channel, a strict day required at least 12 hours of the frozen coverage quantity; a passing window required at least 65 strict days and at least 11 of 13 weeks with five strict days. Eligibility was assessed separately for each core channel; days were never pooled across channels. Sources 3, 6, and 7 had at least 200 participants in each label class and were modelled. Sources 9 and 13 remained audit-only.</p>
<figure><img src="{figure_rel['strict_cohort']}"><figcaption><strong>Figure 2.</strong> Strict source-specific cohorts. Gray bars show the v2 single-source base; colored bars show strict-pass participants by label. Source 6 retains a much smaller fraction; 4,367 of its 4,446 passers qualify through channel 3002.</figcaption></figure>
{strict_table}

<h3>Frozen-split results</h3>
<figure><img src="{figure_rel['auroc']}"><figcaption><strong>Figure 3.</strong> Default and tuned random forests on the same frozen source-specific test folds. These are descriptive point estimates. Source 7 has only 54 test participants, so its apparent ordering is especially unstable.</figcaption></figure>
{side_table}

<h3>What these comparisons show</h3>
<ul>
  <li><strong>Demographics alone were weak.</strong> Their frozen-test AUROC was about 0.56–0.60.</li>
  <li><strong>Recording and epoch summaries carried most of the prediction.</strong> The recording representation improved substantially over demographics in sources 3 and 6, and less precisely in source 7.</li>
  <li><strong>The useful time scope was source-specific.</strong> The 91-day block worked well for source 3; the full recording and combined blocks were stronger for source 6 after regularization.</li>
  <li><strong>More columns were not automatically better.</strong> Under default RF settings, the all-block design could be diluted by wide, redundant inputs.</li>
</ul>
</section>

<section>
<h2>5. Tuning: regularization mattered</h2>
<p>The tuning extension searched 48 random-forest configurations using five-fold cross-validation on each train-plus-validation pool. The frozen test fold was not used for selection. Final models used 500 trees.</p>
<div class="cards">
  <div class="card"><div class="number">10</div><div class="label">The larger minimum leaf size was the clearest and most consistent useful setting.</div></div>
  <div class="card"><div class="number">≤ 0.006</div><div class="label">CV AUROC gap between the fixed G1 configuration and each searched cell optimum.</div></div>
</div>
<p>A practical shared setting emerged: <code>max_features=0.4</code>, <code>min_samples_leaf=10</code>, unrestricted depth, and <code>class_weight=balanced_subsample</code>. The default sklearn configuration ranked near the bottom of the complete tuning surfaces. The main improvement therefore came from regularization, not merely from using 500 rather than 100 trees.</p>
<p>The most defensible performance gain is the approximately 0.015–0.020 increase in cross-validated AUROC over the default settings. Frozen-test gains are supportive but remain a second look at one split. The later repeated-split phases used the shared regularized configuration with 100 trees to control compute.</p>
</section>

<section>
<h2>6. Repeated-split feature selection and engineering</h2>
<p>Both phases used 30 stratified 70/15/15 participant splits per source and the same RF seeds for paired arms. Their estimand is the average effect over the prescribed split/model randomization, conditional on the observed strict-coverage cohort. Reusing these participants and partitions does not provide independent validation.</p>

<h3>Feature selection</h3>
<ul>
  <li><strong>A0:</strong> standard preprocessing.</li>
  <li><strong>A1:</strong> remove missingness-indicator columns and near-zero-variance terms.</li>
  <li><strong>A2:</strong> A1 plus greedy correlation pruning at |rho| &gt; 0.95.</li>
  <li><strong>A3:</strong> top 50 or 100 terms selected by validation-fold permutation importance.</li>
</ul>
<p>A1 produced no detected mean AUROC change in any source-by-representation cell while markedly reducing the widest matrices. This supports it as a computational hygiene step, but it is not a formal equivalence result and does not isolate indicator removal from near-zero-variance removal. A2 and A3 produced small, source-dependent effects; top-k sets were unstable across splits.</p>

<h3>Feature engineering</h3>
<p>The tested families were: products of weekday mean HR and time-of-day mean HR (E1, 24 terms), channel ratios (E2, 12 terms), and 30-day minus 7-day rolling changes (E3, 18 terms). These are deterministic re-expressions of existing measurements; they may make a tree split easier but do not add a new data source.</p>
<p>No primary comparison showed a detectable general AUROC improvement. Some secondary cells moved in either direction, but the effects did not transport consistently across sources or related representations.</p>

<figure><img src="{figure_rel['effects']}"><figcaption><strong>Figure 4.</strong> Corrected simultaneous intervals for the declared repeated-split primary families. The feature-selection family has two comparisons; the feature-engineering family has four. Every interval crosses zero.</figcaption></figure>
{effects_table}
<div class="callout important"><strong>Decision:</strong> keep the simpler hygiene preprocessing for efficiency. Do not adopt correlation pruning, top-k selection, or these engineered families as general defaults. This is a decision about the tested methods, not proof that every possible representation or model would fail.</div>
</section>

<section>
<h2>7. Proposed next study: recording missingness as signal</h2>
<p><strong>Status: not run.</strong> The existing coverage rule discards participants with sparse recordings, yet gate failure is strongly associated with the recorded outcome. The observation process may therefore contain useful predictive information. The right question is not “can MNAR be proved?” but “how much discrimination is carried by recording availability and pattern?”</p>
<figure><img src="{figure_rel['missingness']}"><figcaption><strong>Figure 5.</strong> Observed association motivating the missingness study. This is descriptive and does not identify why recording differs.</figcaption></figure>

<h3>Two target populations</h3>
{table(
    ["Code", "Participants", "Definition", "Interpretation"],
    [
        ["G: gated", "20,485", "Existing v1 gate passers; reuse the frozen split", "Performance among sustained recorders. The old test set has already informed later choices, so this is exploratory."],
        ["U: ungated", "25,784", "Gate passers plus 5,299 labelled participants below the gate; freeze a new split", "Performance in the broader measured population, with gate status treated as a predictor rather than an exclusion."],
    ],
)}

<h3>Simple recommended implementation</h3>
<ol>
  <li><strong>Build a recording-process block.</strong> Use observed and adequate days, event counts, span, duty cycle, longest gaps, hours covered, channel presence, time-of-day event fractions, and device/source fields.</li>
  <li><strong>Keep a value block.</strong> Use HR level and variability summaries. With current frozen artifacts, time-of-day means must use all valid events or be omitted; adequate-day-only diurnal means cannot be recovered because the hourly table has no date dimension.</li>
  <li><strong>Fit small, nested comparisons.</strong> Include source-only and gate-only anchors, recording-process only, HR values only, and their combination. This separates “mostly device/gate” from richer recording-pattern signal.</li>
  <li><strong>Use two value encodings.</strong> A train-median, no-indicator version largely suppresses missingness information; a native-NaN version allows the RF to use missingness implicitly. Label the latter honestly as “values plus implicit missingness.”</li>
  <li><strong>Keep evaluation participant-level and paired.</strong> Fit preprocessing on training participants only. Reuse common resamples across variants. If only ten bootstrap draws are retained for direct v1 compatibility, treat them as a rough spread, not an interval.</li>
</ol>

{table(
    ["Candidate model", "Inputs", "What it answers"],
    [
        ["Source/gate anchors", "Source and multisource; gate status in U", "How much prediction comes from coarse device or selection information alone?"],
        ["R", "Detailed recording-process features, reported with and without source", "Do coverage, gaps, timing, and channel presence add information beyond source?"],
        ["V-imputed", "HR values; training-median fill; no missing bits", "A practical value-focused comparator with missingness mostly suppressed."],
        ["V-native", "HR values with NaNs passed to the RF", "Values plus implicit per-feature missingness."],
        ["R + V-native", "Both blocks", "Best missingness-aware prediction using the existing data."],
    ],
)}

<div class="callout warning"><strong>Interpretation limit:</strong> this design can show that missingness or recording patterns are predictive. It cannot establish an MNAR mechanism, recover the 11 missing outcomes, or show that the signal is physiological. A value can be missing because of wear, device capability, software, study engagement, or data transfer.</div>

<h3>Other possible implementations</h3>
<p>The simplest alternative is the existing median-imputation pipeline with explicit missingness indicators. It is easy to audit but can become wide and the prior strict-cohort indicators were mostly unused. A more complex two-model approach—one model for recording pattern and one for measured values, with their scores combined on validation data—could make block contributions clearer, but it is not necessary for the first pass.</p>
</section>

<section>
<h2>8. What can and cannot be concluded</h2>
<h3>Supported conclusions</h3>
<ul>
  <li>Epoch-derived participant summaries predict recorded salutation above chance in these selected cohorts.</li>
  <li>Age and BMI groups alone explain little of the source-specific discrimination.</li>
  <li>Recording context and broad HR summaries are the strongest available feature families.</li>
  <li>RF regularization is more useful than cell-specific feature selection or the tested hand-built terms.</li>
  <li>The coverage gate changes the target population and removes an outcome-associated part of the observation process.</li>
</ul>
<h3>Not supported</h3>
<ul>
  <li>A biological-sex or gender interpretation of the classifier.</li>
  <li>A causal physiological effect or a vendor ranking.</li>
  <li>Transport to new participants, devices, software versions, or recording policies without external evaluation.</li>
  <li>Claims that feature selection or engineering is exactly equivalent to baseline; the studies establish a resolution limit, not equality.</li>
  <li>Claims that all model-side improvements are exhausted.</li>
</ul>
<div class="callout"><strong>Most valuable next evidence:</strong> a pre-frozen ungated missingness analysis followed by evaluation on genuinely new participants or an external export.</div>
</section>

<section>
<h2>9. Verification and provenance</h2>
<ul>
  <li>The pooled v1 and v2 verification artifacts report that all declared checks passed.</li>
  <li>The strict per-source phase reports 135/135 checks passed, covering eligibility, windows, splits, features, predictions, and bootstrap reconstruction.</li>
  <li>The feature-selection and feature-engineering result tables each contain 2,700 complete rows with 30 repeats per recorded group and no duplicate keys or non-finite metrics.</li>
  <li>Feature-engineering arm B0 reproduces feature-selection arm A1 to a maximum AUROC difference of 5.6e-17.</li>
  <li>This meta-report recomputed its tables and figures from saved JSON/CSV/Parquet artifacts. It fitted no models.</li>
</ul>

<h3>Primary artifact index</h3>
{table(
    ["Topic", "Artifact"],
    [
        ["Pooled experiment", "reports/REPORT.md; experiments/artifacts/metrics.json; experiments/artifacts_v2/metrics.json"],
        ["Strict source cohorts", "experiments/artifacts_sq/source_cohorts.csv; experiments/artifacts_sq/sidequest_report.md"],
        ["Tuning", "experiments/artifacts_sq/tuned_500_comparison.md; source_*/tuned_500/"],
        ["Feature selection", "experiments/artifacts_sq/fsplit/fsplit_metrics.csv; feat_sel_report.md"],
        ["Feature engineering", "experiments/artifacts_sq/feng/feng_metrics.csv; feat_eng_report.md"],
        ["Missingness proposal", "plans/MNAR_PLAN.md"],
    ],
)}

<p class="small">Reporting note: this synthesis uses corrected simultaneous confidence levels for the repeated-split primary families and describes E1 using the actual source feature semantics (time-of-day mean HR, not event share). Secondary repeated-split results remain exploratory and are not used to declare winners.</p>
<div class="footer-note">Generated from local frozen artifacts on 20 September 2026. PDF output: <code>{html.escape(str(OUT_PDF.relative_to(ROOT)))}</code>.</div>
</section>
</body></html>"""


def main() -> None:
    figures = build_figures()
    document = build_html(figures)
    OUT_HTML.write_text(document, encoding="utf-8")
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in figures.values()}
    manifest = {
        "html": str(OUT_HTML.relative_to(ROOT)),
        "pdf": str(OUT_PDF.relative_to(ROOT)),
        "figures": hashes,
        "note": "Reporting only; no model fitting.",
    }
    (REPORT_DIR / "recorded_salutation_meta_report_2026-09-20_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(OUT_HTML)


if __name__ == "__main__":
    main()
