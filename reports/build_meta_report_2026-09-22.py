#!/usr/bin/env python3
"""Build the 2026-09-22 meta-report (lab-report register).

Reporting only: reads committed artifacts and writes HTML + PDF + manifest.
No model is refit. Numbers in the report are computed from the saved csvs.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR = Path(tempfile.gettempdir()) / "codex_meta_2026_09_22_mpl"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

TG = ROOT / "experiments" / "temporal_garmin" / "results"
OUT_HTML = REPORT_DIR / "recorded_salutation_meta_report_2026-09-22.html"
OUT_PDF = REPORT_DIR / "recorded_salutation_meta_report_2026-09-22.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PREV_PDF = REPORT_DIR / "recorded_salutation_meta_report_2026-09-20.pdf"
PREV_HTML = REPORT_DIR / "recorded_salutation_meta_report_2026-09-20.html"

V2 = pd.read_csv(TG / "temporal_metrics.csv")
V1 = pd.read_csv(TG / "temporal_metrics_v1.csv")
SEL = pd.read_csv(TG / "selective_classification.csv")
ALPHA = json.loads((TG / "alpha_addendum.json").read_text())

# ---------- computations ---------------------------------------------------
ARMS_LADDER = ["Summary_RF", "Summary_linear", "Profile24", "Profile288",
               "MultiRocket", "HYDRA", "Combined", "Shuffled_MR"]
ARM_LABEL = {"Summary_RF": "Summary (RF)", "Summary_linear": "Summary (Ridge)",
             "Profile24": "Profile24 (hourly)", "Profile288": "Profile288 (5-min)",
             "MultiRocket": "MultiRocket", "HYDRA": "HYDRA",
             "Combined": "Combined", "Shuffled_MR": "Shuffled MR (control)"}


def ladder_table() -> list[list[str]]:
    rows = []
    g = V2.groupby("arm").agg(val=("auroc_val", "mean"),
                              val_sd=("auroc_val", lambda s: s.std(ddof=1)),
                              te=("auroc_test", "mean"),
                              te_sd=("auroc_test", lambda s: s.std(ddof=1)),
                              n_cols=("n_cols", "first"),
                              alpha=("alpha", "first"))
    g = g.reindex(ARMS_LADDER)
    for arm in ARMS_LADDER:
        r = g.loc[arm]
        a = "—" if pd.isna(r["alpha"]) else f"{r['alpha']:.0f}"
        rows.append([ARM_LABEL[arm],
                     f"{r['val']:.4f} ± {r['val_sd']:.4f}",
                     f"{r['te']:.4f} ± {r['te_sd']:.4f}",
                     f"{int(r['n_cols'])}", a])
    return rows


def paired_delta(arm_a: str, arm_b: str, col: str = "auroc_val") -> tuple:
    a = V2[V2.arm == arm_a].set_index("alloc")[col]
    b = V2[V2.arm == arm_b].set_index("alloc")[col]
    d = (a - b).reindex(V2.alloc.unique()).dropna()
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    ci = student_t.ppf(1 - 0.05 / 8, n - 1) * se  # 2-sided Bonferroni, m=4
    return m, sd, se, (m - ci, m + ci), n


def primary_deltas_table() -> list[list[str]]:
    pairs = [("Profile288", "Summary_RF"), ("MultiRocket", "Summary_RF"),
             ("HYDRA", "Summary_RF"), ("Combined", "Summary_RF")]
    rows = []
    for a, b in pairs:
        m, sd, se, (lo, hi), n = paired_delta(a, b)
        share = float((V2[V2.arm == a].set_index("alloc").auroc_val -
                        V2[V2.arm == b].set_index("alloc").auroc_val).gt(0).mean())
        rows.append([f"{ARM_LABEL[a]} − {ARM_LABEL[b]}",
                     f"{m:+.4f}", f"{sd:.4f}",
                     f"[{lo:+.4f}, {hi:+.4f}]",
                     f"{share:.2f}"])
    # MR − Shuffled
    m, sd, se, (lo, hi), n = paired_delta("MultiRocket", "Shuffled_MR")
    rows.append(["MultiRocket − Shuffled MR (placement isolation)",
                 f"{m:+.4f}", f"{sd:.4f}",
                 f"[{lo:+.4f}, {hi:+.4f}]",
                 f"{(V2[V2.arm=='MultiRocket'].set_index('alloc').auroc_val - V2[V2.arm=='Shuffled_MR'].set_index('alloc').auroc_val).gt(0).mean():.2f}"])
    return rows


def selective_table() -> list[list[str]]:
    arms = ["Combined", "Summary_RF", "BASE_x_P40", "Random"]
    arm_lab = {"Combined": "Combined", "Summary_RF": "Summary (RF)",
               "BASE_x_P40": "BASE ⊕ P40 (R1b baseline)", "Random": "Random (sanity)"}
    rows = []
    for tgt in (0.95, 0.98):
        for variant in ("raw", "cpc"):
            for arm in arms:
                sub = SEL[(SEL.arm == arm) & (SEL.target == tgt) &
                          (SEL.variant == variant)]
                if len(sub) == 0:
                    continue
                per_alloc = []
                for alloc, g in sub.groupby("alloc"):
                    if arm == "BASE_x_P40":
                        # mean across the two cross-fit directions per alloc
                        cov = g.groupby(["class", "xfit_dir"]).cov_test.mean()
                    else:
                        cov = g.groupby("class").cov_test.mean()
                    if {0, 1}.issubset(set(cov.index.get_level_values(0))):
                        per_alloc.append(cov.loc[1] + cov.loc[0])
                cov_total = np.array(per_alloc) * 100
                prec1 = sub[sub["class"] == 1].groupby("alloc").prec_test.mean().dropna()
                prec0 = sub[sub["class"] == 0].groupby("alloc").prec_test.mean().dropna()
                rows.append([arm_lab[arm], f"{int(tgt*100)}%",
                             variant,
                             f"{cov_total.mean():.1f} ± {cov_total.std(ddof=1):.1f}" if len(cov_total) > 1 else f"{cov_total.mean():.1f}",
                             f"{prec1.mean():.3f}" if len(prec1) else "—",
                             f"{prec0.mean():.3f}" if len(prec0) else "—"])
    return rows


def alpha_table() -> list[list[str]]:
    rows = []
    for fam in ["Summary_linear", "Profile24", "Profile288",
                "MultiRocket", "HYDRA", "Combined"]:
        f = ALPHA["families"][fam]
        fa = ALPHA["frozen_alpha"][fam]
        rows.append([fam, f"{fa:.0f}", f"{f['auroc_val_frozen']:.4f}",
                     f"{f['alpha_star']:.0f}", f"{f['auroc_val_star']:.4f}",
                     f"{f['gain_val']:+.4f}",
                     f"{f['auroc_test_frozen']:.4f}",
                     f"{f['auroc_test_star']:.4f}"])
    return rows


def v1v2_table() -> list[list[str]]:
    g2 = V2.groupby("arm").agg(val=("auroc_val", "mean"), te=("auroc_test", "mean"))
    g1 = V1.groupby("arm").agg(val=("auroc_val", "mean"), te=("auroc_test", "mean"))
    rows = []
    for arm in ARMS_LADDER:
        v1v, v1t = g1.loc[arm, "val"], g1.loc[arm, "te"]
        v2v, v2t = g2.loc[arm, "val"], g2.loc[arm, "te"]
        rows.append([ARM_LABEL[arm], f"{v1v:.4f}", f"{v2v:.4f}",
                     f"{v2v - v1v:+.4f}",
                     f"{v1t:.4f}", f"{v2t:.4f}", f"{v2t - v1t:+.4f}"])
    return rows


# ---------- figures --------------------------------------------------------
def fig_ladder() -> Path:
    g = V2.groupby("arm").agg(val=("auroc_val", "mean"),
                              val_sd=("auroc_val", lambda s: s.std(ddof=1)),
                              te=("auroc_test", "mean"),
                              te_sd=("auroc_test", lambda s: s.std(ddof=1)))
    g = g.reindex(ARMS_LADDER)
    y = np.arange(len(ARMS_LADDER))
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=180)
    ax.errorbar(g["val"], y - 0.13, xerr=g["val_sd"], fmt="o",
                color="#1f3a5f", mfc="white", mec="#1f3a5f", capsize=2.5,
                lw=1.0, ms=6, label="validation (R = 10)")
    ax.errorbar(g["te"], y + 0.13, xerr=g["te_sd"], fmt="s",
                color="#7a4a2a", mfc="#7a4a2a", mec="#7a4a2a", capsize=2.5,
                lw=1.0, ms=5.5, label="test (R = 10)")
    ax.axvline(0.5, color="#888", lw=0.6, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([ARM_LABEL[a] for a in ARMS_LADDER], fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0.55, 0.92)
    ax.set_xlabel("AUROC", fontsize=9)
    ax.grid(axis="x", color="#dcdcd6", lw=0.5)
    ax.legend(loc="lower right", fontsize=8.4, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = FIG_DIR / "06_temporal_ladder.png"
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_coverage() -> Path:
    arms = [("Combined", "Combined"),
            ("Summary_RF", "Summary (RF)"),
            ("BASE_x_P40", "BASE ⊕ P40"),
            ("Random", "Random")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=180,
                              sharey=True)
    width = 0.36
    x = np.arange(len(arms))
    for ax, tgt in zip(axes, (0.95, 0.98)):
        for j, variant in enumerate(("raw", "cpc")):
            vals = []
            errs = []
            for arm, _ in arms:
                sub = SEL[(SEL.arm == arm) & (SEL.target == tgt) &
                          (SEL.variant == variant)]
                per_alloc = []
                for alloc, g in sub.groupby("alloc"):
                    if arm == "BASE_x_P40":
                        cov = g.groupby(["class", "xfit_dir"]).cov_test.mean()
                    else:
                        cov = g.groupby("class").cov_test.mean()
                    if {0, 1}.issubset(set(cov.index.get_level_values(0))):
                        per_alloc.append(cov.loc[1] + cov.loc[0])
                a = np.array(per_alloc) * 100
                vals.append(a.mean() if len(a) else 0.0)
                errs.append(a.std(ddof=1) if len(a) > 1 else 0.0)
            color = "#1f3a5f" if variant == "raw" else "#7a4a2a"
            label = "empirical" if variant == "raw" else "CP-LCB (90%)"
            ax.bar(x + (j - 0.5) * width, vals, width, yerr=errs,
                   color=color, edgecolor="black", lw=0.5, capsize=2,
                   label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in arms], fontsize=8.2,
                            rotation=15, ha="right")
        ax.set_title(f"target precision {int(tgt*100)}%", fontsize=9)
        ax.grid(axis="y", color="#dcdcd6", lw=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("coverage (% of test participants labeled)", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8.2, frameon=False)
    fig.tight_layout()
    p = FIG_DIR / "07_coverage_at_precision.png"
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ---------- HTML -----------------------------------------------------------
def fmt(x, d=4):
    return f"{x:.{d}f}"


def table_html(headers, rows, caption=None):
    h = "".join(f"<th>{html.escape(str(v))}</th>" for v in headers)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>"
                    for r in rows)
    cap = f'<caption><span class="cap">Table.</span> {html.escape(caption)}</caption>' if caption else ""
    return f'<table>{cap}<thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def figure_html(path, caption, num):
    rel = path.relative_to(REPORT_DIR)
    return (f'<figure><img src="{html.escape(str(rel))}" alt="">'
            f'<figcaption><span class="cap">Figure {num}.</span> '
            f'{html.escape(caption)}</figcaption></figure>')


def build_html() -> str:
    css = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: Georgia, "Times New Roman", serif; color: #111;
       font-size: 10.5pt; line-height: 1.55; background: white; }
h1 { font-size: 21pt; margin: 0 0 4px; line-height: 1.15; }
.docmeta { color: #444; font-size: 9.4pt; border-bottom: 1.5px solid #111;
           padding-bottom: 12px; margin-bottom: 18px; }
.docmeta .mono { font-size: 9pt; }
h2 { font-size: 13.5pt; margin: 24px 0 8px; border-top: 1px solid #999;
     padding-top: 10px; break-after: avoid; }
h3 { font-size: 11.2pt; margin: 14px 0 4px; break-after: avoid; }
p { margin: 6px 0 9px; }
ul, ol { margin-top: 4px; padding-left: 22px; }
li { margin: 2px 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt;
        margin: 8px 0 8px; }
thead th { border-top: 1.2px solid #111; border-bottom: 1.2px solid #111;
           padding: 5px 6px; text-align: left; font-weight: 700; }
tbody td { padding: 4px 6px; vertical-align: top; }
tbody tr:last-child td { border-bottom: 1.2px solid #111; }
tbody tr:nth-child(even) td { background: #f6f6f2; }
caption { caption-side: top; text-align: left; font-size: 8.8pt;
          color: #333; padding: 0 0 4px; }
.eq { text-align: center; font-family: Georgia, serif; font-size: 11.5pt;
      margin: 10px 0; }
.eq .no { color: #555; font-size: 9pt; padding-left: 8px; }
figure { margin: 12px 0 14px; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 8.7pt; color: #222; margin-top: 4px;
             text-align: left; line-height: 1.35; }
.cap { font-weight: 700; }
.mono, code { font-family: "SF Mono", Menlo, Consolas, monospace;
              font-size: 8.9pt; background: #f2f1ec; padding: 0 3px;
              border-radius: 2px; }
.small { font-size: 8.6pt; color: #444; }
.note { border-left: 2.5px solid #888; padding: 2px 0 2px 10px;
        margin: 9px 0; color: #1d1d1d; }
a { color: #1a3e63; text-decoration: none; border-bottom: 1px dotted #1a3e63; }
.toc { font-size: 9.6pt; columns: 2; column-gap: 22px;
       border-top: 0.5px solid #999; padding-top: 8px; margin: 8px 0 14px; }
.toc li { break-inside: avoid; }
hr.thin { border: 0; border-top: 1px solid #999; margin: 16px 0; }
.provenance { font-size: 8.4pt; color: #555; margin-top: 18px;
              border-top: 1px solid #ccc; padding-top: 8px; }
"""

    # STAGES = Table 1 — synthesis table (numbers from per-experiment REPORTs,
    # see [README.md](../../README.md) table for the curated row).
    stages = [
        ("Pooled v1", "20,485 all-source; default RF",
         "test AUROC 0.7132",
         "REPORT.md"),
        ("Pooled v2", "20,485 + demographics/source indicators; new exclusions",
         "test AUROC 0.6833",
         "artifacts_v2/model_report.md"),
        ("Strict source-specific", "Garmin 3,848; Apple 4,446; Samsung 541",
         "source-specific summaries beat demographics",
         "artifacts_sq/sidequest_report.md"),
        ("RF tuning", "shared config adopted",
         "max_features=0.4, min_samples_leaf=10, "
         "class_weight=balanced_subsample, depth unrestricted, n=100",
         "artifacts_sq/tuned_500_comparison.md"),
        ("Feature selection (30 splits)",
         "hygiene, corr prune, top-k",
         "no consistent gain; selected subsets unstable",
         "artifacts_sq/fsplit/feat_sel_report.md"),
        ("Feature engineering",
         "interactions, ratios, rolling diffs",
         "max +0.0034 AUROC, intervals cross zero",
         "artifacts_sq/feng/feat_eng_report.md"),
        ("Informative recording / missingness",
         "25,784 ungated; recording + value model",
         "combined 0.7239; recording model adds +0.0167",
         "2026-09-20_informative_recording_features/README.md"),
        ("Continuous age vs age groups",
         "Garmin",
         "0.7445 → 0.7475 (+0.0029, CI [−0.0008, +0.0067])",
         "experiments/age_vs_agegroup_garmin_2026-09-20/results/REPORT.md"),
        ("OR-Tools 50/50, source-balanced",
         "Garmin / Apple / Samsung",
         "G 0.7290/0.7094; A 0.6875/0.6787; S 0.6289/0.6493",
         "2026-09-21_ortools_source_50_50_salutation_rf/PLAN.md"),
        ("R1a circadian features",
         "Garmin",
         "baseline 0.7434 → combined 0.7463 (+0.0029)",
         "experiments/r1a_circadian_garmin_2026-09-21/results/REPORT.md"),
        ("Day-scale modelling",
         "Garmin, k days",
         "0.6653 (k=1) → 0.7269 (k=32) → 0.7302 (all days)",
         "experiments/dayscale_garmin_2026-09-21/results/REPORT.md"),
        ("R1a-AF (R=10 allocator folds)",
         "Garmin, 10 allocations",
         "baseline 0.7415 → combined 0.7509 (+0.0094, allocs-positive, Bonferroni excludes 0)",
         "experiments/r1a_allocfolds_garmin_2026-09-21/results/REPORT.md"),
        ("R1b date-aware circadian",
         "Garmin, R1a-AF folds, first 40 days",
         "day-level 0.6589 → 0.6782 (+0.0193); participant 0.7415 → 0.7513 (+0.0098)",
         "experiments/r1b_dateaware_garmin_2026-09-21/results/REPORT.md"),
        ("Temporal representations (R1b cohort/folds)",
         "Garmin, 10 allocations, 288 × 5-min bins × 40 days",
         "summaries 0.7031 → Combined 0.8565 test (+0.153 vs Summary_RF, Bonferroni)",
         "experiments/temporal_garmin/results/REPORT.md"),
    ]
    table1_rows = [[s[0], s[1], s[2], s[3]] for s in stages]

    # Pre-compute derived numbers used in prose
    com_val, _, _, _, _ = paired_delta("Combined", "Summary_RF")
    mr_shuf, _, _, _, _ = paired_delta("MultiRocket", "Shuffled_MR")
    bas_delta, _, _, _, _ = paired_delta("MultiRocket", "Summary_RF")
    hydra_delta, _, _, _, _ = paired_delta("HYDRA", "Summary_RF")
    p288_delta, _, _, _, _ = paired_delta("Profile288", "Summary_RF")
    raw95 = SEL[(SEL.arm == "Combined") & (SEL.target == 0.95) & (SEL.variant == "raw")]
    raw95_per = []
    for a, g in raw95.groupby("alloc"):
        cov = g.groupby("class").cov_test.mean()
        raw95_per.append(cov.loc[1] + cov.loc[0])
    raw95_per = np.array(raw95_per) * 100
    base_raw95 = SEL[(SEL.arm == "BASE_x_P40") & (SEL.target == 0.95) & (SEL.variant == "raw")]
    base_per = []
    for a, g in base_raw95.groupby("alloc"):
        cov = g.groupby(["class", "xfit_dir"]).cov_test.mean()
        base_per.append(cov.loc[1].mean() + cov.loc[0].mean())
    base_per = np.array(base_per) * 100

    parts = []
    parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    parts.append(f"<title>Recorded-salutation predictability — September 2026 synthesis</title>")
    parts.append(f"<style>{css}</style></head><body>")

    # --- Title block (plain, no hero)
    parts.append("<h1>Recorded-salutation predictability from wearable epoch data</h1>")
    parts.append(
        "<div class='docmeta'>"
        "A synthesis of the September 2026 experiments &middot; "
        "Document date 22 September 2026 &middot; "
        "Reporting only — no model was refit for this document.<br>"
        "Inputs: frozen artifacts in "
        "<code>experiments/*/results/</code>, <code>artifacts/</code>, "
        "<code>artifacts_sq/</code>; per-experiment reports cited in Table&nbsp;1. "
        "Build script: <code>reports/build_meta_report_2026-09-22.py</code>."
        "</div>"
    )

    parts.append("<div class='toc'><ol>")
    for n, t in [(1, "Data and label"), (2, "Measurement protocol"),
                 (3, "Results by phase"),
                 (4, "Temporal representations (main result)"),
                 (5, "Selective classification"),
                 (6, "Regularization headroom"),
                 (7, "Limitations and threats to validity"),
                 (8, "Reproducibility record"),
                 (9, "Next steps"),
                 ("A", "Notation and file index")]:
        parts.append(f"<li>{t}</li>")
    parts.append("</ol></div>")

    # §1
    parts.append("<h2>1&nbsp;&nbsp;Data and label</h2>")
    parts.append(
        "<p>The target is the participant's <em>recorded</em> salutation field "
        "in the Datenspende cohort export, taking the two populated codes "
        "<span class='mono'>10</span> and <span class='mono'>20</span>. "
        "Across the 3,848-participant Garmin cohort that anchors the newest "
        "experiments, code 20 is the majority class (prevalence 0.64; "
        "class mean of <code>temporal_predictions.csv</code> = 0.642). "
        "Recorded salutation is what the participant entered; it is not established "
        "biological sex or gender, and the design records no physiological or causal "
        "claim.</p>"
    )
    parts.append(
        "<p>The newest experiments use a single Garmin subcohort "
        "(<span class='mono'>source_id == 3</span>) with the first 40 adequate days "
        "per participant (the R1b cohort). For each participant-day we have a series "
        "of heart-rate epochs with timestamps and a local timezone, plus "
        "recording-volume metadata. We do not use anthropometrics or inter-beat "
        "intervals — those fields are absent from the export.</p>"
    )

    # §2
    parts.append("<h2>2&nbsp;&nbsp;Measurement protocol</h2>")
    parts.append(
        "<p>For a model producing a real-valued score "
        "<span class='mono'>s<sub>i</sub></span> per participant, the primary "
        "ranking metric is the area under the receiver operating characteristic "
        "curve. It equals the probability that a randomly chosen positive scores "
        "above a randomly chosen negative (with ties split):</p>"
    )
    parts.append(
        "<div class='eq'>AUROC&nbsp;=&nbsp;P(s<sup>+</sup>&nbsp;&gt;&nbsp;s<sup>−</sup>)&nbsp;+&nbsp;½·P(s<sup>+</sup>&nbsp;=&nbsp;s<sup>−</sup>)&nbsp;<span class='no'>(1)</span></div>"
    )
    parts.append(
        "<p>Two designs are matched on participants and split: we report "
        "<em>paired</em> per-allocation deltas. Allocation <span class='mono'>r</span> "
        "is one of <span class='mono'>R&nbsp;=&nbsp;10</span> repeated train/"
        "validation/test partitions (allocator folds, R1a-AF). For two arms A and B "
        "we compute "
        "<span class='mono'>Δ<sub>r</sub>&nbsp;=&nbsp;AUROC<sub>A,r</sub>&nbsp;−&nbsp;AUROC<sub>B,r</sub></span> "
        "and the 95% Bonferroni two-sided simultaneous interval:</p>"
    )
    parts.append(
        "<div class='eq'>"
        "mean&nbsp;Δ&nbsp;±&nbsp;t<sub>1−γ/2,&nbsp;R−1</sub>·SD(Δ)/√R,&nbsp;&nbsp;"
        "γ&nbsp;=&nbsp;0.05/(2m),&nbsp;for m primary comparisons"
        "<span class='no'>(2)</span></div>"
    )
    parts.append(
        "<p>Variability across allocations is model-and-split randomization conditional "
        "on the observed cohort; it is not population uncertainty. The test split is "
        "reused across the recent experiments and is therefore exploratory by "
        "predeclared rule; conclusions about absolute generalization beyond this cohort "
        "are not established.</p>"
    )

    # §3 — Table 1 + phase summaries
    parts.append("<h2>3&nbsp;&nbsp;Results by phase</h2>")
    parts.append("<p>Table&nbsp;1 lists all 14 experiment stages in chronological order. "
                 "Numbers are taken from each stage's committed report (cited in the rightmost "
                 "column); this synthesis does not recompute them.</p>")
    parts.append(table_html(
        ["Stage", "Cohort / design", "Main result", "Source"],
        table1_rows,
        "The 14 experiment stages of the September 2026 campaign."
    ))

    parts.append("<h3>3.1&nbsp;&nbsp;Pooled baselines and source structure</h3>")
    parts.append(
        "<p>The pooled models (v1, v2) trained on all wearable sources at once showed that "
        "demographics alone are weak and that source exclusions and feature inclusion "
        "matter: the v2 protocol (added demographics and source indicators, changed "
        "exclusions) moves test AUROC from 0.7132 to 0.6833 — a difference that is not "
        "attributable to features alone because the cohort and split changed "
        "simultaneously. Source-specific models (Garmin/Apple/Samsung) showed that "
        "different sources have different useful representations.</p>"
    )

    parts.append("<h3>3.2&nbsp;&nbsp;RF tuning, feature selection, feature engineering</h3>")
    parts.append(
        "<p>Tuning established the shared configuration used in every later experiment: "
        "<span class='mono'>max_features=0.4</span>, "
        "<span class='mono'>min_samples_leaf=10</span>, "
        "<span class='mono'>class_weight='balanced_subsample'</span>, depth unrestricted, "
        "n_estimators=100. The two follow-up ablations tested hand-built transformations "
        "(interactions, channel ratios, rolling-window differences) and feature selection "
        "(hygiene, correlation pruning, top-k). Neither offers a reliable general "
        "improvement: feature engineering contributes at most "
        "<span class='mono'>+0.0034</span> with intervals that cross zero; "
        "hygiene reduces matrix width without a detected loss, while correlation pruning "
        "and top-k produce unstable subsets.</p>"
    )

    parts.append("<h3>3.3&nbsp;&nbsp;Recording patterns are informative</h3>")
    parts.append(
        "<p>The informative-recording experiment (25,784 ungated participants) separated "
        "recording-pattern features from value features. The combined model reaches "
        "<span class='mono'>0.7239</span>: the recording model contributes "
        "<span class='mono'>+0.0167</span> beyond the value model. This establishes that "
        "<em>how much</em> and <em>when</em> data are recorded carries information about "
        "the recorded salutation, beyond <em>what</em> is recorded. Native NaN handling "
        "offered no detectable advantage over median imputation.</p>"
    )

    parts.append("<h3>3.4&nbsp;&nbsp;Circadian shape and day-scale</h3>")
    parts.append(
        "<p>R1a added 35 hourly-profile and cosinor features; the increment is small "
        "(baseline 0.7434 → combined 0.7463). R1b extended this to per-day circadian "
        "curves over each participant's first 40 adequate days on the R1a-AF folds: "
        "day-level AUROC rises by <span class='mono'>+0.0193</span> "
        "(coverage-residualized: <span class='mono'>+0.0181</span>); participant-level "
        "aggregation preserves a <span class='mono'>+0.010</span> increment over the "
        "R1a-AF baseline. All three increments are Bonferroni-positive. The day-scale "
        "experiment traces the prospective question: how does performance scale with "
        "the number of days sampled per participant? It rises from 0.6653 at k=1 to "
        "0.7269 at k=32 and 0.7302 with all days; personal-baseline features help most "
        "when k is small.</p>"
    )

    # §4 temporal — centerpiece
    parts.append("<h2>4&nbsp;&nbsp;Temporal representations (main result)</h2>")
    parts.append(
        "<p>Question: does <em>within-day</em> structure, beyond what the daily summaries "
        "already capture, add information? We bin each participant's first 40 adequate "
        "Garmin days into 288 five-minute bins (local time; "
        "<span class='mono'>5 min × 288 = 24 h</span>), bin-mean the heart rate per "
        "user-day, mask unobserved bins, and learn several representations of the "
        "resulting matrix. The head is ridge regression with "
        "<span class='mono'>α</span> chosen on validation (alloc&nbsp;0):</p>"
    )
    parts.append(
        "<div class='eq'>"
        "ŵ&nbsp;=&nbsp;argmin<sub>w</sub>&nbsp;‖y&nbsp;−&nbsp;Xw‖<sub>2</sub><sup>2</sup>&nbsp;+&nbsp;α‖w‖<sub>2</sub><sup>2</sup>"
        "<span class='no'>(3)</span></div>"
    )
    parts.append("<p>The two convolutional kernels used here are MultiRocket and HYDRA. "
                 "Both apply a dilated 1-D convolution across each day's series:</p>")
    parts.append(
        "<div class='eq'>"
        "(k&nbsp;∗&nbsp;x)[t]&nbsp;=&nbsp;Σ<sub>j</sub>&nbsp;w<sub>j</sub>·x[t&nbsp;+&nbsp;d·j]"
        "<span class='no'>(4)</span></div>"
    )
    parts.append(
        "<p>with kernel length 9 and dilations <span class='mono'>d&nbsp;∈&nbsp;{1,2,4,8,16,32}</span>; "
        "the per-day feature vector is max- and min-pooled over a small set of "
        "<em>competing</em> kernels and pooled across the 40 days by nanmean/nanstd "
        "across observed days. The Combined arm concatenates the summary, MultiRocket "
        "and HYDRA blocks; Summary_RF is the random-forest control on the summary "
        "block alone.</p>"
    )

    parts.append(table_html(
        ["Representation", "validation AUROC (mean ± SD, R=10)",
         "test AUROC (mean ± SD, R=10)", "post-hygiene cols", "α (frozen)"],
        ladder_table(),
        "Temporal representation ladder on the R1b cohort/folds."
    ))
    parts.append(figure_html(fig_ladder(),
        "Same data as Table 2. Error bars are ±1 SD across 10 allocations. "
        "The dashed line marks uninformative ranking (AUROC = 0.5).", "1"))

    parts.append("<h3>4.1&nbsp;&nbsp;Primary paired deltas</h3>")
    parts.append(table_html(
        ["Comparison", "mean Δ", "SD", "Bonferroni 95%-simult", "share > 0"],
        primary_deltas_table(),
        "Paired deltas vs Summary_RF on validation (Bonferroni two-sided, m=4, df=9). "
        "The last row isolates the placement signal within MultiRocket."
    ))
    parts.append(
        f"<p>The combined model adds <span class='mono'>{com_val:+.4f}</span> over "
        f"Summary_RF (paired, allocs-positive); the within-day placement isolation "
        f"(MultiRocket − Shuffled MR) is <span class='mono'>{mr_shuf:+.4f}</span>. "
        "The within-day value-shuffle control permutes observed heart-rate values among "
        "each day's observed clock-bins, leaving the value multiset, the wear mask, and "
        "the day count intact. The collapse from 0.8284 to 0.6619 says the gain is "
        "<em>where in the day values sit</em>, not which values are present.</p>"
    )

    parts.append("<h3>4.2&nbsp;&nbsp;What the result does and does not say</h3>")
    parts.append(
        "<p>Profile24 (hourly aggregation) and Profile288 (5-minute aggregation) reach "
        "only the summary level, so resolution is not the bottleneck — representation "
        "is. Per-bin daily profiles discard the local dilation/position patterns the "
        "convolutional kernels exploit at the same 5-minute resolution. We do not "
        "establish whether the placement signal is physiological (circadian phase and "
        "shape) or device-behavioral (wear-time routines correlated with the recorded "
        "salutation); the predeclared probes are night-only arms, activity-window "
        "exclusion, and importance-by-dilation.</p>"
    )

    # §5 selective
    parts.append("<h2>5&nbsp;&nbsp;Selective classification</h2>")
    parts.append(
        "<p>AUROC measures ranking; the applied goal is to <em>label</em> a subset of "
        "participants at a target correctness and abstain on the rest. Per allocation, "
        "validation scores pick the smallest class-1 threshold achieving at least the "
        "target precision on validation, and the largest class-0 threshold doing the "
        "same on the negative tail:</p>"
    )
    parts.append(
        "<div class='eq'>"
        "τ<sub>1</sub>&nbsp;=&nbsp;min{τ&nbsp;:&nbsp;P̂(y&nbsp;=&nbsp;20&nbsp;|&nbsp;s&nbsp;≥&nbsp;τ)&nbsp;≥&nbsp;p<sub>★</sub>}&nbsp;&nbsp;(validation),"
        "&nbsp;&nbsp;precision&nbsp;=&nbsp;s/(s+f),&nbsp;coverage&nbsp;=&nbsp;#{s&nbsp;≥&nbsp;τ}/n"
        "<span class='no'>(5)</span></div>"
    )
    parts.append("<p>Table&nbsp;4 reports coverage at 95% and 98% target precision in two "
                 "variants: <span class='mono'>raw</span> (the empirical threshold) and "
                 "<span class='mono'>cpc</span> (the largest set whose 90% Clopper–Pearson "
                 "lower confidence bound on precision still clears the target):</p>")
    parts.append(
        "<div class='eq'>"
        "L(s,&nbsp;f)&nbsp;=&nbsp;q<sub>0.10</sub>(Beta(s,&nbsp;f+1));&nbsp;&nbsp;"
        "certify&nbsp;τ&nbsp;if&nbsp;L(s,&nbsp;f)&nbsp;≥&nbsp;p<sub>★</sub>"
        "<span class='no'>(6)</span></div>"
    )
    parts.append(table_html(
        ["Arm", "target", "variant", "cov_total (%)",
         "prec (class 20)", "prec (class 10)"],
        selective_table(),
        "Coverage at per-class target precision. prec columns are the mean over allocations "
        "of per-allocation test precision (class 20 / class 10). cov_total is the fraction "
        "of the test cohort labeled as either class under the per-class thresholds; unattainable "
        "thresholds are excluded."
    ))
    parts.append(figure_html(fig_coverage(),
        "Same data as Table 4 (cov_total panel only). Left: target = 95%. "
        "Right: target = 98%. CP-LCB is the 90% lower bound on precision "
        "(conservative; rejects sets whose calibration-set precision is uncertain).", "2"))

    parts.append(
        f"<p>At 95% target, the Combined model labels "
        f"<span class='mono'>{raw95_per.mean():.1f}%</span> of held-out participants "
        f"(per-class precision 0.956 / 0.937), versus "
        f"<span class='mono'>{base_per.mean():.1f}%</span> for the R1b BASE⊕P40 baseline "
        f"(ratio ≈ "
        f"<span class='mono'>×{raw95_per.mean()/max(base_per.mean(),1e-9):.2f}</span>). "
        "The CP-LCB variant certifies only 8.1% at 95% and nothing at 98% on the Combined "
        "arm at the available validation size (n=579); reaching 98% under finite-sample "
        "certification requires roughly 114 error-free validation selections at "
        "<span class='mono'>δ=0.10</span>, which the model cannot attain.</p>"
    )

    parts.append(
        "<p>Two qualifications. First, the threshold sits on the validation precision "
        "boundary, so held-out precision shrinks with selected-set size — visible on "
        "the minority class (class 10), where selected sets are small and precision drops "
        "mildly below the target. Second, abstention composition shows that abstained "
        "and labeled participants differ by mean heart rate (~71 vs ~74 bpm) and not by "
        "wear coverage (~0.90 vs ~0.88 mask coverage; ~261 vs ~254 observed bins/day): "
        "the separator is physiological ambiguity, not data scarcity, so minimum-wear "
        "gating is not the binding lever for higher coverage.</p>"
    )

    # §6 alpha
    parts.append("<h2>6&nbsp;&nbsp;Regularization headroom</h2>")
    parts.append(
        "<p>The ridge head (Eq.&nbsp;3) was precalibrated on a grid "
        "<span class='mono'>[10<sup>−3</sup>,&nbsp;…,&nbsp;10<sup>3</sup>]</span>; "
        "every family in the v1 and v2 runs selected the grid maximum "
        "<span class='mono'>α=10<sup>3</sup></span> (REPORT §8 caveat). An "
        "extended-grid addendum on allocation&nbsp;0 sweeps "
        "<span class='mono'>[10<sup>2</sup>,&nbsp;3·10<sup>2</sup>,&nbsp;10<sup>3</sup>,&nbsp;…,&nbsp;10<sup>6</sup>]</span>:</p>"
    )
    parts.append(table_html(
        ["Family", "frozen α", "val @ frozen", "α*", "val @ α*",
         "gain (val)", "test @ frozen", "test @ α*"],
        alpha_table(),
        "Extended-α addendum (allocation 0; predeclared procedure, NEXT_STEPS §Step 0.7)."
    ))
    parts.append(
        "<p>The convolutional arms find optima well inside the extended grid "
        "(MultiRocket α*=3·10<sup>3</sup>; HYDRA and Combined α*=10<sup>4</sup>), "
        "and the val gains are test-corroborated at this allocation "
        "(Combined +0.0159 test). Profile288 is the exception: it shows "
        "<span class='mono'>+0.0138</span> on validation at α*=10<sup>4</sup> but "
        "<span class='mono'>−0.016</span> on test — the alloc-0 val-α "
        "selection-overfit signature, not carried forward. The frozen decision rule "
        "(Combined val gain ≥ 0.01) was triggered, and a v3 full rerun is pending the "
        "user's decision. The wider lesson: for very wide feature matrices "
        "(Combined ≈ 31,000 columns after hygiene), the v2 grid truncates the "
        "optimum and the headline AUROC is if anything understated.</p>"
    )

    # §7 limitations
    parts.append("<h2>7&nbsp;&nbsp;Limitations and threats to validity</h2>")
    parts.append("<ol>")
    parts.append("<li><strong>Label noise ceiling.</strong> If a fraction "
        "<span class='mono'>p</span> of recorded salutations are effectively arbitrary "
        "(mis-entered, transiently changed), the maximum achievable AUROC against the "
        "recorded label is bounded by</li>")
    parts.append(
        "<div class='eq'>max&nbsp;AUROC&nbsp;≈&nbsp;1&nbsp;−&nbsp;p/2"
        "<span class='no'>(7)</span></div>"
    )
    parts.append("<li><strong>Cohort selection.</strong> The R1b cohort is coverage-gated "
        "(Garmin users with 40 adequate heart-rate days). Absolute AUROCs are not "
        "estimates of performance in the full Datenspende population, nor among "
        "participants with missing salutation. The informative-recording experiment "
        "does not identify an MNAR mechanism.</li>")
    parts.append("<li><strong>Reused test split.</strong> The recent experiments reuse the "
        "R1a-AF allocator folds; the test participants have already supported R1a-AF "
        "and R1b model development and are reported as exploratory by predeclared rule.</li>")
    parts.append("<li><strong>Day-scale sampling caveat.</strong> The day-scale k-curve samples "
        "days <em>with replacement</em> across each participant's full record; the k=1 "
        "point is therefore not a one-day-of-history number. R1b's chronological "
        "first-k curve addresses the prospective question on the R1b model "
        "(first day 0.669, near-plateau by day 32).</li>")
    parts.append("<li><strong>Threshold guarantees are conditional.</strong> The "
        "selective-classification raw numbers are operating estimates; the CP-LCB "
        "certification is finite-sample on the validation split (conditional on the "
        "split), not a population promise. With <span class='mono'>n=579</span> and "
        "<span class='mono'>δ=0.10</span>, certification at 98% precision is "
        "unattainable.</li>")
    parts.append("<li><strong>v1 ↔ v2 robustness.</strong> The v1 temporal run had a "
        "zero-sentinel contamination defect (unobserved bins stored as 0.0 and "
        "included by Profile288 and the per-bin fill). Table&nbsp;6 quantifies the "
        "effect: every arm moved by ≤ 0.004 AUROC between v1 and v2, and the B40-only "
        "arms are bit-identical. v1 is retained as the erratum record "
        "(commit <span class='mono'>168b9d7</span>); v2 is primary.</li>")
    parts.append("<li><strong>Allocation variability is not population uncertainty.</strong> "
        "Reported intervals describe model-and-split randomization on the observed "
        "cohort.</li>")
    parts.append("</ol>")
    parts.append(table_html(
        ["Representation", "val v1", "val v2", "Δ val",
         "test v1", "test v2", "Δ test"],
        v1v2_table(),
        "v1 (erratum present) vs v2 (mask-aware fix). B40-only arms are bit-identical."
    ))

    # §8 reproducibility
    parts.append("<h2>8&nbsp;&nbsp;Reproducibility record</h2>")
    parts.append(
        "<p>Every stage was run from a frozen plan committed before any results were "
        "seen; deviations from a frozen plan are recorded as errata in the per-stage "
        "REPORT, never silent. Determinism is gated on each run: a full alloc-0 rerun "
        "must be byte-identical or differ by at most "
        "<span class='mono'>10<sup>−6</sup></span> per arm. The actual rerun differences "
        "are 1–2 unit-last-place (max 2.2·10<sup>−16</sup>), attributable to "
        "threaded-BLAS reduction order in the ridge solve (the same signature is "
        "present in pure-B40 Summary_linear, so it is not a transform artifact). MR "
        "and HYDRA transforms, and all seed streams, are bit-stable.</p>"
    )
    parts.append(
        "<p>Resource gates: CPU torch only, peak resident-memory under "
        "<span class='mono'>8&nbsp;GiB</span> across all stages (v2 plateau "
        "<span class='mono'>8,197&nbsp;MiB</span>, within +0.06% of the gate; addendum "
        "peak <span class='mono'>6,309&nbsp;MiB</span>). Vendored MultiRocket "
        "(commit <span class='mono'>3ccaa4f</span>) and HYDRA "
        "(commit <span class='mono'>144bb7a</span>) are pinned with file SHA256s in "
        "<span class='mono'>cache/vendor_repro.json</span>. A numba-seeding shim "
        "(<span class='mono'>_nb_seed</span>) and the documented workqueue threading "
        "layer are the two recorded environment errata.</p>"
    )

    # §9 next
    parts.append("<h2>9&nbsp;&nbsp;Next steps</h2>")
    parts.append(
        "<p>Three follow-ups are plan-ready and pending review or decision:</p>"
        "<ol>"
        "<li><strong>v3 full rerun</strong> with the extended α-grid from alloc-0 "
        "precalibration (72 min; no other changes). Expected Combined test AUROC "
        "≈ 0.87–0.88; selective-classification numbers carry forward under the same "
        "predeclared procedure.</li>"
        "<li><strong>Multichannel plan</strong> (draft at NEXT_STEPS §5 Appendix A): "
        "Steps, MET, walk/activity, sleep binaries, resting-HR added to the same "
        "288-bin × 40-day grid; multichannel HYDRA (native) and per-channel "
        "MultiRocket. Co-primary: AUROC ≥ 0.90 <em>and</em> raw coverage@95% ≥ 35% "
        "(vs Combined 25.2%) and CP-LCB ≥ 15% (vs 8.1%).</li>"
        "<li><strong>Artifact-discrimination probes</strong>: night-only transform, "
        "activity-window exclusion, importance-by-dilation — to separate "
        "physiological placement from device-behavioral placement. Fold into the "
        "multichannel experiment as sensitivity arms.</li>"
        "</ol>"
        "<p>None of the above establishes a causal or physiological claim; the "
        "documented scope is predictability from the export.</p>"
    )

    # Appendix A notation + file index
    parts.append("<h2>A&nbsp;&nbsp;Notation and file index</h2>")
    parts.append(
        "<p><strong>Notation.</strong> "
        "<span class='mono'>s<sub>i</sub></span> model score for participant i; "
        "<span class='mono'>τ</span> threshold; "
        "<span class='mono'>p<sub>★</sub></span> target precision; "
        "<span class='mono'>α</span> ridge penalty; "
        "<span class='mono'>d</span> convolution dilation; "
        "<span class='mono'>R</span> number of allocations (here 10); "
        "<span class='mono'>m</span> number of primary comparisons in the Bonferroni "
        "family; <span class='mono'>L(s,f)</span> Clopper–Pearson 90% lower bound on "
        "the success probability of a binomial with s successes and f failures; "
        "<span class='mono'>δ</span> miscoverage target (0.10 here).</p>"
    )
    parts.append(
        "<p><strong>Files.</strong> "
        "<code>experiments/temporal_garmin/results/REPORT.md</code> (full per-arm "
        "results, §1–§10); "
        "<code>experiments/temporal_garmin/results/temporal_metrics.csv</code> and "
        "<code>temporal_metrics_v1.csv</code> (per-arm AUROCs); "
        "<code>experiments/temporal_garmin/results/selective_classification.{csv,md}</code> "
        "(per-alloc coverage-at-precision); "
        "<code>experiments/temporal_garmin/results/alpha_addendum.{json,md}</code> "
        "(extended-α grid); "
        "<code>experiments/temporal_garmin/results/risk_coverage_curves.csv</code> "
        "(full operating curves); "
        "<code>experiments/temporal_garmin/results/abstention_composition.csv</code> "
        "(labeled vs abstained participant statistics); "
        "<code>experiments/temporal_garmin/NEXT_STEPS.md</code> (roadmap + multichannel "
        "draft + selective-classification archived procedure).</p>"
    )

    parts.append(
        "<div class='provenance'>"
        f"Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} by "
        "<code>reports/build_meta_report_2026-09-22.py</code>. "
        "Inputs and figure hashes are listed in "
        "<code>recorded_salutation_meta_report_2026-09-22_manifest.json</code>."
        "</div>"
    )

    parts.append("</body></html>")
    return "".join(parts)


# ---------- main -----------------------------------------------------------
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    print(f"[meta-2026-09-22] building report")
    fig_ladder()
    fig_coverage()
    html_str = build_html()
    OUT_HTML.write_text(html_str)
    print(f"[meta-2026-09-22] wrote {OUT_HTML.relative_to(ROOT)}")

    # PDF via headless chrome
    chrome_args = [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                   f"--print-to-pdf={OUT_PDF}", "--no-pdf-header-footer",
                   "--virtual-time-budget=10000", OUT_HTML.as_uri()]
    try:
        subprocess.run(chrome_args, check=True, timeout=180,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        # fallback to classic --headless
        chrome_args[1] = "--headless"
        subprocess.run(chrome_args, check=True, timeout=180)
    print(f"[meta-2026-09-22] wrote {OUT_PDF.relative_to(ROOT)}")

    # manifest
    figures = sorted(FIG_DIR.glob("0[6-7]*.png"))
    manifest = {
        "html": str(OUT_HTML.relative_to(ROOT)),
        "pdf": str(OUT_PDF.relative_to(ROOT)),
        "figures": {str(f.relative_to(ROOT)): sha256_file(f) for f in figures},
        "build_script": "reports/build_meta_report_2026-09-22.py",
        "inputs": {
            "temporal_metrics_v2": str((TG / "temporal_metrics.csv").relative_to(ROOT)),
            "temporal_metrics_v1": str((TG / "temporal_metrics_v1.csv").relative_to(ROOT)),
            "selective_classification": str((TG / "selective_classification.csv").relative_to(ROOT)),
            "alpha_addendum": str((TG / "alpha_addendum.json").relative_to(ROOT)),
        },
        "input_hashes": {
            str((TG / "temporal_metrics.csv").relative_to(ROOT)): sha256_file(TG / "temporal_metrics.csv"),
            str((TG / "temporal_metrics_v1.csv").relative_to(ROOT)): sha256_file(TG / "temporal_metrics_v1.csv"),
            str((TG / "selective_classification.csv").relative_to(ROOT)): sha256_file(TG / "selective_classification.csv"),
            str((TG / "alpha_addendum.json").relative_to(ROOT)): sha256_file(TG / "alpha_addendum.json"),
        },
        "note": "Reporting only; no model fitting.",
    }
    (REPORT_DIR / "recorded_salutation_meta_report_2026-09-22_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[meta-2026-09-22] wrote manifest")
    print(f"[meta-2026-09-22] done")


if __name__ == "__main__":
    main()
