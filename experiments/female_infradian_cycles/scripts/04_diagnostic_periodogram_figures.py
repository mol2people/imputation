"""Diagnostic figures from stored per-participant LS/cosinor metrics (no raw series needed).

Outputs to experiments/female_infradian_cycles/figures/diagnostics/.
Note: LS peak period is searched only within 25-35 d, so peak location is bounded;
pile-up at window edges indicates red-noise leakage (monotone spectrum), not a rhythm.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "female_infradian_cycles/figures/diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(ROOT / "female_infradian_cycles/results/infradian_metrics_by_participant.csv")
d = d[d.screened_pass].copy()
sl = pd.read_csv(ROOT / "female_infradian_extensions/results/sliding_window_stability.csv")
ac = pd.read_csv(ROOT / "female_infradian_extensions/results/acrophase_uniformity_tests.csv")

d["grp"] = d.sex + " / " + d.age_group
order = [f"{s} / {a}" for a in ["Young Adults", "Middle Ager", "Elderly"] for s in ["Female", "Male"]]
col = {g: c for g, c in zip(order, ["#d62728", "#1f77b4", "#e377c2", "#17becf", "#ff9896", "#aec7e8"])}
BINS = np.arange(24.5, 35.6, 1)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


# 1. Peak-period histograms per group, with uniform expectation (edge pile-up check)
fig, axs = plt.subplots(2, 3, figsize=(13, 6), sharey=False)
for ax, g in zip(axs.T.ravel(), order):
    x = d.loc[d.grp == g, "ls_period_hr_p10"].dropna()
    ax.hist(x, BINS, color=col[g], alpha=.8)
    ax.axhline(len(x) / (len(BINS) - 1), ls="--", c="k", lw=1)
    ax.axvspan(27, 30, color="gray", alpha=.12)
    ax.set_title(f"{g} (n={len(x)})", fontsize=10)
    ax.set_xlabel("LS peak period, hr_p10 (d)")
fig.suptitle("LS peak-period histogram; dashed = uniform null; shaded = 27-30 d")
save(fig, "d1_peak_period_histograms.png")

# 2. Peak power vs period, hexbin-free scatter by group (female vs male, young)
fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
for ax, a in zip(axs, ["Young Adults", "Middle Ager", "Elderly"]):
    for s, c in [("Male", "#1f77b4"), ("Female", "#d62728")]:
        x = d[(d.sex == s) & (d.age_group == a)]
        ax.scatter(x.ls_period_hr_p10, x.ls_power_hr_p10, s=6, alpha=.25, c=c, label=f"{s} n={len(x)}")
    ax.set_title(a)
    ax.set_xlabel("peak period (d)")
    ax.legend(markerscale=3)
axs[0].set_ylabel("LS peak power (hr_p10)")
save(fig, "d2_power_vs_period_scatter.png")

# 3. ECDF of LS SNR and power by group (distributional, not just means)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, m in zip(axs, ["ls_power_hr_p10", "ls_snr_hr_p10"]):
    for g in order:
        x = np.sort(d.loc[d.grp == g, m].dropna())
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), c=col[g], label=g, lw=1.6)
    ax.set_xlabel(m)
    ax.set_ylabel("ECDF")
    ax.set_xscale("log")
axs[0].legend(fontsize=8)
save(fig, "d3_ecdf_power_snr.png")

# 4. Physiology vs wear-proxy: power of hr_p10 vs occupancy / nocturnal compliance (per-participant, young)
sigs = ["hr_p10", "night_hr", "day_hr", "dipping_ratio", "occ_daily", "nocturnal_compliance", "gap_episodes"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, a in zip(axs, ["Young Adults", "Middle Ager"]):
    rows = []
    for s in ["Female", "Male"]:
        x = d[(d.sex == s) & (d.age_group == a)]
        rows.append([x[f"ls_power_{k}"].median() for k in sigs])
    w = .38
    ax.bar(np.arange(len(sigs)) - w / 2, rows[0], w, color="#d62728", label="Female")
    ax.bar(np.arange(len(sigs)) + w / 2, rows[1], w, color="#1f77b4", label="Male")
    ax.set_xticks(range(len(sigs)))
    ax.set_xticklabels(sigs, rotation=35, ha="right")
    ax.set_title(f"{a}: median LS peak power per signal")
    ax.legend()
axs[0].set_ylabel("median LS power")
save(fig, "d4_power_by_signal_sex.png")

# 5. Cross-signal period concordance in young females (is hr_p10 peak shared with night_hr / wear proxies?)
yf = d[(d.sex == "Female") & (d.age_group == "Young Adults")]
fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, k in zip(axs, ["night_hr", "day_hr", "occ_daily"]):
    ax.scatter(yf.ls_period_hr_p10 + np.random.default_rng(0).normal(0, .12, len(yf)),
               yf[f"ls_period_{k}"] + np.random.default_rng(1).normal(0, .12, len(yf)), s=8, alpha=.4)
    ax.plot([25, 35], [25, 35], "k--", lw=1)
    r = yf[["ls_period_hr_p10", f"ls_period_{k}"]].corr(method="spearman").iloc[0, 1]
    ax.set_title(f"Young F: hr_p10 vs {k}\nSpearman={r:.2f}")
    ax.set_xlabel("hr_p10 peak period (d)")
    ax.set_ylabel(f"{k} peak period (d)")
save(fig, "d5_cross_signal_period_concordance.png")

# 6. Sliding-window stability: persistence ratio and period drift by group
sl["grp"] = sl.sex + " / " + sl.age_group
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (m, lab) in zip(axs, [("hr_p10_persistence_ratio", "persistence ratio"),
                              ("hr_p10_period_iqr_drift", "period IQR drift (d)"),
                              ("hr_p10_mean_power", "mean window power")]):
    gs = sorted(sl.grp.unique())
    ax.violinplot([sl.loc[sl.grp == g, m].dropna() for g in gs], showmedians=True)
    ax.set_xticks(range(1, len(gs) + 1))
    ax.set_xticklabels([f"{g}\n(n={sl.grp.eq(g).sum()})" for g in gs], fontsize=7, rotation=20)
    ax.set_title(lab)
save(fig, "d6_sliding_window_stability.png")

# 7. Acrophase Rayleigh summary: R-bar with Rayleigh p
fig, ax = plt.subplots(figsize=(9, 4.5))
lab = ac.cohort + " | " + ac.signal + " | " + ac.phase_type
ax.barh(lab, ac.mean_resultant_R_bar, color=np.where(ac.p_value < .05, "#d62728", "#7f7f7f"))
for i, (r, p) in enumerate(zip(ac.mean_resultant_R_bar, ac.p_value)):
    ax.text(r, i, f" p={p:.3g}", va="center", fontsize=7)
ax.set_xlabel("mean resultant length R-bar (red: Rayleigh p<0.05, uncorrected)")
save(fig, "d7_rayleigh_rbar.png")

# 8. Window-edge pile-up statistic: fraction of peaks within 1 d of window edges vs 2/11 uniform
rows = []
for g in order:
    x = d.loc[d.grp == g, "ls_period_hr_p10"].dropna()
    rows.append((g, ((x <= 26) | (x >= 34)).mean(), len(x)))
e = pd.DataFrame(rows, columns=["grp", "edge_frac", "n"])
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(e.grp, e.edge_frac, color=[col[g] for g in e.grp])
ax.axhline(2 / 10, ls="--", c="k")
ax.set_ylabel("fraction of peaks in [25,26]U[34,35]")
ax.set_xticklabels(e.grp, rotation=25, ha="right", fontsize=8)
ax.set_title("Edge pile-up (dashed = uniform expectation 0.20)")
save(fig, "d8_edge_pileup.png")
print(e)
