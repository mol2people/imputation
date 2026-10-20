#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""
04_replication_figures.py -- figures comparing Apple v2 (k=15), Apple k=10 and Garmin k=15.
Writes experiments/garmin_infradian_replication/figures/R1..R6; refuses to overwrite existing files.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(EXP, "../.."))
FIG = os.path.join(EXP, "figures")
os.makedirs(FIG, exist_ok=True)
V2R = os.path.join(ROOT, "experiments/female_infradian_cycles_v2/results")

ap15 = pd.read_parquet(os.path.join(V2R, "recalibration/null2_participants.parquet"))
ap10 = pd.read_parquet(os.path.join(EXP, "results/apple_k10/participants.parquet"))
ga15 = pd.read_parquet(os.path.join(EXP, "results/garmin_k15/participants.parquet"))
for d in (ap10, ga15):
    d.dropna(subset=["E_i2"], inplace=True)
COH = {"Apple k=15 (v2)": ap15, "Apple k=10": ap10, "Garmin k=15": ga15}
COL = {"Apple k=15 (v2)": "#444444", "Apple k=10": "#1f77b4", "Garmin k=15": "#2ca02c"}
ages = ["Young Adults", "Middle Ager", "Elderly"]
SX = {0: "Female", 1: "Male"}
SC = {0: "#d62728", 1: "#1f77b4"}


def save(f, n):
    path = os.path.join(FIG, n)
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; refusing to overwrite")
    f.tight_layout()
    f.savefig(path, dpi=200)
    plt.close(f)


# R1 forest plot of F-M contrasts (E_i, two-component null)
c = {"Apple k=15 (v2)": pd.read_csv(os.path.join(V2R, "recalibration/contrasts_null2.csv")),
     "Apple k=10": pd.read_csv(os.path.join(EXP, "results/apple_k10/contrasts.csv")),
     "Garmin k=15": pd.read_csv(os.path.join(EXP, "results/garmin_k15/contrasts.csv"))}
tests = ["Young Adults", "Middle Ager", "Elderly", "Standardized"]
fig, ax = plt.subplots(figsize=(9, 4.8))
for j, (name, t) in enumerate(c.items()):
    t = t[t.endpoint == "E_i2"].set_index("test")
    y = np.arange(len(tests)) + (j - 1) * .22
    ax.errorbar(t.loc[tests, "est"], y, xerr=1.96 * t.loc[tests, "se"], fmt="o", c=COL[name], label=name, capsize=3)
ax.axvline(0, c="k", lw=.8)
ax.set_yticks(range(len(tests)))
ax.set_yticklabels(tests)
ax.invert_yaxis()
ax.set_xlabel("Female - male difference in excess score E_i (95% HC3 CI; AR(1)+slow null)")
ax.legend()
save(fig, "R1_forest_contrasts.png")

# R2 rejection rates
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, a in zip(axs, ages):
    x = np.arange(2)
    for j, (name, d) in enumerate(COH.items()):
        r = [(d[(d.y == s) & (d.age_group == a)].p_i2 < .05).mean() for s in (0, 1)]
        ax.bar(x + (j - 1) * .26, r, .26, color=COL[name], label=name)
    ax.axhline(.05, ls="--", c="k")
    ax.set_xticks(x)
    ax.set_xticklabels(["Female", "Male"])
    ax.set_title(a)
axs[0].set_ylabel("fraction with p < 0.05 (AR(1)+slow null)")
axs[0].legend(fontsize=8)
save(fig, "R2_rejection_rates.png")

# R3 QQ plots, young and middle
fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
for i, a in enumerate(["Young Adults", "Middle Ager"]):
    for j, (name, d) in enumerate(COH.items()):
        ax = axs[i, j]
        for s in (0, 1):
            p = np.sort(d.loc[(d.y == s) & (d.age_group == a), "p_i2"].values)
            e = (np.arange(1, len(p) + 1) - .5) / len(p)
            ax.plot(-np.log10(e), -np.log10(p), c=SC[s], label=f"{SX[s]} n={len(p)}")
        ax.plot([0, 2.5], [0, 2.5], "k--", lw=1)
        ax.set_title(f"{name} | {a}", fontsize=9)
        ax.legend(fontsize=8)
axs[1, 0].set_xlabel("expected -log10 p")
axs[1, 0].set_ylabel("observed -log10 p")
save(fig, "R3_pvalue_qq.png")

# R4 distribution of E_i2
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, (name, d) in zip(axs, COH.items()):
    data, pos, cols = [], [], []
    for k, a in enumerate(ages):
        for s in (0, 1):
            data.append(d.loc[(d.y == s) & (d.age_group == a), "E_i2"].dropna().values)
            pos.append(k * 3 + s)
            cols.append(SC[s])
    vp = ax.violinplot(data, pos, showmedians=True, widths=.9)
    for b, cc in zip(vp["bodies"], cols):
        b.set_facecolor(cc)
        b.set_alpha(.5)
    ax.set_xticks([.5, 3.5, 6.5])
    ax.set_xticklabels(ages)
    ax.axhline(0, c="k", lw=.7)
    ax.set_title(name)
axs[0].set_ylabel("E_i (AR(1)+slow null)")
save(fig, "R4_excess_score_distributions.png")

# R5 persistence
rng = np.random.default_rng(0)
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, name in zip(axs[:2], ["Apple k=10", "Garmin k=15"]):
    d = COH[name]
    d = d[d.delta_T.notna()]
    for s, a, ls in [(0, "Young Adults", "-"), (0, "Middle Ager", ":"), (1, "Young Adults", "-"), (1, "Middle Ager", ":")]:
        x = np.sort(d.loc[(d.y == s) & (d.age_group == a), "delta_T"].values)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), c=SC[s], ls=ls, label=f"{SX[s]} {a} (n={len(x)})")
    x = np.sort(np.abs(d.best_T.values - rng.permutation(d.best_T_w2.values)))
    ax.plot(x, np.arange(1, len(x) + 1) / len(x), "k--", label="shuffled partners")
    ax.set_xlim(0, 10)
    ax.set_xlabel("|T1 - T2| (d)")
    ax.set_title(name)
    ax.legend(fontsize=7)
bins = np.linspace(-1, 1, 31)
for s in (0, 1):
    for name, ls in [("Apple k=10", "-"), ("Garmin k=15", "--")]:
        d = COH[name]
        v = d.loc[(d.y == s) & (d.age_group == "Young Adults"), "corr_persist"].dropna()
        axs[2].hist(v, bins, histtype="step", color=SC[s], ls=ls, density=True, label=f"{SX[s]} {name} (n={len(v)})")
axs[2].axvline(0, c="k", lw=.7)
axs[2].set_xlabel("corr(window-2 residual, frozen window-1 harmonic)")
axs[2].set_title("Frozen period and phase, young adults")
axs[2].legend(fontsize=7)
save(fig, "R5_persistence.png")

# R6 Garmin spectra vs AR(1)
Z = np.load(os.path.join(EXP, "results/garmin_k15/spectra.npz"), allow_pickle=True)
w = Z["wide"]
fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for j, a in enumerate(ages):
    for i, s in enumerate((0, 1)):
        ax = axs[i, j]
        m = (Z["y"] == s) & (Z["age"].astype(str) == a)
        ar = Z["ar"][m].mean(0)
        lo, hi = np.percentile(ar, [2.5, 97.5], axis=0)
        ax.fill_between(w, lo, hi, color="gray", alpha=.35, label="AR(1) surrogate 95% (10 reps)")
        ax.plot(w, ar.mean(0), "k--", lw=1, label="AR(1) mean")
        ax.plot(w, Z["obs"][m].mean(0), c=SC[s], lw=1.8, label="observed")
        ax.axvspan(25, 35, color="gold", alpha=.15)
        ax.set_xscale("log")
        ax.set_title(f"Garmin {SX[s]} / {a} (n={m.sum()})", fontsize=10)
        ax.set_xlabel("period (d)")
axs[0, 0].legend(fontsize=7)
save(fig, "R6_garmin_mean_periodogram.png")
print(sorted(os.listdir(FIG)))
