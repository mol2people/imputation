#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""06_plot_long_period_check.py -- figure R7 and numeric summary from long_period_check.npz (refuses to overwrite)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
Z = np.load(os.path.join(EXP, "results/garmin_k15/long_period_check.npz"), allow_pickle=True)
w, y, age = Z["wide"], Z["y"], Z["age"].astype(str)
SX, SC = {0: "Female", 1: "Male"}, {0: "#d62728", 1: "#1f77b4"}
path = os.path.join(EXP, "figures/R7_long_period_male_rise.png")
if os.path.exists(path):
    raise SystemExit("exists; refusing to overwrite")

fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
for j, (lab, o, a, t) in enumerate([("v2 baseline", "obs", "ar", "tc"), ("baseline + quadratic + cubic", "obs_poly", "ar_poly", "tc_poly")]):
    pass
cols = [("v2 baseline", "obs", "ar", "tc"), ("+ quadratic and cubic trend", "obs_poly", "ar_poly", "tc_poly")]
m_all = {s: (y == s) & (age != "Elderly") for s in (0, 1)}
for i, s in enumerate((0, 1)):
    for j, (lab, o, a, t) in enumerate(cols):
        ax = axs[i, j]
        m = m_all[s]
        ax.plot(w, Z[o][m].mean(0), c=SC[s], lw=2, label="observed")
        ax.plot(w, Z[a][m].mean((0, 1)), "k--", label="AR(1) null")
        ax.plot(w, Z[t][m].mean((0, 1)), c="green", label="AR(1)+slow null")
        ax.axvspan(25, 35, color="gold", alpha=.15)
        ax.set_xscale("log")
        ax.set_title(f"Garmin {SX[s]}, age 18-64 (n={m.sum()}) | {lab}", fontsize=9)
        ax.set_xlabel("period (d)")
        ax.set_ylabel("mean partial R$^2$")
    ax = axs[i, 2]
    m = m_all[s]
    ax.plot(w, Z["obs"][m].mean(0) - Z["tc"][m].mean((0, 1)), c="gray", label="v2 baseline")
    ax.plot(w, Z["obs_poly"][m].mean(0) - Z["tc_poly"][m].mean((0, 1)), c=SC[s], label="poly baseline")
    ax.axhline(0, c="k", lw=.7)
    ax.axvspan(25, 35, color="gold", alpha=.15)
    ax.set_xscale("log")
    ax.set_title(f"{SX[s]}: observed minus AR(1)+slow null", fontsize=9)
    ax.set_xlabel("period (d)")
axs[0, 0].legend(fontsize=8)
axs[0, 2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(path, dpi=200)

print("period of max mean excess over AR(1)+slow null (d) by sex/baseline (all ages 18-64):")
for s in (0, 1):
    m = m_all[s]
    for lab, o, t in [("v2", "obs", "tc"), ("poly", "obs_poly", "tc_poly")]:
        ex = Z[o][m].mean(0) - Z[t][m].mean((0, 1))
        sel = w >= 20
        print(SX[s], lab, "argmax period:", round(float(w[sel][ex[sel].argmax()]), 1), "max excess:", round(float(ex[sel].max()), 4),
              "| mean excess 40-120d:", round(float(ex[w >= 40].mean()), 4), "| 25-35d:", round(float(ex[(w >= 25) & (w <= 35)].mean()), 4))
for s in (0, 1):
    m = m_all[s]
    pk = w[Z["obs"][m].argmax(1)]
    print(SX[s], "per-participant peak period (v2 baseline, 3-120 d): median", np.median(pk), "frac at >=100d", float((pk >= 100).mean()),
          "frac 25-35d", float(((pk >= 25) & (pk <= 35)).mean()))
    pk = w[Z["obs_poly"][m].argmax(1)]
    print(SX[s], "poly baseline: median", np.median(pk), "frac >=100d", float((pk >= 100).mean()), "frac 25-35d", float(((pk >= 25) & (pk <= 35)).mean()))
