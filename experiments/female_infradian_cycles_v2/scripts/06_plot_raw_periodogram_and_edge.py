#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""
06_plot_raw_periodogram_and_edge.py -- figures/statistics from spectra.npz (see 05_*.py).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.abspath(os.path.join(HERE, "../results/raw_periodogram"))
Z = np.load(os.path.join(D, "spectra.npz"), allow_pickle=True)
y, age, phi = Z["y"], Z["age"].astype(str), Z["phi"]
wide, narrow = Z["wide"], Z["narrow"]
ages = ["Young Adults", "Middle Ager", "Elderly"]
sexes = {0: "Female", 1: "Male"}
freq = 1.0 / wide


def save(fig, n):
    fig.tight_layout()
    fig.savefig(os.path.join(D, n), dpi=200)
    plt.close(fig)


# ---- Fig A: mean spectrum observed vs AR(1) envelope, per sex x age -------------------------
fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for j, a in enumerate(ages):
    for i, s in enumerate([0, 1]):
        ax = axs[i, j]
        m = (y == s) & (age == a)
        obs = Z["w_obs"][m].mean(0)
        ar = Z["w_ar"][m].mean(0)                       # (B, K) mean over participants per replicate
        lo, hi = np.percentile(ar, [2.5, 97.5], axis=0)
        ax.fill_between(wide, lo, hi, color="gray", alpha=.35, label="AR(1) surrogate 95% (30 reps)")
        ax.plot(wide, ar.mean(0), "k--", lw=1, label="AR(1) surrogate mean")
        ax.plot(wide, obs, c="#d62728" if s == 0 else "#1f77b4", lw=1.8, label="observed")
        ax.axvspan(25, 35, color="gold", alpha=.15)
        ax.set_xscale("log")
        ax.set_title(f"{sexes[s]} / {a} (n={m.sum()})", fontsize=10)
        ax.set_xlabel("period (d)")
        ax.set_ylabel("mean partial R$^2$ of harmonic")
axs[0, 0].legend(fontsize=7)
save(fig, "A_mean_periodogram_vs_ar1.png")

# ---- Fig B: excess = observed - surrogate mean, with surrogate band, young F vs M ------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for j, a in enumerate(ages):
    ax = axs[j]
    for s, c in [(0, "#d62728"), (1, "#1f77b4")]:
        m = (y == s) & (age == a)
        ar = Z["w_ar"][m].mean(0)
        ex = Z["w_obs"][m].mean(0) - ar.mean(0)
        lo, hi = np.percentile(ar - ar.mean(0), [2.5, 97.5], axis=0)
        ax.plot(wide, ex, c=c, label=sexes[s])
        ax.fill_between(wide, lo, hi, color=c, alpha=.15)
    ax.axhline(0, c="k", lw=.7)
    ax.axvspan(25, 35, color="gold", alpha=.15)
    ax.set_xscale("log")
    ax.set_title(a)
    ax.set_xlabel("period (d)")
axs[0].set_ylabel("observed - surrogate mean")
axs[0].legend()
save(fig, "B_excess_over_ar1_by_sex.png")

# ---- Fig C: argmax distribution on 25-35 d grid, observed vs AR(1) vs white ------------------
def argmax_hist(S):
    idx = S.argmax(-1).ravel()
    return np.bincount(idx, minlength=len(narrow)) / len(idx)


rows = []
fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
for j, a in enumerate(ages):
    for i, s in enumerate([0, 1]):
        ax = axs[i, j]
        m = (y == s) & (age == a)
        ho, ha, hw = argmax_hist(Z["n_obs"][m]), argmax_hist(Z["n_ar"][m]), argmax_hist(Z["n_wn"][m])
        w = .27
        ax.bar(narrow - w, ho, w, color="#d62728" if s == 0 else "#1f77b4", label="observed")
        ax.bar(narrow, ha, w, color="gray", label="AR(1) surrogate")
        ax.bar(narrow + w, hw, w, color="gold", label="white noise")
        ax.set_title(f"{sexes[s]} / {a}", fontsize=10)
        ax.set_xlabel("argmax period (d)")
        rows.append((sexes[s], a, m.sum(), ho[-1], ha[-1], hw[-1], ho[0], ha[0], hw[0]))
axs[0, 0].legend(fontsize=8)
save(fig, "C_argmax_obs_vs_null.png")

# ---- Fig D: spectral window (design bias) + score profile on 25-35 grid ---------------------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
axs[0].plot(narrow, Z["n_wn"].mean((0, 1)), c="gold", lw=2, label="white noise (design window)")
axs[0].plot(narrow, Z["n_ar"].mean((0, 1)), c="gray", lw=2, label="AR(1) surrogate")
for s, c in [(0, "#d62728"), (1, "#1f77b4")]:
    axs[0].plot(narrow, Z["n_obs"][y == s].mean(0), c=c, lw=2, label=f"observed {sexes[s]}")
axs[0].set_xlabel("period (d)")
axs[0].set_ylabel("mean score")
axs[0].legend(fontsize=8)
axs[0].set_title("Mean score across 25-35 d grid")
# P(argmax = 35) vs phi tercile
q = np.quantile(phi, [0, .25, .5, .75, 1.0])
bins = np.digitize(phi, q[1:-1])
x, po, pa = [], [], []
for b in range(4):
    m = bins == b
    x.append(f"{phi[m].mean():.2f}")
    po.append((Z["n_obs"][m].argmax(1) == len(narrow) - 1).mean())
    pa.append((Z["n_ar"][m].argmax(2) == len(narrow) - 1).mean())
xx = np.arange(4)
axs[1].bar(xx - .2, po, .4, label="observed", color="#555")
axs[1].bar(xx + .2, pa, .4, label="AR(1) surrogate", color="gray", alpha=.6)
axs[1].axhline(1 / len(narrow), ls="--", c="k")
axs[1].set_xticks(xx)
axs[1].set_xticklabels(x)
axs[1].set_xlabel("fitted $\\phi$ (quartile means)")
axs[1].set_ylabel("P(argmax = 35 d)")
axs[1].legend()
save(fig, "D_window_bias_and_phi_dependence.png")

import pandas as pd
t = pd.DataFrame(rows, columns=["sex", "age", "n", "P35_obs", "P35_ar1", "P35_wn", "P25_obs", "P25_ar1", "P25_wn"])
t.to_csv(os.path.join(D, "edge_peak_rates.csv"), index=False)
print(t.round(3).to_string())
print("phi quartile P35 obs:", np.round(po, 3), "ar1:", np.round(pa, 3))
print("mean score 25 vs 35 (wn/ar/obs):",
      Z["n_wn"].mean((0, 1))[[0, -1]], Z["n_ar"].mean((0, 1))[[0, -1]], Z["n_obs"].mean(0)[[0, -1]])
