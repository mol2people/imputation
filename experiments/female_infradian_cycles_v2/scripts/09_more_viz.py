#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""09_more_viz.py -- additional diagnostics from v2 results (p-value calibration, per-participant spectra heatmap,
window-to-window period persistence, example participants)."""
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
R = os.path.join(ROOT, "experiments/female_infradian_cycles_v2/results")
OUT = os.path.join(R, "extra_viz")
os.makedirs(OUT, exist_ok=True)
t = pd.read_parquet(os.path.join(R, "recalibration/null2_participants.parquet"))
sp = np.load(os.path.join(R, "raw_periodogram/spectra.npz"), allow_pickle=True)
assert (sp["cid"] == t.customer_id.values).all()
wide = sp["wide"]
ages = ["Young Adults", "Middle Ager", "Elderly"]
S = {0: "Female", 1: "Male"}
C = {0: "#d62728", 1: "#1f77b4"}


def save(f, n):
    f.tight_layout()
    f.savefig(os.path.join(OUT, n), dpi=200)
    plt.close(f)


# 1. p-value histograms (null calibration) under v2 and two-component null
fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
for j, a in enumerate(ages):
    for i, s in enumerate([0, 1]):
        m = (t.y == s) & (t.age_group == a)
        ax = axs[i, j]
        ax.hist(t.p_i[m], bins=20, range=(0, 1), alpha=.6, color=C[s], label="v2 AR(1)")
        ax.hist(t.p_i2[m], bins=20, range=(0, 1), histtype="step", color="k", lw=1.5, label="AR(1)+slow")
        ax.axhline(m.sum() / 20, ls="--", c="gray")
        ax.set_title(f"{S[s]} / {a} (n={m.sum()})", fontsize=10)
        ax.set_xlabel("empirical p")
axs[0, 0].legend(fontsize=8)
save(fig, "V1_pvalue_histograms.png")

# 2. QQ of -log10 p vs uniform
fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
for j, a in enumerate(ages):
    ax = axs[j]
    for s in [0, 1]:
        p = np.sort(t.p_i2[(t.y == s) & (t.age_group == a)].values)
        e = (np.arange(1, len(p) + 1) - .5) / len(p)
        ax.plot(-np.log10(e), -np.log10(p), c=C[s], label=S[s])
    ax.plot([0, 2.5], [0, 2.5], "k--", lw=1)
    ax.set_title(a)
    ax.set_xlabel("expected -log10 p")
axs[0].set_ylabel("observed -log10 p (AR(1)+slow null)")
axs[0].legend()
save(fig, "V2_pvalue_qq.png")

# 3. Per-participant spectra heatmaps (sorted by 25-35 d peak power), young F vs young M
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
for ax, s in zip(axs, [0, 1]):
    m = ((t.y == s) & (t.age_group == "Young Adults")).values
    X = sp["w_obs"][m]
    band = (wide >= 25) & (wide <= 35)
    X = X[np.argsort(-X[:, band].max(1))]
    im = ax.imshow(X, aspect="auto", origin="upper", extent=[np.log10(wide[0]), np.log10(wide[-1]), X.shape[0], 0],
                   vmax=np.percentile(X, 99), cmap="magma")
    ax.axvline(np.log10(25), c="w", lw=.6)
    ax.axvline(np.log10(35), c="w", lw=.6)
    ax.set_xlabel("log10 period (d)")
    ax.set_title(f"Young {S[s]} (n={m.sum()}), sorted by 25-35 d peak")
    plt.colorbar(im, ax=ax, label="partial R$^2$")
save(fig, "V3_spectra_heatmap_young.png")

# 4. Period persistence window1 -> window2
w = t[t.eligible_w2 & t.w2_best_T.notna()]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, a in zip(axs, ages):
    for s in [0, 1]:
        d = w[(w.y == s) & (w.age_group == a)]
        ax.scatter(d.best_T + np.random.default_rng(0).normal(0, .15, len(d)),
                   d.w2_best_T + np.random.default_rng(1).normal(0, .15, len(d)), s=6, alpha=.3, c=C[s], label=f"{S[s]} n={len(d)}")
    ax.plot([25, 35], [25, 35], "k--", lw=1)
    ax.set_title(a)
    ax.set_xlabel("window-1 best T (d)")
    ax.set_ylabel("window-2 best T (d)")
    ax.legend(markerscale=3, fontsize=8)
save(fig, "V4_period_persistence_scatter.png")

fig, ax = plt.subplots(figsize=(8, 4.5))
hi = w[w.E_i2 > w.E_i2.quantile(.0)]
for s in [0, 1]:
    for a, ls in zip(["Young Adults", "Middle Ager"], ["-", ":"]):
        x = np.sort(w.loc[(w.y == s) & (w.age_group == a), "delta_T_w1_w2"].values)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), c=C[s], ls=ls, label=f"{S[s]} {a}")
# random-pairing reference: |T1 - T2| for shuffled partners
rng = np.random.default_rng(0)
x = np.sort(np.abs(w.best_T.values - rng.permutation(w.w2_best_T.values)))
ax.plot(x, np.arange(1, len(x) + 1) / len(x), c="gray", ls="--", label="shuffled partners")
ax.set_xlabel("|T1 - T2| (d)")
ax.set_ylabel("ECDF")
ax.legend(fontsize=8)
save(fig, "V5_delta_period_ecdf_vs_shuffle.png")

# 5. E_i vs A_i and E_i vs phi (heteroscedasticity / dependence on noise memory)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for s in [1, 0]:
    d = t[(t.y == s) & (t.age_group == "Young Adults")]
    axs[0].scatter(d.A_i, d.E_i2, s=8, alpha=.4, c=C[s], label=S[s])
    axs[1].scatter(d.phi1, d.E_i2, s=8, alpha=.4, c=C[s], label=S[s])
axs[0].set_xlabel("amplitude A_i (bpm)")
axs[0].set_ylabel("E_i (new null)")
axs[1].set_xlabel("fitted fast-component phi1")
axs[0].legend()
axs[0].set_title("Young adults")
save(fig, "V6_E_vs_amplitude_and_phi.png")

# 6. Example participants: young females with highest E_i2 and median-E_i2 male; residual + fitted harmonic
V = importlib.util.module_from_spec(importlib.util.spec_from_file_location("v", os.path.join(HERE, "00_verify_v2_methods.py")))
V.__spec__.loader.exec_module(V)
P = importlib.util.module_from_spec(importlib.util.spec_from_file_location("p", os.path.join(HERE, "01_extract_and_calibrate_infradian.py")))
sys.modules["verify_v2_methods"] = V
P.__spec__.loader.exec_module(P)
el = pd.read_parquet(os.path.join(R, "window_eligibility.parquet")).set_index("customer_id")
yf = t[(t.y == 0) & (t.age_group == "Young Adults")].sort_values("E_i2", ascending=False)
ym = t[(t.y == 1) & (t.age_group == "Young Adults")].sort_values("E_i2", ascending=False)
picks = [("F top", yf.iloc[0]), ("F top-10", yf.iloc[9]), ("F median", yf.iloc[len(yf) // 2]), ("M top", ym.iloc[0]),
         ("M top-10", ym.iloc[9]), ("M median", ym.iloc[len(ym) // 2])]
fig, axs = plt.subplots(2, 3, figsize=(16, 7))
for ax, (lab, r) in zip(axs.ravel(), picks):
    cid = int(r.customer_id)
    with np.load(os.path.join(ROOT, f"experiments/apple_cohort_scaling/artifacts/cache_5min/customer_{cid}.npz")) as z:
        hr, dates = z["hr"], z["date"]
    w1 = np.asarray(el.loc[cid, "w1_selected_dates"], dtype=np.int64)
    p10, clock, _, _ = P.extract_features_on_dates(hr, dates, w1)
    _, _, proj = V.build_nonperiodic_baseline(w1, clock)
    e0 = proj @ p10
    tt = (w1 - w1[0]).astype(float)
    T = r.best_T
    Xh = proj @ np.column_stack([np.cos(2 * np.pi * tt / T), np.sin(2 * np.pi * tt / T)])
    b = np.linalg.lstsq(Xh, e0, rcond=None)[0]
    grid = np.linspace(0, tt.max(), 400)
    ax.plot(tt, e0, ".", c="gray")
    ax.plot(grid, b[0] * np.cos(2 * np.pi * grid / T) + b[1] * np.sin(2 * np.pi * grid / T), c=C[int(r.y)])
    ax.set_title(f"{lab}: cid {cid}  T={T:.1f} d  E={r.E_i2:.3f}  p={r.p_i2:.3f}", fontsize=9)
    ax.set_xlabel("day in window")
    ax.set_ylabel("baseline-adjusted hr_p10 (bpm)")
save(fig, "V7_example_participants.png")
print("done", sorted(os.listdir(OUT)))
