#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""08_plot_recalibration_check.py -- does the 2-component null reproduce the observed spectrum and edge peaks?"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../results"))
t = pd.read_parquet(os.path.join(R, "recalibration/null2_participants.parquet"))
x = np.load(os.path.join(R, "recalibration/null2_extras.npz"))
sp = np.load(os.path.join(R, "raw_periodogram/spectra.npz"), allow_pickle=True)
assert (sp["cid"] == x["cid"]).all() and (t.customer_id.values == x["cid"]).all()
wide, y, age = sp["wide"], t.y.values, t.age_group.values
ages = ["Young Adults", "Middle Ager", "Elderly"]

fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
rows = []
for j, a in enumerate(ages):
    for i, s in enumerate([0, 1]):
        m = (y == s) & (age == a)
        ax = axs[i, j]
        ar1 = sp["w_ar"][m].mean(0).mean(0)
        n2 = x["spec_null"][m].astype(float).mean(0)
        lo, hi = np.percentile(n2, [2.5, 97.5], axis=0)
        ax.fill_between(wide, lo, hi, color="green", alpha=.2)
        ax.plot(wide, n2.mean(0), c="green", label="AR(1)+slow null (mean, 95%)")
        ax.plot(wide, ar1, "k--", lw=1, label="AR(1) null (v2)")
        ax.plot(wide, sp["w_obs"][m].mean(0), c="#d62728" if s == 0 else "#1f77b4", lw=1.8, label="observed")
        ax.axvspan(25, 35, color="gold", alpha=.15)
        ax.set_xscale("log")
        ax.set_title(f"{'Female' if s == 0 else 'Male'} / {a} (n={m.sum()})", fontsize=10)
        ax.set_xlabel("period (d)")
        po = (t.argmax_obs[m] == 20).mean()
        pn = (x["argmax_null"][m] == 20).mean()
        rows.append(dict(sex="F" if s == 0 else "M", age=a, n=int(m.sum()), P35_obs=po, P35_null2=pn,
                         obs_over_30d_minus_null2=float(sp["w_obs"][m][:, wide > 40].mean() - n2.mean(0)[wide > 40].mean())))
axs[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(R, "recalibration/E_spectrum_two_component_null.png"), dpi=200)
print(pd.DataFrame(rows).round(3).to_string())
print("median s2 fraction by sex:", t.assign(f=t.s2 / (t.s1 + t.s2)).groupby("y").f.median().round(3).to_dict())
