#!/usr/bin/env python
"""Plot daily predicted salutation probability for the arm-A user with the
largest (or smallest) variance across days (dayscale r=0 predictions).

Usage:
  python plot_top_variance_user.py [max|min]

Read-only w.r.t. the experiment; writes a PNG next to results/.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "day_predictions_r0.parquet"
ARM = "A"
MIN_DAYS = 30  # ignore tiny day pools

direction = (sys.argv[1] if len(sys.argv) > 1 else "max").lower()
assert direction in {"max", "min"}, "arg must be 'max' or 'min'"
OUT = HERE / "results" / f"user_{direction}_daily_prob_variance.png"

df = pd.read_parquet(PRED)
df = df[df.arm == ARM]

stats = df.groupby("user")["p"].agg(["count", "mean", "var", "std"])
stats = stats[stats["count"] >= MIN_DAYS]
user = int(stats["var"].idxmax() if direction == "max" else stats["var"].idxmin())

d = df[df.user == user].sort_values("date").copy()
d["dt"] = pd.to_datetime(d["date"], unit="D", origin="1970-01-01")
y = int(d["y"].iloc[0])
mean_p = d["p"].mean()
roll = d["p"].rolling(7, center=True, min_periods=3).mean()

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.scatter(d["dt"], d["p"], s=14, alpha=0.55, color="steelblue",
           label="daily predicted P(salutation=20)")
ax.plot(d["dt"], roll, color="darkorange", lw=2, label="7-day centered mean")
ax.axhline(mean_p, color="crimson", ls="--", lw=1.2,
           label=f"user mean = {mean_p:.3f}")
ax.axhline(0.5, color="grey", ls=":", lw=1, label="0.5")
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("date")
ax.set_ylabel("predicted P(salutation = 20)")
ax.set_title(
    f"Dayscale arm {ARM} — user {user} (label y={y}), "
    f"n={len(d)} days, var={d['p'].var():.4f}, std={d['p'].std():.3f}"
)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=150)

print(f"[{direction}] user={user} y={y} n={len(d)} var={d['p'].var():.6f} "
      f"std={d['p'].std():.6f} min={d['p'].min():.3f} max={d['p'].max():.3f}")
print(f"wrote {OUT}")
