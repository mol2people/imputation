"""Reuse the existing adequate-day cache, preserving its selected study cohort.

Alternative to 01_prepare.py: avoids rescanning raw epoch files. Does not
rerun the upstream cohort gates. Each exported cache is private/gitignored.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from heartbeat_lab.data import STATS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--days", type=int, default=0, help="0 = all days; positive = first N (different experiment)")
    args = parser.parse_args()
    if args.days < 0 or args.days == 1:
        raise ValueError("Use 0 for all history or at least 2 days.")
    repo = args.repository
    cache = repo / "experiments/temporal_garmin/cache/bins_all"
    days = pd.read_parquet(repo / "experiments/r1b_dateaware_garmin_2026-09-21/cache/dayall_table.parquet")
    days = days.sort_values(["user", "date"], kind="stable").reset_index(drop=True)
    hr = np.load(cache / "hr_all.npy", mmap_mode="r")
    mask = np.load(cache / "mask_all.npy", mmap_mode="r")
    if not (np.array_equal(days.user, np.load(cache / "user_all.npy"))
            and np.array_equal(days.date, np.load(cache / "date_all.npy"))
            and hr.shape == mask.shape == (len(days), 288)):
        raise ValueError("Day cache does not align with the bin cache.")
    labels = pd.read_parquet(repo / "experiments/artifacts_sq/split_manifest_sq.parquet")
    labels = labels[labels.source_id.eq(3)].set_index("user_id").y
    if not labels.index.is_unique or not set(days.user).issubset(labels.index):
        raise ValueError("Customer labels do not align with the day cache.")
    args.output.mkdir(parents=True, exist_ok=False)
    customers = []
    for index, (user, group) in enumerate(days.groupby("user", sort=True)):
        if args.days:
            group = group.head(args.days)
        indices = group.index.to_numpy()
        filename = f"customer_{index:06d}.npz"
        np.savez_compressed(args.output / filename,
                            hr=np.where(mask[indices], hr[indices], np.nan).astype(np.float32),
                            date=group.date.to_numpy(np.int64),
                            stats=group[["d_ch3000_" + c for c in STATS]].to_numpy(np.float32))
        customers.append(dict(customer_id=str(user), y=int(labels.loc[user]), file=filename,
                              adequate_days=len(group)))
    pd.DataFrame(customers).to_csv(args.output / "customers.csv", index=False)
    print(f"Prepared {len(customers)} customers using {'all' if not args.days else args.days} days per customer.")


if __name__ == "__main__":
    main()
