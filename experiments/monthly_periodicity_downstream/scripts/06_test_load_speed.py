"""
06_test_load_speed.py
Test loading speed of customer npz files using ThreadPoolExecutor.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

def load_user(row: dict) -> dict:
    cid = row["customer_id"]
    sel_dates = np.array(row["w1_selected_dates"])
    cache_path = f"experiments/apple_cohort_scaling/artifacts/cache_5min/customer_{cid}.npz"
    with np.load(cache_path) as npz:
        dates = npz["date"]
        hr = npz["hr"]
        stats = npz["stats"] if "stats" in npz else None
        
    date_to_idx = {d: i for i, d in enumerate(dates)}
    indices = [date_to_idx[d] for d in sel_dates]
    
    hr_sel = hr[indices] # (90, 288)
    stats_sel = stats[indices] if stats is not None else None # (90, 8)
    
    return {
        "customer_id": cid,
        "hr_sel": hr_sel,
        "stats_sel": stats_sel,
        "dates_sel": sel_dates,
    }

def main() -> None:
    elig_path = "experiments/female_infradian_cycles_v2/results/window_eligibility.parquet"
    elig_df = pd.read_parquet(elig_path)
    eligible = elig_df[elig_df["eligible_w1"] == True].head(200).to_dict("records")
    
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(load_user, eligible))
    elapsed = time.time() - t0
    print(f"Loaded {len(results)} users in {elapsed:.2f}s ({elapsed/len(results)*1000:.1f} ms/user)")
    est_total = (elapsed / len(results)) * 4631
    print(f"Estimated time for 4,631 users: {est_total:.1f}s")

if __name__ == "__main__":
    main()
