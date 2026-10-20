"""
01_inspect_cache.py
Inspect customer npz files for eligible participants, checking date mapping and stats.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

def main() -> None:
    elig_path = "experiments/female_infradian_cycles_v2/results/window_eligibility.parquet"
    elig_df = pd.read_parquet(elig_path)
    eligible = elig_df[elig_df["eligible_w1"] == True].copy()
    
    print(f"Total eligible: {len(eligible)}")
    sample_cids = eligible["customer_id"].head(5).tolist()
    
    for cid in sample_cids:
        cache_path = f"experiments/apple_cohort_scaling/artifacts/cache_5min/customer_{cid}.npz"
        if not os.path.exists(cache_path):
            print(f"Error: {cache_path} does not exist!")
            continue
        with np.load(cache_path) as npz:
            keys = list(npz.keys())
            dates = npz["date"]
            hr = npz["hr"]
            stats = npz["stats"] if "stats" in npz else None
            
        row = eligible[eligible["customer_id"] == cid].iloc[0]
        sel_dates = np.array(row["w1_selected_dates"])
        
        # Check date membership
        sel_in_cache = np.isin(sel_dates, dates)
        all_in = np.all(sel_in_cache)
        
        print(f"\nCustomer {cid}:")
        print(f"  npz keys: {keys}")
        print(f"  total cache days: {len(dates)}, hr shape: {hr.shape}")
        if stats is not None:
            print(f"  stats shape: {stats.shape}, dtype: {stats.dtype}")
        print(f"  selected 90 dates all in cache? {all_in}")
        
        # Check indices
        date_to_idx = {d: i for i, d in enumerate(dates)}
        indices = [date_to_idx[d] for d in sel_dates]
        hr_sel = hr[indices]
        valid_bins_per_day = np.isfinite(hr_sel).sum(axis=1)
        print(f"  min valid bins per selected day: {np.min(valid_bins_per_day)} (>= 120 required)")

if __name__ == "__main__":
    main()
