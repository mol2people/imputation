"""
00_inspect_inputs.py
Inspect input parquet schemas, participant counts across alloc_0 splits,
and verify compatibility with the experimental protocol.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def main() -> None:
    print("Python executable:", sys.executable)
    
    # 1. Inspect apple_10_allocations.parquet
    alloc_path = "experiments/apple_cohort_scaling/artifacts/apple_10_allocations.parquet"
    alloc_df = pd.read_parquet(alloc_path)
    print(f"\n--- Apple Allocations ({alloc_path}) ---")
    print("Shape:", alloc_df.shape)
    print("Columns:", list(alloc_df.columns))
    print("alloc_0 value counts:\n", alloc_df["alloc_0"].value_counts(dropna=False).sort_index())
    
    # 2. Inspect window_eligibility.parquet
    elig_path = "experiments/female_infradian_cycles_v2/results/window_eligibility.parquet"
    elig_df = pd.read_parquet(elig_path)
    print(f"\n--- Window Eligibility ({elig_path}) ---")
    print("Shape:", elig_df.shape)
    print("Columns:", list(elig_df.columns))
    print("eligible_w1 value counts:\n", elig_df["eligible_w1"].value_counts(dropna=False))
    
    alloc_df["customer_id"] = alloc_df["customer_id"].astype(str)
    elig_df["customer_id"] = elig_df["customer_id"].astype(str)
    merged = pd.merge(alloc_df, elig_df[["customer_id", "eligible_w1", "w1_start_date", "w1_actual_span", "w1_longest_gap", "w1_selected_dates", "S_i", "best_T", "A_i"]], on="customer_id", how="inner")
    print("\n--- Merged Data ---")
    print("Total merged:", len(merged))
    
    eligible = merged[merged["eligible_w1"] == True].copy()
    print("Eligible w1 count:", len(eligible))
    print("Eligible by alloc_0:\n", eligible["alloc_0"].value_counts(dropna=False).sort_index())
    print("Eligible by age_group:\n", eligible["age_group"].value_counts(dropna=False))
    print("Eligible by age_group x salutation:\n", pd.crosstab(eligible["age_group"], eligible["salutation"]))
    print("Eligible by alloc_0 x salutation:\n", pd.crosstab(eligible["alloc_0"], eligible["salutation"]))
    
    # Check sample w1_selected_dates
    sample_dates = eligible["w1_selected_dates"].iloc[0]
    print(f"Sample w1_selected_dates (len={len(sample_dates)}):", sample_dates[:10], "...", sample_dates[-5:])
    
    # Check R_obs (A_i, S_i, best_T)
    print("\nR_obs statistics on eligible participants:")
    print("A_i (amplitude): finite count =", np.isfinite(eligible["A_i"]).sum(), "mean =", eligible["A_i"].mean(), "min =", eligible["A_i"].min(), "max =", eligible["A_i"].max())
    print("S_i (score): finite count =", np.isfinite(eligible["S_i"]).sum(), "mean =", eligible["S_i"].mean(), "min =", eligible["S_i"].min(), "max =", eligible["S_i"].max())
    print("best_T (period): finite count =", np.isfinite(eligible["best_T"]).sum(), "value counts:\n", eligible["best_T"].value_counts(dropna=False).sort_index())

if __name__ == "__main__":
    main()
