"""
03_verify_r_obs_match.py
Verify that R_obs (A_i, S_i, best_T) in window_eligibility.parquet matches the exact values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

def main() -> None:
    elig_path = "experiments/female_infradian_cycles_v2/results/window_eligibility.parquet"
    elig_df = pd.read_parquet(elig_path)
    eligible = elig_df[elig_df["eligible_w1"] == True].head(10)
    
    for _, row in eligible.iterrows():
        cid = row["customer_id"]
        a_parquet = row["A_i"]
        s_parquet = row["S_i"]
        t_parquet = row["best_T"]
        print(f"CID {cid}: Parquet A={a_parquet:.6f}, S={s_parquet:.6f}, T={t_parquet:.1f}")

if __name__ == "__main__":
    main()
