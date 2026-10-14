"""Step 1: Stratified sampling of N=1,000 single-source eligible Apple Watch users.

Reads experiments/artifacts_v2/cohort_manifest.parquet, filters for single-source
Apple Watch (source 6) users with valid binary recorded salutation, verifies raw CSV
presence in out/{user_id}.csv, and draws a stratified sample of 1,000 participants.
"""
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
OUT_CSV_DIR = REPO_ROOT / "out"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = REPO_ROOT / "experiments/artifacts_v2/cohort_manifest.parquet"
    print(f"Loading cohort manifest from {manifest_path}...")
    cm = pd.read_parquet(manifest_path)
    
    # 1. Filter to single-source eligible Apple users with binary salutation
    apple_mask = (
        (cm["epoch_src_6"] == 1) &
        (cm["epoch_multisource"] == False) &
        (cm["cohort_category"] == "eligible") &
        (cm["y"].isin([0.0, 1.0]))
    )
    apple_df = cm[apple_mask].copy()
    print(f"Total eligible single-source Apple users: {len(apple_df)}")
    
    # 2. Verify raw CSV existence
    apple_df["user_id"] = apple_df["user_id"].astype(int)
    apple_df["csv_path"] = apple_df["user_id"].apply(lambda uid: f"out/{uid}.csv")
    exists_mask = apple_df["user_id"].apply(lambda uid: (OUT_CSV_DIR / f"{uid}.csv").exists())
    apple_df = apple_df[exists_mask].copy()
    print(f"Users with existing raw CSV in out/: {len(apple_df)}")
    
    # 3. Clean demographic covariates
    apple_df["customer_id"] = apple_df["user_id"].astype(str)
    apple_df["y"] = apple_df["y"].astype(int)
    apple_df["salutation"] = apple_df["y"].map({0: 10, 1: 20}).astype(int)
    apple_df["age_group"] = apple_df["age_group"].fillna("Unknown").astype(str)
    apple_df["bmi_group"] = apple_df["bmi_grp"].fillna("Unknown").astype(str)
    
    # Stratum key for balanced sampling
    apple_df["stratum"] = apple_df["y"].astype(str) + "__" + apple_df["age_group"]
    
    # 4. Stratified random sample of N = 1,000
    N_PILOT = 1000
    SEED = 20261014
    rng = np.random.default_rng(SEED)
    
    # Sample proportionally across strata
    sample_indices = []
    strata_counts = apple_df["stratum"].value_counts()
    for stratum, count in strata_counts.items():
        stratum_indices = apple_df[apple_df["stratum"] == stratum].index.tolist()
        stratum_n = int(np.round(N_PILOT * count / len(apple_df)))
        if stratum_n > 0:
            picked = rng.choice(stratum_indices, min(stratum_n, len(stratum_indices)), replace=False)
            sample_indices.extend(picked)
            
    # Adjust to exactly N_PILOT if rounding caused slight deviation
    if len(sample_indices) < N_PILOT:
        remaining = list(set(apple_df.index) - set(sample_indices))
        extra = rng.choice(remaining, N_PILOT - len(sample_indices), replace=False)
        sample_indices.extend(extra)
    elif len(sample_indices) > N_PILOT:
        sample_indices = list(rng.choice(sample_indices, N_PILOT, replace=False))
        
    pilot_df = apple_df.loc[sample_indices].copy().sort_values("user_id").reset_index(drop=True)
    
    # 5. Save manifest
    out_manifest = OUTPUT_DIR / "pilot_manifest.csv"
    cols = ["customer_id", "user_id", "salutation", "y", "age_group", "bmi_group", "csv_path"]
    pilot_df[cols].to_csv(out_manifest, index=False)
    
    print(f"\nSuccessfully sampled {len(pilot_df)} participants into {out_manifest}")
    print("\nPilot demographics summary:")
    print(f"Salutation 10 (F): {(pilot_df['y'] == 0).sum()} ({(pilot_df['y'] == 0).mean()*100:.1f}%)")
    print(f"Salutation 20 (M): {(pilot_df['y'] == 1).sum()} ({(pilot_df['y'] == 1).mean()*100:.1f}%)")
    print("\nAge group distribution:")
    print(pilot_df["age_group"].value_counts())
    print("\nBMI group distribution:")
    print(pilot_df["bmi_group"].value_counts())


if __name__ == "__main__":
    main()
