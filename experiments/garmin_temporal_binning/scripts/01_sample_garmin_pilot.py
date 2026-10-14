"""Step 1: Sample N=1,000 single-source Garmin participants for the temporal binning ablation.

Reads HEARTBEAT_LAB/artifacts/relaxed_garmin_manifest.csv (all 4,410 Tier 1c users),
stratifies on salutation (y) and age_group, and draws a representative sample of 1,000 users.
"""
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PILOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PILOT_DIR / "artifacts"
MANIFEST_PATH = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "relaxed_garmin_manifest.csv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading Garmin relaxed manifest from {MANIFEST_PATH}...")
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"customer_id": str})
    
    t1c = manifest[manifest["tier_1c"] == True].copy()
    print(f"Total Tier 1c single-source Garmin users: {len(t1c)}")
    
    # Stratum key for stratified sampling
    t1c["stratum"] = t1c["y"].astype(str) + "__" + t1c["age_group"].astype(str)
    
    N_SAMPLE = 1000
    SEED = 20261014
    rng = np.random.default_rng(SEED)
    
    sample_indices = []
    strata_counts = t1c["stratum"].value_counts()
    for stratum, count in strata_counts.items():
        stratum_indices = t1c[t1c["stratum"] == stratum].index.tolist()
        stratum_n = int(np.round(N_SAMPLE * count / len(t1c)))
        if stratum_n > 0:
            picked = rng.choice(stratum_indices, min(stratum_n, len(stratum_indices)), replace=False)
            sample_indices.extend(picked)
            
    if len(sample_indices) < N_SAMPLE:
        remaining = list(set(t1c.index) - set(sample_indices))
        extra = rng.choice(remaining, N_SAMPLE - len(sample_indices), replace=False)
        sample_indices.extend(extra)
    elif len(sample_indices) > N_SAMPLE:
        sample_indices = list(rng.choice(sample_indices, N_SAMPLE, replace=False))
        
    sample_df = t1c.loc[sample_indices].copy().sort_values("user_id").reset_index(drop=True)
    out_csv = OUTPUT_DIR / "garmin_binning_manifest.csv"
    sample_df.to_csv(out_csv, index=False)
    
    print(f"\nSaved {len(sample_df)} sampled Garmin participants to {out_csv}")
    print("\nSample demographics summary:")
    print(f"Salutation 10 (F): {(sample_df['y'] == 0).sum()} ({(sample_df['y'] == 0).mean()*100:.1f}%)")
    print(f"Salutation 20 (M): {(sample_df['y'] == 1).sum()} ({(sample_df['y'] == 1).mean()*100:.1f}%)")
    print(f"Tier 0 (Strict window pass): {sample_df['tier_0'].sum()} ({sample_df['tier_0'].mean()*100:.1f}%)")
    print(f"Delta 1c - 0 (Wear-fail users): {sample_df['delta_1c_0'].sum()} ({sample_df['delta_1c_0'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
