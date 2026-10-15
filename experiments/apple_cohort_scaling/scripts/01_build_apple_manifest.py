"""Step 1: Build full candidate manifest for single-source Apple Watch users.

Reads experiments/artifacts_v2/cohort_manifest.parquet, filters for single-source
Apple Watch (source 6) participants with valid recorded binary salutation,
verifies physical presence of out/{user_id}.csv, and writes artifacts/apple_full_manifest.csv.
"""
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_DIR / "artifacts"
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

    # 4. Save manifest
    out_manifest = OUTPUT_DIR / "apple_full_manifest.csv"
    cols = ["customer_id", "user_id", "salutation", "y", "age_group", "bmi_group", "csv_path"]
    apple_df = apple_df.sort_values("user_id").reset_index(drop=True)
    apple_df[cols].to_csv(out_manifest, index=False)

    print(f"\nSaved {len(apple_df)} participants to {out_manifest}")
    print("\nCohort demographics summary:")
    print(f"Salutation 10 (F): {(apple_df['y'] == 0).sum()} ({(apple_df['y'] == 0).mean()*100:.1f}%)")
    print(f"Salutation 20 (M): {(apple_df['y'] == 1).sum()} ({(apple_df['y'] == 1).mean()*100:.1f}%)")
    print("\nAge group distribution:")
    print(apple_df["age_group"].value_counts())
    print("\nBMI group distribution:")
    print(apple_df["bmi_group"].value_counts())


if __name__ == "__main__":
    main()
