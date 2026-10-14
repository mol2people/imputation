"""Build relaxed Garmin cohort manifest with tier assignments and covariates.

Tiers:
  - tier_0: Strict 13-week window, single-source Garmin (N=3,848)
  - tier_1c: Single-source Garmin with >=14 adequate days (N=4,410)
  - delta_1c_0: The 562 single-source Garmin users who failed tier_0
  - tier_2: Primary Garmin from ungated core-HR (N=5,636)
  - tier_3: Any Garmin core-HR user with >=14 adequate days (N=5,678)
"""
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts"


def main():
    print("Loading upstream manifests...")
    cm = pd.read_parquet(REPO_ROOT / "experiments/artifacts_v2/cohort_manifest.parquet")
    el = pd.read_parquet(REPO_ROOT / "experiments/artifacts_sq/eligibility_by_channel.parquet")
    
    # 1. Filter to Garmin core users (epoch_src_3 == 1)
    garmin_cm = cm[cm["epoch_src_3"] == 1].copy()
    
    # Restrict to valid binary salutation (y in {0.0, 1.0})
    garmin_cm = garmin_cm[garmin_cm["y"].isin([0.0, 1.0])].copy()
    garmin_cm["user_id"] = garmin_cm["user_id"].astype(int)
    garmin_cm["salutation"] = garmin_cm["y"].map({0.0: 10, 1.0: 20}).astype(int)
    
    # 2. Get SQ eligibility for channel 3000
    el_ch3000 = el[(el["source_id"] == 3) & (el["channel"] == 3000)].copy()
    el_ch3000["user_id"] = el_ch3000["user_id"].astype(int)
    el_ch3000 = el_ch3000.set_index("user_id")
    
    # Also get SQ any-core pass for source 3 (Tier 0 reference)
    el_s3 = el[el["source_id"] == 3]
    tier_0_users = set(el_s3[el_s3["pass"] == True]["user_id"])
    print(f"Tier 0 users in eligibility table: {len(tier_0_users)}")
    
    # 3. Demographic covariates from cohort_manifest and ortools manifest
    ortools_manifest_path = REPO_ROOT / "experiments/2026-09-21_ortools_source_50_50_salutation_rf/output/split_manifest.csv"
    if ortools_manifest_path.exists():
        ortools_m = pd.read_csv(ortools_manifest_path)
        g_ortools = ortools_m[ortools_m["source_id"] == 3].set_index("user_id")
        g_ortools_users = set(g_ortools.index.astype(int))
    else:
        g_ortools = pd.DataFrame()
        g_ortools_users = set()
    print(f"Tier 2 primary Garmin users in OR-Tools manifest: {len(g_ortools_users)}")
    
    records = []
    for _, row in garmin_cm.iterrows():
        uid = int(row["user_id"])
        is_single_source = not bool(row["epoch_multisource"])
        is_v2_eligible = row["cohort_category"] == "eligible"
        
        # Channel 3000 info if available
        if uid in el_ch3000.index:
            ch_info = el_ch3000.loc[uid]
            n_day_rows = int(ch_info["n_day_rows"])
            fail_reason = str(ch_info["reason"]) if not ch_info["pass"] else "pass"
            sq_ch3000_pass = bool(ch_info["pass"])
        else:
            n_day_rows = int(row["n_days"]) if pd.notna(row["n_days"]) else 0
            fail_reason = "not_in_sq_ch3000"
            sq_ch3000_pass = False
            
        is_tier_0 = uid in tier_0_users
        is_tier_1c = is_single_source and is_v2_eligible
        is_delta_1c_0 = is_tier_1c and (not is_tier_0)
        is_tier_2 = uid in g_ortools_users
        is_tier_3 = is_v2_eligible
        
        rec = {
            "customer_id": str(uid),
            "user_id": uid,
            "salutation": row["salutation"],
            "y": int(row["y"]),
            "is_single_source": is_single_source,
            "n_day_rows": n_day_rows,
            "fail_reason": fail_reason,
            "tier_0": is_tier_0,
            "tier_1c": is_tier_1c,
            "delta_1c_0": is_delta_1c_0,
            "tier_2": is_tier_2,
            "tier_3": is_tier_3,
            "csv_path": f"out/{uid}.csv",
        }
        
        # Demographics from cohort_manifest
        rec["age_group"] = str(row["age_group"]) if pd.notna(row["age_group"]) else "Unknown"
        rec["bmi_group"] = str(row["bmi_grp"]) if pd.notna(row["bmi_grp"]) else "Unknown"
        if uid in g_ortools.index:
            rec["age_band_5y"] = str(g_ortools.loc[uid, "age_band_5y"])
        else:
            rec["age_band_5y"] = "Unknown"
            
        records.append(rec)
        
    df = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "relaxed_garmin_manifest.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved manifest to {out_csv} ({len(df)} total participants).")
    
    print("\nSummary of participant counts:")
    print(f"Tier 0 (Strict window baseline):       {df['tier_0'].sum():5d} (sal10: {(df['tier_0'] & (df['y']==0)).sum()}, sal20: {(df['tier_0'] & (df['y']==1)).sum()})")
    print(f"Tier 1c (Single-source v2-eligible):    {df['tier_1c'].sum():5d} (sal10: {(df['tier_1c'] & (df['y']==0)).sum()}, sal20: {(df['tier_1c'] & (df['y']==1)).sum()})")
    print(f"  -> Delta 1c - 0 (Wear-fail users):    {df['delta_1c_0'].sum():5d} (sal10: {(df['delta_1c_0'] & (df['y']==0)).sum()}, sal20: {(df['delta_1c_0'] & (df['y']==1)).sum()})")
    print(f"Tier 2 (Primary-source Garmin):        {df['tier_2'].sum():5d} (sal10: {(df['tier_2'] & (df['y']==0)).sum()}, sal20: {(df['tier_2'] & (df['y']==1)).sum()})")
    print(f"Tier 3 (All Garmin >=14d core):        {df['tier_3'].sum():5d} (sal10: {(df['tier_3'] & (df['y']==0)).sum()}, sal20: {(df['tier_3'] & (df['y']==1)).sum()})")
    print(f"Tier 4 (All core Garmin archive):      {len(df):5d} (sal10: {(df['y']==0).sum()}, sal20: {(df['y']==1).sum()})")


if __name__ == "__main__":
    main()
