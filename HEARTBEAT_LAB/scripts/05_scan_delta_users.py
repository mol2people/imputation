"""Stream and prepare the 562 delta Garmin users from raw out/ CSVs into 288-bin NPZs.

These users passed v2 single-source eligibility (>=14 adequate days) but failed the
strict 13-week continuous window gate (due to fragmented wear or span <91 days).
"""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "HEARTBEAT_LAB" / "src"))
from heartbeat_lab import prepare_days
from heartbeat_lab.data import STATS

MANIFEST_PATH = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "relaxed_garmin_manifest.csv"
OUTPUT_DIR = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "relaxed_cache" / "delta_562"


def process_single_customer(task):
    idx, cid, sal, y, fail_reason, csv_rel = task
    csv_file = REPO_ROOT / csv_rel
    filename = f"customer_{idx:06d}.npz"
    target_npz = OUTPUT_DIR / filename
    
    try:
        if not csv_file.exists():
            return None, {"customer_id": cid, "reason": f"File not found: {csv_file}"}
        
        df = pd.read_csv(csv_file)
        days, hr, quality = prepare_days(df)
        if len(days) < 2:
            return None, {"customer_id": cid, "reason": f"Fewer than 2 adequate days: {len(days)}"}
            
        np.savez_compressed(
            target_npz,
            hr=hr,
            date=days.date.to_numpy(dtype="datetime64[D]").astype("int64"),
            stats=days[STATS].to_numpy(np.float32),
        )
        
        meta = {
            "customer_id": cid,
            "y": int(y),
            "salutation": int(sal),
            "file": filename,
            "adequate_days": len(days),
            "fail_reason": fail_reason,
            "input_rows": quality.get("input_rows", 0),
            "retained_rows": quality.get("retained_rows", 0),
            "observed_days": quality.get("observed_days", 0),
        }
        return meta, None
    except Exception as e:
        return None, {"customer_id": cid, "reason": str(e)}


def main():
    print(f"Loading manifest from {MANIFEST_PATH}...")
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"customer_id": str})
    delta_df = manifest[manifest["delta_1c_0"] == True].copy().reset_index(drop=True)
    print(f"Found {len(delta_df)} delta users to scan.")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    for idx, row in delta_df.iterrows():
        tasks.append((
            idx,
            str(row["customer_id"]),
            int(row["salutation"]),
            int(row["y"]),
            str(row["fail_reason"]),
            str(row["csv_path"])
        ))
        
    print(f"Scanning raw epoch files with ProcessPoolExecutor (4 workers)...")
    included, excluded = [], []
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        for meta, err in executor.map(process_single_customer, tasks):
            if meta is not None:
                included.append(meta)
            else:
                excluded.append(err)
                
    inc_df = pd.DataFrame(included)
    exc_df = pd.DataFrame(excluded)
    
    inc_df.to_csv(OUTPUT_DIR / "customers.csv", index=False)
    exc_df.to_csv(OUTPUT_DIR / "exclusions.csv", index=False)
    
    print(f"\nProcessing complete!")
    print(f"Successfully prepared: {len(inc_df)} / {len(delta_df)} customers.")
    if len(exc_df) > 0:
        print(f"Excluded: {len(exc_df)} customers (see {OUTPUT_DIR / 'exclusions.csv'}).")
        print("Exclusion reasons:\n", exc_df["reason"].value_counts())
        
    print(f"\nAdequate days distribution for prepared delta users:")
    print(inc_df["adequate_days"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    print(f"Class breakdown: sal10 = {(inc_df['y'] == 0).sum()}, sal20 = {(inc_df['y'] == 1).sum()}")


if __name__ == "__main__":
    main()
