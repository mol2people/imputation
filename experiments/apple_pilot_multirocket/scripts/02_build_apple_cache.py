"""Step 2: Parse raw CSVs and generate 5-min (288-bin) and 15-min (96-bin) caches.

Reads out/{user_id}.csv for all 1,000 pilot participants, validates source-6 / type-3000
heart rate events, aggregates adequate days (>=8 hours AND >=60 readings), and saves
compressed .npz arrays to cache_5min/ and cache_15min/.
"""
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PILOT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PILOT_DIR / "artifacts" / "pilot_manifest.csv"
CACHE_5MIN_DIR = PILOT_DIR / "artifacts" / "cache_5min"
CACHE_15MIN_DIR = PILOT_DIR / "artifacts" / "cache_15min"

STATS = ["mean", "median", "sd", "vmin", "vmax", "n", "cov_h", "hours"]


def union_hours(start, end):
    """Interval union in hours assigned to start dates."""
    if len(start) == 0:
        return 0.0
    order = np.argsort(start, kind="stable")
    s, e = start[order], end[order]
    cum = np.maximum.accumulate(e)
    first = np.r_[0, np.flatnonzero(s[1:] > cum[:-1]) + 1]
    last = np.r_[first[1:] - 1, len(s) - 1]
    return float(np.maximum(0, cum[last] - s[first]).sum() / 3_600_000)


def process_single_customer(row_tuple):
    cid, csv_rel_path = row_tuple
    csv_path = REPO_ROOT / csv_rel_path
    
    if not csv_path.exists():
        return None
        
    try:
        # Read only required columns
        cols = ["startTimestamp", "endTimestamp", "timezoneOffset", "longValue", "source", "type"]
        df = pd.read_csv(csv_path, usecols=cols)
        
        # Filter source 6 (Apple), type 3000 (HR), valid range [25, 230]
        valid = (
            (df["source"] == 6) &
            (df["type"] == 3000) &
            (df["longValue"].between(25, 230)) &
            (df["startTimestamp"] > 0) &
            np.isfinite(df["startTimestamp"]) &
            np.isfinite(df["timezoneOffset"]) &
            np.isfinite(df["longValue"])
        )
        sub = df.loc[valid].copy()
        if len(sub) == 0:
            return None
            
        start = sub["startTimestamp"].to_numpy(np.int64)
        tz_offset = sub["timezoneOffset"].to_numpy(np.float64)
        local_ms = start + (tz_offset * 60_000).astype(np.int64)
        
        # Local timestamps and day indices
        local_dates = pd.to_datetime(local_ms, unit="ms")
        sub["date"] = local_dates.normalize()
        sub["ms_in_day"] = local_ms % 86_400_000
        sub["bin_5m"] = sub["ms_in_day"] // 300_000    # 288 bins
        sub["bin_15m"] = sub["ms_in_day"] // 900_000   # 96 bins
        sub["hour"] = sub["bin_5m"] // 12
        
        end = sub["endTimestamp"].to_numpy()
        sub["endTimestamp"] = np.where(np.isfinite(end) & (end >= start), end, start)
        
        # Day-level grouping
        dates_list = []
        stats_list = []
        bins_5m_list = []
        bins_15m_list = []
        
        for date, day in sub.groupby("date", sort=True):
            n_events = len(day)
            hours_covered = day["hour"].nunique()
            # Apple day adequacy: >=8 clock hours covered AND >=60 readings
            if hours_covered < 8 or n_events < 60:
                continue
                
            vals = day["longValue"].to_numpy()
            cov_h = union_hours(day["startTimestamp"].to_numpy(), day["endTimestamp"].to_numpy())
            
            # 5-minute curve (288 bins)
            curve_5m = day.groupby("bin_5m")["longValue"].mean().reindex(range(288)).to_numpy(np.float32)
            # 15-minute curve (96 bins)
            curve_15m = day.groupby("bin_15m")["longValue"].mean().reindex(range(96)).to_numpy(np.float32)
            
            dates_list.append(np.datetime64(date, "D"))
            stats_list.append([
                vals.mean(),
                np.median(vals),
                vals.std(ddof=1) if len(vals) > 1 else 0.0,
                vals.min(),
                vals.max(),
                n_events,
                cov_h,
                hours_covered,
            ])
            bins_5m_list.append(curve_5m)
            bins_15m_list.append(curve_15m)
            
        n_adequate = len(dates_list)
        if n_adequate < 14:
            return None
            
        dates_arr = np.array(dates_list, dtype="datetime64[D]")
        stats_arr = np.array(stats_list, dtype=np.float32)
        hr_5m_arr = np.stack(bins_5m_list).astype(np.float32)
        hr_15m_arr = np.stack(bins_15m_list).astype(np.float32)
        
        # Save 5-min cache
        fname_5m = f"customer_{cid}.npz"
        np.savez_compressed(
            CACHE_5MIN_DIR / fname_5m,
            hr=hr_5m_arr,
            date=dates_arr.view("int64"),
            stats=stats_arr,
        )
        
        # Save 15-min cache
        fname_15m = f"customer_{cid}.npz"
        np.savez_compressed(
            CACHE_15MIN_DIR / fname_15m,
            hr=hr_15m_arr,
            date=dates_arr.view("int64"),
            stats=stats_arr,
        )
        
        return {
            "customer_id": str(cid),
            "file": fname_5m,
            "n_days": n_adequate,
            "mean_finite_bins_5m": float(np.isfinite(hr_5m_arr).sum(axis=1).mean()),
            "occupancy_rate_5m": float(np.isfinite(hr_5m_arr).mean()),
            "mean_finite_bins_15m": float(np.isfinite(hr_15m_arr).sum(axis=1).mean()),
            "occupancy_rate_15m": float(np.isfinite(hr_15m_arr).mean()),
        }
    except Exception as e:
        print(f"Error processing customer {cid}: {e}", file=sys.stderr)
        return None


def main():
    CACHE_5MIN_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_15MIN_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading pilot manifest from {MANIFEST_PATH}...")
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"customer_id": str})
    tasks = list(zip(manifest["customer_id"], manifest["csv_path"]))
    
    print(f"Processing {len(tasks)} customers across 6 worker processes...")
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(process_single_customer, tasks))
        
    valid_results = [r for r in results if r is not None]
    dt = time.time() - t0
    print(f"Completed processing {len(tasks)} customers in {dt:.1f}s.")
    print(f"Successfully qualified customers with >=14 adequate days: {len(valid_results)} / {len(tasks)} ({len(valid_results)/len(tasks)*100:.1f}%)")
    
    res_df = pd.DataFrame(valid_results)
    res_df.to_csv(CACHE_5MIN_DIR / "customers.csv", index=False)
    res_df.to_csv(CACHE_15MIN_DIR / "customers.csv", index=False)
    
    print("\nData Occupancy Summary across Qualified Apple Cohort:")
    print(f"Mean adequate days per user: {res_df['n_days'].mean():.1f} (median: {res_df['n_days'].median():.1f}, min: {res_df['n_days'].min()}, max: {res_df['n_days'].max()})")
    print(f"5-Minute Bins (288):  {res_df['mean_finite_bins_5m'].mean():.1f} / 288 bins occupied ({res_df['occupancy_rate_5m'].mean()*100:.1f}%)")
    print(f"15-Minute Bins (96):   {res_df['mean_finite_bins_15m'].mean():.1f} / 96 bins occupied ({res_df['occupancy_rate_15m'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
