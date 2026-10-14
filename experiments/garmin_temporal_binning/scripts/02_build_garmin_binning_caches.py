"""Step 2: Parse raw CSVs and generate 1-min, 5-min, and 15-min caches for Garmin.

Reads out/{user_id}.csv for 1,000 sampled Garmin participants across 8 worker processes,
filters source-3 / type-3000 heart rate events, applies the standard day adequacy gate,
and exports compressed .npz arrays to:
  - cache_1min/ (1,440 bins/day)
  - cache_5min/ (288 bins/day)
  - cache_15min/ (96 bins/day)
"""
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PILOT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PILOT_DIR / "artifacts" / "garmin_binning_manifest.csv"

CACHE_1MIN_DIR = PILOT_DIR / "artifacts" / "cache_1min"
CACHE_5MIN_DIR = PILOT_DIR / "artifacts" / "cache_5min"
CACHE_15MIN_DIR = PILOT_DIR / "artifacts" / "cache_15min"

STATS = ["mean", "median", "sd", "vmin", "vmax", "n", "cov_h", "hours"]


def union_hours(start, end):
    """Interval union in hours assigned to start date."""
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
        cols = ["startTimestamp", "endTimestamp", "timezoneOffset", "longValue", "source", "type"]
        df = pd.read_csv(csv_path, usecols=cols)
        
        valid = (
            (df["source"] == 3) &
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
        
        local_dates = pd.to_datetime(local_ms, unit="ms")
        sub["date"] = local_dates.normalize()
        sub["ms_in_day"] = local_ms % 86_400_000
        
        # Bins for all three resolutions
        sub["bin_1m"] = sub["ms_in_day"] // 60_000     # 1,440 bins
        sub["bin_5m"] = sub["ms_in_day"] // 300_000    # 288 bins
        sub["bin_15m"] = sub["ms_in_day"] // 900_000   # 96 bins
        sub["hour"] = sub["bin_5m"] // 12
        
        end = sub["endTimestamp"].to_numpy()
        sub["endTimestamp"] = np.where(np.isfinite(end) & (end >= start), end, start)
        
        dates_list = []
        stats_list = []
        bins_1m_list = []
        bins_5m_list = []
        bins_15m_list = []
        
        for date, day in sub.groupby("date", sort=True):
            n_events = len(day)
            hours_covered = day["hour"].nunique()
            cov_h = union_hours(day["startTimestamp"].to_numpy(), day["endTimestamp"].to_numpy())
            
            # Standard gate: >=8 clock-hours AND (>=8h coverage OR >=60 readings)
            if hours_covered < 8 or (cov_h < 8 and n_events < 60):
                continue
                
            vals = day["longValue"].to_numpy()
            
            # 1-min curve (1440 bins)
            curve_1m = day.groupby("bin_1m")["longValue"].mean().reindex(range(1440)).to_numpy(np.float32)
            # 5-min curve (288 bins)
            curve_5m = day.groupby("bin_5m")["longValue"].mean().reindex(range(288)).to_numpy(np.float32)
            # 15-min curve (96 bins)
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
            bins_1m_list.append(curve_1m)
            bins_5m_list.append(curve_5m)
            bins_15m_list.append(curve_15m)
            
        n_adequate = len(dates_list)
        if n_adequate < 14:
            return None
            
        dates_arr = np.array(dates_list, dtype="datetime64[D]")
        stats_arr = np.array(stats_list, dtype=np.float32)
        hr_1m_arr = np.stack(bins_1m_list).astype(np.float32)
        hr_5m_arr = np.stack(bins_5m_list).astype(np.float32)
        hr_15m_arr = np.stack(bins_15m_list).astype(np.float32)
        
        fname = f"customer_{cid}.npz"
        
        # Save 1-min cache
        np.savez_compressed(
            CACHE_1MIN_DIR / fname,
            hr=hr_1m_arr,
            date=dates_arr.view("int64"),
            stats=stats_arr,
        )
        # Save 5-min cache
        np.savez_compressed(
            CACHE_5MIN_DIR / fname,
            hr=hr_5m_arr,
            date=dates_arr.view("int64"),
            stats=stats_arr,
        )
        # Save 15-min cache
        np.savez_compressed(
            CACHE_15MIN_DIR / fname,
            hr=hr_15m_arr,
            date=dates_arr.view("int64"),
            stats=stats_arr,
        )
        
        return {
            "customer_id": str(cid),
            "file": fname,
            "n_days": n_adequate,
            "mean_finite_bins_1m": float(np.isfinite(hr_1m_arr).sum(axis=1).mean()),
            "occupancy_rate_1m": float(np.isfinite(hr_1m_arr).mean()),
            "mean_finite_bins_5m": float(np.isfinite(hr_5m_arr).sum(axis=1).mean()),
            "occupancy_rate_5m": float(np.isfinite(hr_5m_arr).mean()),
            "mean_finite_bins_15m": float(np.isfinite(hr_15m_arr).sum(axis=1).mean()),
            "occupancy_rate_15m": float(np.isfinite(hr_15m_arr).mean()),
        }
    except Exception as e:
        print(f"Error processing customer {cid}: {e}", file=sys.stderr)
        return None


def main():
    CACHE_1MIN_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_5MIN_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_15MIN_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Garmin manifest from {MANIFEST_PATH}...")
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"customer_id": str})
    tasks = list(zip(manifest["customer_id"], manifest["csv_path"]))
    
    print(f"Processing {len(tasks)} Garmin customers across 8 worker processes...")
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_single_customer, tasks))
        
    valid_results = [r for r in results if r is not None]
    dt = time.time() - t0
    print(f"\nCompleted processing {len(tasks)} customers in {dt:.1f}s.")
    print(f"Qualified customers with >=14 adequate days: {len(valid_results)} / {len(tasks)} ({len(valid_results)/len(tasks)*100:.1f}%)")
    
    res_df = pd.DataFrame(valid_results)
    res_df.to_csv(CACHE_1MIN_DIR / "customers.csv", index=False)
    res_df.to_csv(CACHE_5MIN_DIR / "customers.csv", index=False)
    res_df.to_csv(CACHE_15MIN_DIR / "customers.csv", index=False)
    
    print("\nGarmin Temporal Binning Data Occupancy Summary:")
    print(f"Mean adequate days per user: {res_df['n_days'].mean():.1f} (median: {res_df['n_days'].median():.1f})")
    print(f" 1-Minute Bins (1440): {res_df['mean_finite_bins_1m'].mean():.1f} / 1440 bins occupied ({res_df['occupancy_rate_1m'].mean()*100:.1f}%)")
    print(f" 5-Minute Bins (288):   {res_df['mean_finite_bins_5m'].mean():.1f} / 288 bins occupied ({res_df['occupancy_rate_5m'].mean()*100:.1f}%)")
    print(f"15-Minute Bins (96):    {res_df['mean_finite_bins_15m'].mean():.1f} / 96 bins occupied ({res_df['occupancy_rate_15m'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
