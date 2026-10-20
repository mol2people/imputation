#!/usr/bin/env python3
"""
01_build_manifests_and_windows.py: Milestone 1 Cohort Extraction, Samsung Cache Builder,
Demographic Joins, and Standardized Window Selection.

Tasks:
1. Validate and read existing Apple (N=11,139) and Garmin Tier 1c (N=4,410) 5m caches without modifying them.
2. Construct missing source-specific 5m cache for Samsung candidate participants from `out/{user_id}.csv`
   (channel 3000, 25-230 bpm, valid timezone offset, arithmetic mean per 5m bin, date arrays with >=120 bins/day filter),
   saved into `experiments/infradian_age_dependency/artifacts/cache_5min/`.
3. Demographic joins on `data/13Aug_1222.csv`: `user_id`, `salutation` (10=female, 20=male), `birth_date` (birth year),
   continuous age under December-1 convention (`age = start.year - birth_year - 1 + I(start.month == 12)`),
   plus January-1 and December-31 sensitivities.
4. Execute window selection across all candidates for `expanded90` (primary: 180-day window, 6 blocks of 30 days,
   >=10 qualified dates/block, greedy allocation to 90 total) and `balanced90` (reference: 15/block).
5. Sample selected dates using PCG64 with seed derived from SHA-256 of
   `20261020|infradian_age|{source}|{user_id}|{window_start_day}|dates_block_{b}`.
6. Persist artifacts:
   - `experiments/infradian_age_dependency/artifacts/manifests/participants.parquet`
   - `experiments/infradian_age_dependency/artifacts/windows/windows.parquet`
   - `experiments/infradian_age_dependency/artifacts/windows/selected_dates.parquet`
   - `experiments/infradian_age_dependency/artifacts/manifests/age_support.csv` (5-year band counts)
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Add scripts dir to path
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import (
    REPO_ROOT,
    EXPERIMENT_DIR,
    ARTIFACTS_DIR,
    CACHE_5MIN_DIR,
    MANIFESTS_DIR,
    WINDOWS_DIR,
    UPSTREAM_APPLE_CACHE,
    UPSTREAM_GARMIN_CACHE,
    DEMOGRAPHICS_CSV,
    EPOCH_SOURCES_PARQUET,
    RAW_OUT_DIR,
    compute_all_ages,
    greedy_quota_allocation,
    derive_block_seed,
    sample_block_dates,
    compute_selected_dates_hash,
    find_qualifying_windows,
    get_window_block_capacities,
)


# =============================================================================
# Helper: Samsung 5m Cache Construction Worker
# =============================================================================
def _build_samsung_cache_for_user(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Processes raw CSV for a Samsung participant:
      - Filter: source == 7, type == 3000, HR in [25, 230], positive start, finite tz
      - Local time: startTimestamp + timezoneOffset * 60000 ms
      - 5m clock bins: (local_ms % 86400000) // 300000
      - Arithmetic mean HR per bin
      - Filter: retain only dates with >= 120 occupied bins
      - Saves .npz if >= 1 qualified date
    """
    csv_path = RAW_OUT_DIR / f"{user_id}.csv"
    if not csv_path.exists():
        return None
        
    out_npz = CACHE_5MIN_DIR / f"customer_{user_id}.npz"
    if out_npz.exists():
        try:
            with np.load(out_npz) as d:
                return {
                    "user_id": user_id,
                    "n_dates": len(d["date"]),
                    "cached": True,
                }
        except Exception:
            pass  # re-create corrupted file

    try:
        cols = ["startTimestamp", "endTimestamp", "timezoneOffset", "longValue", "source", "type"]
        df = pd.read_csv(csv_path, usecols=cols)
        ok = (
            (df["source"] == 7) &
            (df["type"] == 3000) &
            df["longValue"].between(25, 230) &
            (df["startTimestamp"] > 0) &
            np.isfinite(df["startTimestamp"]) &
            np.isfinite(df["timezoneOffset"]) &
            np.isfinite(df["longValue"])
        )
        sub = df.loc[ok].copy()
        if sub.empty:
            return None
            
        start = sub["startTimestamp"].to_numpy(np.int64)
        tz = sub["timezoneOffset"].to_numpy(np.float64)
        local_ms = start + (tz * 60_000).astype(np.int64)
        
        sub["date"] = pd.to_datetime(local_ms, unit="ms").normalize()
        sub["bin"] = (local_ms % 86_400_000) // 300_000
        sub["hour"] = sub["bin"] // 12
        
        # Aggregate daily
        dates_list = []
        hr_list = []
        stats_list = []
        
        for date_val, day_df in sub.groupby("date", sort=True):
            n_bins = day_df["bin"].nunique()
            if n_bins < 120:
                continue
                
            vals = day_df["longValue"].to_numpy(dtype=np.float32)
            n_readings = len(vals)
            hours_covered = day_df["hour"].nunique()
            
            # 5m bin arithmetic means
            bin_means = day_df.groupby("bin")["longValue"].mean().reindex(range(288)).to_numpy(np.float32)
            
            day_dt = np.datetime64(date_val, "D")
            dates_list.append(day_dt)
            hr_list.append(bin_means)
            stats_list.append([
                float(vals.mean()),
                float(np.median(vals)),
                float(vals.std(ddof=1)) if n_readings > 1 else 0.0,
                float(vals.min()),
                float(vals.max()),
                float(n_readings),
                float(n_bins * 5.0 / 60.0),  # approx occupied hours
                float(hours_covered),
            ])
            
        if len(dates_list) == 0:
            return None
            
        dates_arr = np.array(dates_list, dtype="datetime64[D]").view("int64")
        hr_arr = np.stack(hr_list).astype(np.float32)
        stats_arr = np.array(stats_list, dtype=np.float32)
        
        np.savez_compressed(
            out_npz,
            hr=hr_arr,
            date=dates_arr,
            stats=stats_arr,
        )
        return {
            "user_id": user_id,
            "n_dates": len(dates_arr),
            "cached": False,
        }
    except Exception as e:
        print(f"Error processing Samsung user {user_id}: {e}", file=sys.stderr)
        return None


# =============================================================================
# Helper: Participant Window Processing Worker
# =============================================================================
def _process_participant_windows(task: Tuple[int, int, str]) -> Dict[str, Any]:
    """
    Loads cache for a participant (user_id, source, cache_path_str),
    computes qualified dates (>=120 finite bins),
    and evaluates window eligibility for expanded90 and balanced90.
    """
    user_id, source, cache_path_str = task
    cache_path = Path(cache_path_str)
    
    res = {
        "user_id": user_id,
        "source": source,
        "n_qual_dates": 0,
        "expanded90": {"eligible": False, "windows": []},
        "balanced90": {"eligible": False, "windows": []},
    }
    
    if not cache_path.exists():
        return res
        
    try:
        with np.load(cache_path) as d:
            hr = d["hr"]
            dates = d["date"]
            
        valid_bins_per_day = np.isfinite(hr).sum(axis=1)
        qual_mask = (valid_bins_per_day >= 120)
        qual_dates = np.sort(dates[qual_mask])
        res["n_qual_dates"] = len(qual_dates)
        
        if len(qual_dates) < 60:
            return res
            
        # Mapping from date to occupied bins
        date_to_bins = dict(zip(dates[qual_mask].tolist(), valid_bins_per_day[qual_mask].tolist()))
        
        for protocol in ["expanded90", "balanced90"]:
            s1, s2 = find_qualifying_windows(qual_dates, protocol=protocol)
            if s1 is None:
                continue
                
            res[protocol]["eligible"] = True
            
            # Process Window 1
            cap1 = get_window_block_capacities(qual_dates, s1)
            q1 = greedy_quota_allocation(cap1, 90, 10) if protocol == "expanded90" else np.full(6, 15, dtype=np.int64)
            
            w1_sampled_dates = []
            w1_block_indices = []
            w1_occupied_bins = []
            
            for b in range(6):
                b_s = s1 + 30 * b
                b_e = s1 + 30 * (b + 1)
                b_dates = qual_dates[(qual_dates >= b_s) & (qual_dates < b_e)]
                b_seed = derive_block_seed(source, user_id, s1, b)
                b_sample = sample_block_dates(b_dates, q1[b], b_seed)
                for d_val in b_sample:
                    w1_sampled_dates.append(int(d_val))
                    w1_block_indices.append(b)
                    w1_occupied_bins.append(int(date_to_bins[int(d_val)]))
                    
            w1_hash = compute_selected_dates_hash(np.array(w1_sampled_dates, dtype=np.int64))
            
            res[protocol]["windows"].append({
                "window_index": 1,
                "start_day": s1,
                "end_day": s1 + 180,
                "num_qualified_dates": int(cap1.sum()),
                "selected_dates_hash": w1_hash,
                "dates": w1_sampled_dates,
                "block_indices": w1_block_indices,
                "occupied_bins": w1_occupied_bins,
            })
            
            # Process Window 2 (if present)
            if s2 is not None:
                cap2 = get_window_block_capacities(qual_dates, s2)
                q2 = greedy_quota_allocation(cap2, 90, 10) if protocol == "expanded90" else np.full(6, 15, dtype=np.int64)
                
                w2_sampled_dates = []
                w2_block_indices = []
                w2_occupied_bins = []
                
                for b in range(6):
                    b_s = s2 + 30 * b
                    b_e = s2 + 30 * (b + 1)
                    b_dates = qual_dates[(qual_dates >= b_s) & (qual_dates < b_e)]
                    b_seed = derive_block_seed(source, user_id, s2, b)
                    b_sample = sample_block_dates(b_dates, q2[b], b_seed)
                    for d_val in b_sample:
                        w2_sampled_dates.append(int(d_val))
                        w2_block_indices.append(b)
                        w2_occupied_bins.append(int(date_to_bins[int(d_val)]))
                        
                w2_hash = compute_selected_dates_hash(np.array(w2_sampled_dates, dtype=np.int64))
                
                res[protocol]["windows"].append({
                    "window_index": 2,
                    "start_day": s2,
                    "end_day": s2 + 180,
                    "num_qualified_dates": int(cap2.sum()),
                    "selected_dates_hash": w2_hash,
                    "dates": w2_sampled_dates,
                    "block_indices": w2_block_indices,
                    "occupied_bins": w2_occupied_bins,
                })
                
        return res
    except Exception as e:
        print(f"Error processing windows for user {user_id}: {e}", file=sys.stderr)
        return res


# =============================================================================
# Main Pipeline Implementation
# =============================================================================
def main() -> None:
    t_start = time.perf_counter()
    print("================================================================================")
    print("STARTING MILESTONE 1: Manifests, Samsung Cache Builder, & Window Selection")
    print("================================================================================")
    
    # 1. Ensure target directories
    CACHE_5MIN_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load Demographics (data/13Aug_1222.csv)
    print("\n--- [Step 1/5] Loading Demographics from data/13Aug_1222.csv ---")
    demo_cols = ["user_id", "salutation", "birth_date", "weight", "height"]
    df_demo = pd.read_csv(DEMOGRAPHICS_CSV, usecols=demo_cols)
    print(f"  Loaded {len(df_demo)} demographic rows.")
    
    # Compute BMI: weight (kg) / (height (m))^2
    w = df_demo["weight"].to_numpy(dtype=np.float64)
    h = df_demo["height"].to_numpy(dtype=np.float64)
    valid_wh = (w > 0) & (h > 0)
    bmi = np.full(len(df_demo), np.nan, dtype=np.float64)
    bmi[valid_wh] = w[valid_wh] / ((h[valid_wh] / 100.0) ** 2)
    df_demo["bmi"] = bmi

    # Create fast demographic lookup dict: user_id -> dict
    demo_dict = {}
    for _, row in df_demo.iterrows():
        uid = int(row["user_id"])
        sal = int(row["salutation"]) if pd.notna(row["salutation"]) else None
        by = int(row["birth_date"]) if pd.notna(row["birth_date"]) else None
        wt = float(row["weight"]) if pd.notna(row["weight"]) else np.nan
        ht = float(row["height"]) if pd.notna(row["height"]) else np.nan
        bm = float(row["bmi"]) if pd.notna(row["bmi"]) else np.nan
        demo_dict[uid] = {
            "salutation": sal,
            "birth_year": by,
            "weight": wt,
            "height": ht,
            "bmi": bm,
        }
    print(f"  Demographic dictionary indexed for {len(demo_dict)} users.")

    # 3. Build Samsung 5m Cache
    print("\n--- [Step 2/5] Building Missing Samsung 5m Cache in artifacts/cache_5min/ ---")
    df_epoch = pd.read_parquet(EPOCH_SOURCES_PARQUET)
    samsung_candidates = df_epoch[(df_epoch["source"] == 7) & (df_epoch["channel"] == 3000)]["user"].unique().tolist()
    print(f"  Found {len(samsung_candidates)} Samsung channel-3000 candidates in epoch_sources.")
    
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as ex:
        samsung_res = list(ex.map(_build_samsung_cache_for_user, samsung_candidates, chunksize=50))
    samsung_cached = [r for r in samsung_res if r is not None]
    print(f"  Samsung 5m cache complete: {len(samsung_cached)} participants with >=1 qualified date (took {time.perf_counter() - t0:.1f}s).")

    # 4. Cohort Manifests and Cache Path Assembly
    print("\n--- [Step 3/5] Assembling Cohort Candidates and Existing Caches ---")
    
    # Apple Watch (N=11,139)
    apple_alloc = pd.read_parquet(REPO_ROOT / "experiments/apple_cohort_scaling/artifacts/apple_10_allocations.parquet")
    apple_users = apple_alloc["customer_id"].astype(int).tolist()
    print(f"  Apple Watch candidates: N = {len(apple_users)}")
    
    # Garmin Tier 1c (N=4,410)
    garmin_mf = pd.read_csv(REPO_ROOT / "experiments/garmin_infradian_replication/artifacts/garmin_tier1c_cached_manifest.csv")
    garmin_users = garmin_mf["customer_id"].astype(int).tolist()
    print(f"  Garmin Tier 1c candidates: N = {len(garmin_users)}")
    
    # Samsung (N=2,089)
    print(f"  Samsung candidates: N = {len(samsung_candidates)}")

    cohort_tasks: List[Tuple[int, int, str]] = []
    for u in apple_users:
        cohort_tasks.append((u, 6, str(UPSTREAM_APPLE_CACHE / f"customer_{u}.npz")))
    for u in garmin_users:
        cohort_tasks.append((u, 3, str(UPSTREAM_GARMIN_CACHE / f"customer_{u}.npz")))
    for u in samsung_candidates:
        cohort_tasks.append((u, 7, str(CACHE_5MIN_DIR / f"customer_{u}.npz")))
        
    print(f"  Total candidate evaluations across 3 sources: {len(cohort_tasks)}")

    # 5. Window Selection Scan across all Candidates
    print("\n--- [Step 4/5] Executing Standardized Window Selection & PCG64 Sampling ---")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as ex:
        window_results = list(ex.map(_process_participant_windows, cohort_tasks, chunksize=100))
    print(f"  Window selection scan completed in {time.perf_counter() - t0:.1f}s.")

    # 6. Build Manifest and Window Artifacts
    print("\n--- [Step 5/5] Generating Artifacts and Enforcing Parity ---")
    
    participants_rows = []
    windows_rows = []
    selected_dates_rows = []
    
    # Index window results by (user_id, source)
    for res in window_results:
        uid = res["user_id"]
        src = res["source"]
        demo = demo_dict.get(uid, {})
        sal = demo.get("salutation")
        by = demo.get("birth_year")
        wt = demo.get("weight", np.nan)
        ht = demo.get("height", np.nan)
        bm = demo.get("bmi", np.nan)
        
        # Primary expanded90 window 1 for age computation
        e90_info = res["expanded90"]
        s_e90 = None
        if e90_info["eligible"] and len(e90_info["windows"]) > 0:
            s_e90 = e90_info["windows"][0]["start_day"]
            
        if s_e90 is not None and by is not None:
            age_dec1, age_jan1, age_dec31 = compute_all_ages(s_e90, by)
        else:
            age_dec1, age_jan1, age_dec31 = pd.NA, pd.NA, pd.NA
            
        participants_rows.append({
            "user_id": uid,
            "source": src,
            "salutation": sal,
            "birth_year": by,
            "age_dec1": age_dec1,
            "age_jan1": age_jan1,
            "age_dec31": age_dec31,
            "weight": wt,
            "height": ht,
            "bmi": bm,
        })
        
        # Windows and Selected Dates
        for protocol in ["expanded90", "balanced90"]:
            proto_info = res[protocol]
            if proto_info["eligible"] and len(proto_info["windows"]) > 0:
                for win in proto_info["windows"]:
                    w_idx = win["window_index"]
                    w_start = win["start_day"]
                    w_end = win["end_day"]
                    w_n_qual = win["num_qualified_dates"]
                    w_hash = win["selected_dates_hash"]
                    
                    windows_rows.append({
                        "user_id": uid,
                        "source": src,
                        "protocol": protocol,
                        "window_index": w_idx,
                        "start_day": w_start,
                        "end_day": w_end,
                        "num_qualified_dates": w_n_qual,
                        "is_eligible": True,
                    })
                    
                    dates = win["dates"]
                    blocks = win["block_indices"]
                    bins = win["occupied_bins"]
                    for d_val, b_val, occ_val in zip(dates, blocks, bins):
                        selected_dates_rows.append({
                            "user_id": uid,
                            "source": src,
                            "protocol": protocol,
                            "window_index": w_idx,
                            "selected_dates_hash": w_hash,
                            "block_index": b_val,
                            "date": d_val,
                            "occupied_bins": occ_val,
                        })
            else:
                # Record non-eligible candidate row
                windows_rows.append({
                    "user_id": uid,
                    "source": src,
                    "protocol": protocol,
                    "window_index": 1,
                    "start_day": pd.NA,
                    "end_day": pd.NA,
                    "num_qualified_dates": res["n_qual_dates"],
                    "is_eligible": False,
                })

    # DataFrames
    df_participants = pd.DataFrame(participants_rows)
    # Ensure correct pandas dtypes
    df_participants["user_id"] = df_participants["user_id"].astype("int64")
    df_participants["source"] = df_participants["source"].astype("int64")
    df_participants["salutation"] = df_participants["salutation"].astype("Int64")
    df_participants["birth_year"] = df_participants["birth_year"].astype("Int64")
    df_participants["age_dec1"] = df_participants["age_dec1"].astype("Int64")
    df_participants["age_jan1"] = df_participants["age_jan1"].astype("Int64")
    df_participants["age_dec31"] = df_participants["age_dec31"].astype("Int64")
    df_participants["weight"] = df_participants["weight"].astype("float32")
    df_participants["height"] = df_participants["height"].astype("float32")
    df_participants["bmi"] = df_participants["bmi"].astype("float32")

    df_windows = pd.DataFrame(windows_rows)
    df_windows["user_id"] = df_windows["user_id"].astype("int64")
    df_windows["source"] = df_windows["source"].astype("int64")
    df_windows["protocol"] = df_windows["protocol"].astype("string")
    df_windows["window_index"] = df_windows["window_index"].astype("int64")
    df_windows["start_day"] = df_windows["start_day"].astype("Int64")
    df_windows["end_day"] = df_windows["end_day"].astype("Int64")
    df_windows["num_qualified_dates"] = df_windows["num_qualified_dates"].astype("int64")
    df_windows["is_eligible"] = df_windows["is_eligible"].astype("bool")

    df_selected_dates = pd.DataFrame(selected_dates_rows)
    df_selected_dates["user_id"] = df_selected_dates["user_id"].astype("int64")
    df_selected_dates["source"] = df_selected_dates["source"].astype("int64")
    df_selected_dates["protocol"] = df_selected_dates["protocol"].astype("string")
    df_selected_dates["window_index"] = df_selected_dates["window_index"].astype("int64")
    df_selected_dates["selected_dates_hash"] = df_selected_dates["selected_dates_hash"].astype("string")
    df_selected_dates["block_index"] = df_selected_dates["block_index"].astype("int64")
    df_selected_dates["date"] = df_selected_dates["date"].astype("int64")
    df_selected_dates["occupied_bins"] = df_selected_dates["occupied_bins"].astype("int64")

    # Persist parquet files
    p_participants = MANIFESTS_DIR / "participants.parquet"
    p_windows = WINDOWS_DIR / "windows.parquet"
    p_selected_dates = WINDOWS_DIR / "selected_dates.parquet"

    df_participants.to_parquet(p_participants, index=False)
    df_windows.to_parquet(p_windows, index=False)
    df_selected_dates.to_parquet(p_selected_dates, index=False)
    print(f"\n  Saved participants.parquet ({len(df_participants)} rows): {p_participants}")
    print(f"  Saved windows.parquet ({len(df_windows)} rows): {p_windows}")
    print(f"  Saved selected_dates.parquet ({len(df_selected_dates)} rows): {p_selected_dates}")

    # 7. Generate Age Support Table (5-Year Bands)
    print("\n--- Generating Age Support Summary (artifacts/manifests/age_support.csv) ---")
    bins_5y = list(range(0, 130, 5))
    labels_5y = [f"[{b},{b+5})" for b in bins_5y[:-1]]
    
    # Merge eligible window 1 with demographics to get age per protocol
    df_elig_w1 = df_windows[(df_windows["is_eligible"]) & (df_windows["window_index"] == 1)].copy()
    df_elig_w1 = df_elig_w1.merge(df_participants[["user_id", "source", "salutation", "birth_year"]], on=["user_id", "source"], how="left")
    
    # Compute age for this specific protocol's window start
    w_ages = []
    for _, row in df_elig_w1.iterrows():
        s = int(row["start_day"])
        by = int(row["birth_year"])
        age_dec1, _, _ = compute_all_ages(s, by)
        w_ages.append(age_dec1)
    df_elig_w1["age"] = w_ages
    df_elig_w1["age_band"] = pd.cut(df_elig_w1["age"], bins=bins_5y, right=False, labels=labels_5y)
    
    support_rows = []
    source_names = {6: "Apple", 3: "Garmin", 7: "Samsung"}
    for src in [6, 3, 7]:
        for proto in ["expanded90", "balanced90"]:
            sub = df_elig_w1[(df_elig_w1["source"] == src) & (df_elig_w1["protocol"] == proto)]
            for band in labels_5y:
                band_sub = sub[sub["age_band"] == band]
                f_cnt = int((band_sub["salutation"] == 10).sum())
                m_cnt = int((band_sub["salutation"] == 20).sum())
                o_cnt = int((~band_sub["salutation"].isin([10, 20])).sum())
                tot = len(band_sub)
                support_rows.append({
                    "source": src,
                    "source_name": source_names[src],
                    "protocol": proto,
                    "age_band": band,
                    "female_count": f_cnt,
                    "male_count": m_cnt,
                    "other_count": o_cnt,
                    "total_count": tot,
                    "female_ge_30": bool(f_cnt >= 30),
                    "shared_ge_30": bool(f_cnt >= 30 and m_cnt >= 30),
                })
                
    df_support = pd.DataFrame(support_rows)
    p_support = MANIFESTS_DIR / "age_support.csv"
    df_support.to_csv(p_support, index=False)
    print(f"  Saved age_support.csv ({len(df_support)} rows): {p_support}")

    # 8. Empirical Verification & Parity Assertions
    print("\n--- Verifying Parity Against Authoritative Survey 2 Benchmark ---")
    
    # Apple expanded90: 5,390 (1,765 female, 3,625 male)
    apple_e90 = df_elig_w1[(df_elig_w1["source"] == 6) & (df_elig_w1["protocol"] == "expanded90")]
    apple_e90_f = (apple_e90["salutation"] == 10).sum()
    apple_e90_m = (apple_e90["salutation"] == 20).sum()
    print(f"  Apple expanded90: Total = {len(apple_e90)} (Female = {apple_e90_f}, Male = {apple_e90_m})")
    assert len(apple_e90) == 5390, f"Expected 5,390 Apple expanded90, got {len(apple_e90)}"
    assert apple_e90_f == 1765, f"Expected 1,765 Apple expanded90 females, got {apple_e90_f}"
    assert apple_e90_m == 3625, f"Expected 3,625 Apple expanded90 males, got {apple_e90_m}"

    # Apple balanced90: 4,631 (1,478 female, 3,153 male)
    apple_b90 = df_elig_w1[(df_elig_w1["source"] == 6) & (df_elig_w1["protocol"] == "balanced90")]
    apple_b90_f = (apple_b90["salutation"] == 10).sum()
    apple_b90_m = (apple_b90["salutation"] == 20).sum()
    print(f"  Apple balanced90: Total = {len(apple_b90)} (Female = {apple_b90_f}, Male = {apple_b90_m})")
    assert len(apple_b90) == 4631, f"Expected 4,631 Apple balanced90, got {len(apple_b90)}"
    assert apple_b90_f == 1478, f"Expected 1,478 Apple balanced90 females, got {apple_b90_f}"
    assert apple_b90_m == 3153, f"Expected 3,153 Apple balanced90 males, got {apple_b90_m}"

    # Garmin expanded90: 3,832 (1,376 female, 2,456 male)
    garmin_e90 = df_elig_w1[(df_elig_w1["source"] == 3) & (df_elig_w1["protocol"] == "expanded90")]
    garmin_e90_f = (garmin_e90["salutation"] == 10).sum()
    garmin_e90_m = (garmin_e90["salutation"] == 20).sum()
    print(f"  Garmin expanded90: Total = {len(garmin_e90)} (Female = {garmin_e90_f}, Male = {garmin_e90_m})")
    assert len(garmin_e90) == 3832, f"Expected 3,832 Garmin expanded90, got {len(garmin_e90)}"
    assert garmin_e90_f == 1376, f"Expected 1,376 Garmin expanded90 females, got {garmin_e90_f}"
    assert garmin_e90_m == 2456, f"Expected 2,456 Garmin expanded90 males, got {garmin_e90_m}"

    # Garmin balanced90: 3,760 (1,339 female, 2,421 male)
    garmin_b90 = df_elig_w1[(df_elig_w1["source"] == 3) & (df_elig_w1["protocol"] == "balanced90")]
    garmin_b90_f = (garmin_b90["salutation"] == 10).sum()
    garmin_b90_m = (garmin_b90["salutation"] == 20).sum()
    print(f"  Garmin balanced90: Total = {len(garmin_b90)} (Female = {garmin_b90_f}, Male = {garmin_b90_m})")
    assert len(garmin_b90) == 3760, f"Expected 3,760 Garmin balanced90, got {len(garmin_b90)}"
    assert garmin_b90_f == 1339, f"Expected 1,339 Garmin balanced90 females, got {garmin_b90_f}"
    assert garmin_b90_m == 2421, f"Expected 2,421 Garmin balanced90 males, got {garmin_b90_m}"

    # Samsung: 8 expanded90, 6 balanced90
    samsung_e90 = df_elig_w1[(df_elig_w1["source"] == 7) & (df_elig_w1["protocol"] == "expanded90")]
    samsung_b90 = df_elig_w1[(df_elig_w1["source"] == 7) & (df_elig_w1["protocol"] == "balanced90")]
    print(f"  Samsung expanded90: Total = {len(samsung_e90)}")
    print(f"  Samsung balanced90: Total = {len(samsung_b90)}")
    assert len(samsung_e90) == 8, f"Expected 8 Samsung expanded90, got {len(samsung_e90)}"
    assert len(samsung_b90) == 6, f"Expected 6 Samsung balanced90, got {len(samsung_b90)}"

    # Date Sampling Integrity: Each eligible window must have exactly 90 selected dates
    dates_per_win = df_selected_dates.groupby(["user_id", "source", "protocol", "window_index"]).size()
    assert (dates_per_win == 90).all(), "Every eligible window must have exactly 90 selected dates!"
    print(f"  Selected dates verified: {len(dates_per_win)} windows, all with exactly 90 sampled observations.")
    
    # Occupied bins integrity: All sampled dates must have >=120 occupied bins
    assert (df_selected_dates["occupied_bins"] >= 120).all(), "All selected dates must have >= 120 occupied bins!"
    print(f"  Occupied bins verified: min = {df_selected_dates['occupied_bins'].min()}, all >= 120.")

    total_time = time.perf_counter() - t_start
    print("\n================================================================================")
    print(f"MILESTONE 1 COMPLETE: All deliverables verified and persisted in {total_time:.1f}s.")
    print("================================================================================")


if __name__ == "__main__":
    main()
