"""Shared manifest and cache helpers for the RHR definition sensitivity run."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
ARTIFACTS = EXPERIMENT_DIR / "artifacts"
PERCENTILES = {"HRp10": 10.0, "HRp5": 5.0, "HRp1": 1.0}
CACHE_PATHS = {
    6: REPO_ROOT / "experiments/apple_cohort_scaling/artifacts/cache_5min",
    3: REPO_ROOT / "experiments/garmin_infradian_replication/artifacts/cache_5min",
    7: ARTIFACTS / "cache_5min",
}


def load_manifests() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load participant, window, and frozen selected-date manifests."""
    part = pd.read_parquet(ARTIFACTS / "manifests/participants.parquet")
    windows = pd.read_parquet(ARTIFACTS / "windows/windows.parquet")
    dates = pd.read_parquet(ARTIFACTS / "windows/selected_dates.parquet")
    return part, windows, dates


def build_tasks(arm: str, protocol: str, limit: int | None = None,
                window_keys: set[tuple[int, int, int]] | None = None) -> list[dict[str, Any]]:
    """Build worker tasks from unchanged manifests and channel-3000 cache paths."""
    if arm not in PERCENTILES:
        raise ValueError(f"Unknown arm {arm!r}; choose {tuple(PERCENTILES)}")
    part, windows, selected = load_manifests()
    selected = selected.loc[selected.protocol.eq(protocol)].copy()
    demos = part.set_index(["user_id", "source"]).to_dict("index")
    starts = windows.loc[windows.is_eligible].set_index(
        ["user_id", "source", "protocol", "window_index"]
    )["start_day"].to_dict()
    tasks: list[dict[str, Any]] = []
    grouped = selected.groupby(["user_id", "source"], sort=True)
    for (uid, src), group in grouped:
        uid, src = int(uid), int(src)
        if src not in CACHE_PATHS:
            continue
        demo_row = demos.get((uid, src))
        if demo_row is None:
            continue
        demo = {
            "salutation": int(demo_row["salutation"]) if pd.notna(demo_row["salutation"]) else None,
            "birth_year": int(demo_row["birth_year"]) if pd.notna(demo_row["birth_year"]) else None,
            "age_dec1": int(demo_row["age_dec1"]) if pd.notna(demo_row["age_dec1"]) else None,
            "age_jan1": int(demo_row["age_jan1"]) if pd.notna(demo_row["age_jan1"]) else None,
            "age_dec31": int(demo_row["age_dec31"]) if pd.notna(demo_row["age_dec31"]) else None,
            "weight": float(demo_row["weight"]) if pd.notna(demo_row["weight"]) else np.nan,
            "height": float(demo_row["height"]) if pd.notna(demo_row["height"]) else np.nan,
            "bmi": float(demo_row["bmi"]) if pd.notna(demo_row["bmi"]) else np.nan,
        }
        win_tasks = []
        for (proto, widx, h), dates in group.groupby(
            ["protocol", "window_index", "selected_dates_hash"], sort=True
        ):
            widx = int(widx)
            if window_keys is not None and (uid, src, widx) not in window_keys:
                continue
            start = int(starts[(uid, src, str(proto), widx)])
            date_arr = np.sort(dates.date.to_numpy(dtype=np.int64))
            actual_hash = hashlib.sha256(np.asarray(date_arr, dtype="<i8").tobytes()).hexdigest()
            if actual_hash != str(h):
                raise ValueError(f"Selected-date hash mismatch for {(uid, src, proto, widx)}")
            win_tasks.append({"protocol": str(proto), "window_index": widx,
                              "start_day": start, "selected_dates_hash": str(h),
                              "dates": date_arr.tolist()})
        if not win_tasks:
            continue
        tasks.append({"user_id": uid, "source": src,
                      "cache_path": str(CACHE_PATHS[src] / f"customer_{uid}.npz"),
                      "demographics": demo, "windows": win_tasks,
                      "percentile": PERCENTILES[arm], "arm": arm})
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def validate_percentile_order(task: dict[str, Any]) -> None:
    """Check p1 <= p5 <= p10 on every selected date for one cache task."""
    with np.load(task["cache_path"]) as cache:
        date_rows = {int(d): cache["hr"][i] for i, d in enumerate(cache["date"])}
    for window in task["windows"]:
        for day in window["dates"]:
            values = date_rows[int(day)]
            values = values[np.isfinite(values)]
            if len(values) < 120:
                raise ValueError(f"Selected date {day} has only {len(values)} finite bins")
            q1, q5, q10 = np.percentile(values, [1, 5, 10])
            if not q1 <= q5 <= q10:
                raise ValueError(f"Percentile order failed on selected date {day}")


def build_supplier_tasks(prepared: pd.DataFrame, endpoint: str, protocol: str,
                         limit: int | None = None) -> list[dict[str, Any]]:
    """Build worker tasks from supplier.py's exact-date complete-case output."""
    part, windows, _ = load_manifests()
    demos = part.set_index(["user_id", "source"]).to_dict("index")
    starts = windows.loc[windows.is_eligible].set_index(
        ["user_id", "source", "protocol", "window_index"]
    )["start_day"].to_dict()
    data = prepared.loc[prepared.endpoint.eq(endpoint) & prepared.protocol.eq(protocol)].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    data["date_day"] = data["date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    tasks: list[dict[str, Any]] = []
    for (uid, src), group in data.groupby(["user_id", "source"], sort=True):
        uid, src = int(uid), int(src)
        if src not in CACHE_PATHS or (uid, src) not in demos:
            continue
        row = demos[(uid, src)]
        demo = {
            "salutation": int(row["salutation"]) if pd.notna(row["salutation"]) else None,
            "birth_year": int(row["birth_year"]) if pd.notna(row["birth_year"]) else None,
            "age_dec1": int(row["age_dec1"]) if pd.notna(row["age_dec1"]) else None,
            "age_jan1": int(row["age_jan1"]) if pd.notna(row["age_jan1"]) else None,
            "age_dec31": int(row["age_dec31"]) if pd.notna(row["age_dec31"]) else None,
            "weight": float(row["weight"]) if pd.notna(row["weight"]) else np.nan,
            "height": float(row["height"]) if pd.notna(row["height"]) else np.nan,
            "bmi": float(row["bmi"]) if pd.notna(row["bmi"]) else np.nan,
        }
        win_tasks = []
        for (proto, widx, h), win in group.groupby(
            ["protocol", "window_index", "selected_dates_hash"], sort=True
        ):
            dates = win["date_day"].to_numpy(dtype=np.int64)
            value_map = dict(zip(dates.tolist(), win["value"].astype(float).tolist()))
            win_tasks.append({"protocol": str(proto), "window_index": int(widx),
                              "start_day": int(starts[(uid, src, str(proto), int(widx))]),
                              "selected_dates_hash": str(h), "dates": np.sort(dates).tolist(),
                              "daily_values": value_map})
        if not win_tasks:
            continue
        # The engine accepts one daily-value map per participant task; keep each window
        # separate so exact-date membership remains explicit.
        for win in win_tasks:
            tasks.append({"user_id": uid, "source": src,
                          "cache_path": str(CACHE_PATHS[src] / f"customer_{uid}.npz"),
                          "demographics": demo, "windows": [{k: v for k, v in win.items() if k != "daily_values"}],
                          "daily_values": win["daily_values"], "percentile": 10.0,
                          "arm": str(endpoint)})
    return tasks[:limit] if limit is not None else tasks


def date_manifest_hash() -> str:
    """Hash sorted date-manifest records to capture fixed selection provenance."""
    path = ARTIFACTS / "windows/selected_dates.parquet"
    return hashlib.sha256(path.read_bytes()).hexdigest()
