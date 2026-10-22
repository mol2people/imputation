"""Stage 2: daily-endpoint qualifying dates and deterministic windows.

Daily arms (type 65 RHR and type 66 sleep HR) use the independent daily-value
ascertainment rule from PLAN.md: exactly one resolved finite value in
[25, 230] bpm per exported calendar date, with no channel-3000 bin-density gate.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import common


def daily_for_arm(
    daily_clean: pd.DataFrame, source: int, endpoint: str
) -> pd.DataFrame:
    """Return sorted finite daily values for one source/endpoint arm."""
    mask = daily_clean["source"].eq(int(source)) & daily_clean["endpoint"].eq(endpoint)
    out = daily_clean.loc[mask, ["user_id", "source", "endpoint", "date", "value"]].copy()
    out = out.loc[np.isfinite(out["value"].to_numpy(dtype=float))]
    return out.sort_values(["user_id", "date"]).reset_index(drop=True)


def qualifying_dates_by_user(daily_arm: pd.DataFrame) -> dict[int, np.ndarray]:
    return {
        int(uid): np.sort(group["date"].to_numpy(dtype=np.int64))
        for uid, group in daily_arm.groupby("user_id", sort=True)
    }


def _select_block_dates(
    qual_dates: np.ndarray,
    start_day: int,
    protocol: str,
    source: int,
    user_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected dates, block indices, and quotas for one window."""
    capacities = common.get_window_block_capacities(qual_dates, int(start_day))
    if protocol == "expanded90":
        quotas = common.greedy_quota_allocation(capacities, target_total=90, base_quota=10)
    elif protocol == "balanced90":
        if not np.all(capacities >= 15):
            raise ValueError(f"balanced90 requires 15 per block, got {capacities.tolist()}")
        quotas = np.full(6, 15, dtype=np.int64)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    selected: list[np.ndarray] = []
    blocks: list[np.ndarray] = []
    for b in range(6):
        lo = int(start_day) + 30 * b
        hi = int(start_day) + 30 * (b + 1)
        block_dates = qual_dates[(qual_dates >= lo) & (qual_dates < hi)]
        seed = common.derive_block_seed(int(source), int(user_id), int(start_day), b)
        chosen = common.sample_block_dates(block_dates, int(quotas[b]), seed)
        selected.append(chosen)
        blocks.append(np.full(len(chosen), b, dtype=np.int64))
    all_dates = np.concatenate(selected)
    all_blocks = np.concatenate(blocks)
    order = np.argsort(all_dates)
    return (
        all_dates[order].astype(np.int64),
        all_blocks[order].astype(np.int64),
        quotas.astype(np.int64),
    )


def build_windows_for_arm(
    daily_arm: pd.DataFrame,
    source: int,
    endpoint: str,
    protocols: Iterable[str] = common.PROTOCOLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build window and selected-date tables for one source/endpoint arm."""
    dates_by_user = qualifying_dates_by_user(daily_arm)
    value_lookup = {
        int(uid): dict(zip(group["date"].astype(int), group["value"].astype(float)))
        for uid, group in daily_arm.groupby("user_id", sort=True)
    }
    window_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for protocol in protocols:
        for user_id, qual_dates in dates_by_user.items():
            first, second = common.find_qualifying_windows(qual_dates, protocol)
            for window_index, start in ((1, first), (2, second)):
                eligible = start is not None
                row: dict[str, Any] = {
                    "user_id": int(user_id),
                    "source": int(source),
                    "endpoint": endpoint,
                    "protocol": str(protocol),
                    "window_index": int(window_index),
                    "start_day": int(start) if eligible else pd.NA,
                    "end_day": int(start) + 179 if eligible else pd.NA,
                    "num_qualified_dates": int(len(qual_dates)),
                    "is_eligible": bool(eligible),
                }
                if not eligible:
                    window_rows.append(row)
                    continue
                capacities = common.get_window_block_capacities(qual_dates, int(start))
                row["block_capacities"] = ",".join(str(int(c)) for c in capacities)
                row["num_window_qualified_dates"] = int(capacities.sum())
                selected, block_idx, quotas = _select_block_dates(
                    qual_dates, int(start), protocol, int(source), int(user_id)
                )
                row["block_quotas"] = ",".join(str(int(q)) for q in quotas)
                digest = common.compute_selected_dates_hash(selected)
                row["selected_dates_hash"] = digest
                row["n_selected"] = int(len(selected))
                window_rows.append(row)
                for day, block in zip(selected, block_idx):
                    selected_rows.append(
                        {
                            "user_id": int(user_id),
                            "source": int(source),
                            "endpoint": endpoint,
                            "protocol": str(protocol),
                            "window_index": int(window_index),
                            "selected_dates_hash": digest,
                            "block_index": int(block),
                            "date": int(day),
                            "value": float(value_lookup[int(user_id)][int(day)]),
                        }
                    )

    windows = pd.DataFrame(window_rows)
    selected_df = pd.DataFrame(
        selected_rows,
        columns=[
            "user_id",
            "source",
            "endpoint",
            "protocol",
            "window_index",
            "selected_dates_hash",
            "block_index",
            "date",
            "value",
        ],
    )
    return windows, selected_df


def _build_arm_windows(args: tuple[pd.DataFrame, int, str, tuple[str, ...]]
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_arm, source, endpoint, protocols = args
    return build_windows_for_arm(daily_arm, int(source), str(endpoint), protocols)


def build_daily_manifests(
    daily_clean: pd.DataFrame,
    demographics: pd.DataFrame,
    sources: Iterable[int] = common.SCOPE_SOURCES,
    endpoints: Iterable[str] = ("RHR", "HR_during_sleep"),
    protocols: Iterable[str] = common.PROTOCOLS,
    workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and age-annotate manifests for every available daily arm.

    Arms are independent, so each source x endpoint window search runs in its
    own worker process.
    """
    protocols = tuple(protocols)
    jobs = []
    for source in sources:
        for endpoint in endpoints:
            daily_arm = daily_for_arm(daily_clean, source, endpoint)
            if daily_arm.empty:
                continue
            users = np.sort(daily_arm["user_id"].unique())
            # Chunk large arms so one source/endpoint cannot pin a single core.
            n_parts = max(1, min(int(workers), int(np.ceil(len(users) / 600))))
            for part in np.array_split(users, n_parts):
                if len(part) == 0:
                    continue
                subset = daily_arm.loc[daily_arm["user_id"].isin(part)]
                jobs.append((subset, int(source), str(endpoint), protocols))
    if not jobs:
        return pd.DataFrame(), pd.DataFrame()

    results: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_build_arm_windows, jobs, chunksize=1))
    else:
        results = [_build_arm_windows(job) for job in jobs]

    windows_frames = [r[0] for r in results if not r[0].empty]
    selected_frames = [r[1] for r in results if not r[1].empty]
    windows = pd.concat(windows_frames, ignore_index=True) if windows_frames else pd.DataFrame()
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    if not windows.empty:
        windows = common.attach_ages(windows, demographics, start_col="start_day")
    return windows, selected


def eligibility_summary(windows: pd.DataFrame) -> pd.DataFrame:
    """Count eligible participants per source/endpoint/protocol/window."""
    if windows.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in windows.groupby(
        ["source", "endpoint", "protocol", "window_index"], sort=True
    ):
        eligible = group.loc[group["is_eligible"]]
        rows.append(
            {
                "source": int(keys[0]),
                "source_name": common.source_name(int(keys[0])),
                "endpoint": str(keys[1]),
                "protocol": str(keys[2]),
                "window_index": int(keys[3]),
                "candidates": int(len(group)),
                "eligible": int(len(eligible)),
                "female_eligible": int((eligible["sex"].eq(1)).sum()),
                "male_eligible": int((eligible["sex"].eq(0)).sum()),
                "median_age": float(eligible["age"].median()) if len(eligible) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["source", "endpoint", "protocol", "window_index"])
