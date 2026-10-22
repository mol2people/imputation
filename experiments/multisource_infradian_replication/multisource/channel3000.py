"""Channel-3000 cache construction and Analysis A manifests.

Existing Apple/Garmin/Samsung caches are reused only for their own source
directory.  New caches are source-tagged so a multisource participant never
collides with another device trajectory.
"""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import common, daily, reml

UPSTREAM_CACHE_DIRS: dict[int, Path] = {
    6: common.REPO_ROOT / "experiments/apple_cohort_scaling/artifacts/cache_5min",
    3: common.REPO_ROOT / "experiments/garmin_infradian_replication/artifacts/cache_5min",
    7: common.UPSTREAM_EXP / "artifacts/cache_5min",
}


def candidate_users(daily_clean: pd.DataFrame, sources: Iterable[int]) -> pd.DataFrame:
    """Users worth scanning or caching, from daily inventory and epoch metadata."""
    frames = []
    for source in sources:
        users = set(
            daily_clean.loc[daily_clean["source"].eq(int(source)), "user_id"].astype(int)
        )
        frames.append(pd.DataFrame({"user_id": sorted(users), "source": int(source)}))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["user_id", "source"]
    )
    if common.EPOCH_SOURCES_PARQUET.exists():
        epoch = pd.read_parquet(common.EPOCH_SOURCES_PARQUET)
        dense = epoch.loc[epoch["channel"].eq(common.DENSE_CHANNEL)]
        extra = pd.DataFrame(
            {
                "user_id": pd.to_numeric(dense["user"], errors="raise").astype(int),
                "source": pd.to_numeric(dense["source"], errors="raise").astype(int),
            }
        )
        out = pd.concat([out, extra], ignore_index=True)
    return out.drop_duplicates(["user_id", "source"]).sort_values(["source", "user_id"]).reset_index(drop=True)


def _scan_one(args: tuple[int, int, str]) -> dict[str, Any] | None:
    user_id, source, path_str = args
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"customer", "type", "source"}.issubset(reader.fieldnames):
                return None
            rows = 0
            for row in reader:
                try:
                    if int(row["type"]) != common.DENSE_CHANNEL:
                        continue
                    if int(row["source"]) != int(source):
                        continue
                except (TypeError, ValueError):
                    continue
                rows += 1
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    if rows == 0:
        return None
    return {"user_id": int(user_id), "source": int(source), "dense_rows": int(rows)}


def scan_raw_index(
    candidates: pd.DataFrame,
    raw_dir: Path = common.RAW_OUT_DIR,
    workers: int = 8,
    limit: int | None = None,
) -> pd.DataFrame:
    """Count raw channel-3000 rows for candidate user/source pairs.

    The scan is a metadata gate, not eligibility.  It reads only three columns
    and never emits participant-level event values.
    """
    tasks = [
        (int(r.user_id), int(r.source), str(raw_dir / f"{int(r.user_id)}.csv"))
        for r in candidates.itertuples(index=False)
    ]
    if limit is not None:
        tasks = tasks[: int(limit)]
    rows: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for result in ex.map(_scan_one, tasks, chunksize=max(1, len(tasks) // (workers * 4) + 1)):
                if result is not None:
                    rows.append(result)
    else:
        for task in tasks:
            result = _scan_one(task)
            if result is not None:
                rows.append(result)
    return pd.DataFrame(rows, columns=["user_id", "source", "dense_rows"])


def _scan_all_one(args: tuple[str, tuple[int, ...]]) -> list[dict[str, Any]]:
    """One-pass scan of a raw CSV for channel-3000 rows in scope sources."""
    path_str, sources = args
    path = Path(path_str)
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return []
            try:
                i_type = header.index("type")
                i_source = header.index("source")
                i_customer = header.index("customer")
            except ValueError:
                return []
            counts: dict[tuple[int, int], int] = {}
            for row in reader:
                try:
                    if int(row[i_type]) != common.DENSE_CHANNEL:
                        continue
                    src = int(row[i_source])
                    if src not in sources:
                        continue
                    uid = int(row[i_customer])
                except (TypeError, ValueError, IndexError):
                    continue
                counts[(uid, src)] = counts.get((uid, src), 0) + 1
    except (OSError, UnicodeDecodeError, csv.Error):
        return []
    return [
        {"user_id": uid, "source": src, "dense_rows": n, "raw_path": str(path)}
        for (uid, src), n in counts.items()
    ]


def scan_raw_index_all(
    raw_dir: Path = common.RAW_OUT_DIR,
    sources: Iterable[int] = common.SCOPE_SOURCES,
    workers: int = 8,
    limit: int | None = None,
) -> pd.DataFrame:
    """Scan every raw CSV once for channel-3000 users in scope sources.

    This is the exhaustive cohort source for sources 2 and 19, which are absent
    from the epoch metadata.  It counts rows only; no event values are emitted.
    """
    paths = sorted(Path(raw_dir).glob("*.csv"))
    if limit is not None:
        paths = paths[: int(limit)]
    tasks = [(str(p), tuple(int(s) for s in sources)) for p in paths]
    rows: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for result in ex.map(_scan_all_one, tasks, chunksize=max(1, len(tasks) // (workers * 4) + 1)):
                rows.extend(result)
    else:
        for task in tasks:
            rows.extend(_scan_all_one(task))
    out = pd.DataFrame(rows, columns=["user_id", "source", "dense_rows", "raw_path"])
    if out.empty:
        return out
    return (
        out.groupby(["user_id", "source"], as_index=False)
        .agg({"dense_rows": "sum", "raw_path": "first"})
        .sort_values(["source", "user_id"])
        .reset_index(drop=True)
    )


def build_raw_index(
    candidates: pd.DataFrame,
    raw_dir: Path = common.RAW_OUT_DIR,
    sources: Iterable[int] = common.SCOPE_SOURCES,
    workers: int = 8,
    limit: int | None = None,
    exhaustive: bool = True,
) -> pd.DataFrame:
    """Choose the raw-index strategy.

    ``exhaustive`` scans every raw file once, which is required to avoid
    conditioning Analysis A on daily-endpoint availability for sources 2 and
    19.  Candidates for sources with known upstream caches are unioned in so
    cached users whose raw export is unavailable are not lost.
    """
    sources = tuple(int(s) for s in sources)
    if exhaustive:
        raw = scan_raw_index_all(raw_dir, sources=sources, workers=workers, limit=limit)
        extra = candidates.loc[candidates["source"].isin((3, 6, 7)), ["user_id", "source"]].copy()
        if raw.empty:
            out = extra
            out["raw_path"] = ""
            out["dense_rows"] = np.nan
        else:
            left = raw[["user_id", "source", "dense_rows", "raw_path"]]
            out = pd.concat([left, extra], ignore_index=True, sort=False)
        out["raw_path"] = out["raw_path"].fillna("")
        out = out.drop_duplicates(["user_id", "source"]).sort_values(["source", "user_id"])
        return out.reset_index(drop=True)
    return scan_raw_index(candidates, raw_dir=raw_dir, workers=workers, limit=limit)


def _build_one(args: tuple[int, int, str, str]) -> dict[str, Any] | None:
    """Build or reuse a source-tagged 5-minute cache for one user/source pair."""
    user_id, source, raw_path, cache_dir = args
    raw = Path(raw_path) if raw_path else common.RAW_OUT_DIR / f"{int(user_id)}.csv"
    existing = resolve_cache_path(int(user_id), int(source), Path(cache_dir))
    if existing is not None:
        try:
            with np.load(existing) as d:
                n_dates = int(len(d["date"]))
            return {
                "user_id": int(user_id),
                "source": int(source),
                "n_dates": n_dates,
                "cache_path": str(existing),
                "cached": True,
            }
        except Exception:
            # A corrupt tagged cache may be rebuilt; a corrupt upstream cache
            # is left untouched and treated as unavailable.
            if Path(existing).parent == Path(cache_dir):
                Path(existing).unlink(missing_ok=True)
    out_npz = Path(cache_dir) / f"customer_{int(user_id)}_source_{int(source)}.npz"
    if not raw.exists():
        return None
    try:
        df = pd.read_csv(
            raw,
            usecols=["customer", "startTimestamp", "timezoneOffset", "longValue", "source", "type"],
        )
    except (ValueError, OSError, UnicodeDecodeError):
        return None
    for col in ("source", "type", "longValue", "startTimestamp", "timezoneOffset"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ok = (
        df["source"].eq(int(source))
        & df["type"].eq(common.DENSE_CHANNEL)
        & df["longValue"].between(common.DENSE_VALUE_MIN, common.DENSE_VALUE_MAX)
        & (df["startTimestamp"] > 0)
        & np.isfinite(df["startTimestamp"])
        & np.isfinite(df["timezoneOffset"])
        & np.isfinite(df["longValue"])
    )
    sub = df.loc[ok, ["startTimestamp", "timezoneOffset", "longValue"]]
    if sub.empty:
        return None
    start = sub["startTimestamp"].to_numpy(np.int64)
    tz = sub["timezoneOffset"].to_numpy(np.float64)
    local_ms = start + (tz * 60_000).astype(np.int64)
    date_day = (local_ms // 86_400_000).astype(np.int64)
    five_min = ((local_ms % 86_400_000) // 300_000).astype(np.int64)

    order = np.lexsort((five_min, date_day))
    date_day = date_day[order]
    five_min = five_min[order]
    values = sub["longValue"].to_numpy(np.float64)[order]

    # Collapse to unique (date, bin) means without interpolating.
    pair = date_day.astype(np.int64) * 288 + five_min
    uniq, inverse = np.unique(pair, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    means = sums / counts
    u_date = uniq // 288
    u_bin = uniq % 288

    unique_dates, date_inverse = np.unique(u_date, return_inverse=True)
    occupied = np.bincount(date_inverse)
    keep_dates = unique_dates[occupied >= common.DENSE_MIN_BINS]
    keep_set = set(int(d) for d in keep_dates)
    keep_mask = np.array([int(d) in keep_set for d in u_date], dtype=bool)
    if not keep_mask.any():
        return None
    hr = np.full((len(keep_dates), 288), np.nan, dtype=np.float32)
    date_pos = {int(d): i for i, d in enumerate(keep_dates)}
    kept_rows = np.array(
        [date_pos[int(d)] for d in u_date[keep_mask]], dtype=np.int64
    )
    hr[kept_rows, u_bin[keep_mask]] = means[keep_mask]

    # Precompute the daily HRp10 endpoint once per participant.  Stage 2 and
    # the matched stage previously re-evaluated the 10th percentile from the
    # 288-bin matrix on every read.  Store float64 so the cached value is
    # bit-compatible with the on-the-fly percentile path.
    finite_counts = np.isfinite(hr).sum(axis=1)
    hrp10 = np.full(len(keep_dates), np.nan, dtype=np.float64)
    dense = finite_counts >= common.DENSE_MIN_BINS
    if np.any(dense):
        # ``hr`` has NaNs for unobserved bins.  The fallback path removes
        # those bins before calling ``np.percentile``; ``nanpercentile`` is
        # the vectorized equivalent and is bit-identical on this array.
        hrp10[dense] = np.nanpercentile(
            hr[dense], common.HRP10_PERCENTILE, axis=1
        )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz, date=keep_dates, hr=hr, hrp10=hrp10,
        occupied_bins=occupied[occupied >= common.DENSE_MIN_BINS],
    )
    return {
        "user_id": int(user_id),
        "source": int(source),
        "n_dates": int(len(keep_dates)),
        "cache_path": str(out_npz),
        "cached": False,
    }


def build_caches(
    candidates: pd.DataFrame,
    cache_dir: Path,
    raw_dir: Path = common.RAW_OUT_DIR,
    workers: int = 8,
    limit: int | None = None,
) -> pd.DataFrame:
    """Build source-tagged caches; returns a cache index."""
    # A user with fewer total channel rows than the per-day dense-bin gate
    # cannot produce any eligible date; skip the expensive raw read.
    if "dense_rows" in candidates.columns:
        enough = candidates["dense_rows"].isna() | (
            pd.to_numeric(candidates["dense_rows"], errors="coerce") >= common.DENSE_MIN_BINS
        )
        candidates = candidates.loc[enough]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            int(r.user_id),
            int(r.source),
            str(getattr(r, "raw_path", "") or (raw_dir / f"{int(r.user_id)}.csv")),
            str(cache_dir),
        )
        for r in candidates.itertuples(index=False)
    ]
    if limit is not None:
        tasks = tasks[: int(limit)]
    rows: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for result in ex.map(_build_one, tasks, chunksize=max(1, len(tasks) // (workers * 4) + 1)):
                if result is not None:
                    rows.append(result)
    else:
        for task in tasks:
            result = _build_one(task)
            if result is not None:
                rows.append(result)
    out = pd.DataFrame(rows, columns=["user_id", "source", "n_dates", "cache_path", "cached"])
    if out.empty:
        return out
    return out.sort_values(["source", "user_id"]).reset_index(drop=True)


def resolve_cache_path(user_id: int, source: int, cache_dir: Path) -> Path | None:
    """Resolve a source-specific cache without cross-source collision."""
    tagged = Path(cache_dir) / f"customer_{int(user_id)}_source_{int(source)}.npz"
    if tagged.exists():
        return tagged
    upstream = UPSTREAM_CACHE_DIRS.get(int(source))
    if upstream is not None:
        legacy = upstream / f"customer_{int(user_id)}.npz"
        if legacy.exists():
            return legacy
    return None


def _read_cache_rows(args: tuple[int, int, str]) -> list[dict[str, Any]] | None:
    user_id, source, path_str = args
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with np.load(path) as d:
            dates = d["date"].astype(np.int64)
            if "hrp10" in d.files:
                values = d["hrp10"].astype(np.float64)
            else:
                hr = d["hr"]
                values = None
    except Exception:
        return None
    if values is None:
        values = np.full(len(dates), np.nan, dtype=np.float64)
        for i in range(len(dates)):
            row = hr[i]
            finite = np.isfinite(row)
            if finite.sum() >= common.DENSE_MIN_BINS:
                values[i] = float(np.percentile(row[finite], common.HRP10_PERCENTILE))
    keep = np.isfinite(values)
    return [
        {
            "user_id": int(user_id),
            "source": int(source),
            "endpoint": "HRp10",
            "date": int(day),
            "value": float(value),
        }
        for day, value in zip(dates[keep], values[keep])
    ]


def cache_daily_frames(
    cache_index: pd.DataFrame,
    cache_dir: Path,
    workers: int = 8,
) -> pd.DataFrame:
    """Read caches into a daily HRp10 frame for window selection."""
    tasks = []
    for r in cache_index.itertuples(index=False):
        path = resolve_cache_path(int(r.user_id), int(r.source), Path(cache_dir))
        if path is not None:
            tasks.append((int(r.user_id), int(r.source), str(path)))
    rows: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for result in ex.map(_read_cache_rows, tasks, chunksize=max(1, len(tasks) // (workers * 4) + 1)):
                if result:
                    rows.extend(result)
    else:
        for task in tasks:
            result = _read_cache_rows(task)
            if result:
                rows.extend(result)
    return pd.DataFrame(rows, columns=["user_id", "source", "endpoint", "date", "value"])


def _build_source_windows(args: tuple[pd.DataFrame, int, tuple[str, ...]]
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    group, source, protocols = args
    return daily.build_windows_for_arm(group, int(source), "HRp10", protocols)


def build_channel_manifests(
    cache_index: pd.DataFrame,
    demographics: pd.DataFrame,
    cache_dir: Path,
    protocols: Iterable[str] = common.PROTOCOLS,
    workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Analysis A manifests from source-specific channel-3000 caches."""
    frames = cache_daily_frames(cache_index, cache_dir, workers=workers)
    if frames.empty:
        return pd.DataFrame(), pd.DataFrame()
    protocols = tuple(protocols)
    jobs = []
    for source, group in frames.groupby("source", sort=True):
        users = np.sort(group["user_id"].unique())
        n_parts = max(1, min(int(workers), int(np.ceil(len(users) / 600))))
        for part in np.array_split(users, n_parts):
            if len(part) == 0:
                continue
            jobs.append((group.loc[group["user_id"].isin(part)], int(source), protocols))
    results = []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_build_source_windows, jobs, chunksize=1))
    else:
        results = [_build_source_windows(job) for job in jobs]
    windows_frames = [r[0] for r in results if not r[0].empty]
    selected_frames = [r[1] for r in results if not r[1].empty]
    windows = pd.concat(windows_frames, ignore_index=True) if windows_frames else pd.DataFrame()
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    if not windows.empty:
        windows = common.attach_ages(windows, demographics, start_col="start_day")
    return windows, selected


def process_channel_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Worker entry point for one channel-3000 participant task."""
    engine = common.load_endpoint_engine()
    _set_task_fitter(task)
    return engine.process_participant_task(task) or []


def _set_task_fitter(task: dict[str, Any]) -> None:
    fitter = reml.active_fitter()
    if fitter is not None:
        task["reml_fitter"] = fitter


def write_channel_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    """Persist one channel arm's endpoints, diagnostics, and spectra."""
    engine = common.load_endpoint_engine()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra = [r.pop("spectra") for r in rows if "spectra" in r]
    diagnostics = [r.pop("null_diagnostics") for r in rows if "null_diagnostics" in r]
    endpoints = pd.DataFrame(rows)
    if not endpoints.empty:
        endpoints["endpoint"] = "HRp10"
        if "null_status" not in endpoints.columns:
            endpoints["null_status"] = "failed"
        endpoints["null_converged"] = endpoints["null_status"].eq("converged")
        failure = endpoints["status"] if "status" in endpoints.columns else "unknown"
        endpoints["null_failure_reason"] = np.where(
            endpoints["null_converged"], "", failure
        )
    endpoints.to_parquet(output_dir / "endpoints.parquet", index=False)
    if spectra:
        np.savez_compressed(
            output_dir / "spectra_broad.npz",
            broad_freqs=engine.BROAD_FREQS,
            P_obs=np.stack([s["P_obs"] for s in spectra]),
            P_null_mean=np.stack([s["P_null_mean"] for s in spectra]),
            P_null_var=np.stack([s["P_null_var"] for s in spectra]),
            D_excess=np.stack([s["D_excess"] for s in spectra]),
        )
    if diagnostics:
        pd.DataFrame(diagnostics).to_parquet(output_dir / "null_diagnostics.parquet", index=False)
    return endpoints


def run_channel_arm(
    tasks: list[dict[str, Any]],
    output_dir: Path,
    null_replicates: int = common.B_NULL_REPLICATES,
    workers: int = 8,
) -> pd.DataFrame:
    """Run Analysis A through the historical channel-3000 endpoint engine."""
    for task in tasks:
        task["null_replicates"] = int(null_replicates)
        task["save_daily_values"] = True
        _set_task_fitter(task)
    rows: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        chunk = max(1, len(tasks) // (workers * 4) + 1)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for result in ex.map(process_channel_task, tasks, chunksize=chunk):
                if result:
                    rows.extend(result)
    else:
        for task in tasks:
            result = process_channel_task(task)
            if result:
                rows.extend(result)
    return write_channel_outputs(rows, output_dir)


def build_channel_tasks(
    selected: pd.DataFrame,
    windows: pd.DataFrame,
    demographics: pd.DataFrame,
    cache_dir: Path,
    protocol: str,
    percentile: float = common.HRP10_PERCENTILE,
    limit: int | None = None,
    window_indices: tuple[int, ...] = (1,),
    source: int | None = None,
) -> list[dict[str, Any]]:
    """Build tasks for the historical channel-3000 endpoint engine.

    ``source`` must be supplied when callers loop per source; otherwise the
    task list would contain every source in the run.
    """
    sel = selected.loc[selected["protocol"].eq(protocol)].copy()
    if source is not None:
        sel = sel.loc[sel["source"].eq(int(source))]
    demos = demographics.set_index("user_id").to_dict("index")
    win_mask = windows["is_eligible"] & windows["protocol"].eq(protocol)
    if source is not None:
        win_mask &= windows["source"].eq(int(source))
    eligible_starts = windows.loc[win_mask].set_index(
        ["user_id", "source", "window_index"]
    )["start_day"].to_dict()
    tasks: list[dict[str, Any]] = []
    for (uid, src), group in sel.groupby(["user_id", "source"], sort=True):
        uid, src = int(uid), int(src)
        cache_path = resolve_cache_path(uid, src, Path(cache_dir))
        if cache_path is None or uid not in demos:
            continue
        row = demos[uid]
        birth_year = int(row["birth_year"]) if pd.notna(row["birth_year"]) else None
        for (proto, widx, digest), win in group.groupby(
            ["protocol", "window_index", "selected_dates_hash"], sort=True
        ):
            widx = int(widx)
            if widx not in window_indices or (uid, src, widx) not in eligible_starts:
                continue
            start_day = int(eligible_starts[(uid, src, widx)])
            if birth_year is not None:
                age_dec1, age_jan1, age_dec31 = common.compute_all_ages(start_day, birth_year)
            else:
                age_dec1 = age_jan1 = age_dec31 = None
            demo = {
                "salutation": int(row["salutation"]) if pd.notna(row["salutation"]) else None,
                "birth_year": birth_year,
                "age_dec1": int(age_dec1) if age_dec1 is not None else None,
                "age_jan1": int(age_jan1) if age_jan1 is not None else None,
                "age_dec31": int(age_dec31) if age_dec31 is not None else None,
                "weight": float(row["weight"]) if pd.notna(row.get("weight")) else np.nan,
                "height": float(row["height"]) if pd.notna(row.get("height")) else np.nan,
                "bmi": float(row["bmi"]) if pd.notna(row.get("bmi")) else np.nan,
            }
            date_arr = np.sort(win["date"].to_numpy(dtype=np.int64))
            task = {
                "user_id": uid,
                "source": src,
                "cache_path": str(cache_path),
                "demographics": demo,
                "windows": [{
                    "protocol": str(proto),
                    "window_index": widx,
                    "start_day": start_day,
                    "selected_dates_hash": str(digest),
                    "dates": date_arr.tolist(),
                }],
                "percentile": float(percentile),
                "arm": "HRp10",
            }
            _set_task_fitter(task)
            tasks.append(task)
    return tasks[: int(limit)] if limit is not None else tasks
