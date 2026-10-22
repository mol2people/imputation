"""Stage 5: matched endpoint comparisons.

Implements the four explicit paired-comparison requirements from PLAN.md:
common converged participant set, age support recomputed on that set, identical
bootstrap participant draws, and arm-specific failure/retention reporting with
paired-difference intervals.
"""
from __future__ import annotations

import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import bootstrap, channel3000, common, endpoints, models, reml

PAIR_KEYS = ["user_id", "source", "protocol", "window_index"]


def _channel_values(cache_path: str, dates: np.ndarray) -> dict[int, float]:
    with np.load(cache_path) as d:
        date_arr = d["date"].astype(int)
        if "hrp10" in d.files:
            hrp10 = d["hrp10"].astype(float)
            hr = None
        else:
            hr = d["hr"]
            hrp10 = None
    row = {int(day): i for i, day in enumerate(date_arr)}
    out: dict[int, float] = {}
    for day in dates:
        idx = row.get(int(day))
        if idx is None:
            continue
        if hrp10 is not None:
            value = float(hrp10[idx])
            if np.isfinite(value):
                out[int(day)] = value
            continue
        vals = hr[idx]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= common.DENSE_MIN_BINS:
            out[int(day)] = float(np.percentile(vals, common.HRP10_PERCENTILE))
    return out


def _load_channel_values(payload: tuple[str, np.ndarray]) -> dict[int, float]:
    """Worker entry point for P2 parallel cache decompression."""
    cache_path, dates = payload
    if not cache_path:
        return {}
    return _channel_values(cache_path, dates)


def _load_channel_values_many(
    payloads: list[tuple[str, np.ndarray]], workers: int
) -> list[dict[int, float]]:
    """Decompress per-participant HR caches across the worker pool.

    The prior serial loop paid one ``np.load``/``zipfile`` open per participant
    inside the main process, so the pool sat idle during the whole stage.
    """
    if not payloads:
        return []
    if workers > 1 and len(payloads) > 1:
        chunk = max(1, len(payloads) // (int(workers) * 4) + 1)
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            return list(ex.map(_load_channel_values, payloads, chunksize=chunk))
    return [_load_channel_values(payload) for payload in payloads]


def exact_date_cohort(
    channel_selected: pd.DataFrame,
    channel_windows: pd.DataFrame,
    daily_values: pd.DataFrame,
    cache_dir: Path,
    source: int,
    protocol: str,
    window_index: int = 1,
    workers: int = 8,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Build exact-date complete-case tasks for one source/protocol/window.

    Returns (tasks, retention table, matched daily values).
    """
    sel = channel_selected.loc[
        channel_selected["source"].eq(int(source))
        & channel_selected["protocol"].eq(protocol)
        & channel_selected["window_index"].eq(int(window_index))
    ].copy()
    if sel.empty:
        return [], pd.DataFrame(), pd.DataFrame()
    starts = channel_windows.loc[
        channel_windows["source"].eq(int(source))
        & channel_windows["protocol"].eq(protocol)
        & channel_windows["is_eligible"]
    ].set_index(["user_id", "window_index"])["start_day"].to_dict()
    daily_map = (
        daily_values.set_index(["user_id", "date"])["value"].astype(float).to_dict()
    )

    # Phase 1: enumerate work and resolve cache paths in the parent process.
    plans: list[tuple[int, int, str, np.ndarray, int, str]] = []
    payloads: list[tuple[str, np.ndarray]] = []
    for (uid, widx, digest), group in sel.groupby(
        ["user_id", "window_index", "selected_dates_hash"], sort=True
    ):
        uid, widx = int(uid), int(widx)
        dates = np.sort(group["date"].to_numpy(dtype=np.int64))
        start = starts.get((uid, widx))
        if start is None:
            continue
        cache_path = channel3000.resolve_cache_path(uid, int(source), Path(cache_dir))
        path_str = str(cache_path) if cache_path else ""
        plans.append((uid, widx, str(digest), dates, int(start), path_str))
        payloads.append((path_str, dates))

    # Phase 2: decompress HR bins in parallel (P2).
    hr_values = _load_channel_values_many(payloads, workers)

    # Phase 3: assemble retention, matched values, and tasks in the original
    # group order so that downstream task ordering is unchanged.
    tasks: list[dict[str, Any]] = []
    retention: list[dict[str, Any]] = []
    matched_values: list[dict[str, Any]] = []
    for (uid, widx, digest, dates, start, path_str), hrp10 in zip(plans, hr_values):
        daily = {int(d): daily_map.get((uid, int(d)), np.nan) for d in dates}
        n_daily = sum(np.isfinite(list(daily.values())))
        n_hr = len(hrp10)
        complete = (n_daily == len(dates)) and (n_hr == len(dates))
        retention.append({
            "user_id": uid, "source": int(source), "protocol": protocol,
            "window_index": widx, "n_dates": int(len(dates)),
            "n_daily_finite": int(n_daily), "n_hrp10_finite": int(n_hr),
            "exact_date_complete": bool(complete),
            "cache_path": path_str,
        })
        if not complete:
            continue
        matched_values.extend(
            {"user_id": uid, "source": int(source), "protocol": protocol,
             "window_index": widx, "date": int(d), "hrp10": float(hrp10[int(d)]),
             "daily": float(daily[int(d)])}
            for d in dates
        )
        tasks.append({
            "user_id": uid, "source": int(source), "start_day": int(start),
            "protocol": protocol, "window_index": widx,
            "selected_dates_hash": str(digest), "dates": dates.tolist(),
            "hrp10": {int(d): float(hrp10[int(d)]) for d in dates},
            "daily": {int(d): float(daily[int(d)]) for d in dates},
        })
    return tasks, pd.DataFrame(retention), pd.DataFrame(matched_values)


def _run_arm_on_matched(
    tasks: list[dict[str, Any]],
    arm: str,
    value_key: str,
    demographics: pd.DataFrame,
    null_replicates: int,
    workers: int,
) -> pd.DataFrame:
    demos = demographics.set_index("user_id").to_dict("index")
    work: list[dict[str, Any]] = []
    for task in tasks:
        row = demos.get(int(task["user_id"]))
        if row is None:
            continue
        birth_year = int(row["birth_year"]) if pd.notna(row["birth_year"]) else None
        if birth_year is not None:
            age_dec1, age_jan1, age_dec31 = common.compute_all_ages(int(task["start_day"]), birth_year)
        else:
            age_dec1 = age_jan1 = age_dec31 = None
        # Matched tasks bypass the manifest age filter; enforce the same
        # plausibility range before fitting or bootstrapping.
        if age_dec1 is None or not (
            common.AGE_PLAUSIBLE_MIN <= int(age_dec1) <= common.AGE_PLAUSIBLE_MAX
        ):
            continue
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
        work.append({
            "user_id": int(task["user_id"]), "source": int(task["source"]),
            "endpoint": arm, "arm": "matched", "demographics": demo,
            "shared_null_seed": True, "null_replicates": int(null_replicates),
            "windows": [{
                "protocol": task["protocol"], "window_index": int(task["window_index"]),
                "start_day": int(task["start_day"]),
                "selected_dates_hash": task["selected_dates_hash"],
                "dates": task["dates"], "daily_values": task[value_key],
            }],
        })
    # Stage 7 constructs tasks directly instead of going through stage3.py.
    # Inject the active fitter here; otherwise matched REML silently reverts
    # to the historical Nelder–Mead path.
    fitter = reml.active_fitter()
    if fitter is not None:
        for task in work:
            task.setdefault("reml_fitter", fitter)
    with tempfile.TemporaryDirectory(prefix="ms_matched_") as tmp:
        return endpoints.run_daily_arm(work, Path(tmp), null_replicates, workers)


def _common_support_table(
    df: pd.DataFrame,
    bands: list[str],
    source: int,
    endpoint: str,
    protocol: str,
    window_index: int,
) -> pd.DataFrame:
    """Band-level female/male counts on the common matched set."""
    work = df.copy()
    work["age_band"] = pd.cut(
        work["age"], bins=models.BAND_EDGES, right=False, labels=models.BAND_LABELS
    )
    rows = []
    for band in models.BAND_LABELS:
        sub = work.loc[work["age_band"].eq(band)]
        rows.append({
            "source": int(source), "endpoint": endpoint, "protocol": protocol,
            "window_index": int(window_index), "age_band": band,
            "female_count": int(sub["sex"].eq(1).sum()),
            "male_count": int(sub["sex"].eq(0).sum()),
            "is_shared_support": bool(band in bands),
        })
    return pd.DataFrame(rows)


def run_matched_source(
    channel_selected: pd.DataFrame,
    channel_windows: pd.DataFrame,
    daily_clean: pd.DataFrame,
    demographics: pd.DataFrame,
    cache_dir: Path,
    source: int,
    daily_endpoint: str,
    protocol: str,
    null_replicates: int = common.B_NULL_REPLICATES,
    n_boot: int = common.N_BOOTSTRAP,
    seed: int = 20261020,
    workers: int = 8,
    output_dir: Path | None = None,
    window_index: int = 1,
) -> dict[str, Any]:
    """Run one matched source/daily-endpoint/protocol comparison."""
    daily_arm = daily_clean.loc[
        daily_clean["source"].eq(int(source)) & daily_clean["endpoint"].eq(daily_endpoint)
    ][["user_id", "source", "endpoint", "date", "value"]]
    tasks, retention, matched_values = exact_date_cohort(
        channel_selected, channel_windows, daily_arm, cache_dir, source, protocol,
        window_index=int(window_index), workers=int(workers),
    )
    if not tasks:
        return {"status": "no_exact_date_complete_cases", "source": source,
                "endpoint": daily_endpoint, "protocol": protocol,
                "window_index": int(window_index)}

    hr = _run_arm_on_matched(tasks, "HRp10", "hrp10", demographics, null_replicates, workers)
    daily_ep = _run_arm_on_matched(tasks, daily_endpoint, "daily", demographics, null_replicates, workers)
    hr["arm"] = "HRp10"
    daily_ep["arm"] = daily_endpoint
    hr["matched_key"] = hr[PAIR_KEYS].astype(str).agg("|".join, axis=1)
    daily_ep["matched_key"] = daily_ep[PAIR_KEYS].astype(str).agg("|".join, axis=1)

    # Requirement 1: common participant set with finite endpoints and converged nulls.
    # Finite Q_i is required in both arms so that support, draws, and paired
    # contrasts are computed on exactly the same rows.
    hr_ok = hr.loc[hr["null_converged"].fillna(False) & np.isfinite(hr["Q_i"])]
    daily_ok = daily_ep.loc[
        daily_ep["null_converged"].fillna(False) & np.isfinite(daily_ep["Q_i"])
    ]
    common_keys = set(hr_ok["matched_key"]).intersection(set(daily_ok["matched_key"]))
    common = pd.concat([
        hr_ok.loc[hr_ok["matched_key"].isin(common_keys)],
        daily_ok.loc[daily_ok["matched_key"].isin(common_keys)],
    ], ignore_index=True)

    # Requirement 4: arm-specific failures and retention.
    arm_status = []
    for arm_name, frame, n_input in (
        ("HRp10", hr, len(tasks)), (daily_endpoint, daily_ep, len(tasks))
    ):
        arm_status.append({
            "source": int(source), "endpoint": daily_endpoint, "protocol": protocol,
            "arm": arm_name, "input_tasks": int(n_input),
            "converged": int(frame["null_converged"].fillna(False).sum()),
            "failed": int((~frame["null_converged"].fillna(False)).sum()),
            "finite_endpoint": int(np.isfinite(frame["Q_i"]).sum()),
            "finite_age": int(np.isfinite(frame["age"]).sum()),
            "common_retained": int(len(common_keys)),
        })

    # Requirement 2: age support on the common set only.
    common_first = common.loc[common["arm"].eq("HRp10")].copy()
    support, bands = models.shared_support(common_first)
    support_table = _common_support_table(
        common_first, bands, int(source), daily_endpoint, protocol, int(window_index)
    )

    # Requirement 3: identical bootstrap draws across arms on the common set.
    paired_contrasts = pd.DataFrame()
    if len(common_keys) >= 30 and bands:
        hr_common = common.loc[common["arm"].eq("HRp10")].sort_values(["user_id", "window_index"]).reset_index(drop=True)
        daily_common = common.loc[common["arm"].eq(daily_endpoint)].sort_values(["user_id", "window_index"]).reset_index(drop=True)
        # Align rows explicitly by key.
        daily_common = daily_common.set_index("matched_key").loc[hr_common["matched_key"]].reset_index()
        draws = bootstrap.draw_indices(hr_common, int(n_boot), int(seed))
        hr_boot = bootstrap.bootstrap_frame(
            hr_common, "Q_i", "HRp10", int(source), protocol, 1,
            int(n_boot), int(seed), shared_bands=bands, draws=draws, scope="sex",
            return_curves=False, curve_only=True,
        )
        daily_boot = bootstrap.bootstrap_frame(
            daily_common, "Q_i", daily_endpoint, int(source), protocol, 1,
            int(n_boot), int(seed), shared_bands=bands, draws=draws, scope="sex",
            return_curves=False, curve_only=True,
        )
        paired_contrasts = _paired_difference_arrays(hr_boot, daily_boot,
                                              int(source), daily_endpoint, protocol)

    # Persist.
    out = Path(output_dir) if output_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        retention.to_csv(out / "retention.csv", index=False)
        matched_values.to_parquet(out / "matched_daily_values.parquet", index=False)
        common.to_parquet(out / "matched_endpoints.parquet", index=False)
        pd.DataFrame(arm_status).to_csv(out / "arm_status.csv", index=False)
        support_table.to_csv(out / "common_age_support.csv", index=False)
        paired_contrasts.to_parquet(out / "paired_contrasts.parquet", index=False)
        requirements = {
            "source": int(source), "endpoint": daily_endpoint, "protocol": protocol,
            "window_index": int(window_index),
            "n_exact_date_complete": int(len(tasks)),
            "common_converged": int(len(common_keys)),
            "common_age_bands": bands,
            "bootstrap_draws_identical": True,
            "bootstrap_seed": int(seed),
            "n_boot": int(n_boot),
            "paired_estimable": bool(len(paired_contrasts) > 0),
            "notes": (
                "Common participant set requires finite endpoint and converged "
                "null in every arm. Age support is the intersection. The same "
                "participant draw indices are used in both arms."
            ),
        }
        (out / "requirements.json").write_text(json.dumps(requirements, indent=2) + "\n", encoding="utf-8")

    return {
        "status": "ok" if len(common_keys) else "no_common_converged",
        "source": int(source), "endpoint": daily_endpoint, "protocol": protocol,
        "n_tasks": int(len(tasks)), "common_converged": int(len(common_keys)),
        "age_bands": bands, "paired_rows": int(len(paired_contrasts)),
    }


def _paired_difference(hr_curves: pd.DataFrame, daily_curves: pd.DataFrame,
                       source: int, endpoint: str, protocol: str) -> pd.DataFrame:
    """Paired curve differences from identical bootstrap draws."""
    if hr_curves.empty or daily_curves.empty:
        return pd.DataFrame()
    rows = []
    for scope in ("sex_female", "sex_male"):
        a = hr_curves.loc[hr_curves["scope"].eq(scope)]
        b = daily_curves.loc[daily_curves["scope"].eq(scope)]
        if a.empty or b.empty:
            continue
        point_a = a.loc[a["draw"].eq(-1)].set_index("age")["prediction"]
        point_b = b.loc[b["draw"].eq(-1)].set_index("age")["prediction"]
        ages = point_a.index.to_numpy(float)
        point_b_values = point_b.reindex(ages).to_numpy(float)
        point_a_values = point_a.to_numpy(float)
        valid = np.isfinite(point_a_values) & np.isfinite(point_b_values)
        ages = ages[valid]
        if len(ages) == 0:
            continue
        point_diff = point_a_values[valid] - point_b_values[valid]
        ba = a.loc[a["draw"].ge(0)].pivot_table(index="draw", columns="age", values="prediction")
        bb = b.loc[b["draw"].ge(0)].pivot_table(index="draw", columns="age", values="prediction")
        ba = ba.reindex(columns=ages)
        bb = bb.reindex(columns=ages)
        common_draws = ba.index.intersection(bb.index)
        if len(common_draws) == 0:
            continue
        diff = ba.loc[common_draws].to_numpy(float) - bb.loc[common_draws].to_numpy(float)
        rows.append(pd.DataFrame({
            "source": int(source), "endpoint": endpoint, "metric": "Q_i",
            "protocol": protocol, "scope": scope, "age": ages,
            "difference": point_diff,
            "lo": np.nanpercentile(diff, 2.5, axis=0),
            "hi": np.nanpercentile(diff, 97.5, axis=0),
            "n_boot": int(len(common_draws)),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _paired_difference_arrays(hr: dict[str, Any], daily: dict[str, Any],
                              source: int, endpoint: str, protocol: str) -> pd.DataFrame:
    """Paired curve differences from aligned bootstrap prediction arrays."""
    rows = []
    for scope in ("sex_female", "sex_male"):
        if scope not in hr["arrays"] or scope not in daily["arrays"]:
            continue
        point_a = np.asarray(hr["point"][scope], dtype=float)
        point_b = np.asarray(daily["point"][scope], dtype=float)
        ages = bootstrap.AGE_GRID
        valid = np.isfinite(point_a) & np.isfinite(point_b)
        ages = ages[valid]
        if len(ages) == 0:
            continue
        all_a = np.asarray(hr["arrays"][scope], dtype=float)
        all_b = np.asarray(daily["arrays"][scope], dtype=float)
        if all_a.shape != all_b.shape:
            raise ValueError(
                f"paired bootstrap arrays for {scope} have different shapes: "
                f"{all_a.shape} and {all_b.shape}"
            )
        # Match pivot_table's removal of draws that failed at every age before
        # restricting columns to the common point-estimate support.
        keep_a = np.isfinite(all_a).any(axis=1)
        keep_b = np.isfinite(all_b).any(axis=1)
        common_draws = keep_a & keep_b
        if not common_draws.any():
            continue
        diff = all_a[common_draws][:, valid] - all_b[common_draws][:, valid]
        rows.append(pd.DataFrame({
            "source": int(source), "endpoint": endpoint, "metric": "Q_i",
            "protocol": protocol, "scope": scope, "age": ages,
            "difference": point_a[valid] - point_b[valid],
            "lo": np.nanpercentile(diff, 2.5, axis=0),
            "hi": np.nanpercentile(diff, 97.5, axis=0),
            "n_boot": int(common_draws.sum()),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_stage5(
    run_dir: Path,
    cache_dir: Path,
    sources: list[int] | None = None,
    endpoints_to_match: list[str] | None = None,
    protocols: list[str] | None = None,
    null_replicates: int = common.B_NULL_REPLICATES,
    n_boot: int = common.N_BOOTSTRAP,
    seed: int = 20261020,
    workers: int = 8,
    window_index: int = 1,
) -> dict[str, Any]:
    """Run all feasible matched comparisons and persist a summary."""
    run_dir = Path(run_dir)
    manifests_dir = run_dir / "manifests"
    if not (manifests_dir / "channel_selected_dates.parquet").exists():
        summary = {
            "comparisons": [],
            "n": 0,
            "status": "no_channel_manifests",
        }
        (run_dir / "stage6_matched_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    channel_selected = pd.read_parquet(manifests_dir / "channel_selected_dates.parquet")
    channel_windows = pd.read_parquet(manifests_dir / "channel_windows.parquet")
    daily_clean = pd.read_parquet(run_dir / "daily_clean.parquet")
    demographics = pd.read_parquet(manifests_dir / "participants.parquet")
    sources = sources or sorted(int(s) for s in channel_selected["source"].unique())
    endpoints_to_match = endpoints_to_match or ["RHR", "HR_during_sleep"]
    protocols = protocols or list(common.PROTOCOLS)

    results = []
    for source in sources:
        for endpoint in endpoints_to_match:
            if daily_clean.loc[
                daily_clean["source"].eq(int(source)) & daily_clean["endpoint"].eq(endpoint)
            ].empty:
                continue
            for protocol in protocols:
                out = run_dir / "matched" / f"source_{source}" / endpoint / protocol
                result = run_matched_source(
                    channel_selected, channel_windows, daily_clean, demographics,
                    Path(cache_dir), int(source), str(endpoint), str(protocol),
                    null_replicates=null_replicates, n_boot=n_boot, seed=seed,
                    workers=workers, output_dir=out, window_index=int(window_index),
                )
                results.append(result)
    summary = {"comparisons": results, "n": len(results), "window_index": int(window_index)}
    (run_dir / "stage6_matched_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
