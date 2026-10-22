"""Repeat-window (longitudinal) summaries for the multisource experiment.

The repeat window is a second independently selected 180-day window for the
same participant, source, endpoint, and protocol.  It is *not* a second
independent participant and it does not identify a causal aging effect.  This
module estimates paired within-participant change, annualized change, and
window-to-window endpoint association for participants with two eligible,
converged windows.

Only observed endpoints are used.  There is no imputation.  Participant
bootstrap resampling is stratified by five-year age band and sex; the same
participant draw is reused across change summaries in one subgroup, while each
correlation uses metric-specific draws after excluding missing values.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import common

PAIR_METRICS = ("Q_i", "E_i2", "A_hat", "sigma_resid", "peak_period")
AGE_EDGES = (20, 35, 42, 47, 52, 60, 75)
AGE_LABELS = ("20-35", "35-42", "42-47", "47-52", "52-60", "60-75")
MIN_PAIR_N = 3


def _seed(*parts: object) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def _draws(
    age: np.ndarray, sex: np.ndarray, n_boot: int, seed: int
) -> Iterable[np.ndarray]:
    """Participant draws stratified by five-year age band and sex."""
    age_band = np.floor(np.asarray(age, dtype=float) / 5.0).astype(int)
    cell = age_band * 2 + np.asarray(sex, dtype=int)
    cells = [np.flatnonzero(cell == c) for c in np.unique(cell)]
    rng = np.random.default_rng(seed)
    for _ in range(int(n_boot)):
        yield np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in cells])


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return np.nan
    x = x[ok]
    y = y[ok]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _percentile_ci(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return np.nan, np.nan
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


def _groups(age: np.ndarray, sex: np.ndarray):
    """Report groups: all, sex strata, and age bands; age restricted to [20,75)."""
    age = np.asarray(age, dtype=float)
    sex = np.asarray(sex, dtype=float)
    base = (
        np.isfinite(age)
        & np.isfinite(sex)
        & (age >= AGE_EDGES[0])
        & (age < AGE_EDGES[-1])
    )
    yield "all", "20-75", "All", base
    codes = np.asarray(pd.cut(
        age, AGE_EDGES, labels=False, right=False, include_lowest=True
    ))
    for code, label in enumerate(AGE_LABELS):
        yield "age_bin", label, "All", base & (codes == code)
    yield "sex", "Female", "Female", base & (sex == 1)
    yield "sex", "Male", "Male", base & (sex == 0)


def _bootstrap_corr(
    x1: np.ndarray,
    x2: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, int]:
    ok = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(age) & np.isfinite(sex)
    if int(ok.sum()) < 3:
        return np.nan, np.nan, np.nan, 0
    x1, x2, age, sex = x1[ok], x2[ok], age[ok], sex[ok]
    point = _corr(x1, x2)
    boot = []
    for draw in _draws(age, sex, n_boot, seed):
        boot.append(_corr(x1[draw], x2[draw]))
    lo, hi = _percentile_ci(boot)
    valid = int(np.isfinite(np.asarray(boot, dtype=float)).sum())
    return point, lo, hi, valid


def _summarize_arm(
    group: pd.DataFrame,
    *,
    source: int,
    endpoint: str,
    protocol: str,
    n_boot: int,
    seed: int,
) -> list[dict]:
    q1 = group["Q_i_w1"].to_numpy(float)
    q2 = group["Q_i_w2"].to_numpy(float)
    age = group["age_w1"].to_numpy(float)
    sex = group["sex_w1"].to_numpy(float)
    gap = group["gap_years"].to_numpy(float)
    change = q2 - q1
    rate = change / gap
    rows: list[dict] = []

    for subgroup, label, sex_label, mask in _groups(age, sex):
        ix = np.flatnonzero(mask)
        if len(ix) < MIN_PAIR_N:
            continue
        g_age, g_sex, g_gap = age[ix], sex[ix], gap[ix]
        g_q1, g_q2, g_change, g_rate = q1[ix], q2[ix], change[ix], rate[ix]

        boot_change: list[float] = []
        boot_rate: list[float] = []
        boot_q1: list[float] = []
        boot_q2: list[float] = []
        for draw in _draws(g_age, g_sex, n_boot, _seed("long", source, endpoint, protocol, subgroup, label)):
            boot_change.append(float(g_change[draw].mean()))
            boot_rate.append(float(g_rate[draw].mean()))
            boot_q1.append(float(g_q1[draw].mean()))
            boot_q2.append(float(g_q2[draw].mean()))
        q_change_lo, q_change_hi = _percentile_ci(boot_change)
        q_rate_lo, q_rate_hi = _percentile_ci(boot_rate)

        row: dict[str, object] = {
            "source": int(source),
            "source_name": common.source_name(int(source)),
            "endpoint": endpoint,
            "protocol": protocol,
            "subgroup": subgroup,
            "group": label,
            "sex": sex_label,
            "n": int(len(ix)),
            "n_female": int((g_sex == 1).sum()),
            "n_male": int((g_sex == 0).sum()),
            "mean_age_w1": float(np.nanmean(g_age)),
            "mean_gap_years": float(np.nanmean(g_gap)),
            "median_gap_years": float(np.nanmedian(g_gap)),
            "mean_Q_w1": float(np.nanmean(g_q1)),
            "mean_Q_w2": float(np.nanmean(g_q2)),
            "mean_Q_i_mc_se_w1": float(np.nanmean(group["Q_i_mc_se_w1"].to_numpy(float)[ix])),
            "mean_Q_i_mc_se_w2": float(np.nanmean(group["Q_i_mc_se_w2"].to_numpy(float)[ix])),
            "mean_Q_change": float(np.nanmean(g_change)),
            "median_Q_change": float(np.nanmedian(g_change)),
            "mean_Q_annualized_rate": float(np.nanmean(g_rate)),
            "Q_change_ci_low": q_change_lo,
            "Q_change_ci_high": q_change_hi,
            "Q_rate_ci_low": q_rate_lo,
            "Q_rate_ci_high": q_rate_hi,
            "Q_n_boot_valid": len(boot_change),
            "n_boot": int(n_boot),
            "stable": bool(len(ix) >= 30 and len(boot_change) >= 100),
        }

        # Metric-specific correlations use participants with both values finite.
        for metric in PAIR_METRICS:
            x1 = group[f"{metric}_w1"].to_numpy(float)[ix]
            x2 = group[f"{metric}_w2"].to_numpy(float)[ix]
            point, lo, hi, valid = _bootstrap_corr(
                x1, x2, g_age, g_sex, n_boot,
                _seed("corr", source, endpoint, protocol, subgroup, label, metric),
            )
            row[f"pearson_{metric}"] = point
            row[f"pearson_{metric}_ci_low"] = lo
            row[f"pearson_{metric}_ci_high"] = hi
            row[f"n_{metric}"] = int((np.isfinite(x1) & np.isfinite(x2)).sum())
            row[f"{metric}_n_boot_valid"] = valid
        rows.append(row)
    return rows


def build_paired_windows(endpoints: pd.DataFrame) -> pd.DataFrame:
    """Return one row per participant with two converged, distinct windows.

    The function raises if a pair violates the prespecified repeat-window rule
    or if participant metadata disagree between windows.  Invalid pairs are not
    repaired.
    """
    need = {
        "user_id", "source", "endpoint", "protocol", "window_index", "start_day",
        "selected_dates_hash", "age", "sex", "null_status", "Q_i", "Q_i_mc_se",
        *PAIR_METRICS,
    }
    missing = need - set(endpoints.columns)
    if missing:
        raise ValueError(f"endpoint table lacks longitudinal fields: {sorted(missing)}")

    keep_status = np.ones(len(endpoints), dtype=bool)
    if "status" in endpoints:
        keep_status = endpoints["status"].eq("ok").to_numpy()
    base = endpoints.loc[
        keep_status
        & endpoints["window_index"].isin([1, 2]).to_numpy()
        & endpoints["null_status"].eq("converged").to_numpy()
        & np.isfinite(endpoints["Q_i"].to_numpy(float))
    ].copy()
    frames: list[pd.DataFrame] = []
    for (source, endpoint, protocol), arm in base.groupby(
        ["source", "endpoint", "protocol"], sort=True
    ):
        w1 = arm.loc[arm["window_index"].eq(1)].copy()
        w2 = arm.loc[arm["window_index"].eq(2)].copy()
        if w1.empty or w2.empty:
            continue
        if w1["user_id"].duplicated().any() or w2["user_id"].duplicated().any():
            raise ValueError(f"duplicate paired endpoint rows for {source}/{endpoint}/{protocol}")
        keep = [
            "user_id", "start_day", "selected_dates_hash", "age", "sex",
            "Q_i", "Q_i_mc_se", "E_i2", "A_hat", "sigma_resid", "peak_period",
        ]
        left = w1[keep].rename(columns={c: f"{c}_w1" for c in keep if c != "user_id"})
        right = w2[keep].rename(columns={c: f"{c}_w2" for c in keep if c != "user_id"})
        merged = left.merge(right, on="user_id", how="inner", validate="one_to_one")
        if merged.empty:
            continue

        gap_days = merged["start_day_w2"].to_numpy(float) - merged["start_day_w1"].to_numpy(float)
        if np.any(gap_days < 180):
            raise ValueError(
                f"repeat window starts less than 180 days after the first for {source}/{endpoint}/{protocol}"
            )
        hashes1 = merged["selected_dates_hash_w1"].astype("string")
        hashes2 = merged["selected_dates_hash_w2"].astype("string")
        if hashes1.isna().any() or hashes2.isna().any():
            raise ValueError(f"missing selected-date hash for {source}/{endpoint}/{protocol}")
        if hashes1.eq(hashes2).any():
            raise ValueError(f"repeat window reuses first-window dates for {source}/{endpoint}/{protocol}")

        sex1 = pd.to_numeric(merged["sex_w1"], errors="coerce")
        sex2 = pd.to_numeric(merged["sex_w2"], errors="coerce")
        bad_sex = sex1.notna() & sex2.notna() & sex1.ne(sex2)
        if bad_sex.any():
            raise ValueError(f"sex changes between windows for {source}/{endpoint}/{protocol}")

        merged = merged.loc[~bad_sex].copy()
        merged["source"] = int(source)
        merged["source_name"] = common.source_name(int(source))
        merged["endpoint"] = str(endpoint)
        merged["protocol"] = str(protocol)
        merged["sex_w1"] = sex1.loc[~bad_sex].to_numpy(float)
        merged["gap_days"] = gap_days[~bad_sex.to_numpy()]
        merged["gap_years"] = merged["gap_days"] / 365.25
        merged["Q_i_change"] = merged["Q_i_w2"].to_numpy(float) - merged["Q_i_w1"].to_numpy(float)
        merged["Q_i_annualized_rate"] = merged["Q_i_change"] / merged["gap_years"]
        merged["age_w1"] = pd.to_numeric(merged["age_w1"], errors="coerce")
        merged["age_w2"] = pd.to_numeric(merged["age_w2"], errors="coerce")
        frames.append(merged)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    metric_order = ["Q_i", "E_i2", "A_hat", "sigma_resid", "peak_period"]
    front = [
        "source", "source_name", "endpoint", "protocol", "user_id",
        "age_w1", "age_w2", "sex_w1", "start_day_w1", "start_day_w2",
        "gap_days", "gap_years", "Q_i_change", "Q_i_annualized_rate",
    ]
    metric_cols = [f"{m}_{w}" for m in metric_order for w in ("w1", "w2")]
    rest = [c for c in out.columns if c not in set(front + metric_cols)]
    return out[front + metric_cols + rest].sort_values(
        ["source", "endpoint", "protocol", "user_id"]
    ).reset_index(drop=True)


def availability_table(endpoints: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (source, endpoint, protocol), group in endpoints.groupby(
        ["source", "endpoint", "protocol"], sort=True
    ):
        def count(window: int, converged: bool = False, q_finite: bool = False) -> int:
            sub = group.loc[group["window_index"].eq(window)]
            if converged:
                sub = sub.loc[sub["null_status"].eq("converged")]
            if q_finite:
                sub = sub.loc[np.isfinite(sub["Q_i"].to_numpy(float))]
            return int(len(sub))

        pair = paired.loc[
            paired["source"].eq(int(source))
            & paired["endpoint"].eq(endpoint)
            & paired["protocol"].eq(protocol)
        ]
        w1 = group.loc[
            group["window_index"].eq(1)
            & group["null_status"].eq("converged")
            & np.isfinite(group["Q_i"].to_numpy(float))
        ]
        q_all = w1["Q_i"].to_numpy(float)
        q_pair = pair["Q_i_w1"].to_numpy(float) if len(pair) else np.array([], dtype=float)
        mean_all = float(np.mean(q_all)) if len(q_all) else np.nan
        mean_pair = float(np.mean(q_pair)) if len(q_pair) else np.nan
        diff = mean_pair - mean_all if np.isfinite(mean_all) and np.isfinite(mean_pair) else np.nan
        if len(q_all) > 1 and len(q_pair) > 1:
            pooled = np.sqrt((np.var(q_all, ddof=1) + np.var(q_pair, ddof=1)) / 2.0)
            smd = float(diff / pooled) if pooled > 1e-12 else np.nan
        else:
            smd = np.nan
        rows.append(
            {
                "source": int(source),
                "source_name": common.source_name(int(source)),
                "endpoint": str(endpoint),
                "protocol": str(protocol),
                "window1_rows": int(group["window_index"].eq(1).sum()),
                "window2_rows": int(group["window_index"].eq(2).sum()),
                "window1_converged_Q": count(1, converged=True, q_finite=True),
                "window2_converged_Q": count(2, converged=True, q_finite=True),
                "paired_converged_Q": int(len(pair)),
                "paired_fraction": float(len(pair) / len(w1)) if len(w1) else np.nan,
                "window1_mean_Q_all": mean_all,
                "window1_mean_Q_paired": mean_pair,
                "selection_difference_Q": diff,
                "selection_smd_Q": smd,
                "window1_mean_age_all": float(w1["age"].mean()) if len(w1) else np.nan,
                "window1_mean_age_paired": float(pair["age_w1"].mean()) if len(pair) else np.nan,
                "paired_female": int((pair["sex_w1"] == 1).sum()) if len(pair) else 0,
                "paired_male": int((pair["sex_w1"] == 0).sum()) if len(pair) else 0,
                "median_gap_days": float(pair["gap_days"].median()) if len(pair) else np.nan,
                "min_gap_days": float(pair["gap_days"].min()) if len(pair) else np.nan,
                "max_gap_days": float(pair["gap_days"].max()) if len(pair) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize(
    paired: pd.DataFrame, *, n_boot: int = 1000, seed: int = 20261020
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (source, endpoint, protocol), group in paired.groupby(
        ["source", "endpoint", "protocol"], sort=True
    ):
        print(
            f"longitudinal summary {source}/{endpoint}/{protocol}: "
            f"{len(group)} paired participants",
            flush=True,
        )
        rows.extend(
            _summarize_arm(
                group, source=int(source), endpoint=str(endpoint),
                protocol=str(protocol), n_boot=int(n_boot),
                seed=_seed("long-arm", seed, source, endpoint, protocol),
            )
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    subgroup_order = {"all": 0, "age_bin": 1, "sex": 2}
    out["_subgroup_order"] = out["subgroup"].map(subgroup_order)
    out = out.sort_values(
        ["source", "endpoint", "protocol", "_subgroup_order", "group"]
    ).drop(columns="_subgroup_order")
    return out.reset_index(drop=True)


def run_stage(
    run_dir: str | Path,
    *,
    n_boot: int = 1000,
    seed: int = 20261020,
) -> dict[str, object]:
    """Write longitudinal paired rows, availability, and summary tables."""
    run_dir = Path(run_dir)
    endpoints_path = run_dir / "endpoints/endpoints.parquet"
    if not endpoints_path.exists():
        raise FileNotFoundError(endpoints_path)
    endpoints = pd.read_parquet(endpoints_path)
    windows = set(pd.to_numeric(endpoints["window_index"], errors="coerce").dropna().astype(int))
    if not {1, 2}.issubset(windows):
        raise ValueError("longitudinal stage requires window_index 1 and 2")

    paired = build_paired_windows(endpoints)
    availability = availability_table(endpoints, paired)

    out = run_dir / "longitudinal"
    out.mkdir(parents=True, exist_ok=True)
    paired.to_parquet(out / "paired_windows.parquet", index=False)
    availability.to_csv(out / "availability.csv", index=False)
    summary = summarize(paired, n_boot=n_boot, seed=seed)
    summary.to_csv(out / "summary.csv", index=False)
    headline = summary.loc[summary["subgroup"].isin(["all", "sex"])].copy()
    headline.to_csv(out / "headline.csv", index=False)
    meta = {
        "n_pairs": int(len(paired)),
        "n_arms": int(summary[["source", "endpoint", "protocol"]].drop_duplicates().shape[0])
        if not summary.empty else 0,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "min_pair_n": int(MIN_PAIR_N),
        "age_range": [int(AGE_EDGES[0]), int(AGE_EDGES[-1])],
        "note": (
            "Repeat-window contrasts are short-interval within-participant "
            "associations, not causal aging effects. Intervals are pointwise "
            "participant-bootstrap percentile intervals."
        ),
    }
    (out / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"paired": int(len(paired)), "summary_rows": int(len(summary)), **meta}
