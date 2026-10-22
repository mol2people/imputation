"""Prepare supplier daily values for conditional, matched RHR arms.

This module consumes normalized source-specific supplier observations. It does
not infer a vendor's daily algorithm, units, or date semantics from export
labels. Those provenance facts must be supplied explicitly before preparation.
"""
from __future__ import annotations

from collections.abc import Mapping
import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from common import (
    compute_selected_dates_hash,
    derive_block_seed,
    greedy_quota_allocation,
    sample_block_dates,
)


REQUIRED_PROVENANCE = (
    "participant_source_linkage", "date_semantics", "daily_value_definition", "units",
)
KEYS = ["user_id", "source", "endpoint", "date"]
TASK_KEYS = ["user_id", "source", "protocol", "window_index"]


def _require_verified(provenance: Mapping[str, Any] | None) -> None:
    if not isinstance(provenance, Mapping) or provenance.get("status") != "verified":
        raise ValueError("Supplier preparation is gated: provide provenance with status='verified'.")
    absent = [k for k in REQUIRED_PROVENANCE if not str(provenance.get(k, "")).strip()]
    if absent:
        raise ValueError("Verified provenance is incomplete: " + ", ".join(absent))


def _normalize_day(values: pd.Series) -> pd.Series:
    """Parse calendar-date keys without applying a timezone conversion."""
    parsed = pd.to_datetime(values, errors="raise", unit="D")
    if getattr(parsed.dt, "tz", None) is not None:
        raise ValueError("Timezone-aware dates require an explicit verified local-date conversion first.")
    return parsed.dt.normalize()


def _validate_supplier(daily: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + ["value"])
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Supplier table lacks required columns: {sorted(missing)}")
    out = daily.copy()
    out["user_id"] = pd.to_numeric(out["user_id"], errors="raise").astype("int64")
    out["source"] = pd.to_numeric(out["source"], errors="raise").astype("int64")
    out["endpoint"] = out["endpoint"].astype("string")
    out["date"] = _normalize_day(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    if out.duplicated(KEYS).any():
        raise ValueError("Duplicate participant/source/endpoint/date rows found; resolve under verified duplicate policy first.")
    return out


def prepare_exact_date_inputs(
    selected_dates: pd.DataFrame,
    supplier_daily: pd.DataFrame,
    endpoint: str,
    provenance: Mapping[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return exact-date complete-case rows and aggregate retention by window.

    `selected_dates` uses the experiment artifact schema (`date` is integer
    days since 1970-01-01). The matched set requires one finite supplier value
    on every selected date. No date is imputed and no participant is removed
    from another arm by this helper.
    """
    _require_verified(provenance)
    need = set(TASK_KEYS + ["date", "selected_dates_hash"])
    missing = need.difference(selected_dates.columns)
    if missing:
        raise ValueError(f"Selected-date table lacks required columns: {sorted(missing)}")
    baseline = selected_dates[list(need)].copy()
    baseline["user_id"] = baseline["user_id"].astype("int64")
    baseline["source"] = baseline["source"].astype("int64")
    baseline["date"] = pd.to_datetime(baseline["date"].astype("int64"), unit="D", origin="unix")
    if baseline.duplicated(TASK_KEYS + ["date"]).any():
        raise ValueError("Selected-date table contains duplicate participant/window/date keys.")
    date_counts = baseline.groupby(TASK_KEYS, sort=False)["date"].nunique()
    if not date_counts.eq(90).all():
        raise ValueError("Exact-date comparison requires exactly 90 selected dates per window.")
    supplier = _validate_supplier(supplier_daily)
    supplier = supplier.loc[supplier["endpoint"].eq(endpoint) & np.isfinite(supplier["value"])].copy()

    expected = baseline.groupby(TASK_KEYS, sort=False)["date"].nunique().rename("expected_dates").reset_index()
    join_cols = ["user_id", "source", "date"]
    matched = baseline.merge(supplier, on=join_cols, how="left", validate="many_to_one", suffixes=("", "_supplier"))
    # A source mismatch is not a match, even when the user/date keys coincide.
    source_col = "source_supplier" if "source_supplier" in matched else "source"
    if source_col != "source":
        matched.loc[matched["source"] != matched[source_col], "value"] = np.nan
        matched.drop(columns=[source_col], inplace=True)
    good = matched[np.isfinite(matched["value"])].copy()
    observed = good.groupby(TASK_KEYS, sort=False)["date"].nunique().rename("observed_dates").reset_index()
    status = expected.merge(observed, on=TASK_KEYS, how="left")
    status["observed_dates"] = status["observed_dates"].fillna(0).astype("int64")
    status["complete"] = status["observed_dates"].eq(status["expected_dates"])
    status["retained"] = status["complete"]
    keep = status.loc[status["complete"], TASK_KEYS]
    complete = good.merge(keep, on=TASK_KEYS, how="inner", validate="many_to_one")
    complete["date"] = complete["date"].dt.strftime("%Y-%m-%d")
    status["endpoint"] = endpoint
    status["retention_fraction"] = np.where(status["expected_dates"] > 0,
                                            status["observed_dates"] / status["expected_dates"], np.nan)
    return complete, status


def select_common_dates(
    qualified_channel3000: pd.DataFrame,
    supplier_daily: pd.DataFrame,
    endpoint: str,
    protocol: str,
    provenance: Mapping[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feasibility-only common-date reselection using fixed window blocks.

    Input rows must be channel-3000-qualified dates from the original 180-day
    window and include `window_start_day` (integer calendar day). Supplier
    availability is intersected before any RHR value is examined. The existing
    deterministic quota allocation and block sampler are reused.
    """
    _require_verified(provenance)
    if protocol not in {"expanded90", "balanced90"}:
        raise ValueError("Common-date fallback is defined only for expanded90 or balanced90.")
    required = set(TASK_KEYS + ["date", "window_start_day"])
    missing = required.difference(qualified_channel3000.columns)
    if missing:
        raise ValueError(f"Qualified-date table lacks required columns: {sorted(missing)}")
    supplier = _validate_supplier(supplier_daily)
    supplier = supplier.loc[supplier["endpoint"].eq(endpoint) & np.isfinite(supplier["value"]),
                            ["user_id", "source", "endpoint", "date", "value"]]
    candidates = qualified_channel3000.copy()
    if candidates.duplicated(TASK_KEYS + ["date"]).any():
        raise ValueError("Qualified-date input contains duplicate participant/window/date keys.")
    candidates["date"] = pd.to_datetime(candidates["date"].astype("int64"), unit="D", origin="unix")
    candidates["window_start_day"] = candidates["window_start_day"].astype("int64")
    candidates = candidates.merge(supplier, on=["user_id", "source", "date"], how="inner", validate="many_to_one")
    out_rows, audits = [], []
    for key, group in candidates.groupby(TASK_KEYS, sort=False):
        start = int(group["window_start_day"].iloc[0])
        day = group["date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
        block = day - start
        in_window = (block >= 0) & (block < 180)
        group = group.loc[in_window].copy()
        day = day[in_window]
        block_idx = ((day - start) // 30).astype(int)
        group["_day"] = day
        group["_block"] = block_idx
        capacities = np.array([(group["_block"] == b).sum() for b in range(6)], dtype=np.int64)
        feasible = bool(np.all(capacities >= (10 if protocol == "expanded90" else 15)))
        if protocol == "expanded90":
            feasible = feasible and capacities.sum() >= 90
        if not feasible:
            audits.append(dict(zip(TASK_KEYS, key), endpoint=endpoint, feasible=False,
                               common_available_dates=int(len(group)), complete_dates=0,
                               block_capacities=",".join(map(str, capacities))))
            continue
        quotas = greedy_quota_allocation(capacities) if protocol == "expanded90" else np.full(6, 15, dtype=np.int64)
        user_id, source, _, window_index = key
        chosen = []
        for b, quota in enumerate(quotas):
            dates = np.sort(group.loc[group["_block"].eq(b), "_day"].unique())
            chosen.extend(sample_block_dates(dates, int(quota),
                          derive_block_seed(int(source), int(user_id), start, b)).tolist())
        chosen = np.asarray(sorted(chosen), dtype=np.int64)
        chosen_dates = pd.to_datetime(chosen, unit="D", origin="unix")
        selected = group.set_index("_day").loc[chosen].reset_index()
        selected["date"] = selected["date"].dt.strftime("%Y-%m-%d")
        selected["selected_dates_hash"] = compute_selected_dates_hash(chosen)
        out_rows.append(selected.drop(columns=["_block"]))
        audits.append(dict(zip(TASK_KEYS, key), endpoint=endpoint, feasible=True,
                           common_available_dates=int(len(group)), complete_dates=int(len(chosen)),
                           block_capacities=",".join(map(str, capacities))))
    selected = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    return selected, pd.DataFrame(audits)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-exact", help="create exact-date complete-case supplier inputs")
    prep.add_argument("--selected-dates", type=Path, required=True)
    prep.add_argument("--supplier-daily", type=Path, required=True, help="normalized long table; see README")
    prep.add_argument("--endpoint", required=True)
    prep.add_argument("--provenance", type=Path, required=True, help="short JSON record confirming endpoint/source/date mapping")
    prep.add_argument("--output", type=Path, required=True, help="participant-level output table")
    prep.add_argument("--retention-output", type=Path, required=True)
    args = parser.parse_args()
    selected = _read_table(args.selected_dates)
    supplier = _read_table(args.supplier_daily)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    prepared, retention = prepare_exact_date_inputs(selected, supplier, args.endpoint, provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.retention_output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(args.output, index=False)
    retention.to_csv(args.retention_output, index=False)
    print(json.dumps({"endpoint": args.endpoint, "complete_case_rows": len(prepared),
                      "matched_windows": int(retention["complete"].sum()),
                      "eligible_windows": int(len(retention)),
                      "output": str(args.output), "retention_output": str(args.retention_output)}, indent=2))


if __name__ == "__main__":
    main()
