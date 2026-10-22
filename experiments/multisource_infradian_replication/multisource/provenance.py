"""Stage 1: inventory, provenance, and duplicate/conflict gates.

This module never invents vendor semantics.  It records what the upstream
exports assert (source ID, value type) and applies only the fixed provisional
QC rules stated in PLAN.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import common

KEYS = ["user_id", "source", "endpoint", "date"]


def load_daily(path: Path = common.DAILY_PARQUET) -> pd.DataFrame:
    """Load the normalized supplier daily table and validate its schema."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Daily table not found: {path}")
    df = pd.read_parquet(path)
    required = set(KEYS + ["value"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Daily table lacks required columns: {sorted(missing)}")
    out = df.copy()
    out["user_id"] = pd.to_numeric(out["user_id"], errors="raise").astype("int64")
    out["source"] = pd.to_numeric(out["source"], errors="raise").astype("int64")
    out["endpoint"] = out["endpoint"].astype("string").str.strip()
    out["date"] = pd.to_numeric(out["date"], errors="raise").astype("int64")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def collapse_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the plan's duplicate policy.

    * Exact duplicate rows are collapsed and counted.
    * A participant/source/endpoint/date key with >1 distinct finite value is a
      conflict and is excluded, because no documented chronology is available
      to resolve it.  Conflicting dates are never averaged.
    """
    key = KEYS
    n_raw = len(df)
    exact_dupes = int(df.duplicated(key + ["value"], keep="first").sum())

    grouped = df.groupby(key, sort=False, dropna=False)["value"]
    nunique = grouped.nunique(dropna=True)
    conflict_keys = nunique[nunique > 1].index
    n_conflict = int(len(conflict_keys))

    # Keep one row per key.  Prefer a finite value over a missing value when a
    # key contains both; non-conflicting finite values are identical by
    # construction.
    ordered = df.assign(_finite=np.isfinite(df["value"].to_numpy(dtype=float)))
    ordered = ordered.sort_values(
        key + ["_finite"], ascending=[True] * len(key) + [False], kind="stable"
    )
    collapsed = ordered.drop_duplicates(key, keep="first").drop(columns="_finite").copy()
    if n_conflict:
        conflict_index = pd.MultiIndex.from_tuples(
            [tuple(k) for k in conflict_keys], names=key
        )
        collapsed = collapsed.set_index(key)
        collapsed = collapsed.drop(index=conflict_index, errors="ignore")
        collapsed = collapsed.reset_index()
    meta = {
        "raw_rows": int(n_raw),
        "exact_duplicate_rows": exact_dupes,
        "conflicting_keys": n_conflict,
        "retained_rows": int(len(collapsed)),
    }
    return collapsed, meta


def apply_daily_qc(
    df: pd.DataFrame,
    value_min: float = common.DAILY_VALUE_MIN,
    value_max: float = common.DAILY_VALUE_MAX,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the fixed daily-value QC convention without imputation."""
    n_in = len(df)
    out = df.copy()
    nonfinite = ~np.isfinite(out["value"].to_numpy(dtype=float))
    out_of_range = out["value"].lt(value_min) | out["value"].gt(value_max)
    keep = np.isfinite(out["value"].to_numpy(dtype=float)) & ~out_of_range
    meta = {
        "input_rows": int(n_in),
        "nonfinite_rows": int(nonfinite.sum()),
        "out_of_range_rows": int(out_of_range.sum()),
        "retained_rows": int(keep.sum()),
        "value_min": float(value_min),
        "value_max": float(value_max),
    }
    return out.loc[keep].copy(), meta


def inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Per source/endpoint date-coverage inventory."""
    rows = []
    for (source, endpoint), group in df.groupby(["source", "endpoint"], sort=True):
        per_user = group.groupby("user_id")["date"].nunique()
        rows.append(
            {
                "source": int(source),
                "source_name": common.source_name(int(source)),
                "endpoint": str(endpoint),
                "type_id": common.DAILY_ENDPOINTS.get(str(endpoint)),
                "participants_any": int(len(per_user)),
                "participants_ge_90": int((per_user >= 90).sum()),
                "participants_ge_180": int((per_user >= 180).sum()),
                "median_dates": float(per_user.median()) if len(per_user) else np.nan,
                "rows": int(len(group)),
                "date_min": int(group["date"].min()),
                "date_max": int(group["date"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["source", "endpoint"]).reset_index(drop=True)


def channel_inventory(epoch_sources: pd.DataFrame | None = None) -> pd.DataFrame:
    """Channel-3000 inventory from epoch metadata, when available."""
    if epoch_sources is None:
        path = common.EPOCH_SOURCES_PARQUET
        if not path.exists():
            return pd.DataFrame()
        epoch_sources = pd.read_parquet(path)
    dense = epoch_sources.loc[epoch_sources["channel"].eq(common.DENSE_CHANNEL)].copy()
    dense["user"] = pd.to_numeric(dense["user"], errors="raise").astype("int64")
    dense["source"] = pd.to_numeric(dense["source"], errors="raise").astype("int64")
    dense = dense.loc[dense["source"].isin(common.SCOPE_SOURCES)].copy()
    dense["days"] = pd.to_numeric(dense["days"], errors="coerce")
    rows = []
    for source, group in dense.groupby("source", sort=True):
        rows.append(
            {
                "source": int(source),
                "source_name": common.source_name(int(source)),
                "participants_any": int(group["user"].nunique()),
                "participants_ge_90": int(group.loc[group["days"] >= 90, "user"].nunique()),
                "participants_ge_180": int(group.loc[group["days"] >= 180, "user"].nunique()),
                "median_days": float(group["days"].median()) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_provenance(
    daily_raw: pd.DataFrame,
    daily_qc: pd.DataFrame,
    duplicate_meta: dict[str, Any],
    qc_meta: dict[str, Any],
    source_mapping: pd.DataFrame,
    value_mapping: pd.DataFrame,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    """Assemble the auditable provenance record for stage 1."""
    present_sources = sorted(int(s) for s in daily_qc["source"].unique())
    present_endpoints = sorted(str(e) for e in daily_qc["endpoint"].unique())
    source_rows = {
        int(r.id): str(r.code) for r in source_mapping.itertuples(index=False)
    }
    value_rows = {
        int(r.id): str(r.code) for r in value_mapping.itertuples(index=False)
    }
    checks = {
        "source_ids_resolved": {
            str(s): source_rows.get(int(s), "UNRESOLVED") for s in present_sources
        },
        "value_type_ids_resolved": {
            str(common.DAILY_ENDPOINTS[e]): value_rows.get(common.DAILY_ENDPOINTS[e], "UNRESOLVED")
            for e in present_endpoints
            if e in common.DAILY_ENDPOINTS
        },
        "source_namespace": "pg_daily (mapping/pg_daily_sources.csv)",
        "value_type_namespace": "pg_daily (mapping/pg_daily_value_types.csv)",
        "thryve_namespace_warning": (
            "Thryve source IDs are a different namespace and must never be "
            "substituted for pg_daily IDs."
        ),
        "date_semantics": (
            "literal exported calendar dates, stored as integer days since "
            "1970-01-01; no timezone offset is applied to daily values"
        ),
        "daily_value_definition": (
            "one resolved value per user/source/endpoint/date; conflicting "
            "values are excluded, never averaged"
        ),
        "units": "bpm as exported; provisional QC range [25, 230]",
        "participant_linkage": "user_id joined directly to data/13Aug_1222.csv",
        "present_sources": present_sources,
        "present_endpoints": present_endpoints,
        "unresolved_sources_present": [
            s for s in present_sources if s in common.UNRESOLVED_SOURCES
        ],
        "excluded_sources_present": [
            s for s in present_sources if s in common.EXCLUDED_SOURCES
        ],
    }
    return {
        "stage": 1,
        "created_utc": common.utc_now_iso(),
        "input_hashes": input_hashes,
        "duplicates": duplicate_meta,
        "daily_qc": qc_meta,
        "checks": checks,
        "status": "verified_for_scope_sources",
        "scope_note": (
            "Only sources 2,3,6,7,9,13,19 enter endpoint analysis.  Polar (4) "
            "is excluded; unknown 38/46/48 remain inventory-only."
        ),
    }


def run_stage1(run_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    """Execute the inventory/provenance gate and persist its artifacts."""
    run_dir = common.ensure_run_dir(Path(run_dir), overwrite=overwrite)
    source_mapping = common.load_source_mapping()
    value_mapping = common.load_value_type_mapping()
    daily_raw = load_daily()
    collapsed, dup_meta = collapse_duplicates(daily_raw)
    daily_qc, qc_meta = apply_daily_qc(collapsed)

    inv = inventory(daily_qc)
    inv.to_csv(run_dir / "inventory_daily.csv", index=False)
    chan = channel_inventory()
    if not chan.empty:
        chan.to_csv(run_dir / "inventory_channel3000.csv", index=False)

    hashes = {
        "daily_parquet": common.sha256_file(common.DAILY_PARQUET),
        "source_mapping": common.sha256_file(common.SOURCE_MAPPING_CSV),
        "value_type_mapping": common.sha256_file(common.VALUE_TYPE_MAPPING_CSV),
    }
    prov = build_provenance(
        daily_raw, daily_qc, dup_meta, qc_meta, source_mapping, value_mapping, hashes
    )
    (run_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")

    # Persist the cleaned daily table so later stages do not repeat QC.
    daily_qc.to_parquet(run_dir / "daily_clean.parquet", index=False)

    summary = {
        "duplicates": dup_meta,
        "daily_qc": qc_meta,
        "inventory_rows": int(len(inv)),
        "channel_rows": int(len(chan)),
        "sources_present": sorted(int(s) for s in daily_qc["source"].unique()),
    }
    (run_dir / "stage1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
