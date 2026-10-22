#!/usr/bin/env python3
"""Validate a repeat-window run against the primary-window run.

The primary-window endpoints must reproduce exactly because the run uses the
same source data, seed, protocol, null replicates, and optimizer.  This script
also checks key uniqueness, window counts, and the paired-window availability
table written by stage 8.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["source", "endpoint", "protocol", "user_id", "window_index"]
METRICS = ["Q_i", "Q_i_mc_se", "E_i2", "A_hat", "sigma_resid", "p_value_discrete"]


def _endpoints(run: Path) -> pd.DataFrame:
    return pd.read_parquet(run / "endpoints/endpoints.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-run", type=Path, required=True)
    parser.add_argument("--repeat-run", type=Path, required=True)
    args = parser.parse_args()

    primary = _endpoints(args.primary_run)
    repeat = _endpoints(args.repeat_run)
    p1 = primary.loc[primary["window_index"].eq(1)].copy()
    r1 = repeat.loc[repeat["window_index"].eq(1)].copy()
    r2 = repeat.loc[repeat["window_index"].eq(2)].copy()

    out: dict[str, object] = {
        "primary_rows": int(len(primary)),
        "repeat_rows": int(len(repeat)),
        "repeat_window1_rows": int(len(r1)),
        "repeat_window2_rows": int(len(r2)),
        "repeat_window_counts": {
            str(k): int(v) for k, v in repeat["window_index"].value_counts().sort_index().items()
        },
        "repeat_status": {
            str(k): int(v) for k, v in repeat["status"].value_counts(dropna=False).items()
        },
        "repeat_null_status": {
            str(k): int(v) for k, v in repeat["null_status"].value_counts(dropna=False).items()
        },
        "duplicate_keys": int(repeat.duplicated(KEYS).sum()),
        "primary_keys": int(len(p1)),
        "repeat_window1_keys": int(len(r1)),
    }

    merged = p1[KEYS + METRICS].merge(
        r1[KEYS + METRICS], on=KEYS, how="outer", suffixes=("_primary", "_repeat"),
        validate="one_to_one", indicator=True,
    )
    out["window1_merge_counts"] = {
        str(k): int(v) for k, v in merged["_merge"].value_counts().items()
    }
    both = merged.loc[merged["_merge"].eq("both")]
    diffs = {}
    for metric in METRICS:
        a = both[f"{metric}_primary"].to_numpy(float)
        b = both[f"{metric}_repeat"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        diffs[metric] = {
            "compared": int(ok.sum()),
            "max_abs_diff": float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else None,
            "n_diff_gt_0": int((a[ok] != b[ok]).sum()) if ok.any() else 0,
        }
    out["window1_metric_diffs"] = diffs

    # Window-1 selected dates must reproduce exactly for shared keys.
    h = p1[["source", "endpoint", "protocol", "user_id", "selected_dates_hash"]].merge(
        r1[["source", "endpoint", "protocol", "user_id", "selected_dates_hash"]],
        on=["source", "endpoint", "protocol", "user_id"], how="outer",
        suffixes=("_primary", "_repeat"), validate="one_to_one", indicator=True,
    )
    hb = h.loc[h["_merge"].eq("both")]
    out["window1_hash_compared"] = int(len(hb))
    out["window1_hash_mismatch"] = int(
        hb["selected_dates_hash_primary"].astype("string").ne(
            hb["selected_dates_hash_repeat"].astype("string")
        ).sum()
    )

    avail_path = args.repeat_run / "longitudinal/availability.csv"
    if avail_path.exists():
        avail = pd.read_csv(avail_path)
        out["longitudinal_availability_rows"] = int(len(avail))
        out["paired_total"] = int(avail["paired_converged_Q"].sum())
        out["paired_fraction_min"] = float(avail["paired_fraction"].min())
        out["paired_fraction_max"] = float(avail["paired_fraction"].max())
        out["median_gap_days_min"] = float(avail["median_gap_days"].min())
        out["median_gap_days_max"] = float(avail["median_gap_days"].max())

    paired_path = args.repeat_run / "longitudinal/paired_windows.parquet"
    if paired_path.exists():
        paired = pd.read_parquet(paired_path, columns=["source", "user_id", "age_w1"])
        out["paired_unique_source_participants"] = int(
            paired[["source", "user_id"]].drop_duplicates().shape[0]
        )
        in_range = paired.loc[paired["age_w1"].ge(20) & paired["age_w1"].lt(75)]
        out["paired_unique_source_participants_age_20_75"] = int(
            in_range[["source", "user_id"]].drop_duplicates().shape[0]
        )

    path = args.repeat_run / "validation.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(path)


if __name__ == "__main__":
    main()
