#!/usr/bin/env python3
"""Run and validate a bounded HRp5/HRp1 real-cache slice against frozen HRp10 manifests."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import shared


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--null-replicates", type=int, default=199)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("choose a new output directory; this validation will not overwrite artifacts")
    cli = Path(__file__).with_name("endpoints.py")
    subprocess.run([sys.executable, str(cli), "--arms", "HRp5", "HRp1", "--protocols",
                    "expanded90", "balanced90", "--limit", str(args.limit),
                    "--null-replicates", str(args.null_replicates), "--output-dir",
                    str(args.output_dir)], check=True)

    baseline = pd.read_parquet(shared.ARTIFACTS / "endpoints/endpoints.parquet")
    key = ["user_id", "source", "protocol", "window_index", "selected_dates_hash"]
    for protocol in ("expanded90", "balanced90"):
        frames = {arm: pd.read_parquet(args.output_dir / arm / protocol / "endpoints.parquet")
                  for arm in ("HRp5", "HRp1")}
        ref = shared.build_tasks("HRp10", protocol, args.limit)
        expected = {(t["user_id"], t["source"], w["protocol"], w["window_index"], w["selected_dates_hash"])
                    for t in ref for w in t["windows"]}
        for arm, frame in frames.items():
            observed = set(map(tuple, frame[key].itertuples(index=False, name=None)))
            assert observed == expected, f"fixed participant/window/date hash mismatch for {arm}/{protocol}"
            assert frame["n_selected_dates"].eq(90).all()
            assert frame["n_qualified_dates"].eq(90).all()
            assert frame["null_status"].eq("converged").all(), f"null fit failed for {arm}/{protocol}"
        base_slice = baseline.loc[baseline.protocol.eq(protocol)]
        for frame in frames.values():
            merged = frame.merge(base_slice, on=["user_id", "source", "protocol", "window_index"],
                                 suffixes=("_arm", "_ref"), validate="one_to_one")
            assert (merged.selected_dates_hash_arm == merged.selected_dates_hash_ref).all()
        for task in ref:
            shared.validate_percentile_order(task)
        for arm in ("HRp5", "HRp1"):
            daily = pd.read_parquet(args.output_dir / arm / protocol / "daily_values.parquet")
            assert len(daily.loc[daily.arm.eq(arm)]) == 90 * len(frames[arm])
            assert len(daily.loc[daily.arm.eq("HRp10")]) == 90 * len(frames[arm])
    print("Bounded validation passed: fixed rows/hashes, 90 selected dates, converged nulls, and p1 <= p5 <= p10.")


if __name__ == "__main__":
    main()
