"""Side-quest section 10: targeted raw-file manifest and parser benchmark.

Produces the inputs the plan requires before any runtime statement:

  artifacts_sq/raw_file_manifest.parquet  one row per model-eligible participant
                                          (bytes of out/<uid>.csv, by source)
  artifacts_sq/scan_benchmark.json        per-source benchmark of the exact
                                          window-diurnal parser on a
                                          deterministic sample + extrapolation

Run:  python src/sidequest/raw_manifest.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sq_config import ARTIFACTS_SQ, EPOCH_DIR  # noqa: E402


def build_manifest(model_users: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, src in zip(model_users["user_id"].to_numpy(),
                        model_users["source_id"].to_numpy()):
        p: Path = EPOCH_DIR / f"{uid}.csv"
        if p.exists():
            rows.append({"user_id": int(uid), "source_id": int(src),
                         "bytes": p.stat().st_size, "exists": True})
        else:
            rows.append({"user_id": int(uid), "source_id": int(src),
                         "bytes": 0, "exists": False})
    return pd.DataFrame(rows)


def benchmark(man: pd.DataFrame, n_per_source: int = 15) -> dict:
    from diurnal import process_window_user
    rng_sample = (man[man["exists"]]
                  .sort_values(["source_id", "user_id"])
                  .groupby("source_id", group_keys=False).head(n_per_source))
    per_source = {}
    for src, g in rng_sample.groupby("source_id"):
        bytes_read = 0
        t0 = time.perf_counter()
        n_err = 0
        for _, r in g.iterrows():
            try:
                rows = process_window_user(int(r["user_id"]), int(r["source_id"]),
                                           int(r["window_start"]), int(r["window_end"]))
                bytes_read += int(r["bytes"])
            except Exception as exc:  # benchmark counts parser failures
                n_err += 1
                print(f"  benchmark parse failure user {r['user_id']}: {exc}", flush=True)
        dt = time.perf_counter() - t0
        per_source[int(src)] = {
            "files": int(len(g)), "parse_failures": int(n_err),
            "bytes": int(bytes_read), "seconds": float(dt),
            "mb_per_s_per_file_stream": float(bytes_read / 1e6 / dt) if dt > 0 else 0.0,
        }
    est = {}
    for src, stats in per_source.items():
        sub = man[(man["source_id"] == src) & man["exists"]]
        total_bytes = int(sub["bytes"].sum())
        rate = stats["mb_per_s_per_file_stream"]
        est[str(src)] = {
            "files_total": int(len(sub)), "bytes_total": total_bytes,
            "gib_total": round(total_bytes / 2**30, 2),
            "rate_mbps": round(rate, 1),
            "est_seconds_single_worker": round(total_bytes / 1e6 / rate, 1) if rate > 0 else None,
        }
    return {"per_source_sample": per_source, "extrapolation": est,
            "sample_size_per_source": n_per_source}


def main() -> None:
    ARTIFACTS_SQ.mkdir(parents=True, exist_ok=True)
    model = pd.read_parquet(ARTIFACTS_SQ / "cohort_manifest_model_sources.parquet")
    man = build_manifest(model)
    man.to_parquet(ARTIFACTS_SQ / "raw_file_manifest.parquet", index=False)
    n_missing = int((~man["exists"]).sum())
    print(f"raw manifest: {len(man):,} files, missing/nonexistent: {n_missing}")

    bench = benchmark(man.merge(
        model[["user_id", "window_start", "window_end"]], on="user_id", how="left"))
    total_gib = sum(v["gib_total"] for v in bench["extrapolation"].values())
    bench["total_gib_all_model_files"] = round(total_gib, 2)
    bench["workers_planned"] = 8
    with open(ARTIFACTS_SQ / "scan_benchmark.json", "w") as fh:
        json.dump(bench, fh, indent=2)
    for src, v in bench["extrapolation"].items():
        print(f"  source {src}: {v['files_total']:,} files, {v['gib_total']} GiB, "
              f"{v['rate_mbps']} MB/s/file, ~{v['est_seconds_single_worker']} s single-worker")
    print(f"total targeted raw: {total_gib:.2f} GiB")


if __name__ == "__main__":
    main()
