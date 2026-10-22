#!/usr/bin/env python3
"""Run fixed-date HR percentile endpoint arms using the existing REML engine."""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import shared

ENGINE_PATH = HERE.parent / "scripts" / "02_compute_endpoints_and_nulls.py"
SPEC = importlib.util.spec_from_file_location("infradian_endpoint_engine", ENGINE_PATH)
engine = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(engine)


def _process_task(task):
    return engine.process_participant_task(task)


def run_arm(arm: str, protocol: str, output_root: Path, limit: int | None,
            null_replicates: int, tasks: list[dict] | None = None,
            workers: int = 1) -> Path:
    tasks = tasks if tasks is not None else shared.build_tasks(arm, protocol, limit)
    if not tasks:
        raise RuntimeError(f"No manifest tasks for {arm}/{protocol}")
    for task in tasks:
        task["null_replicates"] = null_replicates
        task["save_daily_values"] = True

    rows = []
    if workers > 1:
        chunk = max(1, len(tasks) // (workers * 4) + 1)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for result in ex.map(_process_task, tasks, chunksize=chunk):
                rows.extend(result)
    else:
        for task in tasks:
            rows.extend(engine.process_participant_task(task))

    out = output_root / arm / protocol
    spectra_dir = out / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    spectra_records = [r for r in rows if "spectra" in r]
    endpoint_rows = [{k: v for k, v in r.items() if k not in {"spectra", "null_diagnostics"}}
                     for r in rows]
    endpoints = pd.DataFrame(endpoint_rows)
    if not endpoints.empty:
        endpoints["null_converged"] = endpoints["null_status"].eq("converged")
        endpoints["null_failure_reason"] = np.where(endpoints["null_converged"], "", endpoints["status"])
    endpoints.to_parquet(out / "endpoints.parquet", index=False)
    daily_rows = []
    for r in rows:
        if "daily_values" not in r:
            continue
        win = next(w for task in tasks if task["user_id"] == r["user_id"] and task["source"] == r["source"]
                   for w in task["windows"] if w["protocol"] == r["protocol"] and w["window_index"] == r["window_index"])
        daily_rows.extend({"user_id": r["user_id"], "source": r["source"], "arm": arm,
                           "protocol": protocol, "window_index": r["window_index"],
                           "selected_dates_hash": r["selected_dates_hash"], "date": int(day),
                           "value": float(value)}
                          for day, value in zip(win["dates"], r["daily_values"]))
    pd.DataFrame(daily_rows).to_parquet(out / "daily_values.parquet", index=False)
    diag_cols = ["user_id", "source", "arm", "protocol", "window_index", "selected_dates_hash",
                 "status", "null_status", "null_converged", "null_failure_reason", "rss_0",
                 "v1", "phi1", "v2", "phi2", "Q_i_mc_se"]
    endpoints[[c for c in diag_cols if c in endpoints]].to_parquet(out / "null_diagnostics.parquet", index=False)

    spec_rows = []
    for r in spectra_records:
        spec = r["spectra"]
        spec_rows.append({k: r[k] for k in ("user_id", "source", "arm", "protocol", "window_index",
                                             "selected_dates_hash") if k in r} | {
            "broad_freqs": engine.BROAD_FREQS.tolist(),
            "P_obs": spec["P_obs"].tolist(), "P_null_mean": spec["P_null_mean"].tolist(),
            "P_null_var": spec["P_null_var"].tolist(), "D_excess": spec["D_excess"].tolist(),
        })
    pd.DataFrame(spec_rows).to_parquet(spectra_dir / "spectra_summary.parquet", index=False)
    if spectra_records:
        np.savez_compressed(spectra_dir / "spectra_broad.npz",
            user_id=np.asarray([r["user_id"] for r in spectra_records], dtype=np.int64),
            source=np.asarray([r["source"] for r in spectra_records], dtype=np.int64),
            protocol=np.asarray([r["protocol"] for r in spectra_records], dtype=str),
            window_index=np.asarray([r["window_index"] for r in spectra_records], dtype=np.int64),
            selected_dates_hash=np.asarray([r["selected_dates_hash"] for r in spectra_records], dtype=str),
            broad_freqs=engine.BROAD_FREQS,
            P_obs=np.stack([r["spectra"]["P_obs"] for r in spectra_records]),
            P_null_mean=np.stack([r["spectra"]["P_null_mean"] for r in spectra_records]),
            P_null_var=np.stack([r["spectra"]["P_null_var"] for r in spectra_records]),
            D_excess=np.stack([r["spectra"]["D_excess"] for r in spectra_records]))

    (out / "provenance.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "arm": arm,
        "percentile": shared.PERCENTILES.get(arm), "protocol": protocol,
        "endpoint": None if arm in shared.PERCENTILES else arm,
        "null_replicates": null_replicates, "date_manifest_sha256": shared.date_manifest_hash(),
        "HRp10_reference": str(shared.ARTIFACTS / "endpoints/endpoints.parquet"),
        "fixed_seed_tag": "20261020|infradian_age|source|user_id|window_start|null_two_component|selected_dates_hash",
        "rng": "existing engine PCG64 stream; arm omitted from seed; identical draws for same participant/window across arms",
        "task_count": len(tasks), "engine": str(ENGINE_PATH),
    }, indent=2) + "\n")
    return out


def append_saved_hrp10_daily(output_path: Path, tasks: list[dict]) -> None:
    """Append cheap HRp10 daily values from the fixed channel-3000 cache; no model fit."""
    rows = []
    for task in tasks:
        if not Path(task["cache_path"]).exists():
            continue
        with np.load(task["cache_path"]) as cache:
            hr, date_arr = cache["hr"], cache["date"]
        date_to_row = {int(day): i for i, day in enumerate(date_arr)}
        for win in task["windows"]:
            for day in win["dates"]:
                vals = hr[date_to_row[int(day)]]
                vals = vals[np.isfinite(vals)]
                value = float(np.percentile(vals, 10)) if len(vals) else np.nan
                rows.append({"user_id": task["user_id"], "source": task["source"], "arm": "HRp10",
                             "protocol": win["protocol"], "window_index": win["window_index"],
                             "selected_dates_hash": win["selected_dates_hash"],
                             "date": int(day), "value": value})
    if rows:
        path = output_path / "daily_values.parquet"
        current = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=rows[0].keys())
        combined = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
        combined.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=("HRp5", "HRp1"), default=["HRp5", "HRp1"])
    parser.add_argument("--protocols", nargs="+", choices=("expanded90", "balanced90"),
                        default=["expanded90", "balanced90"])
    parser.add_argument("--limit", type=int, help="Bounded number of participant tasks per arm/protocol.")
    parser.add_argument("--null-replicates", type=int, default=199)
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes (default 8)")
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts/runs/latest")
    parser.add_argument("--supplier-daily", type=Path,
                        help="Normalized supplier table; requires a verified provenance JSON.")
    parser.add_argument("--supplier-endpoint", help="Exact exported supplier endpoint name.")
    parser.add_argument("--supplier-provenance", type=Path,
                        help="Verified provenance JSON required for supplier comparisons.")
    parser.add_argument("--supplier-only", action="store_true",
                        help="Run only the supplier arm; requires the three supplier arguments.")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.null_replicates < 2:
        parser.error("--null-replicates must be at least 2")
    supplier_args = (args.supplier_daily, args.supplier_endpoint, args.supplier_provenance)
    if any(supplier_args) and not all(supplier_args):
        parser.error("--supplier-daily, --supplier-endpoint, and --supplier-provenance must be supplied together")
    if args.supplier_only and not all(supplier_args):
        parser.error("--supplier-only requires --supplier-daily, --supplier-endpoint, and --supplier-provenance")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    for protocol in args.protocols:
        if not args.supplier_only:
            alternative_paths = []
            for arm in args.arms:
                path = run_arm(arm, protocol, args.output_dir, args.limit, args.null_replicates, workers=args.workers)
                alternative_paths.append(path)
                print(f"Wrote {arm}/{protocol}: {path}")
            fixed_tasks = shared.build_tasks(args.arms[0], protocol, args.limit)
            for path in alternative_paths:
                append_saved_hrp10_daily(path, fixed_tasks)
        if args.supplier_daily:
            import json
            import supplier
            daily = pd.read_parquet(args.supplier_daily) if args.supplier_daily.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(args.supplier_daily)
            provenance = json.loads(args.supplier_provenance.read_text(encoding="utf-8"))
            selected = shared.load_manifests()[2]
            prepared, retention = supplier.prepare_exact_date_inputs(selected, daily,
                args.supplier_endpoint, provenance)
            supp_tasks = shared.build_supplier_tasks(prepared, args.supplier_endpoint, protocol, args.limit)
            if supp_tasks:
                path = run_arm(args.supplier_endpoint, protocol, args.output_dir, args.limit, args.null_replicates,
                               tasks=supp_tasks, workers=args.workers)
                append_saved_hrp10_daily(path, supp_tasks)
                retention.to_csv(path / "retention.csv", index=False)
                window_keys = {(int(r.user_id), int(r.source), int(r.window_index))
                               for r in prepared.loc[prepared.protocol.eq(protocol)].itertuples()}
                matched_path = args.output_dir / f"HRp10_matched_{args.supplier_endpoint}" / protocol
                matched_path.mkdir(parents=True)
                saved = pd.read_parquet(shared.ARTIFACTS / "endpoints/endpoints.parquet")
                saved = saved.loc[saved.protocol.eq(protocol) & saved.apply(
                    lambda r: (int(r.user_id), int(r.source), int(r.window_index)) in window_keys, axis=1)].copy()
                saved["arm"] = "HRp10"
                saved.to_parquet(matched_path / "endpoints.parquet", index=False)
                append_saved_hrp10_daily(matched_path, supp_tasks)
                saved_spectra = pd.read_parquet(shared.ARTIFACTS / "spectra/spectra_summary.parquet")
                saved_spectra = saved_spectra.loc[saved_spectra.protocol.eq(protocol) & saved_spectra.apply(
                    lambda r: (int(r.user_id), int(r.source), int(r.window_index)) in window_keys, axis=1)]
                saved_spectra.to_parquet(matched_path / "spectra_summary.parquet", index=False)
                matched = matched_path
                print(f"Wrote supplier matched cohort: {path}; HRp10 restriction: {matched}")
            else:
                print(f"No exact-date complete cases for {args.supplier_endpoint}/{protocol}; supplier arm stopped")


if __name__ == "__main__":
    main()
