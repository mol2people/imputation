#!/usr/bin/env python3
"""Orchestrate the full multisource pipeline for one run namespace.

Example:
    python scripts/run_all.py --run-dir artifacts/runs/run_20261021 \
        --workers 9 --n-boot 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multisource.runtime import configure_single_threaded_blas

configure_single_threaded_blas()

import pandas as pd

from multisource import bootstrap, channel3000, common, daily, longitudinal, manifests, matched, models, provenance, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", type=int, default=list(common.SCOPE_SOURCES))
    parser.add_argument("--protocols", nargs="+", default=list(common.PROTOCOLS))
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--null-replicates", type=int, default=common.B_NULL_REPLICATES)
    parser.add_argument("--n-boot", type=int, default=common.N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--limit", type=int, help="Bounded smoke run across stages.")
    parser.add_argument("--skip-channel", action="store_true")
    parser.add_argument("--skip-matched", action="store_true")
    parser.add_argument("--include-repeat-windows", action="store_true",
                        help="Also compute the secondary repeat window (window_index 2).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = common.ensure_run_dir(Path(args.run_dir), overwrite=args.overwrite)
    print(f"workers: {args.workers}; null replicates: {args.null_replicates}; bootstrap: {args.n_boot}")
    print("[1/7] inventory and provenance")
    provenance.run_stage1(run_dir, overwrite=True)

    if not args.skip_channel:
        print("[2/7] channel-3000 caches")
        daily_clean = pd.read_parquet(run_dir / "daily_clean.parquet")
        candidates = channel3000.candidate_users(daily_clean, args.sources)
        index = channel3000.build_raw_index(
            candidates, raw_dir=common.RAW_OUT_DIR, sources=args.sources,
            workers=args.workers, limit=args.limit, exhaustive=True,
        )
        index.to_parquet(run_dir / "raw_index.parquet", index=False)
        build_cols = ["user_id", "source"] + [
            c for c in ("dense_rows", "raw_path") if c in index.columns
        ]
        cache_index = channel3000.build_caches(
            index[build_cols], run_dir / "cache_5min",
            raw_dir=common.RAW_OUT_DIR, workers=args.workers, limit=args.limit,
        )
        cache_index.to_parquet(run_dir / "cache_index.parquet", index=False)
        stage1b = {
            "candidates": int(len(candidates)),
            "raw_index_rows": int(len(index)),
            "caches_built": int(len(cache_index)),
            "cache_dir": str(run_dir / "cache_5min"),
            "by_source": (
                cache_index.groupby("source").size().to_dict()
                if not cache_index.empty else {}
            ),
        }
        (run_dir / "stage1b_summary.json").write_text(
            json.dumps(stage1b, indent=2) + "\n", encoding="utf-8"
        )

    print("[3/7] manifests")
    manifests.run_stage2(
        run_dir, sources=args.sources, protocols=args.protocols,
        cache_dir=run_dir / "cache_5min", workers=args.workers,
    )

    print("[4/7] endpoints and nulls")
    import subprocess
    cmd = [
        sys.executable, str(Path(__file__).with_name("03_compute_endpoints.py")),
        "--run-dir", str(run_dir), "--mode", "daily" if args.skip_channel else "both",
        "--workers", str(args.workers), "--null-replicates", str(args.null_replicates),
        "--protocols", *args.protocols, "--sources", *[str(s) for s in args.sources],
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.include_repeat_windows:
        cmd += ["--include-repeat-windows"]
    subprocess.run(cmd, check=True)

    print("[5/7] models")
    window_indices = (1, 2) if args.include_repeat_windows else (1,)
    models.run_stage3(run_dir, workers=args.workers, window_indices=window_indices)

    print("[6/7] bootstrap")
    bootstrap.run_stage4(
        run_dir, n_boot=args.n_boot, seed=args.seed, workers=args.workers,
        window_indices=window_indices,
    )

    if args.include_repeat_windows:
        print("[8/9] longitudinal repeat-window summaries")
        longitudinal.run_stage(
            run_dir, n_boot=args.n_boot, seed=args.seed,
        )

    matched_label = "[9/9]" if args.include_repeat_windows else "[7/7]"
    if not args.skip_matched and not args.skip_channel:
        print(f"{matched_label} matched comparisons")
        matched.run_stage5(
            run_dir, run_dir / "cache_5min", sources=args.sources,
            protocols=args.protocols, null_replicates=args.null_replicates,
            n_boot=args.n_boot, seed=args.seed, workers=args.workers,
        )
    else:
        print(f"{matched_label} matched comparisons skipped")

    print("report")
    out = report.run_stage6(run_dir)
    print(json.dumps({"report": str(out)}, indent=2))


if __name__ == "__main__":
    main()
