#!/usr/bin/env python3
"""Stage 3: compute daily and channel-3000 endpoints and nulls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multisource.runtime import configure_single_threaded_blas

configure_single_threaded_blas()

import pandas as pd

from multisource import channel3000, common, endpoints, stage3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("daily", "channel", "both"), default="both")
    parser.add_argument("--sources", nargs="+", type=int)
    parser.add_argument("--endpoints", nargs="+", default=["RHR", "HR_during_sleep"])
    parser.add_argument("--protocols", nargs="+", default=list(common.PROTOCOLS))
    parser.add_argument("--null-replicates", type=int, default=common.B_NULL_REPLICATES)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--include-repeat-windows", action="store_true",
        help="Also compute the secondary repeat window (window_index 2).",
    )
    args = parser.parse_args()
    window_indices = (1, 2) if args.include_repeat_windows else (1,)

    run_dir = Path(args.run_dir)
    manifests_dir = run_dir / "manifests"
    cache_dir = args.cache_dir or (run_dir / "cache_5min")
    output_root = run_dir / "endpoints"
    output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict] = []

    if args.mode in ("daily", "both"):
        selected = pd.read_parquet(manifests_dir / "daily_selected_dates.parquet")
        windows = pd.read_parquet(manifests_dir / "daily_windows.parquet")
        demographics = pd.read_parquet(manifests_dir / "participants.parquet")
        sources = args.sources or sorted(int(s) for s in selected["source"].unique())
        for source in sources:
            for endpoint in args.endpoints:
                for protocol in args.protocols:
                    sub = selected.loc[
                        selected["source"].eq(int(source))
                        & selected["endpoint"].eq(endpoint)
                        & selected["protocol"].eq(protocol)
                    ]
                    if sub.empty:
                        continue
                    tasks = endpoints.build_daily_tasks(
                        selected, windows, demographics, int(source), endpoint, protocol,
                        args.limit, window_indices=window_indices,
                    )
                    if not tasks:
                        continue
                    out = output_root / "daily" / f"source_{source}" / endpoint / protocol
                    label = f"daily {source}/{endpoint}/{protocol}"
                    print(f"{label}: {len(tasks)} tasks, workers={args.workers}")
                    jobs.append({
                        "kind": "daily", "tasks": tasks, "output_dir": out,
                        "label": label,
                    })

    if args.mode in ("channel", "both") and (manifests_dir / "channel_selected_dates.parquet").exists():
        selected = pd.read_parquet(manifests_dir / "channel_selected_dates.parquet")
        windows = pd.read_parquet(manifests_dir / "channel_windows.parquet")
        demographics = pd.read_parquet(manifests_dir / "participants.parquet")
        sources = args.sources or sorted(int(s) for s in selected["source"].unique())
        for source in sources:
            for protocol in args.protocols:
                sub = selected.loc[
                    selected["source"].eq(int(source)) & selected["protocol"].eq(protocol)
                ]
                if sub.empty:
                    continue
                tasks = channel3000.build_channel_tasks(
                    selected, windows, demographics, Path(cache_dir), protocol,
                    limit=args.limit, window_indices=window_indices,
                    source=int(source),
                )
                if not tasks:
                    continue
                out = output_root / "channel" / f"source_{source}" / "HRp10" / protocol
                label = f"channel {source}/HRp10/{protocol}"
                print(f"{label}: {len(tasks)} tasks, workers={args.workers}")
                jobs.append({
                    "kind": "channel", "tasks": tasks, "output_dir": out,
                    "label": label,
                })

    frames = stage3.run_arms(
        jobs, workers=args.workers, null_replicates=args.null_replicates
    )

    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined.to_parquet(output_root / "endpoints.parquet", index=False)
        summary = {
            "rows": int(len(combined)),
            "arms": sorted(
                {f"{int(r.source)}:{r.endpoint}:{r.protocol}" for r in combined.itertuples(index=False)}
            ),
            "converged": int(combined.get("null_converged", pd.Series(dtype=bool)).fillna(False).sum()),
        }
    else:
        summary = {"rows": 0, "arms": [], "converged": 0}
    (run_dir / "stage3_endpoints_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
