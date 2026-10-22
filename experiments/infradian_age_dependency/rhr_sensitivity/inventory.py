"""Metadata-only supplier inventory for the RHR sensitivity experiment.

The default CLI inspects path presence and CSV headers only. Raw rows and
participant identifiers are never printed. Optional scans are explicitly
bounded or streamed and emit aggregate counts only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DAILY_PATH = REPO / "data/15Sep_2230_PG_type65-66.csv"
RAW_DIR = REPO / "out"
CHANNELS = (3001, 3002)
DAILY_TYPES = (65, 66)


def csv_header(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f), None)


def raw_file_candidates(user_ids: Iterable[int], raw_dir: Path = RAW_DIR) -> dict[str, int]:
    """Count candidate per-user raw files without opening their contents."""
    ids = list(user_ids)
    present = sum((raw_dir / f"{int(uid)}.csv").is_file() for uid in ids)
    return {"candidate_ids": len(ids), "files_present": present, "files_missing": len(ids) - present}


def scan_daily_export(path: Path = DAILY_PATH, chunksize: int = 500_000) -> dict:
    """Stream daily export and return aggregate source/type coverage only.

    The scan reports literal date strings and does not infer their timezone or
    local-date semantics. It uses pandas for chunked CSV parsing.
    """
    import pandas as pd

    required = {"user_id", "date", "type", "value", "source", "timezone_offset"}
    header = csv_header(path)
    if header is None:
        return {"present": False, "status": "missing"}
    missing = sorted(required.difference(header))
    if missing:
        return {"present": True, "status": "schema_incomplete", "missing_columns": missing}

    agg: dict[tuple[int, int], dict[str, set]] = {}
    rows: dict[tuple[int, int], int] = {}
    for chunk in pd.read_csv(path, usecols=list(required), chunksize=chunksize):
        chunk = chunk.loc[chunk["type"].isin(DAILY_TYPES)]
        for (source, typ), part in chunk.groupby(["source", "type"], dropna=False):
            key = (int(source), int(typ))
            state = agg.setdefault(key, {"users": set(), "dates": set(), "date_strings": set()})
            state["users"].update(part["user_id"].dropna().astype("int64").unique().tolist())
            state["dates"].update(part["date"].dropna().astype(str).unique().tolist())
            state["date_strings"].update(part["date"].dropna().astype(str).head(1).tolist())
            rows[key] = rows.get(key, 0) + len(part)
    return {
        "present": True,
        "status": "scanned",
        "date_semantics": "unverified; literal date field only",
        "groups": [
            {"source": s, "type": t, "rows": rows[(s, t)], "unique_users": len(v["users"]),
             "unique_date_strings": len(v["dates"]), "example_date_format": next(iter(v["date_strings"]), None)}
            for (s, t), v in sorted(agg.items())
        ],
    }


def inventory(scan_daily: bool = False) -> dict:
    daily_header = csv_header(DAILY_PATH)
    raw_example = next(RAW_DIR.glob("*.csv"), None) if RAW_DIR.exists() else None
    cache_dirs = {
        "apple_channel3000": REPO / "experiments/apple_cohort_scaling/artifacts/cache_5min",
        "garmin_channel3000": REPO / "experiments/garmin_infradian_replication/artifacts/cache_5min",
        "samsung_channel3000": HERE / "artifacts/cache_5min",
    }
    result = {
        "scope": "path/schema inventory; no participant rows emitted",
        "channels": {
            str(ch): {"exported_name": name, "raw_schema": csv_header(raw_example) if raw_example else None,
                      "candidate_file_count": "not scanned"}
            for ch, name in ((3001, "HeartRateResting"), (3002, "HeartRateRestingHourly"))
        },
        "daily_export": {
            "path": str(DAILY_PATH.relative_to(REPO)), "present": DAILY_PATH.is_file(),
            "size_bytes": DAILY_PATH.stat().st_size if DAILY_PATH.is_file() else None,
            "columns": daily_header,
            "types": {"65": "RHR (export label)", "66": "HR during sleep (export label)"},
            "date_semantics": "unverified from column name/schema alone",
        },
        "channel3000_cache_presence": {
            name: {"present": path.is_dir(), "npz_files": sum(1 for _ in path.glob("customer_*.npz")) if path.is_dir() else 0}
            for name, path in cache_dirs.items()
        },
        "feasibility": "unresolved until participant/source/local-date linkage and endpoint definitions are verified",
    }
    if scan_daily:
        result["daily_export_scan"] = scan_daily_export()
    return result


def scan_raw_sample(n_files: int) -> dict:
    """Scan channel labels in at most N raw per-user files; emit counts only."""
    if n_files < 1:
        raise ValueError("n_files must be positive")
    counts: dict[tuple[int, int], dict[str, int]] = {}
    files_seen = 0
    for path in sorted(RAW_DIR.glob("*.csv"))[:n_files]:
        files_seen += 1
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"source", "type"}.issubset(reader.fieldnames):
                continue
            file_groups: set[tuple[int, int]] = set()
            for row in reader:
                try:
                    source, channel = int(row["source"]), int(row["type"])
                except (TypeError, ValueError):
                    continue
                if channel in CHANNELS:
                    key = source, channel
                    counts.setdefault(key, {"rows": 0, "files": 0})["rows"] += 1
                    file_groups.add(key)
            for key in file_groups:
                counts[key]["files"] += 1
    return {"status": "bounded_sample", "files_scanned": files_seen, "requested_files": n_files,
            "groups": [{"source": s, "channel": ch, **v} for (s, ch), v in sorted(counts.items())]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-daily", action="store_true", help="stream the full 374 MB daily export; output aggregate counts")
    parser.add_argument("--sample-raw-files", type=int, help="scan channel labels in at most N raw per-user files")
    parser.add_argument("--output", type=Path, help="write JSON inventory (otherwise print JSON)")
    args = parser.parse_args()
    data = inventory(scan_daily=args.scan_daily)
    if args.sample_raw_files:
        data["raw_file_sample"] = scan_raw_sample(args.sample_raw_files)
    rendered = json.dumps(data, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
