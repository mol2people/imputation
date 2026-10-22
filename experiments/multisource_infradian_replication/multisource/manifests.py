"""Stage 2 orchestration: manifests, eligibility, and age support."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import channel3000, common, daily

BAND_EDGES = np.arange(0, 125, common.AGE_BAND_WIDTH)
BAND_LABELS = [
    f"[{BAND_EDGES[i]},{BAND_EDGES[i + 1]})" for i in range(len(BAND_EDGES) - 1)
]


def age_support_table(windows: pd.DataFrame) -> pd.DataFrame:
    """5-year-band female and male counts on eligible windows."""
    if windows.empty:
        return pd.DataFrame()
    eligible = windows.loc[windows["is_eligible"] & windows["age"].notna()].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible["age_band"] = pd.cut(
        eligible["age"], bins=BAND_EDGES, right=False, labels=BAND_LABELS
    )
    rows = []
    for keys, group in eligible.groupby(
        ["source", "endpoint", "protocol", "window_index"], sort=True
    ):
        ct = pd.crosstab(group["age_band"], group["sex"])
        for band in BAND_LABELS:
            n_f = int(ct.loc[band, 1]) if band in ct.index and 1 in ct.columns else 0
            n_m = int(ct.loc[band, 0]) if band in ct.index and 0 in ct.columns else 0
            rows.append(
                {
                    "source": int(keys[0]),
                    "source_name": common.source_name(int(keys[0])),
                    "endpoint": str(keys[1]),
                    "protocol": str(keys[2]),
                    "window_index": int(keys[3]),
                    "age_band": band,
                    "female_count": n_f,
                    "male_count": n_m,
                    "total_count": n_f + n_m,
                    "is_female_support": n_f >= common.MIN_FEMALE_PER_BAND,
                    "is_shared_support": (
                        n_f >= common.MIN_FEMALE_PER_BAND
                        and n_m >= common.MIN_MALE_PER_BAND
                    ),
                }
            )
    return pd.DataFrame(rows)


def shared_support_bands(support: pd.DataFrame, source: int, endpoint: str,
                         protocol: str, window_index: int) -> list[str]:
    subset = support.loc[
        support["source"].eq(int(source))
        & support["endpoint"].eq(endpoint)
        & support["protocol"].eq(protocol)
        & support["window_index"].eq(int(window_index))
        & support["is_shared_support"]
    ]
    return subset["age_band"].astype(str).tolist()


def run_stage2(
    run_dir: Path,
    sources: Iterable[int] = common.SCOPE_SOURCES,
    protocols: Iterable[str] = common.PROTOCOLS,
    cache_dir: Path | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    """Build daily manifests and, when a cache index exists, channel manifests."""
    run_dir = Path(run_dir)
    manifests_dir = run_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    daily_clean = pd.read_parquet(run_dir / "daily_clean.parquet")
    demographics = common.load_demographics()

    windows, selected = daily.build_daily_manifests(
        daily_clean, demographics, sources=sources, protocols=protocols, workers=workers
    )
    windows.to_parquet(manifests_dir / "daily_windows.parquet", index=False)
    selected.to_parquet(manifests_dir / "daily_selected_dates.parquet", index=False)
    elig = daily.eligibility_summary(windows)
    elig.to_csv(manifests_dir / "daily_eligibility.csv", index=False)
    support = age_support_table(windows)
    support.to_csv(manifests_dir / "daily_age_support.csv", index=False)

    channel_summary: dict[str, Any] = {"built": False}
    cache_index_path = run_dir / "cache_index.parquet"
    if cache_dir is not None and cache_index_path.exists():
        cache_index = pd.read_parquet(cache_index_path)
        cw, cs = channel3000.build_channel_manifests(
            cache_index, demographics, Path(cache_dir), protocols=protocols, workers=workers
        )
        if not cw.empty:
            cw.to_parquet(manifests_dir / "channel_windows.parquet", index=False)
            cs.to_parquet(manifests_dir / "channel_selected_dates.parquet", index=False)
            daily.eligibility_summary(cw).to_csv(
                manifests_dir / "channel_eligibility.csv", index=False
            )
            age_support_table(cw).to_csv(
                manifests_dir / "channel_age_support.csv", index=False
            )
            channel_summary = {
                "built": True,
                "windows": int(len(cw)),
                "eligible": int(cw["is_eligible"].sum()),
                "sources": sorted(int(s) for s in cw["source"].unique()),
            }

    demographics.to_parquet(manifests_dir / "participants.parquet", index=False)
    common.audit_demographics().to_csv(manifests_dir / "demographics_audit.csv", index=False)
    summary = {
        "daily_windows": int(len(windows)),
        "daily_eligible_windows": int(windows["is_eligible"].sum()) if not windows.empty else 0,
        "daily_sources": sorted(int(s) for s in windows["source"].unique()) if not windows.empty else [],
        "daily_endpoints": sorted(windows["endpoint"].unique().tolist()) if not windows.empty else [],
        "channel": channel_summary,
    }
    (run_dir / "stage2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
