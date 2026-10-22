"""Stage 6: synthesis report from the run artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import common


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def _read_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_No rows._\n"
    view = df.head(max_rows)
    cols = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in view.iterrows():
        vals = []
        for c in view.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append("" if pd.isna(v) else f"{v:.4g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines) + "\n"


def run_stage6(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    prov = _read_json(run_dir / "provenance.json")
    s1b = _read_json(run_dir / "stage1b_summary.json")
    s2 = _read_json(run_dir / "stage2_summary.json")
    s3 = _read_json(run_dir / "stage3_endpoints_summary.json")
    s4 = _read_json(run_dir / "stage4_models_summary.json")
    s5 = _read_json(run_dir / "stage5_bootstrap_summary.json")
    s6 = _read_json(run_dir / "stage6_matched_summary.json")
    # Backward compatibility with the earlier smoke namespace.
    if not s4:
        s4 = _read_json(run_dir / "stage3_summary.json")
    if not s5:
        s5 = _read_json(run_dir / "stage4_summary.json")
    if not s6:
        s6 = _read_json(run_dir / "stage5_summary.json")

    inv_daily = _read_csv(run_dir / "inventory_daily.csv")
    inv_chan = _read_csv(run_dir / "inventory_channel3000.csv")
    demo_audit = _read_csv(run_dir / "manifests/demographics_audit.csv")
    elig = _read_csv(run_dir / "manifests/daily_eligibility.csv")
    chan_elig = _read_csv(run_dir / "manifests/channel_eligibility.csv")
    model_summary = _read_csv(run_dir / "models/model_summary.csv")
    age_support = _read_csv(run_dir / "models/age_support.csv")
    if not age_support.empty:
        positive = age_support["female_count"].fillna(0) + age_support["male_count"].fillna(0)
        age_support = age_support.loc[positive.gt(0)].copy()
    chunk = _read_csv(run_dir / "models/chunk_tests.csv")
    boot_tests = _read_csv(run_dir / "bootstrap/bootstrap_chunk_summary.csv")
    boot_metrics = _read_csv(run_dir / "bootstrap/transition_metrics.csv")
    long_avail = _read_csv(run_dir / "longitudinal/availability.csv")
    long_head = _read_csv(run_dir / "longitudinal/headline.csv")
    long_meta = _read_json(run_dir / "longitudinal/summary.json")
    matched_req = pd.DataFrame(s6.get("comparisons", []))
    arm_rows = []
    for path in sorted((run_dir / "matched").glob("**/arm_status.csv")):
        frame = _read_csv(path)
        if not frame.empty:
            frame["path"] = str(path.relative_to(run_dir))
            arm_rows.append(frame)
    matched_arms = pd.concat(arm_rows, ignore_index=True) if arm_rows else pd.DataFrame()
    paired_rows = []
    for path in sorted((run_dir / "matched").glob("**/paired_contrasts.parquet")):
        frame = pd.read_parquet(path)
        if not frame.empty:
            frame["path"] = str(path.relative_to(run_dir))
            paired_rows.append(frame)
    paired = pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame()
    long_avail_view = long_avail.copy()
    if not long_avail_view.empty:
        avail_cols = [
            "source_name", "endpoint", "protocol", "window1_converged_Q",
            "window2_converged_Q", "paired_converged_Q", "paired_fraction",
            "selection_smd_Q", "median_gap_days",
        ]
        long_avail_view = long_avail_view[[c for c in avail_cols if c in long_avail_view.columns]]
    long_view = long_head.copy()
    if not long_view.empty:
        long_cols = [
            "source_name", "endpoint", "protocol", "subgroup", "group", "n",
            "mean_gap_years", "mean_Q_change", "Q_change_ci_low",
            "Q_change_ci_high", "mean_Q_annualized_rate", "Q_rate_ci_low",
            "Q_rate_ci_high", "pearson_Q_i", "pearson_Q_i_ci_low",
            "pearson_Q_i_ci_high", "stable",
        ]
        long_view = long_view[[c for c in long_cols if c in long_view.columns]]

    lines = [
        "# Multisource infradian replication — implementation report",
        "",
        f"Run: `{run_dir.name}`  ",
        f"Generated: {common.utc_now_iso()}  ",
        "Status: implementation output. Age associations are observational and source-specific.",
        "",
        "## Provenance gate",
        "",
        f"- Status: `{prov.get('status', 'missing')}`",
        f"- Source namespace: {prov.get('checks', {}).get('source_namespace', 'unknown')}",
        f"- Value-type namespace: {prov.get('checks', {}).get('value_type_namespace', 'unknown')}",
        f"- Thryve warning: {prov.get('checks', {}).get('thryve_namespace_warning', '')}",
        f"- Inventory-only unresolved sources present: "
        f"{prov.get('checks', {}).get('unresolved_sources_present', [])}; "
        "they are excluded from endpoint analysis.",
        f"- Daily QC: {prov.get('daily_qc', {})}",
        f"- Duplicate policy: {prov.get('duplicates', {})}",
        "",
        "### Channel cache index",
        "",
        f"- Candidate index rows: {s1b.get('raw_index_rows', 0)}; caches available: {s1b.get('caches_built', 0)}.",
        f"- By source: {s1b.get('by_source', {})}",
        "",
        "### Daily inventory (type 65 / type 66)",
        "",
        _md_table(inv_daily),
        "### Channel-3000 inventory",
        "",
        _md_table(inv_chan),
        "### Demographic exclusions",
        "",
        _md_table(demo_audit, max_rows=40),
        "## Eligibility",
        "",
        f"Daily arms: {s2.get('daily_eligible_windows', 0)} eligible windows across "
        f"{s2.get('daily_sources', [])}.",
        f"Endpoint rows: {s3.get('rows', 0)}; converged: {s3.get('converged', 0)}; "
        f"arms: {s3.get('arms', [])}.",
        "",
        _md_table(elig, max_rows=80),
        "### Channel-3000 eligibility",
        "",
        _md_table(chan_elig, max_rows=80),
        "## Age models",
        "",
        f"Fitted models: {s4.get('models_fitted', 0)}.",
        "",
        _md_table(model_summary, max_rows=80),
        "### Age support",
        "",
        _md_table(age_support, max_rows=80),
        "### Chunk tests (point estimates)",
        "",
        _md_table(chunk, max_rows=80),
        "## Bootstrap",
        "",
        f"Replicates: {s5.get('n_boot', common.N_BOOTSTRAP)}; arms: {s5.get('arms', 0)}; "
        f"jobs: {s5.get('jobs', 0)}.",
        "",
        "Bootstrap chunk summaries describe the participant-resampling distribution "
        "of the fitted Wald statistic. They are **not null p-values**. Hypothesis "
        "tests use the HC3 `p_chi2`/`p_f` columns in the chunk-test table.",
        "",
        _md_table(boot_tests.drop(columns=["inference"], errors="ignore"), max_rows=80),
        "### Transition metrics",
        "",
        _md_table(boot_metrics, max_rows=40),
        "## Longitudinal repeat windows",
        "",
    ]
    if long_meta:
        lines += [
            f"Paired participant-arm observations: **{long_meta.get('n_pairs', 0)}**; "
            f"unique source-participants: "
            f"**{long_meta.get('n_unique_source_participants', 'NA')}** "
            f"(**{long_meta.get('n_unique_source_participants_age_20_75', 'NA')}** "
            f"in the `[20,75)` summary stratum); arms: "
            f"**{long_meta.get('n_arms', 0)}**; participant bootstrap resamples: "
            f"**{long_meta.get('n_boot', 'NA')}**. Counts across arms are not "
            "distinct people; a participant can contribute multiple endpoint or "
            "protocol arms.",
            "",
            "These are paired within-participant summaries for participants with two "
            "eligible, converged, non-overlapping windows under the same source, "
            "endpoint, and protocol. Age is first-window age. Intervals are pointwise "
            "participant-bootstrap percentile intervals stratified by five-year age "
            "band and sex. They are not causal aging effects and are not longitudinal "
            "replications of the cross-sectional age curves. The availability table "
            "reports the paired fraction and the standardized difference in "
            "first-window `Q_i` between paired and all eligible participants; a large "
            "value indicates selection by two-window availability. `Q_i` retains "
            "finite-null Monte Carlo error; the paired table carries `Q_i_mc_se_w1/w2` "
            "and the summary reports their means.",
            "",
            "### Repeat-window availability",
            "",
            _md_table(long_avail_view, max_rows=40),
            "### Paired change and repeatability (all and sex strata)",
            "",
            _md_table(long_view, max_rows=80),
        ]
    else:
        lines += [
            "_No repeat-window artifacts in this run namespace. The primary-window "
            "analysis does not estimate within-participant change._",
            "",
        ]
    lines += [
        "## Matched comparisons",
        "",
        _md_table(matched_req, max_rows=40),
        "### Per-arm retention and fit status",
        "",
        _md_table(matched_arms, max_rows=80),
        "### Paired-difference intervals",
        "",
        _md_table(paired, max_rows=40),
        "## Interpretation limits",
        "",
        "- Arms A (channel-3000 HRp10), B (type 65 RHR), and C (type 66 sleep HR) have different ascertainment and must not be pooled.",
        "- Daily summaries have no within-day mask; their nuisance design omits clock-coverage fractions.",
        "- Cross-sectional age associations do not identify within-person decline or menopause onset.",
        "- Repeat-window change is a short-interval within-participant association, not a "
        "causal aging effect; two-window eligibility is a selected subset and intervals "
        "are not adjusted for multiple comparisons.",
        "- Supplier labels and units are accepted only under the recorded provenance gate.",
        "- No endpoint is imputed and no conflicting daily value is averaged.",
        "",
    ]
    out = run_dir / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
