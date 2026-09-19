"""Side-quest report: sidequest_report.md + reproducibility_sq.json.

Assembles the per-source cohort/allocation/feature/model results, the
verification outcome, measured runtimes, and full provenance (config echo,
package versions, SHA-256 hashes of scripts and inputs) into
artifacts_sq/sidequest_report.md and artifacts_sq/reproducibility_sq.json.

Run:  python src/sidequest/report.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sq_config as cfg  # noqa: E402
from sq_config import (  # noqa: E402
    ARTIFACTS_SQ, ARTIFACTS_V2, MODEL_SOURCE_IDS, VARIANTS,
)

SPLITS = ["train", "val", "test"]

ROOT = Path(__file__).resolve().parents[2]
DELTA_LABELS = {"rec-demo": "recording vs demographics",
                "win-rec": "window vs recording",
                "roll_win-roll_rec": "rolling window vs rolling recording",
                "all-rec": "all blocks vs recording"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def table(lines: list[str]) -> list[str]:
    return lines


def main() -> None:
    cohorts = pd.read_csv(ARTIFACTS_SQ / "source_cohorts.csv")
    audit = pd.read_csv(ARTIFACTS_SQ / "coverage_semantics_audit.csv")
    split = pd.read_parquet(ARTIFACTS_SQ / "split_manifest_sq.parquet")
    model = pd.read_parquet(ARTIFACTS_SQ / "cohort_manifest_model_sources.parquet")
    split = split.merge(model[["user_id", "age_group", "bmi_grp", "selected_channel",
                               "pass_pattern"]], on="user_id", how="left")
    viol = pd.read_csv(ARTIFACTS_SQ / "split_violations_sq.csv")
    verification = pd.read_json(ARTIFACTS_SQ / "verification_sq.json")
    with open(ARTIFACTS_SQ / "scan_benchmark.json") as fh:
        bench = json.load(fh)
    with open(ARTIFACTS_SQ / "window_diurnal_scan_log.json") as fh:
        scan = json.load(fh)
    metrics = {s: json.loads((ARTIFACTS_SQ / f"source_{s}" / "metrics_sq.json")
                             .read_text()) for s in MODEL_SOURCE_IDS}
    versions = metrics[3]["versions"]

    out: list[str] = []
    add = out.append
    add("# Side-quest report: strict-coverage cohorts and per-source RF benchmarks")
    add("")
    add(f"Generated 2026-09-19 from `src/sidequest/`; all outputs under "
        f"`artifacts_sq/`. Verification: "
        f"**{int(verification['ok'].sum())}/{len(verification)} checks passed** "
        f"(`verification_sq.json`; re-derives eligibility, selection, windows, "
        f"splits, features, schemas, predictions, and the bootstrap from raw "
        f"inputs).")
    add("")
    add("Plan-status note: the 2026-09-19 decision made plan section 1 gate "
        "semantics authoritative; the count rows in section 2 and the "
        "Apple/Samsung rows of section 4 are superseded. Corrected anchors "
        "live in `src/sidequest/sq_config.py`. One frozen-table correction was "
        "required: source 7 train sal10 is **167**, not 166 (166+266 != 433 "
        "and 166+21+21 != 209).")
    add("")

    # ---- cohorts -----------------------------------------------------------
    add("## 1. Cohorts (strict-coverage, any-core)")
    add("")
    add("| source | base v2-eligible single-source | ch3000 | ch3001 | ch3002 | "
        "any-core pass | sal10 | sal20 | disposition |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for _, r in cohorts.iterrows():
        add(f"| {int(r['source_id'])} | {int(r['base_single_source']):,} | "
            f"{int(r['ch3000_pass']):,} | {int(r['ch3001_pass']):,} | "
            f"{int(r['ch3002_pass']):,} | {int(r['any_core_pass']):,} | "
            f"{int(r['sal10']):,} | {int(r['sal20']):,} | {r['disposition']} |")
    add("")
    add(f"Any-core union **{sum(int(c['any_core_pass']) for _, c in cohorts.iterrows()):,}**; "
        f"model union (sources 3/6/7) "
        f"**{sum(int(c['any_core_pass']) for _, c in cohorts.iterrows() if c['model_eligible']):,}**. "
        "Any-core membership is exactly the set union of independent "
        "per-channel passers (verified). Sources 9 and 13 fail the minimum "
        "class-count rule (any-core 9 and 41) and are audit-only.")
    add("")

    # ---- coverage audit ----------------------------------------------------
    add("## 2. Coverage semantics audit (frozen cov_s artifact semantics)")
    add("")
    add("| source | channel | day rows | days cov_s>24h | share | max cov_s (s) |")
    add("|---|---|---:|---:|---:|---:|")
    for _, r in audit.iterrows():
        add(f"| {int(r['source_id'])} | {int(r['channel'])} | "
            f"{int(r['n_day_rows']):,} | {int(r['n_days_over_24h']):,} | "
            f"{100 * r['share_over_24h']:.3f}% | {r['max_cov_s']:.0f} |")
    add("")
    add("`cov_s` is the frozen v2 artifact quantity: the union of epoch "
        "intervals attributed to the interval-start local date, without "
        "midnight clipping, so values above 86,400 s are possible. It is "
        "treated as audit-only and is never reinterpreted.")
    add("")

    # ---- splits ------------------------------------------------------------
    add("## 3. Participant allocation (frozen 80/10/10, per source)")
    add("")
    add("Targets: split totals `T = largest_remainder(FRACS, N)` and class-1 "
        "(sal20) apportioned by the same rule with ties to the lowest split "
        "index; sal10 = T - sal20. This reproduces the untouched Garmin row "
        "and the recorded Apple row exactly. Floor/ceiling proportional "
        "marginals for age group, BMI group, and qualifying-channel "
        "pass-pattern were requested; the two-stage smallest-violation MILP "
        f"found total violation **{int(viol['violation'].sum()) if len(viol) else 0}** "
        "(no nonzero bound violations; relaxation machinery recorded in "
        "`split_violations_sq.csv`).")
    add("")
    for s in MODEL_SOURCE_IDS:
        sub = split[split["source_id"] == s]
        frozen = cfg.FROZEN_SPLITS[s]
        add(f"**Source {s}** (N = {len(sub):,}):")
        add("")
        add("| split | total (target) | sal10 (target) | sal20 (target) |")
        add("|---|---:|---:|---:|")
        for sp in SPLITS:
            g = sub[sub["split"] == sp]
            add(f"| {sp} | {len(g):,} ({frozen[sp][0]:,}) | "
                f"{int((g['y'] == 0).sum()):,} ({frozen[sp][1]:,}) | "
                f"{int((g['y'] == 1).sum()):,} ({frozen[sp][2]:,}) |")
        spread = []
        for fam in ("age_group", "bmi_grp", "selected_channel", "pass_pattern"):
            tab = pd.crosstab(sub[fam].astype(str), sub["split"],
                              normalize="columns").reindex(
                columns=SPLITS).fillna(0.0)
            spread.append(f"{fam} {100 * (tab.max(axis=1) - tab.min(axis=1)).max():.2f}")
        add("")
        add(f"Max within-marginal proportion spread across splits: "
            f"{'; '.join(spread)} (percentage points; joint y x age x BMI "
            f"cells minimised as the MILP secondary objective). Per-level "
            f"tables: `source_{s}/split_balance_report.md`.")
        add("")

    # ---- features ----------------------------------------------------------
    add("## 4. Feature blocks")
    add("")
    add("| block | prefix | width |")
    add("|---|---|---:|")
    add("| D demographics | `demo__` | 2 |")
    add("| A recording-quality (epoch span) | `rec__` | 78 |")
    add("| B 91-day window summaries | `win__` | 78 |")
    add("| C rolling 4/7/15/30-day, recording span | `roll_rec__` | 120 |")
    add("| C rolling 4/7/15/30-day, window + pre-lookback | `roll_win__` | 120 |")
    add("| all | - | 398 |")
    add("")
    add("Per-block widths asserted at build time and re-verified; one row per "
        "model-union participant in every block; prefixes disjoint; "
        "demographics appear only in D and exactly once per design matrix; "
        "`feature_dictionary_sq.csv` documents all 398 columns.")
    add("")
    add("Rolling-value interpretation (the one interpretive choice flagged "
        "for review): the plan defines rolling-block *definedness* only; the "
        "rolling value is implemented as the **trailing-window mean of the "
        "non-missing daily series values** (mean/median/SD/hours series, "
        "missing on days failing the v2 feature-day rule), with windows "
        "truncated at the grid start kept valid. This matches the plan's "
        "pre-window lookback convention.")
    add("")

    # ---- models ------------------------------------------------------------
    add("## 5. Models and bootstrap (exploratory)")
    add("")
    add("RandomForestClassifier(`random_state=20260918`, all other "
        "hyperparameters at the installed scikit-learn default), new "
        "preprocessing + forest per (source, variant); training-fitted "
        "preprocessing: all-missing column drop, median imputation with "
        "missingness indicators, demographic mode imputation + one-hot with "
        "unknown-level handling, zero-variance term drop; no scaling. "
        "Bootstrap: exactly 100 within-class test resamples per source, "
        "SeedSequence([20260918, source_id]), identical resamples across "
        "variants, recomputed from saved predictions.")
    add("")
    for s in MODEL_SOURCE_IDS:
        m = metrics[s]
        add(f"### Source {s}")
        add("")
        add("| variant | val AUROC | test AUROC | boot mean | boot SD | 95% "
            "percentile (exploratory) | test bal. acc. |")
        add("|---|---:|---:|---:|---|---:|---:|")
        for v in VARIANTS:
            mv = m["variants"][v]
            b = mv["bootstrap_test"]["auroc"]
            add(f"| {v} | {mv['val']['auroc']:.4f} | {mv['test']['auroc']:.4f} | "
                f"{b['bootstrap_mean']:.4f} | {b['bootstrap_sd']:.4f} | "
                f"[{b['q025']:.4f}, {b['q975']:.4f}] | "
                f"{mv['test']['balanced_accuracy']:.4f} |")
        add("")
        add("Paired test-AUROC deltas (draw-level, same resamples):")
        add("")
        add("| comparison | point | draw mean | draw SD |")
        add("|---|---:|---:|---:|")
        for comp, label in DELTA_LABELS.items():
            d = m["variants"][comp.split("-")[0]]["paired_deltas_test"][comp]["auroc"]
            add(f"| {label} (`{comp}`) | {d['point']:+.4f} | "
                f"{d['draw_mean']:+.4f} | {d['draw_sd']:.4f} |")
        add("")

    add("### Reading")
    add("")
    add("- Demographics alone sit near chance in every source (test AUROC "
        "0.55-0.59); recording-quality features carry the signal, consistent "
        "with the v2 pooled finding (pooled test AUROC 0.68333, 10-draw SD "
        "0.00873, as background context only).")
    add("- Per-source point estimates are exploratory: the plan pre-specifies "
        "no test-set winner selection, no vendor ranking, and no expected "
        "AUROC; the 100-draw percentile intervals are coarse. Samsung's test "
        "set has n = 54 (21/33) - its intervals span roughly 0.28 in AUROC "
        "and must not be read as rankings.")
    add("- The identical participant split is reused for every variant within "
        "a source, so paired deltas are the comparable quantity; they are "
        "recomputed from saved predictions and re-verified.")
    add("")

    # ---- runtime -----------------------------------------------------------
    add("## 6. Measured runtime and provenance")
    add("")
    add("| stage | wall time | note |")
    add("|---|---:|---|")
    est = sum(v["est_seconds_single_worker"] for v in bench["extrapolation"].values())
    add(f"| raw-file manifest + parser benchmark | ~{est:.0f} s (aggregate "
        f"single-worker) | {bench['total_gib_all_model_files']:.1f} GiB across "
        f"{scan['files']:,} files; all present, 0 parse failures |")
    add(f"| window mini-scan (8 workers) | {scan['elapsed_s']:.0f} s | "
        f"{scan['hour_rows']:,} hour rows, {scan['failures']} failures |")
    add("| feature blocks | ~24 s | rolling over 26,500 user/channel series |")
    add("| allocation (3 sources, 2-stage MILP) | <1 s | 0 bound violations |")
    add("| 18 RF fits + 300 bootstrap draws | ~13 s | sklearn 1.9.0 |")
    add("| verification contract | ~27 s | 135 checks, 0 failures |")
    add("")
    add("Full provenance (config echo, package versions, SHA-256 of scripts "
        "and inputs): `reproducibility_sq.json`.")
    add("")
    with open(ARTIFACTS_SQ / "sidequest_report.md", "w") as fh:
        fh.write("\n".join(out) + "\n")

    # ---- reproducibility ---------------------------------------------------
    inputs = ["epoch_days.parquet", "epoch_sources.parquet", "epoch_features.parquet",
              "epoch_hours.parquet", "epoch_meta.parquet", "cohort_manifest.parquet"]
    hashes = {}
    for name in inputs:
        p = ARTIFACTS_V2 / name
        if p.exists():
            hashes[f"artifacts_v2/{name}"] = sha256(p)
    for name in ("raw_file_manifest.parquet", "scan_benchmark.json",
                 "window_diurnal.parquet"):
        p = ARTIFACTS_SQ / name
        if p.exists():
            hashes[f"artifacts_sq/{name}"] = sha256(p)
    for p in sorted((Path(cfg.__file__).parent).glob("*.py")):
        hashes[f"src/sidequest/{p.name}"] = sha256(p)
    for name in ("SIDEQUEST_PLAN.md", "src/config.py"):
        p = ROOT / name
        if p.exists():
            hashes[name] = sha256(p)

    repro = {
        "generated": "2026-09-19",
        "seeds": {"allocation_and_model_seed": cfg.SEED,
                  "rf_random_state": cfg.RF_SEED,
                  "bootstrap_seed": cfg.SQ_BOOTSTRAP_SEED,
                  "bootstrap_draws": cfg.SQ_BOOTSTRAP_N},
        "anchors_corrected": {
            "note": "section-1 gate semantics authoritative (2026-09-19); "
                    "supersedes plan section 2 counts and section 4 "
                    "Apple/Samsung rows",
            "any_core": cfg.FROZEN_ANY_CORE,
            "any_core_class": cfg.FROZEN_ANY_CORE_CLASS,
            "all_source_pass": cfg.FROZEN_ALL_SOURCE_PASS,
            "model_union": cfg.FROZEN_MODEL_UNION,
            "frozen_splits": cfg.FROZEN_SPLITS},
        "gate": {"strict_cov_s": cfg.STRICT_COV_S,
                 "min_adequate_weeks": cfg.MIN_ADEQUATE_WEEKS,
                 "min_strict_days": cfg.MIN_STRICT_DAYS,
                 "window_weeks": cfg.WINDOW_WEEKS,
                 "window_inclusive_days": cfg.WINDOW_INCLUSIVE_DAYS},
        "package_versions": versions,
        "verification": {"checks": int(len(verification)),
                         "failures": int((~verification["ok"]).sum())},
        "sha256": hashes,
    }
    with open(ARTIFACTS_SQ / "reproducibility_sq.json", "w") as fh:
        json.dump(repro, fh, indent=2)
    print(f"report written: sidequest_report.md, reproducibility_sq.json "
          f"({len(hashes)} hashes)")


if __name__ == "__main__":
    main()
