"""Pre-fit assertions and reproducibility record.

Fails loudly if any plan completion check is violated.  Writes
artifacts/reproducibility.json and prints a verification summary.
Run:  python src/verify.py
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    ARTIFACTS, PROJECT, DAILY_CSV, SALUTATION_CSV, WHO_CSV, EPOCH_DIR,
    EXCLUDED_SOURCES, MIN_ADEQUATE_DAYS, RF_SEED, SEED, BOOTSTRAP_SEED, KEEP_SALUTATIONS,
    ADEQUATE_HOURS, ADEQUATE_COV_H, ADEQUATE_N, INCLUDE_DEMOGRAPHIC_PREDICTORS,
)

FORBIDDEN = {"salutation", "gender", "birth_date", "weight", "height", "age",
             "bmi_grp", "age_group", "who_average", "n_session", "user_id"}
checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    man = pd.read_parquet(ARTIFACTS / "cohort_manifest.parquet")
    split = pd.read_csv(ARTIFACTS / "split_manifest.csv")
    feats = pd.read_parquet(ARTIFACTS / "epoch_features.parquet")
    esrc = pd.read_parquet(ARTIFACTS / "epoch_sources.parquet")
    days = pd.read_parquet(ARTIFACTS / "epoch_days.parquet")
    el = man[man["eligible"]]

    # 1) labels exclusively 10/20
    check("labels_exclusively_10_20",
          set(el["salutation"].unique()) <= set(KEEP_SALUTATIONS),
          sorted(el["salutation"].unique().tolist()))
    check("code_30_absent_from_eligible", (el["salutation"] == 30).sum() == 0)

    # 2) sources 38/46/48 contribute nothing
    bad_daily = [c for c in man.columns
                 if c.startswith("daily_src_") and c.split("_")[-1] in [str(s) for s in EXCLUDED_SOURCES]]
    check("no_excluded_daily_source_columns", not bad_daily, bad_daily)
    check("no_excluded_epoch_sources", not set(esrc["source"].unique()) & set(EXCLUDED_SOURCES),
          sorted(set(esrc["source"].unique())))

    # 3) every eligible passes gate and appears exactly once
    check("all_eligible_pass_gate", bool((el["gate_pass"]).all()))
    check("all_eligible_adequate_ge_min",
          bool((el["adequate_days_max"] >= MIN_ADEQUATE_DAYS).all()))
    check("eligible_once_in_split", split["user_id"].nunique() == len(split) == len(el))
    check("split_users_equal_eligible", set(split["user_id"]) == set(el["user_id"]))

    # 4) disjoint splits and complete coverage
    sets = {s: set(split.loc[split.split == s, "user_id"]) for s in ["train", "val", "test"]}
    check("splits_disjoint",
          not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"])
          and not (sets["val"] & sets["test"]))
    check("splits_complete", len(sets["train"] | sets["val"] | sets["test"]) == len(el))

    # 5) feature ownership and no prohibited predictors
    check("feature_rows_match_eligible", set(feats["user_id"]) == set(el["user_id"])
          and len(feats) == len(el))
    feat_cols = [c for c in feats.columns if c != "user_id"]
    allowed = {"age_group", "bmi_grp"} if INCLUDE_DEMOGRAPHIC_PREDICTORS else set()
    bad = (FORBIDDEN - allowed) & set(feat_cols)
    check("no_prohibited_metadata_in_features", not bad, sorted(bad))
    check("no_salutation_in_features", "salutation" not in feat_cols and "y" not in feat_cols)

    # 6) numerical features reach the estimator (re-load saved feature names)
    fn = json.load(open(ARTIFACTS / "feature_names.json"))
    check("numerical_features_transformed", any(n.startswith("num__") for n in fn),
          f"{len(fn)} transformed features")

    # 7) coverage-day rule recomputation consistency
    d = days.copy()
    d["adequate"] = ((d["hours"] >= ADEQUATE_HOURS) &
                     ((d["cov_s"] >= ADEQUATE_COV_H * 3600) | (d["n"] >= ADEQUATE_N)))
    rec = d[d["adequate"]].groupby(["user", "channel"]).size().groupby("user").max()
    elig = el.set_index("user_id")["adequate_days_max"].astype(int)
    rec_el = rec.reindex(elig.index).fillna(0).astype(int)
    check("adequate_day_recomputation_matches", bool((elig.values == rec_el.values).all()),
          f"max abs diff {int(np.abs(elig.values - rec_el.values).max())}")

    # reproducibility record
    def real(p):
        p = Path(p)
        try:
            return str(p.resolve())
        except Exception:  # noqa: BLE001
            return str(p)

    inv = {}
    for p in [DAILY_CSV, SALUTATION_CSV, WHO_CSV, EPOCH_DIR]:
        rp = real(p)
        if Path(rp).is_dir():
            files = os.listdir(rp)
            inv[str(p)] = {"resolved": rp, "kind": "dir", "n_files": len(files)}
        else:
            inv[str(p)] = {"resolved": rp, "kind": "file",
                           "size_bytes": os.path.getsize(rp) if os.path.exists(rp) else None}

    try:
        git = subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        git = None

    record = {
        "inputs": inv,
        "seed": SEED, "rf_seed": RF_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
        "package_versions": {
            "python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "platform": platform.platform()},
        "skill": "salutation-epoch-rf-v1",
        "eligibility": {"min_adequate_days": MIN_ADEQUATE_DAYS,
                        "adequate_hours": ADEQUATE_HOURS, "adequate_cov_h": ADEQUATE_COV_H,
                        "adequate_n": ADEQUATE_N, "excluded_sources": EXCLUDED_SOURCES},
        "git_commit": git,
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
    }
    with open(ARTIFACTS / "reproducibility.json", "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nall checks passed: {record['all_passed']}")
    if not record["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
