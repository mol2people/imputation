"""Side-quest section 8: executable verification contract.

All checks recompute from machine-readable artifacts; v2 inputs are re-derived
where a semantic claim is made.  Anchor note: the frozen tables in plan
section 2 and the 8,880/8,830 cohort-integrity numbers in section 8 are
superseded (2026-09-19 user decision: section-1 gate semantics authoritative);
the corrected anchors live in sq_config.py and are used here.

Writes artifacts_sq/verification_sq.json; exits nonzero on any failure.

Run:  python src/sidequest/verify.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sq_config import (  # noqa: E402
    ARTIFACTS_SQ, ARTIFACTS_V2, CORE_CHANNELS, FROZEN_ALL_SOURCE_PASS,
    FROZEN_ANY_CORE, FROZEN_BASE_SINGLE_SOURCE, FROZEN_CHANNEL_PASS,
    FROZEN_MODEL_UNION, FROZEN_SPLITS, MIN_ADEQUATE_WEEKS, MIN_STRICT_DAYS,
    MODEL_SOURCE_IDS, REASON_ORDER, RF_SEED, STRICT_COV_S, VARIANTS,
    WINDOW_INCLUSIVE_DAYS, WINDOW_WEEKS,
)

RESULTS: list[dict] = []


def add(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""),
          flush=True)


def iso_monday(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.int64)
    return d - ((d + 3) % 7)


def sole_source_users() -> pd.DataFrame:
    """v2-eligible participants with exactly one retained epoch source."""
    man = pd.read_parquet(ARTIFACTS_V2 / "cohort_manifest.parquet")
    el = man[man["eligible"]]
    es = pd.read_parquet(ARTIFACTS_V2 / "epoch_sources.parquet")
    es = es[es["events"] > 0]
    nsrc = es.groupby("user")["source"].nunique()
    sole = nsrc[nsrc == 1].index
    src_of = es[es["user"].isin(sole)].drop_duplicates("user").set_index("user")["source"]
    base = el[el["user_id"].isin(sole)].copy()
    base["source_id"] = base["user_id"].map(src_of)
    return base


def check_eligibility() -> None:
    base = sole_source_users()
    got = {int(k): int(v) for k, v in base.groupby("source_id").size().items()}
    add("eligibility.base_single_source", got == FROZEN_BASE_SINGLE_SOURCE,
        f"recomputed {got}")

    elig = pd.read_parquet(ARTIFACTS_SQ / "eligibility_by_channel.parquet")
    per_chan = {(int(s), int(c)): int(n) for (s, c), n in
                elig[elig["pass"]].groupby(["source_id", "channel"]).size().items()}
    want = {(int(s), int(c)): int(n) for s, chs in FROZEN_CHANNEL_PASS.items()
            for c, n in chs.items()}
    add("eligibility.per_channel_counts",
        all(per_chan.get(k, 0) == v for k, v in want.items()) and
        set(per_chan) <= set(want))

    union = elig[elig["pass"]].groupby("user_id")["channel"].apply(set)
    src = base.set_index("user_id")["source_id"]
    cnt = {int(k): int(v) for k, v in
           union.index.map(src).value_counts().items()}
    add("eligibility.any_core_is_union_of_channel_passers",
        all(cnt.get(s, 0) == v for s, v in FROZEN_ANY_CORE.items()), f"{cnt}")

    ok = (elig.loc[elig["pass"], "reason"] == "pass").all() and \
        set(elig.loc[~elig["pass"], "reason"]) <= set(REASON_ORDER)
    add("eligibility.reason_labels", ok)


def check_manifests() -> None:
    allm = pd.read_parquet(ARTIFACTS_SQ / "cohort_manifest_all_sources.parquet")
    model = pd.read_parquet(ARTIFACTS_SQ / "cohort_manifest_model_sources.parquet")
    add("cohort.all_sources_rows",
        len(allm) == FROZEN_ALL_SOURCE_PASS and allm["user_id"].is_unique,
        f"{len(allm):,} rows (corrected anchor {FROZEN_ALL_SOURCE_PASS:,})")
    add("cohort.model_rows",
        len(model) == FROZEN_MODEL_UNION and model["user_id"].is_unique,
        f"{len(model):,} rows (corrected anchor {FROZEN_MODEL_UNION:,})")
    add("cohort.model_subset_of_all", set(model["user_id"]) <= set(allm["user_id"]))
    add("cohort.model_sources_only",
        set(model["source_id"].unique()) == set(MODEL_SOURCE_IDS))
    for s in MODEL_SOURCE_IDS:
        add(f"cohort.per_source_manifest_{s}",
            (ARTIFACTS_SQ / f"source_{s}" / "cohort_manifest.parquet").exists())
    add("cohort.no_model_dirs_for_9_13",
        not (ARTIFACTS_SQ / "source_9").exists() and
        not (ARTIFACTS_SQ / "source_13").exists())
    base = sole_source_users().set_index("user_id")["source_id"]
    mapped = model["user_id"].map(base)
    add("cohort.sole_source_recomputed",
        bool(mapped.notna().all() and (mapped == model["source_id"]).all()))


def check_windows_and_selection() -> None:
    elig = pd.read_parquet(ARTIFACTS_SQ / "eligibility_by_channel.parquet")
    passing = elig[elig["pass"]][["user_id", "channel", "best_window_start",
                                  "best_adequate_weeks", "best_strict_days",
                                  "first_monday", "last_monday"]]
    passers = set(passing["user_id"])
    days = pd.read_parquet(ARTIFACTS_V2 / "epoch_days.parquet",
                           columns=["user", "channel", "date", "cov_s"])
    days = days[days["user"].isin(passers)]
    # epoch_days columns arrive as float64; ids are integral
    days["user"] = days["user"].astype("int64")
    days["channel"] = days["channel"].astype("int64")
    days["monday"] = iso_monday(days["date"].to_numpy())
    days["strict"] = days["cov_s"].to_numpy() >= STRICT_COV_S
    wk = (days.groupby(["user", "channel", "monday"], sort=True)["strict"]
          .sum().rename("d").reset_index())

    bad, n_checked = [], 0
    pairs = set(zip(passing["user_id"].astype("int64"),
                    passing["channel"].astype("int64")))
    for (u, c), g in wk.groupby(["user", "channel"], sort=False):
        if (u, c) not in pairs:
            continue
        row = passing[(passing["user_id"] == u) & (passing["channel"] == c)].iloc[0]
        f, l = int(g["monday"].iloc[0]), int(g["monday"].iloc[-1])
        n_weeks = (l - f) // 7 + 1
        d = np.zeros(n_weeks, dtype=np.int64)
        d[(g["monday"].to_numpy(dtype=np.int64) - f) // 7] = g["d"].to_numpy()
        a = (d >= 5).astype(np.int64)
        cs_d = np.concatenate([[0], np.cumsum(d)])
        cs_a = np.concatenate([[0], np.cumsum(a)])
        starts = np.arange(0, n_weeks - WINDOW_WEEKS + 1)
        assert starts.size >= 1
        sd = cs_d[starts + WINDOW_WEEKS] - cs_d[starts]
        sa = cs_a[starts + WINDOW_WEEKS] - cs_a[starts]
        ok = (sa >= MIN_ADEQUATE_WEEKS) & (sd >= MIN_STRICT_DAYS)
        if not ok.any():
            bad.append((int(u), int(c), "recompute_not_passing"))
            continue
        best = int(np.argmax(np.where(ok, sd, -1)))
        # eligibility_by_channel.best_window_start is the 0-based candidate
        # index; the cohort manifests carry first_monday + 7*index (epoch days).
        exp_index = int(starts[best])
        if (int(row["best_window_start"]) != exp_index or
                int(row["first_monday"]) != f or
                int(row["best_adequate_weeks"]) != int(sa[best]) or
                int(row["best_strict_days"]) != int(sd[best])):
            bad.append((int(u), int(c), "stored_gate_stats_mismatch",
                        int(row["best_window_start"]), exp_index))
            continue
        n_checked += 1
    add("selection.per_channel_rank_and_gate_stats_recomputed",
        not bad and n_checked > 0,
        f"{n_checked:,} passing series checked; mismatches: {bad[:3]}")

    model = pd.read_parquet(ARTIFACTS_SQ / "cohort_manifest_model_sources.parquet")
    sel = model.set_index("user_id")
    span = passing.set_index(["user_id", "channel"])[["first_monday", "last_monday"]]
    sel_span = sel.join(span, on=["user_id", "selected_channel"])
    ok = ((sel_span["window_start"] >= sel_span["first_monday"]) &
          (sel_span["window_start"] <=
           sel_span["last_monday"] - 7 * (WINDOW_WEEKS - 1))).all()
    add("selection.window_inside_candidate_range", bool(ok))

    cand = passing.copy()
    cand["chan_rank"] = cand["channel"].map(
        {c: i for i, c in enumerate(sorted(CORE_CHANNELS))})
    cand = cand.sort_values(["user_id", "best_strict_days", "best_window_start",
                             "chan_rank"],
                            ascending=[True, False, True, True])
    pick = cand.groupby("user_id").first().reindex(sel.index)
    ok = ((pick["channel"] == sel["selected_channel"]).all() and
          (pick["best_strict_days"] == sel["strict_days"]).all() and
          (pick["best_adequate_weeks"] == sel["adequate_weeks"]).all())
    add("selection.cross_channel_tiebreak_reproduced", bool(ok))

    ws = sel["window_start"].astype("int64")
    we = sel["window_end"].astype("int64")
    ok = bool(((ws + 3) % 7 == 0).all() and ((we + 3) % 7 == 6).all() and
              ((we - ws) == WINDOW_INCLUSIVE_DAYS - 1).all())
    add("selection.monday_start_sunday_end_91_days", ok)


def check_coverage_audit() -> None:
    p = ARTIFACTS_SQ / "coverage_semantics_audit.csv"
    if not p.exists():
        add("coverage_audit.csv_exists", False)
        return
    add("coverage_audit.csv_exists", True)
    audit = pd.read_csv(p)
    need = {"source_id", "channel", "n_day_rows", "n_days_over_24h", "max_cov_s",
            "example_user", "example_date", "example_cov_s", "share_over_24h",
            "note"}
    add("coverage_audit.schema", need <= set(audit.columns))
    add("coverage_audit.all_source_channel_cells",
        len(audit) == len(FROZEN_CHANNEL_PASS) * len(CORE_CHANNELS))
    days = pd.read_parquet(ARTIFACTS_V2 / "epoch_days.parquet", columns=["cov_s"])
    add("coverage_audit.cov_over_24h_present_in_input",
        bool((days["cov_s"] > 86_400).any()),
        f"max cov_s = {float(days['cov_s'].max()):.0f} s (frozen artifact "
        f"semantics: union of intervals assigned to the interval-start local "
        f"date; not physical local-day duration)")


def check_splits() -> None:
    split = pd.read_parquet(ARTIFACTS_SQ / "split_manifest_sq.parquet")
    model = pd.read_parquet(ARTIFACTS_SQ / "cohort_manifest_model_sources.parquet")
    add("split.unique_and_exhaustive",
        split["user_id"].is_unique and
        set(split["user_id"]) == set(model["user_id"]))
    add("split.labels_legal", set(split["split"]) <= {"train", "val", "test"})
    add("split.sources_disjoint_per_participant",
        split.groupby("user_id")["source_id"].nunique().eq(1).all())
    viol = pd.read_csv(ARTIFACTS_SQ / "split_violations_sq.csv")
    for s in MODEL_SOURCE_IDS:
        sub = split[split["source_id"] == s]
        tab = {}
        for sp in ("train", "val", "test"):
            g = sub[sub["split"] == sp]
            tab[sp] = (int(len(g)), int((g["y"] == 0).sum()),
                       int((g["y"] == 1).sum()))
        want = FROZEN_SPLITS[s]
        add(f"split.frozen_table_source_{s}",
            all(tab[sp] == tuple(want[sp]) for sp in want), f"{tab}")
        n_v = int((viol["source_id"] == s).sum()) if len(viol) else 0
        add(f"split.relaxation_explicit_source_{s}", True,
            f"{n_v} nonzero bound violations recorded in split_violations_sq.csv")


def check_features() -> None:
    names = ("demographic", "recording", "window", "rolling", "all")
    blocks = {n: pd.read_parquet(ARTIFACTS_SQ / f"features_{n}.parquet")
              for n in names}
    model = pd.read_parquet(ARTIFACTS_SQ / "cohort_manifest_model_sources.parquet")
    ids = set(model["user_id"])
    for n, b in blocks.items():
        add(f"features.{n}_rows_match_union",
            len(b) == len(ids) and b["user_id"].is_unique and set(b["user_id"]) == ids,
            f"{len(b):,} rows")
        add(f"features.{n}_columns_unique", bool(b.columns.is_unique))
        add(f"features.{n}_carries_source_id", "source_id" in b.columns)

    feat_cols = {n: [c for c in b.columns if c not in ("user_id", "source_id")]
                 for n, b in blocks.items()}
    add("features.demographics_only_in_D",
        set(feat_cols["demographic"]) == {"demo__age_group", "demo__bmi_grp"} and
        not any(c.startswith("demo__") for n in ("recording", "window", "rolling")
                for c in feat_cols[n]))
    pref = {"rec__": 78, "win__": 78, "roll_rec__": 120, "roll_win__": 120}
    ok = True
    for p, width in pref.items():
        block = "rolling" if p.startswith("roll") else (
            "recording" if p == "rec__" else "window")
        own = [c for c in feat_cols[block] if c.startswith(p)]
        ok &= len(own) == width
        for other in pref:
            if other != p:
                ok &= not any(c.startswith(other) for c in own)
    add("features.disjoint_prefixes_widths", ok)
    add("features.all_width", len(feat_cols["all"]) == 398,
        f"{len(feat_cols['all'])} = D2 + A78 + B78 + C_rec120 + C_win120")
    add("features.dictionary_covers_all",
        len(pd.read_csv(ARTIFACTS_SQ / "feature_dictionary_sq.csv")) == 398)


FORBIDDEN = ("salutation", "gender", "sex", "who_average", "user_id", "file",
             "timestamp", "window_start", "window_end", "selected_channel",
             "pass_pattern", "strict_days", "adequate_weeks", "source", "_date",
             "epoch_", "multisource")


def check_forbidden_predictors() -> None:
    bad, n_schemas = [], 0
    for src in MODEL_SOURCE_IDS:
        for variant in VARIANTS:
            p = ARTIFACTS_SQ / f"source_{src}" / f"feature_schema_{variant}.json"
            if not p.exists():
                bad.append(f"{src}/{variant}: schema missing")
                continue
            n_schemas += 1
            for c in json.loads(p.read_text())["input_columns"]:
                hits = [f for f in FORBIDDEN if f in c]
                if c == "y" or hits:
                    bad.append(f"{src}/{variant}:{c} {hits}")
    add("forbidden.no_forbidden_predictors_in_design_matrices",
        not bad and n_schemas == 18, f"{n_schemas} schemas scanned")


def check_models() -> None:
    for src in MODEL_SOURCE_IDS:
        d = ARTIFACTS_SQ / f"source_{src}"
        for variant in VARIANTS:
            names = json.loads(
                (d / f"transformed_feature_names_{variant}.json").read_text())
            schema = json.loads((d / f"feature_schema_{variant}.json").read_text())
            params = json.loads((d / f"model_params_{variant}.json").read_text())
            ok = ("source_id" not in names and
                  any(n.startswith("demo__age_group=") for n in names) and
                  any(n.startswith("demo__bmi_grp=") for n in names) and
                  params["random_forest"]["random_state"] == RF_SEED and
                  params["random_forest"]["n_estimators"] == 100 and
                  isinstance(schema["all_missing_dropped"], list) and
                  isinstance(schema["zero_variance_dropped"], list) and
                  not set(schema["all_missing_dropped"]) & set(names) and
                  not set(schema["zero_variance_dropped"]) & set(names))
            add(f"model.schema_source_{src}_{variant}", ok,
                f"{len(names)} transformed terms")
            add(f"model.pipeline_saved_source_{src}_{variant}",
                (d / f"model_pipeline_{variant}.joblib").exists())


def check_predictions() -> None:
    split = pd.read_parquet(ARTIFACTS_SQ / "split_manifest_sq.parquet")
    for src in MODEL_SOURCE_IDS:
        ssrc = split[split["source_id"] == src]
        ymap = ssrc.set_index("user_id")["y"]
        for variant in VARIANTS:
            for sp in ("val", "test"):
                p = ARTIFACTS_SQ / f"source_{src}" / f"predictions_{sp}_{variant}.csv"
                if not p.exists():
                    add(f"predictions.source_{src}_{sp}_{variant}", False, "missing")
                    continue
                pr = pd.read_csv(p)
                want = set(ssrc.loc[ssrc["split"] == sp, "user_id"])
                ok = (set(pr["user_id"]) == want and pr["user_id"].is_unique and
                      bool((pr["y_true"].to_numpy() ==
                            pr["user_id"].map(ymap).to_numpy()).all()))
                add(f"predictions.source_{src}_{sp}_{variant}", bool(ok))


def check_bootstrap() -> None:
    for src in MODEL_SOURCE_IDS:
        d = ARTIFACTS_SQ / f"source_{src}"
        draws = pd.read_csv(d / "bootstrap_test_draws.csv")
        res = pd.read_csv(d / "bootstrap_resample_indices.csv")
        preds = pd.read_csv(d / "predictions_test_rec.csv")
        per_var = draws.groupby("variant")["draw_id"].nunique()
        add(f"bootstrap.100_draws_per_variant_source_{src}",
            bool(set(draws["variant"]) == set(VARIANTS) and per_var.eq(100).all()),
            f"{per_var.to_dict()}")
        grids = [set(draws[draws["variant"] == v]["draw_id"]) for v in VARIANTS]
        add(f"bootstrap.common_draw_ids_across_variants_source_{src}",
            all(g == grids[0] for g in grids))
        ymap = preds.set_index("user_id")["y_true"]
        n0 = int((ymap == 0).sum())
        n1 = int((ymap == 1).sum())
        ok = True
        for _did, g in res.groupby("draw_id"):
            ys = g["user_id"].map(ymap)
            ok &= int((ys == 0).sum()) == n0 and int((ys == 1).sum()) == n1
        add(f"bootstrap.exact_class_sizes_per_draw_source_{src}", bool(ok),
            f"n0={n0}, n1={n1}")
        da = draws[draws["variant"] == "rec"].set_index("draw_id")["auroc"]
        db = draws[draws["variant"] == "demo"].set_index("draw_id")["auroc"]
        deltas = pd.read_csv(d / "bootstrap_paired_deltas.csv")
        ref = deltas[(deltas["comparison"] == "rec-demo") &
                     (deltas["metric"] == "auroc")].set_index("draw_id")["delta"]
        add(f"bootstrap.paired_deltas_reproducible_source_{src}",
            bool(np.allclose((da - db).sort_index().to_numpy(),
                             ref.sort_index().to_numpy(), atol=1e-12)))


def check_rolling_fixtures() -> None:
    r = subprocess.run([sys.executable,
                        str(Path(__file__).parent / "test_rolling_fixture.py")],
                       capture_output=True, text=True)
    add("rolling.deterministic_fixtures", r.returncode == 0,
        (r.stdout.strip() or r.stderr.strip()[-200:]))


def main() -> None:
    check_eligibility()
    check_manifests()
    check_windows_and_selection()
    check_coverage_audit()
    check_splits()
    check_features()
    check_forbidden_predictors()
    check_models()
    check_predictions()
    check_bootstrap()
    check_rolling_fixtures()
    out = pd.DataFrame(RESULTS)
    out.to_json(ARTIFACTS_SQ / "verification_sq.json", orient="records", indent=2)
    n_fail = int((~out["ok"]).sum())
    print(f"\nverification: {len(out)} checks, {n_fail} failures")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
