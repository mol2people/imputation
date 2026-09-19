"""Side-quest model extension: 500-tree RF with wide-grid tuning per (source, variant).

Design (user decision 2026-09-19):
  * tuning pool = frozen train + val participants; the frozen test split stays
    a single-look holdout;
  * 5-fold stratified CV (shuffle, seed SEED) over the pool; SQPreprocessor
    refit on each fold's training portion (no leakage); selection by pooled
    out-of-fold AUROC, ties broken by fixed grid order;
  * grid: max_features {sqrt, 0.2, 0.4} x min_samples_leaf {1, 2, 5, 10} x
    max_depth {None, 12} x class_weight {None, "balanced_subsample"} = 48
    configs (min_samples_split omitted: dominated by min_samples_leaf);
  * CV fits use 500 trees as well (user request) with n_jobs=-1, which does
    not change RF results given random_state;
  * final fit: best config, n_estimators=500, on the full pool; one test
    prediction; then the identical 100-draw within-class bootstrap
    (SeedSequence([SQ_BOOTSTRAP_SEED, source_id])) and paired deltas as the
    default run, so draw-level comparisons across the two runs are valid.

Outputs per source under artifacts_sq/source_<s>/tuned_500/:
  tuning_results.csv, best_params_<variant>.json, model_pipeline_<variant>.joblib,
  model_params_<variant>.json, transformed_feature_names_<variant>.json,
  feature_schema_<variant>.json, predictions_{val,test}_<variant>.csv,
  bootstrap_resample_indices.csv, bootstrap_test_draws.csv,
  bootstrap_paired_deltas.csv, metrics_sq.json

Run:  python src/sidequest/model_tune.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (  # noqa: E402
    SPLITS, bootstrap_source, load_blocks, metric_dict, paired_deltas,
    variant_matrix,
)
from preprocess import SQPreprocessor  # noqa: E402
from sq_config import (  # noqa: E402
    ARTIFACTS_SQ, MODEL_SOURCE_IDS, RF_SEED, SEED, SQ_BOOTSTRAP_N,
    SQ_BOOTSTRAP_SEED, VARIANTS,
)

GRID = [dict(max_features=mf, min_samples_leaf=msl, max_depth=md,
             class_weight=cw)
        for mf, msl, md, cw in itertools.product(
            ("sqrt", 0.2, 0.4), (1, 2, 5, 10), (None, 12),
            (None, "balanced_subsample"))]
N_FOLDS = 5
TUNE_TREES = 500
FINAL_TREES = 500


def oof_cv_auroc(X: pd.DataFrame, y: np.ndarray, params: dict) -> tuple[float, float]:
    """Pooled out-of-fold AUROC / balanced accuracy for one config."""
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, va in skf.split(X, y):
        prep = SQPreprocessor().fit(X.iloc[tr])
        rf = RandomForestClassifier(n_estimators=TUNE_TREES, random_state=RF_SEED,
                                    n_jobs=-1, **params)
        rf.fit(prep.transform(X.iloc[tr]), y[tr])
        p1 = rf.predict_proba(prep.transform(X.iloc[va]))[:, list(rf.classes_).index(1)]
        oof[va] = p1
    assert not np.isnan(oof).any()
    pred = (oof >= 0.5).astype(int)
    tp = float(((y == 1) & (pred == 1)).sum())
    tn = float(((y == 0) & (pred == 0)).sum())
    n1, n0 = float((y == 1).sum()), float((y == 0).sum())
    return float(roc_auc_score(y, oof)), 0.5 * (tp / n1 + tn / n0)


def main() -> None:
    blocks = load_blocks()
    split = pd.read_parquet(ARTIFACTS_SQ / "split_manifest_sq.parquet")
    model = pd.read_parquet(ARTIFACTS_SQ / "cohort_manifest_model_sources.parquet")
    print(f"blocks loaded; grid={len(GRID)} configs x {N_FOLDS} folds x "
          f"{TUNE_TREES} trees; sklearn {sklearn.__version__}", flush=True)

    for src in MODEL_SOURCE_IDS:
        d = ARTIFACTS_SQ / f"source_{src}" / "tuned_500"
        d.mkdir(parents=True, exist_ok=True)
        ssrc = split[split["source_id"] == src]
        y_all = ssrc.set_index("user_id")["y"].astype(int)
        ids = {s: set(ssrc.loc[ssrc["split"] == s, "user_id"]) for s in SPLITS}

        test_preds: dict[str, pd.DataFrame] = {}
        metrics: dict[str, dict] = {}
        for variant in VARIANTS:
            pfile = d / f"predictions_test_{variant}.csv"
            bpfile = d / f"best_params_{variant}.json"
            if pfile.exists() and bpfile.exists() and \
                    (d / f"model_pipeline_{variant}.joblib").exists():
                bp = json.loads(bpfile.read_text())
                pr = pd.read_csv(pfile)
                assert set(pr["user_id"]) == ids["test"], (src, variant)
                test_preds[variant] = pr
                metrics[variant] = {
                    "tuning": {"grid_size": len(GRID), "n_folds": N_FOLDS,
                               "n_estimators_tuning": TUNE_TREES,
                               "tuning_pool": "train+val", "best": bp["params"],
                               "cv_auroc": bp["cv_auroc"],
                               "cv_balanced_accuracy": bp["cv_balanced_accuracy"]},
                    "test": metric_dict(pr["y_true"].to_numpy(dtype=int),
                                        pr["p_class1"].to_numpy(),
                                        pr["predicted"].to_numpy(dtype=int)),
                    "schema": json.loads((d / f"feature_schema_{variant}.json")
                                         .read_text())}
                print(f"source {src} variant {variant}: cached; skipping",
                      flush=True)
                continue
            print(f"source {src} variant {variant}: CV tuning "
                  f"({len(GRID)}x{N_FOLDS} fits)", flush=True)
            fr = variant_matrix(blocks, variant).copy()
            fr = fr[fr["source_id"] == src].copy()
            fr["y"] = fr["user_id"].map(y_all)
            feat_cols = [c for c in fr.columns if c not in ("user_id", "source_id", "y")]
            pool = fr[fr["user_id"].isin(ids["train"] | ids["val"])].sort_values("user_id")
            test = fr[fr["user_id"].isin(ids["test"])].sort_values("user_id")
            assert len(pool) == len(ids["train"]) + len(ids["val"])
            assert len(test) == len(ids["test"]) and set(test["user_id"]) == ids["test"]
            Xp, yp = pool[feat_cols], pool["y"].to_numpy(dtype=int)

            rows = []
            for gi, params in enumerate(GRID):
                auc, bacc = oof_cv_auroc(Xp, yp, params)
                rows.append({"variant": variant, "config_id": gi, **params,
                             "cv_auroc": auc, "cv_balanced_accuracy": bacc})
            tune = pd.DataFrame(rows)
            best = tune.sort_values(["cv_auroc", "config_id"],
                                    ascending=[False, True]).iloc[0]
            tune.to_csv(d / f"tuning_results_{variant}.csv", index=False)
            best_params = dict(GRID[int(best["config_id"])])
            print(f"  best config {int(best['config_id'])}: {best_params} "
                  f"cv_auroc={best['cv_auroc']:.4f}", flush=True)
            with open(d / f"best_params_{variant}.json", "w") as fh:
                json.dump({"config_id": int(best["config_id"]),
                           "params": best_params,
                           "cv_auroc": float(best["cv_auroc"]),
                           "cv_balanced_accuracy": float(best["cv_balanced_accuracy"]),
                           "n_folds": N_FOLDS, "n_estimators_tuning": TUNE_TREES},
                          fh, indent=2)

            prep = SQPreprocessor().fit(Xp)
            rf = RandomForestClassifier(n_estimators=FINAL_TREES,
                                        random_state=RF_SEED, n_jobs=-1,
                                        **best_params)
            rf.fit(prep.transform(Xp), yp)

            Xt = prep.transform(test[feat_cols])
            p1 = rf.predict_proba(Xt)[:, list(rf.classes_).index(1)]
            pred = (p1 >= 0.5).astype(int)
            yte = test["y"].to_numpy(dtype=int)
            pred_df = pd.DataFrame({"user_id": test["user_id"].to_numpy(),
                                    "y_true": yte, "p_class1": p1,
                                    "predicted": pred})
            pred_df.to_csv(d / f"predictions_test_{variant}.csv", index=False)
            test_preds[variant] = pred_df
            # val predictions from the pooled fit are in-sample; saved only
            # for completeness, not comparable to the default run's val.
            valp = fr[fr["user_id"].isin(ids["val"])].sort_values("user_id")
            pv1 = rf.predict_proba(prep.transform(valp[feat_cols]))[
                :, list(rf.classes_).index(1)]
            pd.DataFrame({"user_id": valp["user_id"].to_numpy(),
                          "y_true": valp["y"].to_numpy(dtype=int),
                          "p_class1": pv1,
                          "predicted": (pv1 >= 0.5).astype(int)}).to_csv(
                d / f"predictions_val_{variant}.csv", index=False)

            metrics[variant] = {
                "tuning": {"grid_size": len(GRID), "n_folds": N_FOLDS,
                           "n_estimators_tuning": TUNE_TREES,
                           "tuning_pool": "train+val",
                           "best": best_params,
                           "cv_auroc": float(best["cv_auroc"]),
                           "cv_balanced_accuracy": float(best["cv_balanced_accuracy"])},
                "test": metric_dict(yte, p1, pred),
                "schema": {"input_columns": feat_cols,
                           "all_missing_dropped": prep.all_missing_,
                           "missing_indicators": prep.missing_cols_,
                           "zero_variance_dropped": prep.zero_variance_,
                           "transformed_features": prep.feature_names_out_}}

            joblib.dump({"preprocessor": prep, "model": rf,
                         "variant": variant, "source_id": src,
                         "n_estimators": FINAL_TREES, "tuned": True},
                        d / f"model_pipeline_{variant}.joblib")
            with open(d / f"model_params_{variant}.json", "w") as fh:
                json.dump({"random_forest": rf.get_params(),
                           "n_estimators_tuning": TUNE_TREES,
                           "selection": "pooled 5-fold OOF AUROC on train+val",
                           "preprocessing": {
                               "all_missing_dropped": prep.all_missing_,
                               "missing_indicators": prep.missing_cols_,
                               "zero_variance_dropped": prep.zero_variance_,
                               "medians": prep.medians_,
                               "categorical_modes": prep.modes_,
                               "onehot_levels": prep.levels_}}, fh, indent=2)
            with open(d / f"transformed_feature_names_{variant}.json", "w") as fh:
                json.dump(prep.get_feature_names_out(), fh, indent=2)
            with open(d / f"feature_schema_{variant}.json", "w") as fh:
                json.dump(metrics[variant]["schema"], fh, indent=2)
            # per-variant checkpoint (resume skips cells with this file present)
            with open(d / f"metrics_{variant}.json", "w") as fh:
                json.dump(metrics[variant], fh, indent=2)

        resamples, draws = bootstrap_source(src, test_preds)
        resamples.to_csv(d / "bootstrap_resample_indices.csv", index=False)
        draws.to_csv(d / "bootstrap_test_draws.csv", index=False)
        deltas = paired_deltas(draws)
        deltas.to_csv(d / "bootstrap_paired_deltas.csv", index=False)

        for variant in VARIANTS:
            dv = draws[draws["variant"] == variant]
            boot = {}
            for m in ("auroc", "accuracy", "balanced_accuracy"):
                v = dv[m].to_numpy()
                boot[m] = {"point": metrics[variant]["test"][m],
                           "bootstrap_mean": float(v.mean()),
                           "bootstrap_sd": float(v.std(ddof=1)),
                           "q025": float(np.quantile(v, 0.025, method="linear")),
                           "q975": float(np.quantile(v, 0.975, method="linear"))}
            metrics[variant]["bootstrap_test"] = {"n_draws": int(SQ_BOOTSTRAP_N),
                                                  **boot}
            metrics[variant]["paired_deltas_test"] = {
                f"{a}-{b}": {m: {
                    "point": metrics[a]["test"][m] - metrics[b]["test"][m],
                    "draw_mean": float((draws[draws["variant"] == a][m].to_numpy()
                                        - draws[draws["variant"] == b][m].to_numpy()).mean()),
                    "draw_sd": float((draws[draws["variant"] == a][m].to_numpy()
                                      - draws[draws["variant"] == b][m].to_numpy()).std(ddof=1)),
                } for m in ("auroc", "balanced_accuracy", "accuracy")}
                for a, b in (("rec", "demo"), ("win", "rec"),
                             ("roll_win", "roll_rec"), ("all", "rec"))}

        with open(d / "metrics_sq.json", "w") as fh:
            json.dump({"source_id": src, "n_estimators": FINAL_TREES,
                       "seed": int(RF_SEED), "bootstrap_seed": int(SQ_BOOTSTRAP_SEED),
                       "n_draws": int(SQ_BOOTSTRAP_N), "variants": metrics}, fh, indent=2)
        a = {v: metrics[v]["test"]["auroc"] for v in VARIANTS}
        print(f"source {src}: tuned-500 test AUROC " +
              " ".join(f"{v}={a[v]:.4f}" for v in VARIANTS), flush=True)

    print("18 tuned RF fits complete", flush=True)


if __name__ == "__main__":
    main()
