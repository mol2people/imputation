"""Side-quest section 5: 18 RF fits, saved predictions, fixed-100 bootstrap.

Per model-eligible source (3, 6, 7) and each of the six prespecified variants
(demo, rec, win, roll_rec, roll_win, all):

  * a NEW SQPreprocessor + RandomForestClassifier(random_state=RF_SEED), every
    other hyperparameter at the installed scikit-learn default; no fitted state
    crosses the source or variant boundary (plan section 7.3);
  * preprocessing fitted on the source's training participants only;
  * validation/test metrics and saved prediction CSVs first;
  * then exactly 100 test bootstrap resamples per source (with replacement
    separately within salutation class; SeedSequence([SQ_BOOTSTRAP_SEED,
    source_id]); identical index sets reused for every variant), recomputed
    from the saved predictions without refitting, plus aligned paired deltas.

Run:  python src/sidequest/model.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import SQPreprocessor  # noqa: E402
from sq_config import (  # noqa: E402
    ARTIFACTS_SQ, MODEL_SOURCE_IDS, RF_SEED, SQ_BOOTSTRAP_N, SQ_BOOTSTRAP_SEED,
    VARIANTS,
)

SPLITS = ["train", "val", "test"]
PAIRED = [("rec", "demo"), ("win", "rec"), ("roll_win", "roll_rec"), ("all", "rec")]


def load_blocks() -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(ARTIFACTS_SQ / f"features_{name}.parquet")
            for name in ("demographic", "recording", "window", "rolling", "all")}


def variant_matrix(blocks: dict[str, pd.DataFrame], variant: str) -> pd.DataFrame:
    demo, roll = blocks["demographic"], blocks["rolling"]
    if variant == "demo":
        return demo
    if variant == "rec":
        return demo.merge(blocks["recording"], on=["user_id", "source_id"])
    if variant == "win":
        return demo.merge(blocks["window"], on=["user_id", "source_id"])
    if variant == "roll_rec":
        cols = ["user_id", "source_id"] + \
            [c for c in roll.columns if c.startswith("roll_rec__")]
        return demo.merge(roll[cols], on=["user_id", "source_id"])
    if variant == "roll_win":
        cols = ["user_id", "source_id"] + \
            [c for c in roll.columns if c.startswith("roll_win__")]
        return demo.merge(roll[cols], on=["user_id", "source_id"])
    if variant == "all":
        return blocks["all"]
    raise ValueError(variant)


def metric_dict(y_true: np.ndarray, p1: np.ndarray, pred: np.ndarray) -> dict:
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1], zero_division=0)
    return {"n": int(len(y_true)),
            "auroc": float(roc_auc_score(y_true, p1)),
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "precision": [float(v) for v in prec],
            "recall": [float(v) for v in rec],
            "f1": [float(v) for v in f1],
            "support": [int(v) for v in sup],
            "confusion_matrix": [[int(v) for v in row] for row in
                                 confusion_matrix(y_true, pred, labels=[0, 1])]}


def quick_metrics(y_true: np.ndarray, p1: np.ndarray) -> dict:
    pred = (p1 >= 0.5).astype(int)
    return {"auroc": float(roc_auc_score(y_true, p1)),
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred))}


def bootstrap_source(src: int, test_preds: dict[str, pd.DataFrame]):
    """100 aligned draws; returns (resample-index table, per-variant metrics)."""
    y = test_preds["rec"]["y_true"].to_numpy(dtype=int)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    rng = np.random.default_rng(np.random.SeedSequence([SQ_BOOTSTRAP_SEED, src]))
    rows_idx, rows_met = [], []
    for d in range(SQ_BOOTSTRAP_N):
        take = np.concatenate([rng.choice(idx0, size=n0, replace=True),
                               rng.choice(idx1, size=n1, replace=True)])
        for pos, i in enumerate(take):
            rows_idx.append({"draw_id": d, "position": pos,
                             "user_id": int(test_preds["rec"]["user_id"].iloc[i])})
        for variant, pred in test_preds.items():
            rows_met.append({"variant": variant, "draw_id": d,
                             **quick_metrics(y[take],
                                             pred["p_class1"].to_numpy()[take])})
    return pd.DataFrame(rows_idx), pd.DataFrame(rows_met)


def paired_deltas(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in PAIRED:
        for metric in ("auroc", "balanced_accuracy", "accuracy"):
            da = draws[draws["variant"] == a].set_index("draw_id")[metric]
            db = draws[draws["variant"] == b].set_index("draw_id")[metric]
            delta = (da - db).dropna()
            for did, dv in delta.items():
                rows.append({"comparison": f"{a}-{b}", "metric": metric,
                             "draw_id": int(did), "delta": float(dv)})
    return pd.DataFrame(rows)


def main() -> None:
    blocks = load_blocks()
    split = pd.read_parquet(ARTIFACTS_SQ / "split_manifest_sq.parquet")
    assert split["user_id"].is_unique, "duplicate participant in split manifest"
    assert set(split["user_id"]) == set(blocks["all"]["user_id"]), \
        "split manifest and feature blocks disagree on participants"
    assert split.groupby("user_id")["split"].nunique().eq(1).all()

    versions = {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                "pandas": pd.__version__, "numpy": np.__version__,
                "joblib": joblib.__version__}
    with open(ARTIFACTS_SQ / "package_versions_sq.json", "w") as fh:
        json.dump(versions, fh, indent=2)
    print(f"feature blocks loaded; sklearn {versions['sklearn']}", flush=True)

    for src in MODEL_SOURCE_IDS:
        d = ARTIFACTS_SQ / f"source_{src}"
        d.mkdir(parents=True, exist_ok=True)
        ssrc = split[split["source_id"] == src]
        assert ssrc["user_id"].is_unique
        ids = {s: set(ssrc.loc[ssrc["split"] == s, "user_id"]) for s in SPLITS}
        assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"]
                    or ids["val"] & ids["test"]), f"split overlap source {src}"
        assert sum(len(v) for v in ids.values()) == len(ssrc), \
            f"splits not exhaustive source {src}"

        test_preds: dict[str, pd.DataFrame] = {}
        metrics: dict[str, dict] = {}
        y_map = ssrc.set_index("user_id")["y"].astype(int)
        for variant in VARIANTS:
            print(f"source {src} variant {variant}: fitting", flush=True)
            fr = variant_matrix(blocks, variant).copy()
            fr = fr[fr["source_id"] == src].copy()
            fr["y"] = fr["user_id"].map(y_map)
            assert fr["y"].notna().all(), f"variant {variant} missing labels"
            feat_cols = [c for c in fr.columns if c not in ("user_id", "source_id", "y")]
            parts = {}
            for s in SPLITS:
                part = fr[fr["user_id"].isin(ids[s])].sort_values("user_id")
                assert len(part) == len(ids[s]), f"variant {variant} split {s}"
                parts[s] = part
            Xtr = parts["train"][feat_cols]
            ytr = parts["train"]["y"].to_numpy(dtype=int)

            prep = SQPreprocessor().fit(Xtr)
            rf = RandomForestClassifier(random_state=RF_SEED)
            rf.fit(prep.transform(Xtr), ytr)

            ytr_maj = float(max(ytr.mean(), 1.0 - ytr.mean()))
            metrics[variant] = {"train_majority_accuracy": ytr_maj,
                                "schema": {
                                    "input_columns": feat_cols,
                                    "all_missing_dropped": prep.all_missing_,
                                    "missing_indicators": prep.missing_cols_,
                                    "zero_variance_dropped": prep.zero_variance_,
                                    "transformed_features": prep.feature_names_out_}}
            for s in ("val", "test"):
                part = parts[s]
                p1 = rf.predict_proba(prep.transform(part[feat_cols]))[
                    :, list(rf.classes_).index(1)]
                pred = (p1 >= 0.5).astype(int)
                y = part["y"].to_numpy(dtype=int)
                assert set(part["user_id"]) == ids[s]
                pred_df = pd.DataFrame({"user_id": part["user_id"].to_numpy(),
                                        "y_true": y, "p_class1": p1,
                                        "predicted": pred})
                pred_df.to_csv(d / f"predictions_{s}_{variant}.csv", index=False)
                metrics[variant][s] = metric_dict(y, p1, pred)
                if s == "test":
                    test_preds[variant] = pred_df

            joblib.dump({"preprocessor": prep, "model": rf,
                         "variant": variant, "source_id": src},
                        d / f"model_pipeline_{variant}.joblib")
            with open(d / f"model_params_{variant}.json", "w") as fh:
                json.dump({"random_forest": rf.get_params(),
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

        resamples, draws = bootstrap_source(src, test_preds)
        resamples.to_csv(d / "bootstrap_resample_indices.csv", index=False)
        draws.to_csv(d / "bootstrap_test_draws.csv", index=False)
        deltas = paired_deltas(draws)
        deltas.to_csv(d / "bootstrap_paired_deltas.csv", index=False)

        for variant in VARIANTS:
            dv = draws[draws["variant"] == variant]
            boot = {}
            for metric in ("auroc", "accuracy", "balanced_accuracy"):
                v = dv[metric].to_numpy()
                boot[metric] = {
                    "point": metrics[variant]["test"][metric],
                    "bootstrap_mean": float(v.mean()),
                    "bootstrap_sd": float(v.std(ddof=1)),
                    "q025": float(np.quantile(v, 0.025, method="linear")),
                    "q975": float(np.quantile(v, 0.975, method="linear")),
                }
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
                for a, b in PAIRED}

        with open(d / "metrics_sq.json", "w") as fh:
            json.dump({"source_id": src, "seed": int(RF_SEED),
                       "bootstrap_seed": int(SQ_BOOTSTRAP_SEED),
                       "n_draws": int(SQ_BOOTSTRAP_N),
                       "versions": versions, "variants": metrics}, fh, indent=2)

        a = {v: metrics[v]["test"]["auroc"] for v in VARIANTS}
        print(f"source {src}: test AUROC " +
              " ".join(f"{v}={a[v]:.4f}" for v in VARIANTS), flush=True)

    print("18 RF fits complete", flush=True)


if __name__ == "__main__":
    main()
