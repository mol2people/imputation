"""Fit one default random forest on the training split and evaluate.

No hyperparameter tuning, cross-validation, calibration or feature selection.
The validation split is a descriptive evaluation of the same fixed model; the
test split is evaluated once.  Ten within-class bootstrap resamples of test
participants use the already-saved predictions.

Outputs under artifacts/:
  model_pipeline.joblib, model_params.json, feature_names.json
  predictions_val.csv, predictions_test.csv
  metrics.json, bootstrap_test.csv, model_report.md
Run:  python src/model.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ARTIFACTS, RF_SEED, BOOTSTRAP_N, BOOTSTRAP_SEED, EXTRA_CATEGORICAL  # noqa: E402

CAT_COLS = ["epoch_primary_source", "epoch_multisource"] + EXTRA_CATEGORICAL


def metrics(y_true, proba, pred):
    return {
        "auroc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_0": float(precision_score(y_true, pred, pos_label=0, zero_division=0)),
        "precision_1": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "recall_0": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "recall_1": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "f1_0": float(f1_score(y_true, pred, pos_label=0, zero_division=0)),
        "f1_1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "n": int(len(y_true)),
        "confusion": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def bootstrap_metrics(y, proba, pred, n_boot, seed):
    rng = np.random.default_rng(seed)
    idx_by_class = {c: np.where(y == c)[0] for c in np.unique(y)}
    rows = []
    for b in range(n_boot):
        picks = []
        for c, idx in idx_by_class.items():
            picks.append(rng.choice(idx, size=len(idx), replace=True))
        sel = np.concatenate(picks)
        rows.append({
            "resample": b + 1,
            "auroc": float(roc_auc_score(y[sel], proba[sel])),
            "accuracy": float(accuracy_score(y[sel], pred[sel])),
            "balanced_accuracy": float(balanced_accuracy_score(y[sel], pred[sel])),
        })
    return pd.DataFrame(rows)


def main():
    sm = pd.read_csv(ARTIFACTS / "split_manifest.csv")
    feats = pd.read_parquet(ARTIFACTS / "epoch_features.parquet")
    man = pd.read_parquet(ARTIFACTS / "cohort_manifest.parquet")[["user_id", "y", "salutation"]]
    df = (sm.merge(feats, on="user_id", how="left")
            .merge(man, on="user_id", how="left"))
    assert df["split"].isin(["train", "val", "test"]).all()
    assert df["y"].notna().all()
    print(f"rows={len(df)}  split={df.split.value_counts().to_dict()}")

    feature_cols = [c for c in feats.columns if c != "user_id"]
    X = df[feature_cols].copy()
    y = df["y"].astype(int).to_numpy()
    split = df["split"].to_numpy()

    tr, va, te = split == "train", split == "val", split == "test"
    # training-derived all-missing column mask
    keep = X[tr].notna().any(axis=0)
    dropped = [c for c in feature_cols if not keep[c]]
    X = X[keep[keep].index]
    for c in CAT_COLS:
        X[c] = X[c].astype("object")
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    print(f"kept {X.shape[1]} features (dropped all-missing-in-train: {dropped})")

    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]

    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, CAT_COLS)])
    clf = RandomForestClassifier(random_state=RF_SEED)          # all other params default
    pipe = Pipeline([("pre", pre), ("rf", clf)])

    print("RF params:", clf.get_params())
    pipe.fit(Xtr, ytr)

    # verify numerical features reached the estimator
    fn = pipe.named_steps["pre"].get_feature_names_out()
    print(f"transformed feature count: {len(fn)}")
    assert any(n.startswith("num__") for n in fn), "no numerical features reached estimator"

    pred_va = pipe.predict(Xva); proba_va = pipe.predict_proba(Xva)[:, list(pipe.classes_).index(1)]
    pred_te = pipe.predict(Xte); proba_te = pipe.predict_proba(Xte)[:, list(pipe.classes_).index(1)]

    maj = int(round(ytr.mean()))  # training majority class
    maj_acc = float(max(ytr.mean(), 1 - ytr.mean()))

    mv = metrics(yva, proba_va, pred_va)
    mt = metrics(yte, proba_te, pred_te)
    boot = bootstrap_metrics(yte, proba_te, pred_te, BOOTSTRAP_N, BOOTSTRAP_SEED)

    pd.DataFrame({"user_id": df.loc[va, "user_id"], "y": yva, "proba_1": proba_va,
                  "pred": pred_va}).to_csv(ARTIFACTS / "predictions_val.csv", index=False)
    pd.DataFrame({"user_id": df.loc[te, "user_id"], "y": yte, "proba_1": proba_te,
                  "pred": pred_te}).to_csv(ARTIFACTS / "predictions_test.csv", index=False)
    boot.to_csv(ARTIFACTS / "bootstrap_test.csv", index=False)

    joblib.dump(pipe, ARTIFACTS / "model_pipeline.joblib")
    with open(ARTIFACTS / "model_params.json", "w") as fh:
        json.dump({"sklearn_version": sklearn.__version__,
                   "rf_params": clf.get_params(),
                   "n_features_in": int(X.shape[1]),
                   "n_transformed_features": int(len(fn)),
                   "dropped_all_missing_in_train": dropped,
                   "train_majority_class": maj,
                   "minority_class_metric_warning": "accuracy inflated by 66.9% class-1 prevalence",
                   "as_of_recorded_sex": "retrospective classification of recorded salutation"}, fh, indent=2)
    with open(ARTIFACTS / "feature_names.json", "w") as fh:
        json.dump(list(fn), fh, indent=2)

    summary = {
        "majority_class_reference_accuracy": maj_acc,
        "validation": mv,
        "test": mt,
        "bootstrap_test": {
            "n": int(len(boot)),
            "auroc_mean": float(boot.auroc.mean()), "auroc_sd": float(boot.auroc.std(ddof=1)),
            "auroc_min": float(boot.auroc.min()), "auroc_max": float(boot.auroc.max()),
            "accuracy_mean": float(boot.accuracy.mean()), "accuracy_sd": float(boot.accuracy.std(ddof=1)),
            "balanced_accuracy_mean": float(boot.balanced_accuracy.mean()),
            "balanced_accuracy_sd": float(boot.balanced_accuracy.std(ddof=1)),
        },
    }
    with open(ARTIFACTS / "metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    lines = ["# Model report: one default random forest", "",
             f"scikit-learn {sklearn.__version__}; seed RF={RF_SEED}; "
             f"participants train/val/test = {int(tr.sum())}/{int(va.sum())}/{int(te.sum())}.", "",
             f"Majority-class reference accuracy (training majority = class {maj}): {maj_acc:.4f}. "
             "Accuracy is inflated by the 66.9% class-1 prevalence; AUROC and balanced accuracy are primary.", "",
             "## Validation (descriptive)", "",
             f"- AUROC {mv['auroc']:.4f}; accuracy {mv['accuracy']:.4f}; balanced accuracy {mv['balanced_accuracy']:.4f}",
             f"- class 0 P/R/F1 {mv['precision_0']:.3f}/{mv['recall_0']:.3f}/{mv['f1_0']:.3f}",
             f"- class 1 P/R/F1 {mv['precision_1']:.3f}/{mv['recall_1']:.3f}/{mv['f1_1']:.3f}",
             f"- confusion [[TN,FP],[FN,TP]] = {mv['confusion']}", "",
             "## Test (evaluated once)", "",
             f"- AUROC {mt['auroc']:.4f}; accuracy {mt['accuracy']:.4f}; balanced accuracy {mt['balanced_accuracy']:.4f}",
             f"- class 0 P/R/F1 {mt['precision_0']:.3f}/{mt['recall_0']:.3f}/{mt['f1_0']:.3f}",
             f"- class 1 P/R/F1 {mt['precision_1']:.3f}/{mt['recall_1']:.3f}/{mt['f1_1']:.3f}",
             f"- confusion [[TN,FP],[FN,TP]] = {mt['confusion']}", "",
             f"## Test bootstrap ({BOOTSTRAP_N} within-class resamples, seed {BOOTSTRAP_SEED})", "",
             f"- AUROC mean {summary['bootstrap_test']['auroc_mean']:.4f} "
             f"(sample SD {summary['bootstrap_test']['auroc_sd']:.4f}, "
             f"min/max {summary['bootstrap_test']['auroc_min']:.4f}/"
             f"{summary['bootstrap_test']['auroc_max']:.4f})",
             f"- accuracy mean {summary['bootstrap_test']['accuracy_mean']:.4f} "
             f"(SD {summary['bootstrap_test']['accuracy_sd']:.4f})",
             f"- balanced accuracy mean {summary['bootstrap_test']['balanced_accuracy_mean']:.4f} "
             f"(SD {summary['bootstrap_test']['balanced_accuracy_sd']:.4f})",
             "", "Bootstrap summaries condition on the fitted model, fixed split and observed class "
             "counts; they are a rough variability check, not a reliable 95% CI."]
    with open(ARTIFACTS / "model_report.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
