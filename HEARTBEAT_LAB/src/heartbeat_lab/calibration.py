"""Validation-only sigmoid calibration of a fixed model's customer scores."""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def probability_m(scores, calibration):
    """Map scores to P(M); retain missing predictions (e.g. discarded days)."""
    z = calibration["slope"] * np.asarray(scores, dtype=float) + calibration["intercept"]
    with np.errstate(invalid="ignore"):  # Missing day predictions stay NaN.
        return np.exp(-np.logaddexp(0, -z))


def fit_calibration(predictions):
    """One observation per customer; test labels never enter the sigmoid fit."""
    if predictions.customer_id.duplicated().any():
        raise ValueError("Calibration requires disjoint customers and one score per customer.")
    val = predictions.loc[predictions.split.eq("val")]
    if set(val.y) != {0, 1} or not np.isfinite(val.score).all():
        raise ValueError("Calibration needs finite validation scores and both F/M labels.")
    # Weak fixed regularization keeps fitting finite under complete separation.
    sigmoid = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, tol=1e-9)
    sigmoid.fit(val[["score"]], val.y)
    return dict(method="sigmoid logistic regression, C=1e6", fit_split="val",
                unit="one full/budgeted history per customer", n=len(val),
                n_F=int(val.y.eq(0).sum()), n_M=int(val.y.eq(1).sum()),
                slope=float(sigmoid.coef_[0, 0]), intercept=float(sigmoid.intercept_[0]),
                timeline="transferred calibration; daily and prefix probabilities are unvalidated")


def calibrate_bundle(path):
    """Add a small JSON sidecar; keep original model and raw predictions intact."""
    predictions_path = path / "predictions.csv"
    predictions = pd.read_csv(predictions_path, dtype={"customer_id": str})
    assignments = pd.read_csv(path / "assignments.csv", dtype={"customer_id": str})
    if assignments.customer_id.duplicated().any():
        raise ValueError("Duplicate customers in assignments.")
    joined = predictions.merge(assignments[["customer_id", "split"]], on="customer_id",
                               how="left", suffixes=("", "_assigned"), validate="one_to_one")
    if not joined.split.eq(joined.split_assigned).all() or not joined.split.isin(["val", "test"]).all():
        raise ValueError("Calibration predictions must match held-out assignments.")
    calibration = fit_calibration(predictions)
    calibration["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    test = predictions.loc[predictions.split.eq("test")]
    if len(test):
        p = probability_m(test.score, calibration)
        calibration["test"] = dict(n=len(test), brier=float(brier_score_loss(test.y, p)),
                                   log_loss=float(log_loss(test.y, p, labels=[0, 1])))
    (path / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    return calibration


def probability_columns(table, calibrations):
    """Preserve raw scores and add probability columns for plots and downloads."""
    result = table.copy()
    for name, calibration in calibrations.items():
        rows = result.model.eq(name)
        for period in ("daily", "cumulative"):
            result.loc[rows, f"{period}_p_m"] = probability_m(result.loc[rows, f"{period}_score"], calibration)
    return result
