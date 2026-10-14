"""Read-only cohort diagnostics; save aggregate evidence, not customer records."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def main():
    out = ROOT / "artifacts/audit"
    out.mkdir(parents=True, exist_ok=True)
    folds = pd.read_parquet(REPO / "experiments/r1b_dateaware_garmin_2026-09-21/cache/folds.parquet")
    tr0 = set(folds.loc[folds.alloc.eq(0) & folds.fold.eq("train"), "user_id"])
    va0 = set(folds.loc[folds.alloc.eq(0) & folds.fold.eq("val"), "user_id"])
    overlap = []
    for alloc, group in folds.groupby("alloc"):
        if group.user_id.duplicated().any():
            raise ValueError("Duplicate customer assignment within original allocation.")
        for split in ("val", "test"):
            ids = set(group.loc[group.fold.eq(split), "user_id"])
            overlap.append(dict(allocation=int(alloc), split=split, customers=len(ids),
                                in_alloc0_transform_training=len(ids & tr0),
                                in_alloc0_alpha_validation=len(ids & va0)))
    pd.DataFrame(overlap).to_csv(out / "historical_split_exposure.csv", index=False)
    days = pd.read_parquet(REPO / "experiments/r1b_dateaware_garmin_2026-09-21/cache/dayall_table.parquet",
                           columns=["user", "date", "d_ch3000_n", "d_ch3000_cov_h", "d_ch3000_hours"])
    features = days.groupby("user").agg(day_count=("date", "size"), first_day=("date", "min"),
                                       last_day=("date", "max"), events_per_day=("d_ch3000_n", "mean"),
                                       coverage_hours=("d_ch3000_cov_h", "mean"),
                                       observed_hours=("d_ch3000_hours", "mean"))
    features["calendar_span"] = features.last_day - features.first_day + 1
    features["adequate_day_fraction"] = features.day_count / features.calendar_span
    splits = pd.read_csv(ROOT / "artifacts/splits.csv", dtype={"customer_id": "int64"}).set_index("customer_id")
    table = splits.join(features, validate="one_to_one")
    models = {"calendar_only": ["first_day", "last_day"],
              "recording_only": ["day_count", "calendar_span", "events_per_day", "coverage_hours", "observed_hours", "adequate_day_fraction"],
              "calendar_and_recording": list(features.columns)}
    metrics = []
    train = table.split.eq("train")
    for name, cols in models.items():
        fit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1000))
        fit.fit(table.loc[train, cols], table.loc[train, "y"])
        row = dict(model=name, n_features=len(cols))
        for split in ("val", "test"):
            mask = table.split.eq(split)
            row[split + "_auroc"] = float(roc_auc_score(table.loc[mask, "y"], fit.predict(table.loc[mask, cols])))
        metrics.append(row)
    # Fixed-pipeline negative control: permute TRAINING customer labels, never days.
    # Test labels remain fixed. This is a wiring check for this nuisance model,
    # not a full permutation test of the historical model-selection procedure.
    cols = models["calendar_and_recording"]
    rng = np.random.default_rng(20260930)
    null = []
    for _ in range(100):
        fit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1000))
        fit.fit(table.loc[train, cols], rng.permutation(table.loc[train, "y"]))
        mask = table.split.eq("test")
        null.append(float(roc_auc_score(table.loc[mask, "y"], fit.predict(table.loc[mask, cols]))))
    result = dict(n_customers=len(table), n_days=len(days), nuisance_models=metrics,
                  nuisance_train_label_permutation=dict(repeats=100, mean=float(np.mean(null)),
                      q025=float(np.quantile(null, .025)), q975=float(np.quantile(null, .975))),
                  day_count_by_label=table.groupby("y").day_count.agg(["count", "min", "median", "max"]).to_dict("index"))
    (out / "audit_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
