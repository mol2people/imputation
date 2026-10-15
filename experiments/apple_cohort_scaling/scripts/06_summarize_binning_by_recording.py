"""Compare saved 5m/15m test scores by recording pattern; no model fitting.

Reads existing adequate-day caches, allocations, and test predictions. Subgroup
terciles use each allocation's training participants. Descriptors always use the
full cached history, so memberships are identical across grids and budgets.
Writes aggregate CSVs only; original benchmark outputs are not modified.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
OUTPUT = ARTIFACTS / "binning_by_recording"
DESCRIPTORS = {
    "history": "n_days",
    "occupancy": "occupancy_rate_5m",
    "continuity": "adequate_day_density",
    "occupancy_variability": "daily_occupancy_sd",
}
LEVELS = ["low", "middle", "high"]


def load_descriptors():
    customers = pd.read_csv(
        ARTIFACTS / "cache_5min" / "customers.csv", dtype={"customer_id": str}
    ).set_index("customer_id")
    rows = []
    for i, (cid, row) in enumerate(customers.iterrows(), start=1):
        with np.load(ARTIFACTS / "cache_5min" / row["file"]) as cache:
            dates = cache["date"]
            daily_occupancy = np.isfinite(cache["hr"]).mean(axis=1)
        span = int(dates[-1] - dates[0]) + 1
        rows.append({
            "customer_id": cid,
            "adequate_date_span": span,
            "adequate_day_density": len(dates) / span,
            "max_gap_days": int(np.diff(dates).max()) - 1,
            "daily_occupancy_sd": daily_occupancy.std(ddof=1),
        })
        if i % 2000 == 0:
            print(f"Read existing cache descriptors: {i}/{len(customers)}", flush=True)
    return customers.join(pd.DataFrame(rows).set_index("customer_id"))


def subgroup_memberships(info, train_ids):
    labels = pd.DataFrame(index=info.index)
    cutpoints = []
    for dimension, column in DESCRIPTORS.items():
        lo, hi = info.loc[train_ids, column].quantile([1 / 3, 2 / 3])
        labels[dimension] = np.where(
            info[column] <= lo, "low", np.where(info[column] <= hi, "middle", "high")
        )
        cutpoints.append({"dimension": dimension, "lower": lo, "upper": hi})
    groups = [("overall", "all", info.index)]
    for dimension in DESCRIPTORS:
        for level in LEVELS:
            groups.append((dimension, level, info.index[labels[dimension] == level]))
    for first in ["history", "continuity"]:
        for left in LEVELS:
            for right in LEVELS:
                keep = (labels[first] == left) & (labels["occupancy"] == right)
                groups.append((f"{first}_x_occupancy", f"{left}/{right}", info.index[keep]))
    return groups, cutpoints


def main():
    info = load_descriptors()
    allocations = pd.read_parquet(ARTIFACTS / "apple_10_allocations.parquet").set_index("customer_id")
    predictions = pd.read_parquet(ARTIFACTS / "scaling_predictions_test.parquet")
    predictions = predictions[predictions["representation"] == "multirocket"]
    rows, cuts, descriptions = [], [], []
    for allocation in range(10):
        train_ids = allocations.index[allocations[f"alloc_{allocation}"] == "train"]
        groups, cutpoints = subgroup_memberships(info, train_ids)
        cuts.extend({"allocation": allocation, **cut} for cut in cutpoints)
        pred = predictions[predictions["allocation"] == allocation]
        for (budget, classifier), condition in pred.groupby(["budget", "classifier"]):
            # Merge rather than pool: each AUROC pair uses identical participants.
            five = condition[condition["resolution"] == "5m"].set_index("customer_id")
            fifteen = condition[condition["resolution"] == "15m"].set_index("customer_id")
            paired = five[["y_true", "y_score"]].rename(columns={"y_score": "score_5m"}).join(
                fifteen[["y_score"]].rename(columns={"y_score": "score_15m"}), how="inner"
            )
            for dimension, subgroup, ids in groups:
                part = paired.loc[paired.index.isin(ids)]
                n0 = int((part.y_true == 0).sum())
                n1 = int((part.y_true == 1).sum())
                auc5 = roc_auc_score(part.y_true, part.score_5m) if n0 and n1 else np.nan
                auc15 = roc_auc_score(part.y_true, part.score_15m) if n0 and n1 else np.nan
                rows.append({
                    "allocation": allocation, "budget": budget, "classifier": classifier,
                    "dimension": dimension, "subgroup": subgroup,
                    "n_test": len(part), "n_class_0": n0, "n_class_1": n1,
                    "auroc_5m": auc5, "auroc_15m": auc15,
                    "delta_5m_minus_15m": auc5 - auc15,
                })
                if budget == "all_days" and classifier == "ridge_fixed":
                    meta = info.loc[part.index]
                    descriptions.append({
                        "allocation": allocation, "dimension": dimension, "subgroup": subgroup,
                        "n_days_median": meta.n_days.median(),
                        "occupancy_5m_mean": meta.occupancy_rate_5m.mean(),
                        "occupancy_15m_mean": meta.occupancy_rate_15m.mean(),
                        "adequate_day_density_median": meta.adequate_day_density.median(),
                        "max_gap_days_median": meta.max_gap_days.median(),
                    })
    metrics = pd.DataFrame(rows)
    keys = ["budget", "classifier", "dimension", "subgroup"]
    summary = metrics.groupby(keys, sort=False).agg(
        n_allocations=("auroc_5m", "count"),
        n_test_mean=("n_test", "mean"), n_test_min=("n_test", "min"),
        n_class_0_min=("n_class_0", "min"), n_class_1_min=("n_class_1", "min"),
        auroc_5m_mean=("auroc_5m", "mean"), auroc_5m_sd=("auroc_5m", "std"),
        auroc_15m_mean=("auroc_15m", "mean"), auroc_15m_sd=("auroc_15m", "std"),
        delta_mean=("delta_5m_minus_15m", "mean"),
        delta_sd=("delta_5m_minus_15m", "std"),
        n_5m_better=("delta_5m_minus_15m", lambda x: int((x > 0).sum())),
        n_15m_better=("delta_5m_minus_15m", lambda x: int((x < 0).sum())),
    ).reset_index()
    OUTPUT.mkdir(exist_ok=True)
    metrics.to_csv(OUTPUT / "metrics_by_allocation.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    pd.DataFrame(cuts).to_csv(OUTPUT / "training_cutpoints.csv", index=False)
    pd.DataFrame(descriptions).to_csv(OUTPUT / "subgroup_descriptors.csv", index=False)
    info[list(DESCRIPTORS.values()) + ["adequate_date_span", "max_gap_days", "occupancy_rate_15m"]].describe(
        percentiles=[1 / 3, .5, 2 / 3]
    ).to_csv(OUTPUT / "cohort_descriptors.csv")
    for classifier in ["ridge_fixed", "ridge_selected"]:
        selected = summary[(summary.budget == "all_days") & (summary.classifier == classifier)]
        print(f"\nAll-history {classifier}; positive delta favors 5 minutes:")
        print(selected[["dimension", "subgroup", "n_test_mean", "auroc_5m_mean", "auroc_15m_mean", "delta_mean", "delta_sd", "n_5m_better"]].to_string(index=False))
    print(f"\nSaved aggregate summaries to {OUTPUT}")


if __name__ == "__main__":
    main()
