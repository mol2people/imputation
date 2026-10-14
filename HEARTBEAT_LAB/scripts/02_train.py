"""Fit ONE complete model/preprocessing bundle using saved customer splits.

Run this again with the same assignments and another model/fill to compare.
All learned preprocessing sees training customers only; fixed alpha=1000.
"""
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from heartbeat_lab import allocate_customers, validate_assignments
from heartbeat_lab.splits import temporal_customers
from heartbeat_lab.data import STATS
from heartbeat_lab.model import HeartbeatModel
from heartbeat_lab.calibration import calibrate_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path, help="New bundle directory under artifacts/models/")
    parser.add_argument("--assignments", type=Path, required=True, help="Existing CSV, or new path to save once")
    parser.add_argument("--model", choices=["summary", "multirocket"], default="multirocket")
    parser.add_argument("--fill", choices=["clock", "global"], default="clock")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--alpha", type=float, default=1000)
    parser.add_argument("--days", type=int, default=0, help="First N adequate days per window; 0 means all")
    parser.add_argument("--max-customers", type=int, default=0, help="Seeded label-stratified subset; 0 means all")
    parser.add_argument("--protocol", choices=["random", "temporal"], default="random")
    parser.add_argument("--train-before", default="2021-08-10")
    parser.add_argument("--test-from", default="2021-12-22")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.days < 0 or args.days == 1 or args.max_customers < 0 or args.alpha <= 0:
        raise ValueError("Use nonnegative budgets, at least 2 days, and positive alpha.")
    customers = pd.read_csv(args.prepared / "customers.csv", dtype={"customer_id": str})
    if 0 < args.max_customers < len(customers):
        customers, _ = train_test_split(customers, train_size=args.max_customers,
                                         stratify=customers.y, random_state=args.seed)
        customers = customers.sort_values("customer_id").reset_index(drop=True)
    args.output.mkdir(parents=True, exist_ok=False)
    data, hashes = [], {}
    print(f"Loading {len(customers)} customer histories...", flush=True)
    for row in customers.itertuples(index=False):
        path = args.prepared / row.file
        hashes[row.file] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as z:
            days = pd.DataFrame(z["stats"], columns=STATS)
            days.insert(0, "date", pd.to_datetime(z["date"], unit="D"))
            data.append((days, z["hr"].copy()))
    if args.protocol == "temporal":
        customers, data, exclusions = temporal_customers(customers, data, args.train_before, args.test_from)
        generated = customers[["customer_id", "y", "split"]].copy()
        exclusions.to_csv(args.output / "exclusions.csv", index=False)
        if args.assignments.exists():
            assignments = pd.read_csv(args.assignments, dtype={"customer_id": str})
            pd.testing.assert_frame_equal(assignments.sort_values("customer_id").reset_index(drop=True),
                                          generated.sort_values("customer_id").reset_index(drop=True))
        else:
            assignments = generated
    else:
        assignments = (pd.read_csv(args.assignments, dtype={"customer_id": str}) if args.assignments.exists()
                       else allocate_customers(customers, seed=args.seed))
        customers = customers.merge(assignments[["customer_id", "split"]], on="customer_id", validate="one_to_one")
    validate_assignments(customers, assignments)
    if not args.assignments.exists():
        args.assignments.parent.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(args.assignments, index=False)
    if args.days:
        data = [(d.head(args.days).reset_index(drop=True), h[:args.days]) for d, h in data]
    cohort = []
    for split in ("train", "val", "test"):
        indices = np.flatnonzero(customers.split.eq(split))
        cohort.append(dict(split=split, customers=len(indices), class_20=int(customers.iloc[indices].y.sum()),
                           days=sum(len(data[i][0]) for i in indices),
                           first_date=str(min(data[i][0].date.min() for i in indices).date()),
                           last_date=str(max(data[i][0].date.max() for i in indices).date())))
    (args.output / "cohort.json").write_text(json.dumps(cohort, indent=2) + "\n")
    print(json.dumps(cohort), flush=True)
    tr = np.flatnonzero(customers.split.eq("train"))
    model = HeartbeatModel(alpha=args.alpha, seed=args.seed, representation=args.model, filling=args.fill)
    print(f"Fitting {args.model}/{args.fill} on {len(tr)} training customers...", flush=True)
    fit_start = time.perf_counter()
    model.fit([data[i] for i in tr], customers.iloc[tr].y.to_numpy())
    fit_seconds = time.perf_counter() - fit_start
    print("Evaluating validation and test customers...", flush=True)
    predictions, metrics = [], {}
    for split in ("val", "test"):
        indices = np.flatnonzero(customers.split.eq(split))
        scores = model.predict([data[i] for i in indices])
        part = customers.iloc[indices][["customer_id", "y", "split"]].copy()
        part["score"] = scores
        predictions.append(part)
        metrics[split] = dict(n=len(part), auroc=float(roc_auc_score(part.y, scores)))
    metadata = dict(name=args.output.name, model=args.model,
                    filling=args.fill if args.model == "multirocket" else "not applicable",
                    seed=args.seed, alpha=model.alpha, num_features_per_transform=model.num_features,
                    target={"0": "salutation 10 (F)", "1": "salutation 20 (M)"},
                    score="uncalibrated ridge output; not a probability", training_days=model.training_days_,
                    training_customers=len(tr), metrics=metrics,
                    protocol=args.protocol, day_budget=args.days, max_customers=args.max_customers,
                    train_before=args.train_before if args.protocol == "temporal" else None,
                    test_from=args.test_from if args.protocol == "temporal" else None,
                    cohort=cohort, fit_seconds=fit_seconds, total_seconds=time.perf_counter()-started,
                    assignments_sha256=hashlib.sha256(args.assignments.read_bytes()).hexdigest(),
                    customers_sha256=hashlib.sha256((args.prepared / "customers.csv").read_bytes()).hexdigest(),
                    prepared_sha256=hashes,
                    versions={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "numba", "joblib"]})
    reference = Path(__file__).resolve().parents[1] / "vendor/MultiRocket/provenance.json"
    if args.model == "multirocket":
        metadata["reference"] = json.loads(reference.read_text())
    joblib.dump(model, args.output / "model.joblib")
    pd.concat(predictions).to_csv(args.output / "predictions.csv", index=False)
    assignments.to_csv(args.output / "assignments.csv", index=False)
    calibrate_bundle(args.output)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
