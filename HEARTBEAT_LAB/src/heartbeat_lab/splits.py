"""Split customers first; join their assignments onto ALL their observations."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def _customers(rows):
    required = {"customer_id", "y"}
    if not required.issubset(rows) or rows[list(required)].isna().any().any():
        raise ValueError("Need nonmissing customer_id and y columns.")
    if not rows.y.isin([0, 1]).all():
        raise ValueError("y must be 0/1 (recorded salutation 10/20).")
    if rows.groupby("customer_id").y.nunique().gt(1).any():
        raise ValueError("A customer has conflicting labels.")
    return rows[["customer_id", "y"]].drop_duplicates().sort_values("customer_id").reset_index(drop=True)


def allocate_customers(rows, seed=20260930, train=0.7, validation=0.15):
    """Label-stratified customer holdout; no row/day can cross split boundaries.

    For age/BMI/source balance, use the repository's existing cohort_allocator.
    Tiny classes fail explicitly through sklearn instead of losing stratification.
    """
    if not 0 < train < 1 or not 0 < validation < 1 - train:
        raise ValueError("Require positive train, validation and test fractions.")
    customers = _customers(rows)
    # Avoid ceil(40 * 0.15000000000000005) spuriously allocating seven tests.
    development, test = train_test_split(customers, test_size=round(1-train-validation, 12),
                                         stratify=customers.y, random_state=seed)
    train_rows, val = train_test_split(development, test_size=validation/(train+validation),
                                       stratify=development.y, random_state=seed+1)
    result = pd.concat([train_rows.assign(split="train"), val.assign(split="val"),
                        test.assign(split="test")]).sort_values("customer_id").reset_index(drop=True)
    validate_assignments(rows, result)
    return result


def validate_assignments(rows, assignments):
    """Reject omissions, extra IDs, duplicate membership and single-class splits."""
    customers = _customers(rows)
    if not {"customer_id", "split"}.issubset(assignments):
        raise ValueError("Assignments need customer_id and split.")
    if assignments.customer_id.isna().any() or assignments.customer_id.duplicated().any():
        raise ValueError("Each customer must appear exactly once in assignments.")
    if set(assignments.customer_id) != set(customers.customer_id):
        raise ValueError("Assignment IDs do not match the eligible customers.")
    if assignments.split.isna().any() or set(assignments.split) != {"train", "val", "test"}:
        raise ValueError("Use exactly train, val and test splits.")
    joined = customers.merge(assignments[["customer_id", "split"]], on="customer_id", validate="one_to_one")
    if joined.groupby("split").y.nunique().ne(2).any():
        raise ValueError("Each split needs both labels.")


def customer_cv(rows, n_splits=5, seed=20260930):
    """Yield positional train/validation indices for a possibly repeated-row table.

    Call on the OUTER TRAIN partition for hyperparameter tuning. Instantiate
    and fit a fresh complete preprocessing/model pipeline inside every fold.
    """
    customers = _customers(rows)
    if customers.y.value_counts().min() < n_splits or customers.y.nunique() != 2:
        raise ValueError("Need at least n_splits customers per class.")
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, va in cv.split(customers, customers.y):
        yield (np.flatnonzero(rows.customer_id.isin(customers.iloc[tr].customer_id)),
               np.flatnonzero(rows.customer_id.isin(customers.iloc[va].customer_id)))


def temporal_customers(customers, histories, train_before, test_from):
    """Later customers, strictly separated calendar windows, no shared IDs.

    Customer time is the FIRST ADEQUATE RECORDED DAY, not account enrollment.
    Train: first day < A, observations < A. Validation: first day in [A,B),
    observations in [A,B). Test: first day >= B, observations >= B.
    Exclude histories with fewer than two usable days after truncation.
    Returns eligible customers with split, sliced histories and exclusions.
    """
    a, b = pd.Timestamp(train_before), pd.Timestamp(test_from)
    if a >= b or len(customers) != len(histories):
        raise ValueError("Require train_before < test_from and aligned customer histories.")
    kept, data, excluded = [], [], []
    for (_, row), (days, hr) in zip(customers.iterrows(), histories):
        first = days.date.min()
        split = "train" if first < a else "val" if first < b else "test"
        mask = days.date.lt(a) if split == "train" else days.date.ge(b) if split == "test" else days.date.ge(a) & days.date.lt(b)
        if mask.sum() < 2:
            excluded.append({"customer_id": row.customer_id, "reason": "fewer than two days within temporal window"})
            continue
        kept.append({**row.to_dict(), "split": split, "first_date": first.isoformat()})
        data.append((days.loc[mask].reset_index(drop=True), hr[mask.to_numpy()]))
    result = pd.DataFrame(kept)
    if result.empty:
        raise ValueError("No customers remain in these calendar windows.")
    validate_assignments(result, result[["customer_id", "split"]])
    return result, data, pd.DataFrame(excluded, columns=["customer_id", "reason"])
