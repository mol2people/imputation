"""Generate 10 balanced 70/15/15 allocations for all 4,410 single-source Garmin users using OR-Tools CP-SAT.

Balances salutation (exact proportional integer targets), age_group, and bmi_group
using hierarchical joint-cell and marginal deviation penalties.
"""
import time
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "relaxed_garmin_manifest.csv"
OUTPUT_DIR = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "allocations_tier_1c"


def proportional_targets(values: pd.Series, target: int) -> dict[int, int]:
    counts = values.value_counts()
    ideal = counts.to_numpy(dtype=float) * target / len(values)
    assigned = np.floor(ideal).astype(int)
    remaining = target - int(assigned.sum())
    labels = counts.index.tolist()
    order = sorted(range(len(labels)), key=lambda index: (-(ideal[index] - assigned[index]), labels[index]))
    for index in order[:remaining]:
        assigned[index] += 1
    return {int(label): int(count) for label, count in zip(labels, assigned)}


def allocate_two_way(data: pd.DataFrame, test_frac: float, stratify_cols: list[str], seed: int = 20261014):
    n = len(data)
    test_target = int(round(n * test_frac))
    ratio = Fraction(test_frac).limit_denominator(100)
    test_num, test_den = ratio.numerator, ratio.denominator
    
    model = cp_model.CpModel()
    test_vars = [model.new_bool_var(f"t_{i}") for i in range(n)]
    model.add(sum(test_vars) == test_target)
    
    if "y" in stratify_cols or "salutation" in stratify_cols:
        y_col = "y" if "y" in stratify_cols else "salutation"
        class_targets = proportional_targets(data[y_col], test_target)
        for label, target in class_targets.items():
            pos = np.flatnonzero(data[y_col].to_numpy(dtype=int) == label).tolist()
            model.add(sum(test_vars[i] for i in pos) == target)
            
    # Joint deviations
    joint_devs = []
    groups = data.groupby(stratify_cols, dropna=False, sort=True).indices
    for number, positions in enumerate(groups.values()):
        pos = list(map(int, positions))
        c_size = len(pos)
        expr = sum(test_vars[p] for p in pos)
        dev = model.new_int_var(0, test_den * c_size, f"j_{number}")
        model.add_abs_equality(dev, test_den * expr - test_num * c_size)
        joint_devs.append(dev)
        
    # Marginal deviations
    marg_devs = []
    for col in stratify_cols:
        if col in ("y", "salutation"):
            continue
        col_groups = data.groupby([col], dropna=False, sort=True).indices
        for number, positions in enumerate(col_groups.values()):
            pos = list(map(int, positions))
            c_size = len(pos)
            expr = sum(test_vars[p] for p in pos)
            dev = model.new_int_var(0, test_den * c_size, f"m_{col}_{number}")
            model.add_abs_equality(dev, test_den * expr - test_num * c_size)
            marg_devs.append(dev)
            
    rng = np.random.default_rng(seed)
    tie_weights = rng.integers(1, 10, size=n).astype(int)
    
    obj = 10_000_000 * sum(joint_devs) + 1_000 * sum(marg_devs) + sum(int(w) * v for w, v in zip(tie_weights, test_vars))
    model.minimize(obj)
    
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    solver.parameters.max_time_in_seconds = 15.0
    status = solver.solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), f"Solver failed: {solver.status_name(status)}"
    return [bool(solver.value(v)) for v in test_vars]


def allocate_70_15_15(data: pd.DataFrame, stratify_cols: list[str], seed: int):
    # Step 1: 70% train vs 30% heldout
    is_heldout = allocate_two_way(data, 0.30, stratify_cols, seed=seed)
    heldout_indices = [i for i, h in enumerate(is_heldout) if h]
    
    # Step 2: 50% val vs 50% test on heldout
    heldout_df = data.iloc[heldout_indices].copy().reset_index(drop=True)
    is_test = allocate_two_way(heldout_df, 0.50, stratify_cols, seed=seed + 1)
    
    folds = np.array(["train"] * len(data), dtype=object)
    for i, test_flag in zip(heldout_indices, is_test):
        folds[i] = "test" if test_flag else "val"
    return folds


def main():
    print(f"Loading manifest from {MANIFEST_PATH}...")
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"customer_id": str})
    t1c = manifest[manifest["tier_1c"] == True].copy().sort_values("customer_id").reset_index(drop=True)
    n = len(t1c)
    print(f"Tier 1c participants: {n}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stratify_cols = ["salutation", "age_group", "bmi_group"]
    
    alloc_records = t1c[["customer_id", "user_id", "salutation", "y", "age_group", "bmi_group", "delta_1c_0"]].copy()
    
    print("\nGenerating 10 balanced allocations with OR-Tools CP-SAT...")
    t0_all = time.time()
    
    for alloc in range(10):
        t0 = time.time()
        seed = 20261014 + alloc * 10
        folds = allocate_70_15_15(t1c, stratify_cols, seed=seed)
        alloc_records[f"alloc_{alloc}"] = folds
        t_sec = time.time() - t0
        counts = pd.Series(folds).value_counts().to_dict()
        print(f"  Allocation {alloc:1d} (seed {seed}): train={counts.get('train', 0)}, val={counts.get('val', 0)}, test={counts.get('test', 0)} ({t_sec:.2f}s)")
        
    print(f"\nAll 10 allocations completed in {time.time() - t0_all:.2f}s.")
    
    out_csv = OUTPUT_DIR / "tier1c_10_allocations.csv"
    out_parquet = OUTPUT_DIR / "tier1c_10_allocations.parquet"
    alloc_records.to_csv(out_csv, index=False)
    alloc_records.to_parquet(out_parquet, index=False)
    print(f"Saved allocations to {out_csv} and {out_parquet}.")


if __name__ == "__main__":
    main()
