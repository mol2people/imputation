"""Step 3: Generate balanced 70/15/15 demographic splits using OR-Tools CP-SAT for Garmin.

Generates 5 balanced allocations (700 train / 150 val / 150 test) on the 1,000 Garmin pilot
participants, perfectly matching salutation, age_group, and bmi_group marginals.
"""
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

PILOT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PILOT_DIR / "artifacts" / "garmin_binning_manifest.csv"
OUTPUT_PATH = PILOT_DIR / "artifacts" / "garmin_binning_5_allocations.parquet"


def proportional_targets(series: pd.Series, target_total: int) -> dict[int, int]:
    counts = series.value_counts().to_dict()
    total = len(series)
    apportioned = {}
    remainders = []
    for label, count in counts.items():
        exact = target_total * count / total
        base = int(np.floor(exact))
        apportioned[label] = base
        remainders.append((exact - base, label))
    remainder_needed = target_total - sum(apportioned.values())
    remainders.sort(reverse=True, key=lambda x: (x[0], -int(x[1])))
    for _, label in remainders[:remainder_needed]:
        apportioned[label] += 1
    return apportioned


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
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), f"Solver failed: {solver.status_name(status)}"
    return [bool(solver.value(v)) for v in test_vars]


def allocate_70_15_15(data: pd.DataFrame, stratify_cols: list[str], seed: int):
    is_heldout = allocate_two_way(data, 0.30, stratify_cols, seed=seed)
    heldout_indices = [i for i, h in enumerate(is_heldout) if h]
    
    heldout_df = data.iloc[heldout_indices].copy().reset_index(drop=True)
    is_test = allocate_two_way(heldout_df, 0.50, stratify_cols, seed=seed + 1)
    
    folds = np.array(["train"] * len(data), dtype=object)
    for i, test_flag in zip(heldout_indices, is_test):
        folds[i] = "test" if test_flag else "val"
    return folds


def main():
    print(f"Loading Garmin manifest from {MANIFEST_PATH}...")
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"customer_id": str})
    n = len(manifest)
    print(f"Total participants to allocate: {n}")
    
    stratify_cols = ["salutation", "age_group", "bmi_group"]
    
    print("\nSolving OR-Tools CP-SAT for 5 allocations (700 train / 150 val / 150 test)...")
    result_df = manifest[["customer_id", "user_id", "salutation", "y", "age_group", "bmi_group", "tier_0", "delta_1c_0"]].copy()
    
    for a in range(5):
        seed = 20261014 + a * 100
        folds = allocate_70_15_15(result_df, stratify_cols, seed=seed)
        col_name = f"alloc_{a}"
        result_df[col_name] = folds
        
        tr_cnt = (folds == "train").sum()
        va_cnt = (folds == "val").sum()
        te_cnt = (folds == "test").sum()
        
        tr_sal10 = ((folds == "train") & (result_df["salutation"] == 10)).sum()
        va_sal10 = ((folds == "val") & (result_df["salutation"] == 10)).sum()
        te_sal10 = ((folds == "test") & (result_df["salutation"] == 10)).sum()
        
        print(f"  Alloc {a}: Train={tr_cnt} (sal10={tr_sal10}), Val={va_cnt} (sal10={va_sal10}), Test={te_cnt} (sal10={te_sal10})")
        
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved 5 allocations to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
