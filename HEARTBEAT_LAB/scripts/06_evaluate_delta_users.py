"""Evaluate leakage-free HEARTBEAT_LAB models on the 562 delta Garmin users.

These users passed v2 single-source eligibility (>=14 adequate days) but failed the
strict 13-week continuous window gate. They were NEVER seen during training of mr_global
or mr_clock, providing an untouched out-of-cohort zero-shot transportability test.
"""
import sys
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss, log_loss

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "HEARTBEAT_LAB" / "src"))
from heartbeat_lab.data import STATS

CACHE_DIR = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "relaxed_cache" / "delta_562"
MODELS_DIR = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "models"
OUTPUT_DIR = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "eval_delta_562"


def bootstrap_ci(y_true, y_score, n_boot=1000, seed=20261014):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    idx0 = np.flatnonzero(y_true == 0)
    idx1 = np.flatnonzero(y_true == 1)
    
    boot_aurocs = []
    for _ in range(n_boot):
        b0 = rng.choice(idx0, size=len(idx0), replace=True)
        b1 = rng.choice(idx1, size=len(idx1), replace=True)
        b_idx = np.r_[b0, b1]
        try:
            boot_aurocs.append(roc_auc_score(y_true[b_idx], y_score[b_idx]))
        except Exception:
            pass
    if not boot_aurocs:
        return np.nan, np.nan
    return float(np.percentile(boot_aurocs, 2.5)), float(np.percentile(boot_aurocs, 97.5))


def load_delta_customers():
    cust_df = pd.read_csv(CACHE_DIR / "customers.csv")
    customers_data = []
    for row in cust_df.itertuples(index=False):
        npz = np.load(CACHE_DIR / row.file)
        date_series = pd.to_datetime(npz["date"], unit="D")
        days = pd.DataFrame(npz["stats"], columns=STATS)
        days["date"] = date_series
        hr = npz["hr"]
        customers_data.append((days, hr))
    return cust_df, customers_data


def main():
    print(f"Loading 562 delta customers from {CACHE_DIR}...")
    cust_df, customers_data = load_delta_customers()
    n_total = len(cust_df)
    y_true = cust_df["y"].to_numpy()
    print(f"Loaded {n_total} customers (sal10: {(y_true == 0).sum()}, sal20: {(y_true == 1).sum()}).")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    models_to_test = ["summary", "mr_clock", "mr_global"]
    results = {}
    predictions_df = cust_df[["customer_id", "y", "salutation", "adequate_days", "fail_reason"]].copy()
    
    for mname in models_to_test:
        model_path = MODELS_DIR / mname / "model.joblib"
        cal_path = MODELS_DIR / mname / "calibration.json"
        if not model_path.exists():
            print(f"Skipping {mname}: {model_path} not found.")
            continue
            
        print(f"\n--- Evaluating {mname} ---")
        model = joblib.load(model_path)
        scores = model.predict(customers_data)
        predictions_df[f"score_{mname}"] = scores
        
        # Calibration
        if cal_path.exists():
            with open(cal_path) as f:
                cal_data = json.load(f)
            slope = cal_data.get("slope", 1.0)
            intercept = cal_data.get("intercept", 0.0)
            prob = 1.0 / (1.0 + np.exp(-(slope * scores + intercept)))
            predictions_df[f"prob_{mname}"] = prob
            brier = brier_score_loss(y_true, prob)
            logloss = log_loss(y_true, prob)
            bal_acc = balanced_accuracy_score(y_true, prob >= 0.5)
        else:
            brier, logloss = np.nan, np.nan
            bal_acc = balanced_accuracy_score(y_true, scores >= 0.5)
            
        auroc = roc_auc_score(y_true, scores)
        ci_lo, ci_hi = bootstrap_ci(y_true, scores)
        
        print(f"Overall AUROC: {auroc:.4f} [95% CI: {ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"Balanced Accuracy: {bal_acc:.4f}, Brier: {brier:.4f}, Log loss: {logloss:.4f}")
        
        # Baseline reference from AUDIT_REPORT / model metadata
        meta_path = MODELS_DIR / mname / "metadata.json"
        ref_test_auroc = np.nan
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            ref_test_auroc = meta.get("metrics", {}).get("test", {}).get("auroc", np.nan)
        print(f"Reference Strict-Cohort Test AUROC: {ref_test_auroc:.4f}")
        print(f"Delta (Transport Drop): {auroc - ref_test_auroc:+.4f}")
        
        # 1. Stratification by history length
        strata_days = {
            "<40 days": cust_df["adequate_days"] < 40,
            "40-79 days": cust_df["adequate_days"].between(40, 79),
            ">=80 days": cust_df["adequate_days"] >= 80,
        }
        days_breakdown = {}
        print("\nBreakdown by History Length:")
        for sname, smask in strata_days.items():
            sy = y_true[smask]
            ss = scores[smask]
            if len(np.unique(sy)) == 2:
                s_auroc = roc_auc_score(sy, ss)
                s_lo, s_hi = bootstrap_ci(sy, ss)
            else:
                s_auroc, s_lo, s_hi = np.nan, np.nan, np.nan
            days_breakdown[sname] = {
                "n": int(smask.sum()),
                "n_F": int((sy == 0).sum()),
                "n_M": int((sy == 1).sum()),
                "auroc": s_auroc,
                "ci_95": [s_lo, s_hi]
            }
            print(f"  {sname:12s} (n={smask.sum():3d}, F={days_breakdown[sname]['n_F']:2d}, M={days_breakdown[sname]['n_M']:2d}): AUROC = {s_auroc:.4f} [{s_lo:.4f}, {s_hi:.4f}]")
            
        # 2. Stratification by failure reason
        reason_breakdown = {}
        print("\nBreakdown by Failure Reason:")
        for rname in cust_df["fail_reason"].unique():
            rmask = cust_df["fail_reason"] == rname
            ry = y_true[rmask]
            rs = scores[rmask]
            if len(np.unique(ry)) == 2:
                r_auroc = roc_auc_score(ry, rs)
                r_lo, r_hi = bootstrap_ci(ry, rs)
            else:
                r_auroc, r_lo, r_hi = np.nan, np.nan, np.nan
            reason_breakdown[rname] = {
                "n": int(rmask.sum()),
                "n_F": int((ry == 0).sum()),
                "n_M": int((ry == 1).sum()),
                "auroc": r_auroc,
                "ci_95": [r_lo, r_hi]
            }
            print(f"  {rname:30s} (n={rmask.sum():3d}): AUROC = {r_auroc:.4f} [{r_lo:.4f}, {r_hi:.4f}]")
            
        results[mname] = {
            "n": n_total,
            "auroc": auroc,
            "ci_95": [ci_lo, ci_hi],
            "ref_test_auroc": ref_test_auroc,
            "delta_vs_strict": auroc - ref_test_auroc,
            "balanced_accuracy": bal_acc,
            "brier": brier,
            "log_loss": logloss,
            "by_history_length": days_breakdown,
            "by_fail_reason": reason_breakdown
        }
        
    predictions_df.to_csv(OUTPUT_DIR / "delta_predictions.csv", index=False)
    with open(OUTPUT_DIR / "delta_evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nAll evaluations saved to {OUTPUT_DIR}/delta_predictions.csv and delta_evaluation_report.json")


if __name__ == "__main__":
    main()
