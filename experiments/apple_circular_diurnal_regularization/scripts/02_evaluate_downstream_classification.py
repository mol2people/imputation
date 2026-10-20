"""Milestone 3: Downstream Classification Benchmark for Circular Diurnal Regularization.

Evaluates downstream sex classification performance on Apple Watch Allocation 0 (N=11,139):
  - Training split: N = 7,797
  - Validation split: N = 1,671
  - Test split: N = 1,671

Compares two arms:
  1. Baseline: H_clock (raw user diurnal clock median template)
  2. Regularized: H_circ_gauss_s1 (user diurnal clock median template regularized
     with circular Gaussian smoothing sigma=1 bin, i.e. 5 minutes)

Evaluates:
  - Primary representation: Diurnal Template + Static Circadian Summary features (595 cols)
  - Isolated representation: Diurnal Template only (288 cols)
  - Regularization grid: alpha in {100, 500, 1000, 5000, 10000, 100000} tuned on validation AUROC
  - Overall test AUROC, Average Precision (PR-AUC)
  - Stratified test AUROC across occupancy tiers:
      * Low (< 50% occupancy)
      * Medium (50% - 80% occupancy)
      * High (> 80% occupancy)
  - Delta AUROC = AUROC(H_circ_gauss_s1) - AUROC(H_clock) with 2,000 stratified bootstrap
    iterations for mean, standard error, and 95% confidence intervals.

Outputs:
  - experiments/apple_circular_diurnal_regularization/results/downstream_classification_metrics.csv
  - experiments/apple_circular_diurnal_regularization/results/downstream_test_predictions.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Thread configuration for BLAS/Numba
os.environ["NUMBA_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
REPO_ROOT = PROJECT_DIR.parent.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from circular_smoothing import circular_gaussian_smooth

# Paths to authoritative caches & allocations
ALLOC_PATH = REPO_ROOT / "experiments" / "apple_cohort_scaling" / "artifacts" / "apple_10_allocations.parquet"
CACHE_5M_DIR = REPO_ROOT / "experiments" / "apple_cohort_scaling" / "artifacts" / "cache_5min"

# Add heartbeat_lab for STATS definitions
sys.path.insert(0, str(REPO_ROOT / "HEARTBEAT_LAB" / "src"))
from heartbeat_lab.data import STATS

# Candidate alpha regularization grid (extended grid including 100k)
ALPHAS = [100.0, 500.0, 1000.0, 5000.0, 10000.0, 100000.0]
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 20261020


def extract_participant_features(
    cids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract raw user clock templates, circular smoothed templates, summary features, and occupancies.

    Args:
        cids: List of customer IDs.

    Returns:
        X_clock: Array of shape (N, 288), raw user diurnal median templates.
        X_circ_s1: Array of shape (N, 288), circular Gaussian smoothed templates (sigma=1 bin).
        X_static: Array of shape (N, 307), static summary circadian features.
        occupancies: Array of shape (N,), overall wear occupancy fractions.
    """
    print(f"Extracting diurnal templates and circadian features for {len(cids)} participants...", flush=True)
    t0 = time.time()
    templates_raw = []
    static_features = []
    occupancies = []

    for cid in cids:
        npz_path = CACHE_5M_DIR / f"customer_{cid}.npz"
        with np.load(npz_path) as npz:
            hr = npz["hr"]  # shape (D_u, 288)
            date_series = pd.to_datetime(npz["date"], unit="D")
            days = pd.DataFrame(npz["stats"], columns=STATS)
            days["date"] = date_series

            valid = np.count_nonzero(np.isfinite(hr))
            occupancies.append(valid / hr.size)

            # 1. User-specific diurnal clock median template
            with np.errstate(invalid="ignore"):
                t_clock = np.nanmedian(hr, axis=0).astype(np.float64)

            # Fallback to participant overall median if any bin is completely unobserved
            if np.isnan(t_clock).any():
                v = hr[np.isfinite(hr)]
                med_glob = float(np.nanmedian(v)) if v.size > 0 else 60.0
                t_clock = np.where(np.isnan(t_clock), med_glob, t_clock)

            templates_raw.append(t_clock)

            # 2. Static circadian summary features B_L(u) (matching apple_cohort_scaling)
            weekend = days.date.dt.dayofweek.ge(5)
            if (~weekend).any() and weekend.any():
                contrast = float(days.loc[~weekend, "mean"].mean() - days.loc[weekend, "mean"].mean())
            else:
                contrast = 0.0
            if np.isnan(contrast):
                contrast = 0.0

            s_feats = np.r_[
                days[STATS].mean().to_numpy(),
                days[STATS].std(ddof=1).to_numpy() if len(days) > 1 else np.zeros(len(STATS)),
                contrast,
                np.isfinite(hr).mean(axis=0),
            ].astype(np.float32)
            static_features.append(s_feats)

    X_clock = np.array(templates_raw, dtype=np.float32)
    X_static = np.array(static_features, dtype=np.float32)
    occupancies = np.array(occupancies, dtype=np.float32)

    # 3. Circular Gaussian regularized templates across midnight
    X_circ_s1 = circular_gaussian_smooth(X_clock, sigma=1.0, axis=-1, nan_policy="normalize")

    dt = time.time() - t0
    print(f"Feature extraction completed in {dt:.2f}s ({len(cids) / dt:.1f} users/s).", flush=True)
    return X_clock, X_circ_s1, X_static, occupancies


def compute_stratified_bootstrap_contrast(
    y_true: np.ndarray,
    score_ref: np.ndarray,
    score_alt: np.ndarray,
    n_boot: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Compute bootstrap statistics for Delta AUROC = AUROC(alt) - AUROC(ref).

    Uses stratified bootstrap resampling by sex label to guarantee valid AUROC
    computation across all bootstrap resamples, even in small subgroups.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    delta_point = roc_auc_score(y_true, score_alt) - roc_auc_score(y_true, score_ref)

    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])

        yb = y_true[b_idx]
        a_ref = roc_auc_score(yb, score_ref[b_idx])
        a_alt = roc_auc_score(yb, score_alt[b_idx])
        deltas[b] = a_alt - a_ref

    boot_mean = float(np.mean(deltas))
    boot_se = float(np.std(deltas, ddof=1))
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))

    # Empirical two-sided p-value
    p_le = np.mean(deltas <= 0.0)
    p_ge = np.mean(deltas >= 0.0)
    p_val = float(2.0 * min(p_le, p_ge))
    p_val = min(1.0, max(0.0, p_val))

    return {
        "delta_point": delta_point,
        "boot_mean": boot_mean,
        "boot_se": boot_se,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "p_value": p_val,
    }


def fit_and_evaluate_head(
    X_full: np.ndarray,
    tr_mask: np.ndarray,
    va_mask: np.ndarray,
    te_mask: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    y_te: np.ndarray,
    te_tiers: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, float]:
    """Fit Ridge regression on train, select alpha on validation, evaluate on test and occupancy tiers."""
    X_tr = X_full[tr_mask]
    X_va = X_full[va_mask]
    X_te = X_full[te_mask]

    # Preprocessing pipeline fit strictly on training set
    pipe = make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(0.0), StandardScaler())
    X_tr_std = pipe.fit_transform(X_tr)
    X_va_std = pipe.transform(X_va)
    X_te_std = pipe.transform(X_te)

    # Grid search over candidate alphas
    val_aucs = {}
    models = {}
    for alpha in ALPHAS:
        clf = Ridge(alpha=alpha)
        clf.fit(X_tr_std, y_tr)
        va_score = clf.predict(X_va_std)
        va_auc = roc_auc_score(y_va, va_score)
        val_aucs[alpha] = va_auc
        models[alpha] = clf

    # Select best alpha on validation AUROC; break exact ties with larger alpha
    max_auc = max(val_aucs.values())
    best_alpha = max([alp for alp, auc in val_aucs.items() if np.isclose(auc, max_auc, atol=1e-12)])
    best_clf = models[best_alpha]

    tr_score = best_clf.predict(X_tr_std)
    va_score = best_clf.predict(X_va_std)
    te_score = best_clf.predict(X_te_std)

    tr_auc = roc_auc_score(y_tr, tr_score)
    va_auc = roc_auc_score(y_va, va_score)
    te_auc = roc_auc_score(y_te, te_score)
    te_ap = average_precision_score(y_te, te_score)

    tier_aucs = {}
    for tier in ["low", "medium", "high"]:
        idx = te_tiers == tier
        tier_aucs[tier] = roc_auc_score(y_te[idx], te_score[idx])

    res = {
        "selected_alpha": best_alpha,
        "train_auroc": tr_auc,
        "val_auroc": va_auc,
        "test_auroc": te_auc,
        "test_ap": te_ap,
        "test_auroc_low": tier_aucs["low"],
        "test_auroc_med": tier_aucs["medium"],
        "test_auroc_high": tier_aucs["high"],
    }
    return res, te_score, best_alpha


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate downstream classification for circular diurnal regularization.")
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED, help="Random seed for bootstrap resampling.")
    parser.add_argument("--boot-rounds", type=int, default=BOOTSTRAP_ROUNDS, help="Number of bootstrap resamples.")
    args = parser.parse_args()

    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MILESTONE 3: DOWNSTREAM SEX CLASSIFICATION BENCHMARK")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"Allocation file: {ALLOC_PATH}")
    print(f"Cache directory: {CACHE_5M_DIR}")
    print("=" * 80, flush=True)

    # 1. Load Allocations
    alloc_df = pd.read_parquet(ALLOC_PATH)
    alloc_df["customer_id"] = alloc_df["customer_id"].astype(str)

    tr_mask = (alloc_df["alloc_0"] == "train").to_numpy()
    va_mask = (alloc_df["alloc_0"] == "val").to_numpy()
    te_mask = (alloc_df["alloc_0"] == "test").to_numpy()

    cids_all = alloc_df["customer_id"].tolist()
    y_all = alloc_df["y"].to_numpy(dtype=int)

    y_tr = y_all[tr_mask]
    y_va = y_all[va_mask]
    y_te = y_all[te_mask]

    n_tr, n_va, n_te = len(y_tr), len(y_va), len(y_te)
    print(f"\nCohort Allocation 0 splits: Train N={n_tr}, Val N={n_va}, Test N={n_te} (Total N={len(y_all)})")

    # 2. Extract participant features
    X_clock, X_circ_s1, X_static, occupancies = extract_participant_features(cids_all)

    # Stratify test cohort into occupancy tiers: Low (<50%), Medium (50-80%), High (>80%)
    occ_te = occupancies[te_mask]
    conditions = [
        occ_te < 0.50,
        (occ_te >= 0.50) & (occ_te <= 0.80),
        occ_te > 0.80,
    ]
    tier_choices = ["low", "medium", "high"]
    te_tiers = np.select(conditions, tier_choices, default="unknown")
    tier_counts = pd.Series(te_tiers).value_counts().to_dict()
    print("\nTest cohort occupancy tier breakdown:")
    for t_name in ["low", "medium", "high"]:
        print(f"  Tier {t_name:6s}: N = {tier_counts.get(t_name, 0):4d} ({tier_counts.get(t_name, 0)/n_te*100:5.1f}%)")

    # 3. Setup representations to benchmark
    representations = [
        (
            "diurnal_plus_circadian",
            "Diurnal Template + Circadian Summary Features (595 cols)",
            np.hstack([X_clock, X_static]),
            np.hstack([X_circ_s1, X_static]),
        ),
        (
            "diurnal_template_only",
            "Diurnal Template Only (288 cols)",
            X_clock,
            X_circ_s1,
        ),
    ]

    metrics_records = []
    test_pred_df = pd.DataFrame({
        "customer_id": alloc_df.loc[te_mask, "customer_id"].values,
        "y_true": y_te,
        "occupancy": occ_te,
        "occupancy_tier": te_tiers,
    })

    print("\n" + "=" * 80)
    print("FITTING & EVALUATING CLASSIFICATION ARMS")
    print("=" * 80)

    for rep_key, rep_label, X_rep_clock, X_rep_s1 in representations:
        print(f"\n--- Representation: {rep_label} ---")

        # Evaluate Arm 1: H_clock
        m_clock, score_clock, alpha_clock = fit_and_evaluate_head(
            X_rep_clock, tr_mask, va_mask, te_mask, y_tr, y_va, y_te, te_tiers
        )
        test_pred_df[f"score_{rep_key}_H_clock"] = score_clock

        # Evaluate Arm 2: H_circ_gauss_s1
        m_s1, score_s1, alpha_s1 = fit_and_evaluate_head(
            X_rep_s1, tr_mask, va_mask, te_mask, y_tr, y_va, y_te, te_tiers
        )
        test_pred_df[f"score_{rep_key}_H_circ_gauss_s1"] = score_s1

        print(f"  H_clock:          alpha={alpha_clock:7.1f} | Val={m_clock['val_auroc']:.4f} | Test={m_clock['test_auroc']:.4f} | AP={m_clock['test_ap']:.4f}")
        print(f"    By Tier: Low={m_clock['test_auroc_low']:.4f} | Med={m_clock['test_auroc_med']:.4f} | High={m_clock['test_auroc_high']:.4f}")

        print(f"  H_circ_gauss_s1:  alpha={alpha_s1:7.1f} | Val={m_s1['val_auroc']:.4f} | Test={m_s1['test_auroc']:.4f} | AP={m_s1['test_ap']:.4f}")
        print(f"    By Tier: Low={m_s1['test_auroc_low']:.4f} | Med={m_s1['test_auroc_med']:.4f} | High={m_s1['test_auroc_high']:.4f}")

        # Compute overall contrast
        overall_boot = compute_stratified_bootstrap_contrast(
            y_te, score_clock, score_s1, n_boot=args.boot_rounds, seed=args.seed
        )
        print(f"  Overall Delta AUROC: {overall_boot['delta_point']:+.6f} "
              f"(95% CI: [{overall_boot['ci_lower_95']:+.6f}, {overall_boot['ci_upper_95']:+.6f}], "
              f"SE: {overall_boot['boot_se']:.6f}, p: {overall_boot['p_value']:.4f})")

        # Record Arm rows
        metrics_records.append({
            "representation": rep_key,
            "category": "arm",
            "name": "H_clock",
            "occupancy_tier": "all",
            "n_eval": n_te,
            "selected_alpha": alpha_clock,
            "train_auroc": m_clock["train_auroc"],
            "val_auroc": m_clock["val_auroc"],
            "test_auroc": m_clock["test_auroc"],
            "test_ap": m_clock["test_ap"],
            "delta_auroc": 0.0,
            "boot_mean": 0.0,
            "boot_se": 0.0,
            "ci_lower_95": 0.0,
            "ci_upper_95": 0.0,
            "p_value": np.nan,
        })
        metrics_records.append({
            "representation": rep_key,
            "category": "arm",
            "name": "H_circ_gauss_s1",
            "occupancy_tier": "all",
            "n_eval": n_te,
            "selected_alpha": alpha_s1,
            "train_auroc": m_s1["train_auroc"],
            "val_auroc": m_s1["val_auroc"],
            "test_auroc": m_s1["test_auroc"],
            "test_ap": m_s1["test_ap"],
            "delta_auroc": overall_boot["delta_point"],
            "boot_mean": overall_boot["boot_mean"],
            "boot_se": overall_boot["boot_se"],
            "ci_lower_95": overall_boot["ci_lower_95"],
            "ci_upper_95": overall_boot["ci_upper_95"],
            "p_value": overall_boot["p_value"],
        })

        # Record Contrast row: Overall
        metrics_records.append({
            "representation": rep_key,
            "category": "contrast",
            "name": "delta_circ_minus_clock",
            "occupancy_tier": "all",
            "n_eval": n_te,
            "selected_alpha": alpha_s1,
            "train_auroc": m_s1["train_auroc"] - m_clock["train_auroc"],
            "val_auroc": m_s1["val_auroc"] - m_clock["val_auroc"],
            "test_auroc": overall_boot["delta_point"],
            "test_ap": m_s1["test_ap"] - m_clock["test_ap"],
            "delta_auroc": overall_boot["delta_point"],
            "boot_mean": overall_boot["boot_mean"],
            "boot_se": overall_boot["boot_se"],
            "ci_lower_95": overall_boot["ci_lower_95"],
            "ci_upper_95": overall_boot["ci_upper_95"],
            "p_value": overall_boot["p_value"],
        })

        # Record Contrast rows across Occupancy Tiers
        for t_name in ["low", "medium", "high"]:
            idx_t = te_tiers == t_name
            n_t = int(np.sum(idx_t))
            tier_boot = compute_stratified_bootstrap_contrast(
                y_te[idx_t], score_clock[idx_t], score_s1[idx_t], n_boot=args.boot_rounds, seed=args.seed + hash(t_name) % 1000
            )
            print(f"    Tier {t_name:6s} Delta AUROC: {tier_boot['delta_point']:+.6f} "
                  f"(95% CI: [{tier_boot['ci_lower_95']:+.6f}, {tier_boot['ci_upper_95']:+.6f}], "
                  f"SE: {tier_boot['boot_se']:.6f}, p: {tier_boot['p_value']:.4f})")

            metrics_records.append({
                "representation": rep_key,
                "category": "contrast",
                "name": "delta_circ_minus_clock",
                "occupancy_tier": t_name,
                "n_eval": n_t,
                "selected_alpha": alpha_s1,
                "train_auroc": np.nan,
                "val_auroc": np.nan,
                "test_auroc": tier_boot["delta_point"],
                "test_ap": np.nan,
                "delta_auroc": tier_boot["delta_point"],
                "boot_mean": tier_boot["boot_mean"],
                "boot_se": tier_boot["boot_se"],
                "ci_lower_95": tier_boot["ci_lower_95"],
                "ci_upper_95": tier_boot["ci_upper_95"],
                "p_value": tier_boot["p_value"],
            })

    # Save output metrics CSV
    metrics_df = pd.DataFrame(metrics_records)
    out_csv_path = RESULTS_DIR / "downstream_classification_metrics.csv"
    metrics_df.to_csv(out_csv_path, index=False)
    print(f"\nSaved metrics table to: {out_csv_path}")

    # Save predictions parquet
    out_pred_path = RESULTS_DIR / "downstream_test_predictions.parquet"
    test_pred_df.to_parquet(out_pred_path, index=False)
    print(f"Saved test predictions to: {out_pred_path}")

    total_time = time.time() - t_start
    print(f"\n======================================================================")
    print(f"BENCHMARK COMPLETED SUCCESSFULLY in {total_time:.2f}s")
    print(f"======================================================================")


if __name__ == "__main__":
    main()
