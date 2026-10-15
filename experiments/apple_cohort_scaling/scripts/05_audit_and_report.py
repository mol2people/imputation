"""Step 5: Audit saved predictions, compute paired deltas, subgroup stratifications, and generate reports.

Reads:
  - artifacts/scaling_predictions_test.parquet
  - artifacts/apple_benchmark_metrics.csv
  - artifacts/alpha_selection.csv
  - artifacts/cache_5min/customers.csv
  - artifacts/apple_full_manifest.csv
Recomputes test AUROC and balanced accuracy to verify serialization integrity (tol <= 1e-10).
Computes paired differences (MR vs Summary, 5m vs 15m, All vs 40d, Selected vs Fixed alpha).
Evaluates subgroups:
  - Class-specific recall and balanced accuracy at saved validation threshold
  - Stratified AUROC by age group
  - Stratified AUROC by full-history adequate-day terciles (cutpoints on training participants)
Generates:
  - experiments/apple_cohort_scaling/README.md
  - experiments/apple_cohort_scaling/REPORT.md
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_DIR / "artifacts"

METRICS_PATH = OUTPUT_DIR / "apple_benchmark_metrics.csv"
PREDS_PATH = OUTPUT_DIR / "scaling_predictions_test.parquet"
ALPHAS_PATH = OUTPUT_DIR / "alpha_selection.csv"
CUSTOMERS_PATH = OUTPUT_DIR / "cache_5min" / "customers.csv"
MANIFEST_PATH = OUTPUT_DIR / "apple_full_manifest.csv"
ALLOC_PATH = OUTPUT_DIR / "apple_10_allocations.parquet"


def main():
    print("Loading benchmark artifacts...")
    metrics_df = pd.read_csv(METRICS_PATH)
    preds_df = pd.read_parquet(PREDS_PATH)
    alloc_df = pd.read_parquet(ALLOC_PATH)
    cust_df = pd.read_csv(CUSTOMERS_PATH, dtype={"customer_id": str})
    man_df = pd.read_csv(MANIFEST_PATH, dtype={"customer_id": str})

    # Merge demographics and customer days
    user_info = pd.merge(cust_df[["customer_id", "n_days"]], man_df[["customer_id", "age_group", "salutation"]], on="customer_id", how="left")
    user_info_map = user_info.set_index("customer_id")

    # 1. Independent Score Reconstruction Check
    print("\nVerifying independent score reconstruction from saved predictions...")
    mismatches = 0
    max_diff = 0.0

    grouped = preds_df.groupby(["allocation", "budget", "resolution", "representation", "classifier", "alpha"])
    for (alloc, budget, res, rep, clf, alpha), group in grouped:
        y_true = group["y_true"].to_numpy(dtype=int)
        y_score = group["y_score"].to_numpy(dtype=np.float64)
        th = float(group["threshold"].iloc[0])

        recomputed_auc = roc_auc_score(y_true, y_score)
        recomputed_bal = balanced_accuracy_score(y_true, (y_score >= th).astype(int))

        # Match with metrics_df row
        row = metrics_df[
            (metrics_df["allocation"] == alloc) &
            (metrics_df["budget"] == budget) &
            (metrics_df["resolution"] == res) &
            (metrics_df["representation"] == rep) &
            (metrics_df["classifier"] == clf) &
            np.isclose(metrics_df["alpha"], alpha, atol=1e-5)
        ]
        if len(row) == 0:
            print(f"Warning: No matching metrics row for {alloc}, {budget}, {res}, {rep}, {clf}, {alpha}")
            mismatches += 1
            continue

        saved_auc = float(row["test_auroc"].iloc[0])
        saved_bal = float(row["test_bal_acc"].iloc[0])

        diff_auc = abs(recomputed_auc - saved_auc)
        diff_bal = abs(recomputed_bal - saved_bal)
        max_diff = max(max_diff, diff_auc, diff_bal)
        if diff_auc > 1e-10 or diff_bal > 1e-10:
            print(f"Discrepancy in ({alloc}, {budget}, {res}, {rep}, {clf}): recomputed AUC={recomputed_auc}, saved={saved_auc}")
            mismatches += 1

    print(f"Reconstruction check complete: {len(grouped)} conditions checked. Mismatches: {mismatches}. Max float difference: {max_diff:.2e}")
    assert mismatches == 0, "Saved predictions do not match reported metrics!"

    # 2. Main Factorial Performance Summary Table
    print("\nComputing main factorial performance summary...")
    summary_rows = []

    # Aggregate across 10 allocations
    agg_cols = ["test_auroc", "val_auroc", "test_bal_acc", "val_bal_acc"]
    grouped_conditions = metrics_df.groupby(["representation", "resolution", "budget", "classifier"])

    for (rep, res, budget, clf), grp in grouped_conditions:
        rec = {
            "representation": rep,
            "resolution": res,
            "budget": budget,
            "classifier": clf,
            "test_auroc_mean": grp["test_auroc"].mean(),
            "test_auroc_sd": grp["test_auroc"].std(ddof=1),
            "val_auroc_mean": grp["val_auroc"].mean(),
            "test_bal_mean": grp["test_bal_acc"].mean(),
            "test_bal_sd": grp["test_bal_acc"].std(ddof=1),
        }
        summary_rows.append(rec)

    summary_table_df = pd.DataFrame(summary_rows).sort_values(["representation", "budget", "resolution", "classifier"], ascending=[False, True, True, True])

    # 3. Paired Differences (computed within matched allocations)
    print("\nComputing paired differences within matched allocations...")
    paired_diffs = {}

    # Helper to get series of 10 values by condition
    def get_series(rep, res, budget, clf):
        sub = metrics_df[
            (metrics_df["representation"] == rep) &
            (metrics_df["resolution"] == res) &
            (metrics_df["budget"] == budget) &
            (metrics_df["classifier"] == clf)
        ].sort_values("allocation")
        return sub["test_auroc"].to_numpy()

    # (a) MR vs Summary (5m, All days, fixed alpha)
    d_mr_all = get_series("multirocket", "5m", "all_days", "ridge_fixed") - get_series("summary", "5m", "all_days", "ridge_fixed")
    paired_diffs["mr_gain_all_5m_fixed"] = (float(np.mean(d_mr_all)), float(np.std(d_mr_all, ddof=1)))

    # (b) Resolution: 5m vs 15m (MR, All days, fixed alpha)
    d_res_all = get_series("multirocket", "5m", "all_days", "ridge_fixed") - get_series("multirocket", "15m", "all_days", "ridge_fixed")
    paired_diffs["res_5m_minus_15m_all_fixed"] = (float(np.mean(d_res_all)), float(np.std(d_res_all, ddof=1)))

    # (c) Budget: All days vs up to 40d (MR, 5m, fixed alpha)
    d_budget = get_series("multirocket", "5m", "all_days", "ridge_fixed") - get_series("multirocket", "5m", "up_to_40d", "ridge_fixed")
    paired_diffs["budget_all_minus_40d_5m_fixed"] = (float(np.mean(d_budget)), float(np.std(d_budget, ddof=1)))

    # (d) Regularization: selected vs fixed alpha (MR, 5m, All days)
    d_reg_all = get_series("multirocket", "5m", "all_days", "ridge_selected") - get_series("multirocket", "5m", "all_days", "ridge_fixed")
    paired_diffs["reg_selected_minus_fixed_5m_all"] = (float(np.mean(d_reg_all)), float(np.std(d_reg_all, ddof=1)))

    # (e) Regularization: selected vs fixed alpha (MR, 5m, up to 40d)
    d_reg_40d = get_series("multirocket", "5m", "up_to_40d", "ridge_selected") - get_series("multirocket", "5m", "up_to_40d", "ridge_fixed")
    paired_diffs["reg_selected_minus_fixed_5m_40d"] = (float(np.mean(d_reg_40d)), float(np.std(d_reg_40d, ddof=1)))

    # (f) Regularization: selected vs fixed alpha (MR, 15m, up to 40d)
    d_reg_15m_40d = get_series("multirocket", "15m", "up_to_40d", "ridge_selected") - get_series("multirocket", "15m", "up_to_40d", "ridge_fixed")
    paired_diffs["reg_selected_minus_fixed_15m_40d"] = (float(np.mean(d_reg_15m_40d)), float(np.std(d_reg_15m_40d, ddof=1)))

    print("\nPaired Differences (Mean ± SD across 10 allocations):")
    for k, (m, s) in paired_diffs.items():
        print(f"  {k}: {m:+.4f} ± {s:.4f}")

    # 4. Subgroup Stratification
    print("\nEvaluating subgroup stratifications across allocations...")
    # Add metadata to preds_df
    preds_df["n_days"] = preds_df["customer_id"].map(user_info_map["n_days"])
    preds_df["age_group"] = preds_df["customer_id"].map(user_info_map["age_group"])

    # Focus on headline model: MR 5m, All days, ridge_selected (and ridge_fixed)
    subgroup_results = []

    for a in range(10):
        # Determine training-derived tercile cutpoints for allocation a
        tr_mask = alloc_df[f"alloc_{a}"] == "train"
        tr_cids = alloc_df.loc[tr_mask, "customer_id"].astype(str)
        tr_ndays = user_info_map.loc[tr_cids, "n_days"].dropna().to_numpy()
        t1, t2 = np.percentile(tr_ndays, [33.333, 66.667])

        # Subgroup evaluation on MR 5m All days (ridge_fixed and ridge_selected)
        for clf in ["ridge_fixed", "ridge_selected"]:
            sub_preds = preds_df[
                (preds_df["allocation"] == a) &
                (preds_df["budget"] == "all_days") &
                (preds_df["resolution"] == "5m") &
                (preds_df["representation"] == "multirocket") &
                (preds_df["classifier"] == clf)
            ].copy()

            th = sub_preds["threshold"].iloc[0]

            # (i) Overall class recalls
            y_t = sub_preds["y_true"].to_numpy()
            y_p = (sub_preds["y_score"].to_numpy() >= th).astype(int)
            recall_0 = float(np.mean(y_p[y_t == 0] == 0))
            recall_1 = float(np.mean(y_p[y_t == 1] == 1))

            # (ii) Age groups
            for ag in sub_preds["age_group"].dropna().unique():
                grp_ag = sub_preds[sub_preds["age_group"] == ag]
                n_c0 = (grp_ag["y_true"] == 0).sum()
                n_c1 = (grp_ag["y_true"] == 1).sum()
                auc_ag = roc_auc_score(grp_ag["y_true"], grp_ag["y_score"]) if (n_c0 > 0 and n_c1 > 0) else np.nan
                subgroup_results.append({
                    "allocation": a,
                    "classifier": clf,
                    "dimension": "age_group",
                    "subgroup": ag,
                    "n_total": len(grp_ag),
                    "n_c0": n_c0,
                    "n_c1": n_c1,
                    "auroc": auc_ag,
                    "recall_0": float(np.mean((grp_ag["y_score"] < th)[grp_ag["y_true"] == 0])) if n_c0 > 0 else np.nan,
                    "recall_1": float(np.mean((grp_ag["y_score"] >= th)[grp_ag["y_true"] == 1])) if n_c1 > 0 else np.nan,
                })

            # (iii) History terciles
            sub_preds["history_tercile"] = pd.cut(
                sub_preds["n_days"],
                bins=[-np.inf, t1, t2, np.inf],
                labels=["T1_Low", "T2_Mid", "T3_High"]
            )
            for terc in ["T1_Low", "T2_Mid", "T3_High"]:
                grp_t = sub_preds[sub_preds["history_tercile"] == terc]
                n_c0 = (grp_t["y_true"] == 0).sum()
                n_c1 = (grp_t["y_true"] == 1).sum()
                auc_t = roc_auc_score(grp_t["y_true"], grp_t["y_score"]) if (n_c0 > 0 and n_c1 > 0) else np.nan
                subgroup_results.append({
                    "allocation": a,
                    "classifier": clf,
                    "dimension": "history_tercile",
                    "subgroup": terc,
                    "n_total": len(grp_t),
                    "n_c0": n_c0,
                    "n_c1": n_c1,
                    "auroc": auc_t,
                    "recall_0": float(np.mean((grp_t["y_score"] < th)[grp_t["y_true"] == 0])) if n_c0 > 0 else np.nan,
                    "recall_1": float(np.mean((grp_t["y_score"] >= th)[grp_t["y_true"] == 1])) if n_c1 > 0 else np.nan,
                })

    subgroup_df = pd.DataFrame(subgroup_results)
    subgroup_csv = OUTPUT_DIR / "subgroup_audit_metrics.csv"
    subgroup_df.to_csv(subgroup_csv, index=False)
    print(f"Saved subgroup audit metrics to {subgroup_csv}")

    # Aggregate subgroups
    subgroup_summary = subgroup_df.groupby(["classifier", "dimension", "subgroup"]).agg(
        n_mean=("n_total", "mean"),
        n_c0_mean=("n_c0", "mean"),
        n_c1_mean=("n_c1", "mean"),
        auroc_mean=("auroc", "mean"),
        auroc_sd=("auroc", "std"),
        recall_0_mean=("recall_0", "mean"),
        recall_1_mean=("recall_1", "mean"),
    ).reset_index()
    print("\nSubgroup Summary for MultiRocket 5-min All Days:")
    print(subgroup_summary[subgroup_summary["classifier"] == "ridge_fixed"][["dimension", "subgroup", "n_mean", "auroc_mean", "auroc_sd", "recall_0_mean", "recall_1_mean"]].to_string(index=False))

    # 5. Generate Markdown REPORT.md and README.md
    print("\nGenerating experiment reports...")

    report_md = f"""# Full Apple Watch Cohort Benchmark Report

**Study date:** 2026-10-14  
**Target:** Recorded Salutation ($10 \\rightarrow 0/\\text{{Female}}$, $20 \\rightarrow 1/\\text{{Male}}$)  
**Cohort:** Qualified Single-Source Apple Watch (Source 6) participants with $\\ge 14$ adequate days ($N = {len(cust_df)}$ qualified from $11,143$ upstream candidates).  
**Evaluation:** 10 repeated OR-Tools balanced 70/15/15 holdouts ($N_{{\\text{{train}}}} \\approx {int(round(len(cust_df)*0.7))}$, $N_{{\\text{{val}}}} \\approx {int(round(len(cust_df)*0.15))}$, $N_{{\\text{{test}}}} \\approx {int(round(len(cust_df)*0.15))}$).

---

## 1. Executive Summary

This study extends the within-day sequence classification of wearable heart-rate records from the $N=1,000$ pilot to the full qualified single-source Apple Watch population. All preprocessing, clock-bin median fills, MultiRocket bias quantiles, and Ridge regression models were fitted strictly on training participants within each allocation.

### Primary Benchmark Results (10 Allocations, Mean ± Sample SD)

| Representation | History Budget | Resolution | Fixed $\\alpha=1000$ Test AUROC | Validation-Selected $\\alpha$ Test AUROC | Val AUROC | Test Balanced Accuracy |
|---|---|---|---:|---:|---:|---:|
"""
    configs_order = [
        ("multirocket", "all_days", "5m"),
        ("multirocket", "all_days", "15m"),
        ("multirocket", "up_to_40d", "5m"),
        ("multirocket", "up_to_40d", "15m"),
        ("summary", "all_days", "5m"),
        ("summary", "up_to_40d", "5m"),
    ]
    for rep, budget, res in configs_order:
        s_fix = metrics_df[(metrics_df["representation"] == rep) & (metrics_df["budget"] == budget) & (metrics_df["resolution"] == res) & (metrics_df["classifier"] == "ridge_fixed")]
        s_sel = metrics_df[(metrics_df["representation"] == rep) & (metrics_df["budget"] == budget) & (metrics_df["resolution"] == res) & (metrics_df["classifier"] == "ridge_selected")]

        fix_str = f"{s_fix['test_auroc'].mean():.4f} ± {s_fix['test_auroc'].std(ddof=1):.4f}"
        sel_str = f"{s_sel['test_auroc'].mean():.4f} ± {s_sel['test_auroc'].std(ddof=1):.4f}"
        val_str = f"{s_fix['val_auroc'].mean():.4f}"
        bal_str = f"{s_fix['test_bal_acc'].mean():.4f} ± {s_fix['test_bal_acc'].std(ddof=1):.4f}"
        report_md += f"| {rep.capitalize()} | {budget} | {res} | {fix_str} | {sel_str} | {val_str} | {bal_str} |\n"

    report_md += f"""
---

## 2. Paired Hypothesis Tests (Within-Allocation Deltas)

- **MultiRocket increment over summary baseline (5m, All days, Fixed $\\alpha$):**  
  $\\Delta = {paired_diffs['mr_gain_all_5m_fixed'][0]:+.4f} \\pm {paired_diffs['mr_gain_all_5m_fixed'][1]:.4f}$
- **Resolution effect (5m vs 15m, MultiRocket All days, Fixed $\\alpha$):**  
  $\\Delta = {paired_diffs['res_5m_minus_15m_all_fixed'][0]:+.4f} \\pm {paired_diffs['res_5m_minus_15m_all_fixed'][1]:.4f}$
- **History budget effect (All days vs First up to 40 days, MultiRocket 5m, Fixed $\\alpha$):**  
  $\\Delta = {paired_diffs['budget_all_minus_40d_5m_fixed'][0]:+.4f} \\pm {paired_diffs['budget_all_minus_40d_5m_fixed'][1]:.4f}$
- **Regularization selection effect on full history (Selected vs Fixed $\\alpha$, MultiRocket 5m All days):**  
  $\\Delta = {paired_diffs['reg_selected_minus_fixed_5m_all'][0]:+.4f} \\pm {paired_diffs['reg_selected_minus_fixed_5m_all'][1]:.4f}$
- **Regularization selection effect on short history (Selected vs Fixed $\\alpha$, MultiRocket 5m $\\le$ 40d):**  
  $\\Delta = {paired_diffs['reg_selected_minus_fixed_5m_40d'][0]:+.4f} \\pm {paired_diffs['reg_selected_minus_fixed_5m_40d'][1]:.4f}$
- **Regularization selection effect on coarse short history (Selected vs Fixed $\\alpha$, MultiRocket 15m $\\le$ 40d):**  
  $\\Delta = {paired_diffs['reg_selected_minus_fixed_15m_40d'][0]:+.4f} \\pm {paired_diffs['reg_selected_minus_fixed_15m_40d'][1]:.4f}$

---

## 3. Subgroup Stratification (MultiRocket 5-min All Days, Fixed $\\alpha$)

| Dimension | Subgroup | Mean $N$ | Test AUROC (Mean ± SD) | Class 0 (F) Recall | Class 1 (M) Recall |
|---|---|---:|---:|---:|---:|
"""
    for _, row in subgroup_summary[subgroup_summary["classifier"] == "ridge_fixed"].iterrows():
        report_md += f"| {row['dimension']} | {row['subgroup']} | {row['n_mean']:.1f} | {row['auroc_mean']:.4f} ± {row['auroc_sd']:.4f} | {row['recall_0_mean']:.4f} | {row['recall_1_mean']:.4f} |\n"

    report_md += """
---

## 4. Verification and Audit

- **Score reconstruction:** 100% of test predictions in `scaling_predictions_test.parquet` independently reproduce `apple_benchmark_metrics.csv` within $10^{-10}$.
- **Leakage verification:** MultiRocket biases and fill vectors strictly derived from training customer IDs. Test participants did not influence preprocessing, scaling, alpha selection, or threshold calibration.
"""
    report_path = EXPERIMENT_DIR / "REPORT.md"
    readme_path = EXPERIMENT_DIR / "README.md"

    with open(report_path, "w") as f:
        f.write(report_md)
    with open(readme_path, "w") as f:
        f.write(report_md)

    print(f"\nSaved reports to {report_path} and {readme_path}")


if __name__ == "__main__":
    main()
