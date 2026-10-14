"""Visual checks using saved held-out customer predictions."""
import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import streamlit as st
from heartbeat_lab.calibration import probability_m
from charts import class_color, probability_axis
from model_info import with_model_labels


def show_diagnostics(root, bundles, selected, calibrations, model_color, model_dash, labels):
    with st.expander("Cohort diagnostics: errors and recording history"):
        split = st.radio("Diagnostic split", ["test", "val"], horizontal=True)
        st.caption("Uses saved customer-level predictions. History-length associations are descriptive, not causal. "
                   "Treat test-set exploration as exploratory; freeze a new evaluation before choosing settings from these plots.")
        st.caption("Calibration fits validation customer scores only. Test checks are held out from that fit, "
                   "but this cohort has already been explored. Validation calibration plots are in-sample for the sigmoid. "
                   "These checks do not validate daily, prefix, perturbed or uploaded-data probabilities.")
        customers = pd.read_csv(root / "artifacts/prepared/customers.csv", dtype={"customer_id": str})
        tables = []
        for name in selected:
            p = pd.read_csv(bundles[name] / "predictions.csv", dtype={"customer_id": str})
            p = p.loc[p.split.eq(split)].merge(customers[["customer_id", "adequate_days"]], on="customer_id", validate="one_to_one")
            p["model"] = name
            p["label"] = p.y.map({0: "10 (F)", 1: "20 (M)"})
            p["p_m"] = probability_m(p.score, calibrations[name])
            p["error"] = (p.p_m >= .5).astype(int).ne(p.y)
            tables.append(p)
        if not tables:
            return
        predictions = pd.concat(tables, ignore_index=True)
        predictions = with_model_labels(predictions, labels)
        st.altair_chart(alt.Chart(predictions).mark_bar(opacity=.9).encode(
            x=alt.X("p_m:Q", bin=alt.Bin(step=.05, extent=[0, 1]), title="Estimated P(M)", axis=alt.Axis(format=".0%")),
            y=alt.Y("count():Q", title="Customers"), color=class_color(), row="label:N", column=alt.Column("model_label:N", title="Model")), width="stretch", theme=None)
        st.altair_chart(alt.Chart(predictions).mark_point(opacity=.7, size=40).encode(
            x=alt.X("adequate_days:Q", title="Total available adequate days"),
            y=probability_axis("p_m:Q", "Estimated P(M)"),
            color=class_color(), shape="error:N", column=alt.Column("model_label:N", title="Model"),
            tooltip=["adequate_days:Q", alt.Tooltip("p_m:Q", format=".1%"), "label:N", "error:N"]), width="stretch", theme=None)
        st.write("**Probability reliability — customer histories**")
        reliability, metrics = [], []
        for name, p in predictions.groupby("model"):
            bins = pd.cut(p.p_m, np.linspace(0, 1, 11), include_lowest=True)
            curve = p.groupby(bins, observed=True).agg(predicted=("p_m", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index(drop=True)
            curve["model"] = name
            reliability.append(curve)
            metrics.append(dict(model=labels[name], n=len(p), calibration_customers=calibrations[name]["n"],
                                Brier=brier_score_loss(p.y, p.p_m), Log_loss=log_loss(p.y, p.p_m, labels=[0, 1])))
        diagonal = alt.Chart(pd.DataFrame({"p": [0, 1]})).mark_line(color="#777777", strokeDash=[3, 4]).encode(x="p:Q", y="p:Q")
        curve = alt.Chart(with_model_labels(pd.concat(reliability), labels)).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("predicted:Q", title="Mean estimated P(M)", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            y=probability_axis("observed:Q", "Observed fraction M"), color=model_color, strokeDash=model_dash,
            tooltip=[alt.Tooltip("model_label:N", title="Model"), "n:Q", alt.Tooltip("predicted:Q", format=".1%"), alt.Tooltip("observed:Q", format=".1%")])
        st.altair_chart(curve + diagonal, width="stretch", theme=None)
        st.dataframe(pd.DataFrame(metrics), hide_index=True)
        st.caption("Closer to the diagonal indicates better agreement. Bins can be small: hover for customer counts. "
                   "Brier and log loss assess overall probabilistic accuracy, not calibration alone; lower is better.")
        rows = []
        for name, p in predictions.groupby("model"):
            for band, group in p.groupby(pd.qcut(p.adequate_days, 4, duplicates="drop"), observed=True):
                rows.append(dict(model=labels[name], available_days=str(band), n=len(group),
                                  n_F=int(group.y.eq(0).sum()), n_M=int(group.y.eq(1).sum()),
                                  AUROC=float(roc_auc_score(group.y, group.score)) if group.y.nunique()==2 else np.nan,
                                  error_at_05=float(group.error.mean())))
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Error flags use estimated P(M) ≥50%. Available days can exceed a model's training/prediction day budget. "
                   "Small subgroup estimates are noisy; no subgroup confidence intervals are claimed.")
