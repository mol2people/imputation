"""Run: streamlit run app.py --server.address 127.0.0.1"""
import hashlib
import json
import os
from pathlib import Path

import joblib
import altair as alt
import pandas as pd
import streamlit as st
from heartbeat_lab import prepare_days
from heartbeat_lab.calibration import probability_columns
from charts import BLUE, model_styles, probability_axis, threshold
from model_info import model_labels, describe_model, with_model_labels, show_model_details, show_methods
from stress_panel import show_stress_panel
from training_panel import show_training_panel
from diagnostics_panel import show_diagnostics

ROOT = Path(__file__).resolve().parent
EPOCH_DIR = Path(os.environ.get("HEARTBEAT_EPOCH_DIR", ROOT.parent / "out")).resolve()


@st.cache_data(show_spinner=False)
def study_customers(manifest, modified):
    return pd.read_csv(manifest, dtype={"customer_id": str})


@st.cache_data(show_spinner=False, max_entries=8)
def local_days(path, modified):
    return prepare_days(pd.read_csv(path))


@st.cache_resource(show_spinner=False)
def load_model(path, modified):
    # Only local operator-created bundles, never uploaded model files.
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def model_assignments(path, modified):
    return pd.read_csv(path, dtype={"customer_id": str}).set_index("customer_id")


st.title("Heartbeat Lab")
st.write("Explore daily predictions from heart-rate history and compare models.")
st.caption("Target: recorded salutation 10 (F) / 20 (M). Percentages run from F (0% M) to M (100% M).")
show_methods()
show_training_panel(ROOT)
source = st.radio("Data source", ["Study customer", "Upload CSV"], horizontal=True)
customer_id = None
if source == "Study customer":
    manifest = ROOT / "artifacts/prepared/customers.csv"
    if not manifest.exists():
        st.info("Prepare the study customers with scripts/01_from_repository.py first.")
        st.stop()
    customers = study_customers(str(manifest), manifest.stat().st_mtime_ns)
    customer_id = st.selectbox("Customer ID", customers.customer_id.tolist(), index=0,
                              placeholder="Select or search for a customer ID")
    if customer_id is None:
        st.stop()
    path = EPOCH_DIR / f"{customer_id}.csv"
    if not path.is_file():
        st.error("This customer's CSV was not found in the configured local epoch directory.")
        st.stop()
    record = customers.set_index("customer_id").loc[customer_id]
    st.write(f"Recorded salutation: **{'10 (F)' if record.y == 0 else '20 (M)'}**")
    st.caption("Reads the original customer CSV from the local epoch directory.")
    input_key = (source, customer_id, path.stat().st_mtime_ns)
    with st.spinner("Reading customer history"):
        try:
            days, hr, quality = local_days(str(path), path.stat().st_mtime_ns)
        except (ValueError, OverflowError, OSError) as error:
            st.error(str(error))
            st.stop()
else:
    st.write("Upload one person's Garmin epoch HR CSV.")
    st.caption("Uploads are not written to disk. Local hosting processes them on this machine; "
               "remote hosting sends the CSV to that server.")
    with st.expander("CSV format"):
        st.code("timestamp,hr\n2026-09-01T08:00:00+02:00,68\n2026-09-01T08:05:00+02:00,71", language="text")
        st.write("Include each timestamp's timezone offset. Optional end_timestamp gives interval coverage. "
                 "Native CSVs also work: startTimestamp,endTimestamp,timezoneOffset,longValue,source,type. "
                 "Native times are UTC milliseconds; offsets are minutes east of UTC. One person per file.")
        st.download_button("Download synthetic example", (ROOT / "examples/synthetic_hr.csv").read_bytes(), "synthetic_hr.csv")
    upload = st.file_uploader("Upload customer CSV", type="csv")
    if upload is None:
        st.stop()
    input_key = (source, hashlib.sha256(upload.getbuffer()).hexdigest())
    try:
        days, hr, quality = prepare_days(pd.read_csv(upload))
    except (ValueError, OverflowError) as error:
        st.error(str(error))
        st.stop()

with st.expander("Input quality and heart-rate preview"):
    st.write(quality)
    st.line_chart(days.set_index("date")[["mean"]], y_label="Mean HR (bpm)", color=BLUE)
bundles = {p.parent.name: p.parent for p in sorted((ROOT / "artifacts/models").glob("*/metadata.json"))
           if (p.parent / "model.joblib").exists()}
if not bundles:
    st.info("Train a model bundle with scripts/02_train.py to enable prediction.")
    st.stop()
default = "mr_clock" if "mr_clock" in bundles else next(iter(bundles))
metadata = {name: json.loads((path / "metadata.json").read_text()) for name, path in bundles.items()}
# The original three bundles predate these metadata fields; their saved run used
# a random customer split and all adequate days. Later runs record both fields.
for name in {"mr_clock", "mr_global", "summary"} & metadata.keys():
    metadata[name].setdefault("protocol", "random")
    metadata[name].setdefault("day_budget", 0)
labels = model_labels(metadata)
selected = st.multiselect("Compare models and preprocessing", list(bundles), default=[default], max_selections=3,
                          format_func=labels.get)
view = st.radio("Timeline", ["Individual days (exploratory)", "Cumulative history"], horizontal=True)
st.caption("The sigmoid mapping is fitted on validation customers' whole/budgeted histories. "
           "Daily and cumulative percentages are pseudo-probabilities: calibration at these history lengths "
           "has not been validated. Cumulative predictions use only data up to each date.")
calibrations = {}
for name in selected:
    path = bundles[name] / "calibration.json"
    if not path.exists():
        st.error(f"Calibrate {name} first with scripts/03_calibrate.py artifacts/models/{name}")
        st.stop()
    calibrations[name] = json.loads(path.read_text())
model_color, model_dash = model_styles(bundles, selected, labels)
for name in selected:
    info = metadata[name]
    detail = f"**{labels[name]}**"
    if customer_id is not None:
        path = bundles[name] / "assignments.csv"
        assignments = model_assignments(str(path), path.stat().st_mtime_ns)
        split = assignments.loc[customer_id, "split"] if customer_id in assignments.index else "not in this model's cohort"
        detail += f"; customer split: **{split}**"
        if split == "train":
            detail += " (in-sample)"
    st.write(detail)
    st.write(describe_model(info))
    show_model_details(name, info, labels[name], calibrations[name])
    if calibrations[name]["n"] < 100:
        st.caption(f"{labels[name]}: small calibration sample ({calibrations[name]['n']} validation customers); "
                   "probability estimates may be unstable. Inspect the reliability plot below.")
if len({metadata[name]["assignments_sha256"] for name in selected}) > 1:
    st.warning("These bundles use different customer splits; this is not a matched model comparison.")
key = (input_key, tuple((name, (bundles[name] / "model.joblib").stat().st_mtime_ns) for name in selected))
if selected and st.session_state.get("prediction_key") != key:
    tables = []
    for name in selected:
        path = bundles[name] / "model.joblib"
        model = load_model(str(path), path.stat().st_mtime_ns)
        with st.spinner(f"Running {labels[name]}"):
            try:
                table = model.timeline(days, hr)
            except RuntimeError as error:
                st.error(str(error))
                st.stop()
        table["model"] = name
        tables.append(table)
    st.session_state["predictions"] = pd.concat(tables, ignore_index=True)
    st.session_state["prediction_key"] = key
if st.session_state.get("prediction_key") != key:
    st.stop()
predictions = probability_columns(st.session_state["predictions"], calibrations)
predictions = with_model_labels(predictions, labels)
predictions["daily_prediction"] = predictions.daily_p_m.ge(0.5).map({True: 20, False: 10})
predictions["cumulative_prediction"] = predictions.cumulative_p_m.ge(0.5).map({True: 20, False: 10})
predictions["daily_label"] = predictions.daily_prediction.map({10: "10 (F)", 20: "20 (M)"})
predictions["cumulative_label"] = predictions.cumulative_prediction.map({10: "10 (F)", 20: "20 (M)"})
column = "daily_p_m" if view.startswith("Individual") else "cumulative_p_m"
label_column = "daily_label" if view.startswith("Individual") else "cumulative_label"
st.subheader("Predictions")
st.caption("Label: M at ≥50%, otherwise F. Dashed horizontal line: 50%. Raw ridge scores remain in the CSV download.")
for name, group in predictions.groupby("model", sort=False):
    last = group.iloc[-1]
    st.metric(f"{labels[name]} — latest {'daily' if view.startswith('Individual') else 'cumulative'} prediction", last[label_column],
              help=f"Date: {last.date.date()}")
    st.write(f"**F: {1-last[column]:.1%} · M: {last[column]:.1%}** (pseudo-probabilities)")
line = alt.Chart(predictions).mark_line(strokeWidth=3).encode(
    x=alt.X("date:T", title="Date"), y=probability_axis(f"{column}:Q"),
    color=model_color, strokeDash=model_dash,
    tooltip=["date:T", alt.Tooltip("model_label:N", title="Model"), alt.Tooltip(f"{column}:Q", title="Pseudo P(M)", format=".1%")])
st.altair_chart((line + threshold()).interactive(bind_y=False), width="stretch", theme=None)
display = predictions[["date", "model_label", label_column, column, "observed_bins"]].copy()
display[column] = display[column].map(lambda p: f"{p:.1%}")
st.dataframe(display.rename(
    columns={"date": "Date", "model_label": "Model", label_column: "Prediction (F/M)",
             column: "Pseudo P(M)", "observed_bins": "Observed bins"}), hide_index=True)
st.download_button("Download predictions", predictions.to_csv(index=False), "heartbeat_scores.csv", mime="text/csv")

# Keep original events available on demand; perturb raw readings, then rebuild
# both daily summaries and bins through the same preparation functions.
if source == "Study customer":
    raw_path = EPOCH_DIR / f"{customer_id}.csv"
    raw = lambda: pd.read_csv(raw_path)
else:
    import io
    raw = lambda: pd.read_csv(io.BytesIO(upload.getvalue()))
models = {name: load_model(str(bundles[name] / "model.joblib"),
                          (bundles[name] / "model.joblib").stat().st_mtime_ns) for name in selected}
show_stress_panel(raw, days, hr, models, key, calibrations, model_color, model_dash, labels)
show_diagnostics(ROOT, bundles, selected, calibrations, model_color, model_dash, labels)
st.download_button("Download project audit report", (ROOT / "AUDIT_REPORT.md").read_text(),
                   "heartbeat_project_audit.md", mime="text/markdown")
