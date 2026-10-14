"""One local background training job at a time; no queue or external service."""
from datetime import date, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time

import pandas as pd
import streamlit as st
from model_info import MODEL_NAMES, FILL_NAMES, describe_model


@st.cache_resource
def job_state():
    return {"process": None, "output": None, "log": None, "started": None, "lock": Lock()}


def show_training_panel(root):
    with st.expander("Retrain locally and test later customers"):
        st.write("Start a separate background process. New completed bundles appear in the model selector after refresh.")
        model = st.selectbox("Model to train", ["summary", "multirocket"], format_func=MODEL_NAMES.get)
        fill = st.selectbox("Training fill", ["clock", "global"], disabled=model == "summary", format_func=FILL_NAMES.get,
                            help="Applies to missing HR bins in MultiRocket. HR summaries do not need bin-value filling.")
        st.caption(describe_model({"model": model, "filling": fill}))
        protocol = st.radio("Validation design", ["temporal", "random"], horizontal=True,
                            format_func={"temporal": "Time split — later customers", "random": "Random customer split"}.get)
        st.caption("Temporal: train on earlier customers and pre-cutoff days; validate and test on later customers in separate "
                   "calendar windows. Customer time means first adequate recorded day. This still uses the retrospectively selected study cohort.")
        cutoff = st.date_input("Training observations strictly before", value=date(2021, 8, 10))
        test_from = st.date_input("Test customers first observed on or after", value=date(2021, 12, 22))
        customers = st.selectbox("Customer budget", [200, 500, 1000, 3848], index=0)
        day_budget = st.selectbox("Adequate days per customer/window", [7, 40, 80, "All"], index=1)
        seed = st.number_input("Training seed", min_value=0, max_value=2**32-2, value=20260930)
        st.caption("Smaller budgets change the experiment. Summary ridge is the fast check; full-history MultiRocket takes minutes. "
                   "Repeatedly choosing settings using these test results makes the comparison exploratory.")
        state = job_state()
        process = state["process"]
        running = process is not None and process.poll() is None
        ready = (root / "artifacts/prepared/customers.csv").exists()
        if st.button("Start local training", disabled=running or not ready):
            if protocol == "temporal" and cutoff >= test_from:
                st.error("Training cutoff must precede the test start.")
            else:
                with state["lock"]:
                    process = state["process"]
                    if process is None or process.poll() is not None:
                        tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        job = root / "artifacts/jobs" / tag
                        job.mkdir(parents=True)
                        output = root / "artifacts/models" / f"{protocol}_{model}_{fill}_{tag}"
                        command = [sys.executable, str(root / "scripts/02_train.py"),
                                   str(root / "artifacts/prepared"), str(output),
                                   "--assignments", str(job / "assignments.csv"), "--model", model,
                                   "--fill", fill, "--protocol", protocol, "--train-before", str(cutoff),
                                   "--test-from", str(test_from), "--max-customers", str(customers),
                                   "--days", str(0 if day_budget == "All" else day_budget), "--seed", str(seed)]
                        env = {**os.environ, "PYTHONUNBUFFERED": "1", "OPENBLAS_NUM_THREADS": "4",
                               "VECLIB_MAXIMUM_THREADS": "4", "OMP_NUM_THREADS": "4"}
                        log = job / "training.log"
                        with log.open("w") as stream:
                            state["process"] = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                                                  cwd=root, env=env)
                        state.update(output=output, log=log, started=time.time())
        st.button("Refresh training status")
        process = state["process"]
        if process is not None:
            elapsed = time.time() - state["started"]
            status = process.poll()
            if status is None:
                st.info(f"Training in background — {elapsed:.0f} seconds elapsed.")
            elif status == 0:
                st.success("Training completed. The new run is available under Compare models and preprocessing.")
            else:
                st.error("Training failed. The log below gives the reason; existing bundles were retained.")
            cohort = state["output"] / "cohort.json"
            if cohort.exists():
                counts = pd.DataFrame(json.loads(cohort.read_text()))
                counts["F (10)"] = counts.customers - counts.class_20
                st.dataframe(counts.rename(columns={"class_20": "M (20)"}), hide_index=True)
            metadata = state["output"] / "metadata.json"
            if metadata.exists():
                info = json.loads(metadata.read_text())
                st.write({"fit_seconds": round(info["fit_seconds"], 1), "total_seconds": round(info["total_seconds"], 1),
                          "validation_AUROC": info["metrics"]["val"]["auroc"], "test_AUROC": info["metrics"]["test"]["auroc"]})
            st.code(state["log"].read_text()[-4000:], language="text")
