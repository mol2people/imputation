"""Small Streamlit view for repeated within-customer perturbations."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from heartbeat_lab.data import normalize_epochs
from heartbeat_lab.perturb import MODES, repeated_predictions, perturb_events, shuffle_bins
from heartbeat_lab.calibration import probability_columns
from charts import BLUE, ORANGE, probability_axis, input_color
from model_info import with_model_labels


def show_stress_panel(raw, days, hr, models, input_key, calibrations, model_color, model_dash, labels):
    with st.expander("Stress tests: shuffle, generate, thin or shift days", expanded=True):
        st.write("Choose dates to modify. All other days stay unchanged; fitted models stay fixed.")
        mode = st.selectbox("Perturbation", list(MODES), format_func=MODES.get)
        options = days.date.dt.strftime("%Y-%m-%d").tolist()
        all_days = st.checkbox("Modify all adequate days", value=False)
        selected = options if all_days else st.multiselect("Days to modify", options, default=options[-3:])
        fraction = 1.0
        shift = 6
        if mode in {"shuffle", "bin_shuffle"}:
            fraction = st.slider("Fraction of readings to shuffle", 0.0, 1.0, 1.0, 0.05)
            if mode == "shuffle":
                st.caption("Permutes raw readings within each selected day: timestamps, gaps and each day's HR multiset stay fixed. "
                           "Binning afterward can change the distribution of bin means.")
            else:
                st.caption("Permutes observed five-minute means, preserving their multiset, missing-bin mask and original daily summaries. "
                           "This is a model-input intervention matching the historical shuffle level; it does not produce raw epoch CSVs.")
        elif mode == "resample":
            st.caption("Draws the same number of readings with replacement from each selected day's observed HR values. "
                       "Timestamps stay fixed; the exact moments and temporal dependence need not be preserved.")
        elif mode == "thin":
            fraction = st.slider("Fraction of observations to remove", 0.0, 1.0, 0.5, 0.05)
            st.caption("Each reading is independently removed with this probability. Days failing the original quality gate "
                       "lose their prediction; they are not silently filled back in.")
        else:
            shift = st.slider("Maximum random clock shift (hours)", 0, 12, 6)
            st.caption("For each date/run, shifts all readings by a random 5-minute multiple within ±this range, "
                       "wrapping inside the same date. Preserves event durations and moves the recording mask too.")
        repeats = st.number_input("Repetitions", min_value=1, max_value=200, value=30, step=1)
        seed = st.number_input("Random seed", min_value=0, max_value=2**32-1, value=42, step=1)
        view = st.radio("Stress-test timeline", ["Daily", "Cumulative"], horizontal=True)
        settings = (input_key, tuple(models), mode, tuple(selected), repeats, seed, fraction, shift)
        if st.button("Run repeated stress test", disabled=not selected or not models):
            events, _ = normalize_epochs(raw())
            bar = st.progress(0.0)
            baseline, results = repeated_predictions(events, days, hr, models, selected, mode,
                                                     int(repeats), int(seed), fraction, shift,
                                                     progress=bar.progress)
            first_seed = int(np.random.SeedSequence(int(seed)).generate_state(int(repeats))[0])
            preview_date = pd.Timestamp(selected[0])
            if mode == "bin_shuffle":
                positions = np.flatnonzero(days.date.isin(pd.to_datetime(selected)))
                generated = shuffle_bins(hr[positions], first_seed, fraction)
                i = int(np.flatnonzero(days.iloc[positions].date.eq(preview_date))[0])
                preview = pd.DataFrame({"original": hr[positions[i]], "generated": generated[i]})
                distribution = preview.melt(var_name="kind", value_name="longValue").dropna()
                generated_csv = None
            else:
                sample = perturb_events(events, selected, mode, first_seed, fraction, shift)
                original_day = events.loc[events.date.eq(preview_date)]
                generated_day = sample.loc[sample.date.eq(preview_date)]
                preview = pd.DataFrame({"original": original_day.groupby("bin").longValue.mean(),
                                        "generated": generated_day.groupby("bin").longValue.mean()}).reindex(range(288))
                distribution = pd.concat([original_day[["longValue"]].assign(kind="original"),
                                           generated_day[["longValue"]].assign(kind="generated")], ignore_index=True)
                columns = ["startTimestamp", "endTimestamp", "timezoneOffset", "longValue", "source", "type"]
                generated_csv = sample[columns].to_csv(index=False)
            st.session_state["stress_result"] = (settings, baseline, results, preview, distribution,
                                                  generated_csv)
            bar.empty()
        saved = st.session_state.get("stress_result")
        if saved is None or saved[0] != settings:
            return
        _, baseline, results, preview, distribution, generated_csv = saved
        baseline = probability_columns(baseline, calibrations)
        results = probability_columns(results, calibrations)
        results = with_model_labels(results, labels)
        column = "daily_p_m" if view == "Daily" else "cumulative_p_m"
        stats = results.groupby(["model", "date"])[column].agg(
            median="median", lower=lambda x: x.quantile(.05), upper=lambda x: x.quantile(.95),
            valid_runs="count").reset_index()
        stats = with_model_labels(stats, labels)
        plotted = stats.merge(baseline[["model", "date", column]], on=["model", "date"], validate="one_to_one")
        plotted = plotted.rename(columns={column: "Original", "median": "Perturbed median"})
        chart = alt.Chart(plotted)
        band = chart.mark_area(opacity=.18, color=ORANGE).encode(
            x="date:T", y=probability_axis("lower:Q"), y2="upper:Q")
        lines = chart.transform_fold(["Original", "Perturbed median"], as_=["Series", "p_m"]).mark_line(strokeWidth=2.5).encode(
            x="date:T", y=probability_axis("p_m:Q"),
            color=alt.Color("Series:N", scale=alt.Scale(domain=["Original", "Perturbed median"], range=[BLUE, ORANGE])),
            strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=["Original", "Perturbed median"], range=[[7, 4], [1, 0]])),
            tooltip=["date:T", alt.Tooltip("model_label:N", title="Model"), "Series:N", alt.Tooltip("p_m:Q", format=".1%"), "valid_runs:Q"])
        st.altair_chart((band + lines).properties(height=220).facet(row=alt.Row("model_label:N", title="Model")).interactive(bind_y=False), width="stretch", theme=None)
        st.caption("Blue dashed: original. Orange solid: perturbed median; orange band: 5th–95th percentiles. "
                   "Separate panel per model. Percentages use the same fixed calibration as the main timeline. "
                   "These are sensitivity ranges, not confidence intervals or validated daily probabilities.")
        matched = results.merge(baseline[["date", "model", column]].rename(columns={column: "original"}),
                                on=["date", "model"], validate="many_to_one")
        matched["change"] = 100 * (matched[column] - matched.original)
        by_run = matched.loc[matched.selected_day].groupby(["model", "run"]).change.mean().reset_index()
        finite = with_model_labels(by_run.dropna(subset=["change"]), labels)
        if not finite.empty:
            st.altair_chart(alt.Chart(finite).mark_bar(opacity=.65).encode(
                x=alt.X("change:Q", bin=alt.Bin(maxbins=25), title="Mean change in P(M), percentage points (surviving selected days)"),
                y=alt.Y("count():Q", title="Repetitions"), color=model_color, column=alt.Column("model_label:N", title="Model")), width="stretch", theme=None)
        retention = results.loc[results.selected_day].groupby(["model", "date"]).adequate.mean().reset_index()
        retention = with_model_labels(retention, labels)
        st.altair_chart(alt.Chart(retention).mark_line(strokeWidth=2.5, point=True).encode(
            x="date:T", y=probability_axis("adequate:Q", "Runs retaining an adequate day"),
            color=model_color, strokeDash=model_dash), width="stretch", theme=None)
        st.caption(f"{by_run.change.isna().sum()} model/run combinations had no surviving selected day; "
                   "they are excluded from the score-change histogram. Always inspect retention alongside score changes.")
        st.download_button("Download all perturbation predictions", results.to_csv(index=False),
                           "perturbation_predictions.csv", mime="text/csv")
        st.download_button("Download perturbation summary", stats.to_csv(index=False),
                           "perturbation_summary.csv", mime="text/csv")
        with st.expander(f"Inspect one generated day: {selected[0]}"):
            preview_long = preview.rename_axis("bin").reset_index().melt(id_vars="bin", var_name="kind", value_name="hr")
            st.altair_chart(alt.Chart(preview_long).mark_line(strokeWidth=2.5).encode(
                x=alt.X("bin:Q", title="Five-minute clock bin"), y=alt.Y("hr:Q", title="HR (bpm)", scale=alt.Scale(zero=False)),
                color=input_color(), strokeDash=alt.StrokeDash("kind:N", scale=alt.Scale(
                    domain=["original", "generated"], range=[[7, 4], [1, 0]]))), width="stretch", theme=None)
            edges = np.histogram_bin_edges(distribution.longValue, bins=30)
            histograms = []
            for kind, group in distribution.groupby("kind"):
                counts, _ = np.histogram(group.longValue, bins=edges)
                histograms.append(pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts, "kind": kind}))
            st.altair_chart(alt.Chart(pd.concat(histograms)).mark_bar(opacity=.85).encode(
                x=alt.X("left:Q", title="HR (bpm)"), x2="right:Q",
                y=alt.Y("count:Q", title="Readings / bins"), color=input_color(), row="kind:N"), width="stretch", theme=None)
            if generated_csv is not None:
                st.download_button("Download first generated customer CSV", generated_csv,
                                   "generated_customer.csv", mime="text/csv")
