"""Shared explicit colors and percentage axes; never use theme color cycles."""
import altair as alt
import pandas as pd

BLUE, ORANGE, PURPLE = "#0072B2", "#D55E00", "#AA4499"
PALETTE = [BLUE, ORANGE, PURPLE, "#009E73", "#CC79A7", "#666666"]
DASHES = [[1, 0], [8, 4], [2, 3], [10, 3, 2, 3], [4, 2], [12, 5]]


def model_styles(names, selected, labels):
    # Canonical bundles retain colors when selections change; new models follow.
    domain = ["mr_clock", "mr_global", "summary"]
    domain += sorted(set(names) - set(domain))
    legend = alt.Legend(values=[labels[name] for name in selected], labelLimit=500)
    readable_domain = [labels.get(name, name) for name in domain]
    return (alt.Color("model_label:N", scale=alt.Scale(domain=readable_domain, range=[PALETTE[i % len(PALETTE)] for i in range(len(domain))]), title="Model", legend=legend),
            alt.StrokeDash("model_label:N", scale=alt.Scale(domain=readable_domain, range=[DASHES[i % len(DASHES)] for i in range(len(domain))]), title="Model", legend=legend))


def probability_axis(field, title="Pseudo-probability of M"):
    return alt.Y(field, title=title, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%"))


def threshold():
    return alt.Chart(pd.DataFrame({"threshold": [.5]})).mark_rule(color="#777777", strokeDash=[3, 4]).encode(y="threshold:Q")


def class_color():
    return alt.Color("label:N", scale=alt.Scale(domain=["10 (F)", "20 (M)"], range=[BLUE, ORANGE]), title="Recorded label")


def input_color():
    return alt.Color("kind:N", scale=alt.Scale(domain=["original", "generated"], range=[BLUE, ORANGE]), title="Data")
