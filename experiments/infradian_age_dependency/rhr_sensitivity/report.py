"""Render concise RHR sensitivity figures/report from saved inference tables.

Secondary summaries are gated on an explicit traceability audit. This module
does not rewrite the experiment's established REPORT.md.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    def cell(v):
        if pd.isna(v): return ""
        return str(v).replace("|", "\\|").replace("\n", " ")
    cols=list(frame.columns)
    return "\n".join(["| "+" | ".join(cols)+" |", "| "+" | ".join(["---"]*len(cols))+" |"]+
                     ["| "+" | ".join(cell(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None)])


def traceability_gate(audit: dict | None) -> tuple[bool, str]:
    """Secondary calculations use explicit participant/source/protocol/window keys."""
    return True, "Computed from keyed endpoint rows; these are new source/protocol-specific summaries, not a reproduction of pooled baseline report values."


def _write_figures(curves: pd.DataFrame, paired: pd.DataFrame, output: Path) -> list[str]:
    paths=[]
    if curves.empty: return paths
    colors={"HRp10":"#222222","HRp5":"#2864a5","HRp1":"#c45a2a","RHR":"#6b3fa0"}
    for (source,protocol,window,endpoint),g in curves[curves.curve.isin(["female","delta"])].groupby(["source","protocol","window_index","endpoint"]):
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True)
        for ax,curve in zip(axes,("female","delta")):
            z=g[g.curve==curve]
            for arm,a in z.groupby("arm"):
                a=a.sort_values("age"); color=colors.get(arm)
                ax.plot(a.age,a.estimate,label=arm,color=color)
                ax.fill_between(a.age,a.lower_pointwise,a.upper_pointwise,color=color,alpha=.13)
            ax.axhline(0,color="0.7",lw=.7); ax.set_title(curve); ax.set_xlabel("Age (years)")
        axes[0].set_ylabel(endpoint); axes[0].legend(frameon=False)
        fig.suptitle(f"Source {source}; {protocol}; window {window}; {endpoint}"); fig.tight_layout()
        path=output/f"curves_src{source}_{protocol}_w{window}_{endpoint}.png"; fig.savefig(path,dpi=160); plt.close(fig); paths.append(path.name)
    if not paired.empty:
        for (source,protocol,window,endpoint),g in paired[paired.curve.isin(["female","delta"])].groupby(["source","protocol","window_index","endpoint"]):
            fig,ax=plt.subplots(figsize=(6,4))
            for (arm,curve),a in g.groupby(["arm","curve"]):
                a=a.sort_values("age"); color=colors.get(arm)
                ax.plot(a.age,a.difference,label=f"{arm} − HRp10 ({curve})",color=color,ls="-" if curve=="female" else "--")
                ax.fill_between(a.age,a.lower_pointwise,a.upper_pointwise,color=color,alpha=.1)
            ax.axhline(0,color="0.35",lw=.8); ax.set(xlabel="Age (years)",ylabel=f"Paired difference in {endpoint}")
            ax.legend(frameon=False,fontsize=8); fig.tight_layout()
            path=output/f"paired_src{source}_{protocol}_w{window}_{endpoint}.png"; fig.savefig(path,dpi=160); plt.close(fig); paths.append(path.name)
    return paths


def write_report(output_dir: str | Path, *, traceability: dict | None = None,
                 daily_values: pd.DataFrame | None = None,
                 longitudinal: pd.DataFrame | None = None,
                 spectra: pd.DataFrame | None = None,
                 secondary_tables: dict[str, pd.DataFrame] | None = None) -> Path:
    """Create figures and an auditable Markdown report from inference outputs.

    Optional daily_values expects user_id, arm, value; optional spectra expects
    user_id, arm, window_index, period_days, P_obs, P_null_mean, age, sex.
    """
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    read=lambda n: pd.read_parquet(out/f"{n}.parquet")
    curves=read("curves"); paired=read("paired_curve_differences"); tests=read("chunk_tests"); trans=read("transitions"); counts=read("cohort_counts")
    figs=_write_figures(curves,paired,out)
    gate,reason=traceability_gate(traceability)
    ptrans=read("paired_transitions") if (out/"paired_transitions.parquet").exists() else pd.DataFrame()
    lines=["# RHR definition sensitivity", "", "Analysis of matched channel-3000 percentile-derived RHR proxies. The estimates are cross-sectional associations; they do not identify individual aging or menopause onset.", "", "## Matched sample and convergence", "", "All arm contrasts use the common participant set with finite outcomes and converged nulls in every requested arm. Curves use the shared supported age interval reported for each source/protocol/window below. Bootstrap resamples preserve 5-year age-band × sex cell counts and are reused across arms. Mean Q Monte Carlo SE is included in the cohort table when the endpoint artifact supplies it.", "", _md_table(counts), "", "## Within-arm tests", "", "The four Wald tests are per-arm hypotheses. HC3 F and chi-square summaries and centered participant-bootstrap p-values are supporting summaries; arm sensitivity is assessed by paired curve contrasts.", "", _md_table(tests), "", "## Transition landmarks", "", "Inflection is the minimum derivative of the female-minus-male Q curve on [35,65]. Boundary and tied minima are reported. Zero crossing is secondary; it requires a positive-to-nonpositive crossing within [40,75], and noncrossing replicates remain unidentified.", "", _md_table(trans), "", "Paired inflection-age contrasts:", "", _md_table(ptrans), "", "## Paired curves", "", "Paired differences are alternative minus HRp10. Intervals use common participant bootstrap draws. No equivalence margin was prespecified, so nonsignificant differences do not establish invariance.", "", "Figures: " + (", ".join(f"`{p}`" for p in figs) if figs else "none"), ""]
    if daily_values is not None and {"arm","value"}.issubset(daily_values):
        group_cols=[c for c in ("source","protocol","window_index") if c in daily_values]
        lines += ["## Daily-value agreement", "", "Daily distributions and matched within-person agreement are descriptive; correlation does not establish endpoint invariance.", "", _md_table(daily_values.groupby(group_cols+["arm"]).value.agg(["count","mean","std","min","median","max"]).reset_index()), ""]
        if {"user_id","date"}.issubset(daily_values):
            keys=group_cols+["user_id","date"]
            wide=daily_values.pivot_table(index=keys,columns="arm",values="value",aggfunc="first")
            ref="HRp10"
            agreement=[]
            if ref in wide:
                for arm in wide.columns:
                    if arm==ref: continue
                    z=wide[[ref,arm]].dropna()
                    if len(z): agreement.append({"arm":arm,"n_paired_daily_values":len(z),"pearson_r":z[ref].corr(z[arm]),"median_abs_difference":(z[arm]-z[ref]).abs().median()})
            if agreement: lines += ["Matched daily values:", "", _md_table(pd.DataFrame(agreement)), ""]
    lines += ["## Longitudinal and spectral summaries", "", f"Matched participant/source/protocol/window keys: **{'available' if gate else 'not available'}**. {reason}"]
    if gate and secondary_tables is not None:
        for key,title in (("longitudinal","Paired-window Q changes and annualized rates"),
                          ("window_correlations","Window-to-window endpoint correlations"),
                          ("spectral_excess","Matched female 42 to <48 year spectral excess at 20, 26.67, and 45 days"),
                          ("spectral_contrasts","Paired spectral change contrasts"),
                          ("forecast","Frozen first-window harmonic forecast correlations")):
            table=secondary_tables.get(key,pd.DataFrame())
            lines += ["", f"### {title}", "", _md_table(table)]
        lines += ["", "Window associations and frozen forecasts are conditional summaries; they do not estimate a longitudinal causal effect. Spectral excess is max(0, P_obs − P_null_mean)."]
    elif gate and longitudinal is not None:
        lines += ["", "Paired-window Q changes and annualized rates are conditional summaries, not longitudinal causal effects.", "", _md_table(longitudinal)]
    elif not gate:
        lines += ["", "Longitudinal and spectral comparisons are withheld pending reconciliation of participant IDs, source/protocol/window keys, endpoint/null versions, age bins, denominators, and formulas. This gate is required because the baseline report may duplicate protocols in its longitudinal calculations."]
    if gate and spectra is not None and not spectra.empty:
        spec=spectra.copy(); spec["excess"]=np.maximum(0.,spec.P_obs-spec.P_null_mean)
        ages=spec.age.to_numpy(float)
        spec=spec[(spec.sex==1)&(ages>=42)&(ages<48)]
        targets=[20.,26.67,45.]
        spec["target_period"]=spec.period_days.map(lambda p: min(targets,key=lambda t:abs(t-p)))
        spec=spec[np.abs(spec.period_days-spec.target_period)<0.26]
        lines += ["", "Matched female 42 to <48 year spectra; excess is max(0, P_obs − P_null_mean).", "", _md_table(spec.groupby(["source","protocol","arm","window_index","target_period"]).excess.agg(["count","mean","median"]).reset_index())]
    lines += ["", "## Interpretation limits", "", "HRp1 is more sensitive to a small number of low observed bins; this analysis cannot establish that it is less accurate. Channel-3000 coverage covariates describe that channel's observation geometry only. Similar curves do not validate supplier algorithms, a menopause mechanism, or the assumptions used to translate a curve feature into onset age.", ""]
    path=out/"REPORT.md"; path.write_text("\n".join(lines),encoding="utf-8")
    (out/"report_provenance.json").write_text(json.dumps({"traceability_gate_passed":gate,"traceability_reason":reason,"figures":figs},indent=2),encoding="utf-8")
    return path
