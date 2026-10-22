"""Paired RHR-arm inference for the fixed-date sensitivity analysis.

Input is a long endpoint table with one row per user/source/arm/protocol/window.
The analysis is deliberately downstream of endpoint construction: it never
selects dates or changes convergence status.
"""
from __future__ import annotations

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SCRIPT_DIR = EXP / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from common import DEFAULT_KNOTS, rcs_stone_koo

ARMS = ("HRp10", "HRp5", "HRp1")
ENDPOINTS = ("Q_i", "E_i2", "A_hat", "sigma_resid")
TESTS = {
    "total_female_age": ([1, 2, 3], [5, 6, 7], 3),
    "nonlinear_female_curvature": ([2, 3], [6, 7], 2),
    "total_sex_x_age_interaction": ([], [5, 6, 7], 3),
    "nonlinear_sex_x_age_interaction": ([], [6, 7], 2),
}


def design(age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    r1, r2 = rcs_stone_koo(age, DEFAULT_KNOTS)
    return np.column_stack((np.ones(len(age)), age, r1, r2, sex, sex*age, sex*r1, sex*r2))


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """OLS and HC3 covariance, matching the existing unadjusted RCS model."""
    if len(y) <= x.shape[1] or np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("rank deficient or insufficient common sample")
    inv = np.linalg.inv(x.T @ x)
    beta = inv @ x.T @ y
    h = np.sum(x * (x @ inv), axis=1)
    if np.any(1.0 - h < 1e-10):
        raise ValueError("HC3 leverage singularity")
    resid = (y - x @ beta) / (1.0 - h)
    cov = inv @ ((x * resid[:, None]).T @ (x * resid[:, None])) @ inv
    return beta, cov, len(y)-x.shape[1]


def _wald(beta: np.ndarray, cov: np.ndarray, name: str, df_resid: int) -> dict:
    c = _contrast(name)
    df=c.shape[0]
    v = c @ cov @ c.T
    w = float((c @ beta).T @ np.linalg.pinv(v) @ (c @ beta))
    w=max(0.,w); f=w/df
    return {"test": name, "df": df, "wald_hc3": w,"f_hc3":f,
            "p_chi2": float(stats.chi2.sf(w, df)),"p_f":float(stats.f.sf(f,df, df_resid))}


def _contrast(name: str) -> np.ndarray:
    left,right,df=TESTS[name]; c=np.zeros((df,8))
    for j in range(df):
        if left: c[j,left[j]]=1.
        c[j,right[j]]=1.
    return c


def _curves(beta: np.ndarray, grid: np.ndarray) -> dict[str, np.ndarray]:
    r1, r2 = rcs_stone_koo(grid, DEFAULT_KNOTS)
    z = np.column_stack((np.ones(len(grid)), grid, r1, r2))
    return {"male": z @ beta[:4], "female": z @ (beta[:4] + beta[4:]),
            "delta": z @ beta[4:]}


def _strata(df: pd.DataFrame) -> list[np.ndarray]:
    band = np.floor(df.age.to_numpy(float) / 5).astype(int)
    sex = df.sex.to_numpy(int)
    key = band * 2 + sex
    return [np.flatnonzero(key == k) for k in np.unique(key)]


def _transition(beta: np.ndarray) -> tuple[float, bool, float | None, bool]:
    grid = np.linspace(35, 65, 601)
    r1, r2 = rcs_stone_koo(grid, DEFAULT_KNOTS)
    d1 = 3*np.maximum(grid-25, 0)**2/1600 - (40/13)*3*np.maximum(grid-52, 0)**2/1600 + (27/13)*3*np.maximum(grid-65, 0)**2/1600
    d2 = 3*np.maximum(grid-42, 0)**2/1600 - (23/13)*3*np.maximum(grid-52, 0)**2/1600 + (10/13)*3*np.maximum(grid-65, 0)**2/1600
    deriv = beta[1] + beta[2]*d1 + beta[3]*d2
    mn = np.flatnonzero(np.isclose(deriv, deriv.min(), rtol=1e-8, atol=1e-10))
    age = float(grid[mn[0]])
    boundary = bool(age <= 35.05 or age >= 64.95)
    z = np.linspace(40, 75, 701)
    a, b = rcs_stone_koo(z, DEFAULT_KNOTS)
    delta = beta[0] + beta[1]*z + beta[2]*a + beta[3]*b
    hit = np.flatnonzero(delta <= 0)
    # A value already below zero at the lower search boundary is not a
    # crossing within the interval. Require a positive-to-nonpositive bracket.
    root = None
    if len(hit) and hit[0] > 0 and np.any(delta[:hit[0]] > 0):
        k=hit[0]; root=float(z[k-1] - delta[k-1]*(z[k]-z[k-1])/(delta[k]-delta[k-1]))
    multiple = bool(np.sum(np.isclose(deriv, deriv.min(), atol=max(1e-9, abs(deriv.min())*1e-5)) & (np.abs(grid-grid[mn[0]]) > .1)) > 0)
    return age, boundary, root, multiple


def run_inference(endpoints: pd.DataFrame, output_dir: str | Path, *, n_boot: int = 2000,
                  seed: int = 20261020, arms: tuple[str, ...] = ARMS) -> dict[str, pd.DataFrame]:
    """Run common-convergence matched arm fits and participant-cluster bootstrap.

    Each source/protocol/window comparison is restricted to users with finite
    endpoints and converged nulls in every requested arm. The same stratified
    resampled user indices are applied to each arm. Seed excludes arm identity.
    """
    required = {"user_id", "source", "arm", "protocol", "window_index", "age", "sex", "null_status", *ENDPOINTS}
    missing = required - set(endpoints.columns)
    if missing:
        raise ValueError(f"endpoint table missing columns: {sorted(missing)}")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    curves, paired, tests, transitions, counts = [], [], [], [], []
    boot_inflections={}
    keys = ["source", "protocol", "window_index"]
    for key, group in endpoints[endpoints.arm.isin(arms)].groupby(keys, dropna=False):
        source, protocol, window = key
        wide = group.pivot(index="user_id", columns="arm", values=list(ENDPOINTS))
        # All-arm complete cases, preserving a single common participant set.
        ids = wide.index
        ok = np.ones(len(ids), dtype=bool)
        for arm in arms:
            if arm not in wide.columns.get_level_values(1):
                ok[:] = False; continue
            sub = group[group.arm == arm].set_index("user_id").reindex(ids)
            ok &= sub.null_status.eq("converged").to_numpy()
            for ep in ENDPOINTS:
                ok &= np.isfinite(sub[ep].to_numpy(float))
        eligible = group[group.arm == arms[0]].set_index("user_id").reindex(ids)
        data = eligible.loc[ok].copy()
        diag={"source":source,"protocol":protocol,"window_index":window}
        for arm in arms:
            ag=group[group.arm==arm]
            diag[f"n_{arm}_input"]=len(ag)
            diag[f"n_{arm}_converged"]=int(ag.null_status.eq("converged").sum())
            diag[f"n_{arm}_failed"]=int((~ag.null_status.eq("converged")).sum())
            if "Q_i_mc_se" in ag:
                diag[f"mean_{arm}_Q_mc_se_all_converged"]=float(ag.loc[ag.null_status.eq("converged"),"Q_i_mc_se"].mean())
        data=data[(data.age>=25.)&(data.age<75.)]
        if data.empty:
            counts.append({**diag,"n_common":0,"status":"no_common_converged_sample"}); continue
        # Enforce age/sex metadata consistency across arms.
        for arm in arms[1:]:
            m = group[group.arm == arm].set_index("user_id").reindex(data.index)
            if not (np.allclose(data.age, m.age, equal_nan=True) and np.array_equal(data.sex, m.sex)):
                raise ValueError(f"age/sex mismatch across arms for {key}, arm={arm}")
        for col in ("selected_dates_hash", "n_selected_dates"):
            if col in group and (col != "n_selected_dates" or group[col].notna().all()):
                aligned=[]
                for arm in arms:
                    vals=group[group.arm==arm].set_index("user_id").reindex(data.index)[col]
                    aligned.append(vals.to_numpy())
                if any(not np.array_equal(aligned[0],v) for v in aligned[1:]):
                    raise ValueError(f"{col} differs across matched arms for {key}")
        n = len(data); ages=data.age.to_numpy(float); sex=data.sex.to_numpy(float)
        age_band=np.floor(ages/5).astype(int)
        support_counts=pd.DataFrame({"band":age_band,"sex":sex.astype(int)}).groupby(["band","sex"]).size()
        good=[b for b in range(5,15) if all(support_counts.get((b,sx),0)>=30 for sx in (0,1))]
        runs=[]
        for b in good:
            if not runs or b != runs[-1][-1]+1: runs.append([b])
            else: runs[-1].append(b)
        support=max(runs,key=len) if runs else []
        if not support:
            counts.append({**diag,"n_common":n,"status":"insufficient_shared_age_support","n_by_sex_age_json":json.dumps({f"{int(b)}:{int(s)}":int(v) for (b,s),v in support_counts.items()},sort_keys=True)})
            continue
        lo_age,hi_age=5.*support[0],5.*(support[-1]+1)
        keep=(ages>=lo_age)&(ages<hi_age); data=data.loc[keep].copy(); ages=data.age.to_numpy(float); sex=data.sex.to_numpy(float); n=len(data)
        x=design(ages, sex); grid=np.arange(lo_age, hi_age, .25)
        if np.linalg.matrix_rank(x) != 8:
            counts.append({**diag,"n_common":n,"support_lower_age":lo_age,"support_upper_age_exclusive":hi_age,"status":"rank_deficient_common_support"})
            continue
        transition_supported=(lo_age<=35.0 and hi_age>=65.0)
        group_key = f"{source}|{protocol}|{window}|{','.join(arms)}"
        seed_group = int(seed + sum(group_key.encode("utf-8")) * 1009)
        rng=np.random.Generator(np.random.PCG64(seed_group)); strata=_strata(data)
        # Draw once; indices identify the common participant rows for every arm.
        draws=np.empty((n_boot,n),dtype=np.int32)
        for b in range(n_boot):
            draws[b]=np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in strata])
        fits={}
        for arm in arms:
            sub=group[group.arm==arm].set_index("user_id").reindex(data.index)
            fit_rows={}
            for ep in ENDPOINTS:
                y=sub[ep].to_numpy(float); beta,cov,dfres=_fit(x,y); fit_rows[ep]=(beta,cov,dfres)
                pred=_curves(beta,grid)
                boot={k:np.empty((n_boot,len(grid)),dtype=np.float32) for k in ("female","male","delta")}
                transitions_boot=[]; bootstrap_wald={name:[] for name in TESTS}
                for b,idx in enumerate(draws):
                    xb=x[idx]
                    if np.linalg.matrix_rank(xb) != 8:
                        for c in boot: boot[c][b]=np.nan
                        if ep == "Q_i": transitions_boot.append((np.nan,np.nan,np.nan))
                        continue
                    bb=np.linalg.lstsq(xb,y[idx],rcond=None)[0]
                    for c,v in _curves(bb,grid).items(): boot[c][b]=v
                    inv_b=np.linalg.inv(xb.T@xb); h_b=np.sum(xb*(xb@inv_b),axis=1)
                    if np.any(1.-h_b<1e-10): continue
                    residual=(y[idx]-xb@bb)/(1.-h_b)
                    cov_b=inv_b@((xb*residual[:,None]).T@(xb*residual[:,None]))@inv_b
                    for name in TESTS:
                        cmat=_contrast(name); d=cmat@(bb-beta)
                        bootstrap_wald[name].append(float(max(0.,d.T@np.linalg.pinv(cmat@cov_b@cmat.T)@d)))
                    if ep == "Q_i": transitions_boot.append(_transition(bb[4:]) if transition_supported else (np.nan,False,None,False))
                for c,arr in boot.items():
                    valid=np.all(np.isfinite(arr),axis=1); arr_ok=arr[valid]
                    if len(arr_ok)<2: raise ValueError(f"fewer than two valid bootstrap fits for {key}/{arm}/{ep}/{c}")
                    se=arr_ok.std(axis=0,ddof=1)
                    lo,hi=np.percentile(arr_ok,[2.5,97.5],axis=0)
                    sup=np.percentile(np.max(np.abs(arr_ok-pred[c])/np.maximum(se,1e-12),axis=1),95)
                    for j,a in enumerate(grid):
                        curves.append(dict(source=source,protocol=protocol,window_index=window,arm=arm,endpoint=ep,curve=c,age=a,estimate=pred[c][j],lower_pointwise=lo[j],upper_pointwise=hi[j],lower_simultaneous=pred[c][j]-sup*se[j],upper_simultaneous=pred[c][j]+sup*se[j],n=n,n_boot=n_boot,n_boot_valid=int(valid.sum())))
                for name in TESTS:
                    observed=_wald(beta,cov,name,dfres); dist=np.asarray(bootstrap_wald[name])
                    observed["p_bootstrap_centered"]=(1.+np.sum(dist>=observed["wald_hc3"]-1e-12))/(len(dist)+1.) if len(dist) else np.nan
                    observed["n_boot_valid"]=len(dist)
                    tests.append(dict(source=source,protocol=protocol,window_index=window,arm=arm,endpoint=ep,n=n,df_resid=dfres,**observed))
                if ep=="Q_i":
                    inf,bound,root,multiple=_transition(beta[4:]) if transition_supported else (np.nan,False,None,False)
                    tb=np.array([[v[0], np.nan if v[2] is None else v[2]] for v in transitions_boot],float)
                    infarr=tb[np.isfinite(tb[:,0]),0]; rootarr=tb[np.isfinite(tb[:,1]),1]
                    boot_inflections[(source,protocol,window,arm)]=tb[:,0]
                    transitions.append(dict(source=source,protocol=protocol,window_index=window,arm=arm,inflection_age=inf,inflection_ci_low=np.percentile(infarr,2.5) if len(infarr) else np.nan,inflection_ci_high=np.percentile(infarr,97.5) if len(infarr) else np.nan,inflection_boundary=bound,inflection_multiple_minima=multiple,transition_status="estimated" if transition_supported else "support_does_not_cover_35_65",bootstrap_boundary_fraction=np.mean((infarr<=35.05)|(infarr>=64.95)) if len(infarr) else np.nan,bootstrap_identifiable_fraction=len(infarr)/n_boot,bootstrap_fit_failures=n_boot-len(infarr),zero_crossing_age=root,zero_crossing_status=("support_does_not_cover_35_65" if not transition_supported else ("not_reached_in_window" if root is None else "crossed_from_positive")),zero_crossing_ci_low=np.percentile(rootarr,2.5) if len(rootarr) else np.nan,zero_crossing_ci_high=np.percentile(rootarr,97.5) if len(rootarr) else np.nan,zero_crossing_unidentified=n_boot-len(rootarr),n_boot=n_boot))
                fits[(arm,ep)]=(pred,boot)

        # Paired arm minus reference curves from the identical resamples.
        ref=arms[0]
        for arm in arms[1:]:
            for ep in ENDPOINTS:
                for c in ("female","delta"):
                    p0,b0=fits[(ref,ep)]; p1,b1=fits[(arm,ep)]
                    diff=p1[c]-p0[c]; bd=b1[c]-b0[c]
                    valid=np.all(np.isfinite(bd),axis=1); bd_ok=bd[valid]
                    if len(bd_ok)<2: raise ValueError(f"fewer than two paired bootstrap fits for {key}/{arm}/{ep}/{c}")
                    se=bd_ok.std(axis=0,ddof=1)
                    lo,hi=np.percentile(bd_ok,[2.5,97.5],axis=0)
                    sup=np.percentile(np.max(np.abs(bd_ok-diff)/np.maximum(se,1e-12),axis=1),95)
                    for j,a in enumerate(grid): paired.append(dict(source=source,protocol=protocol,window_index=window,arm=arm,reference=ref,endpoint=ep,curve=c,age=a,difference=diff[j],lower_pointwise=lo[j],upper_pointwise=hi[j],lower_simultaneous=diff[j]-sup*se[j],upper_simultaneous=diff[j]+sup*se[j],n=n,n_boot=n_boot,n_boot_valid=int(valid.sum())))
        cell_counts=data.assign(age_band=np.floor(data.age/5)*5).groupby(["age_band","sex"]).size()
        cells={f"{int(b)}:{int(s)}":int(v) for (b,s),v in cell_counts.items()}
        counts.append({**diag,"n_common":n,"support_lower_age":lo_age,"support_upper_age_exclusive":hi_age,"n_by_sex_age_json":json.dumps(cells,sort_keys=True),"n_boot":n_boot,"bootstrap_seed":seed_group,"status":"ok"})
    # Paired inflection-age differences retain the shared bootstrap draws.
    paired_trans=[]
    for (source,protocol,window),g in endpoints[endpoints.arm.isin(arms)].groupby(keys,dropna=False):
        ref=arms[0]
        for arm in arms[1:]:
            x=boot_inflections.get((source,protocol,window,ref)); y=boot_inflections.get((source,protocol,window,arm))
            if x is None or y is None: continue
            valid=np.isfinite(x)&np.isfinite(y); dif=y[valid]-x[valid]
            point={r["arm"]:r["inflection_age"] for r in transitions if r["source"]==source and r["protocol"]==protocol and r["window_index"]==window}
            paired_trans.append(dict(source=source,protocol=protocol,window_index=window,arm=arm,reference=ref,measure="inflection_age",difference=point.get(arm,np.nan)-point.get(ref,np.nan),ci_low=np.percentile(dif,2.5) if len(dif) else np.nan,ci_high=np.percentile(dif,97.5) if len(dif) else np.nan,n_boot_valid=len(dif),n_boot=n_boot))
    tables={"curves":pd.DataFrame(curves),"paired_curve_differences":pd.DataFrame(paired),"chunk_tests":pd.DataFrame(tests),"transitions":pd.DataFrame(transitions),"paired_transitions":pd.DataFrame(paired_trans),"cohort_counts":pd.DataFrame(counts)}
    for name,df in tables.items(): df.to_parquet(out/f"{name}.parquet",index=False)
    return tables


def longitudinal_summaries(endpoints: pd.DataFrame, *, n_boot: int = 2000, seed: int = 20261020,
                           arms: tuple[str, ...] = ARMS) -> pd.DataFrame:
    """Paired Q change/rate by protocol and source, clustered by participant.

    Requires two converged windows per user and arm. No causal interpretation.
    """
    need={"user_id","source","arm","protocol","window_index","start_day","age","sex","Q_i","null_status"}
    if need-set(endpoints): raise ValueError(f"missing longitudinal columns: {sorted(need-set(endpoints))}")
    rows=[]
    for (source,protocol),g0 in endpoints.groupby(["source","protocol"]):
        group_arms=tuple(a for a in arms if a in set(g0.arm))
        if len(group_arms)<2: continue
        vals={}; ages=None; sexes=None; common=None
        for arm in group_arms:
            g=g0[(g0.arm==arm)&g0.null_status.eq("converged")&g0.Q_i.notna()]
            q=g.pivot(index="user_id",columns="window_index",values="Q_i")
            starts=g.pivot(index="user_id",columns="window_index",values="start_day")
            if not {1,2}.issubset(q.columns) or not {1,2}.issubset(starts.columns):
                common=pd.Index([]); vals[arm]=(q,starts); continue
            ok=q.dropna(subset=[1,2]).index.intersection(starts.dropna(subset=[1,2]).index)
            common=ok if common is None else common.intersection(ok)
            meta=g[g.window_index==1].set_index("user_id")
            if ages is None:
                ages=meta.age; sexes=meta.sex
            vals[arm]=(q,starts)
        if common is None or len(common)==0: continue
        qref,sref=vals[group_arms[0]]
        for arm in group_arms[1:]:
            q,s=vals[arm]
            for w in (1,2):
                if not np.array_equal(sref.loc[common,w].to_numpy(),s.loc[common,w].to_numpy()):
                    raise ValueError(f"window start mismatch across arms: source={source}, protocol={protocol}, arm={arm}")
        gap=(sref.loc[common,2]-sref.loc[common,1]).to_numpy(float)/365.25
        per_arm={a:{"Q_change":(q.loc[common,2]-q.loc[common,1]).to_numpy(float),"Q_annualized_rate":(q.loc[common,2]-q.loc[common,1]).to_numpy(float)/gap} for a,(q,_) in vals.items()}
        # Stratified participant clusters by first-window five-year age band x sex.
        meta=g0[(g0.arm==group_arms[0])&(g0.window_index==1)].set_index("user_id").loc[common]
        strata=_strata(meta)
        rng=np.random.Generator(np.random.PCG64(seed+int(source)*10007+len(str(protocol))))
        draws=np.empty((n_boot,len(common)),dtype=int)
        for b in range(n_boot): draws[b]=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in strata])
        for a in group_arms:
            for label,refvals in per_arm[a].items():
                boot=refvals[draws].mean(axis=1)
                rows.append(dict(source=source,arm=a,reference=arms[0],contrast="arm",protocol=protocol,measure=label,n=len(common),mean=float(refvals.mean()),ci_low=float(np.percentile(boot,2.5)),ci_high=float(np.percentile(boot,97.5)),n_boot=n_boot,seed=seed))
        ref=group_arms[0]
        for a in group_arms[1:]:
            for label in ("Q_change","Q_annualized_rate"):
                diff=per_arm[a][label]-per_arm[ref][label]; boot=diff[draws].mean(axis=1)
                rows.append(dict(source=source,arm=a,reference=ref,contrast="paired_arm_minus_reference",protocol=protocol,measure=label,n=len(common),mean=float(diff.mean()),ci_low=float(np.percentile(boot,2.5)),ci_high=float(np.percentile(boot,97.5)),n_boot=n_boot,seed=seed))
    return pd.DataFrame(rows)
