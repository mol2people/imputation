#!/usr/bin/env python
"""Selective classification analysis for the temporal arms.

Predeclared procedure (NEXT_STEPS.md Step 0.5, timestamped by commit
168b9d7 before any coverage/precision numbers were computed):

- Per alloc r in {0..9}, per arm {Combined (primary), Summary_RF,
  BASE_x_P40, Random}: thresholds selected on val (n=579), evaluated
  on test (n=573).
- Per assigned class, targets 95% and 98%.
  - Class-1: smallest tau1 s.t. empirical P(y=1 | score >= tau1) >= target.
  - Class-0: largest  tau0 s.t. empirical P(y=0 | score <= tau0) >= target.
  - All ties included on both sides (no within-tie cherry-picking).
- Two threshold variants:
  (a) raw empirical (smallest/largest tau meeting target).
  (b) finite-sample-corrected via one-sided Clopper-Pearson lower
      confidence bound on the calibration selected-set precision:
      L = beta.ppf(delta, s, f+1), delta = 0.10 (s = correct, f = wrong
      among calibration-selected); require L >= target.
      RECORDED DEVIATION FROM PREDECLARATION: NEXT_STEPS Step 0.5 named
      "conformal risk control (CRC)" (Angelopoulos et al. 2022). CRC
      proper controls expectations of bounded per-instance losses and
      does not directly apply to precision (a ratio of expectations).
      The CP-LCB is the exact, distribution-free rate guarantee in the
      same spirit: P(true selected-population precision >= L) >= 1-delta
      over the calibration draw; test participants selected at tau are
      exchangeable with calibration, so expected test precision inherits
      the bound (sampling noise on the test side reported via Wilson
      CIs, not guaranteed away).
- Evaluation on test: per-class precision (Wilson 95% CI), per-class
  coverage (fraction of cohort labeled with that class), recall
  (true positives / class size), abstention fraction.
- BASE_x_P40: R1b saved preds are test-only (r1b_predictions_part.csv,
  no split col) -> 50/50 half-sample cross-fit on test per alloc,
  both directions averaged.
- Random baseline: seeded uniform scores per (alloc, user), same
  protocol using real val/test users from Combined.
- Aggregates: cross-alloc mean +- SD (alloc = repeat unit); alloc-0 val
  touched by alpha selection (flagged, not excluded).
- Risk-coverage curves: one row per unique val threshold per (arm,
  alloc, side) for Combined, Summary_RF, Random (BASE_x_P40 excluded
  — test-only preds make half-sample curves artifacts). Includes
  test-side evaluation at each tau.
- Abstention composition (Combined, raw and CP-LCB at 95% target):
  mean mask coverage, observed-bins-per-day, and d40 d_ch3000_* stats
  for labeled vs abstained test participants.

Seeds: SeedSequence([20260922, component]) with components:
  RANDOM=5 (Random arm scores), HALF_SPLIT=6 (BASE_x_P40 splits).
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------- paths -----------------------------------------------------------
ROOT = Path(__file__).parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
PREDS = RESULTS / "temporal_predictions.csv"
R1B_PREDS = Path("/Users/bulat/Documents/datenspende_IMPUTE_sex/"
                  "experiments/r1b_dateaware_garmin_2026-09-21/results/"
                  "r1b_predictions_part.csv")
D40_PARQUET = Path("/Users/bulat/Documents/datenspende_IMPUTE_sex/"
                    "experiments/r1b_dateaware_garmin_2026-09-21/cache/"
                    "day40_table.parquet")
BINS_NPZ = CACHE / "day_bins.npz"

ARMS_PRIMARY = ("Combined", "Summary_RF")
ARMS_CURVE = ("Combined", "Summary_RF", "Random")
TARGETS = (0.95, 0.98)
DELTA = 0.10
Z95 = 1.959963984540054
COMP_RANDOM = 5
COMP_HALFSPLIT = 6
ROOT_SEED = 20260922


def seed_for(*component: int) -> np.random.Generator:
    """Seeded generator via SeedSequence([ROOT_SEED, *component])."""
    return np.random.default_rng(
        np.random.SeedSequence([ROOT_SEED, *component]).spawn(1)[0])


# ---------- statistical helpers --------------------------------------------
def cp_lcb(s: int, f: int, delta: float = DELTA) -> float:
    """One-sided Clopper-Pearson lower bound on precision with s correct
    and f wrong among n=s+f selected, confidence level 1-delta.
    P(true precision >= L) >= 1-delta."""
    if s <= 0:
        return 0.0
    return float(stats.beta.ppf(delta, s, f + 1))


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ---------- threshold selection --------------------------------------------
def _select_high(scores: np.ndarray, y: np.ndarray, target: float,
                  variant: str, assigned: int = 1):
    """Class-1: smallest tau s.t. precision(y=assigned | score >= tau)
    meets the variant's criterion. Returns (tau, n_sel, prec) or
    (nan, 0, nan) if unattainable. All ties included (overwrite)."""
    n = len(scores)
    if n == 0:
        return (float("nan"), 0, float("nan"))
    pos = (y == assigned).astype(np.int64)
    # Sort DESCENDING so ties are last -> overwrite gives full tie count
    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    p_sorted = pos[order]
    cum_correct = np.cumsum(p_sorted)
    # Build per-tau (n_sel, correct) using the LAST occurrence (max inclusive set)
    seen = {}
    for k in range(n):
        seen[float(s_sorted[k])] = (k + 1, int(cum_correct[k]))
    # Iterate tau ASCENDING -> first satisfying = smallest satisfying tau
    for tau in sorted(seen.keys()):
        k_sel, correct = seen[tau]
        if k_sel == 0:
            continue
        f_sel = k_sel - correct
        prec = correct / k_sel
        if variant == "raw":
            if prec >= target:
                return (float(tau), int(k_sel), float(prec))
        else:
            if cp_lcb(correct, f_sel) >= target:
                return (float(tau), int(k_sel), float(prec))
    return (float("nan"), 0, float("nan"))


def _select_low(scores: np.ndarray, y: np.ndarray, target: float,
                 variant: str, assigned: int = 0):
    """Class-0: largest tau s.t. precision(y=assigned | score <= tau)
    meets the criterion. All ties included."""
    n = len(scores)
    if n == 0:
        return (float("nan"), 0, float("nan"))
    pos = (y == assigned).astype(np.int64)
    # Sort ASCENDING; ties last -> overwrite gives full tie count for that tau
    order = np.argsort(scores, kind="stable")
    s_sorted = scores[order]
    p_sorted = pos[order]
    cum_correct = np.cumsum(p_sorted)
    seen = {}
    for k in range(n):
        seen[float(s_sorted[k])] = (k + 1, int(cum_correct[k]))
    # Iterate tau DESCENDING -> first satisfying = largest satisfying tau
    for tau in sorted(seen.keys(), reverse=True):
        k_sel, correct = seen[tau]
        if k_sel == 0:
            continue
        f_sel = k_sel - correct
        prec = correct / k_sel
        if variant == "raw":
            if prec >= target:
                return (float(tau), int(k_sel), float(prec))
        else:
            if cp_lcb(correct, f_sel) >= target:
                return (float(tau), int(k_sel), float(prec))
    return (float("nan"), 0, float("nan"))


# ---------- evaluation on test ---------------------------------------------
def _eval_high(tau: float, scores_te: np.ndarray, y_te: np.ndarray,
                assigned: int = 1):
    if np.isnan(tau):
        sel = np.zeros(len(scores_te), dtype=bool)
    else:
        sel = scores_te >= tau
    n_sel = int(sel.sum())
    if n_sel == 0:
        return (0, 0, 0.0, (float("nan"), float("nan")))
    correct = int(((y_te == assigned) & sel).sum())
    prec = correct / n_sel
    return (n_sel, correct, prec, wilson_ci(correct, n_sel))


def _eval_low(tau: float, scores_te: np.ndarray, y_te: np.ndarray,
               assigned: int = 0):
    if np.isnan(tau):
        sel = np.zeros(len(scores_te), dtype=bool)
    else:
        sel = scores_te <= tau
    n_sel = int(sel.sum())
    if n_sel == 0:
        return (0, 0, 0.0, (float("nan"), float("nan")))
    correct = int(((y_te == assigned) & sel).sum())
    prec = correct / n_sel
    return (n_sel, correct, prec, wilson_ci(correct, n_sel))


# ---------- arm-level analyses ---------------------------------------------
def analyze_arm(preds: pd.DataFrame, arm: str, targets=TARGETS):
    rows = []
    for r in range(10):
        sub_v = preds[(preds.arm == arm) & (preds.alloc == r)
                       & (preds.split == "val")].sort_values("user")
        sub_t = preds[(preds.arm == arm) & (preds.alloc == r)
                       & (preds.split == "test")].sort_values("user")
        if len(sub_v) == 0 or len(sub_t) == 0:
            continue
        s_v, y_v = sub_v.score.to_numpy(), sub_v.y.to_numpy()
        s_t, y_t = sub_t.score.to_numpy(), sub_t.y.to_numpy()
        n_test = len(s_t)
        n_class = {1: int((y_t == 1).sum()), 0: int((y_t == 0).sum())}
        for tgt in targets:
            for variant in ("raw", "cpc"):
                tau1, nsv1, prv1 = _select_high(
                    s_v, y_v, tgt, variant, assigned=1)
                ns1, c1, p1, (lo1, hi1) = _eval_high(
                    tau1, s_t, y_t, assigned=1)
                cov1 = ns1 / n_test
                rec1 = c1 / n_class[1] if n_class[1] else float("nan")
                rows.append({
                    "arm": arm, "alloc": r, "target": tgt,
                    "variant": variant, "class": 1, "tau": tau1,
                    "n_sel_val": nsv1, "prec_val": prv1,
                    "n_sel_test": ns1, "prec_test": p1,
                    "wilson_lo": lo1, "wilson_hi": hi1,
                    "cov_test": cov1, "recall_test": rec1,
                })
                tau0, nsv0, prv0 = _select_low(
                    s_v, y_v, tgt, variant, assigned=0)
                ns0, c0, p0, (lo0, hi0) = _eval_low(
                    tau0, s_t, y_t, assigned=0)
                cov0 = ns0 / n_test
                rec0 = c0 / n_class[0] if n_class[0] else float("nan")
                rows.append({
                    "arm": arm, "alloc": r, "target": tgt,
                    "variant": variant, "class": 0, "tau": tau0,
                    "n_sel_val": nsv0, "prec_val": prv0,
                    "n_sel_test": ns0, "prec_test": p0,
                    "wilson_lo": lo0, "wilson_hi": hi0,
                    "cov_test": cov0, "recall_test": rec0,
                })
    return rows


def analyze_r1b_crossfit(r1b: pd.DataFrame, targets=TARGETS):
    rows = []
    for r in range(10):
        sub = r1b[(r1b.arm == "BASE_x_P40") & (r1b.alloc == r)]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("user").reset_index(drop=True)
        s_all = sub.p_class1.to_numpy()
        y_all = sub.y.to_numpy()
        rng = seed_for(COMP_HALFSPLIT, r)
        perm = rng.permutation(len(sub))
        h = len(sub) // 2
        # Single loop, both cross-fit directions; no double counting
        for cal_idx, te_idx, dir_tag in (
            (perm[:h], perm[h:], "A"),
            (perm[h:], perm[:h], "B"),
        ):
            s_v, y_v = s_all[cal_idx], y_all[cal_idx]
            s_t, y_t = s_all[te_idx], y_all[te_idx]
            n_te = len(s_t)
            n_class = {1: int((y_t == 1).sum()),
                        0: int((y_t == 0).sum())}
            for tgt in targets:
                for variant in ("raw", "cpc"):
                    tau1, nsv, prv = _select_high(
                        s_v, y_v, tgt, variant, assigned=1)
                    ns1, c1, p1, (lo1, hi1) = _eval_high(
                        tau1, s_t, y_t, assigned=1)
                    cov1 = ns1 / n_te
                    rec1 = c1 / n_class[1] if n_class[1] else float("nan")
                    rows.append({
                        "arm": "BASE_x_P40", "alloc": r,
                        "target": tgt, "variant": variant, "class": 1,
                        "tau": tau1, "n_sel_val": nsv, "prec_val": prv,
                        "n_sel_test": ns1, "prec_test": p1,
                        "wilson_lo": lo1, "wilson_hi": hi1,
                        "cov_test": cov1, "recall_test": rec1,
                        "xfit_dir": dir_tag,
                    })
                    tau0, nsv, prv = _select_low(
                        s_v, y_v, tgt, variant, assigned=0)
                    ns0, c0, p0, (lo0, hi0) = _eval_low(
                        tau0, s_t, y_t, assigned=0)
                    cov0 = ns0 / n_te
                    rec0 = c0 / n_class[0] if n_class[0] else float("nan")
                    rows.append({
                        "arm": "BASE_x_P40", "alloc": r,
                        "target": tgt, "variant": variant, "class": 0,
                        "tau": tau0, "n_sel_val": nsv, "prec_val": prv,
                        "n_sel_test": ns0, "prec_test": p0,
                        "wilson_lo": lo0, "wilson_hi": hi0,
                        "cov_test": cov0, "recall_test": rec0,
                        "xfit_dir": dir_tag,
                    })
    return rows


def analyze_random(preds: pd.DataFrame, targets=TARGETS):
    """Random: seeded uniform scores using REAL val/test users per alloc."""
    rows = []
    for r in range(10):
        sub_v = preds[(preds.alloc == r) & (preds.split == "val")
                       & (preds.arm == "Combined")].sort_values("user")
        sub_t = preds[(preds.alloc == r) & (preds.split == "test")
                       & (preds.arm == "Combined")].sort_values("user")
        if len(sub_v) == 0 or len(sub_t) == 0:
            continue
        users_v = sub_v.user.to_numpy(); y_v = sub_v.y.to_numpy()
        users_t = sub_t.user.to_numpy(); y_t = sub_t.y.to_numpy()
        # one score per user (consistent across val/test)
        all_users = np.concatenate([users_v, users_t])
        rng = seed_for(COMP_RANDOM, r)
        scores_all = rng.random(len(all_users))
        smap = {int(u): float(x) for u, x in zip(all_users, scores_all)}
        s_v = np.array([smap[int(u)] for u in users_v])
        s_t = np.array([smap[int(u)] for u in users_t])
        n_te = len(s_t)
        n_class = {1: int((y_t == 1).sum()), 0: int((y_t == 0).sum())}
        for tgt in targets:
            for variant in ("raw", "cpc"):
                tau1, nsv, prv = _select_high(
                    s_v, y_v, tgt, variant, assigned=1)
                ns1, c1, p1, (lo1, hi1) = _eval_high(
                    tau1, s_t, y_t, assigned=1)
                cov1 = ns1 / n_te
                rec1 = c1 / n_class[1] if n_class[1] else float("nan")
                rows.append({
                    "arm": "Random", "alloc": r, "target": tgt,
                    "variant": variant, "class": 1,
                    "tau": tau1, "n_sel_val": nsv, "prec_val": prv,
                    "n_sel_test": ns1, "prec_test": p1,
                    "wilson_lo": lo1, "wilson_hi": hi1,
                    "cov_test": cov1, "recall_test": rec1,
                })
                tau0, nsv, prv = _select_low(
                    s_v, y_v, tgt, variant, assigned=0)
                ns0, c0, p0, (lo0, hi0) = _eval_low(
                    tau0, s_t, y_t, assigned=0)
                cov0 = ns0 / n_te
                rec0 = c0 / n_class[0] if n_class[0] else float("nan")
                rows.append({
                    "arm": "Random", "alloc": r, "target": tgt,
                    "variant": variant, "class": 0,
                    "tau": tau0, "n_sel_val": nsv, "prec_val": prv,
                    "n_sel_test": ns0, "prec_test": p0,
                    "wilson_lo": lo0, "wilson_hi": hi0,
                    "cov_test": cov0, "recall_test": rec0,
                })
    return rows


# ---------- risk-coverage curves -------------------------------------------
def collect_curves(preds: pd.DataFrame, arm: str, r: int):
    """Yield curve rows (arm, alloc, class, tau, n_sel_val, prec_val,
    n_sel_test, prec_test, cov_test) for one (arm, alloc). Class-side
    determined per row."""
    sub_v = preds[(preds.arm == arm) & (preds.alloc == r)
                   & (preds.split == "val")].sort_values("user")
    sub_t = preds[(preds.arm == arm) & (preds.alloc == r)
                   & (preds.split == "test")].sort_values("user")
    if len(sub_v) == 0 or len(sub_t) == 0:
        return []
    s_v, y_v = sub_v.score.to_numpy(), sub_v.y.to_numpy()
    s_t, y_t = sub_t.score.to_numpy(), sub_t.y.to_numpy()
    n_te = len(s_t)
    out = []
    # HIGH side (class 1), tau descending scan
    order = np.argsort(-s_v, kind="stable")
    s_s, p_s = s_v[order], (y_v[order] == 1).astype(np.int64)
    cum = np.cumsum(p_s)
    seen = {}
    for k in range(len(s_v)):
        seen[float(s_s[k])] = (k + 1, int(cum[k]))
    for tau in sorted(seen.keys(), reverse=True):
        k_sel, correct = seen[tau]
        # test eval at this tau
        sel_t = s_t >= tau
        ns = int(sel_t.sum())
        c = int(((y_t == 1) & sel_t).sum())
        out.append({
            "arm": arm, "alloc": r, "class": 1, "tau": tau,
            "n_sel_val": k_sel, "prec_val": correct / k_sel if k_sel else float("nan"),
            "n_sel_test": ns, "prec_test": c / ns if ns else float("nan"),
            "cov_test": ns / n_te,
        })
    # LOW side (class 0), tau ascending scan
    order = np.argsort(s_v, kind="stable")
    s_s, p_s = s_v[order], (y_v[order] == 0).astype(np.int64)
    cum = np.cumsum(p_s)
    seen = {}
    for k in range(len(s_v)):
        seen[float(s_s[k])] = (k + 1, int(cum[k]))
    for tau in sorted(seen.keys()):
        k_sel, correct = seen[tau]
        sel_t = s_t <= tau
        ns = int(sel_t.sum())
        c = int(((y_t == 0) & sel_t).sum())
        out.append({
            "arm": arm, "alloc": r, "class": 0, "tau": tau,
            "n_sel_val": k_sel, "prec_val": correct / k_sel if k_sel else float("nan"),
            "n_sel_test": ns, "prec_test": c / ns if ns else float("nan"),
            "cov_test": ns / n_te,
        })
    return out


def collect_random_curves(preds: pd.DataFrame, r: int):
    sub_v = preds[(preds.alloc == r) & (preds.split == "val")
                   & (preds.arm == "Combined")].sort_values("user")
    sub_t = preds[(preds.alloc == r) & (preds.split == "test")
                   & (preds.arm == "Combined")].sort_values("user")
    if len(sub_v) == 0 or len(sub_t) == 0:
        return []
    users_v = sub_v.user.to_numpy(); y_v = sub_v.y.to_numpy()
    users_t = sub_t.user.to_numpy(); y_t = sub_t.y.to_numpy()
    all_users = np.concatenate([users_v, users_t])
    rng = seed_for(COMP_RANDOM, r)
    smap = {int(u): float(x) for u, x in zip(all_users, rng.random(len(all_users)))}
    s_v = np.array([smap[int(u)] for u in users_v])
    s_t = np.array([smap[int(u)] for u in users_t])
    out = []
    n_te = len(s_t)
    # HIGH
    order = np.argsort(-s_v, kind="stable")
    s_s = s_v[order]; p_s = (y_v[order] == 1).astype(np.int64)
    cum = np.cumsum(p_s)
    seen = {}
    for k in range(len(s_v)):
        seen[float(s_s[k])] = (k + 1, int(cum[k]))
    for tau in sorted(seen.keys(), reverse=True):
        k_sel, correct = seen[tau]
        sel_t = s_t >= tau
        ns = int(sel_t.sum())
        c = int(((y_t == 1) & sel_t).sum())
        out.append({
            "arm": "Random", "alloc": r, "class": 1, "tau": tau,
            "n_sel_val": k_sel, "prec_val": correct / k_sel if k_sel else float("nan"),
            "n_sel_test": ns, "prec_test": c / ns if ns else float("nan"),
            "cov_test": ns / n_te,
        })
    # LOW
    order = np.argsort(s_v, kind="stable")
    s_s = s_v[order]; p_s = (y_v[order] == 0).astype(np.int64)
    cum = np.cumsum(p_s)
    seen = {}
    for k in range(len(s_v)):
        seen[float(s_s[k])] = (k + 1, int(cum[k]))
    for tau in sorted(seen.keys()):
        k_sel, correct = seen[tau]
        sel_t = s_t <= tau
        ns = int(sel_t.sum())
        c = int(((y_t == 0) & sel_t).sum())
        out.append({
            "arm": "Random", "alloc": r, "class": 0, "tau": tau,
            "n_sel_val": k_sel, "prec_val": correct / k_sel if k_sel else float("nan"),
            "n_sel_test": ns, "prec_test": c / ns if ns else float("nan"),
            "cov_test": ns / n_te,
        })
    return out


# ---------- abstention composition ----------------------------------------
def abstention_composition(preds: pd.DataFrame, targets=(0.95,),
                             variants=("raw", "cpc")):
    z = np.load(BINS_NPZ)
    mask = z["mask"]                              # (n_users*DAYS, 288)
    user_per_row = z["user"]
    DAYS = 40; N_BINS = 288
    n_users = user_per_row.size // DAYS
    # mask rows: user-major (users sorted; each user's 40 days consecutive)
    # verify & build user -> index map using unique users in order
    unique_users, first_idx = np.unique(user_per_row, return_index=True)
    assert len(unique_users) == n_users and first_idx[0] == 0
    user_to_idx = {int(u): i for i, u in enumerate(unique_users)}
    mask_pu = mask.reshape(n_users, DAYS * N_BINS)
    cov_per_user = mask_pu.mean(axis=1).astype(np.float32)      # wear coverage
    obs_per_userday = mask.reshape(n_users, DAYS, N_BINS).sum(
        axis=2).mean(axis=1).astype(np.float32)                  # obs bins/day

    d40 = pd.read_parquet(D40_PARQUET)
    d40_cols = [c for c in d40.columns if c.startswith("d_ch3000_")]
    d40u = d40.groupby("user", sort=True)[d40_cols].mean()

    rows = []
    for r in range(10):
        sub_v = preds[(preds.arm == "Combined") & (preds.alloc == r)
                       & (preds.split == "val")].sort_values("user")
        sub_t = preds[(preds.arm == "Combined") & (preds.alloc == r)
                       & (preds.split == "test")].sort_values("user")
        if len(sub_v) == 0 or len(sub_t) == 0:
            continue
        s_v, y_v = sub_v.score.to_numpy(), sub_v.y.to_numpy()
        s_t, y_t = sub_t.score.to_numpy(), sub_t.y.to_numpy()
        users_t = sub_t.user.to_numpy()
        idx_t = np.array([user_to_idx.get(int(u), -1) for u in users_t])
        ok = idx_t >= 0
        for tgt in targets:
            for variant in variants:
                tau1, _, _ = _select_high(s_v, y_v, tgt, variant, assigned=1)
                tau0, _, _ = _select_low(s_v, y_v, tgt, variant, assigned=0)
                lab = np.zeros(len(s_t), dtype=bool)
                if not np.isnan(tau1):
                    lab |= (s_t >= tau1)
                if not np.isnan(tau0):
                    lab |= (s_t <= tau0)
                abst = ~lab
                for label, mask_subset in (
                    ("labeled", lab & ok),
                    ("abstained", abst & ok),
                ):
                    nsub = int(mask_subset.sum())
                    if nsub == 0:
                        continue
                    cov = float(cov_per_user[idx_t[mask_subset]].mean())
                    obsd = float(obs_per_userday[idx_t[mask_subset]].mean())
                    sub_d = d40u.reindex(
                        users_t[mask_subset].astype("int64"))
                    rows.append({
                        "alloc": r, "target": tgt, "variant": variant,
                        "group": label, "n": nsub,
                        "mean_mask_coverage": cov,
                        "mean_obs_bins_per_day": obsd,
                        "mean_d_ch3000_mean": float(
                            sub_d["d_ch3000_mean"].mean()),
                        "mean_d_ch3000_cov_h": float(
                            sub_d["d_ch3000_cov_h"].mean()),
                    })
    return rows


# ---------- markdown summary ------------------------------------------------
def write_md(rows: list[dict], md_path: Path):
    df = pd.DataFrame(rows)
    L = []
    L.append("### Coverage at per-class target precision "
              "(mean ± SD across 10 allocs; alloc = repeat unit)\n")
    L.append("Variant codes: `raw` = empirical threshold; "
              "`cpc` = Clopper-Pearson LCB-corrected "
              f"(δ={DELTA}, confidence {1-DELTA:.2f}). "
              "`BASE_x_P40` = R1b half-sample cross-fit (test-only preds; "
              "n_test per direction ≈ 286). `Random` = seeded uniform "
              "scores using real val/test users. Coverage reported as "
              "fraction of cohort labeled with that class; recall = "
              "true positives / class size. Wilson 95% CI on precision.\n")
    for arm in ("Combined", "Summary_RF", "BASE_x_P40", "Random"):
        sub = df[df.arm == arm]
        if len(sub) == 0:
            continue
        L.append(f"\n#### `{arm}`\n")
        L.append("| target | variant | cov₁ (%) | recall₁ (%) | prec₁ "
                  "(mean [Wilson]) | cov₀ (%) | recall₀ (%) | "
                  "prec₀ (mean [Wilson]) | cov_total (%) | abst (%) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for tgt in TARGETS:
            for variant in ("raw", "cpc"):
                cells = {}
                for cls in (1, 0):
                    s = sub[(sub.target == tgt) & (sub.variant == variant)
                             & (sub["class"] == cls)]
                    if len(s) == 0:
                        cells[(cls, "prec")] = "—"
                        cells[(cls, "cov")] = "—"
                        cells[(cls, "rec")] = "—"
                        continue
                    pp = s[["prec_test", "wilson_lo", "wilson_hi"]].dropna(
                        subset=["prec_test"])
                    if len(pp) == 0:
                        cells[(cls, "prec")] = "—"
                        cells[(cls, "cov")] = "—"
                        cells[(cls, "rec")] = "—"
                        continue
                    cells[(cls, "prec")] = (
                        f"{pp.prec_test.mean():.3f} "
                        f"({pp.wilson_lo.mean():.3f}–{pp.wilson_hi.mean():.3f})")
                    c = s.cov_test
                    r = s.recall_test.dropna()
                    cells[(cls, "cov")] = (
                        f"{c.mean() * 100:.1f} ± {c.std(ddof=1) * 100:.1f}"
                        if len(c) > 1 else f"{c.mean() * 100:.1f}")
                    cells[(cls, "rec")] = (
                        f"{r.mean() * 100:.1f} ± {r.std(ddof=1) * 100:.1f}"
                        if len(r) > 1 else f"{r.mean() * 100:.1f}")
                s1 = sub[(sub.target == tgt) & (sub.variant == variant)
                          & (sub["class"] == 1)]
                s0 = sub[(sub.target == tgt) & (sub.variant == variant)
                          & (sub["class"] == 0)]
                if len(s1) and len(s0):
                    # pair by alloc (+ xfit_dir if present)
                    if "xfit_dir" in s1.columns:
                        m1 = s1.groupby(["alloc", "xfit_dir"]).cov_test.mean()
                        m0 = s0.groupby(["alloc", "xfit_dir"]).cov_test.mean()
                    else:
                        m1 = s1.groupby("alloc").cov_test.mean()
                        m0 = s0.groupby("alloc").cov_test.mean()
                    joined = m1.add(m0, fill_value=0.0)
                    tot = joined.values * 100
                    abst_v = 100 - tot
                    tot_str = (f"{tot.mean():.1f} ± {tot.std(ddof=1):.1f}"
                                if len(tot) > 1 else f"{tot.mean():.1f}")
                    abst_str = (f"{abst_v.mean():.1f} ± {abst_v.std(ddof=1):.1f}"
                                 if len(abst_v) > 1 else f"{abst_v.mean():.1f}")
                else:
                    tot_str = "—"; abst_str = "—"
                L.append(
                    f"| {tgt*100:.0f}% | {variant} | "
                    f"{cells.get((1,'cov'), '—')} | "
                    f"{cells.get((1,'rec'), '—')} | "
                    f"{cells.get((1,'prec'), '—')} | "
                    f"{cells.get((0,'cov'), '—')} | "
                    f"{cells.get((0,'rec'), '—')} | "
                    f"{cells.get((0,'prec'), '—')} | "
                    f"{tot_str} | {abst_str} |")

    if "Combined" in df.arm.unique() and "BASE_x_P40" in df.arm.unique():
        L.append("\n#### Combined − BASE⊕P40 (cpc variant; applied delta)\n")
        L.append("| target | metric | Combined | BASE⊕P40 | Δ |")
        L.append("|---|---|---|---|---|")
        for tgt in TARGETS:
            for cls in (1, 0):
                c = df[(df.arm == "Combined") & (df.target == tgt)
                        & (df.variant == "cpc") & (df["class"] == cls)]
                b = df[(df.arm == "BASE_x_P40") & (df.target == tgt)
                        & (df.variant == "cpc") & (df["class"] == cls)]
                if len(c) == 0 or len(b) == 0:
                    continue
                cc = c.cov_test; bb = b.cov_test
                L.append(f"| {tgt*100:.0f}% | cov{cls} | "
                          f"{cc.mean()*100:.1f}% ± {cc.std(ddof=1)*100:.1f}% | "
                          f"{bb.mean()*100:.1f}% ± {bb.std(ddof=1)*100:.1f}% | "
                          f"{(cc.mean()-bb.mean())*100:+.1f} pp |")
    open(md_path, "w").write("\n".join(L) + "\n")


# ---------- main -----------------------------------------------------------
def main():
    t0 = time.perf_counter()
    print(f"[selective] start {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    if not PREDS.exists():
        raise SystemExit(f"missing {PREDS}")
    preds = pd.read_csv(PREDS)
    rows = []
    for arm in ARMS_PRIMARY:
        print(f"[selective]   arm={arm}")
        rows.extend(analyze_arm(preds, arm))
    if R1B_PREDS.exists():
        r1b = pd.read_csv(R1B_PREDS)
        print(f"[selective]   arm=BASE_x_P40 (cross-fit on {len(r1b)} rows)")
        rows.extend(analyze_r1b_crossfit(r1b))
    else:
        print(f"[selective]   skipping BASE_x_P40 ({R1B_PREDS} not found)")
    rows.extend(analyze_random(preds))
    print(f"[selective]   arm=Random")

    df = pd.DataFrame(rows)
    out_csv = RESULTS / "selective_classification.csv"
    df.to_csv(out_csv, index=False)
    print(f"[selective] wrote {out_csv} ({len(df)} rows)")

    md_path = RESULTS / "selective_classification.md"
    write_md(rows, md_path)
    print(f"[selective] wrote {md_path}")

    # Curves (Combined + Summary_RF + Random)
    print(f"[selective] curves (Combined, Summary_RF, Random)")
    curve_rows = []
    for arm in ("Combined", "Summary_RF"):
        for r in range(10):
            curve_rows.extend(collect_curves(preds, arm, r))
    for r in range(10):
        curve_rows.extend(collect_random_curves(preds, r))
    curve_df = pd.DataFrame(curve_rows)
    curve_csv = RESULTS / "risk_coverage_curves.csv"
    curve_df.to_csv(curve_csv, index=False)
    print(f"[selective] wrote {curve_csv} ({len(curve_df)} rows)")

    # Abstention composition (Combined, 95%, raw + cpc)
    print(f"[selective] abstention composition")
    abst = abstention_composition(preds, targets=(0.95,),
                                    variants=("raw", "cpc"))
    abst_df = pd.DataFrame(abst)
    abst_path = RESULTS / "abstention_composition.csv"
    abst_df.to_csv(abst_path, index=False)
    print(f"[selective] wrote {abst_path} ({len(abst_df)} rows)")

    print(f"[selective] done in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
