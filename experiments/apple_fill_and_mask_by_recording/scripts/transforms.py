"""Core sequence transforms and preprocessing utilities for Apple Watch analysis.

Provides vectorized, leak-free implementations of:
  - compute_clean_diff: Piecewise clean difference withholding gap transitions (Review R6.1)
  - interpolate_short_gaps: Span-matched short-gap linear interpolation (Review R4)
  - compute_mask_stats: Binary observation mask mean and SD via algebraic identity (Review R3)
  - fit_fill_templates: Train-only clock and global median template estimators
"""
from __future__ import annotations

import numpy as np


def compute_clean_diff(
    hr_or_clock: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """Compute piecewise clean difference Delta^{clean} X avoiding NaN * 0 = NaN.

    Definition (Review R6.1, PLAN §4.4):
        Delta^{clean} X_{u,d,t} = hr_{t+1} - hr_{t}  if M_t == 1 and M_{t+1} == 1
                                  0.0                otherwise.

    Args:
        hr_or_clock: Heart rate array (raw with NaNs or clock-filled), shape (..., L).
        mask: Binary observation mask (1=observed, 0=unobserved), shape (..., L).

    Returns:
        clean_diff: Array of shape (..., L-1), float32, with zero NaNs.
    """
    m_bool = (mask == 1)
    mask_both = m_bool[..., :-1] & m_bool[..., 1:]
    with np.errstate(invalid="ignore"):
        diff = hr_or_clock[..., 1:] - hr_or_clock[..., :-1]
        clean = np.where(mask_both, diff, 0.0)
    return clean.astype(np.float32)


def interpolate_short_gaps(
    hr_raw: np.ndarray,
    mask: np.ndarray | None = None,
    clock_template: np.ndarray | None = None,
    resolution: str = "5m",
    k_max: int | None = None,
) -> np.ndarray:
    """Span-matched short-gap linear interpolation (Review R4, PLAN §4.4).

    Interpolates interior missing runs of length 1 <= k <= k_max bounded by
    observed bins (M=1) on the same day with physical separation (k+1)*dt <= 30 min.
      - 5m: k_max = 5 (10..30 min span)
      - 15m: k_max = 1 (30 min span)

    Ineligible gaps (k > k_max), boundary gaps (touching t=0 or t=L-1), and all-missing
    days remain on clock_template. Observed bins (M=1) are preserved bit-exactly.

    Args:
        hr_raw: Heart rate array with NaNs, shape (D, L) or (L,).
        mask: Optional binary mask (1=observed, 0=unobserved). Inferred from hr_raw if None.
        clock_template: Optional fallback template of shape (L,). Computed from hr_raw if None.
        resolution: Grid resolution, "5m" or "15m".
        k_max: Optional explicit integer override for maximum gap run length.

    Returns:
        X_interp: Array of shape (D, L) or (L,), float32.
    """
    if k_max is None:
        if resolution == "5m":
            k_max = 5
        elif resolution == "15m":
            k_max = 1
        else:
            raise ValueError(f"Unknown resolution '{resolution}'; expected '5m' or '15m'")

    orig_1d = (hr_raw.ndim == 1)
    if orig_1d:
        hr_raw = hr_raw[None, :]
        if mask is not None:
            mask = mask[None, :]

    D, L = hr_raw.shape
    if mask is None:
        mask = np.isfinite(hr_raw).astype(np.uint8)
    else:
        mask = (mask == 1).astype(np.uint8)

    if clock_template is None:
        with np.errstate(invalid="ignore"):
            clock_template = np.nanmedian(np.where(mask == 1, hr_raw, np.nan), axis=0).astype(np.float32)
        if np.isnan(clock_template).any():
            med_glob = float(np.nanmedian(hr_raw[mask == 1])) if np.any(mask == 1) else 60.0
            clock_template = np.where(np.isnan(clock_template), med_glob, clock_template)

    # Initialize output: observed positions keep raw values, missing positions take clock_template
    X = np.where(mask == 1, hr_raw, clock_template[None, :]).astype(np.float32)

    if k_max <= 0:
        return X[0] if orig_1d else X

    # Identify contiguous missing runs along time axis (mask == 0)
    p = np.zeros((D, L + 2), dtype=bool)
    p[:, 1:-1] = (mask == 0)
    diff = np.diff(p.astype(np.int8), axis=1)  # shape (D, L + 1)

    starts = np.flatnonzero(diff == 1)
    ends_neg1 = np.flatnonzero(diff == -1)

    if starts.size == 0:
        return X[0] if orig_1d else X

    rows = starts // (L + 1)
    s = (starts % (L + 1)).astype(np.int32)
    e = ((ends_neg1 % (L + 1)) - 1).astype(np.int32)
    k = e - s + 1

    # Eligible interior gap: strictly within day (s >= 1 and e <= L - 2) and 1 <= k <= k_max
    eligible = (s >= 1) & (e <= L - 2) & (k >= 1) & (k <= k_max)
    ei = np.flatnonzero(eligible)

    if ei.size == 0:
        return X[0] if orig_1d else X

    el_rows = rows[ei]
    el_s = s[ei]
    el_k = k[ei]

    t_L = el_s - 1
    t_R = el_s + el_k

    h_L = hr_raw[el_rows, t_L].astype(np.float64)
    h_R = hr_raw[el_rows, t_R].astype(np.float64)

    # Vectorized scatter per gap length k (at most k_max iterations)
    for cur_k in range(1, k_max + 1):
        match_k = (el_k == cur_k)
        if not np.any(match_k):
            continue
        r_k = el_rows[match_k]
        s_k = el_s[match_k]
        hl_k = h_L[match_k]
        hr_k = h_R[match_k]
        slope = (hr_k - hl_k) / (cur_k + 1.0)
        for j in range(1, cur_k + 1):
            X[r_k, s_k + (j - 1)] = (hl_k + j * slope).astype(np.float32)

    return X[0] if orig_1d else X


def compute_mask_stats(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bin across-day mean and sample SD of binary mask (Review R3, PLAN §4.1).

    Algebraic Identity for binary M in {0, 1} and D_u > 1:
        s_{M,u,t}^2 = [D_u / (D_u - 1)] * mean_M * (1 - mean_M)
    When D_u <= 1:
        s_{M,u,t} = 0.0

    Args:
        mask: Observation mask of shape (D_u, L) or (L,).

    Returns:
        mean_M: Shape (L,), float32.
        sd_M: Shape (L,), float32.
    """
    orig_1d = (mask.ndim == 1)
    if orig_1d:
        mask = mask[None, :]

    D, L = mask.shape
    m_f = (mask == 1).astype(np.float64)
    mean_M = np.mean(m_f, axis=0)

    if D > 1:
        var_M = (D / (D - 1.0)) * mean_M * (1.0 - mean_M)
        sd_M = np.sqrt(np.maximum(0.0, var_M))
    else:
        sd_M = np.zeros(L, dtype=np.float64)

    return mean_M.astype(np.float32), sd_M.astype(np.float32)


def fit_fill_templates(
    train_hr: list[np.ndarray] | np.ndarray,
    mask: list[np.ndarray] | np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Fit population clock-bin and scalar global medians strictly from observed training data.

    Matches bit-for-bit with reference fit_train_fill in apple_cohort_scaling.

    Args:
        train_hr: List of participant HR arrays or single concatenated 2D array.
        mask: Optional matching binary mask(s). If provided, unobserved bins are set to NaN.

    Returns:
        clock_template: Per-clock-bin median array of shape (L,), float32.
        global_template: Scalar overall median, float.
    """
    if isinstance(train_hr, list):
        if mask is not None and isinstance(mask, list):
            masked_list = [np.where(m == 1, h, np.nan) for h, m in zip(train_hr, mask)]
            train_all = np.concatenate(masked_list, axis=0)
        else:
            train_all = np.concatenate(train_hr, axis=0)
    else:
        if mask is not None:
            train_all = np.where(mask == 1, train_hr, np.nan)
        else:
            train_all = train_hr

    with np.errstate(invalid="ignore"):
        clock = np.nanmedian(train_all, axis=0).astype(np.float32)
        valid = train_all[np.isfinite(train_all)]
        global_med = float(np.nanmedian(valid)) if valid.size > 0 else 60.0

    if np.isnan(clock).any():
        clock = np.where(np.isnan(clock), global_med, clock)

    return clock, global_med
