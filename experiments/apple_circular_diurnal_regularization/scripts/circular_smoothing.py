"""Modular circular 1D Gaussian smoothing for 288-bin diurnal profiles.

Provides leak-free, mathematically verified circular smoothing across midnight
with periodic wrap-around boundary conditions, mean/energy conservation, and
normalized convolution for missing/NaN bins.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def circular_gaussian_smooth(
    x: np.ndarray,
    sigma: float = 1.0,
    axis: int = -1,
    truncate: float = 4.0,
    nan_policy: str = "normalize",
) -> np.ndarray:
    """Apply 1D Gaussian smoothing with periodic circular wrap-around boundary conditions.

    In a 288-bin diurnal profile (5-min resolution), bin 287 (23:55-24:00) is
    immediately adjacent to bin 0 (00:00-00:05). Periodic wrap-around ensures zero
    boundary flattening, reflection distortion, or truncation across midnight.

    For complete profiles, the operation corresponds to convolution with a normalized
    doubly-stochastic circulant Gaussian kernel, preserving total sum and mean to
    floating-point precision (error <= 1e-10 in float64).

    When NaN values are present, normalized circular convolution is applied:
        smoothed = smooth(x * mask) / smooth(mask)
    which smoothly interpolates across missing runs while avoiding NaN propagation.

    Args:
        x: Input array of shape (..., L), typically L=288. Float32 or Float64.
        sigma: Standard deviation of Gaussian smoothing kernel in bins.
            sigma=1 bin = 5 minutes
            sigma=2 bins = 10 minutes
            sigma=4 bins = 20 minutes
            sigma=0 returns a copy of x unmodified.
        axis: Dimension along which to apply smoothing (default: -1, diurnal bins).
        truncate: Truncate filter at this many standard deviations (default: 4.0).
        nan_policy: Strategy for handling NaNs:
            "normalize": Normalized convolution (interpolates missing bins).
            "omit": Return NaNs where input has NaNs.

    Returns:
        smoothed: Array of same shape and dtype as input x.
    """
    if sigma <= 0.0:
        return x.copy()

    orig_dtype = x.dtype
    # Use float64 internally for numerical stability and exact conservation
    arr = np.asarray(x, dtype=np.float64)

    has_nans = np.isnan(arr).any()

    if not has_nans:
        # Standard circular Gaussian filtering with periodic wrap-around
        smoothed = gaussian_filter1d(
            arr,
            sigma=float(sigma),
            axis=axis,
            mode="wrap",
            truncate=float(truncate),
        )
    else:
        # Normalized convolution for handling missing/NaN bins
        mask = np.isfinite(arr).astype(np.float64)
        if not mask.any():
            # All elements are NaN
            return np.full_like(arr, np.nan, dtype=orig_dtype)

        filled = np.where(mask > 0, arr, 0.0)

        s_val = gaussian_filter1d(
            filled,
            sigma=float(sigma),
            axis=axis,
            mode="wrap",
            truncate=float(truncate),
        )
        s_mask = gaussian_filter1d(
            mask,
            sigma=float(sigma),
            axis=axis,
            mode="wrap",
            truncate=float(truncate),
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = np.where(s_mask > 1e-12, s_val / s_mask, np.nan)

        if nan_policy == "omit":
            smoothed = np.where(mask > 0, smoothed, np.nan)

    return smoothed.astype(orig_dtype)


def discrete_circular_gaussian_kernel(
    sigma: float,
    n_bins: int = 288,
) -> np.ndarray:
    """Generate exact discrete circular Gaussian kernel on Z / n_bins Z.

    Computes normalized periodic Gaussian weights:
        w_k = sum_{m in Z} exp(- (k + m * n_bins)^2 / (2 * sigma^2))
    summing to exactly 1.0.

    Args:
        sigma: Kernel bandwidth standard deviation in bins.
        n_bins: Number of bins in circular period (default: 288).

    Returns:
        kernel: 1D array of shape (n_bins,), symmetric around index 0, sum == 1.0.
    """
    if sigma <= 0.0:
        k = np.zeros(n_bins, dtype=np.float64)
        k[0] = 1.0
        return k

    # Account for periodic wraps (typically m in {-1, 0, 1} is more than sufficient for sigma <= 30)
    idx = np.arange(n_bins, dtype=np.float64)
    kernel = np.zeros(n_bins, dtype=np.float64)

    m_max = int(np.ceil(4.0 * sigma / n_bins)) + 1
    for m in range(-m_max, m_max + 1):
        dist = idx + m * n_bins
        kernel += np.exp(-0.5 * (dist / sigma) ** 2)

    kernel /= np.sum(kernel)
    return kernel


def smooth_diurnal_template(
    template: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Convenience wrapper for smoothing 288-bin diurnal median templates.

    Args:
        template: 1D (288,) or 2D (N, 288) diurnal profile array.
        sigma: Standard deviation in bins (1, 2, or 4).

    Returns:
        smoothed_template: Array with identical shape and dtype.
    """
    return circular_gaussian_smooth(template, sigma=sigma, axis=-1)
