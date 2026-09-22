"""
Multimodal Hemodynamic Correlation and Cross-Correlation Analysis.
Implements signal preprocessing (temporal windowing, NaN interpolation, detrending, centralization)
and cross-correlation estimation for physiological monitoring (ECG, ABP, ICP).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.signal import detrend, resample, correlate
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import pandas as pd


def preprocess_signal(
    signal_data: np.ndarray,
    fs: float,
    max_duration_sec: float = 3600.0,
    new_fs: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    Standardize a physiological waveform by clipping to a fixed window,
    interpolating missing values, removing polynomial trends, and centering.

    Parameters:
        signal_data: 1D array of signal samples.
        fs: Input sampling frequency in Hz.
        max_duration_sec: Temporal cut-off duration (default 1 hour = 3600 s).
        new_fs: Target sampling frequency for resampling (optional).

    Returns:
        Tuple: (preprocessed_signal, effective_sampling_rate)
    """
    sig = np.asarray(signal_data, dtype=float)
    fs = float(fs)

    # Limit to designated observation duration (e.g., first 1 hour)
    max_samples = int(fs * max_duration_sec)
    sig = sig[:max_samples]

    # Impute missing values (NaNs) via linear interpolation
    nan_mask = np.isnan(sig)
    if np.any(nan_mask):
        indices = np.arange(len(sig))
        valid_idx = indices[~nan_mask]
        valid_vals = sig[~nan_mask]
        if len(valid_idx) > 1:
            interpolator = interp1d(
                valid_idx, valid_vals, kind="linear", bounds_error=False, fill_value="extrapolate"
            )
            sig = interpolator(indices)
        else:
            sig = np.nan_to_num(sig, nan=0.0)

    # Detrend linear and slow drifts
    sig = detrend(sig)

    # Zero-mean centralization
    sig = sig - np.mean(sig)

    # Optional resampling
    if new_fs is not None and new_fs > 0 and new_fs != fs:
        target_len = int(len(sig) * (new_fs / fs))
        sig = resample(sig, target_len)
        fs = new_fs

    return sig, fs


def compute_pairwise_correlations(
    ecg: np.ndarray,
    abp: np.ndarray,
    icp: np.ndarray
) -> pd.DataFrame:
    """
    Compute zero-lag instantaneous Pearson correlation coefficient across signal channels.
    """
    min_len = min(len(ecg), len(abp), len(icp))
    e, a, i = ecg[:min_len], abp[:min_len], icp[:min_len]

    r_ea, p_ea = pearsonr(e, a)
    r_ei, p_ei = pearsonr(e, i)
    r_ai, p_ai = pearsonr(a, i)

    return pd.DataFrame({
        "Signal Pair": ["ECG-ABP", "ECG-ICP", "ABP-ICP"],
        "Pearson r": [r_ea, r_ei, r_ai],
        "p-value": [p_ea, p_ei, p_ai]
    })


def lagged_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    max_lag_s: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute normalized lagged cross-correlation between two signals x and y.

    Parameters:
        x: First signal array.
        y: Second signal array.
        fs: Sampling rate in Hz.
        max_lag_s: Search interval bound in seconds (±max_lag_s).

    Returns:
        Tuple:
            - lags_sec: Array of lag offsets in seconds within [-max_lag_s, +max_lag_s].
            - corr_vals: Correlation coefficient values for each lag.
            - lag_max: Lag (in seconds) corresponding to maximum correlation.
            - corr_max: Maximum correlation coefficient.
    """
    min_len = min(len(x), len(y))
    x_seg = x[:min_len]
    y_seg = y[:min_len]

    # Z-score normalization for accurate cross-correlation coefficient calculation
    std_x = np.std(x_seg)
    std_y = np.std(y_seg)
    if std_x < 1e-12 or std_y < 1e-12:
        return np.array([0.0]), np.array([0.0]), 0.0, 0.0

    x_norm = (x_seg - np.mean(x_seg)) / std_x
    y_norm = (y_seg - np.mean(y_seg)) / std_y

    raw_corr = correlate(y_norm, x_norm, mode="full")
    all_lags = np.arange(-len(x_norm) + 1, len(y_norm))
    lags_sec = all_lags / fs

    # Window around ±max_lag_s
    mask = (lags_sec >= -max_lag_s) & (lags_sec <= max_lag_s)
    corr_windowed = raw_corr[mask] / len(x_norm)
    lags_windowed = lags_sec[mask]

    best_idx = int(np.argmax(corr_windowed))
    lag_max = float(lags_windowed[best_idx])
    corr_max = float(corr_windowed[best_idx])

    return lags_windowed, corr_windowed, lag_max, corr_max
