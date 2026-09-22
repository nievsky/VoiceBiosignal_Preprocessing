"""
Automated ECG R-Peak Detection and Heart Rate Estimation.
Provides standard R-peak extraction based on Z-score normalization and peak prominence,
along with instantaneous and mean heart rate calculations.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import scipy.signal as signal
import pandas as pd


def detect_r_peaks(
    ecg_signal: np.ndarray,
    sampling_rate: float,
    min_rr_sec: float = 0.30,
    height_scale: float = 0.0,
    prominence: float = 0.5
) -> np.ndarray:
    """
    Detect R-peaks in an ECG signal using adaptive amplitude & prominence thresholding.

    Parameters:
        ecg_signal: 1D array containing ECG voltage amplitudes.
        sampling_rate: Sampling frequency in Hertz (Hz).
        min_rr_sec: Minimum refractory period between consecutive beats (default 0.3s -> max 200 bpm).
        height_scale: Factor scaling standard deviation for height threshold.
        prominence: Minimum peak prominence in standardized units.

    Returns:
        np.ndarray: Indices of detected R-peak sample positions.
    """
    x = np.asarray(ecg_signal, dtype=float)
    fs = float(sampling_rate)
    if x.size == 0 or fs <= 0:
        return np.array([], dtype=int)

    # Standardize signal (Z-score normalization)
    std_val = np.std(x)
    if std_val < 1e-12:
        return np.array([], dtype=int)
    xz = (x - np.mean(x)) / std_val

    # Determine minimum refractory sample distance and amplitude height
    distance = max(1, int(min_rr_sec * fs))
    height = np.mean(xz) + height_scale * np.std(xz)

    peaks, _ = signal.find_peaks(
        xz,
        height=height,
        distance=distance,
        prominence=prominence
    )

    return peaks.astype(int)


def compute_heart_rate(
    r_peaks_idx: np.ndarray,
    sampling_rate: float,
    min_bpm: float = 30.0,
    max_bpm: float = 210.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute instantaneous and mean heart rate (BPM) from detected R-peaks.

    Parameters:
        r_peaks_idx: Indices of detected R-peaks.
        sampling_rate: Sampling frequency in Hertz (Hz).
        min_bpm: Lower physiological plausibility threshold.
        max_bpm: Upper physiological plausibility threshold.

    Returns:
        Tuple:
            - hr_bpm: Array of instantaneous heart rates in BPM.
            - hr_times_s: Midpoint timestamps in seconds for each RR interval.
            - mean_bpm: Average heart rate over valid intervals.
    """
    r_peaks_idx = np.asarray(r_peaks_idx, dtype=int)
    fs = float(sampling_rate)

    if len(r_peaks_idx) < 2 or fs <= 0:
        return np.array([]), np.array([]), float("nan")

    r_times = r_peaks_idx / fs
    rr = np.diff(r_times)  # Interval duration in seconds

    # Calculate valid physiological RR boundaries
    min_rr = 60.0 / max_bpm
    max_rr = 60.0 / min_bpm
    valid = (rr >= min_rr) & (rr <= max_rr)

    rr_valid = rr[valid]
    times_valid = ((r_times[1:] + r_times[:-1]) / 2.0)[valid]

    if len(rr_valid) == 0:
        return np.array([]), np.array([]), float("nan")

    hr_bpm = 60.0 / rr_valid
    mean_bpm = float(np.mean(hr_bpm))

    return hr_bpm, times_valid, mean_bpm


def process_ecg(
    ecg_signal: np.ndarray,
    sampling_rate: float,
    min_rr_sec: float = 0.30,
    prominence: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    High-level convenience pipeline to detect peaks and evaluate heart rate.

    Returns:
        r_peaks_idx, hr_times_s, hr_bpm, mean_bpm
    """
    r_peaks = detect_r_peaks(
        ecg_signal,
        sampling_rate=sampling_rate,
        min_rr_sec=min_rr_sec,
        prominence=prominence
    )
    hr_bpm, hr_times_s, mean_bpm = compute_heart_rate(r_peaks, sampling_rate)
    return r_peaks, hr_times_s, hr_bpm, mean_bpm


def evaluate_ecg_record(
    ecg_signal: np.ndarray,
    annotated_peaks: np.ndarray,
    sampling_rate: float
) -> Dict[str, float]:
    """
    Evaluate algorithm heart rate detection against physician reference annotations.
    """
    r_peaks, _, _, detected_mean_bpm = process_ecg(ecg_signal, sampling_rate)
    total_duration_min = (len(ecg_signal) / sampling_rate) / 60.0

    actual_hr = len(annotated_peaks) / total_duration_min if total_duration_min > 0 else 0.0
    detected_hr = len(r_peaks) / total_duration_min if total_duration_min > 0 else 0.0

    error_rate = abs(actual_hr - detected_hr) / actual_hr if actual_hr > 0 else 0.0
    accuracy = max(0.0, 100.0 * (1.0 - error_rate))

    return {
        "actual_hr_bpm": actual_hr,
        "detected_hr_bpm": detected_hr,
        "accuracy_pct": accuracy,
        "n_actual_peaks": len(annotated_peaks),
        "n_detected_peaks": len(r_peaks),
        "mean_instantaneous_bpm": detected_mean_bpm
    }
