"""
Acoustic feature extraction for voice pathology detection.
Computes time-domain perturbation parameters (Jitter, Shimmer) and
frequency/quefrency domain Cepstral Peak Prominence (CPP).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.signal import find_peaks
import scipy.fft as fft


def calculate_acoustic_features(
    signal_data: np.ndarray,
    fs: float = 8000.0,
    trim_middle: bool = True
) -> Tuple[float, float]:
    """
    Calculate Jitter (%) and Shimmer (%) from glottal pitch periods and pulse amplitudes.

    Parameters:
        signal_data: 1D array of phonation samples (e.g., sustained vowel /a/).
        fs: Sampling rate in Hz (default: 8000 Hz).
        trim_middle: If True, uses the middle 50% of the recording to avoid vocal onset/offset transients.

    Returns:
        Tuple: (jitter_percent, shimmer_percent)
    """
    sig = np.asarray(signal_data, dtype=float)
    fs = float(fs)

    if len(sig) < int(fs * 0.2):
        return float("nan"), float("nan")

    if trim_middle:
        start = int(len(sig) * 0.25)
        end = int(len(sig) * 0.75)
        trimmed = sig[start:end]
    else:
        trimmed = sig

    # Remove DC bias
    trimmed = trimmed - np.mean(trimmed)

    # Peak detection for glottal pulses (assume pitch <= 500 Hz -> distance >= fs / 500)
    min_distance = max(1, int(fs / 500.0))
    peaks, props = find_peaks(trimmed, height=0.0, distance=min_distance)

    if len(peaks) < 10:
        return float("nan"), float("nan")

    # Time durations between consecutive glottal pulses (Periods)
    peak_times = peaks / fs
    periods = np.diff(peak_times)
    mean_period = np.mean(periods)

    if mean_period <= 0:
        return float("nan"), float("nan")

    # Jitter: Relative cycle-to-cycle frequency perturbation
    jitter_abs = np.mean(np.abs(np.diff(periods)))
    jitter_percent = (jitter_abs / mean_period) * 100.0

    # Shimmer: Relative cycle-to-cycle amplitude perturbation
    amplitudes = props.get("peak_heights", np.array([]))
    if len(amplitudes) < 2 or np.mean(amplitudes) <= 0:
        return float("nan"), float("nan")

    shimmer_abs = np.mean(np.abs(np.diff(amplitudes)))
    shimmer_percent = (shimmer_abs / np.mean(amplitudes)) * 100.0

    return float(jitter_percent), float(shimmer_percent)


def get_cpp(
    signal_data: np.ndarray,
    fs: float = 8000.0,
    frame_len_s: float = 0.04,
    hop_s: float = 0.02,
    f_min: float = 60.0,
    f_max: float = 300.0
) -> float:
    """
    Calculate Cepstral Peak Prominence (CPP) across short-time frames.
    CPP quantifies harmonic periodicity and vocal regularity in the quefrency domain.

    Parameters:
        signal_data: 1D voice waveform.
        fs: Sampling frequency in Hz.
        frame_len_s: Frame length in seconds (default 40 ms).
        hop_s: Step duration between frames (default 20 ms).
        f_min, f_max: Plausible pitch boundaries in Hz defining quefrency search range.

    Returns:
        float: 90th percentile of frame-wise peak cepstral amplitudes.
    """
    sig = np.asarray(signal_data, dtype=float)
    fs = float(fs)
    std_val = np.std(sig)
    if std_val < 1e-12:
        return float("nan")

    # Standardize
    sig = (sig - np.mean(sig)) / std_val

    frame_len = int(frame_len_s * fs)
    hop = int(hop_s * fs)

    # Quefrency indices: tau = fs / f
    min_q = max(1, int(fs / f_max))
    max_q = min(int(frame_len // 2), int(fs / f_min))

    peaks = []
    window = np.hanning(frame_len)

    for i in range(0, len(sig) - frame_len, hop):
        frame = sig[i : i + frame_len] * window
        spectrum = np.abs(fft.rfft(frame))
        log_spec = np.log(spectrum + 1e-9)
        ceps = np.abs(fft.irfft(log_spec))

        if len(ceps) > max_q and max_q > min_q:
            peak_val = np.max(ceps[min_q:max_q])
            peaks.append(peak_val)

    if not peaks:
        return float("nan")

    return float(np.percentile(peaks, 90))


def health_or_pathological_threshold(
    jitter: float,
    shimmer: float,
    jitter_th: float = 1.04,
    shimmer_th: float = 3.81
) -> str:
    """
    Rule-based threshold classification based on classic literature norms:
    Pathological if Jitter > 1.04% and Shimmer > 3.81%.
    """
    if np.isnan(jitter) or np.isnan(shimmer):
        return "Unknown"
    if jitter > jitter_th and shimmer > shimmer_th:
        return "Pathological"
    return "Healthy"
