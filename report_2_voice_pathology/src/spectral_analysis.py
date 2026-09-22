"""
Spectral and timbre analysis of acoustic phonation.
Decomposes power spectral density into distinct clinical frequency bands.
"""

from typing import Tuple
import numpy as np
import scipy.fft as fft


def get_spectral_bands(
    signal: np.ndarray,
    fs: float = 8000.0
) -> Tuple[float, float, float]:
    """
    Divide signal power spectrum into 3 clinically meaningful bands and compute energy ratios:
      - Band 1 (0 - 800 Hz): Fundamental frequency (F0) & lowest harmonics.
      - Band 2 (800 - 2500 Hz): Vowel formants & acoustic timbre information.
      - Band 3 (2500+ Hz): High-frequency turbulent noise, friction & breathiness.

    Parameters:
        signal: 1D array of phonation samples.
        fs: Sampling rate in Hz.

    Returns:
        Tuple: (ratio_low, ratio_mid, ratio_high) normalized such that their sum equals 1.0.
    """
    sig = np.asarray(signal, dtype=float)
    fs = float(fs)

    if len(sig) == 0:
        return 0.0, 0.0, 0.0

    # Windowing to minimize spectral leakage
    windowed = sig * np.hanning(len(sig))
    spectrum = np.abs(fft.rfft(windowed)) ** 2  # Power spectral density
    freqs = fft.rfftfreq(len(sig), 1.0 / fs)

    total_energy = float(np.sum(spectrum))
    if total_energy <= 0:
        return 0.0, 0.0, 0.0

    idx_low = (freqs <= 800.0)
    idx_mid = (freqs > 800.0) & (freqs <= 2500.0)
    idx_high = (freqs > 2500.0)

    e_low = float(np.sum(spectrum[idx_low]))
    e_mid = float(np.sum(spectrum[idx_mid]))
    e_high = float(np.sum(spectrum[idx_high]))

    return e_low / total_energy, e_mid / total_energy, e_high / total_energy
