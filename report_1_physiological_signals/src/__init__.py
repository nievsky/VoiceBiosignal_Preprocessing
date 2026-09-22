"""
ECG and Hemodynamic Signal Processing Module
Part of Seminar Work 1: KI/PZS (Computer Signal Processing), UJEP
Author: Aleksandr Demin
"""

from .ecg_detector import detect_r_peaks, compute_heart_rate, process_ecg, evaluate_ecg_record
from .multimodal_correlation import preprocess_signal, lagged_cross_correlation, compute_pairwise_correlations

__all__ = [
    "detect_r_peaks",
    "compute_heart_rate",
    "process_ecg",
    "evaluate_ecg_record",
    "preprocess_signal",
    "lagged_cross_correlation",
    "compute_pairwise_correlations"
]
