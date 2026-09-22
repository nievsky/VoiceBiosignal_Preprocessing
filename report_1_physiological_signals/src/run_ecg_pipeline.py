"""
Standalone CLI runner for Report 1: ECG and Hemodynamic Analysis.
Can execute against local PhysioNet databases or in demonstration mode with synthesized signals.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .ecg_detector import process_ecg, evaluate_ecg_record
    from .multimodal_correlation import preprocess_signal, compute_pairwise_correlations, lagged_cross_correlation
except (ImportError, ValueError):
    from ecg_detector import process_ecg, evaluate_ecg_record
    from multimodal_correlation import preprocess_signal, compute_pairwise_correlations, lagged_cross_correlation


def run_demo_mode():
    print("=" * 70)
    print("RUNNING REPORT 1 PIPELINE IN SYNTHETIC DEMONSTRATION MODE")
    print("=" * 70)

    # 1. Synthesize 60s of realistic ECG-like signal at 128 Hz
    fs = 128.0
    duration_s = 60.0
    t = np.arange(int(fs * duration_s)) / fs
    heart_rate_target = 72.0  # bpm
    r_interval_s = 60.0 / heart_rate_target

    synthetic_ecg = np.random.normal(0, 0.05, len(t))
    r_peak_locs = []
    curr_t = 0.5
    while curr_t < duration_s - 0.5:
        idx = int(curr_t * fs)
        r_peak_locs.append(idx)
        # Add QRS wave
        synthetic_ecg[idx] += 1.5
        if idx > 2:
            synthetic_ecg[idx - 2] -= 0.2  # Q wave
        if idx < len(synthetic_ecg) - 2:
            synthetic_ecg[idx + 2] -= 0.3  # S wave
        curr_t += r_interval_s + np.random.normal(0, 0.02)

    # Detect R-peaks
    r_peaks, hr_times, hr_bpm, mean_bpm = process_ecg(synthetic_ecg, fs)
    print(f"\n[Part 1: ECG Analysis]")
    print(f"Synthesized beats: {len(r_peak_locs)}")
    print(f"Detected R-peaks:  {len(r_peaks)}")
    print(f"Mean Heart Rate:   {mean_bpm:.2f} BPM (Target: {heart_rate_target:.2f} BPM)")

    # 2. Synthesize multimodal signals with deliberate delay
    abp_synthetic = np.roll(synthetic_ecg, int(fs * 0.25)) + np.random.normal(0, 0.1, len(t))
    icp_synthetic = np.roll(abp_synthetic, int(fs * 0.15)) + np.random.normal(0, 0.08, len(t))

    ecg_clean, _ = preprocess_signal(synthetic_ecg, fs, max_duration_sec=duration_s)
    abp_clean, _ = preprocess_signal(abp_synthetic, fs, max_duration_sec=duration_s)
    icp_clean, _ = preprocess_signal(icp_synthetic, fs, max_duration_sec=duration_s)

    corr_static = compute_pairwise_correlations(ecg_clean, abp_clean, icp_clean)
    print(f"\n[Part 2: Static Pearson Correlations]")
    print(corr_static.to_string(index=False))

    _, _, lag_ea, r_ea = lagged_cross_correlation(ecg_clean, abp_clean, fs, max_lag_s=2.0)
    _, _, lag_ai, r_ai = lagged_cross_correlation(abp_clean, icp_clean, fs, max_lag_s=2.0)
    print(f"\n[Part 2: Lagged Cross-Correlation]")
    print(f"ECG <-> ABP: Max correlation = {r_ea:.3f} at lag = {lag_ea:.3f} s")
    print(f"ABP <-> ICP: Max correlation = {r_ai:.3f} at lag = {lag_ai:.3f} s")
    print("\nDemonstration execution completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="ECG Detection and Multimodal Signal Correlation Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run in self-contained demonstration mode with synthesized data")
    parser.add_argument("--data-dir", type=str, default="", help="Path to base data directory containing PhysioNet databases")
    args = parser.parse_args()

    if args.demo:
        run_demo_mode()
        return

    # Check for PhysioNet datasets
    candidate_paths = [
        args.data_dir,
        os.path.join(os.getcwd(), "data"),
        os.path.join(os.getcwd(), "..", "data"),
        r"C:\Users\alecs\My Drive\UJEP\PZS\SemPrace1\data"
    ]

    found_dir = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            found_dir = p
            break

    if not found_dir:
        print("Notice: No PhysioNet data folder detected in default locations.")
        print("Falling back to demonstration mode. Use --data-dir to specify a local dataset path.")
        run_demo_mode()
        return

    print(f"Using PhysioNet data directory: {found_dir}")
    try:
        import wfdb
    except ImportError:
        print("Error: 'wfdb' package is required to read PhysioNet binaries. Install via 'pip install wfdb'.")
        return

    # 1. Test ECG on Stress Drive 01 if available
    drive_path = os.path.join(found_dir, "stress-recognition-in-automobile-drivers-1.0.0", "stress-recognition-in-automobile-drivers-1.0.0", "drive01")
    if os.path.exists(drive_path + ".hea"):
        print("\n--- Processing Drive01 (Driver Stress Database) ---")
        record = wfdb.rdrecord(drive_path)
        ecg = record.p_signal[:, 0]
        fs = record.fs
        peaks, _, _, mean_bpm = process_ecg(ecg, fs)
        print(f"Total duration: {len(ecg)/fs:.1f} s, Detected peaks: {len(peaks)}, Mean HR: {mean_bpm:.2f} BPM")

    # 2. Test MIT-BIH NSRDB if available
    nsrdb_dir = os.path.join(found_dir, "mit-bih-normal-sinus-rhythm-database-1.0.0", "mit-bih-normal-sinus-rhythm-database-1.0.0")
    if os.path.exists(nsrdb_dir):
        patient_list = ['16265', '16272', '16273', '16420', '16483']
        print(f"\n--- Testing sample patients on MIT-BIH NSRDB ---")
        for pid in patient_list:
            rec_path = os.path.join(nsrdb_dir, pid)
            if os.path.exists(rec_path + ".hea") and os.path.exists(rec_path + ".atr"):
                rec = wfdb.rdrecord(rec_path)
                ann = wfdb.rdann(rec_path, "atr")
                eval_res = evaluate_ecg_record(rec.p_signal[:, 0], ann.sample, rec.fs)
                print(f"Patient {pid}: Actual={eval_res['actual_hr_bpm']:.1f} BPM, Detected={eval_res['detected_hr_bpm']:.1f} BPM, Accuracy={eval_res['accuracy_pct']:.2f}%")

    # 3. Test CHARIS database if available
    charis_dir = os.path.join(found_dir, "charis-database-1.0.0", "charis-database-1.0.0")
    charis6_path = os.path.join(charis_dir, "charis6")
    if os.path.exists(charis6_path + ".hea"):
        print(f"\n--- Testing Multimodal Correlation on CHARIS (charis6) ---")
        rec = wfdb.rdrecord(charis6_path)
        abp, fs = preprocess_signal(rec.p_signal[:, 0], rec.fs)
        ecg, _ = preprocess_signal(rec.p_signal[:, 1], rec.fs)
        icp, _ = preprocess_signal(rec.p_signal[:, 2], rec.fs)

        _, _, lag_ea, r_ea = lagged_cross_correlation(ecg, abp, fs, max_lag_s=5.0)
        _, _, lag_ai, r_ai = lagged_cross_correlation(abp, icp, fs, max_lag_s=5.0)
        print(f"ECG <-> ABP: Max r = {r_ea:.3f} at lag = {lag_ea:.3f} s")
        print(f"ABP <-> ICP: Max r = {r_ai:.3f} at lag = {lag_ai:.3f} s")

    print("\nExecution complete.")


if __name__ == "__main__":
    main()
