"""
Standalone CLI runner for Report 2: Voice Pathology Classification.
Can execute against local VOICED PhysioNet records or in synthetic demonstration mode.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

try:
    from .acoustic_features import calculate_acoustic_features, get_cpp, health_or_pathological_threshold
    from .spectral_analysis import get_spectral_bands
    from .classifiers import train_binary_random_forest, train_multiclass_knn
except (ImportError, ValueError):
    from acoustic_features import calculate_acoustic_features, get_cpp, health_or_pathological_threshold
    from spectral_analysis import get_spectral_bands
    from classifiers import train_binary_random_forest, train_multiclass_knn


def run_demo_mode():
    print("=" * 70)
    print("RUNNING REPORT 2 PIPELINE IN SYNTHETIC DEMONSTRATION MODE")
    print("=" * 70)

    fs = 8000.0
    duration_s = 2.0
    t = np.arange(int(fs * duration_s)) / fs
    f0 = 150.0  # fundamental frequency (Hz)

    records = []
    # Generate 30 synthetic healthy subjects and 30 synthetic pathological subjects
    np.random.seed(42)
    for i in range(60):
        is_healthy = (i < 25)
        jitter_noise = 0.002 if is_healthy else 0.03
        amp_noise = 0.05 if is_healthy else 0.35

        # Harmonic voice model
        pulse_train = np.zeros_like(t)
        period = int(fs / f0)
        idx = 0
        while idx < len(t):
            pulse_train[idx] = 1.0 + np.random.normal(0, amp_noise)
            step = int(period + np.random.normal(0, jitter_noise * period))
            idx += max(10, step)

        sig = np.convolve(pulse_train, np.exp(-np.linspace(0, 5, 50)), mode='same')
        sig += np.random.normal(0, 0.02 if is_healthy else 0.15, len(t))

        jit, shim = calculate_acoustic_features(sig, fs=fs)
        cpp = get_cpp(sig, fs=fs)
        low, mid, high = get_spectral_bands(sig, fs=fs)

        if is_healthy:
            diag = "healthy"
            target = 0
        else:
            diag = np.random.choice(["hyperkinetic dysphonia", "hypokinetic dysphonia", "reflux laryngitis"])
            target = 1

        records.append({
            "Record ID": f"demo_{i:03d}",
            "Jitter": jit if not np.isnan(jit) else 10.0,
            "Shimmer": shim if not np.isnan(shim) else 50.0,
            "CPP": cpp if not np.isnan(cpp) else 0.3,
            "LowFreq": low,
            "MidFreq": mid,
            "HighFreq": high,
            "Target": target,
            "Diagnosis": diag
        })

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} synthetic phonation records.")

    # 1. Literature threshold test
    df["Threshold_Pred"] = [health_or_pathological_threshold(r.Jitter, r.Shimmer) for _, r in df.iterrows()]
    th_acc = np.mean((df["Target"] == 0) == (df["Threshold_Pred"] == "Healthy"))
    print(f"\n[Part 1: Classical Thresholding]")
    print(f"Accuracy using Jitter > 1.04% & Shimmer > 3.81%: {th_acc * 100:.2f}%")

    # 2. Binary Random Forest
    rf_res = train_binary_random_forest(df)
    print(f"\n[Part 2: Random Forest Classification (Healthy vs Pathological)]")
    print(f"Total Dataset Accuracy: {rf_res['full_accuracy'] * 100:.2f}%")
    print("Feature Importances:")
    for feat, imp in rf_res['feature_importances'].items():
        print(f"  - {feat}: {imp:.4f}")

    # 3. Multiclass k-NN
    pathology_df = df[df["Target"] == 1].copy()
    if len(pathology_df) >= 15:
        knn_res = train_multiclass_knn(pathology_df, cv_folds=3)
        print(f"\n[Part 3: Multiclass Pathology Discrimination (k-NN)]")
        print(f"Cross-Validation Accuracy: {knn_res['cv_mean'] * 100:.2f}% (+/- {knn_res['cv_std'] * 100:.2f}%)")

    print("\nDemonstration execution completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Voice Pathology Classification Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run in self-contained demonstration mode")
    parser.add_argument("--data-dir", type=str, default="", help="Path to VOICED database directory")
    args = parser.parse_args()

    if args.demo:
        run_demo_mode()
        return

    candidate_paths = [
        args.data_dir,
        os.path.join(os.getcwd(), "data", "voiced-database-1.0.0", "voice-icar-federico-ii-database-1.0.0"),
        os.path.join(os.getcwd(), "..", "data", "voiced-database-1.0.0", "voice-icar-federico-ii-database-1.0.0"),
        r"C:\Users\alecs\My Drive\UJEP\PZS\SemPrace2\voiced-database-1.0.0\voice-icar-federico-ii-database-1.0.0"
    ]

    found_dir = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            found_dir = p
            break

    if not found_dir:
        print("Notice: No VOICED database folder detected in default locations.")
        print("Falling back to demonstration mode. Use --data-dir to specify a local dataset path.")
        run_demo_mode()
        return

    print(f"Using VOICED database directory: {found_dir}")
    try:
        import wfdb
    except ImportError:
        print("Error: 'wfdb' package is required. Install via 'pip install wfdb'.")
        return

    # Process all records
    print("Loading 208 records from VOICED database...")
    voice_data = []
    for i in range(1, 209):
        rec_name = f"voice{i:03d}"
        rec_path = os.path.join(found_dir, rec_name)
        if not os.path.exists(rec_path + ".hea"):
            continue
        rec = wfdb.rdrecord(rec_path)
        sig = rec.p_signal[:, 0]
        fs = rec.fs

        diagnosis = "unknown"
        for comment in rec.comments:
            if "<diagnoses>:" in comment:
                diagnosis = comment.split("<diagnoses>:")[1].split("<")[0].strip().lower()
                break

        is_healthy = "healthy" in diagnosis
        jit, shim = calculate_acoustic_features(sig, fs)
        cpp = get_cpp(sig, fs)
        low, mid, high = get_spectral_bands(sig, fs)

        voice_data.append({
            "Record ID": rec_name,
            "Diagnosis": diagnosis,
            "Target": 0 if is_healthy else 1,
            "Status": "Healthy" if is_healthy else "Pathological",
            "Jitter": jit,
            "Shimmer": shim,
            "CPP": cpp,
            "LowFreq": low,
            "MidFreq": mid,
            "HighFreq": high
        })

    df = pd.DataFrame(voice_data)
    print(f"Loaded {len(df)} records. Pathological: {sum(df.Target == 1)}, Healthy: {sum(df.Target == 0)}")

    # Train binary Random Forest
    rf_res = train_binary_random_forest(df)
    print(f"\n--- Random Forest Binary Accuracy (Whole Dataset): {rf_res['full_accuracy'] * 100:.2f}% ---")
    print("Confusion Matrix:")
    print(pd.DataFrame(rf_res['confusion_matrix'], index=["Actual Healthy", "Actual Patho"], columns=["Pred Healthy", "Pred Patho"]))

    # Multiclass on the 3 pathologies
    pat_df = df[df["Diagnosis"].isin(["hyperkinetic dysphonia", "hypokinetic dysphonia", "reflux laryngitis"])].copy()
    if len(pat_df) > 0:
        knn_res = train_multiclass_knn(pat_df, cv_folds=10)
        print(f"\n--- Multiclass k-NN 10-Fold Cross-Validation Accuracy: {knn_res['cv_mean'] * 100:.2f}% (+/- {knn_res['cv_std'] * 100:.2f}%) ---")

    print("\nExecution completed successfully!")


if __name__ == "__main__":
    main()
