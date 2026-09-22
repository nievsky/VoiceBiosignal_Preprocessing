# Advanced Biosignal & Acoustic Voice Processing Reports

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data: PhysioNet](https://img.shields.io/badge/Data-PhysioNet-red.svg)](https://physionet.org/)
[![Code Style: Clean](https://img.shields.io/badge/code%20style-modular-brightgreen.svg)]()

**Academic Term:** 2025/2026



## Executive Summary of Projects

| Project | Domain | Key Datasets | Core Techniques | Key Results |
|:-------:|:------:|:------------:|:---------------:|:-----------:|
| **[Report 1](report_1_physiological_signals/README.md)** | **Cardiovascular & Neurocritical Biosignals** | `drivedb`, `nsrdb`, `charisdb` | Z-score prominence R-peak detection, Refractory suppression, Lagged normalized cross-correlation | • **86.00%** mean detection accuracy across 18 MIT-BIH Holter records (up to **99.42%**).<br>• Discovered severe autoregulatory coupling ($\rho$ up to **0.855**) in TBI patients with impaired compliance. |
| **[Report 2](report_2_voice_pathology/README.md)** | **Acoustic Speech & Voice Pathology** | `voiced` (208 subjects) | Jitter, Shimmer, Cepstral Peak Prominence (CPP), Spectral Timbre Bands, Random Forest, k-NN | • **93.27%** accuracy in binary pathology detection via Random Forest.<br>• **48.96%** 10-fold CV accuracy in 3-class differential diagnosis (surpassing 33.3% random chance). |

---

## Key Highlights

### 🫀 [Report 1: ECG Analysis & Multimodal Hemodynamic Cross-Correlation](report_1_physiological_signals/README.md)
* **Automated ECG R-Peak Detection:** Implemented an adaptive prominence-based peak localization algorithm on driver stress data (`drivedb/1.0.0`), followed by rigorous validation against 18 gold-standard clinical recordings from the **MIT-BIH Normal Sinus Rhythm Database** (`nsrdb/1.0.0`).
* **Hemodynamic Cross-Correlation in TBI:** Evaluated simultaneous recordings of Arterial Blood Pressure (ABP), Intracranial Pressure (ICP), and ECG from 13 traumatic brain injury patients in the **CHARIS Database** (`charisdb/1.0.0`). Demonstrated why zero-lag Pearson correlation fails ($r \approx 0$) due to physiological pulse transit delays, whereas lagged cross-correlation exposes critical vascular compliance dynamics.
* [**Read Full Report 1 ➔**](report_1_physiological_signals/README.md)

---

### 🎙️ [Report 2: Acoustic & Cepstral Characterization of Voice Disorders](report_2_voice_pathology/README.md)
* **Failure of Classical Heuristic Thresholds:** Demonstrated that standard clinical rules ($\text{Jitter} > 1.04\%$, $\text{Shimmer} > 3.81\%$) collapse on severe hoarseness by classifying 100% of samples as pathological (0% healthy sensitivity).
* **Noise-Resilient Quefrency Analysis:** Introduced **Cepstral Peak Prominence (CPP)** to measure harmonic voice regularity in the quefrency domain, achieving **93.27% accuracy** using a Random Forest model.
* **Timbre Analysis & Multiclass Differential Diagnosis:** Decomposed power spectra into Low, Mid, and High frequency bands to quantify acoustic timbre across three pathologies (*Hyperkinetic Dysphonia*, *Hypokinetic Dysphonia*, and *Reflux Laryngitis*), achieving a cross-validated accuracy of **48.96%** (max fold **60.00%**) via distance-weighted k-NN.
* [**Read Full Report 2 ➔**](report_2_voice_pathology/README.md)

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sanyanevsky/biosignal-processing-reports.git
cd biosignal-processing-reports
```

### 2. Create and Activate a Virtual Environment
```bash
# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quickstart: Running the Code

Both modules provide standalone Command-Line Interface (CLI) runners equipped with **synthetic demonstration modes**, allowing you to execute and test the complete pipeline instantly without needing multi-gigabyte database downloads:

### Run Report 1 (ECG & Hemodynamics)
```bash
# Synthetic demonstration mode
python report_1_physiological_signals/src/run_ecg_pipeline.py --demo

# Execution on local PhysioNet data
python report_1_physiological_signals/src/run_ecg_pipeline.py --data-dir path/to/data
```

### Run Report 2 (Voice Pathology)
```bash
# Synthetic demonstration mode
python report_2_voice_pathology/src/run_voice_pipeline.py --demo

# Execution on local VOICED data
python report_2_voice_pathology/src/run_voice_pipeline.py --data-dir path/to/voiced-database-1.0.0/voice-icar-federico-ii-database-1.0.0
```

### Running Interactive Jupyter Notebooks
Pre-computed interactive notebooks with full cell outputs are available in:
- `report_1_physiological_signals/notebooks/01_ecg_heart_rate_detection.ipynb`
- `report_1_physiological_signals/notebooks/02_multimodal_signal_correlation.ipynb`
- `report_2_voice_pathology/notebooks/voice_pathology_classification.ipynb`

Launch Jupyter:
```bash
jupyter lab
# or
jupyter notebook
```

---

## Datasets & Acknowledgments

All biomedical records originate from open clinical repositories hosted by [PhysioNet](https://physionet.org):
1. **Stress Recognition in Automobile Drivers (`drivedb`):** Healey & Picard, MIT Media Lab.
2. **MIT-BIH Normal Sinus Rhythm Database (`nsrdb`):** Beth Israel Hospital & MIT.
3. **CHARIS Traumatic Brain Injury Database (`charisdb`):** Kim et al., CHARIS Research Team.
4. **VOICED Voice Icar Federico II Database (`voiced`):** Cesari et al., University of Naples Federico II.

For dataset download procedures, please refer to [`data/README.md`](data/README.md).

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
