# Dataset Acquisition and Structure

This repository uses open-access biomedical and acoustic databases hosted on **PhysioNet** ([physionet.org](https://physionet.org)). Because these datasets exceed multiple gigabytes in raw formats, they are not committed to git version control.

The analysis pipelines can run directly using local downloads of the PhysioNet records or via synthetic demonstration modes.

---

## Required Databases

### 1. Stress Recognition in Automobile Drivers (`drivedb`)
- **Used in:** Report 1 (Part 1: ECG Heart Rate Detection)
- **Source:** [https://physionet.org/content/drivedb/1.0.0/](https://physionet.org/content/drivedb/1.0.0/)
- **Description:** Multi-modal physiological recordings (ECG, EMG, GSR, respiration, heart rate) of drivers under natural driving conditions.
- **Expected location:** `data/stress-recognition-in-automobile-drivers-1.0.0/`

### 2. MIT-BIH Normal Sinus Rhythm Database (`nsrdb`)
- **Used in:** Report 1 (Part 1: R-Peak Detector Validation)
- **Source:** [https://physionet.org/content/nsrdb/1.0.0/](https://physionet.org/content/nsrdb/1.0.0/)
- **Description:** 18 long-term ECG recordings of subjects with normal sinus rhythm, complete with cardiologist-annotated beat labels (`.atr`).
- **Expected location:** `data/mit-bih-normal-sinus-rhythm-database-1.0.0/`

### 3. CHARIS Traumatic Brain Injury Database (`charisdb`)
- **Used in:** Report 1 (Part 2: Multi-Modal Hemodynamic Correlation)
- **Source:** [https://physionet.org/content/charisdb/1.0.0/](https://physionet.org/content/charisdb/1.0.0/)
- **Description:** Simultaneous recordings of arterial blood pressure (ABP), intracranial pressure (ICP), and electrocardiogram (ECG) from 13 patients suffering traumatic brain injury (TBI).
- **Expected location:** `data/charis-database-1.0.0/`

### 4. VOICED Database (`voiced`)
- **Used in:** Report 2 (Acoustic Voice Pathology Classification)
- **Source:** [https://physionet.org/content/voiced/1.0.0/](https://physionet.org/content/voiced/1.0.0/)
- **Description:** Voice records of sustained vowel /a/ from 208 subjects (58 healthy, 150 diagnosed with dysphonia, reflux laryngitis, etc.) recorded at 8,000 Hz.
- **Expected location:** `data/voiced-database-1.0.0/voice-icar-federico-ii-database-1.0.0/`

---

## Automatic Download via Python `wfdb`

You can use the PhysioNet `wfdb` package to automatically download specific records:

```python
import wfdb

# Download a sample record from MIT-BIH Normal Sinus Rhythm
wfdb.dl_database('nsrdb', dl_dir='data/mit-bih-normal-sinus-rhythm-database-1.0.0', records=['16265'])

# Download a sample record from VOICED
wfdb.dl_database('voiced', dl_dir='data/voiced-database-1.0.0', records=['voice001', 'voice002'])
```

Alternatively, download and extract the dataset archives directly into the `data/` folder following the hierarchy shown above.
