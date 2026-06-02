# ECG Image Digitization System
### AI-Based Final Year Project

A Streamlit web app that converts scanned paper ECG images into digital time-series signals using classical image processing. No GPU or pretrained model required.

---

## 🚀 Setup & Run (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open in browser → http://localhost:8501
```

---

## 📁 Project Structure

```
ecg-digitizer/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← All dependencies
└── README.md           ← This file
```

---

## 🔬 Pipeline (Maps to SRS FR1–FR7)

| SRS Requirement | Implementation |
|---|---|
| FR1 — Image Upload | `st.file_uploader()` with PNG/JPG validation |
| FR2 — Preprocessing | Grayscale → Gaussian blur → morphological gridline removal → FFT suppression |
| FR3 — Waveform Detection | Horizontal edge projection → `find_peaks` → strip scoring → top N rows |
| FR4 — Digitization | Column-wise argmax pixel scan → median + Savitzky-Golay smooth → CSV |
| FR5 — Visualization | Side-by-side: cleaned segment image + reconstructed waveform plot |
| FR6 — Export | CSV download (lead, time_s, amplitude_norm) + PNG plot download |
| FR7 — Web Interface | Streamlit app with sidebar controls |
| NFR1 — Performance | Typically 3–8 seconds on CPU |
| NFR6 — Security | Images processed in RAM only, never written to disk |

---

## 📊 Output CSV Format

```
lead,time_s,amplitude_norm
I,0.0,-0.12345
I,0.002,0.03421
...
II,0.0,0.54321
...
```

---

## 🧪 Test ECG Images

Download sample ECG images from:
- `github.com/alphanumericslab/ecg-image-kit` → `sample-data/` folder
- Kaggle: search "ECG image dataset"
- PhysioNet: `physionet.org/content/ecg-image-database/`

---

## 📚 References

- Shivashankara et al. (2024). ECG-Image-Kit. *Physiological Measurement*, IOP Publishing.
- George B. Moody PhysioNet Challenge 2024: Digitization and Classification of ECG Images.
- OpenCV morphological operations for gridline removal.
- Savitzky-Golay filter for signal smoothing (SciPy).
