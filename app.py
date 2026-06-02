import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.signal import savgol_filter, find_peaks
import io
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Image Digitizer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3547;
    }

    /* Cards */
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2e3547;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card .label {
        font-size: 12px;
        color: #7a8499;
        font-family: monospace;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 22px;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-card .sub {
        font-size: 11px;
        color: #3d4560;
        margin-top: 4px;
    }

    /* Step headers */
    .step-header {
        background: #1a1d27;
        border: 1px solid #2e3547;
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 10px 16px;
        margin: 16px 0 10px;
        font-size: 14px;
        font-weight: 600;
        color: #e2e8f0;
    }
    .step-header span {
        font-family: monospace;
        font-size: 11px;
        color: #3b82f6;
        margin-right: 8px;
    }

    /* Info boxes */
    .info-box {
        background: #1e2d4a;
        border: 1px solid #2a4a7a;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #93c5fd;
        margin: 8px 0;
    }
    .warn-box {
        background: #2d2208;
        border: 1px solid #4a3810;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #fcd34d;
        margin: 8px 0;
    }
    .success-box {
        background: #0f2d1a;
        border: 1px solid #1a4a2a;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #86efac;
        margin: 8px 0;
    }

    /* Download button */
    .stDownloadButton > button {
        background-color: #1a1d27 !important;
        border: 1px solid #2e3547 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        border-color: #3b82f6 !important;
        color: #3b82f6 !important;
    }

    /* Rhythm result card */
    .rhythm-card {
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
        border: 1px solid;
    }
    .rhythm-card .rhythm-label {
        font-size: 11px;
        font-family: monospace;
        letter-spacing: .08em;
        margin-bottom: 8px;
        opacity: .7;
    }
    .rhythm-card .rhythm-name {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .rhythm-card .rhythm-abbr {
        font-size: 13px;
        font-family: monospace;
        opacity: .65;
        margin-bottom: 10px;
    }
    .rhythm-card .rhythm-desc {
        font-size: 13px;
        line-height: 1.6;
        opacity: .85;
    }

    /* Color variants */
    .rhythm-normal  { background:#0f2d1a; border-color:#1a4a2a; color:#86efac; }
    .rhythm-tachy   { background:#2d1515; border-color:#5a2020; color:#fca5a5; }
    .rhythm-brady   { background:#1e2d4a; border-color:#2a4a7a; color:#93c5fd; }
    .rhythm-afib    { background:#2d2208; border-color:#5a3810; color:#fcd34d; }
    .rhythm-pvc     { background:#2d1a2d; border-color:#4a2a5a; color:#d8b4fe; }
    .rhythm-unknown { background:#1a1d27; border-color:#2e3547; color:#7a8499; }

    /* Feature pills row */
    .feat-row { display:flex; gap:10px; flex-wrap:wrap; margin:12px 0; }
    .feat-pill {
        background:#1a1d27; border:1px solid #2e3547;
        border-radius:8px; padding:8px 14px;
        font-size:12px; color:#e2e8f0; text-align:center; flex:1; min-width:100px;
    }
    .feat-pill .fp-label { color:#7a8499; font-family:monospace; font-size:10px; margin-bottom:3px; }
    .feat-pill .fp-val   { font-size:18px; font-weight:700; }

    /* Confidence bar */
    .conf-bar-wrap { margin:10px 0 4px; }
    .conf-bar-bg   { background:#2e3547; border-radius:4px; height:6px; }
    .conf-bar-fill { height:6px; border-radius:4px; }

    /* Hide default streamlit elements */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PIPELINE FUNCTIONS  (ported from notebook)
# ─────────────────────────────────────────────

def load_and_grayscale(uploaded_file):
    """FR1 + FR2 (Step 1): Load uploaded image → BGR → Gray"""
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image. Please upload a valid JPG or PNG.")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, gray


def preprocess(gray):
    """FR2 (Steps 4–5): Blur → Edges → Gridline removal → Cleaned binary"""
    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to binary (dark waveform → white after THRESH_BINARY_INV)
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Remove vertical gridlines
    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_vert)

    # Remove horizontal gridlines
    kernel_horz = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_horz)

    # Subtract gridlines from binary image
    grid_removed = bw.copy()
    combined = cv2.add(vertical_lines, horizontal_lines)
    grid_removed = cv2.subtract(bw, combined)

    # Additional: strong vertical kernel for stubborn grid lines
    kernel_vert2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 120))
    vertical_strong = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_vert2)
    cleaned = cv2.subtract(grid_removed, vertical_strong)

    # Edge image for strip detection
    edges = cv2.Canny(blur, 30, 150)

    return blur, bw, cleaned, edges


def detect_lead_strips(gray, edges, pad=70, n_strips=4):
    """FR3 (Step 6): Horizontal projection → find_peaks → top N ECG rows"""
    h, w = gray.shape

    # Horizontal projection of edges
    proj = edges.sum(axis=1).astype(float)
    proj_smooth = gaussian_filter1d(proj, sigma=6)

    # Detect peaks
    peaks, props = find_peaks(
        proj_smooth,
        distance=80,
        height=np.max(proj_smooth) * 0.08
    )

    # Build strips
    all_strips = []
    for i, y in enumerate(peaks):
        y1 = max(0, int(y - pad))
        y2 = min(h, int(y + pad))
        all_strips.append((i, y1, y2))

    # Score strips by normalized edge energy, keep top N
    scores = []
    for idx, y1, y2 in all_strips:
        s = gray[y1:y2, :]
        edge_s = cv2.Canny(cv2.GaussianBlur(s, (5, 5), 0), 20, 120)
        energy = edge_s.sum() / (s.shape[0] * s.shape[1] + 1e-9)
        scores.append((idx, y1, y2, energy))

    scores_sorted = sorted(scores, key=lambda x: x[3], reverse=True)
    top_strips = scores_sorted[:n_strips]

    # Sort by vertical position (top → bottom)
    top_strips = sorted(top_strips, key=lambda x: x[1])

    return [(y1, y2) for (_, y1, y2, _) in top_strips], proj_smooth, peaks


def extract_signal_from_segment(seg_gray, fft_band=6):
    """
    FR3 + FR4 (Step 8): Per-segment classical digitization.
    normalize → FFT grid suppression → argmax column scan →
    median + SavGol smooth → amplitude array in [-1, 1]
    """
    seg = seg_gray.astype(np.float32)
    seg = (seg - seg.min()) / (seg.max() - seg.min() + 1e-9)
    seg_inv = 1.0 - seg  # waveform becomes bright

    # FFT — suppress vertical periodic gridline frequency
    F = np.fft.fft2(seg_inv)
    Fshift = np.fft.fftshift(F)
    rows, cols = seg_inv.shape
    ccol = cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[:, ccol - fft_band: ccol + fft_band] = 0
    Ff = Fshift * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(Ff)))
    img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min() + 1e-9)

    # Column-wise argmax: row index of brightest pixel per column = waveform position
    col_idx = np.argmax(img_back, axis=0)

    # Smooth: median filter then Savitzky-Golay
    col_med = median_filter(col_idx, size=9)
    win = 51 if cols > 101 else (33 if cols > 65 else 11)
    if win >= cols:
        win = cols - 1 if (cols - 1) % 2 == 1 else cols - 2
    if win < 3:
        win = 3
    col_smooth = savgol_filter(col_med.astype(float), window_length=win, polyorder=2)

    # Invert row index → amplitude (higher row = lower on image = lower voltage)
    amp = -(col_smooth - np.mean(col_smooth))
    rng = amp.max() - amp.min()
    if rng > 1e-9:
        amp = (amp - amp.min()) / rng
        amp = (amp - 0.5) * 2.0  # scale to [-1, 1]
    else:
        amp = np.zeros_like(amp)

    return amp, img_back


def digitize_all_leads(gray, strip_ranges):
    """
    FR4: For each detected strip, split into 3 columns → extract 12 leads.
    Standard 12-lead layout: 4 rows × 3 columns
    """
    lead_layout = [
        ["I",   "aVR", "V1", "V4"],
        ["II",  "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]
    # Actually standard layout is 3 rows × 4 columns:
    row_to_leads = [
        ["I",   "II",  "III"],
        ["aVR", "aVL", "aVF"],
        ["V1",  "V2",  "V3"],
        ["V4",  "V5",  "V6"],
    ]

    h, w = gray.shape
    results = {}   # lead_name → amp array
    cleaned_segs = {}  # lead_name → cleaned FFT image for display

    n_rows = min(len(strip_ranges), 4)

    for row_idx in range(n_rows):
        y1, y2 = strip_ranges[row_idx]
        strip = gray[y1:y2, :]
        col_w = w // 3

        for col in range(3):
            x1 = col * col_w
            x2 = w if col == 2 else (col + 1) * col_w
            seg = strip[:, x1:x2]

            lead_name = row_to_leads[row_idx][col]
            amp, img_back = extract_signal_from_segment(seg)
            results[lead_name] = amp
            cleaned_segs[lead_name] = img_back

    return results, cleaned_segs


def build_dataframe(signals, fs_estimate=500):
    """FR4: Build tidy DataFrame — time_s + one column per lead → CSV"""
    rows = []
    for lead, amp in signals.items():
        duration = 10.0 if lead == "II" else 2.5
        t = np.linspace(0, duration, len(amp))
        for i, (ti, ai) in enumerate(zip(t, amp)):
            rows.append({"lead": lead, "time_s": round(ti, 5), "amplitude_norm": round(float(ai), 6)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# MODULE 4 — RHYTHM CLASSIFICATION
# ─────────────────────────────────────────────

# Class definitions with clinical info
RHYTHM_CLASSES = {
    "NSR": {
        "name": "Normal Sinus Rhythm",
        "abbr": "NSR",
        "css": "rhythm-normal",
        "emoji": "✅",
        "hr_range": "60–100 bpm",
        "desc": (
            "The heart is beating in a normal, regular pattern originating "
            "from the sinoatrial (SA) node. RR intervals are consistent and "
            "heart rate is within the normal resting range."
        ),
    },
    "TACHY": {
        "name": "Sinus Tachycardia",
        "abbr": "Tachycardia",
        "css": "rhythm-tachy",
        "emoji": "⚡",
        "hr_range": "> 100 bpm",
        "desc": (
            "Heart rate is elevated above 100 bpm. The rhythm is regular and "
            "still originates from the SA node. Common causes include exercise, "
            "fever, anxiety, or underlying cardiac conditions."
        ),
    },
    "BRADY": {
        "name": "Sinus Bradycardia",
        "abbr": "Bradycardia",
        "css": "rhythm-brady",
        "emoji": "🐢",
        "hr_range": "< 60 bpm",
        "desc": (
            "Heart rate is below 60 bpm. The rhythm is regular. May be normal "
            "in trained athletes, or may indicate SA node dysfunction, "
            "hypothyroidism, or medication effects."
        ),
    },
    "AFIB": {
        "name": "Atrial Fibrillation",
        "abbr": "AFib",
        "css": "rhythm-afib",
        "emoji": "〰️",
        "hr_range": "Variable",
        "desc": (
            "Highly irregular RR intervals with no consistent pattern. "
            "Atrial fibrillation is caused by chaotic electrical activity in "
            "the atria. It is the most common sustained cardiac arrhythmia "
            "and significantly increases stroke risk."
        ),
    },
    "PVC": {
        "name": "Premature Ventricular Contractions",
        "abbr": "PVC / PAC",
        "css": "rhythm-pvc",
        "emoji": "💥",
        "hr_range": "Variable",
        "desc": (
            "Mostly regular rhythm with occasional premature beats detected "
            "as sudden short RR intervals followed by a compensatory pause. "
            "May indicate ectopic ventricular activity (PVC) or atrial "
            "premature contractions (PAC)."
        ),
    },
    "UNKNOWN": {
        "name": "Indeterminate Rhythm",
        "abbr": "Unknown",
        "css": "rhythm-unknown",
        "emoji": "❓",
        "hr_range": "N/A",
        "desc": (
            "Could not reliably detect enough R-peaks to classify the rhythm. "
            "This may be due to image quality, poor waveform extraction, or "
            "a non-standard ECG layout. Try a cleaner image or different lead."
        ),
    },
}


def detect_r_peaks(signal, min_distance_samples=30, min_height_factor=0.4):
    """
    Detect R-peaks in a normalised ECG amplitude array.
    Uses scipy find_peaks with adaptive height threshold.
    Returns array of peak indices.
    """
    if len(signal) < 10:
        return np.array([])

    # Adaptive threshold: fraction of signal range above the median
    median_val = np.median(signal)
    sig_range  = signal.max() - signal.min()
    height_thresh = median_val + min_height_factor * sig_range

    peaks, _ = find_peaks(
        signal,
        height=height_thresh,
        distance=min_distance_samples,
        prominence=0.15 * sig_range,
    )
    return peaks


def compute_rr_features(peaks, duration_s):
    """
    Given R-peak indices and signal duration, compute:
      - heart rate (bpm)
      - mean RR interval (s)
      - RR std deviation
      - RMSSD  (root mean square of successive differences)
      - RR coefficient of variation (CV = std/mean)
      - premature beat ratio
    Returns dict of features.  Returns None if < 3 peaks found.
    """
    if len(peaks) < 3:
        return None

    # peaks are column indices into the signal array.
    # We need the total signal length to convert correctly.
    # Use the last peak as a proxy for signal length (conservative but accurate enough).
    # actual formula: time_s = peak_index / total_signal_length * duration_s
    # We don't have total_signal_length here, so derive it:
    # The peak indices span [0, total_length). Use peaks.max() + a small margin.
    total_len_est = int(peaks[-1] * (1 + 1.0 / max(len(peaks), 1))) + 1

    # Convert peak indices to seconds
    peak_times   = peaks / total_len_est * duration_s
    rr_intervals = np.diff(peak_times)   # seconds between consecutive beats

    if len(rr_intervals) < 2:
        return None

    mean_rr  = float(np.mean(rr_intervals))
    std_rr   = float(np.std(rr_intervals))
    rmssd    = float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2)))
    cv_rr    = std_rr / (mean_rr + 1e-9)           # coefficient of variation
    hr_bpm   = 60.0 / (mean_rr + 1e-9)

    # Premature beat detection: RR interval < 75 % of mean RR
    short_rr = rr_intervals < 0.75 * mean_rr
    premature_ratio = float(short_rr.sum()) / len(rr_intervals)

    return {
        "hr_bpm":          hr_bpm,
        "mean_rr_s":       mean_rr,
        "std_rr_s":        std_rr,
        "rmssd":           rmssd,
        "cv_rr":           cv_rr,
        "premature_ratio": premature_ratio,
        "n_peaks":         len(peaks),
        "rr_intervals":    rr_intervals,
    }


def classify_rhythm(features):
    """
    Rule-based rhythm classifier.
    Returns (rhythm_key, confidence_0_to_1, rule_explanation).

    Rules (applied in priority order):
    1. UNKNOWN  — fewer than 3 R-peaks detected
    2. AFIB     — CV of RR > 0.18  (highly irregular intervals)
    3. PVC/PAC  — premature_ratio > 0.10 AND CV moderate (0.08–0.18)
    4. TACHY    — HR > 100 bpm, regular rhythm
    5. BRADY    — HR < 60  bpm, regular rhythm
    6. NSR      — everything else
    """
    if features is None:
        return "UNKNOWN", 0.0, "Fewer than 3 R-peaks detected — cannot classify."

    hr   = features["hr_bpm"]
    cv   = features["cv_rr"]
    pr   = features["premature_ratio"]
    rmssd = features["rmssd"]
    mean_rr = features["mean_rr_s"]

    # AFib: very irregular RR intervals
    if cv > 0.18:
        conf = min(1.0, (cv - 0.18) / 0.12 + 0.6)
        return "AFIB", conf, (
            f"CV of RR intervals = {cv:.3f} (threshold > 0.18). "
            f"RMSSD = {rmssd*1000:.1f} ms — indicates highly irregular rhythm."
        )

    # PVC/PAC: mostly regular but with premature beats
    if pr > 0.10 and cv < 0.18:
        conf = min(1.0, pr * 3 + 0.5)
        return "PVC", conf, (
            f"{pr*100:.0f}% of RR intervals are premature (< 75% of mean). "
            f"RR CV = {cv:.3f} — mostly regular with ectopic beats."
        )

    # Tachycardia
    if hr > 100:
        conf = min(1.0, (hr - 100) / 50 + 0.7)
        return "TACHY", conf, (
            f"Heart rate = {hr:.0f} bpm (threshold > 100 bpm). "
            f"RR CV = {cv:.3f} — regular rhythm with elevated rate."
        )

    # Bradycardia
    if hr < 60:
        conf = min(1.0, (60 - hr) / 30 + 0.7)
        return "BRADY", conf, (
            f"Heart rate = {hr:.0f} bpm (threshold < 60 bpm). "
            f"RR CV = {cv:.3f} — regular rhythm with slow rate."
        )

    # Normal Sinus Rhythm
    conf = min(1.0, 0.6 + (1 - cv) * 0.4)
    return "NSR", conf, (
        f"Heart rate = {hr:.0f} bpm (60–100 range). "
        f"RR CV = {cv:.3f} — regular rhythm, normal rate. "
        f"Premature beat ratio = {pr*100:.1f}%."
    )


def fig_to_bytes(fig):
    """Convert matplotlib figure to PNG bytes for download"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#0f1117", edgecolor="none")
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🫀 ECG Image Digitizer")
    st.markdown("---")
    st.markdown(
        "Converts a scanned or photographed **paper ECG** into a "
        "digital time-series signal using classical image processing."
    )
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("""
- `M1` Grayscale + blur
- `M1` Gridline removal (morphological + FFT)
- `M2` Lead strip detection (edge projection)
- `M3` Column-wise pixel scan
- `M3` Median + Savitzky-Golay smooth
- `M3` Export → CSV
""")
    st.markdown("---")
    st.markdown("**Settings**")
    n_strips = st.slider("Expected ECG rows", 2, 4, 4,
                         help="How many horizontal lead rows in the ECG image")
    pad = st.slider("Strip crop padding (px)", 40, 120, 70,
                    help="Vertical padding around each detected ECG row")
    fft_band = st.slider("FFT grid filter width", 2, 15, 6,
                         help="Width of vertical frequency band to suppress")

    st.markdown("---")
    st.markdown("**Rhythm Classification**")
    classify_lead = st.selectbox(
        "Classify rhythm from lead",
        options=["II", "I", "V1", "V2", "aVR", "aVL", "aVF",
                 "III", "V3", "V4", "V5", "V6"],
        index=0,
        help="Lead II is best for rhythm analysis (10s strip, clear P-waves and QRS)"
    )
    rpeak_sensitivity = st.slider(
        "R-peak sensitivity", 0.2, 0.7, 0.4, 0.05,
        help="Lower = detect more peaks (use for low-amplitude signals). Higher = stricter."
    )

    st.markdown("---")
    st.caption("Final Year Project · AI-Based ECG Digitization System")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("## AI-Based ECG Image Digitization System")
st.markdown(
    "Upload a scanned paper ECG image (JPG or PNG). "
    "The system will preprocess it, detect all lead strips, "
    "extract each waveform, and export the digitized signal as CSV."
)
st.markdown("---")

# ── FR1: File Upload ──────────────────────────
st.markdown('<div class="step-header"><span>FR1</span> Upload ECG Image</div>',
            unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Choose a scanned ECG image",
    type=["png", "jpg", "jpeg"],
    help="Best results with 300 DPI scans. Standard 12-lead layout recommended."
)

if uploaded is None:
    st.markdown(
        '<div class="info-box">👆 Upload a JPG or PNG ECG image above to begin.</div>',
        unsafe_allow_html=True
    )
    st.stop()

# Validate file size (max 20MB)
if uploaded.size > 20 * 1024 * 1024:
    st.markdown(
        '<div class="warn-box">⚠ File too large (max 20 MB). Please compress or resize the image.</div>',
        unsafe_allow_html=True
    )
    st.stop()

# ── FR2: Load + Preprocess ────────────────────
st.markdown('<div class="step-header"><span>FR2</span> Image Preprocessing</div>',
            unsafe_allow_html=True)

with st.spinner("Loading and preprocessing ECG image…"):
    t_start = time.time()
    try:
        img_rgb, gray = load_and_grayscale(uploaded)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    blur, bw, cleaned, edges = preprocess(gray)

h_img, w_img = gray.shape
col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots(figsize=(5, 3), facecolor="#1a1d27")
    ax.imshow(img_rgb); ax.set_title("Original ECG", color="#e2e8f0", fontsize=11)
    ax.axis("off"); fig.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5, 3), facecolor="#1a1d27")
    ax.imshow(gray, cmap="gray"); ax.set_title("Grayscale", color="#e2e8f0", fontsize=11)
    ax.axis("off"); fig.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col3:
    fig, ax = plt.subplots(figsize=(5, 3), facecolor="#1a1d27")
    ax.imshow(cleaned, cmap="gray")
    ax.set_title("Gridlines Removed (Binary)", color="#e2e8f0", fontsize=11)
    ax.axis("off"); fig.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown(
    f'<div class="info-box">Image loaded: <b>{w_img}×{h_img}px</b> · '
    f'Format: <b>{uploaded.type}</b> · Size: <b>{uploaded.size/1024:.1f} KB</b></div>',
    unsafe_allow_html=True
)

# ── FR3: Lead Strip Detection ─────────────────
st.markdown('<div class="step-header"><span>FR3</span> ECG Lead Strip Detection</div>',
            unsafe_allow_html=True)

with st.spinner("Detecting lead strips via horizontal edge projection…"):
    strip_ranges, proj_smooth, peaks = detect_lead_strips(gray, edges, pad=pad, n_strips=n_strips)

n_detected = len(strip_ranges)

if n_detected == 0:
    st.markdown(
        '<div class="warn-box">⚠ No lead strips detected. Try adjusting the padding slider or uploading a higher-quality image.</div>',
        unsafe_allow_html=True
    )
    st.stop()

# Show projection + detected rows
fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), facecolor="#1a1d27")

# Projection plot
axes[0].set_facecolor("#0f1117")
axes[0].plot(proj_smooth / proj_smooth.max(), color="#3b82f6", linewidth=1.5, label="Edge projection")
for p in peaks:
    axes[0].axvline(p, color="#f87171", linewidth=0.8, alpha=0.7)
axes[0].set_title("Horizontal Edge Projection (peaks = ECG rows)", color="#e2e8f0", fontsize=11)
axes[0].set_xlabel("Row (y)", color="#7a8499"); axes[0].set_ylabel("Normalized energy", color="#7a8499")
axes[0].tick_params(colors="#7a8499")
for spine in axes[0].spines.values(): spine.set_edgecolor("#2e3547")

# Annotated image
axes[1].set_facecolor("#0f1117")
axes[1].imshow(img_rgb)
axes[1].set_title(f"Detected Lead Rows ({n_detected} found)", color="#e2e8f0", fontsize=11)
for i, (y1, y2) in enumerate(strip_ranges):
    axes[1].axhline(y1, color="#4ade80", linewidth=1.5, linestyle="--", alpha=0.9)
    axes[1].axhline(y2, color="#f87171", linewidth=1.0, linestyle=":", alpha=0.7)
    axes[1].text(10, (y1+y2)//2, f"Row {i+1}", color="#fcd34d",
                 fontsize=9, va="center", fontweight="bold")
axes[1].axis("off")
fig.tight_layout(pad=1)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.markdown(
    f'<div class="success-box">✓ Detected <b>{n_detected} lead row(s)</b> → '
    f'up to <b>{n_detected * 3} leads</b> will be extracted (3 per row).</div>',
    unsafe_allow_html=True
)

# ── FR4: Signal Reconstruction ────────────────
st.markdown('<div class="step-header"><span>FR4</span> Signal Reconstruction → CSV</div>',
            unsafe_allow_html=True)

with st.spinner("Extracting waveforms from each lead segment…"):
    signals, cleaned_segs = digitize_all_leads(gray, strip_ranges)
    df_out = build_dataframe(signals)
    t_end = time.time()
    elapsed = t_end - t_start

n_leads = len(signals)
total_samples = len(df_out)

mc1, mc2, mc3, mc4 = st.columns(4)
for col, label, val, sub in zip(
    [mc1, mc2, mc3, mc4],
    ["Leads Extracted", "Total Samples", "Processing Time", "Output Format"],
    [n_leads, total_samples, f"{elapsed:.1f}s", "CSV"],
    ["of 12 standard leads", "across all leads", "on CPU", "time_s + amplitude_norm"],
):
    col.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{val}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True
    )

if elapsed <= 15:
    st.markdown(
        f'<div class="success-box">✓ Processing completed in <b>{elapsed:.1f}s</b> — within SRS NFR1 requirement (≤15s on CPU).</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="warn-box">⚠ Processing took {elapsed:.1f}s (SRS NFR1 target: ≤15s). Consider a smaller or compressed image.</div>',
        unsafe_allow_html=True
    )

# Preview CSV
with st.expander("📄 Preview CSV output (first 20 rows)"):
    st.dataframe(df_out.head(20), use_container_width=True)

# ── FR5: Visualization ────────────────────────
st.markdown('<div class="step-header"><span>FR5</span> Original vs Digitized — Visual Comparison</div>',
            unsafe_allow_html=True)

lead_names = list(signals.keys())
selected_leads = st.multiselect(
    "Select leads to plot",
    options=lead_names,
    default=lead_names[:4] if len(lead_names) >= 4 else lead_names,
)

if selected_leads:
    n_plot = len(selected_leads)
    fig, axes = plt.subplots(n_plot, 2, figsize=(16, 2.8 * n_plot), facecolor="#0f1117")
    if n_plot == 1:
        axes = [axes]

    for i, lead in enumerate(selected_leads):
        amp = signals[lead]
        seg_img = cleaned_segs[lead]
        duration = 10.0 if lead == "II" else 2.5
        t = np.linspace(0, duration, len(amp))

        # Left: cleaned segment image
        ax_img = axes[i][0]
        ax_img.set_facecolor("#0f1117")
        ax_img.imshow(seg_img, cmap="gray", aspect="auto")
        ax_img.set_title(f"Lead {lead} — FFT-cleaned segment", color="#e2e8f0", fontsize=10)
        ax_img.axis("off")

        # Right: reconstructed waveform
        ax_sig = axes[i][1]
        ax_sig.set_facecolor("#0f1117")
        ax_sig.plot(t, amp, color="#3b82f6", linewidth=1.2)
        ax_sig.axhline(0, color="#2e3547", linewidth=0.8, linestyle="--")
        ax_sig.set_title(f"Lead {lead} — Reconstructed Signal", color="#e2e8f0", fontsize=10)
        ax_sig.set_xlabel("Time (s)", color="#7a8499", fontsize=9)
        ax_sig.set_ylabel("Amplitude (norm.)", color="#7a8499", fontsize=9)
        ax_sig.set_ylim(-1.3, 1.3)
        ax_sig.tick_params(colors="#7a8499", labelsize=8)
        for spine in ax_sig.spines.values(): spine.set_edgecolor("#2e3547")

    fig.suptitle("ECG Image vs Reconstructed Digital Signal",
                 color="#e2e8f0", fontsize=13, y=1.01)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    comparison_png = fig_to_bytes(fig)
    plt.close(fig)
else:
    comparison_png = None
    st.info("Select at least one lead above to see the comparison plot.")

# ── FR6: Downloads ────────────────────────────
st.markdown('<div class="step-header"><span>FR6</span> Download Results</div>',
            unsafe_allow_html=True)

csv_bytes = df_out.to_csv(index=False).encode("utf-8")

dl1, dl2 = st.columns(2)

with dl1:
    st.download_button(
        label="⬇ Download Digitized ECG (CSV)",
        data=csv_bytes,
        file_name="ecg_digitized_signal.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(f"Contains {len(df_out):,} rows · columns: lead, time_s, amplitude_norm")

with dl2:
    if comparison_png:
        st.download_button(
            label="⬇ Download Comparison Plot (PNG)",
            data=comparison_png,
            file_name="ecg_comparison_plot.png",
            mime="image/png",
            use_container_width=True,
        )
        st.caption("High-res PNG of original segment vs reconstructed waveform")
    else:
        st.info("Select leads above to enable PNG download.")

st.markdown("---")

# ── FR8: Rhythm Classification ────────────────
st.markdown('<div class="step-header"><span>FR8</span> Heartbeat Rhythm Classification</div>',
            unsafe_allow_html=True)

st.markdown(
    f"Classifying rhythm from **Lead {classify_lead}** "
    f"using R-peak detection + rule-based clinical analysis."
)

# Get signal for selected lead — fall back gracefully
if classify_lead in signals:
    clf_signal = signals[classify_lead]
    clf_duration = 10.0 if classify_lead == "II" else 2.5
else:
    # Pick the first available lead
    classify_lead = list(signals.keys())[0]
    clf_signal = signals[classify_lead]
    clf_duration = 10.0 if classify_lead == "II" else 2.5
    st.markdown(
        f'<div class="warn-box">⚠ Lead {classify_lead} was not extracted. '
        f'Using {classify_lead} instead.</div>',
        unsafe_allow_html=True
    )

with st.spinner("Detecting R-peaks and classifying rhythm…"):
    clf_t = np.linspace(0, clf_duration, len(clf_signal))

    # Estimate min distance between R-peaks:
    # At 40 bpm (very slow), one beat every 1.5s → samples per beat
    # Signal has len(clf_signal) samples over clf_duration seconds
    samples_per_sec_est = len(clf_signal) / clf_duration
    min_dist = max(10, int(samples_per_sec_est * 0.4))  # 0.4s = 150 bpm max

    r_peaks = detect_r_peaks(
        clf_signal,
        min_distance_samples=min_dist,
        min_height_factor=rpeak_sensitivity,
    )
    features = compute_rr_features(r_peaks, clf_duration)
    rhythm_key, confidence, explanation = classify_rhythm(features)

cls = RHYTHM_CLASSES[rhythm_key]

# ── Result card ──────────────────────────────
st.markdown(
    f'<div class="rhythm-card {cls["css"]}">'
    f'  <div class="rhythm-label">CLASSIFICATION RESULT</div>'
    f'  <div class="rhythm-name">{cls["emoji"]} {cls["name"]}</div>'
    f'  <div class="rhythm-abbr">{cls["abbr"]} · HR range: {cls["hr_range"]}</div>'
    f'  <div class="rhythm-desc">{cls["desc"]}</div>'
    f'</div>',
    unsafe_allow_html=True
)

# ── Feature pills ────────────────────────────
if features:
    hr_str   = f'{features["hr_bpm"]:.0f}'
    rr_str   = f'{features["mean_rr_s"]*1000:.0f} ms'
    cv_str   = f'{features["cv_rr"]:.3f}'
    np_str   = str(features["n_peaks"])
    pr_str   = f'{features["premature_ratio"]*100:.1f}%'
    conf_str = f'{confidence*100:.0f}%'
else:
    hr_str = rr_str = cv_str = np_str = pr_str = conf_str = "N/A"

st.markdown(
    f'<div class="feat-row">'
    f'  <div class="feat-pill"><div class="fp-label">HEART RATE</div><div class="fp-val">{hr_str} <span style="font-size:13px;font-weight:400">bpm</span></div></div>'
    f'  <div class="feat-pill"><div class="fp-label">MEAN RR</div><div class="fp-val">{rr_str}</div></div>'
    f'  <div class="feat-pill"><div class="fp-label">RR VARIABILITY (CV)</div><div class="fp-val">{cv_str}</div></div>'
    f'  <div class="feat-pill"><div class="fp-label">R-PEAKS FOUND</div><div class="fp-val">{np_str}</div></div>'
    f'  <div class="feat-pill"><div class="fp-label">PREMATURE BEATS</div><div class="fp-val">{pr_str}</div></div>'
    f'  <div class="feat-pill"><div class="fp-label">CONFIDENCE</div><div class="fp-val">{conf_str}</div></div>'
    f'</div>',
    unsafe_allow_html=True
)

# Confidence bar
bar_color = {
    "NSR": "#4ade80", "TACHY": "#f87171", "BRADY": "#3b82f6",
    "AFIB": "#fcd34d", "PVC": "#a78bfa", "UNKNOWN": "#7a8499"
}[rhythm_key]
bar_width = int(confidence * 100)
st.markdown(
    f'<div class="conf-bar-wrap">'
    f'  <div style="font-size:11px;color:#7a8499;font-family:monospace;margin-bottom:4px;">CLASSIFICATION CONFIDENCE</div>'
    f'  <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{bar_width}%;background:{bar_color};"></div></div>'
    f'  <div style="font-size:11px;color:#7a8499;margin-top:3px;">{explanation}</div>'
    f'</div>',
    unsafe_allow_html=True
)

# ── R-peak plot ──────────────────────────────
st.markdown("#### R-Peak Detection on Lead " + classify_lead)

fig, axes = plt.subplots(2, 1, figsize=(14, 6), facecolor="#0f1117",
                         gridspec_kw={"height_ratios": [3, 1]})

# Top: signal + R-peaks
ax = axes[0]
ax.set_facecolor("#0f1117")
ax.plot(clf_t, clf_signal, color="#3b82f6", linewidth=1.2, label="ECG signal", zorder=2)
ax.axhline(0, color="#2e3547", linewidth=0.7, linestyle="--")

if len(r_peaks) > 0:
    peak_times_plot = r_peaks / len(clf_signal) * clf_duration  # correct: divide by total samples
    ax.scatter(peak_times_plot, clf_signal[r_peaks],
               color=bar_color, s=60, zorder=5, label=f"R-peaks ({len(r_peaks)})",
               marker="v")
    # Shade every other beat for readability
    for i in range(len(peak_times_plot) - 1):
        if i % 2 == 0:
            ax.axvspan(peak_times_plot[i], peak_times_plot[i+1],
                       alpha=0.06, color=bar_color)

ax.set_title(
    f"Lead {classify_lead} — R-Peak Detection  |  "
    f"Rhythm: {cls['name']}  |  HR ≈ {hr_str} bpm",
    color="#e2e8f0", fontsize=11
)
ax.set_ylabel("Amplitude (norm.)", color="#7a8499", fontsize=9)
ax.set_ylim(-1.4, 1.5)
ax.tick_params(colors="#7a8499", labelsize=8)
ax.legend(fontsize=9, facecolor="#1a1d27", edgecolor="#2e3547",
          labelcolor="#e2e8f0", loc="upper right")
for spine in ax.spines.values(): spine.set_edgecolor("#2e3547")

# Bottom: RR interval tachogram
ax2 = axes[1]
ax2.set_facecolor("#0f1117")
if features and len(features["rr_intervals"]) > 1:
    rr_ms = features["rr_intervals"] * 1000
    beat_nums = np.arange(1, len(rr_ms) + 1)
    ax2.bar(beat_nums, rr_ms, color=bar_color, alpha=0.75, width=0.7)
    ax2.axhline(np.mean(rr_ms), color="#e2e8f0", linewidth=1,
                linestyle="--", label=f"Mean RR = {np.mean(rr_ms):.0f} ms")
    ax2.set_title("RR Interval Tachogram (beat-to-beat variability)",
                  color="#e2e8f0", fontsize=10)
    ax2.set_xlabel("Beat number", color="#7a8499", fontsize=9)
    ax2.set_ylabel("RR (ms)", color="#7a8499", fontsize=9)
    ax2.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#2e3547", labelcolor="#e2e8f0")
else:
    ax2.text(0.5, 0.5, "Not enough R-peaks for tachogram",
             ha="center", va="center", color="#7a8499",
             transform=ax2.transAxes, fontsize=10)
    ax2.set_title("RR Interval Tachogram", color="#e2e8f0", fontsize=10)
ax2.tick_params(colors="#7a8499", labelsize=8)
for spine in ax2.spines.values(): spine.set_edgecolor("#2e3547")

fig.tight_layout(pad=1.5)
st.pyplot(fig, use_container_width=True)
classification_png = fig_to_bytes(fig)
plt.close(fig)

# ── Clinical note ─────────────────────────────
st.markdown(
    '<div class="warn-box" style="margin-top:12px;">'
    '⚕️ <b>Clinical disclaimer:</b> This classification is generated by a '
    'rule-based algorithm for educational and research purposes only. '
    'It is <b>not a medical diagnosis</b>. Always consult a qualified cardiologist '
    'for clinical ECG interpretation.'
    '</div>',
    unsafe_allow_html=True
)

# ── Updated downloads ─────────────────────────
st.markdown('<div class="step-header"><span>FR6</span> Download Results</div>',
            unsafe_allow_html=True)

# Add classification to CSV
if features:
    df_out["rhythm"]     = cls["name"]
    df_out["hr_bpm"]     = round(features["hr_bpm"], 1)
    df_out["rr_cv"]      = round(features["cv_rr"], 4)
    df_out["confidence"] = round(confidence, 3)

csv_bytes = df_out.to_csv(index=False).encode("utf-8")

dl1, dl2, dl3 = st.columns(3)

with dl1:
    st.download_button(
        label="⬇ Digitized ECG (CSV)",
        data=csv_bytes,
        file_name="ecg_digitized_signal.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(f"{len(df_out):,} rows · lead, time_s, amplitude_norm, rhythm, hr_bpm")

with dl2:
    if comparison_png:
        st.download_button(
            label="⬇ Waveform Plot (PNG)",
            data=comparison_png,
            file_name="ecg_waveform_plot.png",
            mime="image/png",
            use_container_width=True,
        )
        st.caption("Original segment vs reconstructed signal")
    else:
        st.info("Select leads in FR5 to enable.")

with dl3:
    st.download_button(
        label="⬇ Classification Plot (PNG)",
        data=classification_png,
        file_name="ecg_classification_result.png",
        mime="image/png",
        use_container_width=True,
    )
    st.caption(f"R-peak detection + RR tachogram · {cls['name']}")

st.markdown("---")
st.caption(
    "AI-Based ECG Image Digitization System · Final Year Project · "
    "Modules: OpenCV preprocessing → Lead detection → Signal reconstruction → Rhythm classification · "
    "No GPU required"
)
