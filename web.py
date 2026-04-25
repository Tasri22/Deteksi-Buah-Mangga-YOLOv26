import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import tempfile
import time
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mango Detector",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* Reset & Base */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Background */
.stApp {
    background: #0d0d0d;
    color: #f0ece4;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #2a2a2a;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Header */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #f5c842 0%, #f59e0b 50%, #ef7c12 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1;
}

.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #555;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

/* Cards */
.metric-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
}

.metric-card:hover {
    border-color: #f5c842;
}

.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    color: #f5c842;
    line-height: 1;
    font-family: 'DM Mono', monospace;
}

.metric-label {
    font-size: 0.7rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: 0.3rem;
}

/* Detection box */
.detection-item {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #f5c842;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
}

.detection-label {
    color: #f5c842;
    font-weight: 500;
}

.detection-conf {
    color: #888;
    font-size: 0.75rem;
}

/* Conf bar */
.conf-bar-wrap {
    background: #222;
    border-radius: 4px;
    height: 6px;
    margin-top: 0.4rem;
    overflow: hidden;
}

.conf-bar-fill {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, #f5c842, #ef7c12);
    transition: width 0.5s ease;
}

/* Section label */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 0.5rem;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}

.badge-green  { background: #1a2e1a; color: #4ade80; border: 1px solid #2d5a2d; }
.badge-yellow { background: #2e2600; color: #f5c842; border: 1px solid #5a4a00; }
.badge-red    { background: #2e1a1a; color: #f87171; border: 1px solid #5a2d2d; }
.badge-gray   { background: #1a1a1a; color: #888;    border: 1px solid #333; }

/* Uploader area */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #2a2a2a;
    border-radius: 12px;
    background: #111;
    padding: 0.5rem;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #f5c842;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #f5c842, #ef7c12) !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.4rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: #f5c842 !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #161616 !important;
    border-color: #2a2a2a !important;
    color: #f0ece4 !important;
}

/* Info box */
.info-box {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #888;
    line-height: 1.7;
}

.info-box span { color: #f5c842; }

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 1.5rem 0;
}

/* No detection */
.no-detect {
    text-align: center;
    padding: 2rem;
    color: #333;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_resource
def load_model(model_path):
    """Load YOLO model — cached agar tidak reload setiap interaksi"""
    return YOLO(model_path)


def get_conf_badge(conf):
    if conf >= 0.80:
        return f'<span class="badge badge-green">HIGH {conf:.0%}</span>'
    elif conf >= 0.50:
        return f'<span class="badge badge-yellow">MED {conf:.0%}</span>'
    else:
        return f'<span class="badge badge-red">LOW {conf:.0%}</span>'


def run_inference(model, image_np, conf_thresh, iou_thresh):
    """Jalankan inferensi YOLO dan kembalikan hasil"""
    results = model.predict(
        source=image_np,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False
    )
    return results[0]


def draw_detections(image_np, result, box_color=(245, 200, 66)):
    """Gambar bounding box & label ke gambar dengan style custom"""
    img = image_np.copy()
    h, w = img.shape[:2]

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        label = result.names[cls]

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

        # Label background
        label_text = f"{label}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), box_color, -1)

        # Label text
        cv2.putText(img, label_text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (13, 13, 13), 1, cv2.LINE_AA)

    return img


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="section-label">⚙ Konfigurasi</p>', unsafe_allow_html=True)

    # Upload model
    st.markdown("**Model YOLO**")
    model_file = st.file_uploader(
        "Upload file .pt",
        type=["pt"],
        label_visibility="collapsed"
    )

    # Atau gunakan path
    model_path_input = st.text_input(
        "Atau masukkan path model:",
        placeholder="best.pt",
        label_visibility="visible"
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">🎛 Parameter Deteksi</p>', unsafe_allow_html=True)

    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.95,
        value=0.25, step=0.05,
        help="Semakin tinggi = hanya deteksi yang yakin saja yang ditampilkan"
    )

    iou_thresh = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.10, max_value=0.95,
        value=0.45, step=0.05,
        help="Non-Maximum Suppression — kurangi deteksi duplikat"
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">🎨 Tampilan</p>', unsafe_allow_html=True)

    show_original = st.toggle("Tampilkan Gambar Original", value=True)
    show_details  = st.toggle("Tampilkan Detail Deteksi", value=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <span>Format gambar</span>: JPG, PNG, WEBP<br>
        <span>Format video</span>: MP4, AVI, MOV<br>
        <span>Model</span>: YOLOv8 (.pt)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

model = None

if model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        tmp_path = tmp.name
    try:
        model = load_model(tmp_path)
        st.sidebar.success(f"✓ Model loaded")
    except Exception as e:
        st.sidebar.error(f"Gagal load model: {e}")

elif model_path_input and os.path.exists(model_path_input):
    try:
        model = load_model(model_path_input)
        st.sidebar.success(f"✓ Model loaded")
    except Exception as e:
        st.sidebar.error(f"Gagal load model: {e}")

elif model_path_input == "" and os.path.exists("best.pt"):
    # Auto-load jika best.pt ada di direktori yang sama
    try:
        model = load_model("best.pt")
        st.sidebar.success("✓ Auto-loaded: best.pt")
    except:
        pass


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">🥭 Mango Detector</h1>
    <p class="hero-sub">YOLOv8 · Object Detection · Testing System</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📷 Deteksi Gambar", "🎬 Deteksi Video", "📊 Batch Testing"])


# ════════════════════════════════════════════
# TAB 1 — DETEKSI GAMBAR
# ════════════════════════════════════════════

with tab1:

    if model is None:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#333;">
            <div style="font-size:3rem;">🥭</div>
            <p style="font-family:'DM Mono',monospace; font-size:0.85rem; margin-top:1rem;">
                Upload model <b style="color:#f5c842">.pt</b> di sidebar untuk mulai
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        uploaded_imgs = st.file_uploader(
            "Upload gambar untuk dideteksi",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_imgs:
            for uploaded_file in uploaded_imgs:
                st.markdown(f'<p class="section-label">📁 {uploaded_file.name}</p>',
                            unsafe_allow_html=True)

                # Load gambar
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Inferensi
                with st.spinner("Mendeteksi..."):
                    t_start = time.time()
                    result  = run_inference(model, img_rgb, conf_thresh, iou_thresh)
                    t_infer = (time.time() - t_start) * 1000

                # Gambar dengan bounding box
                img_drawn = draw_detections(img_rgb, result)
                n_detect  = len(result.boxes)

                # Layout: original | hasil
                if show_original:
                    col_orig, col_res = st.columns(2)
                    with col_orig:
                        st.markdown('<p class="section-label">Original</p>',
                                    unsafe_allow_html=True)
                        st.image(img_rgb, use_column_width=True)
                    with col_res:
                        st.markdown('<p class="section-label">Hasil Deteksi</p>',
                                    unsafe_allow_html=True)
                        st.image(img_drawn, use_column_width=True)
                else:
                    st.image(img_drawn, use_column_width=True)

                # Metrics row
                h_img, w_img = img_rgb.shape[:2]
                confs = [float(b.conf[0]) for b in result.boxes]
                avg_conf = sum(confs) / len(confs) if confs else 0

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{n_detect}</div>
                        <div class="metric-label">Mangga Terdeteksi</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{avg_conf:.0%}</div>
                        <div class="metric-label">Avg Confidence</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{t_infer:.0f}</div>
                        <div class="metric-label">Waktu (ms)</div>
                    </div>""", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{w_img}×{h_img}</div>
                        <div class="metric-label">Resolusi</div>
                    </div>""", unsafe_allow_html=True)

                # Detail deteksi
                if show_details:
                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    st.markdown('<p class="section-label">📋 Detail Setiap Deteksi</p>',
                                unsafe_allow_html=True)

                    if n_detect == 0:
                        st.markdown("""
                        <div class="no-detect">
                            Tidak ada mangga terdeteksi.<br>
                            Coba turunkan Confidence Threshold di sidebar.
                        </div>""", unsafe_allow_html=True)
                    else:
                        for i, box in enumerate(result.boxes):
                            conf  = float(box.conf[0])
                            cls   = int(box.cls[0])
                            label = result.names[cls]
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            bw = x2 - x1
                            bh = y2 - y1
                            badge_html = get_conf_badge(conf)

                            st.markdown(f"""
                            <div class="detection-item">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span class="detection-label">#{i+1} — {label.upper()}</span>
                                    {badge_html}
                                </div>
                                <div class="detection-conf" style="margin-top:0.3rem;">
                                    Posisi: ({x1}, {y1}) → ({x2}, {y2}) &nbsp;|&nbsp;
                                    Ukuran: {bw}×{bh}px
                                </div>
                                <div class="conf-bar-wrap">
                                    <div class="conf-bar-fill" style="width:{conf*100:.1f}%"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # Download hasil
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                result_pil = Image.fromarray(img_drawn)
                buf = io.BytesIO()
                result_pil.save(buf, format="JPEG", quality=95)
                st.download_button(
                    label="⬇ Download Hasil Deteksi",
                    data=buf.getvalue(),
                    file_name=f"detected_{uploaded_file.name}",
                    mime="image/jpeg"
                )

                st.markdown('<hr class="divider" style="margin-top:2rem">', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem; color:#333; border:1.5px dashed #222; border-radius:12px;">
                <div style="font-size:2.5rem;">📸</div>
                <p style="font-family:'DM Mono',monospace; font-size:0.82rem; margin-top:0.8rem;">
                    Drag & drop gambar ke sini<br>atau klik untuk memilih file
                </p>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2 — DETEKSI VIDEO
# ════════════════════════════════════════════

with tab2:

    if model is None:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#333;">
            <p style="font-family:'DM Mono',monospace; font-size:0.85rem;">
                Upload model <b style="color:#f5c842">.pt</b> di sidebar terlebih dahulu
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        uploaded_video = st.file_uploader(
            "Upload video",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed"
        )

        if uploaded_video:
            # Simpan video ke temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.flush()

            col_cfg, col_info = st.columns([2, 1])
            with col_cfg:
                max_frames = st.slider(
                    "Maksimal frame yang diproses",
                    min_value=10, max_value=500, value=100, step=10
                )
                frame_skip = st.slider(
                    "Proses setiap N frame (frame skip)",
                    min_value=1, max_value=10, value=2,
                    help="1 = semua frame, 2 = setiap 2 frame, dst"
                )

            with col_info:
                cap_info = cv2.VideoCapture(tfile.name)
                total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_video    = cap_info.get(cv2.CAP_PROP_FPS)
                w_vid = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
                h_vid = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_info.release()

                st.markdown(f"""
                <div class="info-box" style="margin-top:1.5rem;">
                    <span>Total frame</span>: {total_frames}<br>
                    <span>FPS</span>: {fps_video:.1f}<br>
                    <span>Resolusi</span>: {w_vid}×{h_vid}<br>
                    <span>Durasi</span>: {total_frames/fps_video:.1f}s
                </div>
                """, unsafe_allow_html=True)

            if st.button("▶ Mulai Deteksi Video", key="btn_video"):
                cap = cv2.VideoCapture(tfile.name)

                frame_placeholder = st.empty()
                progress_bar      = st.progress(0)
                stats_placeholder = st.empty()

                frame_count    = 0
                process_count  = 0
                total_detected = 0
                detection_log  = []

                while cap.isOpened() and process_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    # Frame skip
                    if frame_count % frame_skip != 0:
                        continue

                    process_count += 1
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Inferensi
                    t0     = time.time()
                    result = run_inference(model, frame_rgb, conf_thresh, iou_thresh)
                    t_ms   = (time.time() - t0) * 1000
                    n_det  = len(result.boxes)
                    total_detected += n_det

                    # Gambar hasil
                    frame_drawn = draw_detections(frame_rgb, result)

                    # Update display
                    frame_placeholder.image(frame_drawn, use_column_width=True)

                    # Update progress
                    prog = min(process_count / max_frames, 1.0)
                    progress_bar.progress(prog)

                    # Update stats
                    detection_log.append(n_det)
                    avg_det = sum(detection_log) / len(detection_log)

                    stats_placeholder.markdown(f"""
                    <div style="display:flex; gap:1rem; margin-top:0.5rem;">
                        <div class="metric-card" style="flex:1">
                            <div class="metric-value">{process_count}</div>
                            <div class="metric-label">Frame Diproses</div>
                        </div>
                        <div class="metric-card" style="flex:1">
                            <div class="metric-value">{n_det}</div>
                            <div class="metric-label">Deteksi Frame Ini</div>
                        </div>
                        <div class="metric-card" style="flex:1">
                            <div class="metric-value">{total_detected}</div>
                            <div class="metric-label">Total Deteksi</div>
                        </div>
                        <div class="metric-card" style="flex:1">
                            <div class="metric-value">{1000/t_ms:.0f}</div>
                            <div class="metric-label">FPS Inferensi</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                cap.release()
                st.success(f"✅ Selesai! Diproses {process_count} frame | "
                           f"Total deteksi: {total_detected} | "
                           f"Rata-rata per frame: {total_detected/process_count:.1f}")


# ════════════════════════════════════════════
# TAB 3 — BATCH TESTING
# ════════════════════════════════════════════

with tab3:

    if model is None:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#333;">
            <p style="font-family:'DM Mono',monospace; font-size:0.85rem;">
                Upload model <b style="color:#f5c842">.pt</b> di sidebar terlebih dahulu
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<p class="section-label">📂 Upload Folder Test (multiple files)</p>',
                    unsafe_allow_html=True)

        batch_files = st.file_uploader(
            "Upload semua gambar test sekaligus",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if batch_files:
            st.markdown(f"""
            <div class="info-box" style="margin-bottom:1rem;">
                <span>{len(batch_files)}</span> gambar siap diproses
            </div>
            """, unsafe_allow_html=True)

            if st.button("🚀 Jalankan Batch Testing", key="btn_batch"):
                progress     = st.progress(0)
                status_text  = st.empty()
                results_data = []

                for i, f in enumerate(batch_files):
                    status_text.markdown(
                        f'<p class="section-label">Memproses {i+1}/{len(batch_files)}: {f.name}</p>',
                        unsafe_allow_html=True
                    )

                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    t0     = time.time()
                    result = run_inference(model, img_rgb, conf_thresh, iou_thresh)
                    t_ms   = (time.time() - t0) * 1000

                    n_det  = len(result.boxes)
                    confs  = [float(b.conf[0]) for b in result.boxes]
                    avg_c  = sum(confs) / len(confs) if confs else 0
                    max_c  = max(confs) if confs else 0

                    results_data.append({
                        "Nama File"      : f.name,
                        "Deteksi"        : n_det,
                        "Avg Conf"       : f"{avg_c:.2%}",
                        "Max Conf"       : f"{max_c:.2%}",
                        "Waktu (ms)"     : f"{t_ms:.1f}",
                        "Status"         : "✅ Terdeteksi" if n_det > 0 else "❌ Tidak Terdeteksi"
                    })

                    progress.progress((i + 1) / len(batch_files))

                status_text.empty()
                progress.empty()

                # ─── Ringkasan ───
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown('<p class="section-label">📊 Ringkasan Hasil Batch</p>',
                            unsafe_allow_html=True)

                total_img  = len(results_data)
                detected   = sum(1 for r in results_data if r["Deteksi"] > 0)
                not_detect = total_img - detected
                total_det  = sum(r["Deteksi"] for r in results_data)
                det_rate   = detected / total_img if total_img > 0 else 0

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{total_img}</div>
                        <div class="metric-label">Total Gambar</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:#4ade80">{detected}</div>
                        <div class="metric-label">Terdeteksi</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:#f87171">{not_detect}</div>
                        <div class="metric-label">Tidak Terdeteksi</div>
                    </div>""", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{det_rate:.0%}</div>
                        <div class="metric-label">Detection Rate</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                # ─── Tabel hasil ───
                st.markdown('<p class="section-label">📋 Tabel Hasil Per Gambar</p>',
                            unsafe_allow_html=True)
                st.dataframe(
                    results_data,
                    use_container_width=True,
                    hide_index=True
                )

                # ─── Download CSV ───
                import csv
                csv_buf = io.StringIO()
                writer  = csv.DictWriter(csv_buf, fieldnames=results_data[0].keys())
                writer.writeheader()
                writer.writerows(results_data)

                st.download_button(
                    label="⬇ Download Hasil sebagai CSV",
                    data=csv_buf.getvalue(),
                    file_name="batch_results.csv",
                    mime="text/csv"
                )

        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem; color:#333;
                        border:1.5px dashed #222; border-radius:12px;">
                <div style="font-size:2.5rem;">📂</div>
                <p style="font-family:'DM Mono',monospace; font-size:0.82rem; margin-top:0.8rem;">
                    Upload semua gambar test sekaligus<br>
                    untuk diproses secara batch
                </p>
            </div>
            """, unsafe_allow_html=True)
