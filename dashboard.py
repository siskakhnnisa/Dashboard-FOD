import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

# Streamlit Page Config
st.set_page_config(
    page_title="FOD Detection Dashboard",
    page_icon="🛰️",
    layout="wide",
)


# Custom CSS
st.markdown("""
    <style>
    .title {
        font-size: 32px;
        font-weight: 600;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #555;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)


# Load Model
@st.cache_resource
def load_model():
    model = YOLO("models/exp1_finetune.pt")
    return model

model = load_model()


# Title UI
st.markdown('<p class="title">🛰️ FOD Detection Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Mendeteksi Foreign Object Debris (FOD) secara otomatis menggunakan YOLOv8</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05)
source_type = st.sidebar.radio("Pilih Input", ["Upload Image", "Upload Video", "Webcam"])


# FUNCTION: Convert frame with detection

def detect_image(img):
    results = model.predict(img, conf=confidence)
    plotted = results[0].plot()
    num_boxes = len(results[0].boxes)
    st.sidebar.success(f"Total Deteksi: {num_boxes}")
    return plotted


# UPLOAD IMAGE

if source_type == "Upload Image":
    img_file = st.file_uploader("Upload gambar untuk dideteksi", type=["jpg", "jpeg", "png"])

    if img_file:
        img = Image.open(img_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Deteksi"):
            with st.spinner("Detecting..."):
                result = detect_image(img)
                st.image(result, caption="Detection Result", use_column_width=True)


# UPLOAD VIDEO
elif source_type == "Upload Video":
    vid_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if vid_file:

        # Simpan video ke file temporer
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        video_path = tfile.name

        # Preview frame pertama
        cap_preview = cv2.VideoCapture(video_path)
        ret, preview_frame = cap_preview.read()
        cap_preview.release()
        if ret:
            st.image(preview_frame, channels="BGR", caption="Preview Video")

        st.markdown("### 🚀 Klik tombol di bawah untuk mulai deteksi video")
        start_detection = st.button("Mulai Deteksi Video")

        if start_detection:

            stframe = st.empty()
            sidebar_det = st.sidebar.empty()
            sidebar_fps = st.sidebar.empty()

            cap = cv2.VideoCapture(video_path)

            while True:
                ret, frame = cap.read()
                if not ret:
                    sidebar_det.warning("Video selesai.")
                    break

                t0 = time.time()

                # YOLO inference
                results = model.predict(frame, conf=confidence, verbose=False)
                annotated_frame = results[0].plot()

                fps = 1 / (time.time() - t0)

                # Render frame stabil
                stframe.image(annotated_frame, channels="BGR")

                sidebar_det.success(f"Deteksi: {len(results[0].boxes)}")
                sidebar_fps.info(f"FPS: {fps:.2f}")

                # Biarkan Streamlit refresh (WAJIB)
                time.sleep(0.001)

                # Force re-render frame-by-frame
                st.experimental_rerun()

# WEBCAM REAL-TIME
elif source_type == "Webcam":
    run = st.checkbox("Nyalakan Webcam")

    if run:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            results = model(frame, conf=confidence)
            annotated = results[0].plot()
            end = time.time()

            fps = 1 / (end - start)

            stframe.image(annotated, channels="BGR", use_column_width=True)
            st.sidebar.write(f"FPS: {fps:.2f}")

            if not st.checkbox("Continous Detection", value=True):
                break

        cap.release()
