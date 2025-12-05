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

# Load model only once
@st.cache_resource
def load_model():
    return YOLO("models/model_finetuning2_part1.pt")

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
    st.sidebar.success(f"Total Deteksi: {len(results[0].boxes)}")
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
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        video_path = tfile.name

        # Preview frame pertama
        cap_preview = cv2.VideoCapture(video_path)
        ret, preview_frame = cap_preview.read()
        cap_preview.release()

        if ret:
            st.image(preview_frame, channels="BGR", caption="Preview Video")

        start_detection = st.button("Mulai Deteksi Video")

        if start_detection:
            stframe = st.empty()
            side_det = st.sidebar.empty()
            side_fps = st.sidebar.empty()

            # Tombol stop
            stop_button = st.sidebar.button("Stop Video")

            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                if stop_button:
                    break

                ret, frame = cap.read()
                if not ret:
                    side_det.warning("Video selesai.")
                    break

                t0 = time.time()
                results = model.predict(frame, conf=confidence, verbose=False)
                annotated_frame = results[0].plot()
                fps = 1 / (time.time() - t0)

                stframe.image(annotated_frame, channels="BGR")
                side_det.success(f"Deteksi: {len(results[0].boxes)}")
                side_fps.info(f"FPS: {fps:.2f}")

                time.sleep(0.01)

            cap.release()


# WEBCAM REAL-TIME
elif source_type == "Webcam":
    run = st.checkbox("Nyalakan Webcam")

    if run:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        stop_webcam = st.sidebar.button("Stop Webcam")

        while cap.isOpened():
            if stop_webcam:
                break

            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            results = model(frame, conf=confidence)
            annotated = results[0].plot()
            fps = 1 / (time.time() - t0)

            stframe.image(annotated, channels="BGR", use_column_width=True)
            st.sidebar.write(f"FPS: {fps:.2f}")

            time.sleep(0.01)

        cap.release()
