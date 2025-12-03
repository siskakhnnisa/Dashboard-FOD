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

        # simpan sebagai file temporary
        if "video_path" not in st.session_state:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(vid_file.read())
            st.session_state.video_path = tfile.name

        st.video(st.session_state.video_path)

        # init state
        if "video_running" not in st.session_state:
            st.session_state.video_running = False

        # tombol kontrol
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Mulai Deteksi Realtime"):
                st.session_state.video_running = True
        with col2:
            if st.button("⏹️ Hentikan"):
                st.session_state.video_running = False

        # tempat menampilkan frame detection
        stframe = st.empty()
        fps_box = st.sidebar.empty()
        det_box = st.sidebar.empty()

        # simpan objek VideoCapture sekali saja!
        if "cap" not in st.session_state:
            st.session_state.cap = cv2.VideoCapture(st.session_state.video_path)

        # LOOP DETEKSI: dilakukan setiap rerun
        if st.session_state.video_running:

            cap = st.session_state.cap
            ret, frame = cap.read()

            # video selesai → reset capture
            if not ret:
                fps_box.warning("Video selesai.")
                st.session_state.cap.release()
                del st.session_state.cap
                st.session_state.video_running = False
                st.stop()

            # deteksi YOLO
            start = time.time()
            results = model.predict(frame, conf=confidence, verbose=False)
            annotated = results[0].plot()
            fps = 1 / (time.time() - start)

            # tampilkan frame hasil deteksi
            stframe.image(annotated, channels="BGR", use_column_width=True)
            fps_box.info(f"FPS Realtime: **{fps:.2f}**")
            det_box.success(f"Jumlah Deteksi: **{len(results[0].boxes)}**")

            # rerun script otomatis tanpa freeze
            time.sleep(0.001)
            st.rerun()
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
