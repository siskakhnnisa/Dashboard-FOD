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
    model = YOLO("models/best.pt")
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
        # Simpan video ke file sementara
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())
        tfile.flush()

        st.video(tfile.name)

        # UI buttons
        cols = st.columns([1, 1, 2])
        with cols[0]:
            start_btn = st.button("🚀 Mulai Deteksi Realtime")
        with cols[1]:
            stop_btn = st.button("⏹️ Stop")
        with cols[2]:
            target_size = st.selectbox(
                "Inference Resolution (lebih kecil = lebih cepat)",
                [480, 640, 800, 960, 1280],
                index=1
            )

        # tempat menampilkan frame
        stframe = st.empty()
        fps_box = st.sidebar.empty()
        det_box = st.sidebar.empty()

        # state
        if "run_video" not in st.session_state:
            st.session_state.run_video = False

        if start_btn:
            st.session_state.run_video = True
        if stop_btn:
            st.session_state.run_video = False

        # PROSES DETEKSI REALTIME
        if st.session_state.run_video:
            cap = cv2.VideoCapture(tfile.name)

            prev_time = time.time()

            while st.session_state.run_video:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame agar inference lebih cepat
                h, w = frame.shape[:2]
                scale = target_size / max(h, w)
                frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

                # Predict
                results = model.predict(
                    frame_resized, 
                    conf=confidence,
                    verbose=False
                )

                annotated = results[0].plot()

                # Hitung FPS
                now = time.time()
                fps = 1 / (now - prev_time)
                prev_time = now

                # Streamlit render
                stframe.image(annotated, channels="BGR", use_column_width=True)
                fps_box.info(f"FPS Realtime: **{fps:.2f}**")
                det_box.success(f"Total Deteksi: **{len(results[0].boxes)}**")

                # Jeda kecil agar UI tidak freeze
                time.sleep(0.001)

            cap.release()
            fps_box.warning("Deteksi dihentikan.")



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
