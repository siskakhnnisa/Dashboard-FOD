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
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())
        tfile.flush()

        st.video(tfile.name)

        # session_state control
        if "video_running" not in st.session_state:
            st.session_state.video_running = False

        cols = st.columns([1, 1, 1])
        with cols[0]:
            if st.button("🚀 Jalankan Deteksi Video (Realtime)"):
                st.session_state.video_running = True
        with cols[1]:
            if st.button("⏹️ Hentikan Deteksi"):
                st.session_state.video_running = False
        with cols[2]:
            # optional: choose target inference size for speed
            target_size = st.selectbox("Max side (px) untuk inference", [640, 800, 960, 1280], index=0)

        stframe = st.empty()
        info_placeholder = st.sidebar.empty()

        if st.session_state.video_running:
            try:
                prev_time = None
                # Let YOLO iterate over the video file frames (internal streaming)
                results = model(source=tfile.name, conf=confidence, stream=True, imgsz=target_size)
                for r in results:
                    # if user stopped from UI, break gracefully
                    if not st.session_state.video_running:
                        break

                    # r is a Results object for one frame
                    annotated = r.plot()  # BGR image
                    # fps calc
                    now = time.time()
                    if prev_time is None:
                        fps = 0.0
                    else:
                        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
                    prev_time = now

                    # display
                    stframe.image(annotated, channels="BGR", use_column_width=True)

                    info_placeholder.write(f"FPS: {fps:.2f}  |  Deteksi: {len(r.boxes)}")

                    # tiny sleep to yield control back to Streamlit (prevents UI freeze)
                    time.sleep(0.001)

            except Exception as e:
                st.error(f"Terjadi error saat deteksi video: {e}")
            finally:
                st.session_state.video_running = False
                info_placeholder.write("Selesai.")



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
