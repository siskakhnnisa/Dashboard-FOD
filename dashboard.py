import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

st.set_page_config(
    page_title="FOD Detection Dashboard",
    page_icon="🛰️",
    layout="wide",
)

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

@st.cache_resource
def load_model():
    return YOLO("models/exp1_finetune.pt")

model = load_model()

def detect_image(img, conf):
    results = model.predict(img, conf=conf)
    plotted = results[0].plot()
    num_boxes = len(results[0].boxes)
    st.sidebar.success(f"Total Deteksi: {num_boxes}")
    return plotted

st.markdown('<p class="title">🛰️ FOD Detection Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Mendeteksi Foreign Object Debris (FOD) secara otomatis menggunakan YOLOv8</p>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Pengaturan")
confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05)
source_type = st.sidebar.radio("Pilih Input", ["Upload Image", "Upload Video", "Webcam"])


@st.cache_resource
def load_model1():
    return YOLO("models/model1.pt")

@st.cache_resource
def load_model2():
    return YOLO("models/model2.pt")

@st.cache_resource
def load_model3():
    return YOLO("models/model3.pt")

# UPLOAD IMAGE
if source_type == "Upload Image":
    img_file = st.file_uploader("Upload gambar untuk dideteksi", type=["jpg", "jpeg", "png"])

    if img_file:
        img = Image.open(img_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # 🔥 Tambahan: Pilihan model
        model_choice = st.radio(
            "Pilih Model Deteksi",
            ("Model 1", "Model 2", "Model 3"),
            horizontal=True
        )

        # 🔥 Mapping model
        if model_choice == "Model 1":
            selected_model = load_model1()
        elif model_choice == "Model 2":
            selected_model = load_model2()
        else:
            selected_model = load_model3()

        # Tombol Deteksi
        if st.button("🔍 Deteksi dengan " + model_choice):
            with st.spinner("Detecting..."):

                results = selected_model.predict(img, conf=confidence)
                plotted = results[0].plot()

                st.image(plotted, caption=f"Hasil Deteksi ({model_choice})", use_column_width=True)
                st.sidebar.success(f"Total Deteksi: {len(results[0].boxes)}")


elif source_type == "Upload Video":
    vid_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        ret, preview_frame = cap.read()
        cap.release()

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
                results = model.predict(frame, conf=confidence, verbose=False)
                annotated_frame = results[0].plot()
                fps = 1 / (time.time() - t0)

                stframe.image(annotated_frame, channels="BGR")
                sidebar_det.success(f"Deteksi: {len(results[0].boxes)}")
                sidebar_fps.info(f"FPS: {fps:.2f}")
                time.sleep(0.001)

            st.experimental_rerun()

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
            fps = 1 / (time.time() - start)

            stframe.image(annotated, channels="BGR", use_column_width=True)
            st.sidebar.write(f"FPS: {fps:.2f}")

            if not st.checkbox("Continous Detection", value=True):
                break

        cap.release()
