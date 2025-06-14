import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile
import os
from collections import Counter
import plotly.express as px
import time

# --- Page Configuration ---
st.set_page_config(page_title="YOLOv12n Detection", layout="wide")
st.markdown(
    """
    <style>
        html, body {
            background: linear-gradient(to right, #e0f7fa, #ffffff);
            font-family: 'Roboto', sans-serif;
            color: #003366;
        }
        .title {
            font-size: 2.5em;
            font-weight: bold;
            padding-bottom: 10px;
        }
        .subtitle {
            font-size: 1.2em;
            padding-bottom: 20px;
        }
        .stButton>button {
            background-color: #007acc;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">YOLOv12n Object Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image or video to detect objects using your YOLOv12n model.</div>', unsafe_allow_html=True)

# --- Sidebar Config ---
with st.sidebar:
    st.markdown("## Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    model_choice = st.selectbox("Select Model Version", ["YOLOv12n - Original", "YOLOv12n - Pruned", "YOLOv12n - Final Quantized"])
    input_type = st.selectbox("Select Input Type", ["Image", "Video"])
    st.markdown("---")
    st.markdown("### Detected Labels")

# --- Load Model ---
@st.cache_resource
def load_model(choice):
    if choice == "YOLOv12n - Original":
        return YOLO("best.pt")
    elif choice == "YOLOv12n - Pruned":
        return YOLO("yolo12n_pruned_ultra.pt")
    else:
        return YOLO("dynamic_quantized.onnx")

model = load_model(model_choice)

# --- Helper: Extract Labels ---
def extract_labels(results):
    names = model.names
    return [names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]

# ======================
# IMAGE DETECTION
# ======================
if input_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        start = time.time()
        results = model(img_array, conf=conf_threshold)
        end = time.time()
        annotated_img = results[0].plot()
        inference_time = end - start

        st.image(annotated_img, channels="BGR", caption="Detected Objects", use_column_width=True)

        detected_labels = extract_labels(results)
        with st.sidebar:
            if detected_labels:
                for label in set(detected_labels):
                    st.success(f"{label}")
            else:
                st.info("No objects detected.")

        if detected_labels:
            label_counts = Counter(detected_labels)
            fig = px.bar(
                x=list(label_counts.keys()),
                y=list(label_counts.values()),
                labels={'x': 'Label', 'y': 'Count'},
                title="Class Distribution",
                color=list(label_counts.keys())
            )
            st.plotly_chart(fig, use_container_width=True)

            confidences = results[0].boxes.conf.cpu().numpy()
            avg_conf = np.mean(confidences) * 100

            st.markdown("### Detection Report")
            st.markdown(f"Labels Detected: {', '.join(set(detected_labels))}")
            st.markdown(f"Total Objects Detected: {len(detected_labels)}")
            st.markdown(f"Inference Time: {inference_time:.2f} seconds")
            st.markdown(f"Average Confidence: {avg_conf:.2f}%")
            st.markdown(f"Input Image Size: {img_array.shape[1]} x {img_array.shape[0]}")
            st.markdown(f"Model Used: {model_choice}")

# ======================
# VIDEO DETECTION
# ======================
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        frame_count = 0
        max_frames = 150
        all_detected = []
        all_confidences = []

        progress = st.progress(0, text="Detecting objects in video...")

        start = time.time()
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf_threshold)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

            detected = extract_labels(results)
            all_detected.extend(detected)

            confs = results[0].boxes.conf.cpu().numpy()
            all_confidences.extend(confs)

            frame_count += 1
            progress.progress(min(frame_count / max_frames, 1.0))

        end = time.time()
        cap.release()
        try:
            os.unlink(video_path)
        except Exception as e:
            st.warning(f"Could not delete temp file: {e}")

        unique_labels = list(set(all_detected))
        with st.sidebar:
            if unique_labels:
                for label in unique_labels:
                    st.success(f"{label}")
            else:
                st.info("No objects detected.")

        if all_detected:
            label_counts = Counter(all_detected)
            fig = px.bar(
                x=list(label_counts.keys()),
                y=list(label_counts.values()),
                labels={'x': 'Label', 'y': 'Count'},
                title="Class Distribution",
                color=list(label_counts.keys())
            )
            st.plotly_chart(fig, use_container_width=True)

            avg_conf = np.mean(all_confidences) * 100
            inference_time = end - start

            st.markdown("### Detection Report")
            st.markdown(f"Labels Detected: {', '.join(set(all_detected))}")
            st.markdown(f"Total Objects Detected: {len(all_detected)}")
            st.markdown(f"Total Inference Time: {inference_time:.2f} seconds")
            st.markdown(f"Average Confidence: {avg_conf:.2f}%")
            st.markdown(f"Processed Frames: {frame_count}")
            st.markdown(f"Model Used: {model_choice}")
