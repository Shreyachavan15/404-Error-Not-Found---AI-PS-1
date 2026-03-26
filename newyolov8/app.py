import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

st.set_page_config(page_title="Plastic Detector", layout="wide")
st.title("♻️ YOLOv8 Plastic Detector")
st.write("Detect plastic types and learn recycling tips!")

# Load YOLOv8 model
model = YOLO("runs/detect/plastic_detector_auto11/weights/best.pt")

# Plastic recycling tips
recycling_tips = {
    "PET": "Plastic bottles, soda bottles\n♻️ Recycle: Curbside recycling, reprocessing into fibers or containers",
    "HDPE": "Milk jugs, detergent bottles\n♻️ Recycle: Curbside recycling, pipelines, plastic lumber",
    "LDPE": "Plastic bags, wrappers\n♻️ Recycle: Drop-off collection, reusable bags, mulch film",
    "PP": "Food containers, caps\n♻️ Recycle: Re-mold into trays, automotive parts, containers",
    "PS": "Cups, cutlery, packaging\n♻️ Recycle: Insulation, light-weight foam, packaging",
    "Others": "Mixed plastics, multi-layer\n♻️ Recycle: Harder to recycle, usually energy recovery"
}

# --- Tabs ---
tab1, tab2 = st.tabs(["Image Upload", "Webcam Stream"])

# ---------- Tab 1: Image Upload ----------
with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(image)
        results = model.predict(img_array, verbose=False, conf=0.25)
        annotated_frame = results[0].plot()

        st.image(annotated_frame, caption="Prediction", use_column_width=True)

        # Show detected classes and recycling tips
        detected_classes = set([model.names[int(box.cls)] for box in results[0].boxes])
        if detected_classes:
            st.subheader("Detected Plastic Types & Recycling Tips")
            for cls in detected_classes:
                st.markdown(f"**{cls}**: {recycling_tips.get(cls,'No info')}")

# ---------- Tab 2: Webcam Stream ----------
with tab2:
    stframe = st.empty()
    start_webcam = st.button("Start Webcam")
    stop_webcam = st.button("Stop Webcam")

    if start_webcam:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, verbose=False, conf=0.25)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            stframe.image(annotated_frame)

            # Exit if Stop button pressed
            if stop_webcam:
                break

        cap.release()
        cv2.destroyAllWindows()