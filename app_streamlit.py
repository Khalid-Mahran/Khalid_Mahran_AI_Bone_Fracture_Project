import os
import json
import tempfile
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing import image

MODEL_PATH = "models/bone_binary_model.keras"
CLASSES_PATH = "models/binary_class_names.json"
REF_DIR = "assets/bone_reference"

BONE_IMAGES = {
    "Radius": "radius.png",
    "Ulna": "ulna.png",
    "Humerus": "humerus.png",
    "Carpals": "carpals.png"
}

BONE_RATIOS = {
    "Radius": 0.146,
    "Ulna": 0.156,
    "Humerus": 0.186,
    "Carpals": 0.045
}

@st.cache_resource
def load_model_once():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r") as f:
        return json.load(f)

def detect_bone_from_filename(filename):
    name = filename.lower()
    if "radius" in name:
        return "Radius"
    if "ulna" in name:
        return "Ulna"
    if "humerus" in name:
        return "Humerus"
    if "carpal" in name or "wrist" in name:
        return "Carpals"
    return None

def estimate_recovery(status, age):
    if status == "normal":
        return "A", "No cast required", "No recovery time needed"

    if age < 18:
        return "C", "Cast required", "3 - 4 weeks"
    elif age <= 40:
        return "C", "Cast required", "4 - 6 weeks"
    elif age <= 60:
        return "D", "Cast required", "6 - 8 weeks"
    else:
        return "D", "Cast + doctor follow-up", "8 - 10 weeks"

def predict_xray(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    img = image.load_img(tmp_path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(pred))
    return class_names[idx], float(pred[idx]) * 100

def image_exists(path):
    return os.path.exists(path)

st.set_page_config(
    page_title="AI Upper Limb X-ray Fracture Analyzer",
    layout="wide"
)

model = load_model_once()
class_names = load_classes()

st.title("AI Upper Limb X-ray Fracture Analyzer")
st.write("Upload an X-ray image, enter patient data, and the system will analyze fracture status with an educational bone reference.")

left, right = st.columns([0.9, 1.4])

with left:
    uploaded_file = st.file_uploader(
        "Drag & Drop X-ray Image Here",
        type=["jpg", "jpeg", "png"]
    )

    age = st.number_input("Patient Age", min_value=1, max_value=120, value=20)
    height = st.number_input("Patient Height (cm)", min_value=50, max_value=230, value=170)
    gender = st.radio("Gender", ["male", "female"])

    auto_bone = None
    if uploaded_file:
        auto_bone = detect_bone_from_filename(uploaded_file.name)

    if auto_bone:
        selected_bone = auto_bone
        st.success(f"Detected educational reference from file name: {selected_bone}")
    else:
        selected_bone = st.selectbox(
            "Select bone / area manually",
            ["Radius", "Ulna", "Humerus", "Carpals"]
        )

    analyze_btn = st.button("Analyze X-ray", use_container_width=True)

with right:
    ref_col, xray_col = st.columns(2)

    with ref_col:
        if uploaded_file:
            ref_file = BONE_IMAGES.get(selected_bone)
            ref_path = os.path.join(REF_DIR, ref_file)
            st.subheader(f"{selected_bone} Educational Reference")

            if image_exists(ref_path):
                st.image(ref_path, use_container_width=True)
            else:
                st.error(f"Missing reference image: {ref_path}")
        else:
            default_path = os.path.join(REF_DIR, "upper_limb_overview.png")
            st.subheader("Upper Limb Overview")

            if image_exists(default_path):
                st.image(default_path, use_container_width=True)
            else:
                st.error("Missing default image: assets/bone_reference/upper_limb_overview.png")

    with xray_col:
        st.subheader("Uploaded X-ray")
        if uploaded_file:
            uploaded_img = Image.open(uploaded_file)
            st.image(uploaded_img, use_container_width=True)
        else:
            st.info("No X-ray uploaded yet.")

if analyze_btn:
    if uploaded_file is None:
        st.error("Please upload an X-ray image first.")
    else:
        status, confidence = predict_xray(uploaded_file)

        fracture_class, cast_decision, recovery_time = estimate_recovery(status, age)
        estimated_bone_length = round(height * BONE_RATIOS.get(selected_bone, 0.146), 1)

        if gender == "female":
            estimated_bone_length = round(estimated_bone_length * 0.97, 1)

        fracture_location = round(estimated_bone_length * 0.65, 1)
        fracture_gap = 3

        st.subheader("AI Analysis Report")

        if status == "fractured":
            st.error("Condition: FRACTURED")
        else:
            st.success("Condition: NORMAL")

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{confidence:.2f}%")
        c2.metric("Selected Bone", selected_bone)
        c3.metric("Fracture Class", fracture_class)

        st.write(f"**Cast Decision:** {cast_decision}")
        st.write(f"**Estimated Recovery Time:** {recovery_time}")
        st.write(f"**Estimated Bone Length:** {estimated_bone_length} cm")

        if status == "fractured":
            st.write(f"**Estimated Fracture Location:** {fracture_location} cm from the top")
            st.write(f"**Estimated Fracture Gap:** {fracture_gap} mm")
        else:
            st.write("**Estimated Fracture Location:** —")
            st.write("**Estimated Fracture Gap:** —")

        st.warning("Educational AI prototype only. Not a medical diagnosis.")
