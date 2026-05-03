import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = "models/artificial_neural_network.joblib"

st.set_page_config(
    page_title="Bone Fracture AI Assistant",
    page_icon="🦴",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def extract_features(uploaded_image):
    image = Image.open(uploaded_image).convert("L")
    img = np.array(image)

    img = cv2.resize(img, (224, 224))

    mean_intensity = float(np.mean(img))
    std_intensity = float(np.std(img))
    min_intensity = float(np.min(img))
    max_intensity = float(np.max(img))
    contrast = max_intensity - min_intensity

    edges = cv2.Canny(img, 100, 200)
    edge_density = float(np.sum(edges > 0) / edges.size)

    laplacian_variance = float(cv2.Laplacian(img, cv2.CV_64F).var())

    height, width = img.shape
    aspect_ratio = float(width / height)

    return np.array([[
        mean_intensity,
        std_intensity,
        min_intensity,
        max_intensity,
        contrast,
        edge_density,
        laplacian_variance,
        width,
        height,
        aspect_ratio
    ]])

def estimate_cast_duration(age, gender, height, prediction):
    if prediction == 0:
        return "No cast duration estimated because the model predicted Not Fractured."

    base_weeks = 6

    if age < 12:
        base_weeks -= 1
    elif age >= 60:
        base_weeks += 2
    elif age >= 40:
        base_weeks += 1

    if height >= 185:
        base_weeks += 1

    if gender == "Female" and age >= 50:
        base_weeks += 1

    min_weeks = max(3, base_weeks - 1)
    max_weeks = base_weeks + 1

    return f"Estimated cast duration: {min_weeks} to {max_weeks} weeks"

def get_recommendation(prediction):
    if prediction == 1:
        return (
            "The system predicts a possible fracture. "
            "The patient should visit an orthopedic specialist for confirmation, "
            "proper immobilization, and treatment planning."
        )
    return (
        "The system predicts no fracture. However, if pain, swelling, or movement limitation exists, "
        "medical examination is still recommended."
    )

model = load_model()

st.markdown(
    """
    <h1 style='text-align:center;'>🦴 AI Bone Fracture Classification Assistant</h1>
    <p style='text-align:center; font-size:18px;'>
    General Bone Fracture X-ray Classification using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Patient Information")

    age = st.number_input(
        "Patient Age",
        min_value=1,
        max_value=120,
        value=25
    )

    height = st.number_input(
        "Patient Height (cm)",
        min_value=50,
        max_value=230,
        value=170
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    uploaded_file = st.file_uploader(
        "Upload X-ray Image",
        type=["jpg", "jpeg", "png"]
    )

with right_col:
    st.subheader("AI Prediction Result")

    if uploaded_file is None:
        st.info("Please upload an X-ray image to start prediction.")
    else:
        st.image(
            uploaded_file,
            caption="Uploaded X-ray Image",
            use_container_width=True
        )

        features = extract_features(uploaded_file)
        prediction = model.predict(features)[0]

        confidence_text = ""
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities) * 100)
            confidence_text = f"{confidence:.2f}%"

        st.divider()

        if prediction == 1:
            st.error("Prediction: Fractured")
        else:
            st.success("Prediction: Not Fractured")

        if confidence_text:
            st.metric("Model Confidence", confidence_text)

        st.subheader("Estimated Cast Duration")
        st.write(estimate_cast_duration(age, gender, height, prediction))

        st.subheader("Recommendation")
        st.write(get_recommendation(prediction))

        st.warning(
            "Disclaimer: This application is for educational purposes only. "
            "It does not replace professional medical diagnosis."
        )

st.divider()

st.subheader("Model Information")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric("Best Model", "Artificial Neural Network")

with info_col2:
    st.metric("Accuracy", "95.81%")

with info_col3:
    st.metric("Features Used", "10")

st.write(
    "The system extracts numerical image features such as intensity, contrast, "
    "edge density, and Laplacian variance, then uses a trained machine learning model "
    "to classify the X-ray image."
)
