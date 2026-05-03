import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = "models/artificial_neural_network.joblib"

st.set_page_config(page_title="Bone Fracture AI", layout="centered")

st.title("AI Bone Fracture Classification")
st.write("Radius, Ulna, and Humerus X-ray Classification")
st.write("Upload an X-ray image to predict whether it is fractured or not fractured.")

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

model = load_model()

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded X-ray Image", use_container_width=True)

    features = extract_features(uploaded_file)
    prediction = model.predict(features)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Fractured")
    else:
        st.success("Not Fractured")

    st.write("The model extracted 10 numerical features from the uploaded image.")
