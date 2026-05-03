import os
from io import BytesIO
from datetime import datetime

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportImage,
)

# =========================
# Python: file paths
# =========================
MODEL_PATH = "models/artificial_neural_network.joblib"
LOGO_PATH = "assets/logo.png"


# =========================
# Python: load trained model
# =========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# =========================
# Python: open image safely from uploaded bytes
# =========================
def get_rgb_gray(file_bytes):
    pil_img = Image.open(BytesIO(file_bytes)).convert("RGB")
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


# =========================
# Python: reject non-Xray images
# logic:
# - X-ray should be mostly grayscale
# - low saturation
# - low color-channel differences
# =========================
def is_valid_xray(file_bytes):
    try:
        rgb, gray = get_rgb_gray(file_bytes)

        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        b = rgb[:, :, 2].astype(np.float32)

        channel_diff = np.mean((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3.0)

        pixel_spread = (rgb.max(axis=2) - rgb.min(axis=2))
        grayscale_ratio = float(np.mean(pixel_spread < 18))

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mean_saturation = float(np.mean(hsv[:, :, 1]))

        # heuristic thresholds
        if grayscale_ratio < 0.72:
            return False, f"Rejected: image is too colorful / not grayscale enough (grayscale ratio={grayscale_ratio:.2f})"

        if mean_saturation > 45:
            return False, f"Rejected: image saturation is too high for an X-ray (mean saturation={mean_saturation:.1f})"

        if channel_diff > 22:
            return False, f"Rejected: image does not look like a grayscale X-ray (color difference={channel_diff:.1f})"

        return True, "Valid X-ray-like image"

    except Exception:
        return False, "Rejected: unsupported or corrupted image"


# =========================
# Python + OpenCV: extract 10 features
# =========================
def extract_features(file_bytes):
    _, gray = get_rgb_gray(file_bytes)
    img = cv2.resize(gray, (224, 224))

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


# =========================
# Python: educational bone image mapping
# =========================
def get_bone_image_path(bone_type):
    mapping = {
        "Radius": "assets/bone_reference/radius.png",
        "Ulna": "assets/bone_reference/ulna.png",
        "Humerus": "assets/bone_reference/humerus.png",
        "Forearm": "assets/bone_reference/forearm.png",
        "Wrist": "assets/bone_reference/wrist.png",
    }
    return mapping.get(bone_type, "assets/bone_reference/radius.png")


# =========================
# Python: support text
# =========================
def get_recommendation(prediction_label):
    if prediction_label == "Fractured":
        return "Suggested: Orthopedic consultation, immobilization, pain relief support, and follow-up imaging if needed."
    return "No fracture detected. Rest and monitor symptoms."


# =========================
# Python: estimated cast duration
# =========================
def estimate_cast_duration(age, bone_type, prediction_label):
    if prediction_label != "Fractured":
        return "No cast duration required."

    base_weeks = 6

    if bone_type == "Humerus":
        base_weeks += 2
    elif bone_type == "Ulna":
        base_weeks += 1
    elif bone_type == "Forearm":
        base_weeks += 1

    if age < 12:
        base_weeks -= 1
    elif age >= 60:
        base_weeks += 2
    elif age >= 40:
        base_weeks += 1

    min_weeks = max(3, base_weeks - 1)
    max_weeks = base_weeks + 1
    return f"{min_weeks} to {max_weeks} weeks"


# =========================
# Python: build data for PDF
# =========================
def build_pdf_data(patient_name, phone, doctor_name, age, height_cm, gender, bone_type,
                   result, confidence, recommendation, cast_duration):
    return {
        "hospital": "Arab Academy Maritime Hospital",
        "doctor_name": doctor_name,
        "patient_name": patient_name if patient_name else "Not entered",
        "phone": phone if phone else "Not entered",
        "age": str(age),
        "height": f"{height_cm} cm",
        "gender": gender,
        "bone_type": bone_type,
        "result": result,
        "confidence": f"{confidence:.2f}%",
        "recommendation": recommendation,
        "cast_duration": cast_duration,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# =========================
# Python + ReportLab: generate PDF
# =========================
def make_pdf(pdf_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = []

    header_row = []

    if os.path.exists(LOGO_PATH):
        logo = ReportImage(LOGO_PATH, width=70, height=70)
        header_row.append([
            logo,
            Paragraph(
                "<b>AI-OrthoCare</b><br/>Integrated Fracture Detection & Prescription System<br/><b>Arab Academy Maritime Hospital</b>",
                styles["Title"]
            )
        ])
    else:
        header_row.append([
            "",
            Paragraph(
                "<b>AI-OrthoCare</b><br/>Integrated Fracture Detection & Prescription System<br/><b>Arab Academy Maritime Hospital</b>",
                styles["Title"]
            )
        ])

    header_table = Table(header_row, colWidths=[85, 410])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(header_table)

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Doctor-Verified Prescription Support Draft</b>", styles["Heading2"]))
    story.append(Spacer(1, 10))

    info_table = Table([
        ["Doctor", pdf_data["doctor_name"], "Date & Time", pdf_data["timestamp"]],
        ["Patient Name", pdf_data["patient_name"], "Phone", pdf_data["phone"]],
        ["Age", pdf_data["age"], "Gender", pdf_data["gender"]],
        ["Height", pdf_data["height"], "Bone Type", pdf_data["bone_type"]],
        ["AI Result", pdf_data["result"], "Confidence", pdf_data["confidence"]],
        ["Estimated Cast Duration", pdf_data["cast_duration"], "", ""],
    ], colWidths=[120, 150, 120, 145])

    info_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(info_table)

    story.append(Spacer(1, 16))
    story.append(Paragraph("<b>AI Recommendation</b>", styles["Heading3"]))
    story.append(Paragraph(pdf_data["recommendation"], styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Suggested Doctor-Reviewed Plan</b>", styles["Heading3"]))

    if pdf_data["result"] == "Fractured":
        plan_lines = [
            "1. Orthopedic consultation is recommended.",
            "2. Immobilization / splint / cast as clinically indicated by the doctor.",
            "3. Pain relief medication only as prescribed by the doctor.",
            "4. Rest and avoid pressure or heavy movement on the affected limb.",
            "5. Follow-up appointment is recommended to monitor healing progress.",
            "6. Urgent review is required if severe pain, numbness, swelling, color change, or loss of movement occurs.",
        ]
    else:
        plan_lines = [
            "1. No fracture detected by the AI model.",
            "2. Rest and monitor symptoms.",
            "3. Pain relief medication only if prescribed by the doctor.",
            "4. Avoid heavy use of the affected limb for a short observation period.",
            "5. Review is recommended if pain, swelling, or limited movement continues.",
        ]

    for line in plan_lines:
        story.append(Paragraph(line, styles["BodyText"]))

    story.append(Spacer(1, 16))
    story.append(Paragraph("<b>Medical Disclaimer</b>", styles["Heading3"]))
    story.append(Paragraph(
        "This system supports the doctor and does not replace medical judgment. "
        "The generated prescription support must be reviewed, edited, and approved by a qualified doctor.",
        styles["BodyText"]
    ))

    story.append(Spacer(1, 26))
    sig = Table([
        ["Doctor Signature", "________________________"],
        ["Doctor Name", pdf_data["doctor_name"]],
    ], colWidths=[150, 320])
    sig.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(sig)

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# =========================
# Python + Streamlit: app config
# =========================
st.set_page_config(page_title="AI-OrthoCare", page_icon="🦴", layout="wide")

# =========================
# HTML/CSS inside Python: GUI styling
# =========================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #071226 45%, #0b1120 100%);
        color: #f8fafc;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 95rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: #f8fafc !important;
    }

    .main-title {
        font-size: 2.7rem;
        font-weight: 850;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #cbd5e1 !important;
        font-size: 1.02rem;
        margin-bottom: 1rem;
    }

    .panel-title {
        font-size: 1.05rem;
        font-weight: 750;
        margin-bottom: 0.6rem;
    }

    .result-good {
        background: #052e16;
        border: 1px solid #22c55e;
        color: #bbf7d0 !important;
        border-radius: 12px;
        padding: 14px;
        font-weight: 800;
        margin-top: 0.8rem;
    }

    .result-bad {
        background: #3b0d0d;
        border: 1px solid #ef4444;
        color: #fecaca !important;
        border-radius: 12px;
        padding: 14px;
        font-weight: 800;
        margin-top: 0.8rem;
    }

    .warn-box {
        background: #422006;
        border: 1px solid #f59e0b;
        color: #fde68a !important;
        border-radius: 12px;
        padding: 12px;
        margin-top: 1rem;
    }

    .metric-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        margin-bottom: 0.6rem;
    }

    .metric-title {
        color: #94a3b8 !important;
        font-size: 0.92rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f8fafc !important;
    }

    .section-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 16px;
        margin-top: 0.6rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 46px;
        background: #0f172a;
        color: white;
        border: 1px solid #475569;
        font-size: 16px;
        font-weight: 500;
    }

    .stButton > button:hover {
        border: 1px solid #60a5fa;
        color: #dbeafe;
    }
</style>
""", unsafe_allow_html=True)

model = load_model()

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "prediction_label" not in st.session_state:
    st.session_state.prediction_label = None
if "confidence_value" not in st.session_state:
    st.session_state.confidence_value = 0.0
if "recommendation" not in st.session_state:
    st.session_state.recommendation = ""
if "cast_duration" not in st.session_state:
    st.session_state.cast_duration = ""
if "report_time" not in st.session_state:
    st.session_state.report_time = ""
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "validation_msg" not in st.session_state:
    st.session_state.validation_msg = ""


# =========================
# Header
# =========================
st.markdown('<div class="main-title">AI-OrthoCare: Integrated Fracture Detection & Prescription System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-assisted fracture detection with doctor-verified prescription support and educational anatomy reference.</div>',
    unsafe_allow_html=True
)

# =========================
# Top Layout: 3 columns
# =========================
left, center, right = st.columns([1.1, 0.9, 1.05], gap="large")

with left:
    st.markdown('<div class="panel-title">Patient & X-ray Input</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])
    patient_name = st.text_input("Patient Name")
    patient_phone = st.text_input("Patient Phone")
    doctor_name = st.text_input("Doctor Name", value="Dr. Mohamed")
    age = st.number_input("Patient Age", min_value=1, max_value=120, value=20)
    height_cm = st.number_input("Patient Height (cm)", min_value=50, max_value=230, value=170)
    gender = st.radio("Gender", ["male", "female"], horizontal=False)

    bone_options = ["Radius", "Ulna", "Humerus", "Forearm", "Wrist"]
    bone_type = st.selectbox("Select bone / area manually", bone_options, index=0)

    analyze_btn = st.button("Analyze X-ray")

with center:
    st.markdown('<div class="panel-title">Educational Bone Reference</div>', unsafe_allow_html=True)
    bone_img = get_bone_image_path(bone_type)
    if os.path.exists(bone_img):
        st.image(bone_img, use_container_width=True)
    else:
        st.info("Educational image not found.")

with right:
    st.markdown('<div class="panel-title">Uploaded X-ray Preview</div>', unsafe_allow_html=True)
    if uploaded_file is None:
        st.info("No X-ray uploaded yet.")
    else:
        st.image(uploaded_file, use_container_width=True)


# =========================
# Analyze Logic
# =========================
if analyze_btn:
    if uploaded_file is None:
        st.session_state.analysis_done = False
        st.session_state.validation_msg = "Please upload an X-ray image first."
        st.warning(st.session_state.validation_msg)
    else:
        file_bytes = uploaded_file.getvalue()

        valid, validation_msg = is_valid_xray(file_bytes)
        st.session_state.validation_msg = validation_msg

        if not valid:
            st.session_state.analysis_done = False
            st.session_state.pdf_bytes = None
            st.error("Invalid image: please upload a real upper-limb X-ray image.")
            st.info(validation_msg)
        else:
            features = extract_features(file_bytes)
            prediction = model.predict(features)[0]

            confidence_value = 0.0
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                confidence_value = float(np.max(proba) * 100)

            prediction_label = "Fractured" if prediction == 1 else "Normal"

            st.session_state.analysis_done = True
            st.session_state.prediction_label = prediction_label
            st.session_state.confidence_value = confidence_value
            st.session_state.recommendation = get_recommendation(prediction_label)
            st.session_state.cast_duration = estimate_cast_duration(age, bone_type, prediction_label)
            st.session_state.report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.pdf_bytes = None


# =========================
# Results Area
# =========================
if st.session_state.analysis_done:
    st.markdown("---")

    # quick result row
    if st.session_state.prediction_label == "Fractured":
        st.markdown(
            f'<div class="result-bad">Fractured - {st.session_state.confidence_value:.2f}% Confidence</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-good">Normal / Not Fractured - {st.session_state.confidence_value:.2f}% Confidence</div>',
            unsafe_allow_html=True
        )

    # summary metrics distributed nicely
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Bone Type</div><div class="metric-value">{bone_type}</div></div>',
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Confidence</div><div class="metric-value">{st.session_state.confidence_value:.2f}%</div></div>',
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Cast Duration</div><div class="metric-value" style="font-size:1.2rem;">{st.session_state.cast_duration}</div></div>',
            unsafe_allow_html=True
        )

    # tabs so everything is not stuck on one side
    tab1, tab2, tab3 = st.tabs(["Diagnostic Summary", "Patient & Case Info", "Prescription PDF"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("AI Diagnostic Summary")
            st.write(f"**Result:** {st.session_state.prediction_label}")
            st.write(f"**Recommendation:** {st.session_state.recommendation}")
            st.write(f"**Date & Time:** {st.session_state.report_time}")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("System Note")
            st.write("This system supports the doctor and does not replace medical judgment.")
            st.write("It is designed to assist in faster screening and early decision support.")
            st.write(f"**Validation:** {st.session_state.validation_msg}")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("Patient Information")
            st.write(f"**Patient Name:** {patient_name if patient_name else 'Not entered'}")
            st.write(f"**Patient Phone:** {patient_phone if patient_phone else 'Not entered'}")
            st.write(f"**Age:** {age}")
            st.write(f"**Height:** {height_cm} cm")
            st.write(f"**Gender:** {gender}")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.subheader("Case Information")
            st.write(f"**Doctor Name:** {doctor_name}")
            st.write(f"**Selected Bone:** {bone_type}")
            st.write(f"**AI Result:** {st.session_state.prediction_label}")
            st.write(f"**Estimated Cast Duration:** {st.session_state.cast_duration}")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="warn-box">Prescription support can only be generated after doctor verification.</div>', unsafe_allow_html=True)
        doctor_verified = st.checkbox("Verified by doctor before prescription generation")

        if st.button("Request Prescription PDF"):
            if not doctor_verified:
                st.error("Doctor verification is required before generating the prescription PDF.")
            else:
                pdf_data = build_pdf_data(
                    patient_name, patient_phone, doctor_name, age, height_cm, gender,
                    bone_type, st.session_state.prediction_label,
                    st.session_state.confidence_value,
                    st.session_state.recommendation,
                    st.session_state.cast_duration
                )
                st.session_state.pdf_bytes = make_pdf(pdf_data)
                st.success("Prescription PDF generated successfully.")

        if st.session_state.pdf_bytes is not None:
            st.download_button(
                "Download Prescription PDF",
                data=st.session_state.pdf_bytes,
                file_name="AI-OrthoCare_prescription_support.pdf",
                mime="application/pdf"
            )
