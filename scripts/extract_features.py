import os
import cv2
import numpy as np
import pandas as pd

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

IGNORE_DIRS = {
    ".git", "venv", "__pycache__", "outputs", "models",
    "scripts", ".ipynb_checkpoints"
}

def detect_label(path):
    p = path.lower().replace("\\", "/")

    negative_words = [
        "not_fractured", "non_fractured", "no_fracture",
        "normal", "negative", "not fractured", "non fractured"
    ]

    positive_words = [
        "fractured", "fracture", "broken", "positive"
    ]

    for word in negative_words:
        if word in p:
            return 0

    for word in positive_words:
        if word in p:
            return 1

    return None

def extract_image_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

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

    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "contrast": contrast,
        "edge_density": edge_density,
        "laplacian_variance": laplacian_variance,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
    }

rows = []

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

    for file in files:
        if not file.lower().endswith(IMAGE_EXTS):
            continue

        path = os.path.join(root, file)
        label = detect_label(path)

        if label is None:
            continue

        features = extract_image_features(path)

        if features is None:
            continue

        features["image_path"] = path
        features["label"] = label
        features["label_name"] = "fractured" if label == 1 else "not_fractured"
        rows.append(features)

if not rows:
    raise SystemExit(
        "No labelled images found. لازم الصور تكون جوه فولدر اسمه fractured و فولدر اسمه normal أو not_fractured."
    )

df = pd.DataFrame(rows)
df.to_csv("outputs/extracted_features.csv", index=False)

print("Done: outputs/extracted_features.csv")
print("Total images:", len(df))
print(df["label_name"].value_counts())
