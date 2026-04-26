import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = "models/bone_model.keras"
CLASSES_PATH = "models/class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)

def estimate_bone_length(bone, height, gender):
    ratios = {
        "humerus": 0.186,
        "radius": 0.146,
        "ulna": 0.156,
        "femur": 0.265,
        "tibia": 0.246,
        "fibula": 0.240
    }
    length = height * ratios.get(bone, 0.15)
    if gender.lower() == "female":
        length *= 0.97
    return round(length, 1)

def recovery_plan(status, age):
    if status == "normal":
        return "A", "No cast", "No fracture detected"

    if age < 18:
        return "C", "Cast required", "3-4 weeks"
    elif age <= 40:
        return "C", "Cast required", "4-6 weeks"
    elif age <= 60:
        return "D", "Cast required", "6-8 weeks"
    else:
        return "D", "Cast required + doctor follow-up", "8-10 weeks"

img_path = input("Enter X-ray image path: ")
age = int(input("Enter patient age: "))
height = float(input("Enter patient height in cm: "))
gender = input("Enter gender male/female: ")

img = image.load_img(img_path, target_size=(224,224))
arr = image.img_to_array(img) / 255.0
arr = np.expand_dims(arr, axis=0)

pred = model.predict(arr)[0]
idx = int(np.argmax(pred))
label = class_names[idx]
confidence = float(pred[idx]) * 100

bone, status = label.rsplit("_", 1)
bone_length = estimate_bone_length(bone, height, gender)
fracture_location = round(bone_length * 0.65, 1) if status == "fractured" else 0
fracture_gap = 3 if status == "fractured" else 0
fracture_class, cast, recovery = recovery_plan(status, age)

print("\n===== AI X-ray Bone Report =====")
print(f"Detected Bone: {bone.capitalize()}")
print(f"Status: {status.capitalize()}")
print(f"Confidence: {confidence:.2f}%")
print(f"Estimated Bone Length: {bone_length} cm")
if status == "fractured":
    print(f"Estimated Fracture Location: {fracture_location} cm from top")
    print(f"Estimated Fracture Gap: {fracture_gap} mm")
print(f"Fracture Class: {fracture_class}")
print(f"Cast Decision: {cast}")
print(f"Estimated Recovery Time: {recovery}")
print("Note: Educational AI prototype, not a medical diagnosis.")
